// strata/b70/pack.hpp - the B70 card library: the pack, the card's own weight format, and its file (host side, no SYCL).
//
// A pack matrix is 4-bit weights with a scale and a zero per output row and 128 inputs (weight = scale x (q - zero)), laid
// out for the matrix engine instead of for a CPU decoder: records of 16 output rows (a tile) x 256 inputs (a chunk), tile
// after tile, chunk after chunk; a record is
//     4 x 128 dwords   the weights: pair p of the chunk's 32-weight sub-blocks (2p in the low nibbles, 2p + 1 in the high
//                      ones), dword d (4 of the 32 bytes) of row i at [128 p + 16 d + i]
//     2 x 16 dwords    group g of the chunk (128 inputs), row i at [512 + 16 g + i]: the FP16 scale in the low half, the
//                      zero (1 .. 16) in the high half
// 2,176 bytes (4.25 bits a weight).  A hardware thread owns a tile and reads its records in order with plain block loads;
// a sub-block's 32 nibbles are one DPAS operand as they are.  The kernels: strata/b70/stream.hpp (decode, verify, batch
// rows) and strata/b70/fat.hpp (prompt chunks).
//
// The file ("B70W4\0\0\1", a u64 length, a JSON index {"tensors": {name: {off, bytes, n_in, n_out[, layout]}}}, the blobs
// at 4096-byte aligned offsets) names its matrices as the model's GGUF names them, so any engine that loads matrices by
// name can take them from a pack.  tools/b70_pack.py writes it from a GPTQ-style int4 checkpoint or from a GGUF.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

namespace strata::b70 {

/// The type a pack matrix carries where an engine keeps a ggml type id (not a ggml type id).
constexpr int kTypeW4 = 1004;

namespace pack {

constexpr int kRecDwords = 544;                           // dwords a record
constexpr int kRecBytes = 4 * kRecDwords;                 // 2,176
constexpr int kTileRows = 16, kChunkInputs = 256, kGroupInputs = 128;
constexpr std::size_t kDeviceAlign = (std::size_t) 2 << 20;   // a matrix's device buffer: 2 MiB (the card maps it with large pages)
constexpr int kMaxCols = 32;                              // activation columns one weight pass takes (stream products)

/// Whether an n_out x n_in matrix can be a pack matrix.
constexpr bool shape_ok(long long n_in, long long n_out) {
    return n_in > 0 && n_out > 0 && n_in % kChunkInputs == 0 && n_out % kTileRows == 0;
}
/// The bytes of a packed n_out x n_in matrix (stream::w4_bytes without the library).
constexpr std::size_t bytes(long long n_in, long long n_out) {
    return (std::size_t) (n_out / kTileRows) * (std::size_t) (n_in / kChunkInputs) * (std::size_t) kRecBytes;
}

struct Entry {
    uint64_t off = 0, bytes = 0;   // from the start of the blob area
    int n_in = 0, n_out = 0;
    std::string layout;            // empty: a plain matrix; else a derived blob (e.g. "gu_merged")
    bool plain() const { return layout.empty(); }
};

struct File {
    std::string path;
    uint64_t blobs = 0;   // the file offset of the blob area
    std::unordered_map<std::string, Entry> t;

    bool open(const std::string& p, std::string& err) {
        std::FILE* f = std::fopen(p.c_str(), "rb");
        if (!f) { err = "cannot open " + p; return false; }
        char magic[8];
        uint64_t n = 0;
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "B70W4\0\0\1", 8) != 0 || std::fread(&n, 8, 1, f) != 1 ||
            n > (64u << 20)) {
            std::fclose(f);
            err = p + " is not a B70W4 pack";
            return false;
        }
        std::string js((size_t) n, '\0');
        const bool ok = std::fread(js.data(), 1, (size_t) n, f) == (size_t) n;
        std::fclose(f);
        if (!ok) { err = p + ": short index"; return false; }
        std::unordered_map<std::string, Entry> tt;
        if (!parse_index(js, tt)) { err = p + ": no tensor index"; return false; }
        blobs = (16 + n + 4095) / 4096 * 4096;
        path = p;
        t = std::move(tt);
        return !t.empty();
    }

    /// The plain matrix `name` when the pack has it with this shape and the record count that goes with it; else nullptr.
    const Entry* find(const std::string& name, long long n_in, long long n_out) const {
        const auto it = t.find(name);
        if (it == t.end() || !it->second.plain()) return nullptr;
        const Entry& e = it->second;
        return e.n_in == n_in && e.n_out == n_out && shape_ok(n_in, n_out) && e.bytes == pack::bytes(n_in, n_out) ? &e : nullptr;
    }

    /// The entry's bytes into dst (e.bytes of them).
    bool read(const Entry& e, void* dst, std::string& err) const {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        const bool ok = f != nullptr && fseeko(f, (off_t) (blobs + e.off), SEEK_SET) == 0 &&
                        std::fread(dst, 1, (size_t) e.bytes, f) == (size_t) e.bytes;
        if (f) std::fclose(f);
        if (!ok) err = "cannot read " + std::to_string(e.bytes) + " bytes at " + std::to_string(blobs + e.off) + " of " + path;
        return ok;
    }

    /// The index as tools/b70_pack.py (json.dumps) writes it: "tensors": {"name": {"off": N, "bytes": N, "n_in": N, "n_out": N[,
    /// "layout": "..."]}, ...}.  An entry without its four numbers ends the read.
    static bool parse_index(const std::string& js, std::unordered_map<std::string, Entry>& out) {
        size_t at = js.find("\"tensors\"");
        if (at == std::string::npos) return false;
        at = js.find('{', at);
        if (at == std::string::npos) return false;
        for (;;) {
            const size_t q0 = js.find('"', at + 1);
            if (q0 == std::string::npos) break;
            const size_t q1 = js.find('"', q0 + 1);
            if (q1 == std::string::npos) break;
            size_t ob = js.find_first_not_of(" \t\r\n", q1 + 1);
            if (ob == std::string::npos || js[ob] != ':') break;
            ob = js.find_first_not_of(" \t\r\n", ob + 1);
            if (ob == std::string::npos || js[ob] != '{') break;
            const size_t oe = js.find('}', ob);
            if (oe == std::string::npos) break;
            Entry e;
            unsigned long long off = 0, nb = 0, n_in = 0, n_out = 0;
            if (!number(js, ob, oe, "off", off) || !number(js, ob, oe, "bytes", nb) || !number(js, ob, oe, "n_in", n_in) ||
                !number(js, ob, oe, "n_out", n_out) || n_in > 0x7fffffffull || n_out > 0x7fffffffull) break;
            e.off = off;
            e.bytes = nb;
            e.n_in = (int) n_in;
            e.n_out = (int) n_out;
            const size_t lk = js.find("\"layout\"", ob);
            if (lk != std::string::npos && lk < oe) {
                const size_t l0 = js.find('"', js.find(':', lk + 8));
                const size_t l1 = l0 == std::string::npos ? l0 : js.find('"', l0 + 1);
                if (l1 != std::string::npos && l1 < oe) e.layout = js.substr(l0 + 1, l1 - l0 - 1);
            }
            out[js.substr(q0 + 1, q1 - q0 - 1)] = e;
            at = js.find_first_not_of(" \t\r\n", oe + 1);
            if (at == std::string::npos || js[at] != ',') break;   // the end of the tensors object
        }
        return true;
    }

  private:
    static bool number(const std::string& js, size_t b, size_t e, const char* key, unsigned long long& v) {
        const std::string k = std::string("\"") + key + "\"";
        size_t p = js.find(k, b);
        if (p == std::string::npos || p >= e) return false;
        p = js.find(':', p + k.size());
        if (p == std::string::npos || p >= e) return false;
        const char* s = js.c_str() + p + 1;
        char* end = nullptr;
        v = std::strtoull(s, &end, 10);
        return end != s;
    }
};

}   // namespace pack
}   // namespace strata::b70
