// Host-only test of the conversation cache's SSD tier (conversation_spill.hpp): no device.  Builds small
// SavedConversations (running state and checkpoints, no K/V), writes them through the spill writer thread and reads
// them back.  Run: conversation_spill_test
#include "strata/core/conversation_spill.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

using namespace strata::core;
namespace fs = std::filesystem;

static int failures = 0;
static const char* g_ctx = "";      // which run of a shared check failed (the K/V store checks run twice)
#define CHECK(cond, what) do { if (!(cond)) { std::fprintf(stderr, "FAIL line %d: %s [%s]\n", __LINE__, what, g_ctx); ++failures; } } while (0)
// the same, with the last error text of the call (a `std::string err` in scope)
#define CHECKE(cond, what) do { if (!(cond)) { std::fprintf(stderr, "FAIL line %d: %s [%s] (%s)\n", __LINE__, what, g_ctx, err.c_str()); ++failures; } } while (0)

static SavedConversation image(int32_t first, size_t tokens, size_t ckpt_at, size_t state_bytes = 4096) {
    SavedConversation a;
    a.layer_lo = 0;
    a.layer_hi = 1;
    a.cvec = true;
    for (size_t i = 0; i < tokens; ++i) a.live.ids.push_back(first + (int32_t) i);
    a.live.gdn.assign(state_bytes, 0x5A);
    if (ckpt_at) {
        ConversationCheckpoint c;
        c.ids.assign(a.live.ids.begin(), a.live.ids.begin() + (std::ptrdiff_t) ckpt_at);
        c.gdn.assign(state_bytes, 0x3C);
        c.used = 7;
        a.checkpoints.push_back(std::move(c));
    }
    return a;
}

static std::vector<int32_t> prompt_of(const SavedConversation& a, size_t keep, size_t extra, int32_t other = 900000) {
    std::vector<int32_t> p(a.live.ids.begin(), a.live.ids.begin() + (std::ptrdiff_t) keep);
    for (size_t i = 0; i < extra; ++i) p.push_back(other + (int32_t) i);
    return p;
}

static SpillConfig config(const std::string& dir) {
    SpillConfig c;
    c.dir = dir;
    c.quota_bytes = 1ull << 30;
    c.min_free_bytes = 0;
    c.min_tokens = 100;
    return c;
}

static std::vector<ConversationSpill::Done> finish(ConversationSpill& s) {
    s.wait_idle();
    return s.drain();
}

// ---- the K/V store file (conversation_kv_store.hpp) and the spill's store entries (park_store, checkout)
using Sizes = std::vector<std::array<size_t, 5>>;           // bytes of each part of each layer
using From = std::vector<std::array<uint64_t, 5>>;

struct KvSet { std::vector<std::array<std::vector<uint8_t>, 5>> data; };

static uint8_t pat(size_t layer, size_t part, size_t i, int gen) {
    return (uint8_t) ((i * 131 + (i >> 7) * 17 + layer * 29 + part * 53 + (size_t) gen * 71) & 0xFF);
}
// Byte i of a part is the first pattern below the part's `change_at` and the pattern of `gen` from there on: a
// conversation that grew and rewrote its tail.
static KvSet make_kv(const Sizes& sizes, const Sizes& change_at, int gen) {
    KvSet s;
    s.data.resize(sizes.size());
    for (size_t j = 0; j < sizes.size(); ++j)
        for (size_t p = 0; p < 5; ++p) {
            auto& v = s.data[j][p];
            v.resize(sizes[j][p]);
            for (size_t i = 0; i < v.size(); ++i) v[i] = pat(j, p, i, i < change_at[j][p] ? 0 : gen);
        }
    return s;
}
static Sizes unchanged(const Sizes& like) {
    Sizes n = like;
    for (auto& r : n) r.fill(SIZE_MAX);
    return n;
}
static Sizes tail_at(const Sizes& old, size_t back) {       // `back` bytes before each part's end (the part's start when shorter)
    Sizes n = old;
    for (auto& r : n)
        for (auto& v : r) v = v > back ? v - back : 0;
    return n;
}
static From from_of(const Sizes& at) {
    From f(at.size());
    for (size_t j = 0; j < at.size(); ++j)
        for (size_t p = 0; p < 5; ++p) f[j][p] = at[j][p];
    return f;
}
static KvStoreLayout layout_of(const Sizes& cap) {
    KvStoreLayout l;
    for (const auto& c : cap) l.capacity.push_back({c[0], c[1], c[2], c[3], c[4]});
    return l;
}
static std::vector<SessionKvSource> sources_of(const KvSet& set, int fail_layer = -1) {
    std::vector<SessionKvSource> out;
    for (size_t j = 0; j < set.data.size(); ++j) {
        SessionKvSource s;
        s.format = (int) j + 1;
        s.cells = 100 + (int64_t) j;
        s.heads = 2 + (int64_t) j;
        s.head_dim = 64;
        s.page_size = 16;
        s.pooled_rows = 7 * ((int64_t) j + 1);
        s.idx_dim = 128;
        for (size_t p = 0; p < 5; ++p) s.sizes[p] = set.data[j][p].size();
        const bool bad = (int) j == fail_layer;
        s.read = [&set, j, bad](size_t part, size_t offset, void* dst, size_t n) {
            if (bad) return false;
            const auto& v = set.data[j][part];
            if (offset > v.size() || n > v.size() - offset) return false;
            if (n) std::memcpy(dst, v.data() + offset, n);
            return true;
        };
        s.error = [bad] { return std::string(bad ? "disk gone" : ""); };
        out.push_back(std::move(s));
    }
    return out;
}
static bool kv_same(const std::vector<ConversationKv>& kv, const KvSet& set, const std::vector<SessionKvSource>& src) {
    if (kv.size() != set.data.size()) return false;
    for (size_t j = 0; j < kv.size(); ++j) {
        const ConversationKv& k = kv[j];
        const SessionKvSource& s = src[j];
        if (k.format != s.format || k.cells != s.cells || k.heads != s.heads || k.head_dim != s.head_dim ||
            k.page_size != s.page_size || k.pooled_rows != s.pooled_rows || k.idx_dim != s.idx_dim) return false;
        const ConversationBuffer* parts[5] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
        for (size_t p = 0; p < 5; ++p) {
            const auto& want = set.data[j][p];
            if (parts[p]->size() != want.size()) return false;
            std::vector<uint8_t> got(want.size());
            if (!want.empty() && !parts[p]->read(got.data(), 0, got.size())) return false;
            if (got != want) return false;
        }
    }
    return true;
}
static uint64_t round4k(uint64_t n) { return (n + 4095) / 4096 * 4096; }
static uint64_t sum_sizes(const Sizes& s) {
    uint64_t n = 0;
    for (const auto& r : s)
        for (size_t v : r) n += v;
    return n;
}
// what kv_store_write reports: every part from the 4096 block of its first changed byte to its end
static uint64_t expect_written(const Sizes& now, const From& from) {
    uint64_t n = 0;
    for (size_t j = 0; j < now.size(); ++j)
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t start = std::min<uint64_t>(from[j][p], now[j][p]);
            if (start < now[j][p]) n += now[j][p] - start / 4096 * 4096;
        }
    return n;
}
static bool has(const std::string& text, const char* part) { return text.find(part) != std::string::npos; }
static std::string slurp(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
static std::vector<std::string> folder(const fs::path& d) {  // the names in a spill folder, without index.log
    std::vector<std::string> v;
    for (const auto& f : fs::directory_iterator(d)) {
        const std::string n = f.path().filename().string();
        if (n != "index.log") v.push_back(n);
    }
    std::sort(v.begin(), v.end());
    return v;
}
// What a crash leaves, or damage: the header block of a K/V store edited on disk; `rehash` writes a valid hash again.
static void patch_header(const std::string& path, size_t layers, const std::function<void(uint8_t*)>& edit, bool rehash) {
    std::vector<uint8_t> h(65536);
    {
        std::ifstream in(path, std::ios::binary);
        in.read((char*) h.data(), (std::streamsize) h.size());
    }
    edit(h.data());
    if (rehash) {
        const size_t used = 48 + 96 * layers + 8;
        const uint64_t x = session_hash64(h.data(), used - 8, 0);
        std::memcpy(h.data() + used - 8, &x, 8);
    }
    std::fstream out(path, std::ios::in | std::ios::out | std::ios::binary);
    out.write((const char*) h.data(), (std::streamsize) h.size());
}
static bool same_meta(const SavedConversation& x, const SavedConversation& y) {
    if (x.live.ids != y.live.ids || x.live.gdn != y.live.gdn || x.cvec != y.cvec || x.checkpoints.size() != y.checkpoints.size()) return false;
    for (size_t i = 0; i < x.checkpoints.size(); ++i)
        if (x.checkpoints[i].ids != y.checkpoints[i].ids || x.checkpoints[i].gdn != y.checkpoints[i].gdn) return false;
    return true;
}

// park_store takes its meta image by reference (it moves the checkpoints out for the writes and back).  The older checks park
// a copy: they do not look at the image afterwards (the checks of the checkpoint files do, and call park_store themselves).
static bool park(ConversationSpill& s, SavedConversation meta, const std::vector<SessionKvSource>& sources,
                 const KvStoreLayout& layout, uint64_t base_seq, const From& from, ConversationSpill::StorePark& out,
                 std::string& why) {
    return s.park_store(meta, sources, layout, base_seq, from, out, why);
}
// An image with a checkpoint at each of the token counts `ckpt_at` (prefixes of its tokens).  The state of a checkpoint
// follows from its first token and its token count only, like the real thing: the same tokens, the same state.  The state
// arrays differ from one checkpoint to the next, so a mix-up shows.
static SavedConversation image_n(int32_t first, size_t tokens, const std::vector<size_t>& ckpt_at, size_t state_bytes = 4096) {
    SavedConversation a = image(first, tokens, 0, state_bytes);
    for (size_t at : ckpt_at) {
        ConversationCheckpoint c;
        c.ids.assign(a.live.ids.begin(), a.live.ids.begin() + (std::ptrdiff_t) at);
        const uint8_t fill = (uint8_t) (0x20 + ((size_t) first * 31 + at / 100) % 200);
        c.gdn.assign(state_bytes, fill);
        c.ple.assign(64, (uint8_t) (fill ^ 0xFF));
        c.tails.assign(48, (uint8_t) (fill + 1));
        c.dead.assign(8, 1);
        c.block_pos.assign(16, (uint8_t) (fill + 2));
        c.used = at;
        a.checkpoints.push_back(std::move(c));
    }
    return a;
}
// The conversation was rewritten from token `pos` on: the tokens there change (every checkpoint past `pos` is a different
// one, even at the same token count) and so does the state it holds.
static SavedConversation rewritten(SavedConversation a, size_t pos) {
    for (size_t i = pos; i < a.live.ids.size(); ++i) a.live.ids[i] += 5000000;
    for (auto& c : a.checkpoints) {
        if (c.ids.size() <= pos) continue;
        for (size_t i = pos; i < c.ids.size(); ++i) c.ids[i] += 5000000;
        for (auto& b : c.gdn) b = (uint8_t) (b ^ 0x55);
    }
    return a;
}
static bool same_ckpt(const ConversationCheckpoint& x, const ConversationCheckpoint& y) {
    return x.ids == y.ids && x.imgs == y.imgs && x.gdn == y.gdn && x.ple == y.ple && x.tails == y.tails && x.dead == y.dead &&
           x.block_pos == y.block_pos && x.used == y.used;
}
// the whole image but its K/V: running state, geometry, steering and every checkpoint, in order
static bool same_image(const SavedConversation& x, const SavedConversation& y) {
    if (x.geometry != y.geometry || x.layer_lo != y.layer_lo || x.layer_hi != y.layer_hi || x.cvec != y.cvec ||
        !same_ckpt(x.live, y.live) || x.checkpoints.size() != y.checkpoints.size()) return false;
    for (size_t i = 0; i < x.checkpoints.size(); ++i)
        if (!same_ckpt(x.checkpoints[i], y.checkpoints[i])) return false;
    return true;
}
static std::string hex16(uint64_t h) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) h);
    return b;
}
// the name of the file of a checkpoint of the store entry c<n>: c<n>.k<tokens>-<16 hex digits of the hash of its ids>
static std::string kfile(uint64_t n, const std::vector<int32_t>& ids) {
    return "c" + std::to_string(n) + ".k" + std::to_string(ids.size()) + "-" +
           hex16(session_hash64(ids.data(), ids.size() * sizeof(int32_t), 0));
}
// every file of the store entry c<n> that holds `m`, sorted as folder() lists them
static std::vector<std::string> entry_files(uint64_t n, const SavedConversation& m) {
    std::vector<std::string> v{"c" + std::to_string(n) + ".kv", "c" + std::to_string(n) + ".meta"};
    for (const auto& c : m.checkpoints) v.push_back(kfile(n, c.ids));
    std::sort(v.begin(), v.end());
    return v;
}
static uint64_t inode_of(const fs::path& p) {
    struct stat st {};
    return ::stat(p.c_str(), &st) == 0 ? (uint64_t) st.st_ino : 0;
}
static uint64_t size_of(const fs::path& p) {
    std::error_code ec;
    const auto n = fs::file_size(p, ec);
    return ec ? UINT64_MAX : (uint64_t) n;
}

static void kv_store_checks(const fs::path& d, const char* mode) {
    g_ctx = mode;
    fs::remove_all(d);
    fs::create_directories(d);
    const SessionFileIdentity id{11, 22};
    std::string err;
    const Sizes cap = {{4000000, 4000000, 40000, 40000, 0}, {4000000, 4000000, 40000, 40000, 100000},
                       {4000000, 20000000, 40000, 40000, 100000}};
    // none a multiple of 4096; one part is zero bytes, one is above 16 MiB (two blocks)
    const Sizes s0 = {{3000017, 2500001, 9001, 9003, 0}, {3000321, 2500777, 9005, 9007, 30011}, {3001001, 17000003, 9011, 9013, 30013}};
    const Sizes s1 = {{3050028, 2540010, 12002, 12004, 0}, {3050332, 2540786, 12006, 12008, 44012}, {3051012, 17040012, 12012, 12014, 44014}};
    const KvStoreLayout layout = layout_of(cap);
    const From zero(cap.size());
    const std::string path = (d / "a.kv").string();
    uint64_t written = 0;

    // the places: aligned, in order, the file as long as the last part's place ends
    CHECK(kv_store_offset(layout, 0, 0) == 65536, "the first part follows the header block");
    bool aligned = true;
    for (size_t j = 0; j < cap.size(); ++j)
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t at = kv_store_offset(layout, j, p);
            aligned = aligned && at % 4096 == 0;
            if (p + 1 < 5) aligned = aligned && kv_store_offset(layout, j, p + 1) == at + round4k(cap[j][p]);
        }
    CHECK(aligned, "every part has an aligned place of its capacity");
    CHECK(kv_store_offset(layout, 1, 0) == kv_store_offset(layout, 0, 4) + round4k(cap[0][4]), "layer 1 follows layer 0");
    CHECK(kv_store_file_size(layout) == kv_store_offset(layout, 2, 4) + round4k(cap[2][4]), "the file ends with the last place");
    CHECK(kv_store_offset(layout, 3, 0) == UINT64_MAX && kv_store_offset(layout, 0, 5) == UINT64_MAX, "no such part");

    // 1. a new file: read back exactly, sparse
    const KvSet a = make_kv(s0, unchanged(s0), 0);
    const auto sa = sources_of(a);
    CHECKE(kv_store_write(path, id, layout, sa, zero, 4242, true, written, err), "create");
    CHECK(written == sum_sizes(s0), "a new file writes every byte of every part");
    struct stat st {};
    CHECK(::stat(path.c_str(), &st) == 0 && (uint64_t) st.st_size == kv_store_file_size(layout), "the file has the layout's size");
    CHECK((uint64_t) st.st_blocks * 512 < (uint64_t) st.st_size, "and is sparse");
    KvStoreInfo info;
    CHECKE(kv_store_info(path, id, layout, info, err), "info");
    CHECK(info.tokens == 4242 && info.layers.size() == 3, "the header keeps the tag and the layer count");
    bool header_ok = info.layers.size() == 3;
    uint64_t stored = 65536;
    for (size_t j = 0; header_ok && j < 3; ++j) {
        header_ok = info.layers[j].format == sa[j].format && info.layers[j].cells == sa[j].cells && info.layers[j].heads == sa[j].heads &&
                    info.layers[j].head_dim == sa[j].head_dim && info.layers[j].page_size == sa[j].page_size &&
                    info.layers[j].pooled_rows == sa[j].pooled_rows && info.layers[j].idx_dim == sa[j].idx_dim;
        for (size_t p = 0; p < 5; ++p) {
            header_ok = header_ok && info.layers[j].sizes[p] == s0[j][p];
            stored += round4k(s0[j][p]);
        }
    }
    CHECK(header_ok, "the header keeps the 7 numbers and the 5 sizes of every layer");
    CHECK(info.stored_bytes == stored, "stored_bytes: the header block and every part rounded up to 4096");
    std::vector<ConversationKv> kv;
    KvStoreInfo got;
    CHECKE(kv_store_read(path, id, layout, kv, got, err), "read");
    CHECK(kv_same(kv, a, sa), "every byte and the 7 numbers come back");
    CHECK(got.tokens == 4242 && got.stored_bytes == info.stored_bytes, "and read agrees with info");

    // 2. it grew and the tail of every part changed from an unaligned byte: only that is written
    const Sizes at = tail_at(s0, 1000);
    const From from = from_of(at);
    const KvSet b = make_kv(s1, at, 1);
    const auto sb = sources_of(b);
    CHECKE(kv_store_write(path, id, layout, sb, from, 5555, false, written, err), "update in place");
    CHECK(written == expect_written(s1, from), "written: each part from its first changed 4096 block");
    CHECK(written * 20 < sum_sizes(s1), "far below the whole");
    auto still_b = [&] {
        std::vector<ConversationKv> k2;
        KvStoreInfo i2;
        std::string e2;
        return kv_store_read(path, id, layout, k2, i2, e2) && kv_same(k2, b, sb) && i2.tokens == 5555;
    };
    CHECK(still_b(), "the updated file reads as the whole new content");

    // 3. refusals; each leaves the file as it was
    From gap = from_of(s1);
    gap[0][2] += 1;
    CHECK(!kv_store_write(path, id, layout, sb, gap, 1, false, written, err) && has(err, "gap"), "a from beyond the stored size is a gap");
    CHECK(still_b(), "and the file was not touched");
    Sizes over = s1;
    over[0][0] = cap[0][0] + 1;
    const KvSet big = make_kv(over, unchanged(over), 2);
    CHECK(!kv_store_write(path, id, layout, sources_of(big), zero, 1, false, written, err) && has(err, "capacity"), "a size above its capacity");
    CHECK(still_b(), "refused before anything was written");
    const SessionFileIdentity other{11, 23};
    CHECK(!kv_store_write(path, other, layout, sb, zero, 1, false, written, err) && has(err, "another"), "another identity cannot update");
    CHECK(!kv_store_info(path, other, layout, info, err) && has(err, "another"), "and cannot read the header");
    CHECK(!kv_store_read(path, other, layout, kv, got, err) && kv.empty(), "or the K/V");
    CHECK(!kv_store_write(path, id, layout, sb, zero, 1, true, written, err), "create on an existing file");
    CHECK(still_b(), "left it as it was");
    CHECK(!kv_store_write((d / "nope.kv").string(), id, layout, sb, zero, 1, false, written, err) && !fs::exists(d / "nope.kv"),
          "update of a missing file does not create it");
    CHECK(!kv_store_write((d / "c.kv").string(), id, layout, sa, from, 1, true, written, err) && has(err, "must be 0") && !fs::exists(d / "c.kv"),
          "a new file is written whole");
    Sizes cap2 = cap;
    cap2[0][0] += 8192;
    CHECK(!kv_store_info(path, id, layout_of(cap2), info, err) && has(err, "does not match"), "another layout of the same layer count");
    CHECK(!kv_store_info(path, id, layout_of(Sizes(cap.begin(), cap.begin() + 2)), info, err) && has(err, "layers"), "another layer count");
    // a header flipped on disk
    patch_header(path, 3, [](uint8_t* h) { h[12] = 1; }, false);
    CHECK(!kv_store_info(path, id, layout, info, err) && has(err, "hash"), "the state byte flipped alone: the hash refuses");
    CHECK(!kv_store_read(path, id, layout, kv, got, err) && kv.empty(), "and so does the read");
    patch_header(path, 3, [](uint8_t* h) { h[12] = 0; }, false);
    CHECK(still_b(), "flipped back");
    patch_header(path, 3, [](uint8_t* h) { h[12] = 1; }, true);          // what a stop in the middle of an update leaves
    CHECK(!kv_store_info(path, id, layout, info, err) && has(err, "being written"), "a file that was being written is refused");
    CHECK(!kv_store_read(path, id, layout, kv, got, err) && has(err, "being written") && kv.empty(), "the read too");
    CHECK(!kv_store_write(path, id, layout, sb, zero, 1, false, written, err) && has(err, "being written"), "and it is not updated");
    patch_header(path, 3, [](uint8_t* h) { h[12] = 0; }, true);
    CHECK(still_b(), "the state back to whole");
    patch_header(path, 3, [](uint8_t* h) { h[33] ^= 0xFF; }, false);
    CHECK(!kv_store_info(path, id, layout, info, err) && has(err, "damaged") && !has(err, "being written"), "a damaged header byte: the hash");
    patch_header(path, 3, [](uint8_t* h) { h[33] ^= 0xFF; }, false);
    patch_header(path, 3, [](uint8_t* h) { h[0] = 'X'; }, false);
    CHECK(!kv_store_info(path, id, layout, info, err) && has(err, "not a K/V store"), "another magic");
    patch_header(path, 3, [](uint8_t* h) { h[0] = 'S'; }, false);
    CHECK(still_b(), "every header repaired");
    // the memory check
    uint64_t asked = 0;
    CHECK(!kv_store_read(path, id, layout, kv, got, err, [&](uint64_t need, std::string& why) { asked = need; why = "no room"; return false; }) &&
          kv.empty() && has(err, "no room") && asked >= sum_sizes(s1), "the memory check can refuse a read");
    CHECKE(kv_store_read(path, id, layout, kv, got, err, [&](uint64_t need, std::string&) { asked = need; return true; }) && kv_same(kv, b, sb),
           "and can agree");

    // more updates: everything again, then a conversation that went back, then nothing new
    CHECKE(kv_store_write(path, id, layout, sb, zero, 6666, false, written, err), "update from byte 0");
    CHECK(written == sum_sizes(s1), "writes every byte");
    CHECKE(kv_store_info(path, id, layout, info, err) && info.tokens == 6666 && still_b() == false, "the tag changed");
    CHECKE(kv_store_write(path, id, layout, sa, from, 7777, false, written, err), "a shorter conversation");
    CHECK(written == expect_written(s0, from), "the tail of the shorter one");
    CHECKE(kv_store_read(path, id, layout, kv, got, err) && kv_same(kv, a, sa) && got.tokens == 7777, "reads as the shorter one");
    CHECKE(kv_store_write(path, id, layout, sa, from_of(s0), 8888, false, written, err) && written == 0, "nothing changed: nothing written");
    CHECKE(kv_store_read(path, id, layout, kv, got, err) && kv_same(kv, a, sa) && got.tokens == 8888, "and it still reads");

    // a read that fails in the middle of an update leaves a file that is refused; in a new file, no file at all
    CHECK(!kv_store_write(path, id, layout, sources_of(b, 1), zero, 9, false, written, err) && has(err, "disk gone"), "a failed source read fails the update");
    CHECK(!kv_store_info(path, id, layout, info, err) && has(err, "being written"), "the file it left is refused");
    CHECK(!kv_store_write((d / "f.kv").string(), id, layout, sources_of(b, 2), zero, 9, true, written, err) && has(err, "disk gone"), "and the new file");
    CHECK(!fs::exists(d / "f.kv"), "is not left behind");

    // the header block holds 682 layers, not one more
    const std::string many_path = (d / "many.kv").string();
    KvStoreLayout many;
    many.capacity.assign(682, std::array<uint64_t, 5>{});
    CHECKE(kv_store_write(many_path, id, many, std::vector<SessionKvSource>(682), From(682), 1, true, written, err), "682 layers");
    CHECKE(kv_store_read(many_path, id, many, kv, got, err) && kv.size() == 682 && got.stored_bytes == 65536, "read back");
    many.capacity.push_back({});
    CHECK(!kv_store_write((d / "more.kv").string(), id, many, std::vector<SessionKvSource>(683), From(683), 1, true, written, err) &&
          has(err, "do not fit") && !fs::exists(d / "more.kv"), "683 layers are refused");
    fs::remove_all(d);
}

// index.log grows with every park; above SpillConfig::index_compact_bytes it is written again from the live entries
static void index_compact_checks(const fs::path& d) {
    fs::remove_all(d);
    const SessionFileIdentity id{11, 22};
    std::string err;
    const Sizes cap = {{600000, 500000, 30000, 30000, 80000}};
    const Sizes s0 = {{200017, 150001, 9001, 9003, 0}};
    const KvStoreLayout layout = layout_of(cap);
    const KvSet a = make_kv(s0, unchanged(s0), 0);
    const auto sa = sources_of(a);
    const From nothing = from_of(s0);            // every part is on disk already: a re-park writes no K/V
    SpillConfig c = config(d.string());
    c.index_compact_bytes = 200000;              // one record of 5000 tokens is about 30 KB
    uint64_t seq = 0, grew = 0;
    bool written_once = true;                    // the checkpoint's file is written by the first park only
    {
        ConversationSpill s(c);
        CHECKE(s.open(id, err), "open");
        for (int i = 0; i < 60; ++i) {
            ConversationSpill::StorePark p;
            CHECKE(park(s, image(1, 5000 + (size_t) i, 2000), sa, layout, seq, nothing, p, err), "park");
            written_once = written_once && p.checkpoints_written == (i == 0 ? 1u : 0u) && p.checkpoints_kept == (i == 0 ? 0u : 1u);
            seq = p.seq;
            std::error_code ec;
            grew = std::max<uint64_t>(grew, fs::file_size(d / "index.log", ec));
        }
        CHECK(s.compactions() >= 2, "the log was written again more than once");
        CHECK(grew < 200000 + 100000, "and never grew far above the limit");
        CHECK(s.files() == 1, "one entry all along");
        CHECK(written_once && folder(d).size() == 3, "its checkpoint's file written once, kept by the 59 updates (K/V, meta, one K file)");
    }
    {
        ConversationSpill s(c);
        CHECKE(s.open(id, err), "reopen after compactions");
        CHECK(s.files() == 1, "the entry is in the rewritten log");
        CHECK(has(slurp(d / "index.log"), "\nK 2000 "), "with its K line");
        SavedConversation back;
        size_t bytes = 0;
        uint64_t got = 0;
        const SavedConversation last = image(1, 5059, 2000);
        CHECKE(s.checkout(s.best(prompt_of(last, 5059, 3), {}, true), SessionReadLimits{}, layout, back, bytes, got, err), "and reads back");
        CHECK(back.live.ids.size() == 5059 && kv_same(back.kv, a, sa), "the last park's tokens and its K/V");
        CHECK(back.checkpoints.size() == 1 && back.checkpoints[0].ids.size() == 2000, "and its checkpoint, from its file");
    }
    {
        SpillConfig off = c;
        off.index_compact_bytes = 0;             // 0: never
        ConversationSpill s(off);
        CHECKE(s.open(id, err), "open, no limit");
        for (int i = 0; i < 12; ++i) {
            ConversationSpill::StorePark p;
            CHECKE(park(s, image(1, 5100 + (size_t) i, 2000), sa, layout, seq, nothing, p, err), "park");
            seq = p.seq;
        }
        std::error_code ec;
        CHECK(s.compactions() == 0 && fs::file_size(d / "index.log", ec) > 200000, "no limit: the log just grows");
    }
    fs::remove_all(d);
}

static void store_entry_checks(const fs::path& sd) {
    g_ctx = "store entries";
    fs::remove_all(sd);
    const SessionFileIdentity id{11, 22};
    std::string err;
    const Sizes cap = {{600000, 500000, 30000, 30000, 80000}, {600000, 500000, 30000, 30000, 80000}};
    const Sizes s0 = {{200017, 150001, 9001, 9003, 0}, {200321, 150777, 9005, 9007, 30011}};
    const Sizes s1 = {{220028, 165010, 10002, 10006, 0}, {220332, 165786, 10006, 10010, 35012}};
    const Sizes sbig = {{500000, 400000, 25000, 25000, 0}, {500000, 400000, 25000, 25000, 70000}};
    const KvStoreLayout layout = layout_of(cap);
    const From zero(cap.size());
    const KvSet a = make_kv(s0, unchanged(s0), 0);
    const Sizes at = tail_at(s0, 1000);
    const KvSet b = make_kv(s1, at, 1);
    const auto sa = sources_of(a), sb = sources_of(b);
    const SavedConversation m0 = image(1, 5000, 2000), m1 = image(1, 5600, 2000);
    // the K/V file and the meta file of an entry (in this order); entry_files() adds the file of each checkpoint.  Every image
    // of these checks has one checkpoint, at 2000 tokens, so every store entry has a third file.
    auto files_of = [&](uint64_t n) { return std::vector<std::string>{"c" + std::to_string(n) + ".kv", "c" + std::to_string(n) + ".meta"}; };
    ConversationSpill::StorePark p0, p1;
    SavedConversation back;
    size_t bytes = 0;
    uint64_t seq = 99, born = 0;                                  // born: the seq an entry was created with, its files' name
    {
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(id, err), "open");

        // 4. a new store entry: both files in the folder, found by its tokens, read back, kept
        CHECKE(park(s, m0, sa, layout, 0, {}, p0, err), "park_store, new files");
        CHECK(p0.full && p0.seq != 0 && p0.written == sum_sizes(s0) && p0.bytes > sum_sizes(s0), "every part written whole");
        CHECK(s.files() == 1 && s.indexed_bytes() == p0.bytes, "one entry of that size");
        CHECK(folder(sd) == entry_files(p0.seq, m0), "its three files (K/V, meta, the checkpoint), named after its seq");
        auto m = s.best(prompt_of(m0, 5000, 10), {}, true);
        CHECK(m.index == 0 && m.tokens == 5000 && m.live && s.is_store(m) && s.bytes_of(m) == p0.bytes, "best() finds it");
        CHECKE(s.checkout(m, SessionReadLimits{}, layout, back, bytes, seq, err), "checkout");
        CHECK(seq == p0.seq && bytes == p0.bytes, "the entry's seq and size");
        CHECK(same_meta(back, m0) && back.kv.size() == 2 && kv_same(back.kv, a, sa), "the meta and the K/V are what was parked");
        CHECK(s.files() == 1 && folder(sd) == entry_files(p0.seq, m0), "the files stay");

        // the same entry with the K/V left in its file and streamed block by block: the same bytes, none of them in the image
        {
            SavedConversation lean;
            size_t lean_bytes = 0;
            uint64_t lean_seq = 0;
            KvStoreInfo left;
            CHECKE(s.checkout(m, SessionReadLimits{}, layout, lean, lean_bytes, lean_seq, err, &left), "checkout, K/V left in the file");
            CHECK(lean.kv.empty() && same_meta(lean, m0) && lean_seq == p0.seq && lean_bytes < bytes, "the meta without the K/V");
            CHECK(left.layers.size() == 2 && left.tokens == 5000, "the K/V file's header");
            std::vector<std::vector<std::vector<uint8_t>>> got(2, std::vector<std::vector<uint8_t>>(5));
            uint64_t next_at = 0;
            size_t last_j = 0, last_p = 0;
            bool in_order = true, began = false;
            KvStoreInfo sinfo;
            CHECKE(s.stream_kv(p0.seq, layout, [&](const KvStoreInfo& i, std::string&) { began = i.layers.size() == 2; return true; },
                               [&](size_t j, size_t p, uint64_t off, const uint8_t* d, size_t n, std::string&) {
                                   if (j != last_j || p != last_p) { next_at = 0; last_j = j; last_p = p; }
                                   in_order = in_order && off == next_at && n <= (16u << 20);
                                   next_at = off + n;
                                   got[j][p].insert(got[j][p].end(), d, d + n);
                                   return true;
                               }, sinfo, err), "stream_kv");
            bool same = began && in_order;
            for (size_t j = 0; j < 2 && same; ++j)
                for (size_t p = 0; p < 5 && same; ++p) {
                    std::vector<uint8_t> want(sa[j].sizes[p]);
                    if (!want.empty()) sa[j].read(p, 0, want.data(), want.size());
                    same = got[j][p] == want;
                }
            CHECK(same, "every part's bytes, in order, equal to what was parked");
            // only a prefix of every part: the first bytes, nothing after them
            {
                std::vector<std::array<uint64_t, 5>> only(2);
                for (size_t j = 0; j < 2; ++j)
                    for (size_t p = 0; p < 5; ++p) only[j][p] = sa[j].sizes[p] / 3 + (p == 1 ? 5000000 : 0);   // (one above its size)
                std::vector<std::vector<std::vector<uint8_t>>> pre(2, std::vector<std::vector<uint8_t>>(5));
                CHECKE(s.stream_kv(p0.seq, layout, {}, [&](size_t j, size_t p, uint64_t, const uint8_t* d, size_t n, std::string&) {
                                       pre[j][p].insert(pre[j][p].end(), d, d + n);
                                       return true;
                                   }, sinfo, err, &only), "stream_kv, a prefix");
                bool pre_ok = true;
                for (size_t j = 0; j < 2 && pre_ok; ++j)
                    for (size_t p = 0; p < 5 && pre_ok; ++p) {
                        const size_t want_n = (size_t) std::min<uint64_t>(only[j][p], sa[j].sizes[p]);
                        pre_ok = pre[j][p].size() == want_n && std::equal(pre[j][p].begin(), pre[j][p].end(), got[j][p].begin());
                    }
                CHECK(pre_ok, "each part's leading bytes and no more");
                std::vector<std::array<uint64_t, 5>> wrong(1);
                std::string why3;
                CHECK(!s.stream_kv(p0.seq, layout, {}, [](size_t, size_t, uint64_t, const uint8_t*, size_t, std::string&) { return true; },
                                   sinfo, why3, &wrong), "one prefix size per layer is asked for");
            }
            std::string why2;
            CHECK(!s.stream_kv(p0.seq, layout, [](const KvStoreInfo&, std::string& w) { w = "no"; return false; },
                               [](size_t, size_t, uint64_t, const uint8_t*, size_t, std::string&) { return true; }, sinfo, why2) &&
                  has(why2, "no"), "begin may refuse");
            CHECK(!s.stream_kv(p0.seq, layout, {}, [](size_t, size_t, uint64_t, const uint8_t*, size_t, std::string& w) { w = "full"; return false; },
                               sinfo, why2) && has(why2, "full"), "the sink may stop it");
            CHECK(s.files() == 1 && !s.stream_kv(424242, layout, {}, [](size_t, size_t, uint64_t, const uint8_t*, size_t, std::string&) { return true; },
                                                 sinfo, why2), "the entry stays; an unknown seq is refused");
        }

        // a held entry survives the quota and a newer park of the same conversation; released, it goes as any other
        {
            SpillConfig tight = config((sd / "held").string());
            fs::remove_all(sd / "held");
            ConversationSpill h(tight);
            CHECKE(h.open(id, err), "open");
            ConversationSpill::StorePark a1, a2, a3;
            SavedConversation i1 = image(7000, 5000, 2000), i2 = image(8000, 5000, 2000), i3 = image(7000, 5600, 2000);
            CHECKE(h.park_store(i1, sa, layout, 0, {}, a1, err), "first entry");
            h.hold(a1.seq);
            CHECKE(h.park_store(i3, sa, layout, 0, {}, a3, err), "a longer park of the same conversation, new files");
            CHECK(h.files() == 2, "the held entry was not superseded");
            h.hold(0);
            CHECKE(h.park_store(i2, sa, layout, 0, {}, a2, err), "another conversation");
            CHECK(h.files() == 3, "three entries");
            fs::remove_all(sd / "held");
        }

        // 5. a re-park of the same conversation, further on: the same files, only the changed bytes
        const From from = from_of(at);
        CHECKE(park(s, m1, sb, layout, p0.seq, from, p1, err), "park_store in place");
        CHECK(!p1.full && p1.seq != p0.seq && p1.written == expect_written(s1, from), "written: what changed");
        CHECK(p1.written * 5 < sum_sizes(s1), "well below the whole");
        CHECK(s.files() == 1 && s.indexed_bytes() == p1.bytes, "still one entry");
        CHECK(folder(sd) == entry_files(p0.seq, m1), "in the same three files (the checkpoint's file is the same: same tokens)");
        const std::string log = slurp(sd / "index.log");
        CHECK(has(log, ("\nD " + std::to_string(p0.seq) + "\n").c_str()) && has(log, ("\nS " + std::to_string(p1.seq) + " c" + std::to_string(p0.seq) + " ").c_str()),
              "the old record is deleted and the new one names the same files");
        m = s.best(prompt_of(m1, 5600, 10), {}, true);
        CHECK(m.index == 0 && m.tokens == 5600 && s.is_store(m), "best() finds the new tokens");
        CHECKE(s.checkout(m, SessionReadLimits{}, layout, back, bytes, seq, err), "checkout again");
        CHECK(seq == p1.seq && bytes == p1.bytes && same_meta(back, m1) && kv_same(back.kv, b, sb), "the new content");
        CHECK(s.common_prefix(p1.seq, prompt_of(m1, 3000, 50)) == 3000 && s.common_prefix(p1.seq, m1.live.ids) == 5600 &&
              s.common_prefix(p0.seq, m1.live.ids) == 0, "common_prefix: the tokens an entry shares with a conversation");
        std::vector<ConversationKv> only_kv;
        CHECKE(s.read_kv(p1.seq, layout, only_kv, err) && kv_same(only_kv, b, sb) && s.files() == 1, "read_kv: the K/V alone, the entry stays");
        CHECK(!s.read_kv(p0.seq, layout, only_kv, err), "and not of an entry that is gone");

        // refusals change nothing
        const auto before = folder(sd);
        SavedConversation pic = m1;
        pic.live.imgs.push_back(ConversationImageKey{});
        SavedConversation with_kv = m1;
        with_kv.kv.resize(1);
        ConversationSpill::StorePark none;
        CHECK(!park(s, image(1, 50, 0), sb, layout, p1.seq, from, none, err) && has(err, "short"), "too short");
        CHECK(!park(s, pic, sb, layout, p1.seq, from, none, err) && has(err, "pictures"), "pictures");
        CHECK(!park(s, with_kv, sb, layout, p1.seq, from, none, err), "a meta image with K/V");
        CHECK(!park(s, m1, sb, layout_of(Sizes(cap.begin(), cap.begin() + 1)), p1.seq, from, none, err), "sources that do not fit the layout");
        CHECK(!park(s, m1, sb, layout, p1.seq, From{}, none, err), "an update without its from");
        CHECK(s.files() == 1 && folder(sd) == before && s.indexed_bytes() == p1.bytes, "the entry and its files are as they were");
        CHECKE(s.checkout(s.best(prompt_of(m1, 5600, 10), {}, true), SessionReadLimits{}, layout, back, bytes, seq, err) && kv_same(back.kv, b, sb),
               "and it reads");

        // a source that fails in the middle of an update: both files go, the conversation is not parked
        const auto sbad = sources_of(b, 1);
        CHECK(!park(s, image(1, 5700, 2000), sbad, layout, p1.seq, zero, none, err) && has(err, "disk gone"), "a failed update");
        CHECK(s.files() == 0 && folder(sd).empty(), "removed both files and the entry");
        CHECK(s.best(prompt_of(m1, 5600, 10), {}, true).tokens == 0, "nothing to find");
        CHECK(has(slurp(sd / "index.log"), ("\nD " + std::to_string(p1.seq) + "\n").c_str()), "and the log says so");

        // the base that is gone: new files, whole
        ConversationSpill::StorePark p2, p3, p4;
        CHECKE(park(s, m0, sa, layout, 0, {}, p2, err), "park again");
        CHECKE(park(s, m1, sb, layout, p2.seq, from, p3, err) && !p3.full, "and again in place");
        CHECKE(park(s, image(1, 5700, 2000), sb, layout, p2.seq, from, p4, err), "park with the seq of an entry that is gone");
        CHECK(p4.full && p4.written == sum_sizes(s1), "new files, every part whole, from ignored");
        CHECK(s.files() == 1 && folder(sd) == entry_files(p4.seq, image(1, 5700, 2000)), "the older entry of the same conversation went");
        born = p4.seq;
    }
    {
        // 6. a restart: the entry is in the index and reads back; files nothing points to are removed, the drive's speed is kept
        std::ofstream(sd / "c999.kv") << "stray";
        std::ofstream(sd / "c999.meta") << "stray";
        std::ofstream(sd / ".c999.meta.0123.tmp") << "stray";
        std::ofstream(sd / spill_speed_file()) << "v1 1500 1000 1073741824 1\n";
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(id, err), "reopen");
        CHECK(s.files() == 1, "the store entry is loaded");
        std::vector<std::string> want = entry_files(born, image(1, 5700, 2000));
        want.push_back(spill_speed_file());
        std::sort(want.begin(), want.end());
        CHECK(folder(sd) == want, "its three files and the drive's speed stay, the rest goes");
        const auto m = s.best(prompt_of(image(1, 5700, 2000), 5700, 10), {}, true);
        CHECK(m.tokens == 5700 && s.is_store(m), "it matches");
        CHECKE(s.checkout(m, SessionReadLimits{}, layout, back, bytes, seq, err), "and reads back");
        CHECK(seq == born && kv_same(back.kv, b, sb) && same_meta(back, image(1, 5700, 2000)), "as it was");
        // an update after the restart goes in place too
        ConversationSpill::StorePark p5;
        CHECKE(park(s, image(1, 5900, 2000), sb, layout, seq, from_of(s1), p5, err), "an update after a restart");
        CHECK(!p5.full && p5.written == 0 && p5.seq != seq && folder(sd) == want, "in place, nothing of the K/V changed");
        CHECK(p5.bytes > 0, "with a size");
    }
    {
        // an entry with one file missing is not loaded, and its other file is removed
        fs::remove(sd / files_of(born)[1]);                       // the meta file
        fs::remove(sd / spill_speed_file());
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(id, err), "reopen with the meta file missing");
        CHECK(s.files() == 0, "an entry with a missing file is not loaded");
        CHECK(folder(sd).empty(), "and its other files are removed (the checkpoint's too)");
    }
    uint64_t one = 0;                                             // what an entry of m0 and `a` takes
    {
        ConversationSpill s(config(sd.string()));
        ConversationSpill::StorePark p;
        CHECKE(s.open(id, err) && park(s, m0, sa, layout, 0, {}, p, err), "a store entry");
        one = p.bytes;
        CHECK(folder(sd) == entry_files(p.seq, m0), "its three files");
    }
    {
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(SessionFileIdentity{11, 23}, err), "open with another configuration");
        CHECK(s.files() == 0 && folder(sd).empty(), "another configuration uses none of it: no entry, no file");
    }
    {
        // the quota removes the oldest entry, both its files
        const fs::path q = sd / "q1";
        SpillConfig c = config(q.string());
        c.quota_bytes = one * 5 / 2;
        ConversationSpill s(c);
        ConversationSpill::StorePark pa, pb, pc;
        CHECKE(s.open(id, err), "open");
        CHECKE(park(s, image(1, 5000, 2000), sa, layout, 0, {}, pa, err), "A");
        CHECKE(park(s, image(1000000, 5000, 2000), sa, layout, 0, {}, pb, err), "B");
        CHECK(s.files() == 2 && folder(q).size() == 6, "two entries, six files");
        CHECKE(park(s, image(2000000, 5000, 2000), sa, layout, 0, {}, pc, err), "C");
        std::vector<std::string> want = entry_files(pb.seq, image(1000000, 5000, 2000));
        for (const auto& n : entry_files(pc.seq, image(2000000, 5000, 2000))) want.push_back(n);
        std::sort(want.begin(), want.end());
        CHECK(s.files() == 2 && s.indexed_bytes() <= c.quota_bytes, "the quota holds");
        CHECK(folder(q) == want, "the oldest entry's three files went");
    }
    {
        // an update never removes its own entry: the oldest of the others goes
        const fs::path q = sd / "q2";
        SpillConfig c = config(q.string());
        c.quota_bytes = one * 3;
        ConversationSpill s(c);
        ConversationSpill::StorePark pa, pb, pu;
        CHECKE(s.open(id, err), "open");
        CHECKE(park(s, image(1, 5000, 2000), sa, layout, 0, {}, pa, err), "A");
        CHECKE(park(s, image(1000000, 5000, 2000), sa, layout, 0, {}, pb, err), "B");
        CHECK(s.files() == 2, "two entries");
        const KvSet big = make_kv(sbig, unchanged(sbig), 3);
        const auto sbg = sources_of(big);
        CHECKE(park(s, image(1, 5600, 2000), sbg, layout, pa.seq, zero, pu, err), "A, much larger, in place");
        CHECK(!pu.full && s.files() == 1 && folder(q) == entry_files(pa.seq, image(1, 5600, 2000)), "B went, A stays in its files");
        CHECKE(s.checkout(s.best(prompt_of(image(1, 5600, 2000), 5600, 10), {}, true), SessionReadLimits{}, layout, back, bytes, seq, err) &&
               kv_same(back.kv, big, sbg), "and reads as the larger one");
        // nothing is written when it cannot fit or the disk keeps its reserve
        SpillConfig tiny = config((sd / "q3").string());
        tiny.quota_bytes = 1000;
        ConversationSpill t(tiny);
        ConversationSpill::StorePark none;
        CHECKE(t.open(id, err), "open");
        CHECK(!park(t, m0, sa, layout, 0, {}, none, err) && has(err, "quota") && t.files() == 0 && folder(sd / "q3").empty(), "above the quota");
        SpillConfig poor = config((sd / "q4").string());
        poor.min_free_bytes = UINT64_MAX / 2;
        ConversationSpill r(poor);
        CHECKE(r.open(id, err), "open");
        CHECK(!park(r, m0, sa, layout, 0, {}, none, err) && has(err, "disk space") && r.files() == 0 && folder(sd / "q4").empty(),
              "the disk keeps its reserve");
    }
    {
        // a whole-file entry and a store entry side by side
        const fs::path q = sd / "mix";
        ConversationSpill s(config(q.string()));
        ConversationSpill::StorePark ps;
        const SavedConversation w = image(3000000, 3000, 0);
        CHECKE(s.open(id, err) && park(s, m0, sa, layout, 0, {}, ps, err), "a store entry");
        SavedConversation copy = w;
        CHECKE(s.submit(std::move(copy), {}, err), "and a whole file");
        finish(s);
        CHECK(s.files() == 2, "two entries");
        auto mw = s.best(prompt_of(w, 3000, 5), {}, true);
        auto ms = s.best(prompt_of(m0, 5000, 10), {}, true);
        CHECK(mw.tokens == 3000 && !s.is_store(mw) && ms.tokens == 5000 && s.is_store(ms), "each is found, one of each kind");
        CHECK(!s.take(ms, SessionReadLimits{}, back, bytes, err) && has(err, "checkout") && s.files() == 2, "take() leaves a store entry alone");
        CHECKE(s.checkout(mw, SessionReadLimits{}, layout, back, bytes, seq, err), "checkout of a whole-file entry");
        CHECK(seq == 0 && same_meta(back, w) && s.files() == 1, "is take(): seq 0, and the file is gone");
        ms = s.best(prompt_of(m0, 5000, 10), {}, true);
        CHECKE(s.checkout(ms, SessionReadLimits{}, layout, back, bytes, seq, err), "checkout of the store entry");
        CHECK(seq == ps.seq && s.files() == 1 && kv_same(back.kv, a, sa) && folder(q) == entry_files(ps.seq, m0), "keeps it");
        SavedConversation copy2 = w;
        CHECKE(s.submit(std::move(copy2), {}, err), "another whole file");
        finish(s);
        mw = s.best(prompt_of(w, 3000, 5), {}, true);
        CHECKE(s.take(mw, SessionReadLimits{}, back, bytes, err) && same_meta(back, w) && s.files() == 1, "take() of it still reads and removes");
    }
    {
        // an entry that cannot be read is removed, both files
        const fs::path q = sd / "bad";
        ConversationSpill s(config(q.string()));
        ConversationSpill::StorePark pk;
        CHECKE(s.open(id, err) && park(s, m0, sa, layout, 0, {}, pk, err), "a store entry");
        fs::resize_file(q / files_of(pk.seq)[1], 100);
        CHECK(!s.checkout(s.best(prompt_of(m0, 5000, 10), {}, true), SessionReadLimits{}, layout, back, bytes, seq, err) && seq == 0,
              "a damaged meta file");
        CHECK(s.files() == 0 && folder(q).empty(), "the entry and its files are removed");
        CHECKE(park(s, m0, sa, layout, 0, {}, pk, err), "park again");
        patch_header((q / files_of(pk.seq)[0]).string(), 2, [](uint8_t* h) { h[12] = 1; }, true);
        CHECK(!s.checkout(s.best(prompt_of(m0, 5000, 10), {}, true), SessionReadLimits{}, layout, back, bytes, seq, err) && has(err, "being written"),
              "a K/V file that was being written");
        CHECK(s.files() == 0 && folder(q).empty(), "removed too");
    }
    for (int verify = 0; verify < 2; ++verify) {
        // the payload hash of the meta file is read only when asked for (like take())
        const fs::path q = sd / "hash";
        fs::remove_all(q);
        SpillConfig c = config(q.string());
        c.verify = verify != 0;
        ConversationSpill s(c);
        ConversationSpill::StorePark pk;
        CHECKE(s.open(id, err) && park(s, m0, sa, layout, 0, {}, pk, err), "a store entry");
        {
            std::fstream f(q / files_of(pk.seq)[1], std::ios::in | std::ios::out | std::ios::binary);
            f.seekg(100);
            char ch = 0;
            f.read(&ch, 1);
            ch = (char) (ch ^ 0xFF);
            f.seekp(100);
            f.write(&ch, 1);
        }
        const bool ok = s.checkout(s.best(prompt_of(m0, 5000, 10), {}, true), SessionReadLimits{}, layout, back, bytes, seq, err);
        CHECK(ok == (verify == 0), verify ? "verify: a flipped byte in the meta file is refused" : "no verify: it is not looked at");
        CHECK(s.files() == (verify ? 0u : 1u), verify ? "and the entry is removed" : "and the entry stays");
    }
    fs::remove_all(sd);
}

// what Entry::bytes must be, taken from the folder: the meta file, every checkpoint file and the K/V store's stored bytes
static uint64_t expect_entry_bytes(const fs::path& d, uint64_t name, const SavedConversation& m, const SessionFileIdentity& id,
                                   const KvStoreLayout& layout) {
    KvStoreInfo ki;
    std::string err;
    if (!kv_store_info((d / ("c" + std::to_string(name) + ".kv")).string(), id, layout, ki, err)) return UINT64_MAX;
    uint64_t n = ki.stored_bytes + size_of(d / ("c" + std::to_string(name) + ".meta"));
    for (const auto& c : m.checkpoints) n += size_of(d / kfile(name, c.ids));
    return n;
}
static std::string ids_line(char tag, const std::vector<int32_t>& ids) {
    std::string o(1, tag);
    for (int32_t t : ids) o += " " + std::to_string(t);
    return o + "\n";
}
// the position of `text` in `log` at or after `from` (npos: not there)
static size_t find_from(const std::string& log, const std::string& text, size_t from) { return log.find(text, from); }

// Every checkpoint of a store entry is a file of its own (c<N>.k<tokens>-<hash>), written once.
static void ckpt_file_checks(const fs::path& sd) {
    g_ctx = "checkpoint files";
    fs::remove_all(sd);
    const SessionFileIdentity id{11, 22};
    std::string err;
    const Sizes cap = {{600000, 500000, 30000, 30000, 80000}, {600000, 500000, 30000, 30000, 80000}};
    const Sizes s0 = {{200017, 150001, 9001, 9003, 0}, {200321, 150777, 9005, 9007, 30011}};
    const Sizes s1 = {{220028, 165010, 10002, 10006, 0}, {220332, 165786, 10006, 10010, 35012}};
    const KvStoreLayout layout = layout_of(cap);
    const From zero(cap.size());
    const KvSet a = make_kv(s0, unchanged(s0), 0);
    const Sizes at = tail_at(s0, 1000);
    const KvSet b = make_kv(s1, at, 1);
    const auto sa = sources_of(a), sb = sources_of(b);
    const From from = from_of(at), stay = from_of(s1);          // stay: the K/V is on disk already
    SavedConversation back;
    size_t bytes = 0;
    uint64_t seq = 99;
    auto meta_name = [](uint64_t n) { return "c" + std::to_string(n) + ".meta"; };
    auto read_file = [&](const fs::path& f, SavedConversation& out) {
        size_t n = 0;
        std::string e;
        return session_file_read(f.string(), id, out, n, e);
    };
    auto best_for = [&](ConversationSpill& s, const SavedConversation& m, size_t keep) { return s.best(prompt_of(m, keep, 10), {}, true); };
    auto checkout = [&](ConversationSpill& s, const SavedConversation& m, size_t keep) {
        return s.checkout(best_for(s, m, keep), SessionReadLimits{}, layout, back, bytes, seq, err);
    };

    const SavedConversation base = image_n(1, 6000, {1500, 3000, 4500});
    const SavedConversation upd = image_n(1, 6600, {1500, 3000, 5200});
    const SavedConversation more = image_n(1, 6700, {1500, 3000, 5200});
    const SavedConversation rew = rewritten(more, 2000);        // the same token counts, other tokens from 2000 on
    uint64_t name = 0;                                          // the entry's file name: the seq it was born with
    uint64_t last = 0;
    {
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(id, err), "open");

        // 1. a new entry with 3 checkpoints
        SavedConversation m = base;
        ConversationSpill::StorePark p;
        CHECKE(s.park_store(m, sa, layout, 0, {}, p, err), "park a conversation with 3 checkpoints");
        name = p.seq;
        CHECK(same_image(m, base) && m.checkpoints.size() == 3, "the meta image is exactly as it came in: its checkpoints are back");
        CHECK(p.full && p.checkpoints_written == 3 && p.checkpoints_kept == 0, "3 checkpoint files written, none kept");
        CHECK(folder(sd) == entry_files(name, base), "the K/V, the meta and one file per checkpoint, named by tokens and hash");
        uint64_t files_bytes = 0;
        for (const auto& c : base.checkpoints) files_bytes += size_of(sd / kfile(name, c.ids));
        const uint64_t meta_size = size_of(sd / meta_name(name));
        CHECK(p.meta == meta_size + files_bytes, "StorePark::meta: the meta file and the checkpoint files this call wrote");
        CHECK(p.bytes == expect_entry_bytes(sd, name, base, id, layout) && s.indexed_bytes() == p.bytes, "the entry counts all its files");
        SavedConversation mf, kf;
        CHECK(read_file(sd / meta_name(name), mf) && mf.checkpoints.empty() && mf.kv.empty() && same_ckpt(mf.live, base.live),
              "the meta file holds the running state and no checkpoint");
        CHECK(read_file(sd / kfile(name, base.checkpoints[1].ids), kf) && same_ckpt(kf.live, base.checkpoints[1]) && kf.checkpoints.empty() &&
              kf.kv.empty() && kf.geometry == base.geometry && kf.layer_lo == base.layer_lo && kf.layer_hi == base.layer_hi && kf.cvec == base.cvec,
              "a checkpoint file holds that checkpoint as its live state, with the meta's geometry");
        const std::string log = slurp(sd / "index.log");
        const size_t at_s = log.find("\nS " + std::to_string(p.seq) + " c" + std::to_string(name) + " ");
        size_t pos = at_s;
        bool in_order = at_s != std::string::npos;
        for (size_t i = 0; in_order && i < 3; ++i) {
            const auto& c = base.checkpoints[i];
            const std::string line = "\nK " + std::to_string(c.ids.size()) + " " +
                                     std::to_string(session_hash64(c.ids.data(), c.ids.size() * sizeof(int32_t), 0)) + " " +
                                     std::to_string(size_of(sd / kfile(name, c.ids))) + "\n";
            const size_t q = find_from(log, line, pos);
            in_order = q != std::string::npos;
            pos = in_order ? q + 1 : pos;
        }
        const size_t at_l = find_from(log, "\nL ", at_s == std::string::npos ? 0 : at_s);
        CHECK(in_order && at_l != std::string::npos && pos < at_l, "the record: S, then one K line per checkpoint file in order, then L");
        auto mt = best_for(s, base, 3500);
        CHECK(mt.tokens == 3000 && !mt.live && s.is_store(mt), "best() finds the checkpoint a prompt that diverges at 3500 can use");
        CHECKE(s.checkout(mt, SessionReadLimits{}, layout, back, bytes, seq, err), "checkout");
        CHECK(seq == p.seq && bytes == p.bytes && same_image(back, base) && kv_same(back.kv, a, sa),
              "the 3 checkpoints come back in order with their state, the K/V too, and `bytes` is the entry's");
        CHECK(folder(sd) == entry_files(name, base), "the files stay");

        // 2. the conversation went on: 2 of the 3 checkpoints stay, 4500 is gone, 5200 is new
        const fs::path k0 = sd / kfile(name, base.checkpoints[0].ids), k1 = sd / kfile(name, base.checkpoints[1].ids);
        const uint64_t ino0 = inode_of(k0), ino1 = inode_of(k1);
        m = upd;
        ConversationSpill::StorePark p2;
        CHECKE(s.park_store(m, sb, layout, p.seq, from, p2, err), "park in place: 2 checkpoints stay, 1 goes, 1 is new");
        CHECK(same_image(m, upd), "the meta image is as it came in");
        CHECK(!p2.full && p2.seq != p.seq && p2.checkpoints_written == 1 && p2.checkpoints_kept == 2, "one file written, two kept");
        CHECK(folder(sd) == entry_files(name, upd), "the entry's files: 4500's file is gone, 5200's is there");
        CHECK(ino0 != 0 && ino1 != 0 && inode_of(k0) == ino0 && inode_of(k1) == ino1, "the kept files were not touched (a rewrite is a new file)");
        CHECK(p2.meta == size_of(sd / meta_name(name)) + size_of(sd / kfile(name, upd.checkpoints[2].ids)),
              "StorePark::meta counts the meta file and the one new file, not the kept ones");
        CHECK(p2.bytes == expect_entry_bytes(sd, name, upd, id, layout) && s.indexed_bytes() == p2.bytes, "the entry counts the 3 files it has");
        const std::string log2 = slurp(sd / "index.log");
        CHECK(has(log2, ("\nD " + std::to_string(p.seq) + "\n").c_str()) && has(log2, ("\nS " + std::to_string(p2.seq) + " c" + std::to_string(name) + " ").c_str()) &&
              has(log2, ("\nK 5200 " + std::to_string(session_hash64(upd.checkpoints[2].ids.data(), 5200 * sizeof(int32_t), 0)) + " ").c_str()),
              "the old record is deleted, the new one names the same files and the new checkpoint");
        CHECKE(checkout(s, upd, 5400), "checkout of the new chain");
        CHECK(best_for(s, upd, 5400).tokens == 5200 && seq == p2.seq && bytes == p2.bytes && same_image(back, upd) && kv_same(back.kv, b, sb),
              "its checkpoints in order (1500, 3000, 5200) with their state");

        // nothing new in the checkpoints: nothing written but the meta file
        m = more;
        ConversationSpill::StorePark p3;
        CHECKE(s.park_store(m, sb, layout, p2.seq, stay, p3, err), "park in place, the same checkpoints");
        CHECK(p3.checkpoints_written == 0 && p3.checkpoints_kept == 3 && p3.written == 0 && p3.meta == size_of(sd / meta_name(name)),
              "no file written but the meta file");
        CHECK(folder(sd) == entry_files(name, more) && inode_of(k0) == ino0 && inode_of(k1) == ino1, "the same three files");

        // 3. a checkpoint with the same token count but other tokens (the conversation was rewritten before it) gets a new file
        m = rew;
        ConversationSpill::StorePark p4;
        CHECKE(s.park_store(m, sb, layout, p3.seq, stay, p4, err), "park in place, the conversation rewritten from token 2000");
        CHECK(same_image(m, rew), "the meta image is as it came in");
        CHECK(p4.checkpoints_written == 2 && p4.checkpoints_kept == 1, "1500 stays; 3000 and 5200 are new files");
        const std::string old3000 = kfile(name, more.checkpoints[1].ids), new3000 = kfile(name, rew.checkpoints[1].ids);
        CHECK(old3000 != new3000 && old3000.substr(0, old3000.find('-')) == new3000.substr(0, new3000.find('-')), "the same token count, another hash: another name");
        CHECK(!fs::exists(sd / old3000) && !fs::exists(sd / kfile(name, more.checkpoints[2].ids)), "the old files are removed");
        CHECK(folder(sd) == entry_files(name, rew) && inode_of(k0) == ino0, "the folder holds exactly the new chain");
        CHECKE(checkout(s, rew, 3500), "checkout");
        CHECK(same_image(back, rew) && !same_ckpt(back.checkpoints[1], more.checkpoints[1]) && kv_same(back.kv, b, sb),
              "the new chain, not the old state at 3000");
        last = p4.seq;
    }
    {
        // 4. a restart: the entry and its checkpoint files are loaded; files nothing points to go
        std::ofstream(sd / "c999.k100-0000000000000000") << "stray";
        std::ofstream(sd / ("." + kfile(name, rew.checkpoints[0].ids) + ".0123.tmp")) << "stray";
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(id, err), "reopen");
        CHECK(s.files() == 1 && folder(sd) == entry_files(name, rew), "the entry and its checkpoint files stay, the strays go");
        CHECK(best_for(s, rew, 5400).tokens == 5200, "best() knows its checkpoints");
        CHECKE(checkout(s, rew, 5400), "and it reads back");
        CHECK(seq == last && same_image(back, rew) && kv_same(back.kv, b, sb) && s.indexed_bytes() == bytes, "as it was");
        SavedConversation m = rew;
        m.live.ids.push_back(7);
        ConversationSpill::StorePark p5;
        CHECKE(s.park_store(m, sb, layout, last, stay, p5, err), "an update after a restart");
        CHECK(p5.checkpoints_written == 0 && p5.checkpoints_kept == 3, "keeps the files the index lists");
        last = p5.seq;
    }
    {
        // a missing checkpoint file: the entry is not loaded, its other files are removed
        fs::remove(sd / kfile(name, rew.checkpoints[1].ids));
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(id, err), "reopen with a checkpoint file missing");
        CHECK(s.files() == 0 && folder(sd).empty(), "the entry is not loaded and none of its files is left");
    }
    {
        // another configuration uses none of it
        ConversationSpill s(config(sd.string()));
        ConversationSpill::StorePark p;
        CHECKE(s.open(id, err) && park(s, base, sa, layout, 0, {}, p, err), "a store entry");
        CHECK(folder(sd) == entry_files(p.seq, base), "with its checkpoint files");
    }
    {
        ConversationSpill s(config(sd.string()));
        CHECKE(s.open(SessionFileIdentity{11, 23}, err), "open with another configuration");
        CHECK(s.files() == 0 && folder(sd).empty(), "no entry and no file, the checkpoint files included");
    }
    {
        // drop(), the quota and a newer image of the same conversation remove every file of an entry
        const fs::path q = sd / "q";
        ConversationSpill s(config(q.string()));
        ConversationSpill::StorePark pa, pb;
        CHECKE(s.open(id, err) && park(s, base, sa, layout, 0, {}, pa, err), "a store entry");
        CHECK(s.drop(pa.seq) && s.files() == 0 && folder(q).empty(), "drop() removes all five files");
        CHECK(has(slurp(q / "index.log"), ("\nD " + std::to_string(pa.seq) + "\n").c_str()), "and logs it");
        CHECKE(park(s, base, sa, layout, 0, {}, pa, err), "park again");
        CHECKE(park(s, upd, sb, layout, 0, {}, pb, err), "park the same conversation, further on, as a new entry");
        CHECK(s.files() == 1 && folder(q) == entry_files(pb.seq, upd), "the older entry's files, checkpoint files included, were superseded");
        // the quota
        const fs::path q2 = sd / "q2";
        SpillConfig c = config(q2.string());
        c.quota_bytes = pa.bytes * 5 / 2;
        ConversationSpill t(c);
        ConversationSpill::StorePark ta, tb, tc;
        const SavedConversation x = image_n(1000000, 6000, {1500, 3000, 4500}), y = image_n(2000000, 6000, {1500, 3000, 4500});
        CHECKE(t.open(id, err) && park(t, base, sa, layout, 0, {}, ta, err) && park(t, x, sa, layout, 0, {}, tb, err), "two entries");
        CHECK(t.files() == 2 && folder(q2).size() == 10, "ten files");
        CHECKE(park(t, y, sa, layout, 0, {}, tc, err), "a third");
        std::vector<std::string> want = entry_files(tb.seq, x);
        for (const auto& n : entry_files(tc.seq, y)) want.push_back(n);
        std::sort(want.begin(), want.end());
        CHECK(t.files() == 2 && t.indexed_bytes() <= c.quota_bytes && folder(q2) == want, "the oldest entry went with all five of its files");
    }
    {
        // the quota counts the checkpoint files: states of 200 KB, a quota that holds the entry without them
        const fs::path q = sd / "q3";
        const SavedConversation plain = image_n(1, 6000, {}, 200000), heavy = image_n(1, 6000, {1500, 3000, 4500}, 200000);
        ConversationSpill::StorePark p0;
        {
            ConversationSpill s(config(q.string()));
            CHECKE(s.open(id, err) && park(s, plain, sa, layout, 0, {}, p0, err), "an entry without checkpoints");
            CHECK(folder(q) == entry_files(p0.seq, plain) && p0.checkpoints_written == 0 && p0.checkpoints_kept == 0, "two files");
            CHECKE(checkout(s, plain, 6000) && back.checkpoints.empty() && same_image(back, plain), "reads back without checkpoints");
        }
        fs::remove_all(q);
        SpillConfig c = config(q.string());
        c.quota_bytes = p0.bytes + 150000;
        ConversationSpill s(c);
        ConversationSpill::StorePark none, p1;
        CHECKE(s.open(id, err), "open");
        CHECK(!park(s, heavy, sa, layout, 0, {}, none, err) && has(err, "quota") && s.files() == 0 && folder(q).empty(),
              "three checkpoint files of 200 KB do not fit: refused, nothing written");
        CHECKE(park(s, plain, sa, layout, 0, {}, p1, err), "the entry without them fits");
        CHECK(!park(s, heavy, sa, layout, p1.seq, from, none, err) && has(err, "quota") && s.files() == 1 && folder(q) == entry_files(p1.seq, plain),
              "an update that adds them is refused too, the entry stays as it was");
    }
    {
        // the disk keeps its reserve: the checkpoint files count (a new one of 64 MB, a reserve 32 MB under the free space now)
        const fs::path q = sd / "q4";
        std::error_code ec;
        fs::create_directories(q, ec);
        SpillConfig c = config(q.string());
        c.min_free_bytes = fs::space(q, ec).available > (96ull << 20) ? fs::space(q, ec).available - (32ull << 20) : 0;
        ConversationSpill s(c);
        ConversationSpill::StorePark p1, none;
        CHECKE(s.open(id, err), "open");
        if (c.min_free_bytes) {
            const SavedConversation small = image_n(1, 6000, {});
            CHECKE(park(s, small, sa, layout, 0, {}, p1, err), "an entry that needs about 2 MB fits above the reserve");
            SavedConversation bigger = image_n(1, 6600, {1500}), fresh = image_n(5000000, 6000, {1500});
            bigger.checkpoints[0].gdn.assign(64u << 20, 7);
            fresh.checkpoints[0].gdn.assign(64u << 20, 7);
            const auto before = folder(q);
            CHECK(!s.park_store(bigger, sb, layout, p1.seq, from, none, err) && has(err, "disk space") && s.files() == 1 && folder(q) == before,
                  "an update that adds a checkpoint file of 64 MB: refused for disk space, nothing changed");
            CHECK(!s.park_store(fresh, sa, layout, 0, {}, none, err) && has(err, "disk space") && s.files() == 1 && folder(q) == before,
                  "and a new entry with one");
        }
    }

    // 5. an entry whose checkpoint files cannot be read
    {
        const fs::path q = sd / "bad";
        ConversationSpill s(config(q.string()));
        ConversationSpill::StorePark pk;
        CHECKE(s.open(id, err) && park(s, base, sa, layout, 0, {}, pk, err), "a store entry");
        const auto k = [&](size_t i) { return q / kfile(pk.seq, base.checkpoints[i].ids); };
        fs::resize_file(k(1), 100);
        CHECK(!checkout(s, base, 3500) && seq == 0 && s.files() == 0 && folder(q).empty(), "a damaged checkpoint file: refused, the entry and all its files are removed");
        CHECKE(park(s, base, sa, layout, 0, {}, pk, err), "park again");
        fs::remove(k(2));
        CHECK(!checkout(s, base, 3500) && s.files() == 0 && folder(q).empty(), "a missing checkpoint file: the same");
        CHECKE(park(s, base, sa, layout, 0, {}, pk, err), "park again");
        fs::copy_file(k(0), k(1), fs::copy_options::overwrite_existing);
        CHECK(!checkout(s, base, 3500) && has(err, "tokens") && s.files() == 0 && folder(q).empty(), "a file that holds the tokens of another: refused, removed");
        CHECKE(park(s, base, sa, layout, 0, {}, pk, err), "park again");
        {
            // the meta file carries checkpoints while the entry lists their files: damaged
            size_t n = 0;
            CHECKE(session_file_write((q / meta_name(pk.seq)).string(), base, id, n, err), "a meta file with checkpoints");
        }
        CHECK(!checkout(s, base, 3500) && has(err, "carries checkpoints") && s.files() == 0 && folder(q).empty(), "refused, removed");
        CHECKE(park(s, base, sa, layout, 0, {}, pk, err), "park again");
        SessionReadLimits few;
        few.max_checkpoints = 2;
        CHECK(!s.checkout(best_for(s, base, 3500), few, layout, back, bytes, seq, err) && has(err, "limit") && s.files() == 0 && folder(q).empty(),
              "more checkpoints than the engine accepts: refused, removed");
        CHECKE(park(s, base, sa, layout, 0, {}, pk, err), "park again");

        // a refusal for RAM keeps the entry and its files, and leaves the caller's image alone.  The memory check is asked once
        // for each file, for that read's own peak.
        std::vector<uint64_t> needs;
        SessionReadLimits ask;
        ask.admit = [&](uint64_t need, std::string&) { needs.push_back(need); return true; };
        CHECKE(s.checkout(best_for(s, base, 3500), ask, layout, back, bytes, seq, err) && same_image(back, base), "checkout with a memory check that agrees");
        std::vector<uint64_t> want_needs = {session_read_peak_bytes(size_of(q / meta_name(pk.seq)))};
        for (size_t i = 0; i < 3; ++i) want_needs.push_back(session_read_peak_bytes(size_of(k(i))));
        KvStoreInfo ki;
        CHECKE(kv_store_info((q / ("c" + std::to_string(pk.seq) + ".kv")).string(), id, layout, ki, err), "info");
        want_needs.push_back(session_read_peak_bytes(ki.stored_bytes));
        CHECK(needs == want_needs, "asked for the meta file, each checkpoint file and the K/V, each for its own peak");
        for (size_t refuse = 1; refuse <= 4; ++refuse) {
            size_t calls = 0;
            SessionReadLimits no;
            no.admit = [&](uint64_t, std::string& why) { why = "no room"; return ++calls != refuse + 1; };
            SavedConversation keep_me = image(77, 300, 0);
            const SavedConversation copy = keep_me;
            const bool ok = s.checkout(best_for(s, base, 3500), no, layout, keep_me, bytes, seq, err);
            CHECK(!ok && seq == 0 && s.files() == 1 && folder(q) == entry_files(pk.seq, base) && same_image(keep_me, copy),
                  "refused for RAM (at the meta file's successors, each checkpoint file, then the K/V): the entry and its files stay, the image is untouched");
        }
        CHECKE(checkout(s, base, 3500) && same_image(back, base), "and a request that is admitted reads it");
    }
    for (int verify = 0; verify < 2; ++verify) {
        // the payload hash of a checkpoint file is read only when asked for (like the meta file's)
        const fs::path q = sd / "hash";
        fs::remove_all(q);
        SpillConfig c = config(q.string());
        c.verify = verify != 0;
        ConversationSpill s(c);
        ConversationSpill::StorePark pk;
        CHECKE(s.open(id, err) && park(s, base, sa, layout, 0, {}, pk, err), "a store entry");
        {
            std::fstream f(q / kfile(pk.seq, base.checkpoints[2].ids), std::ios::in | std::ios::out | std::ios::binary);
            f.seekg(100);
            char ch = 0;
            f.read(&ch, 1);
            ch = (char) (ch ^ 0xFF);
            f.seekp(100);
            f.write(&ch, 1);
        }
        const bool ok = checkout(s, base, 3500);
        CHECK(ok == (verify == 0), verify ? "verify: a flipped byte in a checkpoint file is refused" : "no verify: it is not looked at");
        CHECK(s.files() == (verify ? 0u : 1u), verify ? "and the entry is removed" : "and the entry stays");
    }

    // 6. a park that fails leaves no file of the entry
    {
        const fs::path q = sd / "fail";
        ConversationSpill s(config(q.string()));
        ConversationSpill::StorePark pk, none;
        CHECKE(s.open(id, err) && park(s, base, sa, layout, 0, {}, pk, err), "a store entry with 3 checkpoints");
        SavedConversation m = upd;
        const auto sbad = sources_of(b, 1);
        CHECK(!s.park_store(m, sbad, layout, pk.seq, zero, none, err) && has(err, "disk gone"), "an update with a K/V source that fails");
        CHECK(same_image(m, upd) && m.checkpoints.size() == 3, "the meta image is as it came in");
        CHECK(s.files() == 0 && folder(q).empty(), "no file of the entry is left, the kept checkpoint files neither");
        CHECK(has(slurp(q / "index.log"), ("\nD " + std::to_string(pk.seq) + "\n").c_str()), "and the log says so");
        // a checkpoint file that cannot be written (its name is a folder), after the first new one was written
        SavedConversation two = image_n(1, 6000, {1500, 3000});
        CHECKE(park(s, two, sa, layout, 0, {}, pk, err), "park again, 2 checkpoints");
        SavedConversation want_m = image_n(1, 6600, {1500, 4000, 5000});
        const SavedConversation want_copy = want_m;
        fs::create_directory(q / kfile(pk.seq, want_m.checkpoints[2].ids));
        CHECK(!s.park_store(want_m, sb, layout, pk.seq, from, none, err) && has(err, kfile(pk.seq, want_m.checkpoints[2].ids).c_str()),
              "an update whose last new checkpoint file cannot be written (the error names that file)");
        CHECK(same_image(want_m, want_copy), "the meta image is as it came in");
        CHECK(s.files() == 0 && folder(q).empty(), "every file of the entry went: the K/V, the meta, the kept and the new checkpoint files");
        // a K/V source that throws: the files go, the image has its checkpoints back, the exception goes on
        CHECKE(park(s, two, sa, layout, 0, {}, pk, err), "park again");
        auto sthrow = sources_of(b);
        sthrow[1].read = [](size_t, size_t, void*, size_t) -> bool { throw std::runtime_error("boom"); };
        SavedConversation m3 = image_n(1, 6600, {1500, 4000});
        const SavedConversation m3_copy = m3;
        bool threw = false;
        try {
            s.park_store(m3, sthrow, layout, pk.seq, zero, none, err);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        CHECK(threw, "the exception goes on");
        CHECK(same_image(m3, m3_copy) && m3.checkpoints.size() == 2, "the meta image is as it came in");
        CHECK(s.files() == 0 && folder(q).empty(), "no file of the entry is left");
        // refusals change nothing and give the checkpoints back
        CHECKE(park(s, base, sa, layout, 0, {}, pk, err), "park again");
        const auto before = folder(q);
        SavedConversation dup = base;
        dup.checkpoints.push_back(base.checkpoints[0]);
        const SavedConversation dup_copy = dup;
        CHECK(!s.park_store(dup, sb, layout, pk.seq, from, none, err) && has(err, "same tokens") && same_image(dup, dup_copy), "two checkpoints with the same tokens");
        SavedConversation pic = base;
        pic.checkpoints[1].imgs.push_back(ConversationImageKey{});
        const SavedConversation pic_copy = pic;
        CHECK(!s.park_store(pic, sb, layout, pk.seq, from, none, err) && has(err, "pictures") && same_image(pic, pic_copy), "a picture in a checkpoint");
        CHECK(s.files() == 1 && folder(q) == before, "the entry and its files are as they were");
    }

    // 7. a folder that an older version wrote: its meta file holds the checkpoints, its records have no K lines
    {
        const fs::path q = sd / "old";
        fs::remove_all(q);
        fs::create_directories(q);
        const SavedConversation om = image_n(1, 5000, {1200, 2400});
        uint64_t written = 0;
        size_t meta_bytes = 0;
        KvStoreInfo ki;
        CHECKE(kv_store_write((q / "c1.kv").string(), id, layout, sa, zero, 5000, true, written, err) &&
               kv_store_info((q / "c1.kv").string(), id, layout, ki, err) &&
               session_file_write((q / "c1.meta").string(), om, id, meta_bytes, err), "the files of an old store entry");
        {
            std::ofstream log(q / "index.log", std::ios::binary);
            log << "I 11 22\nS 1 c1 " << (meta_bytes + ki.stored_bytes) << " 1\n" << ids_line('L', om.live.ids);
            for (const auto& c : om.checkpoints) log << ids_line('C', c.ids);
            log << "Z\n";
        }
        ConversationSpill s(config(q.string()));
        CHECKE(s.open(id, err), "open");
        std::vector<std::string> only = {"c1.kv", "c1.meta"};
        CHECK(s.files() == 1 && folder(q) == only, "the old entry is loaded: its two files stay");
        CHECK(best_for(s, om, 2600).tokens == 2400, "best() finds its checkpoints (the C lines)");
        CHECKE(checkout(s, om, 2600), "checkout reads it");
        CHECK(seq == 1 && same_image(back, om) && kv_same(back.kv, a, sa) && bytes == meta_bytes + ki.stored_bytes && bytes == s.indexed_bytes(),
              "the checkpoints come from the meta file, as written");
        // the next update writes the new layout
        SavedConversation nm = image_n(1, 5600, {1200, 2400, 4000});
        ConversationSpill::StorePark p;
        CHECKE(s.park_store(nm, sb, layout, 1, from, p, err), "update");
        CHECK(p.checkpoints_written == 3 && p.checkpoints_kept == 0 && !p.full, "all 3 checkpoints get their files");
        CHECK(folder(q) == entry_files(1, nm), "the entry now has a file per checkpoint");
        SavedConversation mf;
        CHECK(read_file(q / "c1.meta", mf) && mf.checkpoints.empty() && same_ckpt(mf.live, nm.live), "and a meta file without checkpoints");
        CHECKE(checkout(s, nm, 5000), "checkout");
        CHECK(same_image(back, nm) && kv_same(back.kv, b, sb) && bytes == p.bytes, "reads the checkpoints from their files");
    }
    {
        ConversationSpill s(config((sd / "old").string()));
        CHECKE(s.open(id, err), "reopen the converted folder");
        const SavedConversation nm = image_n(1, 5600, {1200, 2400, 4000});
        CHECKE(s.files() == 1 && checkout(s, nm, 5000) && same_image(back, nm), "it reads back from the log's K lines");
    }
    fs::remove_all(sd);
}

int main() {
    const fs::path dir = fs::temp_directory_path() / ("spill_test_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    const SessionFileIdentity id{11, 22};
    std::string err, why;
    {
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "open");
        const SavedConversation a = image(1, 5000, 2000);

        // a write goes through the thread, the index learns it at drain()
        SavedConversation copy = a;
        CHECK(s.submit(std::move(copy), {}, why), "submit");
        const auto done = finish(s);
        CHECK(done.size() == 1 && done[0].ok && done[0].bytes > 0, "the write finished");
        CHECK(s.files() == 1, "one file indexed");

        // a prompt that continues it matches the live tokens; one that diverges after 2500 tokens matches the checkpoint
        auto m = s.best(prompt_of(a, 5000, 10), {}, true);
        CHECK(m.index == 0 && m.tokens == 5000 && m.live, "live match");
        m = s.best(prompt_of(a, 2500, 300), {}, true);
        CHECK(m.index == 0 && m.tokens == 2000 && !m.live, "checkpoint match");
        CHECK(s.best(prompt_of(a, 5000, 10), {}, false).tokens == 0, "another steering does not match");
        CHECK(s.best(prompt_of(a, 0, 50), {}, true).tokens == 0, "no match");

        // the gate: reading a 4 GiB file to save 17K tokens at 1700 MB/s is worth it, a 100-token gain is not
        SpillConfig g;
        g.disk_mbps = 1700;
        g.prefill_us = 300;
        CHECK(spill_worth(g, 17500, 4ull << 30), "17.5K tokens against 4 GiB: read it");
        CHECK(!spill_worth(g, 100, 4ull << 30), "100 tokens against 4 GiB: read the prompt");
        CHECK(!spill_worth(g, 0, 1), "nothing saved");
        g.disk_mbps = 150;
        CHECK(!spill_worth(g, 17500, 4ull << 30), "a slow disk refuses");

        // divergence: a sibling shares the start (less than half of both) and then differs
        std::vector<int32_t> sib = prompt_of(a, 1500, 4000);
        CHECK(s.divergence(sib) == 1500, "sibling divergence");
        CHECK(s.divergence(prompt_of(a, 5000, 10)) == 0, "the same conversation going on diverges nowhere");
        CHECK(s.divergence(prompt_of(a, 4900, 5000)) == 0, "a late rewrite is not a sibling");
        CHECK(conversation_divergence(sib, a.live.ids) == 1500, "free function agrees");

        // take(): the file is read back whole and leaves the folder
        SavedConversation back;
        size_t bytes = 0;
        m = s.best(prompt_of(a, 5000, 10), {}, true);
        CHECK(s.take(m, SessionReadLimits{}, back, bytes, err), "take");
        CHECK(back.live.ids == a.live.ids && back.checkpoints.size() == 1 && back.checkpoints[0].ids.size() == 2000 &&
              back.live.gdn == a.live.gdn && back.checkpoints[0].gdn == a.checkpoints[0].gdn, "the image came back as it was");
        CHECK(s.files() == 0, "the file left the index");
        size_t left = 0;
        for (const auto& f : fs::directory_iterator(dir)) left += f.path().filename() != "index.log";
        CHECK(left == 0, "and the folder");

        // too short, pictures: refused
        CHECK(!s.submit(image(1, 50, 0), {}, why), "short");
        SavedConversation pic = image(1, 500, 0);
        pic.live.imgs.push_back(ConversationImageKey{});
        CHECK(!s.submit(std::move(pic), {}, why), "pictures");
    }
    {
        // a write that finished but was never drained (the engine stopped first) is in the log and survives the restart
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "open");
        CHECK(s.submit(image(555, 3000, 1000), {}, why), "submit, never drained");
        s.wait_idle();                                   // (no drain())
    }
    {
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "reopen after an undrained write");
        CHECK(s.files() == 1, "the undrained file is indexed");
        CHECK(s.best(prompt_of(image(555, 3000, 0), 3000, 5), {}, true).tokens == 3000, "and matches");
        SavedConversation back;
        size_t bytes = 0;
        CHECK(s.take(s.best(prompt_of(image(555, 3000, 0), 3000, 5), {}, true), SessionReadLimits{}, back, bytes, err), "and reads back");
    }
    {
        // a second image of the same conversation (it went on) replaces the older file; the index survives a restart
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "open");
        CHECK(s.submit(image(1, 3000, 1000), {}, why), "first");
        finish(s);
        CHECK(s.submit(image(1, 4000, 1000), {}, why), "the same conversation, further on");
        finish(s);
        CHECK(s.files() == 1 && s.best(prompt_of(image(1, 4000, 0), 4000, 5), {}, true).tokens == 4000, "the older file is gone");
        CHECK(s.submit(image(100000, 3000, 0), {}, why), "another conversation");
        finish(s);
        CHECK(s.files() == 2, "two files");
    }
    {
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "reopen");
        CHECK(s.files() == 2, "the index was rebuilt from the log");
        const SavedConversation x = image(100000, 3000, 0);
        CHECK(s.best(prompt_of(x, 3000, 5), {}, true).tokens == 3000, "and matches");
    }
    {
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(SessionFileIdentity{11, 23}, err), "open with another configuration");
        CHECK(s.files() == 0, "another configuration uses none of the files");
        size_t left = 0;
        for (const auto& f : fs::directory_iterator(dir)) left += f.path().filename() != "index.log";
        CHECK(left == 0, "and the folder is clean");
    }
    {
        // the quota: the oldest file goes to make room
        SpillConfig c = config(dir.string());
        c.quota_bytes = 250000;       // two files of ~102 KB fit, a third makes the oldest go
        ConversationSpill s(c);
        CHECK(s.open(id, err), "open");
        for (int i = 0; i < 4; ++i) {
            CHECK(s.submit(image(1000000 * (i + 1), 400, 0, 100000), {}, why), "submit under quota");
            finish(s);
        }
        CHECK(s.files() == 2 && s.indexed_bytes() <= c.quota_bytes, "the quota holds: the oldest files went");
    }
    {
        // a cancel stops the write, drops the entry and leaves no temporary file
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "open");
        const uint64_t before = s.files();
        CHECK(s.submit(image(7, 400, 0, 300u << 20), {}, why), "big submit");
        s.cancel_inflight();
        const auto done = s.drain();
        CHECK(s.inflight_bytes() == 0, "nothing in flight after the cancel");
        CHECK(done.size() == 1, "one completion");
        CHECK(s.files() == before + (done[0].ok ? 1 : 0), "an unfinished write is not indexed");
        if (!done[0].ok) {
            size_t stray = 0;
            for (const auto& f : fs::directory_iterator(dir)) {
                const std::string n = f.path().filename().string();
                stray += n != "index.log" && n.rfind("s", 0) != 0;   // (a finished file is s<seq>.sess)
            }
            CHECK(stray == 0, "no temporary file is left");
        }
    }
    {
        // the entry being written: a prompt that goes on with it sees how much of itself is on its way to a file
        ConversationSpill s(config(dir.string()));
        CHECK(s.open(id, err), "open");
        const SavedConversation big = image(4242, 400, 0, 300u << 20);
        SavedConversation copy = big;
        CHECK(s.inflight_prefix(prompt_of(big, 400, 5)) == 0, "nothing in flight before a submit");
        CHECK(s.submit(std::move(copy), {}, why), "submit a large image");
        const int64_t mine = s.inflight_prefix(prompt_of(big, 400, 5)), other = s.inflight_prefix(prompt_of(big, 0, 50));
        CHECK(mine == 400 || mine == 0, "the whole conversation while it is written (0: the write was already done)");
        CHECK(other == 0, "another conversation shares nothing with it");
        s.wait_idle();
        CHECK(s.inflight_prefix(prompt_of(big, 400, 5)) == 0, "nothing in flight after the write");
        const auto done = s.drain();
        CHECK(done.size() == 1 && done[0].ok, "the write finished");
        CHECK(s.best(prompt_of(big, 400, 5), {}, true).tokens == 400, "and the file is found");
    }
    {
        // the drive's speed: measured once for a folder, kept in <dir>/drive-speed, read from there afterwards
        const fs::path d2 = dir / "speed";
        SpillDriveSpeed a, b;
        CHECK(spill_drive_speed(d2.string(), 0, a, err, 64u << 20, 2.0), "the drive is measured");
        CHECK(a.measured_now && a.read_mbps > 0 && a.write_mbps > 0 && a.bytes == (64u << 20), "with a read and a write speed");
        CHECK(fs::exists(d2 / spill_speed_file()), "the result is kept in the folder");
        size_t others = 0;
        for (const auto& f : fs::directory_iterator(d2)) others += f.path().filename() != spill_speed_file();
        CHECK(others == 0, "and the measurement's file is gone");
        CHECK(spill_drive_speed(d2.string(), 0, b, err) && !b.measured_now, "a second call reads the file");
        CHECK(b.bytes == a.bytes && b.direct == a.direct && b.read_mbps > 0.99 * a.read_mbps && b.read_mbps < 1.01 * a.read_mbps,
              "with the same numbers");
        {
            ConversationSpill s(config(d2.string()));
            CHECK(s.open(id, err), "open in the measured folder");
        }
        CHECK(fs::exists(d2 / spill_speed_file()), "open() keeps the drive's speed");
        // 3 GiB at 1500 MB/s read and 1000 MB/s write
        SpillDriveSpeed k;
        k.read_mbps = 1500;
        k.write_mbps = 1000;
        CHECK(spill_read_seconds(k, 3ull << 30) > 2.1 && spill_read_seconds(k, 3ull << 30) < 2.2, "seconds to read back");
        CHECK(spill_write_seconds(k, 3ull << 30) > 3.2 && spill_write_seconds(k, 3ull << 30) < 3.3, "seconds to write");
        // a drive without the free space the tier keeps is not measured; a damaged file is measured again
        SpillDriveSpeed n;
        CHECK(!spill_drive_speed((dir / "noroom").string(), UINT64_MAX / 2, n, err), "no room: not measured");
        {
            std::ofstream f(d2 / spill_speed_file(), std::ios::trunc);
            f << "v1 nonsense";
        }
        CHECK(spill_drive_speed(d2.string(), 0, n, err, 32u << 20, 2.0) && n.measured_now, "a damaged file is measured again");
        std::printf("drive under %s: reads %.0f MB/s, writes %.0f MB/s (%s, %llu MiB)\n", d2.string().c_str(), a.read_mbps,
                    a.write_mbps, a.direct ? "direct" : "through the file cache", (unsigned long long) (a.bytes >> 20));
    }
    {
        // the K/V store file (updated in place) and the store entries of the spill (park_store, checkout)
        kv_store_checks(dir / "kv_direct", "K/V store, O_DIRECT where the filesystem takes it");
        ::setenv("STRATA_SESSION_BUFFERED", "1", 1);
        kv_store_checks(dir / "kv_buffered", "K/V store, buffered I/O");
        ::unsetenv("STRATA_SESSION_BUFFERED");
        std::error_code ec;
        const fs::path shm = fs::path("/dev/shm") / ("kv_store_test_" + std::to_string(::getpid()));
        if (fs::is_directory("/dev/shm", ec)) {
            kv_store_checks(shm, "K/V store on /dev/shm (tmpfs)");
            fs::remove_all(shm, ec);
        }
        store_entry_checks(dir / "store");
        ckpt_file_checks(dir / "ckpt");
        index_compact_checks(dir / "compact");
        g_ctx = "";
    }
    fs::remove_all(dir);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("conversation_spill_test: ok");
    return 0;
}
