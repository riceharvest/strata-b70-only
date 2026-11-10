#include "strata/core/conversation_kv_store.hpp"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

static_assert(std::endian::native == std::endian::little, "K/V store files are little-endian only");

namespace strata::core {
namespace {
constexpr uint64_t kAlign = 4096;
constexpr size_t kBlock = 16u << 20;
constexpr size_t kFixedHead = 48;      // magic, version, state, model, config, tokens, layer count
constexpr size_t kLayerHead = 96;      // 7 x i64 and 5 x u64
constexpr uint32_t kVersion = 1, kWhole = 0, kWriting = 1;
constexpr char kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 'K', 'V'};
const char* const kTag = "kv store: ";

uint64_t round_up(uint64_t n) { return n > UINT64_MAX - (kAlign - 1) ? UINT64_MAX : (n + kAlign - 1) / kAlign * kAlign; }

// the place of every part and the size of the file; false when the sum does not fit in a signed file offset
bool place(const KvStoreLayout& layout, std::vector<std::array<uint64_t, 5>>& at, uint64_t& total) {
    at.assign(layout.capacity.size(), {});
    total = kKvStoreHeaderBytes;
    for (size_t j = 0; j < layout.capacity.size(); ++j)
        for (size_t p = 0; p < 5; ++p) {
            at[j][p] = total;
            const uint64_t c = round_up(layout.capacity[j][p]);
            if (c > (uint64_t) std::numeric_limits<int64_t>::max() - total) return false;
            total += c;
        }
    return true;
}
} // namespace

uint64_t kv_store_offset(const KvStoreLayout& layout, size_t layer, size_t part) {
    std::vector<std::array<uint64_t, 5>> at;
    uint64_t total = 0;
    if (layer >= layout.capacity.size() || part >= 5 || !place(layout, at, total)) return UINT64_MAX;
    return at[layer][part];
}

uint64_t kv_store_file_size(const KvStoreLayout& layout) {
    std::vector<std::array<uint64_t, 5>> at;
    uint64_t total = 0;
    return place(layout, at, total) ? total : UINT64_MAX;
}

#ifdef _WIN32

bool kv_store_write(const std::string&, const SessionFileIdentity&, const KvStoreLayout&, const std::vector<SessionKvSource>&,
                    const std::vector<std::array<uint64_t, 5>>&, uint64_t, bool, uint64_t& written, std::string& error) {
    written = 0;
    error = "not supported on Windows";
    return false;
}
bool kv_store_info(const std::string&, const SessionFileIdentity&, const KvStoreLayout&, KvStoreInfo&, std::string& error) {
    error = "not supported on Windows";
    return false;
}
bool kv_store_read(const std::string&, const SessionFileIdentity&, const KvStoreLayout&, std::vector<ConversationKv>&,
                   KvStoreInfo&, std::string& error, const std::function<bool(uint64_t, std::string&)>&) {
    error = "not supported on Windows";
    return false;
}

bool kv_store_stream(const std::string&, const SessionFileIdentity&, const KvStoreLayout&,
                     const std::function<bool(const KvStoreInfo&, std::string&)>&,
                     const std::function<bool(size_t, size_t, uint64_t, const uint8_t*, size_t, std::string&)>&, KvStoreInfo&,
                     std::string& error, const std::vector<std::array<uint64_t, 5>>*) {
    error = "not supported on Windows";
    return false;
}
#else

namespace {
struct FreeBuf { void operator()(uint8_t* p) const { std::free(p); } };
using Buf = std::unique_ptr<uint8_t, FreeBuf>;
Buf aligned_buf(size_t n) { return Buf(static_cast<uint8_t*>(std::aligned_alloc(kAlign, n))); }

bool buffered_forced() {
    const char* e = std::getenv("STRATA_SESSION_BUFFERED");
    return e != nullptr && e[0] == '1';
}

// One file, opened without following a link.  O_DIRECT is switched on the open descriptor: a filesystem that refuses it
// (EINVAL, e.g. tmpfs) keeps buffered I/O, and so does one that refuses a single aligned transfer.
class File {
public:
    ~File() { close(); }
    bool open(const std::string& path, int flags, std::string& error) {
        fd_ = ::open(path.c_str(), flags | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd_ < 0) { error = path + ": " + std::strerror(errno); return false; }
        struct stat st {};
        if (::fstat(fd_, &st) != 0) { error = path + ": " + std::strerror(errno); return false; }
        if (!S_ISREG(st.st_mode)) { error = path + ": not a regular file"; return false; }
        size_ = (uint64_t) st.st_size;
        if (O_DIRECT != 0 && !buffered_forced()) {
            const int fl = ::fcntl(fd_, F_GETFL);
            direct_ = fl >= 0 && ::fcntl(fd_, F_SETFL, fl | O_DIRECT) == 0;
        }
        return true;
    }
    bool is_open() const { return fd_ >= 0; }
    bool direct() const { return direct_; }
    uint64_t size() const { return size_; }
    bool truncate(uint64_t n) {
        if (::ftruncate(fd_, (off_t) n) != 0) { error_ = std::strerror(errno); return false; }
        size_ = n;
        return true;
    }
    uint64_t allocated() const {   // the bytes the filesystem holds for the file now (0 when it cannot say)
        struct stat st {};
        return ::fstat(fd_, &st) == 0 ? (uint64_t) st.st_blocks * 512 : 0;
    }
    bool read_at(uint8_t* p, size_t n, uint64_t off) {
        while (n) {
            const ssize_t r = ::pread(fd_, p, n, (off_t) off);
            if (r < 0 && errno == EINTR) continue;
            if (r < 0 && errno == EINVAL && direct_ && leave_direct()) continue;
            if (r <= 0) { error_ = r < 0 ? std::strerror(errno) : "the file ends early"; return false; }
            p += r; n -= (size_t) r; off += (uint64_t) r;
        }
        return true;
    }
    bool write_at(const uint8_t* p, size_t n, uint64_t off) {
        while (n) {
            const ssize_t w = ::pwrite(fd_, p, n, (off_t) off);
            if (w < 0 && errno == EINTR) continue;
            if (w < 0 && errno == EINVAL && direct_ && leave_direct()) continue;   // nothing was written: go buffered
            if (w <= 0) { error_ = w < 0 ? std::strerror(errno) : "short write"; return false; }
            p += w; n -= (size_t) w; off += (uint64_t) w;
        }
        return true;
    }
    bool sync() {
        if (::fsync(fd_) == 0) return true;
        error_ = std::strerror(errno);
        return false;
    }
    bool close() {
        if (fd_ < 0) return true;
        const bool ok = ::close(fd_) == 0;
        if (!ok) error_ = std::strerror(errno);
        fd_ = -1;
        return ok;
    }
    const std::string& error() const { return error_; }
private:
    bool leave_direct() {
        const int fl = ::fcntl(fd_, F_GETFL);
        if (fl < 0 || ::fcntl(fd_, F_SETFL, fl & ~O_DIRECT) != 0) return false;
        direct_ = false;
        return true;
    }
    int fd_ = -1;
    bool direct_ = false;
    uint64_t size_ = 0;
    std::string error_;
};

size_t header_used(size_t layers) { return kFixedHead + layers * kLayerHead + 8; }

// The header bytes into `h` (a block of at least the returned length, which is a multiple of 4096): the used part, its
// hash, zeros after it.
size_t put_header(uint8_t* h, const SessionFileIdentity& id, uint32_t state, uint64_t tokens,
                  const std::vector<KvStoreLayer>& layers) {
    const size_t used = header_used(layers.size()), len = (size_t) round_up(used);
    std::memset(h, 0, len);
    const uint64_t n = layers.size();
    std::memcpy(h, kMagic, 8);
    std::memcpy(h + 8, &kVersion, 4);
    std::memcpy(h + 12, &state, 4);
    std::memcpy(h + 16, &id.model, 8);
    std::memcpy(h + 24, &id.config, 8);
    std::memcpy(h + 32, &tokens, 8);
    std::memcpy(h + 40, &n, 8);
    uint8_t* at = h + kFixedHead;
    for (const auto& l : layers) {
        for (int64_t v : {l.format, l.cells, l.heads, l.head_dim, l.page_size, l.pooled_rows, l.idx_dim}) {
            std::memcpy(at, &v, 8);
            at += 8;
        }
        for (uint64_t v : l.sizes) {
            std::memcpy(at, &v, 8);
            at += 8;
        }
    }
    const uint64_t hash = session_hash64(h, used - 8, 0);
    std::memcpy(h + used - 8, &hash, 8);
    return len;
}

bool fail(std::string& error, const std::string& msg) {
    error = kTag + msg;
    return false;
}

// Reads the header block and checks it: shape, hash, identity, state, the layout.  `buf` holds 65536 aligned bytes.
bool read_header(File& f, uint8_t* buf, const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                 uint64_t file_size, KvStoreInfo& info, std::string& error) {
    if (f.size() < kKvStoreHeaderBytes) return fail(error, path + " is too short for a header");
    if (!f.read_at(buf, kKvStoreHeaderBytes, 0)) return fail(error, "reading the header of " + path + ": " + f.error());
    if (std::memcmp(buf, kMagic, 8) != 0) return fail(error, path + " is not a K/V store file");
    uint32_t version = 0, state = 0;
    uint64_t model = 0, config = 0, tokens = 0, n = 0;
    std::memcpy(&version, buf + 8, 4);
    std::memcpy(&state, buf + 12, 4);
    std::memcpy(&model, buf + 16, 8);
    std::memcpy(&config, buf + 24, 8);
    std::memcpy(&tokens, buf + 32, 8);
    std::memcpy(&n, buf + 40, 8);
    if (version != kVersion) return fail(error, path + " has version " + std::to_string(version) + ", not " + std::to_string(kVersion));
    if (n > kKvStoreMaxLayers) return fail(error, path + " names " + std::to_string(n) + " layers: more than a header holds");
    const size_t used = header_used((size_t) n);
    uint64_t hash = 0;
    std::memcpy(&hash, buf + used - 8, 8);
    if (hash != session_hash64(buf, used - 8, 0)) return fail(error, "the header of " + path + " is damaged (hash mismatch)");
    if (model != id.model || config != id.config) return fail(error, path + " was written for another model or configuration");
    if (state != kWhole) return fail(error, path + " was being written when it stopped (state " + std::to_string(state) + ")");
    if (n != layout.capacity.size())
        return fail(error, path + " holds " + std::to_string(n) + " layers, the layout has " + std::to_string(layout.capacity.size()));
    if (f.size() != file_size) return fail(error, path + " has a size that does not match the layout");
    info = {};
    info.tokens = tokens;
    info.layers.resize((size_t) n);
    info.stored_bytes = kKvStoreHeaderBytes;
    const uint8_t* at = buf + kFixedHead;
    for (size_t j = 0; j < (size_t) n; ++j) {
        KvStoreLayer& l = info.layers[j];
        int64_t* nums[7] = {&l.format, &l.cells, &l.heads, &l.head_dim, &l.page_size, &l.pooled_rows, &l.idx_dim};
        for (int64_t* v : nums) {
            std::memcpy(v, at, 8);
            at += 8;
        }
        for (size_t p = 0; p < 5; ++p) {
            std::memcpy(&l.sizes[p], at, 8);
            at += 8;
            if (l.sizes[p] > layout.capacity[j][p])
                return fail(error, path + ": layer " + std::to_string(j) + " part " + std::to_string(p) + " holds " +
                                       std::to_string(l.sizes[p]) + " bytes, above its capacity " + std::to_string(layout.capacity[j][p]));
            info.stored_bytes += round_up(l.sizes[p]);
        }
    }
    return true;
}
} // namespace

bool kv_store_write(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                    const std::vector<SessionKvSource>& sources, const std::vector<std::array<uint64_t, 5>>& from,
                    uint64_t tokens, bool create, uint64_t& written, std::string& error) {
    written = 0;
    error.clear();
    // ---- refusals that touch no file
    const size_t layers = layout.capacity.size();
    if (layers > kKvStoreMaxLayers) return fail(error, std::to_string(layers) + " layers do not fit in a header");
    if (sources.size() != layers) return fail(error, std::to_string(sources.size()) + " K/V sources for a layout of " + std::to_string(layers) + " layers");
    if (from.size() != layers) return fail(error, "one `from` per layer is needed");
    std::vector<std::array<uint64_t, 5>> at;
    uint64_t total = 0;
    if (!place(layout, at, total)) return fail(error, "the layout is larger than a file can be");
    for (size_t j = 0; j < layers; ++j)
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t size = sources[j].sizes[p];
            if (size > layout.capacity[j][p])
                return fail(error, "layer " + std::to_string(j) + " part " + std::to_string(p) + " has " + std::to_string(size) +
                                       " bytes, above its capacity " + std::to_string(layout.capacity[j][p]));
            if (create && from[j][p] != 0) return fail(error, "a new file is written whole: `from` must be 0");
            if (size && !sources[j].read) return fail(error, "layer " + std::to_string(j) + " has no reader");
        }
    std::vector<KvStoreLayer> now(layers);
    for (size_t j = 0; j < layers; ++j) {
        const SessionKvSource& s = sources[j];
        now[j].format = s.format;
        now[j].cells = s.cells;
        now[j].heads = s.heads;
        now[j].head_dim = s.head_dim;
        now[j].page_size = s.page_size;
        now[j].pooled_rows = s.pooled_rows;
        now[j].idx_dim = s.idx_dim;
        for (size_t p = 0; p < 5; ++p) now[j].sizes[p] = s.sizes[p];
    }
    const Buf buf = aligned_buf(kBlock);
    if (!buf) return fail(error, "out of memory for the write buffer");

    // ---- the file
    File f;
    bool created = false;
    auto stop = [&](const std::string& msg) {
        f.close();
        if (created) ::unlink(path.c_str());             // a new file that was not finished is not left behind
        return fail(error, msg);
    };
    if (create) {
        if (!f.open(path, O_RDWR | O_CREAT | O_EXCL, error)) { error = kTag + error; return false; }
        created = true;
        if (!f.truncate(total)) return stop("sizing " + path + ": " + f.error());
        // The layout counts on a sparse file: the spill's quota and free-space floor count the bytes stored, not the
        // file's size.  A filesystem that allocates the whole size at once (exFAT, FAT32, some network mounts) would
        // take every part's full capacity for each conversation: refused there, the park is skipped with this line.
        if (total >= (64ull << 20) && f.allocated() >= total / 2)
            return stop(path + ": the folder's filesystem allocated the whole file (no sparse files; ext4, xfs and btrfs "
                               "keep them): the K/V store is not written there");
    } else {
        if (!f.open(path, O_RDWR, error)) { error = kTag + error; return false; }
        KvStoreInfo old;
        if (!read_header(f, buf.get(), path, id, layout, total, old, error)) return false;
        for (size_t j = 0; j < layers; ++j)
            for (size_t p = 0; p < 5; ++p)
                if (from[j][p] > old.layers[j].sizes[p])
                    return fail(error, "layer " + std::to_string(j) + " part " + std::to_string(p) + ": `from` " + std::to_string(from[j][p]) +
                                           " is beyond the " + std::to_string(old.layers[j].sizes[p]) + " bytes the file holds (a gap)");
        // the file is not whole from here until the last header: the old sizes stay in it, with the state "being written"
        const size_t len = put_header(buf.get(), id, kWriting, old.tokens, old.layers);
        if (!f.write_at(buf.get(), len, 0) || !f.sync()) return stop("marking " + path + " as being written: " + f.error());
    }

    // ---- the bytes: from the 4096 block that holds the first changed byte to the end of the part
    for (size_t j = 0; j < layers; ++j)
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t size = sources[j].sizes[p];
            const uint64_t start = std::min(from[j][p], size);   // before `from` the bytes are the file's already
            if (start >= size) continue;
            for (uint64_t pos = start / kAlign * kAlign; pos < size;) {
                const uint64_t n = std::min<uint64_t>(kBlock, size - pos);
                if (!sources[j].read(p, (size_t) pos, buf.get(), (size_t) n)) {
                    std::string why = sources[j].error ? sources[j].error() : std::string();
                    return stop("reading layer " + std::to_string(j) + " part " + std::to_string(p) + " at " + std::to_string(pos) +
                                " failed" + (why.empty() ? "" : ": " + why));
                }
                const uint64_t padded = round_up(n);   // the padding stays inside the part's place: its capacity is rounded up too
                std::memset(buf.get() + n, 0, (size_t) (padded - n));
                if (!f.write_at(buf.get(), (size_t) padded, at[j][p] + pos)) return stop("writing " + path + ": " + f.error());
                written += n;
                pos += n;
            }
        }
    if (!f.sync()) return stop("flushing " + path + ": " + f.error());   // the bytes are on disk before the header says so
    const size_t len = put_header(buf.get(), id, kWhole, tokens, now);
    if (!f.write_at(buf.get(), len, 0) || !f.sync()) return stop("writing the header of " + path + ": " + f.error());
    if (!f.close()) return stop("closing " + path + ": " + f.error());
    return true;
}

bool kv_store_info(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                   KvStoreInfo& info, std::string& error) {
    error.clear();
    if (layout.capacity.size() > kKvStoreMaxLayers) return fail(error, "the layout has more layers than a header holds");
    std::vector<std::array<uint64_t, 5>> at;
    uint64_t total = 0;
    if (!place(layout, at, total)) return fail(error, "the layout is larger than a file can be");
    const Buf buf = aligned_buf((size_t) kKvStoreHeaderBytes);
    if (!buf) return fail(error, "out of memory for the header");
    File f;
    if (!f.open(path, O_RDONLY, error)) { error = kTag + error; return false; }
    KvStoreInfo got;
    if (!read_header(f, buf.get(), path, id, layout, total, got, error)) return false;
    info = std::move(got);
    return true;
}

bool kv_store_read(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                   std::vector<ConversationKv>& kv, KvStoreInfo& info, std::string& error,
                   const std::function<bool(uint64_t need_bytes, std::string& why)>& admit) {
    error.clear();
    kv.clear();
    std::vector<std::array<uint64_t, 5>> at;
    uint64_t total = 0;
    if (layout.capacity.size() > kKvStoreMaxLayers) return fail(error, "the layout has more layers than a header holds");
    if (!place(layout, at, total)) return fail(error, "the layout is larger than a file can be");
    File f;
    KvStoreInfo got;
    {
        const Buf head = aligned_buf((size_t) kKvStoreHeaderBytes);
        if (!head) return fail(error, "out of memory for the header");
        if (!f.open(path, O_RDONLY, error)) { error = kTag + error; return false; }
        if (!read_header(f, head.get(), path, id, layout, total, got, error)) return false;
    }
    for (const auto& l : got.layers)
        if (l.format < INT32_MIN || l.format > INT32_MAX) return fail(error, "invalid K/V format in " + path);
    if (admit) {
        std::string why;
        if (!admit(session_read_peak_bytes(got.stored_bytes), why)) return fail(error, why.empty() ? "refused by the memory check" : why);
    }
    const Buf buf = aligned_buf(kBlock);
    if (!buf) return fail(error, "out of memory for the read buffer");
    try {
        kv.resize(got.layers.size());
        for (size_t j = 0; j < got.layers.size(); ++j) {
            const KvStoreLayer& l = got.layers[j];
            ConversationKv& out = kv[j];
            out.format = (int) l.format;
            out.cells = l.cells;
            out.heads = l.heads;
            out.head_dim = l.head_dim;
            out.page_size = l.page_size;
            out.pooled_rows = l.pooled_rows;
            out.idx_dim = l.idx_dim;
            ConversationBuffer* parts[5] = {&out.k, &out.v, &out.k_scale, &out.v_scale, &out.pooled};
            for (size_t p = 0; p < 5; ++p) {
                const uint64_t size = l.sizes[p];
                ConversationBuffer& b = *parts[p];
                b = {};
                b.resize_for_overwrite((size_t) size);   // every byte is read below before anything sees it
                for (uint64_t pos = 0; pos < size;) {
                    const uint64_t n = std::min<uint64_t>(kBlock, size - pos);
                    if (!f.read_at(buf.get(), (size_t) round_up(n), at[j][p] + pos)) {
                        kv.clear();
                        return fail(error, "reading layer " + std::to_string(j) + " part " + std::to_string(p) + " of " + path + ": " + f.error());
                    }
                    const uint8_t* src = buf.get();
                    b.visit((size_t) pos, (size_t) n, [&](uint8_t* d, size_t c, size_t from_pos) {
                        std::memcpy(d, src + (from_pos - (size_t) pos), c);
                        return true;
                    });
                    pos += n;
                }
            }
        }
    } catch (const std::bad_alloc&) {
        kv.clear();
        return fail(error, "out of memory for the K/V of " + path);
    }
    info = std::move(got);
    return true;
}

bool kv_store_stream(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                     const std::function<bool(const KvStoreInfo& info, std::string& why)>& begin,
                     const std::function<bool(size_t layer, size_t part, uint64_t offset, const uint8_t* data, size_t n,
                                              std::string& why)>& sink,
                     KvStoreInfo& info, std::string& error, const std::vector<std::array<uint64_t, 5>>* only) {
    error.clear();
    std::vector<std::array<uint64_t, 5>> at;
    uint64_t total = 0;
    if (layout.capacity.size() > kKvStoreMaxLayers) return fail(error, "the layout has more layers than a header holds");
    if (!place(layout, at, total)) return fail(error, "the layout is larger than a file can be");
    if (!sink) return fail(error, "no sink for the K/V blocks");
    if (only && only->size() != layout.capacity.size()) return fail(error, "one prefix size per layer is needed");
    File f;
    KvStoreInfo got;
    {
        const Buf head = aligned_buf((size_t) kKvStoreHeaderBytes);
        if (!head) return fail(error, "out of memory for the header");
        if (!f.open(path, O_RDONLY, error)) { error = kTag + error; return false; }
        if (!read_header(f, head.get(), path, id, layout, total, got, error)) return false;
    }
    std::string why;
    if (begin && !begin(got, why)) return fail(error, why.empty() ? "refused" : why);
    const Buf buf = aligned_buf(kBlock);
    if (!buf) return fail(error, "out of memory for the read buffer");
    for (size_t j = 0; j < got.layers.size(); ++j)
        for (size_t p = 0; p < 5; ++p) {
            const uint64_t size = only ? std::min<uint64_t>(got.layers[j].sizes[p], (*only)[j][p]) : got.layers[j].sizes[p];
            for (uint64_t pos = 0; pos < size;) {
                const uint64_t n = std::min<uint64_t>(kBlock, size - pos);
                if (!f.read_at(buf.get(), (size_t) round_up(n), at[j][p] + pos))
                    return fail(error, "reading layer " + std::to_string(j) + " part " + std::to_string(p) + " of " + path + ": " + f.error());
                if (!sink(j, p, pos, buf.get(), (size_t) n, why))
                    return fail(error, why.empty() ? "the K/V could not be taken" : why);
                pos += n;
            }
        }
    info = std::move(got);
    return true;
}


#endif

} // namespace strata::core
