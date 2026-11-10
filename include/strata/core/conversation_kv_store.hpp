// A K/V store file: the K/V of one parked conversation, kept in a file that is updated IN PLACE.  A conversation's K/V is a
// list of layers of five parts each (k, v, k_scale, v_scale, pooled: flat byte arrays).  When the conversation grows, each
// part only grows at its end, so a re-park needs to write only the bytes from the first one that changed.  A whole session
// file (conversation_file.hpp) is written again from the first byte.  This file is not.
//
// Layout of the file (every part has its own fixed place, so the file is sparse until a part is written):
//   [0, 65536)   the header block (the first bytes of it are used, the rest is zero)
//   then, for every layer in order and for every part in order, `capacity` rounded up to 4096 bytes
// The header is little-endian: magic "STRATAKV", version u32 = 1, state u32 (0 whole, 1 being written), model u64,
// config u64, tokens u64, layer count u64; per layer 7 x i64 (format, cells, heads, head_dim, page_size, pooled_rows,
// idx_dim) and 5 x u64 (the bytes of each part now); then a u64 hash (session_hash64, seed 0) of all bytes before it.
// 682 layers fit in the block.
//
// Crash safety: an update first writes the header with state 1 and the old sizes, then the bytes, then the header with
// state 0 and the new sizes.  A file whose state is 1 is not read and not updated: it is thrown away by the caller.
//
// I/O: 4096-byte aligned blocks of up to 16 MiB through pread/pwrite, with O_DIRECT when the filesystem takes it, buffered
// otherwise (STRATA_SESSION_BUFFERED=1 forces buffered, as for session files).  POSIX only.
#pragma once

#include "strata/core/conversation_file.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::core {

struct KvStoreLayout {                       // fixed for one engine configuration
    std::vector<std::array<uint64_t, 5>> capacity;   // bytes each part of each layer may grow to
};
struct KvStoreLayer {                        // what the header keeps per layer
    int64_t format = 0, cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    std::array<uint64_t, 5> sizes{};         // bytes of each part now
};
// stored_bytes: what a read of the whole store touches: the header block plus every part's size rounded up to 4096.
struct KvStoreInfo { uint64_t tokens = 0; std::vector<KvStoreLayer> layers; uint64_t stored_bytes = 0; };

inline constexpr uint64_t kKvStoreHeaderBytes = 65536;
inline constexpr size_t kKvStoreMaxLayers = 682;

// The file offset of a part: the header block, then every layer's parts in order, each capacity rounded up to 4096.
// UINT64_MAX for a layer or part that is not in the layout.
uint64_t kv_store_offset(const KvStoreLayout& layout, size_t layer, size_t part);
// The size the file has (sparse: most of it holds no data).  UINT64_MAX when it does not fit in a file offset.
uint64_t kv_store_file_size(const KvStoreLayout& layout);

// Writes bytes [from[j][p], sources[j].sizes[p]) of every part into its place, then the header.
// create=true: a new file (must not exist; every from must be 0).  create=false: the file must exist with a valid
// header of the same identity and layer count, and every from[j][p] must be <= the size the header holds for that part
// (no gap).  A size above its capacity is refused before anything is written.  `written` = payload bytes written by
// this call (the part's bytes from its first 4096 block that holds a changed byte; the zero padding of the last block
// is not counted).  `tokens` is stored in the header as the caller's tag.  A `from` above the part's new size is the
// same as `from` = that size (the bytes before it are unchanged, and there are no more).
bool kv_store_write(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                    const std::vector<SessionKvSource>& sources, const std::vector<std::array<uint64_t, 5>>& from,
                    uint64_t tokens, bool create, uint64_t& written, std::string& error);
// Reads the header only (identity, state, hash and layout checked).
bool kv_store_info(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                   KvStoreInfo& info, std::string& error);
// Reads every layer into `kv`.  Before allocating, if `admit` is set it is asked for the bytes the read holds at its peak
// (session_read_peak_bytes of the K/V) and a refusal fails the read with its reason.  On a failure `kv` is empty.
bool kv_store_read(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                   std::vector<ConversationKv>& kv, KvStoreInfo& info, std::string& error,
                   const std::function<bool(uint64_t need_bytes, std::string& why)>& admit = {});

// Reads the file without building its layers in RAM: after the header is checked, `begin(info, why)` may refuse (false:
// the call fails with `why`); then every part is handed to `sink(layer, part, offset, data, n, why)` in blocks of up to
// 16 MiB, in file order (layer, part, rising offset).  A false from the sink stops the read with its `why`.
// `only` (optional, one entry per layer): at most that many leading bytes of each part are read and handed over (a
// prefix of the conversation: a part's first bytes are its first tokens).
bool kv_store_stream(const std::string& path, const SessionFileIdentity& id, const KvStoreLayout& layout,
                     const std::function<bool(const KvStoreInfo& info, std::string& why)>& begin,
                     const std::function<bool(size_t layer, size_t part, uint64_t offset, const uint8_t* data, size_t n,
                                              std::string& why)>& sink,
                     KvStoreInfo& info, std::string& error, const std::vector<std::array<uint64_t, 5>>* only = nullptr);

} // namespace strata::core
