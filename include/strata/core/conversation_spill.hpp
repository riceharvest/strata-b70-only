// The conversation cache's SSD tier: a parked conversation the RAM budget evicts is written to a file instead of being
// lost, and a request whose prompt matches it reads it back instead of reading the prompt again (a 70-236K-token prompt
// takes 15-70 s to read; a 2-4 GiB file reads back in about a second on an NVMe drive).
//
// Pure host code (no device calls), one writer thread:
//   serve thread   submit()   an evicted entry goes to the writer thread (one write in flight; another eviction while it
//                             runs is dropped as before); the writer owns the entry's RAM until the file is written
//                  drain()    completed writes enter the index (the serve thread is the only one that touches it)
//                  best()     the longest prefix a spilled file holds for a prompt (its live tokens or any checkpoint)
//                  worth()    the cost gate: reading the file must be cheaper than reading the prompt
//                  take()     reads the file into a SavedConversation (the caller restores it like a taken RAM entry)
//                  cancel_inflight()  the RAM admission needs the memory back now: the write stops at its next block
//   writer thread  session_file_write (conversation_file.hpp: temporary file, 16 MiB direct blocks, flush, rename), paused
//                  while a restore reads (a restore always wins over a write)
// Files and an append-only index.log live in one folder; the index is validated against the files at open().
//
// A second kind of entry, the "store entry" (park_store, checkout): its K/V lives in c<N>.kv, a file that is updated IN
// PLACE (conversation_kv_store.hpp), and everything else in c<N>.meta, an ordinary session file without K/V and without
// checkpoints that is written whole (it is small).  Every checkpoint of the conversation is a file of its own,
// c<N>.k<tokens>-<hash>: a session file whose live state is that checkpoint.  A checkpoint never changes (the state at a
// token count is fixed by the tokens before it), so its file is written once, and a re-park of the same conversation writes
// only the K/V bytes that changed and the checkpoints the entry does not have yet.  It runs on the calling thread, not on
// the writer thread.  A read (checkout) keeps the files: the next park updates them.
#pragma once

#include "strata/core/conversation_file.hpp"
#include "strata/core/conversation_kv_store.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace strata::core {

struct SpillConfig {
    std::string dir;                        // empty: off
    uint64_t quota_bytes = 24ull << 30;     // the files together
    uint64_t min_free_bytes = 40ull << 30;  // the disk keeps at least this free after a file
    int64_t min_tokens = 4096;              // shorter conversations are not worth a write
    double disk_mbps = 1700;                // the SUSTAINED read speed the gate assumes (MB/s; a burst reads faster).  The
                                            // engine puts the folder's measured read speed here (spill_drive_speed)
    double prefill_us = 300;                // what reading one prompt token costs (microseconds)
    bool verify = false;                    // check the payload hash of a session file on a read (a 4 GiB file: seconds).
                                            // A K/V store file (conversation_kv_store.hpp) carries no hash: its header
                                            // and sizes are checked, its bytes are not.
    uint64_t index_compact_bytes = 64ull << 20;   // index.log is written again from the live entries above this size (0: never).
                                            // Every park appends a record with the conversation's tokens (about 1 MB at 150K).
};

// The cost gate: restoring `bytes` from disk to save `saved_tokens` of prompt reading.
inline bool spill_worth(const SpillConfig& c, int64_t saved_tokens, uint64_t bytes) {
    if (saved_tokens <= 0) return false;
    const double read_s = (double) bytes / (std::max(c.disk_mbps, 1.0) * 1e6) + 0.5;
    return (double) saved_tokens * c.prefill_us * 1e-6 >= read_s;
}

// ---- the drive's speed, measured once for a spill folder
// The first start with a spill folder writes and reads back a file there (session_drive_probe; the engine's defaults:
// 1 GiB, at most 3 s each way, --conversation-cache-spill-probe-mib 0 turns it off) and keeps the result in <dir>/drive-speed.  Later starts read that file.  A person sees from it what a
// conversation on the SSD costs against one in RAM, and the cost gate takes its read speed.  Deleting the file
// measures again (another drive, another cable).
struct SpillDriveSpeed {
    double read_mbps = 0, write_mbps = 0;   // MB/s (1e6 bytes a second)
    uint64_t bytes = 0;                      // the size of the file it was measured with
    bool direct = false;                     // false: read through the file cache (read_mbps capped at write_mbps)
    bool measured_now = false;               // this call measured it; false: read from <dir>/drive-speed
};
inline const char* spill_speed_file() { return "drive-speed"; }

inline bool spill_drive_speed_load(const std::string& dir, SpillDriveSpeed& s) {
    std::ifstream f(std::filesystem::path(dir) / spill_speed_file());
    std::string tag;
    SpillDriveSpeed got;
    int direct = 0;
    if (!(f >> tag >> got.read_mbps >> got.write_mbps >> got.bytes >> direct) || tag != "v1") return false;
    if (!(got.read_mbps >= 1 && got.read_mbps < 1e6 && got.write_mbps >= 1 && got.write_mbps < 1e6)) return false;
    got.direct = direct != 0;
    s = got;
    return true;
}

/// The folder's drive speed: from <dir>/drive-speed, else measured now and written there.  `min_free_bytes`: what the
/// drive keeps free (the tier's own rule); a drive without that much and the measurement's file is not measured.
inline bool spill_drive_speed(const std::string& dir, uint64_t min_free_bytes, SpillDriveSpeed& s, std::string& err,
                              uint64_t probe_bytes = 1ull << 30, double max_seconds = 3.0) {
    namespace fs = std::filesystem;
    if (dir.empty()) { err = "no spill folder"; return false; }
    if (spill_drive_speed_load(dir, s)) return true;
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) { err = "cannot create " + dir + ": " + ec.message(); return false; }
    const auto space = fs::space(dir, ec);
    if (!ec && space.available < min_free_bytes + probe_bytes) {
        err = "the drive has less free than the tier keeps free";
        return false;
    }
    SessionDriveSpeed m;
    if (!session_drive_probe(dir, probe_bytes, max_seconds, m, err)) return false;
    s = SpillDriveSpeed{m.read_mbps, m.write_mbps, m.bytes, m.direct, true};
    // written whole under another name and renamed: a start that is stopped here leaves no half file to be read
    const auto tmp = fs::path(dir) / (std::string(spill_speed_file()) + ".tmp");
    {
        std::ofstream f(tmp, std::ios::trunc);
        f << "v1 " << s.read_mbps << ' ' << s.write_mbps << ' ' << s.bytes << ' ' << (s.direct ? 1 : 0) << '\n';
    }
    fs::rename(tmp, fs::path(dir) / spill_speed_file(), ec);   // not kept: the next start measures again
    return true;
}

/// Seconds the drive takes to read back (or to write) a parked conversation of `bytes`.
inline double spill_read_seconds(const SpillDriveSpeed& s, uint64_t bytes) {
    return s.read_mbps > 0 ? (double) bytes / (s.read_mbps * 1e6) : 0.0;
}
inline double spill_write_seconds(const SpillDriveSpeed& s, uint64_t bytes) {
    return s.write_mbps > 0 ? (double) bytes / (s.write_mbps * 1e6) : 0.0;
}

class ConversationSpill {
public:
    struct CheckpointFile {                  // one checkpoint of a store entry: the file c<N>.k<tokens>-<hash>
        uint64_t tokens = 0;                 // the checkpoint's ids.size()
        uint64_t hash = 0;                   // session_hash64 (seed 0) over the bytes of its ids
        uint64_t bytes = 0;                  // the file's size
    };
    struct Entry {
        uint64_t seq = 0;
        std::string file;                    // a name inside the folder (a store entry: the base name c<N>, no suffix)
        uint64_t bytes = 0;                  // what a read touches (a store entry: its meta file, its checkpoint files and its K/V store)
        bool cvec = true;
        bool store = false;                  // a store entry (c<N>.kv, c<N>.meta and c<N>.k*), not one whole file
        ConversationCheckpoint live;         // ids only: what the file holds (the index keeps no state)
        std::vector<ConversationCheckpoint> checkpoints;   // ids only
        // a store entry: the files of its checkpoints, one per item of `checkpoints`, in the same order.  Empty for an entry
        // that an older version wrote: its c<N>.meta holds the checkpoints itself (checkout reads them from there).
        std::vector<CheckpointFile> ckpt_files;
    };
    struct Match {
        size_t index = SIZE_MAX;
        int64_t tokens = 0;
        bool live = false;
    };
    struct Done {                            // a finished write, for the log
        bool ok = false;
        bool cancelled = false;
        uint64_t bytes = 0;
        int64_t tokens = 0;
        std::string file, why;
    };
    struct StorePark {                       // what park_store did
        uint64_t seq = 0;                    // the entry's seq now (a new one on every park)
        uint64_t written = 0;                // K/V payload bytes written (the meta file and the checkpoint files are not counted)
        uint64_t bytes = 0;                  // the entry's Entry::bytes
        bool full = false;                   // new files: every part was written whole
        uint64_t meta = 0;                   // the bytes this park wrote besides the K/V: the meta file (written whole at every
                                             // park) and the checkpoint files THIS call wrote (not the ones it kept)
        uint64_t checkpoints_written = 0, checkpoints_kept = 0;   // checkpoint files written by this call / left as they were
    };

    explicit ConversationSpill(SpillConfig cfg) : cfg_(std::move(cfg)) {}
    ~ConversationSpill() { stop(); }
    ConversationSpill(const ConversationSpill&) = delete;
    ConversationSpill& operator=(const ConversationSpill&) = delete;

    const SpillConfig& config() const { return cfg_; }
    bool enabled() const { return opened_; }
    size_t files() const { return index_.size(); }
    uint64_t indexed_bytes() const {
        uint64_t n = 0;
        for (const auto& e : index_) n += e.bytes;
        return n;
    }
    // RAM an unfinished write still holds (counts against the physical-RAM admission)
    uint64_t inflight_bytes() const { return inflight_.load(); }

    /// Creates the folder, loads the index, drops what does not match the files (or the identity) and starts the writer.
    bool open(const SessionFileIdentity& id, std::string& err) {
        if (cfg_.dir.empty()) { err = "no spill folder"; return false; }
        id_ = id;
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(cfg_.dir, ec);
        if (ec) { err = "cannot create " + cfg_.dir + ": " + ec.message(); return false; }
        load_index();
        // files the index does not know (an interrupted write's temporary file, a dropped entry) go
        std::unordered_set<std::string> known;
        for (const auto& e : index_) {
            if (!e.store) { known.insert(e.file); continue; }
            known.insert(e.file + ".kv");
            known.insert(e.file + ".meta");
            for (const auto& k : e.ckpt_files) known.insert(ckpt_name(e.file, k));
        }
        for (const auto& f : fs::directory_iterator(cfg_.dir, ec)) {
            const std::string name = f.path().filename().string();
            if (name == "index.log" || name == "index.log.tmp" || name == spill_speed_file()) continue;
            if (known.count(name) == 0) fs::remove(f.path(), ec);
        }
        rewrite_index();
        opened_ = true;
        writer_ = std::thread([this] { run(); });
        return true;
    }

    /// An evicted entry: written to a file by the writer thread.  False (the entry is simply dropped, as without a
    /// spill) when it is not worth a file, a write is in flight, or it carries pictures.  `before_hand_off` joins the
    /// entry's background copies (the engine's park/checkpoint async joins), on this thread.
    bool submit(SavedConversation&& image, const std::function<void(const SavedConversation&)>& before_hand_off,
                std::string& why) {
        if (!opened_) { why = "off"; return false; }
        if ((int64_t) image.live.ids.size() < cfg_.min_tokens) { why = "short"; return false; }
        if (!image.live.imgs.empty() || !image.stage_images.empty()) { why = "pictures or a layer split"; return false; }
        for (const auto& c : image.checkpoints)
            if (!c.imgs.empty() || !c.stage_parts.empty()) { why = "pictures or a layer split"; return false; }
        const uint64_t estimate = image.bytes();
        if (estimate > cfg_.quota_bytes) { why = "larger than the quota"; return false; }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (busy_ || job_) { why = "a write is in flight"; return false; }
        }
        if (before_hand_off) before_hand_off(image);
        supersede(image.live.ids);                       // the same conversation, an older file: it goes
        while (!index_.empty() && indexed_bytes() + estimate > cfg_.quota_bytes) {
            const auto it = oldest();
            if (it == index_.end()) break;               // only the held entry is left
            remove_entry(it);
        }
        Job j;
        j.entry.seq = ++seq_;
        j.entry.file = "s" + std::to_string(j.entry.seq) + ".sess";
        j.entry.cvec = image.cvec;
        j.entry.live.ids = image.live.ids;
        for (const auto& c : image.checkpoints) {
            ConversationCheckpoint k;
            k.ids = c.ids;
            j.entry.checkpoints.push_back(std::move(k));
        }
        j.tokens = (int64_t) image.live.ids.size();
        j.image = std::move(image);
        {
            std::lock_guard<std::mutex> lk(mu_);
            inflight_ = estimate;
            busy_ = true;
            inflight_ids_ = j.entry.live.ids;
            job_ = std::move(j);
        }
        cv_.notify_all();
        return true;
    }

    /// How many of the prompt's first tokens the entry that is being written holds (0: no write in flight).  Its file
    /// is not in the index before it is whole, so best() cannot see it: a request that goes on with that conversation
    /// waits for the write (wait_idle, then drain) instead of reading its whole prompt again.
    template<class Token>
    int64_t inflight_prefix(const std::vector<Token>& prompt) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!busy_) return 0;
        const size_t n = std::min(prompt.size(), inflight_ids_.size());
        size_t same = 0;
        while (same < n && (int64_t) prompt[same] == (int64_t) inflight_ids_[same]) ++same;
        return (int64_t) same;
    }

    /// Writes that finished since the last call enter the index.
    std::vector<Done> drain() {
        std::deque<Finished> got;
        {
            std::lock_guard<std::mutex> lk(mu_);
            got.swap(done_);
        }
        std::vector<Done> out;
        for (auto& f : got) {
            if (f.done.ok) {                                         // (its index record is in the log already: the writer
                f.entry.bytes = f.done.bytes;                        //  appended it when the file was complete, so a stop before
                index_.push_back(std::move(f.entry));                //  this call does not lose the file)
            }
            out.push_back(std::move(f.done));
        }
        maybe_compact();
        return out;
    }

    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match m;
        for (size_t i = index_.size(); i-- > 0;) {
            const auto& e = index_[i];
            if (e.cvec != cvec) continue;
            const int64_t n = conversation_prefix(e.live, prompt, images);
            if (n > m.tokens) m = {i, n, true};
            for (const auto& c : e.checkpoints) {
                const int64_t k = conversation_prefix(c, prompt, images);
                if (k > m.tokens) m = {i, k, false};
            }
        }
        return m;
    }
    /// The longest divergence point between the prompt and a spilled conversation (conversation_divergence).
    template<class Token>
    int64_t divergence(const std::vector<Token>& prompt) const {
        int64_t best = 0;
        for (const auto& e : index_) best = std::max(best, conversation_divergence(prompt, e.live.ids));
        return best;
    }
    bool worth(const Match& m, int64_t already_have) const {
        return m.index < index_.size() && spill_worth(cfg_, m.tokens - already_have, index_[m.index].bytes);
    }
    uint64_t bytes_of(const Match& m) const { return m.index < index_.size() ? index_[m.index].bytes : 0; }

    /// Reads the matched file into `image` and removes it from the folder (the conversation goes on in the session; its
    /// next eviction writes it again).  A file that cannot be read is removed too.
    bool take(const Match& m, const SessionReadLimits& limits_in, SavedConversation& image, size_t& bytes, std::string& err) {
        if (m.index >= index_.size()) { err = "no such file"; return false; }
        const Entry e = index_[m.index];
        if (e.store) { err = "a store entry has no whole file: read it with checkout()"; return false; }   // (it stays)
        restore_.fetch_add(1);
        SessionReadLimits limits = limits_in;
        limits.skip_payload_hash = !cfg_.verify;
        SessionStatus st;
        const bool ok = session_file_read((std::filesystem::path(cfg_.dir) / e.file).string(), id_, image, bytes, err,
                                          limits, &st);
        restore_.fetch_sub(1);
        remove_entry(std::find_if(index_.begin(), index_.end(), [&](const Entry& x) { return x.seq == e.seq; }));
        return ok;
    }

    /// Parks a conversation as a store entry, on the calling thread.  `meta` is the image WITHOUT K/V (meta.kv must be
    /// empty); `sources` is its K/V.  The image's checkpoints are not in the meta file: each one goes to a file of its own
    /// (c<N>.k<tokens>-<hash>), written once.  `meta` is moved from for the writes (its checkpoints are held aside, one at a
    /// time a checkpoint is the live state of a temporary image) and is exactly as it came in when the call returns,
    /// whichever way it ends.  Two checkpoints with the same tokens are refused.
    /// `base_seq` != 0 and naming a store entry (StorePark::seq of an earlier park): its files are updated in place -
    /// `from[j][p]` is the first byte of layer j's part p that may differ from what the K/V file holds - and the entry gets
    /// `meta`'s tokens and a new seq.  A checkpoint the base entry already has (the same tokens, the same hash of them, its
    /// file still there) keeps its file; a checkpoint it does not have is written; a file of the base that is no longer
    /// wanted is removed once the new files are whole.  Otherwise (or when that entry is gone) new files are written, every
    /// part whole (`from` is ignored), and out.full is set.  False (nothing was changed) when it is off, too short, carries
    /// pictures, or does not fit the quota or the free space the folder keeps; false after a failed write too: then every
    /// file of the entry is removed (the conversation is simply not parked).
    /// The index keeps the order that makes a stop safe at any point: the old record is deleted BEFORE its files change,
    /// the new record is appended after every file is whole.
    bool park_store(SavedConversation& meta, const std::vector<SessionKvSource>& sources, const KvStoreLayout& layout,
                    uint64_t base_seq, const std::vector<std::array<uint64_t, 5>>& from, StorePark& out, std::string& why) {
        namespace fs = std::filesystem;
        out = {};
        if (!opened_) { why = "off"; return false; }
        if ((int64_t) meta.live.ids.size() < cfg_.min_tokens) { why = "short"; return false; }
        if (!meta.kv.empty()) { why = "the meta image must carry no K/V"; return false; }
        if (!meta.live.imgs.empty() || !meta.live.stage_parts.empty() || !meta.stage_images.empty()) {
            why = "pictures or a layer split";
            return false;
        }
        for (const auto& c : meta.checkpoints)
            if (!c.imgs.empty() || !c.stage_parts.empty()) { why = "pictures or a layer split"; return false; }
        if (sources.size() != layout.capacity.size()) { why = "the K/V sources do not match the layout"; return false; }
        // a checkpoint file is named by its tokens and the hash of them: two of the same name would be one file
        std::vector<CheckpointFile> want;
        uint64_t ckpt_ram = 0;
        for (const auto& c : meta.checkpoints) {
            CheckpointFile f;
            f.tokens = c.ids.size();
            f.hash = ids_hash(c.ids);
            for (const auto& w : want)
                if (w.tokens == f.tokens && w.hash == f.hash) { why = "two checkpoints hold the same tokens"; return false; }
            want.push_back(f);
            ckpt_ram += c.bytes();
        }
        uint64_t kv_bytes = kKvStoreHeaderBytes;         // what the K/V store will hold (Entry::bytes counts it so): an upper bound
        for (const auto& s : sources)
            for (uint64_t n : s.sizes) kv_bytes += (n + 4095) / 4096 * 4096;
        // meta.bytes() holds the checkpoints too: the meta file is the rest of it
        const uint64_t ram = (uint64_t) meta.bytes();
        const uint64_t core = (ram > ckpt_ram ? ram - ckpt_ram : 0) + kv_bytes;       // the meta file and the K/V store
        auto base = base_seq ? std::find_if(index_.begin(), index_.end(), [&](const Entry& x) { return x.store && x.seq == base_seq; })
                             : index_.end();
        const bool update = base != index_.end();
        if (update && from.size() != sources.size()) { why = "one `from` per layer is needed"; return false; }
        // the files of the base that stay: the same tokens and the same hash, the file there with the size the index recorded
        std::vector<CheckpointFile> base_files;
        std::vector<bool> keep(want.size(), false);
        uint64_t base_core = 0;
        if (update) {
            base_files = base->ckpt_files;
            uint64_t base_ckpt = 0;
            for (const auto& b : base_files) base_ckpt += b.bytes;
            base_core = base->bytes > base_ckpt ? base->bytes - base_ckpt : 0;
            for (size_t i = 0; i < want.size(); ++i)
                for (const auto& b : base_files) {
                    if (b.tokens != want[i].tokens || b.hash != want[i].hash) continue;
                    std::error_code ec;
                    const auto sz = fs::file_size(path(ckpt_name(base->file, b)), ec);
                    if (!ec && sz == b.bytes) { keep[i] = true; want[i].bytes = b.bytes; }
                }
        }
        uint64_t new_ckpt = 0, kept_ckpt = 0;            // the checkpoint files this call writes (an upper bound) and keeps
        for (size_t i = 0; i < want.size(); ++i) {
            if (keep[i]) kept_ckpt += want[i].bytes;
            else new_ckpt += (uint64_t) meta.checkpoints[i].bytes() + kCkptFileSlack;
        }
        const uint64_t estimate = core + kept_ckpt + new_ckpt;      // the entry's size: what the quota counts
        if (estimate > cfg_.quota_bytes) { why = "larger than the quota"; return false; }
        if (cfg_.min_free_bytes) {                       // new data only: an update writes what grew, and the new checkpoint files
            std::error_code ec;
            const auto space = fs::space(cfg_.dir, ec);
            const uint64_t need = update ? (core > base_core ? core - base_core : 0) + new_ckpt : estimate;
            if (ec || space.available < need || space.available - need < cfg_.min_free_bytes) {
                why = "not enough disk space";
                return false;
            }
        }
        // ---- from here the folder changes
        Entry e;
        e.store = true;
        e.cvec = meta.cvec;
        e.live.ids = meta.live.ids;
        for (const auto& c : meta.checkpoints) {
            ConversationCheckpoint k;
            k.ids = c.ids;
            e.checkpoints.push_back(std::move(k));
        }
        e.seq = ++seq_;
        if (update) {
            e.file = base->file;
            index_.erase(base);                          // its record is deleted before its files change: a stop leaves
            append_del(base_seq);                        // files no record points to, and open() removes those
        } else {
            e.file = "c" + std::to_string(e.seq);
        }
        supersede(e.live.ids);                           // other older files of the same conversation (the base is out)
        while (!index_.empty() && indexed_bytes() + estimate > cfg_.quota_bytes) {
            const auto it = oldest();
            if (it == index_.end()) break;               // only the held entry is left
            remove_entry(it);
        }
        const std::string kv_path = path(e.file + ".kv").string(), meta_path = path(e.file + ".meta").string();
        // however the call ends before the new record is written, every file of the entry goes (kept checkpoint files too:
        // the entry is gone) and the image gets its checkpoints back
        Cleanup cleanup;
        cleanup.fn = [&] {
            std::error_code ec;
            fs::remove(kv_path, ec);
            fs::remove(meta_path, ec);
            for (const auto& w : want) fs::remove(path(ckpt_name(e.file, w)), ec);
            for (const auto& b : base_files) fs::remove(path(ckpt_name(e.file, b)), ec);
        };
        CheckpointsAside aside(meta);
        std::vector<std::array<uint64_t, 5>> zero;
        if (!update) {
            zero.assign(sources.size(), std::array<uint64_t, 5>{});
            std::error_code ec;
            fs::remove(kv_path, ec);                     // (a name nothing indexes: no stale file may be updated by accident)
        }
        uint64_t written = 0;
        KvStoreInfo info;
        if (!kv_store_write(kv_path, id_, layout, sources, update ? from : zero, (uint64_t) e.live.ids.size(), !update, written,
                            why) ||
            !kv_store_info(kv_path, id_, layout, info, why))
            return false;
        SessionWriteOptions opt;
        opt.min_free_bytes = cfg_.min_free_bytes;
        size_t meta_bytes = 0;
        SessionStatus st;
        if (!session_file_write(meta_path, meta, id_, meta_bytes, why, opt, &st)) return false;   // (meta holds no checkpoints now)
        uint64_t ckpt_written_bytes = 0, n_written = 0, n_kept = 0;
        for (size_t i = 0; i < want.size(); ++i) {
            if (keep[i]) { ++n_kept; continue; }
            size_t n = 0;
            bool ok = false;
            {
                CheckpointAsLive one(meta, aside.held[i]);
                ok = session_file_write(path(ckpt_name(e.file, want[i])).string(), one.image, id_, n, why, opt, &st);
            }
            if (!ok) return false;
            want[i].bytes = (uint64_t) n;
            ckpt_written_bytes += (uint64_t) n;
            ++n_written;
        }
        // every new file is whole: the files of the base that no checkpoint wants any more go
        for (const auto& b : base_files) {
            bool wanted = false;
            for (const auto& w : want) wanted = wanted || (w.tokens == b.tokens && w.hash == b.hash);
            std::error_code ec;
            if (!wanted) fs::remove(path(ckpt_name(e.file, b)), ec);
        }
        uint64_t ckpt_bytes = 0;
        for (const auto& w : want) ckpt_bytes += w.bytes;
        e.ckpt_files = want;
        e.bytes = (uint64_t) meta_bytes + info.stored_bytes + ckpt_bytes;
        append_add(e);
        cleanup.armed = false;
        out.seq = e.seq;
        out.written = written;
        out.bytes = e.bytes;
        out.full = !update;
        out.meta = (uint64_t) meta_bytes + ckpt_written_bytes;
        out.checkpoints_written = n_written;
        out.checkpoints_kept = n_kept;
        index_.push_back(std::move(e));
        maybe_compact();
        return true;
    }

    /// Reads a store entry into `image` (its meta file, then its checkpoint files in the entry's order, then its K/V file)
    /// and KEEPS its files and its index entry: the next park_store of the same conversation updates them in place.  `seq` =
    /// the entry's seq (park_store's base_seq).  `bytes` = what was read (Entry::bytes).  Each read follows the caller's
    /// limits, on its own (`limits.admit` is asked once for each file, for what that read holds at its peak: it does not
    /// count what the earlier reads of the call keep in RAM).
    /// An entry that an older version wrote (it lists no checkpoint files, its meta file holds the checkpoints) is read from
    /// the meta file as it is.  A whole-file entry behaves exactly like take() (the file is read and removed) and sets seq =
    /// 0.  An entry that cannot be read is removed (files too) and the call returns false; one that is only refused for RAM
    /// (the limits' admit) stays.  `image` is replaced only on success.
    /// `kv_left` (a store entry only): the K/V is NOT read - `image.kv` stays empty and `*kv_left` gets the K/V file's
    /// header - for a caller that streams it into place afterwards (stream_kv), without the K/V in RAM.
    bool checkout(const Match& m, const SessionReadLimits& limits_in, const KvStoreLayout& layout, SavedConversation& image,
                  size_t& bytes, uint64_t& seq, std::string& err, KvStoreInfo* kv_left = nullptr) {
        seq = 0;
        if (m.index >= index_.size()) { err = "no such file"; return false; }
        const Entry e = index_[m.index];
        if (!e.store) return take(m, limits_in, image, bytes, err);
        restore_.fetch_add(1);
        SessionReadLimits limits = limits_in;
        limits.skip_payload_hash = !cfg_.verify;
        const std::string kv_path = path(e.file + ".kv").string();
        SavedConversation got;
        size_t meta_bytes = 0;
        SessionStatus st;
        KvStoreInfo info;
        // a read that is refused for RAM is not a damaged entry: the files stay for the conversation's next request
        bool keep = false;
        bool ok = session_file_read(path(e.file + ".meta").string(), id_, got, meta_bytes, err, limits, &st);
        if (!ok && st.error == SessionError::memory) keep = true;
        if (ok && !got.kv.empty()) { err = "the meta file of a store entry carries K/V"; ok = false; }
        if (ok && !e.ckpt_files.empty()) {               // (none listed: an older version's entry, its meta file holds them)
            if (!got.checkpoints.empty()) {
                err = "the meta file of a store entry carries checkpoints that are in files of their own";
                ok = false;
            } else if (e.ckpt_files.size() > limits.max_checkpoints) {
                err = "the entry has " + std::to_string(e.ckpt_files.size()) + " checkpoints, above this engine's limit " +
                      std::to_string(limits.max_checkpoints);
                ok = false;
            }
        }
        ok = ok && kv_store_info(kv_path, id_, layout, info, err);
        if (ok && info.layers.size() > limits.max_kv_layers) {
            err = "the K/V store has " + std::to_string(info.layers.size()) + " layers, above this engine's limit " +
                  std::to_string(limits.max_kv_layers);
            ok = false;
        }
        for (size_t j = 0; ok && j < info.layers.size() && j < limits.max_kv_bytes.size(); ++j)
            for (size_t p = 0; ok && p < 5; ++p)
                if (info.layers[j].sizes[p] > limits.max_kv_bytes[j][p]) {
                    err = "the K/V store holds more bytes in layer " + std::to_string(j) + " part " + std::to_string(p) +
                          " than this engine accepts";
                    ok = false;
                }
        if (ok && info.tokens != (uint64_t) got.live.ids.size()) {
            err = "the K/V store and the meta file of a store entry do not hold the same conversation";
            ok = false;
        }
        // the checkpoint files, in the entry's order
        uint64_t ckpt_bytes = 0;
        for (size_t i = 0; ok && i < e.ckpt_files.size(); ++i) {
            const CheckpointFile& f = e.ckpt_files[i];
            SavedConversation one;
            size_t n = 0;
            st = {};
            ok = session_file_read(path(ckpt_name(e.file, f)).string(), id_, one, n, err, limits, &st);
            if (!ok) {
                if (st.error == SessionError::memory) keep = true;
                break;
            }
            if (!one.kv.empty() || !one.checkpoints.empty()) {
                err = "a checkpoint file of a store entry carries K/V or more checkpoints";
                ok = false;
            } else if (one.live.ids.size() != f.tokens || ids_hash(one.live.ids) != f.hash) {
                err = "a checkpoint file of a store entry does not hold the tokens its name says";
                ok = false;
            }
            if (!ok) break;
            ckpt_bytes += (uint64_t) n;
            got.checkpoints.push_back(std::move(one.live));
        }
        if (ok && limits.admit && !kv_left) {
            std::string why;
            if (!limits.admit(session_read_peak_bytes(info.stored_bytes), why)) {
                err = why.empty() ? "refused by the memory check" : why;
                ok = false;
                keep = true;
            }
        }
        if (!kv_left) ok = ok && kv_store_read(kv_path, id_, layout, got.kv, info, err);   // (the memory check was asked above)
        restore_.fetch_sub(1);
        if (!ok) {
            if (!keep) remove_entry(std::find_if(index_.begin(), index_.end(), [&](const Entry& x) { return x.seq == e.seq; }));
            return false;
        }
        image = std::move(got);
        bytes = meta_bytes + (size_t) ckpt_bytes + (kv_left ? 0 : (size_t) info.stored_bytes);
        if (kv_left) *kv_left = info;
        seq = e.seq;
        return true;
    }
    /// The K/V of the store entry `seq`, handed to `sink` block by block (kv_store_stream) instead of being built in RAM.
    /// A read that fails leaves the entry in the index: the caller decides (drop()).
    bool stream_kv(uint64_t seq, const KvStoreLayout& layout,
                   const std::function<bool(const KvStoreInfo& info, std::string& why)>& begin,
                   const std::function<bool(size_t layer, size_t part, uint64_t offset, const uint8_t* data, size_t n,
                                            std::string& why)>& sink,
                   KvStoreInfo& info, std::string& err, const std::vector<std::array<uint64_t, 5>>* only = nullptr) {
        for (const auto& e : index_) {
            if (e.seq != seq || !e.store) continue;
            restore_.fetch_add(1);
            const bool ok = kv_store_stream(path(e.file + ".kv").string(), id_, layout, begin, sink, info, err, only);
            restore_.fetch_sub(1);
            return ok;
        }
        err = "no such store entry";
        return false;
    }
    /// The entry `seq` is being streamed into the session (checkout with kv_left, then stream_kv): until hold(0) the
    /// quota and a newer park of the same conversation leave its files alone.
    void hold(uint64_t seq) { held_ = seq; }
    bool is_store(const Match& m) const { return m.index < index_.size() && index_[m.index].store; }
    uint64_t seq_of(const Match& m) const { return m.index < index_.size() ? index_[m.index].seq : 0; }
    uint64_t compactions() const { return compactions_; }
    /// Removes the entry `seq` and its files (a park whose file did not pass a check must never be read back).
    bool drop(uint64_t seq) {
        const auto it = std::find_if(index_.begin(), index_.end(), [&](const Entry& x) { return x.seq == seq; });
        if (it == index_.end()) return false;
        remove_entry(it);
        return true;
    }

    /// How many of `ids`' first tokens the entry `seq` holds (its live tokens; 0: no such entry).  The engine asks what a
    /// store entry still has of the conversation it is about to update.
    template<class Token>
    int64_t common_prefix(uint64_t seq, const std::vector<Token>& ids) const {
        for (const auto& e : index_) {
            if (e.seq != seq) continue;
            const size_t n = std::min(ids.size(), e.live.ids.size());
            size_t same = 0;
            while (same < n && (int64_t) ids[same] == (int64_t) e.live.ids[same]) ++same;
            return (int64_t) same;
        }
        return 0;
    }
    /// The K/V of the store entry `seq`, read without changing the entry (the engine's STRATA_SPILL_STORE_VERIFY compares it
    /// with the session's).  False when there is no such store entry; a file that cannot be read stays in the index.
    bool read_kv(uint64_t seq, const KvStoreLayout& layout, std::vector<ConversationKv>& kv, std::string& err) {
        for (const auto& e : index_) {
            if (e.seq != seq || !e.store) continue;
            KvStoreInfo info;
            restore_.fetch_add(1);
            const bool ok = kv_store_read(path(e.file + ".kv").string(), id_, layout, kv, info, err);
            restore_.fetch_sub(1);
            return ok;
        }
        err = "no such store entry";
        return false;
    }

    /// The RAM admission needs an in-flight entry's memory back: the write stops at its next block, the entry is dropped.
    void cancel_inflight() {
        std::unique_lock<std::mutex> lk(mu_);
        if (!busy_) return;
        cancel_ = true;
        idle_.wait(lk, [&] { return !busy_; });
        cancel_ = false;
    }
    /// Blocks until no write is in flight (tests, shutdown).
    void wait_idle() {
        std::unique_lock<std::mutex> lk(mu_);
        idle_.wait(lk, [&] { return !busy_ && !job_; });
    }

private:
    struct Job {
        Entry entry;
        int64_t tokens = 0;
        SavedConversation image;
    };
    struct Finished {
        Entry entry;
        Done done;
    };
    using Iter = std::vector<Entry>::iterator;

    std::filesystem::path path(const std::string& name) const { return std::filesystem::path(cfg_.dir) / name; }
    // what a checkpoint is called in a file name: session_hash64 (seed 0) over the bytes of its ids
    static uint64_t ids_hash(const std::vector<int32_t>& ids) {
        return session_hash64(ids.data(), ids.size() * sizeof(int32_t), 0);
    }
    // c<N>.k<tokens>-<hash>, the hash as 16 lower-case hex digits
    static std::string ckpt_name(const std::string& base, const CheckpointFile& f) {
        char hex[17];
        std::snprintf(hex, sizeof hex, "%016llx", (unsigned long long) f.hash);
        return base + ".k" + std::to_string(f.tokens) + "-" + hex;
    }
    // a file's size beyond what the checkpoint holds in RAM (its header, its counts): the estimate adds this to each one
    static constexpr uint64_t kCkptFileSlack = 512;

    // The checkpoints of an image, held aside while a park writes (the meta file is written without them) and put back when
    // the park ends, however it ends.
    struct CheckpointsAside {
        SavedConversation& image;
        std::vector<ConversationCheckpoint> held;
        explicit CheckpointsAside(SavedConversation& i) : image(i), held(std::move(i.checkpoints)) { image.checkpoints.clear(); }
        CheckpointsAside(const CheckpointsAside&) = delete;
        CheckpointsAside& operator=(const CheckpointsAside&) = delete;
        ~CheckpointsAside() { image.checkpoints = std::move(held); }
    };
    // One checkpoint as the live state of a temporary image (the geometry, layer range and steering of `meta`), so that it is
    // written as a session file without a copy of its state.  It is moved back when the object goes.
    struct CheckpointAsLive {
        ConversationCheckpoint& from;
        SavedConversation image;
        CheckpointAsLive(const SavedConversation& meta, ConversationCheckpoint& c) : from(c) {
            image.geometry = meta.geometry;
            image.layer_lo = meta.layer_lo;
            image.layer_hi = meta.layer_hi;
            image.cvec = meta.cvec;
            image.live = std::move(from);
        }
        CheckpointAsLive(const CheckpointAsLive&) = delete;
        CheckpointAsLive& operator=(const CheckpointAsLive&) = delete;
        ~CheckpointAsLive() { from = std::move(image.live); }
    };
    // Runs `fn` when it goes out of scope unless `armed` was cleared (a park that fails, or throws, removes its files).
    struct Cleanup {
        std::function<void()> fn;
        bool armed = true;
        Cleanup() = default;
        Cleanup(const Cleanup&) = delete;
        Cleanup& operator=(const Cleanup&) = delete;
        ~Cleanup() { if (armed && fn) fn(); }
    };
    // the entry written longest ago, never the held one (index_.end(): there is no other)
    Iter oldest() {
        Iter best = index_.end();
        for (Iter it = index_.begin(); it != index_.end(); ++it)
            if (it->seq != held_ && (best == index_.end() || it->seq < best->seq)) best = it;
        return best;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
            cancel_ = true;
        }
        cv_.notify_all();
        if (writer_.joinable()) writer_.join();
    }

    void remove_entry(Iter it) {
        if (it == index_.end()) return;
        std::error_code ec;
        if (it->store) {
            std::filesystem::remove(path(it->file + ".kv"), ec);
            std::filesystem::remove(path(it->file + ".meta"), ec);
            for (const auto& k : it->ckpt_files) std::filesystem::remove(path(ckpt_name(it->file, k)), ec);
        } else {
            std::filesystem::remove(path(it->file), ec);
        }
        append_del(it->seq);
        index_.erase(it);
    }
    // an older file of the same conversation (its tokens are a prefix of the new image's) is stale
    void supersede(const std::vector<int32_t>& ids) {
        for (size_t i = 0; i < index_.size();) {
            const auto& old = index_[i].live.ids;
            if (index_[i].seq != held_ && !old.empty() && old.size() <= ids.size() && std::equal(old.begin(), old.end(), ids.begin()))
                remove_entry(index_.begin() + (std::ptrdiff_t) i);
            else
                ++i;
        }
    }

    // ---- the index: append-only text records; a torn last record is ignored
    //   I <model> <config>                    the identity every file was written with
    //   A <seq> <file> <bytes> <cvec>  \n  L <ids...>  \n  (C <ids...> \n)*  Z     one file
    //   S <seq> <name> <bytes> <cvec>  \n  (K <tokens> <hash> <bytes> \n)*  L ...  \n  (C ...)*  Z
    //                                         a store entry (<name>.kv, <name>.meta and the checkpoint files).  One K line per
    //                                         checkpoint file <name>.k<tokens>-<hash16> (<hash> is decimal here, 16 hex digits
    //                                         in the name), in the order of the C lines.  A record without K lines is an older
    //                                         version's: its meta file holds the checkpoints.
    //   D <seq>                               an entry removed (a store entry updated in place: its old record)
    static void put_ids(std::ostringstream& o, char tag, const std::vector<int32_t>& ids) {
        o << tag;
        for (int32_t t : ids) o << ' ' << t;
        o << '\n';
    }
    static void put_entry(std::ostringstream& o, const Entry& e) {
        o << (e.store ? 'S' : 'A') << ' ' << e.seq << ' ' << e.file << ' ' << e.bytes << ' ' << (e.cvec ? 1 : 0) << '\n';
        for (const auto& k : e.ckpt_files) o << "K " << k.tokens << ' ' << k.hash << ' ' << k.bytes << '\n';
        put_ids(o, 'L', e.live.ids);
        for (const auto& c : e.checkpoints) put_ids(o, 'C', c.ids);
        o << "Z\n";
    }
    void append(const std::string& text) {
        std::lock_guard<std::mutex> lk(idx_mu_);                  // the writer thread appends too
        std::ofstream f(path("index.log"), std::ios::app | std::ios::binary);
        f << text;
        f.flush();
    }
    void append_add(const Entry& e) {
        std::ostringstream o;
        put_entry(o, e);
        append(o.str());
    }
    void append_del(uint64_t seq) { append("D " + std::to_string(seq) + "\n"); }

    void load_index() {
        index_.clear();
        std::ifstream f(path("index.log"), std::ios::binary);
        std::string line;
        bool identity_ok = false, seen_identity = false;
        std::optional<Entry> cur;
        auto ids_of = [](std::istringstream& s) {
            std::vector<int32_t> v;
            long long t;
            while (s >> t) v.push_back((int32_t) t);
            return v;
        };
        while (std::getline(f, line)) {
            std::istringstream s(line);
            std::string tag;
            s >> tag;
            if (tag == "I") {
                unsigned long long m = 0, c = 0;
                s >> m >> c;
                seen_identity = true;
                identity_ok = m == id_.model && c == id_.config;
                if (!identity_ok) index_.clear();
            } else if (!identity_ok) {
                continue;                                       // another model or configuration: nothing of it is used
            } else if (tag == "A" || tag == "S") {
                cur.emplace();
                int cv = 1;
                s >> cur->seq >> cur->file >> cur->bytes >> cv;
                cur->cvec = cv != 0;
                cur->store = tag == "S";
            } else if (tag == "K" && cur && cur->store) {
                CheckpointFile k;
                if (s >> k.tokens >> k.hash >> k.bytes)
                    cur->ckpt_files.push_back(k);
                else
                    cur.reset();                                // a damaged line: the record is not used
            } else if (tag == "L" && cur) {
                cur->live.ids = ids_of(s);
            } else if (tag == "C" && cur) {
                ConversationCheckpoint c;
                c.ids = ids_of(s);
                cur->checkpoints.push_back(std::move(c));
            } else if (tag == "Z" && cur) {
                seq_ = std::max(seq_, cur->seq);
                index_.push_back(std::move(*cur));
                cur.reset();
            } else if (tag == "D") {
                uint64_t seq = 0;
                s >> seq;
                index_.erase(std::remove_if(index_.begin(), index_.end(), [&](const Entry& e) { return e.seq == seq; }), index_.end());
            }
        }
        (void) seen_identity;
        // an entry whose file is missing or has another size is gone (a store entry: when its K/V file, its meta file or one of
        // its checkpoint files is missing; its K/V file is sparse and updated in place, so its sizes say nothing)
        std::error_code ec;
        index_.erase(std::remove_if(index_.begin(), index_.end(), [&](const Entry& e) {
            if (e.store) {
                if (!std::filesystem::exists(path(e.file + ".kv"), ec) || !std::filesystem::exists(path(e.file + ".meta"), ec)) return true;
                for (const auto& k : e.ckpt_files)
                    if (!std::filesystem::exists(path(ckpt_name(e.file, k)), ec)) return true;
                // the K lines and the C lines describe the same checkpoints (best() matches on the C lines)
                if (!e.ckpt_files.empty()) {
                    if (e.ckpt_files.size() != e.checkpoints.size()) return true;
                    for (size_t i = 0; i < e.ckpt_files.size(); ++i)
                        if (e.checkpoints[i].ids.size() != e.ckpt_files[i].tokens || ids_hash(e.checkpoints[i].ids) != e.ckpt_files[i].hash)
                            return true;
                }
                return false;
            }
            const auto sz = std::filesystem::file_size(path(e.file), ec);
            return ec || sz != e.bytes;
        }), index_.end());
    }
    // the log written again from the index (atomically): the identity first
    // The log holds every record since the folder was opened (a park appends one, an update or a removal a `D`).  Above
    // SpillConfig::index_compact_bytes it is written again from the live entries.  Serve thread only, and only when the
    // writer thread has nothing in flight or undrained: its finished file is in the log before it is in `index_`.
    void maybe_compact() {
        if (!opened_ || !cfg_.index_compact_bytes || compact_failed_) return;
        std::error_code ec;
        const auto size = std::filesystem::file_size(path("index.log"), ec);
        if (ec || size <= cfg_.index_compact_bytes) return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (busy_ || job_ || !done_.empty()) return;
        }
        std::lock_guard<std::mutex> lk(idx_mu_);
        if (rewrite_index()) ++compactions_;
        else compact_failed_ = true;   // a full or failing disk: the log keeps growing, not tried again in this process
    }
    // The new log takes the old one's place only when all of it is on the disk: a full disk or a stop in between
    // leaves the old log, which still names every file (a short or empty log would have every file removed as unknown
    // at the next open).
    bool rewrite_index() {
        std::ostringstream o;
        o << "I " << id_.model << ' ' << id_.config << '\n';
        for (const auto& e : index_) put_entry(o, e);
        const auto tmp = path("index.log.tmp");
        bool ok = false;
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f << o.str();
            f.flush();
            ok = f.good();
        }
        if (ok) {
            const int fd = ::open(tmp.c_str(), O_RDONLY);
            ok = fd >= 0 && ::fsync(fd) == 0;
            if (fd >= 0) ::close(fd);
        }
        std::error_code ec;
        if (ok) {
            std::filesystem::rename(tmp, path("index.log"), ec);
            ok = !ec;
        }
        if (!ok) std::filesystem::remove(tmp, ec);
        return ok;
    }

    // ---- the writer thread
    void run() {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return job_.has_value() || stop_; });
                if (!job_) return;
                j = std::move(*job_);
                job_.reset();
            }
            Finished fin;
            fin.entry = std::move(j.entry);
            fin.done.file = fin.entry.file;
            fin.done.tokens = j.tokens;
            SessionWriteOptions opt;
            opt.min_free_bytes = cfg_.min_free_bytes;
            opt.cancel = [this] {
                // a restore reads from the same disk: it goes first, the write waits for it (or for a cancel)
                while (restore_.load() > 0 && !cancel_.load()) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                return cancel_.load();
            };
            size_t bytes = 0;
            std::string err;
            SessionStatus st;
            const bool ok = !cancel_.load() && session_file_write(path(fin.entry.file).string(), j.image, id_, bytes, err, opt, &st);
            fin.done.ok = ok;
            fin.done.cancelled = cancel_.load();
            fin.done.bytes = bytes;
            if (ok) {
                fin.entry.bytes = bytes;
                append_add(fin.entry);
            }
            fin.done.why = ok ? "" : (fin.done.cancelled || err.empty() ? "cancelled" : err);
            j.image = SavedConversation{};                       // the entry's memory goes back to the kernel here
            {
                std::lock_guard<std::mutex> lk(mu_);
                inflight_ = 0;
                busy_ = false;
                inflight_ids_.clear();
                done_.push_back(std::move(fin));
            }
            idle_.notify_all();
        }
    }

    SpillConfig cfg_;
    SessionFileIdentity id_;
    bool opened_ = false;
    uint64_t seq_ = 0;
    std::vector<Entry> index_;                // serve thread only
    std::thread writer_;
    std::mutex mu_, idx_mu_;                  // mu_: the job and completion queues; idx_mu_: appends to index.log
    std::condition_variable cv_, idle_;
    std::optional<Job> job_;
    std::deque<Finished> done_;
    bool busy_ = false, stop_ = false;
    std::vector<int32_t> inflight_ids_;       // the tokens of the entry being written (under mu_)
    uint64_t compactions_ = 0;
    bool compact_failed_ = false;
    uint64_t held_ = 0;                       // hold(): an entry no eviction removes (0: none)
    std::atomic<bool> cancel_{false};
    std::atomic<int> restore_{0};
    std::atomic<uint64_t> inflight_{0};
};

} // namespace strata::core
