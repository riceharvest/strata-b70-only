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
#pragma once

#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace strata::core {

struct SpillConfig {
    std::string dir;                        // empty: off
    uint64_t quota_bytes = 24ull << 30;     // the files together
    uint64_t min_free_bytes = 40ull << 30;  // the disk keeps at least this free after a file
    int64_t min_tokens = 4096;              // shorter conversations are not worth a write
    double disk_mbps = 1700;                // the SUSTAINED read speed the gate assumes (MB/s; a burst reads faster)
    double prefill_us = 300;                // what reading one prompt token costs (microseconds)
    bool verify = false;                    // check the payload hash on a read (a 4 GiB file: seconds)
};

// The cost gate: restoring `bytes` from disk to save `saved_tokens` of prompt reading.
inline bool spill_worth(const SpillConfig& c, int64_t saved_tokens, uint64_t bytes) {
    if (saved_tokens <= 0) return false;
    const double read_s = (double) bytes / (std::max(c.disk_mbps, 1.0) * 1e6) + 0.5;
    return (double) saved_tokens * c.prefill_us * 1e-6 >= read_s;
}

class ConversationSpill {
public:
    struct Entry {
        uint64_t seq = 0;
        std::string file;                    // a name inside the folder
        uint64_t bytes = 0;
        bool cvec = true;
        ConversationCheckpoint live;         // ids only: what the file holds (the index keeps no state)
        std::vector<ConversationCheckpoint> checkpoints;
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
        for (const auto& f : fs::directory_iterator(cfg_.dir, ec)) {
            const std::string name = f.path().filename().string();
            if (name == "index.log" || name == "index.log.tmp") continue;
            bool known = false;
            for (const auto& e : index_) known = known || e.file == name;
            if (!known) fs::remove(f.path(), ec);
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
        while (!index_.empty() && indexed_bytes() + estimate > cfg_.quota_bytes) remove_entry(oldest());
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
            job_ = std::move(j);
        }
        cv_.notify_all();
        return true;
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
            if (f.done.ok) {
                f.entry.bytes = f.done.bytes;
                append_add(f.entry);
                index_.push_back(std::move(f.entry));
            }
            out.push_back(std::move(f.done));
        }
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
    Iter oldest() { return std::min_element(index_.begin(), index_.end(), [](const Entry& a, const Entry& b) { return a.seq < b.seq; }); }

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
        std::filesystem::remove(path(it->file), ec);
        append_del(it->seq);
        index_.erase(it);
    }
    // an older file of the same conversation (its tokens are a prefix of the new image's) is stale
    void supersede(const std::vector<int32_t>& ids) {
        for (size_t i = 0; i < index_.size();) {
            const auto& old = index_[i].live.ids;
            if (!old.empty() && old.size() <= ids.size() && std::equal(old.begin(), old.end(), ids.begin()))
                remove_entry(index_.begin() + (std::ptrdiff_t) i);
            else
                ++i;
        }
    }

    // ---- the index: append-only text records; a torn last record is ignored
    //   I <model> <config>                    the identity every file was written with
    //   A <seq> <file> <bytes> <cvec>  \n  L <ids...>  \n  (C <ids...> \n)*  Z     one file
    //   D <seq>                               a file removed
    static void put_ids(std::ostringstream& o, char tag, const std::vector<int32_t>& ids) {
        o << tag;
        for (int32_t t : ids) o << ' ' << t;
        o << '\n';
    }
    void append(const std::string& text) {
        std::ofstream f(path("index.log"), std::ios::app | std::ios::binary);
        f << text;
        f.flush();
    }
    void append_add(const Entry& e) {
        std::ostringstream o;
        o << "A " << e.seq << ' ' << e.file << ' ' << e.bytes << ' ' << (e.cvec ? 1 : 0) << '\n';
        put_ids(o, 'L', e.live.ids);
        for (const auto& c : e.checkpoints) put_ids(o, 'C', c.ids);
        o << "Z\n";
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
            } else if (tag == "A") {
                cur.emplace();
                int cv = 1;
                s >> cur->seq >> cur->file >> cur->bytes >> cv;
                cur->cvec = cv != 0;
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
        // an entry whose file is missing or has another size is gone
        std::error_code ec;
        index_.erase(std::remove_if(index_.begin(), index_.end(), [&](const Entry& e) {
            const auto sz = std::filesystem::file_size(path(e.file), ec);
            return ec || sz != e.bytes;
        }), index_.end());
    }
    // the log written again from the index (atomically): the identity first
    void rewrite_index() {
        std::ostringstream o;
        o << "I " << id_.model << ' ' << id_.config << '\n';
        for (const auto& e : index_) {
            o << "A " << e.seq << ' ' << e.file << ' ' << e.bytes << ' ' << (e.cvec ? 1 : 0) << '\n';
            put_ids(o, 'L', e.live.ids);
            for (const auto& c : e.checkpoints) put_ids(o, 'C', c.ids);
            o << "Z\n";
        }
        const auto tmp = path("index.log.tmp");
        {
            std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
            f << o.str();
        }
        std::error_code ec;
        std::filesystem::rename(tmp, path("index.log"), ec);
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
            fin.done.why = ok ? "" : (fin.done.cancelled || err.empty() ? "cancelled" : err);
            j.image = SavedConversation{};                       // the entry's memory goes back to the kernel here
            {
                std::lock_guard<std::mutex> lk(mu_);
                inflight_ = 0;
                busy_ = false;
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
    std::mutex mu_;
    std::condition_variable cv_, idle_;
    std::optional<Job> job_;
    std::deque<Finished> done_;
    bool busy_ = false, stop_ = false;
    std::atomic<bool> cancel_{false};
    std::atomic<int> restore_{0};
    std::atomic<uint64_t> inflight_{0};
};

} // namespace strata::core
