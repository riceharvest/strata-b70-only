// Host-only test of the conversation cache's SSD tier (conversation_spill.hpp): no device.  Builds small
// SavedConversations (running state and checkpoints, no K/V), writes them through the spill writer thread and reads
// them back.  Run: conversation_spill_test
#include "strata/core/conversation_spill.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <unistd.h>

using namespace strata::core;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(cond, what) do { if (!(cond)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, what); ++failures; } } while (0)

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
    fs::remove_all(dir);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::puts("conversation_spill_test: ok");
    return 0;
}
