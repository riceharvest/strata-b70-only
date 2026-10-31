// --session-save-reclaim: what goes when a SAVE needs RAM.  The selected (deepest) checkpoint and explicit pins stay;
// the system prompt's end (root) goes only when no other checkpoint is left to free.
#include "strata/core/session_save_reclaim.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
constexpr size_t MiB = size_t(1) << 20;
ConversationCheckpoint ck(int len, bool root = false, bool pinned = false) {
    ConversationCheckpoint c;
    c.ids.assign((size_t) len, 1);
    c.gdn.resize(MiB);
    c.root = root;
    c.pinned = pinned;
    return c;
}
size_t total(const std::vector<ConversationCheckpoint>& chain) {
    size_t n = 0;
    for (const auto& c : chain) n += c.bytes();
    return n;
}
bool has(const std::vector<ConversationCheckpoint>& chain, int len) {
    for (const auto& c : chain) if ((int) c.ids.size() == len) return true;
    return false;
}
// RAM short of the SAVE's need by `short_by`; every byte the chain frees counts.
SessionSaveReclaimed run(std::vector<ConversationCheckpoint>& chain, size_t short_by) {
    ConversationCache cache(64 * MiB, 4);
    const SessionSaveLive live{};
    const uint64_t need = session_save_peak_bytes(session_deepest_checkpoint(chain), live);
    const size_t start = total(chain);
    auto probe = [&]() -> std::optional<uint64_t> { return need - short_by + (start - total(chain)); };
    return session_save_reclaim(chain, cache, live, 0, probe);
}
} // namespace

int main() {
    {   // freeing the middle one is enough: root stays
        std::vector<ConversationCheckpoint> chain{ck(100, true), ck(200), ck(300)};
        const auto out = run(chain, MiB / 2);
        check(out.checkpoints == 1, "one checkpoint freed");
        check(has(chain, 100) && has(chain, 300) && !has(chain, 200), "root and deepest kept, middle freed");
    }
    {   // a periodic checkpoint inside the system prompt goes before root
        std::vector<ConversationCheckpoint> chain{ck(50), ck(100, true), ck(200), ck(300)};
        const auto out = run(chain, MiB + MiB / 2);
        check(out.checkpoints == 2, "two checkpoints freed");
        check(has(chain, 100) && has(chain, 300) && chain.size() == 2, "root kept over the periodic one");
    }
    {   // nothing else left: root goes, the deepest never does
        std::vector<ConversationCheckpoint> chain{ck(100, true), ck(200), ck(300)};
        const auto out = run(chain, MiB + MiB / 2);
        check(out.checkpoints == 2, "root freed last");
        check(chain.size() == 1 && has(chain, 300), "only the deepest left");
    }
    {   // with a pin the file keeps the pinned prefix (session_deepest_checkpoint): the rest goes, root last
        std::vector<ConversationCheckpoint> chain{ck(100, true), ck(150, false, true), ck(200), ck(300)};
        const auto out = run(chain, 2 * MiB);
        check(out.checkpoints == 2 && chain.size() == 2 && has(chain, 100) && has(chain, 150), "pin and root kept");
        std::vector<ConversationCheckpoint> all{ck(100, true), ck(150, false, true), ck(200), ck(300)};
        const auto out2 = run(all, 8 * MiB);
        check(out2.checkpoints == 3 && all.size() == 1 && has(all, 150), "the pin never goes");
    }
    {   // enough RAM: nothing goes
        std::vector<ConversationCheckpoint> chain{ck(100, true), ck(200), ck(300)};
        const auto out = run(chain, 0);
        check(out.checkpoints == 0 && chain.size() == 3, "no reclaim when the SAVE fits");
    }
    std::printf("session_save_reclaim_test: %d checks passed\n", checks);
}
