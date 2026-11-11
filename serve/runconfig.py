"""serve/runconfig.py - #564: the web page's Settings view of the run config (strata-<model>.json).

A short list of documented keys can be read and changed from Strata's own page (GET / POST /config).  Everything
else in the file - the keys setup writes, the network and security keys (host, api_key, cors_origins,
trusted_origins, allowed_hosts), the MCP servers and before_load (which run programs), sampling keys not listed
here - is kept as it is: a change touches only the keys it names (as setup run again keeps the user's keys, #629).
The earlier file is kept as strata-<model>.json.bak.  The server reads the config when it starts, so a change is
used from the next start on.
"""
from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

from strata_control import user_args
from strata_control import atomic as _atomic_mod

# (key, kind, help).  kind: "bool", "int>=0", "num>=0", ("enum", values), "names", ("int", lo, hi) and ("num", lo, hi,
# lo_open) for a whole number or a number in a range (hi None: no top), or ("sampling", check) for a key of the "sampling"
# block.  An engine option is kept in "args" (and in "user_args"): ("arg", flag[, lo[, hi]]) a whole number, ("argnum",
# flag, lo, hi[, lo_open]) any number in a range, ("argenum", flag, values[, aliases]) one of some words, ("argauto", flag,
# words, lo) one of `words` or a whole number of at least lo, ("path", flag) an absolute path, ("flag", flag) one without a
# value (on or off).  ("env", NAME, lo, hi) is an engine environment variable kept in the "env" block as a whole number
# written as text.  Empty (None) removes the entry: the engine's or the server's own default then applies.
# The page shows the keys in this order, under the heading GROUPS gives each.
EDITABLE = [
    ("sampling.temperature", ("sampling", "num>=0"),
     "Default temperature for requests that send none (0 = greedy; without it the model card's: 1.0 thinking, 0.7 not)"),
    ("sampling.top_p", ("sampling", "0<x<=1"), "Default top_p for requests that send none"),
    ("sampling.top_k", ("sampling", "1..64"), "Default top_k for requests that send none (1-64)"),
    ("sampling.min_p", ("sampling", "0<=x<=1"), "Default min_p for requests that send none"),
    ("reasoning_budget_tokens", "int>=0", "Cap the thinking of every request at this many tokens (0 or empty: no cap)"),
    ("fit_max_tokens", "bool", "Shorten a max_tokens that does not fit the context (on by default); off: answer 400 like llama.cpp"),
    ("anthropic_thinking", ("enum", ["model", "on_request"]),
     "Anthropic requests that do not ask for thinking: think as the model does (model) or not (on_request)"),
    ("effort_position", ("enum", ["start", "end"]),
     "Where a non-default reasoning effort goes: start (the default) or end (keeps the cache when it changes)"),
    ("aliases", "names", "Other model names the server lists and answers to (comma-separated)"),
    ("idle_unload_s", "num>=0", "Unload the model after this many seconds without requests (0 or empty: never)"),
    ("mtp", "bool", "Draft layer (MTP): off starts the engine without it (about 1 GB less graphics memory, slower words per second); "
                    "empty or on: as installed"),
    ("lazy_load", "bool", "Start without loading the model; the first request loads it (text only)"),
    ("engine_silence_s", "num>=0", "End a request when the engine says nothing for this long (default 300 s, 0 = wait)"),
    ("api_monitor", "bool", "Keep the last requests' prompts and answers in memory for /api-monitor (how many: api_monitor_requests, "
                    "100 by default)"),
    ("open_browser", "bool", "Open the chat page in the browser when the model is ready"),
    ("vram_reserve_mib", ("arg", "--vram-reserve-mib"),
     "VRAM in MiB the engine leaves free for other programs (engine default 700)"),
    ("max_context", ("arg", "--max-context", 1024, 2147483639),
     "Context length in tokens: the most one conversation can hold (--max-context, engine default 4096). RAM and VRAM grow "
     "with it, and a length the card cannot hold stops the engine at start. A change drops the conversations saved on the "
     "SSD tier (their files no longer match). Empty: 4096"),
    ("kv_format", ("argenum", "--kv", ("fp16", "int8", "q4_0", "k8v4"), {"q4": "q4_0"}),
     "How the context's K/V is stored (--kv). fp16: the most VRAM, exact. int8: half of fp16's memory. k8v4: int8 keys and "
     "4-bit values, 816 bytes a cell against int8's 1,056. q4_0: 4-bit, about half of int8's memory, slightly lower precision. "
     "Less memory leaves more VRAM to the expert cache. A change drops the conversations saved on the SSD tier. Empty: fp16"),
    ("kv_resident", ("arg", "--kv-resident", 0, 2147483647),
     "K/V streaming (--kv-resident N): keep N cells of each attention layer's K/V in VRAM and the whole K/V in pinned RAM; the "
     "VRAM it frees goes to expert slots. On the B70 at 32768 about 4 MB of K/V a window is read from RAM and 95-96% of the "
     "block reads hit VRAM; 16K cells gave back only about 140 slots. It costs pinned RAM for the whole context. 1 to 20479 "
     "count as 20480, and a context of N cells or fewer is not streamed. A change drops the conversations saved on the SSD "
     "tier. Empty or 0: all of it in VRAM"),
    ("expert_cache", ("argauto", "--expert-cache", ("auto",), 0),
     "Expert slots kept in VRAM (--expert-cache): auto sizes them from the VRAM left after the model, the context and the "
     "reserve, or give a whole number of slots. More slots mean fewer experts read from RAM or the SSD and faster answers, and "
     "less VRAM for everything else; too many may stop the engine at start for lack of VRAM. With an expert profile 0 also "
     "means auto. Empty: the engine's 0 (auto, when it has a profile)"),
    ("expert_copies_max_mib", ("env", "STRATA_MIRROR_MIB", 0, 4194304),
     "The most RAM in MiB the copies of experts may take (STRATA_MIRROR_MIB): the experts that do not fit VRAM are copied once "
     "into pinned RAM and the GPU reads them over PCIe. Too small a cap drops the copies of the prompt loan first (long prompts "
     "read slower), then the VRAM ring of copied experts; and an expert that is in neither VRAM nor a copy makes the engine "
     "refuse to start in the B70 setup (STRATA_VERIFY_NO_HOST=1), so 0 (no copies) refuses too. Too large leaves the desktop "
     "too little RAM (docs/B70.md: keep at least 7 GB free). Empty: free RAM minus the margin below"),
    ("expert_copies_margin_mib", ("env", "STRATA_MIRROR_MARGIN_MIB", 0, 4194304),
     "RAM in MiB the automatic cap on the copies of experts leaves free (STRATA_MIRROR_MARGIN_MIB): the copies may take the "
     "free RAM minus this. Higher keeps more RAM for other programs and can drop copies (see the cap above for what that "
     "costs); lower lets the copies take more. Not used when the cap above is set. Empty: 4096"),
    ("expert_copies_loan_slots", ("env", "STRATA_LEND_MIRROR", 0, 2147483647),
     "How many of the prompt loan's cache slots keep a copy in RAM (STRATA_LEND_MIRROR; the loan is the expert-cache slots "
     "a prompt chunk borrows, about 5,000 at --prefill 16384). Empty: the whole loan. Fewer saves RAM (about 1.27 MiB a slot: "
     "5 GiB between the whole loan and 1536 on a B70) and costs about 4 s on a 180K-token cold read and about 0.5 s on a short "
     "read right after a conversation switch. 0: none. It does not stop the engine from starting; with STRATA_BATCH_LEAN the "
     "slots without a copy wait for a prompt read instead of decoding between its chunks"),
    ("conversation_cache_slots", ("arg", "--conversation-cache-slots", 0, 2147483647),
     "Parked conversations: how many conversations' states the engine keeps in RAM, so agents can take turns without reading "
     "their prompts again (empty: up to 4). The memory below also limits it"),
    ("conversation_cache_mib", ("arg", "--conversation-cache-mib"),
     "RAM in MiB for the parked conversations (empty: sized from the free RAM by the two keys below). One 70-140K-token "
     "conversation takes about 2.6-3 GB. The copy of the conversation in use counts too, so two agents taking turns need "
     "room for two: a budget that is too small skips parks, and every switch then reads the whole prompt again"),
    ("conversation_cache_auto_pct", ("arg", "--conversation-cache-auto-pct", 0, 100),
     "When the RAM for parked conversations is empty above: the percentage of the RAM free above the floor that it takes "
     "(--conversation-cache-auto-pct). Higher parks more conversations in RAM and leaves other programs less. Empty: 40"),
    ("conversation_cache_auto_max_mib", ("arg", "--conversation-cache-auto-max-mib", 0, 4194304),
     "... and the most that automatic size gives, in MiB (--conversation-cache-auto-max-mib). Empty: 8192"),
    ("conversation_cache_min_free_mib", ("arg", "--conversation-cache-min-free-mib", 0, 4194304),
     "RAM in MiB that parking, saving or restoring a conversation must leave free (--conversation-cache-min-free-mib). Higher "
     "keeps more RAM for the desktop and skips more parks when RAM is short (a conversation that is not parked is read "
     "again when its agent returns); lower parks more and leaves the desktop less. Empty: 2560"),
    ("conversation_cache_min_tokens", ("arg", "--conversation-cache-min-tokens", 0, 2147483647),
     "Park only conversations of at least this many tokens (--conversation-cache-min-tokens), so short side requests do not "
     "take a slot a long conversation needs. A short conversation that is not parked is read again from its start. Empty or "
     "0: park any"),
    ("conversation_cache_park_tail", ("arg", "--conversation-cache-park-tail", 0, 2147483647),
     "Do not park a rewind of the live conversation that leaves at most this many tokens (and 1/16 of it) behind "
     "(--conversation-cache-park-tail): that is the client's rewritten last reply, which nothing returns to, and parking "
     "copies the whole conversation. Higher skips more parks and saves RAM or drive writes; if the client does come back to "
     "such a rewind, its tokens are read again. 0: park every rewind. Empty: 4096"),
    ("conversation_cache_park_gain", ("arg", "--conversation-cache-park-gain", 0, 2147483647),
     "Park only after the conversation has advanced this many tokens past the copy it replaces "
     "(--conversation-cache-park-gain), so a turn that grew by a few hundred tokens does not write a whole snapshot (about "
     "3 GB at 150K tokens) that the next turn drops. Higher: fewer writes and less drive wear; the older copy stays and the "
     "tokens after it are read again on a switch back. Empty or 0: park every turn"),
    ("parallel", "int>=0",
     "Requests decoding together (batch slots): 1 or empty = one at a time. Each slot takes about 1 GiB of VRAM that the expert "
     "cache would hold, and on the B70 two streams together measured about the same speed as one alone (it saves waiting, not time)"),
    ("prefill_chunk", ("argauto", "--prefill", ("auto", "auto:16384", "auto:32768"), 0),
     "How many prompt tokens are read per step (--prefill). A bigger chunk reads a long prompt faster (fewer experts streamed "
     "per token) and borrows more of the expert cache's VRAM while the prompt is read (about 4,900 slots at 16384, streamed "
     "back afterwards). auto: the largest chunk of up to 8192 the cache can lend; auto:16384 and auto:32768 allow bigger ones. "
     "A whole number is used as given; 0 reads a prompt one token at a time. Empty: the engine's 0 (setup writes one)"),
    ("spec", ("arg", "--spec", 0, 8),
     "Draft window (--spec): the most tokens one check holds, the draft layer's guesses included. A longer window gains speed "
     "where the guesses are right and wastes work where they are not; with STRATA_SPEC_COST (the B70 default) the engine picks "
     "the size per check, so this is only the upper limit (setup writes 4; prompt lookup may add 2, at most 8 in all). On one "
     "B70 a limit of 6 measured 5 to 15% faster than 4 on code, and 7 or 8 no faster (a seventh row costs twice a sixth). "
     "Above 8 stops the engine at start. Below 2 the draft layer is not used: slower words per second, about 1 GB of VRAM "
     "free. Empty: the engine's 0"),
    ("spec_min_p", ("argnum", "--spec-min-p", 0, 1),
     "How sure the draft layer must be to extend a verify window by another guess (--spec-min-p, 0 to 1; setup writes 0.5). "
     "Higher: shorter windows and fewer wasted checks; lower: longer windows and more wasted work when the guesses miss. 0 "
     "extends every window to the draft window above. Empty: the engine's 0"),
    ("spec_probe_min", ("env", "STRATA_SPEC_PROBE_MIN", 0, 1),
     "How the engine prices a draft window size it has hardly used (STRATA_SPEC_PROBE_MIN; it picks the window size with the "
     "most tokens per millisecond from measured timings). 1: the fastest of a size's first tries is its price, and a size "
     "not timed for a while is timed again. That keeps one slow try (the first use of a size, a busy moment) from ruling a "
     "size out until the engine restarts, which costs up to 15% of the words per second; the price of it is a few slower "
     "checks per size. 0 or empty: the average of the first tries, and a size is not timed again by the draft layer. The "
     "text does not change either way"),
    ("spec_probes", ("env", "STRATA_SPEC_PROBES", 1, 64),
     "How often a draft window size with no known timing is tried before its timing may rule it out (STRATA_SPEC_PROBES). "
     "More: a surer timing, more slower checks at the start of each context range. Empty: 3"),
    ("spec_stale_rounds", ("env", "STRATA_SPEC_STALE", 0, 2147483647),
     "After how many checks without a timing a draft window size is timed again (STRATA_SPEC_STALE). Lower follows a "
     "changing machine faster and spends more checks on sizes that do not pay; 0 never times a size again. Empty: 300"),
    ("iq_xxs_sign_arith", ("env", "STRATA_IQ_XXS_SGNMUL", 0, 1),
     "How the expert kernels work out the signs of IQ2_XXS weights (STRATA_IQ_XXS_SGNMUL). 1: by arithmetic, as IQ2_S "
     "already does, which makes that kernel about 13% faster on a B70 (1 to 3% more words per second with a pack whose "
     "experts are mostly IQ2_XXS, such as UD-IQ1_S). 0 or empty: from a table, as before. The numbers are the same bit for "
     "bit either way; it needs STRATA_IQ_GRID on (the default) and has no effect with STRATA_XE2_IQ"),
    ("f16_products_any_format", ("env", "STRATA_F16X_ANY", 0, 1),
     "Which expert formats the prompt path multiplies straight from the quantized weights (STRATA_F16X_ANY). 1: also the "
     "formats without kernels of their own (Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ3_XXS, IQ2_XS, IQ4_XS, IQ1_M, Q2_0 gate/up; "
     "IQ4_NL, Q5_0, Q5_1, Q8_0 down: with it, every format of Unsloth's files and of the GSQ-RCO packs), by one kernel "
     "that decodes any of them; they otherwise take a dequantize and two library products per "
     "expert. 0 or empty: only IQ2_S / IQ2_XXS with a Q2_0 down, as before. It changes nothing for a pack in those three "
     "formats (UD-IQ1_S). Not measured on a real pack yet: esimd_f16_parity <rows> <experts> <reps> <seed> all times every "
     "format on random blocks, without a model file. The products add in another order than the library's, so the "
     "numbers differ in the last bits"),
    ("mtp_window", ("arg", "--mtp-window", 0, 2147483647),
     "How far back the draft layer looks (--mtp-window): the last N cells of the context, 0 = every cell. A smaller window is "
     "cheaper to draft on a long context (setup picks 8192 above 64K) and a very small one (2048) costs accepted drafts. A "
     "change drops the conversations saved on the SSD tier. Empty: 32768"),
    ("conversation_cache_spill_dir", ("path", "--conversation-cache-spill-dir"),
     "SSD tier for parked conversations: a folder where a conversation pushed out of RAM is written, and read back when "
     "its agent returns, so its whole prompt need not be read again. It costs disk space (the two numbers below) and time: "
     "a conversation comes back slower from the SSD than from RAM. The first start with a folder measures its drive once "
     "(about 1 GB written and read back, a few seconds); the Monitor's Conversation cache card then says how long a "
     "conversation takes from this drive. Give the "
     "path as the engine sees it: the engine runs in a container where the folder that holds the repo is /work, e.g. "
     "/work/strata-spill. Empty: no SSD tier (and the two numbers below go)"),
    ("conversation_cache_spill_mib", ("arg", "--conversation-cache-spill-mib"),
     "Disk space in MiB the SSD tier may use at most (empty: 24576, about 24 GiB). Needs the folder above"),
    ("conversation_cache_spill_min_free_mib", ("arg", "--conversation-cache-spill-min-free-mib"),
     "Disk space in MiB the SSD tier leaves free on that disk (empty: 40960, about 40 GiB). Needs the folder above"),
    ("conversation_cache_park_ssd", ("flag", "--conversation-cache-park-ssd"),
     "Park on the SSD only (--conversation-cache-park-ssd): every parked conversation goes to a file in the folder above and "
     "none stays in RAM, so more agents can take turns without more memory. A switch then costs the drive's seconds (the "
     "Monitor's Conversation cache card says how many). A park writes the model's running state and only the K/V that "
     "changed since the conversation was read back. Needs the folder above. Empty: off (parked conversations stay in RAM, "
     "the SSD takes what RAM pushes out)"),
    ("conversation_cache_park_ssd_whole", ("flag", "--conversation-cache-park-ssd-whole"),
     "With Park on the SSD only: write the whole conversation at every park (about 3 GB at 150K tokens) instead of only "
     "what changed. Simpler files, far more drive wear. Empty: off"),
    ("conversation_cache_park_on_exit", ("flag", "--conversation-cache-park-on-exit"),
     "With Park on the SSD only: park the conversation in use too when the engine stops cleanly (a restart from this page, "
     "an unload), so the next start reads it back from its file in about a second instead of reading its whole prompt "
     "again. The stop takes that write longer. Empty: off"),
    ("conversation_cache_park_ssd_checkpoints", ("arg", "--conversation-cache-park-ssd-checkpoints", 0, 64),
     "With Park on the SSD only: how many checkpoints a parked conversation keeps. Empty or 0: all of them (5-6, about "
     "113 MB each, all read back at every switch: 0.8 s on an NVMe). N: the shallowest one (the start that new "
     "conversations share) and the N deepest (where the next turns resume); a request that rewinds to between them reads "
     "more of its prompt again"),
    ("conversation_cache_park_ssd_stream", ("flag", "--conversation-cache-park-ssd-stream"),
     "With Park on the SSD only: when a conversation is read back, copy its K/V from the file straight into the engine, "
     "block by block, instead of reading the whole conversation into RAM first (about 15 KB a token: 2 GB at 150K tokens, "
     "for the seconds of a switch). On: a switch needs almost no free RAM. Empty: off"),
    ("conversation_cache_spill_index_mib", ("arg", "--conversation-cache-spill-index-mib", 0, 1048576),
     "The SSD tier's index file grows with every park (about 1 MB for a 150K-token conversation). Above this many MiB it is "
     "written again from the entries that are still there (--conversation-cache-spill-index-mib). 0: never. Empty: 64"),
    ("conversation_cache_park_ssd_keep_pct", ("arg", "--conversation-cache-park-ssd-keep-pct", 0, 100),
     "With Park on the SSD only: a conversation's file is updated in place when at least this percentage of its tokens "
     "still holds; below it a new file is written and the old one stays (empty: 50)"),
    ("conversation_cache_park_ssd_deepest_checkpoint", ("flag", "--conversation-cache-park-ssd-deepest-checkpoint"),
     "With Park on the SSD only: keep only the deepest checkpoint of a parked conversation. By default every checkpoint has "
     "its own file, written once (about 113 MB each), so a request that rewinds further back, or a new conversation that "
     "shares the system prompt, does not read its whole prompt again after a read-back. On: less disk and a faster "
     "read-back, and those requests read their prompt again. Empty: off"),
    ("conversation_cache_spill_probe_mib", ("arg", "--conversation-cache-spill-probe-mib"),
     "The size in MiB of the file the drive is measured with at the first start with a folder (empty: 1024; 0: no "
     "measurement, the engine then assumes 1700 MB/s)"),
    ("conversation_cache_spill_probe_s", ("arg", "--conversation-cache-spill-probe-s", 1),
     "The most seconds each direction of that measurement may take (empty: 3)"),
    ("conversation_cache_spill_min_tokens", ("arg", "--conversation-cache-spill-min-tokens", 0, 2147483647),
     "Conversations shorter than this many tokens are not written to the SSD (--conversation-cache-spill-min-tokens): a short "
     "one is not worth a write, and is read again from its start when its agent returns. Needs the folder above. Empty: 4096"),
    ("conversation_cache_spill_mbps", ("arg", "--conversation-cache-spill-mbps", 1, 100000),
     "The read speed in MB/s that the read-back-or-read-again decision assumes (--conversation-cache-spill-mbps). A "
     "conversation comes back from the SSD only when the prompt tokens it saves cost more time (at the number below per "
     "token) than reading its file plus 0.5 s. Higher: more read-backs; lower: more prompts read again. At least 1. Needs the "
     "folder above. Empty: the speed measured on this drive (1700 if it was not measured)"),
    ("conversation_cache_spill_prefill_us", ("arg", "--conversation-cache-spill-prefill-us", 1, 1000000),
     "What reading one prompt token costs, in microseconds, in the same decision (--conversation-cache-spill-prefill-us). "
     "Higher: more conversations come back from the SSD; lower: the prompt is read again sooner. At least 1. Needs the folder "
     "above. Empty: 300"),
    ("conversation_cache_spill_verify", ("flag", "--conversation-cache-spill-verify"),
     "Check the payload hash when a file is read back from the SSD (--conversation-cache-spill-verify). It catches a damaged "
     "file and costs seconds at every read-back (a 4 GB file is read whole to be checked). Needs the folder above. Empty: off"),
    ("batch_mtp", ("flag", "--batch-mtp"),
     "One draft proposal per parallel request (--batch-mtp): only with parallel 2 or more; off, or removed when parallel is "
     "off. Empty: engine default (off)"),
    # the server's own numbers (top-level keys, read when the server starts; serve/server.py SERVER_NUMBERS)
    ("answer_room", "bool",
     "Leave the answer room (on by default): a request with no thinking budget of its own and a max_tokens of the number "
     "below or more has its thinking closed early enough to leave the answer its share. Off: the model may spend all of "
     "max_tokens thinking and be cut off with no answer"),
    ("answer_room_min_cap", ("int", 1, None),
     "From this max_tokens on, thinking is closed early to leave room to answer (answer_room; default 1024). Lower: shorter "
     "replies are protected too, and their thinking is cut. Higher: short caps are left as they are and may end in the middle "
     "of the thinking. (DeepSeek-V4.1 only: caps from 48 up to this are limited to half.) Empty: 1024"),
    ("answer_reserve_min", ("int", 0, None),
     "The least number of tokens kept for the answer when a thinking budget is cut down to fit max_tokens (default 512; the "
     "reserve is the larger of this and a quarter of max_tokens, at most half). Higher: a safer answer and less thinking. "
     "Lower: more thinking and a risk that the answer is cut. Empty: 512"),
    ("repeat_stop_tokens", "int>=0",
     "End a reply that repeats one token this many times in a row (default 256; 0 = never): a model in a loop, or a broken "
     "state that answers one token forever. The reply ends as \"length\". Lower: loops end sooner and a long honest run of "
     "one token (a line of dashes) may be cut. Empty: 256"),
    ("tool_name_repair", ("enum", ["prefix"]),
     "What to do when the model calls a tool the request did not declare. Empty: the call goes back as text (and "
     "strata_rejected_tool_calls names it). prefix: read it as the one declared tool whose name it extends, or that "
     "extends it, at an underscore (process_manage for process). The agent then gets a real call, possibly with "
     "arguments that tool does not take; with two candidates nothing is guessed"),
    ("reasoning_loop_recovery", ("enum", ["stop", "recover"]),
     "Reasoning that repeats whole passages: every few hundred tokens the server looks at the thinking (the loop_* numbers "
     "below) and on a loop ends the reply there (stop) or goes on from the same output with the low-effort instruction in "
     "place of the xhigh one (recover; not for a request with pictures). It costs a scan of the latest words of the thinking at "
     "each look, and a wrong call cuts a long honest check short. Empty: off, the detector does not run and the loop_* numbers "
     "do nothing"),
    ("loop_check_every", ("int", 1, None),
     "With reasoning_loop_recovery on: output tokens between two looks at the thinking (default 512). Smaller: a loop is "
     "caught sooner and the scan runs more often. Larger: a reply may loop that many tokens longer. Empty: 512"),
    ("loop_coverage", ("num", 0, 1, True),
     "With reasoning_loop_recovery on: the share of the last 2,000 words of the thinking that must lie inside 12-word passages "
     "seen three times before it counts as a loop (more than 0, at most 1; default 0.25). Lower: loops are caught sooner and "
     "more honest long checks are cut. Higher: more repeating is put up with (1 acts only when the whole window repeats). "
     "Empty: 0.25"),
    ("loop_history_words", ("int", 2000, 1000000),
     "With reasoning_loop_recovery on: how many of the latest words of the thinking are searched for repeated passages "
     "(default 30000). Larger: sees repeats that lie far apart (a check repeated after a long code draft) and every look scans "
     "longer. Below 2000 the detector could never fire, so 2000 is the least. Empty: 30000"),
    ("loop_tail_chars", ("int", 256, 65536),
     "With reasoning_loop_recovery on: how many characters at the end of the thinking are also checked for a short repeating "
     "period (default 2048). That catches one long word that cycles (a 24,000-digit string), which the word count cannot see. "
     "Larger: it needs a longer run of repeating before it acts. Empty: 2048"),
    ("loop_tail_period", ("int", 1, 1024),
     "With reasoning_loop_recovery on: the longest repeating period in characters that counts in that tail (default 64; at "
     "most half of loop_tail_chars). Larger: longer cycles are caught and the check compares more. Empty: 64"),
    ("yields_max", ("int", 0, None),
     "Only with parallel 2 or more: how often one request's prompt read gives way to a shorter waiting one (default 2). "
     "Higher: short requests wait less behind a long read and the long one finishes later (it is read in more pieces). 0: a "
     "read is never interrupted. Empty: 2"),
    ("admit_short_first", ("int", 0, None),
     "Only with parallel 2 or more: a new request with a prompt of at most this many tokens takes the engine before longer "
     "ones, so a chat question does not wait behind every agent read. A longer prompt that has waited 30 s (the server's "
     "STRATA_ADMIT_MAX_WAIT_S) counts as short, so it is not held back longer. 0 or empty: first come, first served (the server's environment variable "
     "STRATA_ADMIT_SHORT_FIRST does the same; this key wins)"),
    ("unload_wait_s", ("num", 1, 3600),
     "How long POST /unload?wait=1 waits for the running request before it answers busy (default 300 s; wait=<seconds> in "
     "the request chooses its own). Longer: another program that needs the GPU waits longer for it. Shorter: it is told busy "
     "sooner. Empty: 300"),
    ("restart_retry_s", ("num", 1, 3600),
     "Seconds between the tries to start the engine again after it died (default 15): a dying engine's VRAM can take a while "
     "to come back. Shorter: the engine is back sooner and a try may fail while the VRAM is still held. Empty: 15"),
    ("stream_linger_s", ("num", 0, 60),
     "With several agents taking turns: when the agent that was just served has a tool running, hold the other agents' "
     "requests back up to this many seconds for its next request. If it comes back in time the engine goes on with the "
     "conversation it holds (no switch, about 2 s saved each way); if not, the others waited that long for nothing. The "
     "Monitor's Agent threads data says how fast agents come back (return_gap). Empty or 0: off, the engine never waits"),
    ("queue_beat_s", ("num", 1, 600),
     "A request waiting for its turn sends a keep-alive this often, in seconds (default 10), so a client or proxy does not "
     "give up on a silent connection. Shorter: safer behind a proxy with a short idle limit, more traffic. Empty: 10"),
    ("keepalive_s", ("num", 1, 600),
     "A stream that has gone quiet (the server holds back a tool call's arguments until they are whole) sends a ping at least "
     "this often, in seconds (default 10). The same trade as the number above. Empty: 10"),
    ("monitor_requests", ("int", 1, 500),
     "How many of the last finished requests the Monitor lists (default 12). The server keeps up to 500 in memory for the full "
     "list. More: a longer table and a bigger answer every time the page refreshes. Empty: 12"),
    ("api_monitor_requests", ("int", 1, 1000),
     "With the API monitor on: how many requests' prompts and answers it keeps in memory (default 100). One can take about 1 MB "
     "(prompt, answer, thinking and response are each cut at 256K characters), so more costs RAM. Empty: 100"),
]
SPILL_DIR = "conversation_cache_spill_dir"
SPILL_DEPENDENTS = ("conversation_cache_spill_mib", "conversation_cache_spill_min_free_mib", "conversation_cache_park_ssd",
                    "conversation_cache_park_on_exit", "conversation_cache_park_ssd_stream", "conversation_cache_spill_index_mib",
                    "conversation_cache_park_ssd_checkpoints",
                    "conversation_cache_park_ssd_whole", "conversation_cache_park_ssd_keep_pct",
                    "conversation_cache_park_ssd_deepest_checkpoint", "conversation_cache_spill_probe_mib",
                    "conversation_cache_spill_probe_s", "conversation_cache_spill_min_tokens",
                    "conversation_cache_spill_mbps", "conversation_cache_spill_prefill_us", "conversation_cache_spill_verify")
PARALLEL_MAX = 8      # serve/server.py PARALLEL_MAX: the engine's batch window holds at most 8 rows
LOOP_TAIL = ("loop_tail_chars", "loop_tail_period")
LOOP_TAIL_DEFAULTS = (2048, 64)   # serve/server.py LOOP_TAIL_CHARS and LOOP_TAIL_PERIOD (test_runconfig checks they match)
SPEC = {k: kind for k, kind, _ in EDITABLE}

# The headings of the Settings page, in the order shown, and the keys under each (the keys keep their order in EDITABLE).
GROUPS = {
    "Model and context": ("max_context", "kv_format", "kv_resident", "vram_reserve_mib", "expert_cache",
                          "expert_copies_max_mib", "expert_copies_margin_mib", "expert_copies_loan_slots", "lazy_load"),
    "Speed": ("mtp", "spec", "spec_min_p", "spec_probe_min", "spec_probes", "spec_stale_rounds", "iq_xxs_sign_arith", "f16_products_any_format", "mtp_window", "prefill_chunk", "parallel", "batch_mtp", "yields_max",
              "admit_short_first"),
    "Parked conversations": ("conversation_cache_slots", "conversation_cache_mib", "conversation_cache_auto_pct",
                             "conversation_cache_auto_max_mib", "conversation_cache_min_free_mib",
                             "conversation_cache_min_tokens", "conversation_cache_park_tail",
                             "conversation_cache_park_gain"),
    "SSD tier": ("conversation_cache_spill_dir", "conversation_cache_spill_mib", "conversation_cache_spill_min_free_mib",
                 "conversation_cache_park_ssd", "conversation_cache_park_ssd_whole", "conversation_cache_park_on_exit",
                 "conversation_cache_park_ssd_stream", "conversation_cache_park_ssd_checkpoints",
                 "conversation_cache_park_ssd_keep_pct", "conversation_cache_spill_index_mib",
                 "conversation_cache_park_ssd_deepest_checkpoint", "conversation_cache_spill_probe_mib",
                 "conversation_cache_spill_probe_s", "conversation_cache_spill_min_tokens", "conversation_cache_spill_mbps",
                 "conversation_cache_spill_prefill_us", "conversation_cache_spill_verify"),
    "Thinking and answers": ("tool_name_repair", "sampling.temperature", "sampling.top_p", "sampling.top_k", "sampling.min_p",
                             "reasoning_budget_tokens", "fit_max_tokens", "anthropic_thinking", "effort_position",
                             "answer_room", "answer_room_min_cap", "answer_reserve_min", "repeat_stop_tokens",
                             "reasoning_loop_recovery", "loop_check_every", "loop_coverage", "loop_history_words",
                             "loop_tail_chars", "loop_tail_period"),
    "Server": ("stream_linger_s", "aliases", "idle_unload_s", "engine_silence_s", "open_browser", "unload_wait_s", "restart_retry_s",
               "queue_beat_s", "keepalive_s"),
    "Monitor": ("api_monitor", "api_monitor_requests", "monitor_requests"),
}
GROUP_OF = {key: name for name, keys in GROUPS.items() for key in keys}


def _args(cfg: dict) -> list:
    """The engine's arguments as a start gives them: "args" with the person's "user_args" over them (serve/server.py
    engine_args), so a value kept only in "user_args" is seen too."""
    a = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    return user_args.merge(a, cfg["user_args"]) if cfg.get("user_args") else a


def _arg(cfg: dict, flag: str):
    a = _args(cfg)
    if flag in a[:-1]:
        v = str(a[a.index(flag) + 1])
        return int(v) if v.isdigit() else v
    return None


def _num_text(v) -> str:
    """A number as the engine reads it: 1 for 1.0, 0.5 for 0.5."""
    return str(int(v)) if float(v) == int(v) else repr(float(v))


def value_of(cfg: dict, key: str):
    """What the config says for `key` now, as the page shows it (None: not set).  A value a hand edit left in another
    spelling comes back as the one `check` accepts (q4 for q4_0, true for recover, 1536 for "1536")."""
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "sampling":
        s = cfg.get("sampling")
        return s.get(key.split(".", 1)[1]) if isinstance(s, dict) else None
    if k in ("arg", "path", "argauto"):
        return _arg(cfg, kind[1])
    if k == "argnum":
        v = _arg(cfg, kind[1])
        try:
            return None if v is None else float(v)
        except ValueError:
            return v
    if k == "argenum":
        v = _arg(cfg, kind[1])
        return (kind[3] if len(kind) > 3 else {}).get(v, v)
    if k == "flag":
        return True if kind[1] in _args(cfg) else None
    if k == "env":
        e = cfg.get("env")
        v = e.get(kind[1]) if isinstance(e, dict) else None
        if v is None:
            return None
        v = str(v).strip()
        return int(v) if v.isdigit() else v
    v = cfg.get(key)
    if key == "reasoning_loop_recovery":               # the config also takes true (= recover) and false (= off)
        return "recover" if v is True else None if v is False else v
    return v


def _limits(kind) -> dict:
    """The number kinds' range and step for the page's input (the server checks the value again)."""
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "arg":
        return {"step": 1, "min": kind[2] if len(kind) > 2 else 0, **({"max": kind[3]} if len(kind) > 3 and kind[3] is not None else {})}
    if k == "int":
        return {"step": 1, "min": kind[1], **({"max": kind[2]} if kind[2] is not None else {})}
    if k == "env":
        return {"step": 1, "min": kind[2], "max": kind[3]}
    if k == "int>=0":
        return {"step": 1, "min": 0}
    if k == "argnum":
        return {"step": "any", "min": kind[2], "max": kind[3]}
    if k == "num":
        return {"step": "any", "min": kind[1], "max": kind[2]}
    return {}


def view(cfg: dict, path: str | Path) -> dict:
    """GET /config: the editable keys with their values (None: not set, the default applies), each under its group
    (`groups`: the headings in the order to show them)."""
    out = []
    for key, kind, help_ in EDITABLE:
        k = kind[0] if isinstance(kind, tuple) else kind
        choices = kind[1] if k == "enum" else kind[2] if k == "argenum" else None
        out.append({"key": key, "value": value_of(cfg, key), "help": help_, "group": GROUP_OF.get(key, "Other"),
                    "kind": "number" if k in ("sampling", "arg", "argnum", "int", "num", "env", "int>=0", "num>=0") else
                    "bool" if k == "flag" else "text" if k == "path" else "auto" if k == "argauto" else
                    "enum" if k == "argenum" else k,
                    **({"choices": list(choices)} if choices else {}), **_limits(kind)})
    return {"file": Path(path).name, "keys": out, "groups": list(GROUPS) + (["Other"] if any(g["group"] == "Other" for g in out) else []),
            "note": "Saved to the run config; used from the next start of the model."}


def _number(key, v, whole=False, lo=0.0, hi=None, lo_open=False):
    # (json reads NaN and Infinity as numbers: neither is below or above anything, and int(inf) raises OverflowError)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (whole and v != int(v)) or \
            (v <= lo if lo_open else v < lo) or (hi is not None and v > hi):
        rng = f"{'more than' if lo_open else 'at least'} {lo:g}" + (f" and at most {hi:g}" if hi is not None else "")
        raise ValueError(f"{key}: expected a {'whole ' if whole else ''}number, {rng}, not {v!r}")
    return int(v) if whole else v


def check(key: str, v, cfg: dict):
    """The value to store for `key`, or a ValueError naming what is expected.  None removes the key (its default)."""
    if key not in SPEC:
        raise ValueError(f"{key!r} cannot be changed here (only the keys the Settings view lists)")
    if v is None:
        return None
    kind = SPEC[key]
    k = kind[0] if isinstance(kind, tuple) else kind
    if k == "bool":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        if key == "lazy_load" and v and cfg.get("vision"):
            raise ValueError("lazy_load: lazy loading is text-only, and this model reads images (\"vision\")")
        return v
    if k == "enum":
        if v not in kind[1]:
            raise ValueError(f"{key}: expected one of {', '.join(kind[1])}, not {v!r}")
        return v
    if k == "names":
        if isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() and len(x) <= 128 for x in v):
            raise ValueError(f"{key}: expected a list of model names, not {v!r}")
        return v or None
    if k == "int>=0":
        n = _number(key, v, whole=True)
        if key == "parallel":                          # 0 and 1 are "one at a time": the key goes
            if n > PARALLEL_MAX:
                raise ValueError(f"parallel: at most {PARALLEL_MAX} requests decode together, not {n}")
            return n if n >= 2 else None
        return n
    if k == "num>=0":
        return _number(key, v)
    if k == "arg":
        return _number(key, v, whole=True, lo=kind[2] if len(kind) > 2 else 0, hi=kind[3] if len(kind) > 3 else None)
    if k == "int":
        return _number(key, v, whole=True, lo=kind[1], hi=kind[2])
    if k == "num":
        return float(_number(key, v, lo=kind[1], hi=kind[2], lo_open=len(kind) > 3 and bool(kind[3])))
    if k == "argnum":
        return float(_number(key, v, lo=kind[2], hi=kind[3], lo_open=len(kind) > 4 and bool(kind[4])))
    if k == "argenum":
        aliases = kind[3] if len(kind) > 3 else {}
        if not isinstance(v, str) or (v not in kind[2] and v not in aliases):
            raise ValueError(f"{key}: expected one of {', '.join(kind[2])}, not {v!r}")
        return aliases.get(v, v)
    if k == "argauto":                                 # a word such as auto, or a whole number (a digits text is one too)
        if isinstance(v, str):
            t = v.strip().lower()
            if t in kind[2]:
                return t
            if t == "":
                return None
            if not t.isdigit():
                raise ValueError(f"{key}: expected {', '.join(kind[2])} or a whole number of at least {kind[3]}, not {v!r}")
            v = int(t)
        return _number(key, v, whole=True, lo=kind[3], hi=2147483647)
    if k == "env":                                     # a whole number, kept as text in the "env" block
        if isinstance(v, str) and v.strip().isdigit():
            v = int(v.strip())
        return _number(key, v, whole=True, lo=kind[2], hi=kind[3])
    if k == "path":
        if not isinstance(v, str) or not v.strip():
            if isinstance(v, str):                     # an empty text: the folder is not wanted
                return None
            raise ValueError(f"{key}: expected an absolute path, not {v!r}")
        v = v.strip()
        if not v.startswith("/") or len(v) > 4096 or any(ord(c) < 32 or c == "\x7f" for c in v):
            raise ValueError(f"{key}: expected an absolute path as the engine sees it (starting with /, e.g. /work/strata-spill), "
                             f"not {v!r}")
        return v
    if k == "flag":
        if not isinstance(v, bool):
            raise ValueError(f"{key}: expected true or false, not {v!r}")
        return True if v else None                     # off is the flag left out
    rule = kind[1]                                     # a sampling key: the server's own rules
    if rule == "num>=0":
        return float(_number(key, v))
    if rule == "0<x<=1":
        return float(_number(key, v, hi=1, lo_open=True))
    if rule == "0<=x<=1":
        return float(_number(key, v, hi=1))
    return _number(key, v, whole=True, lo=1, hi=64)


def apply(cfg: dict, changes: dict) -> tuple[dict, list[str]]:
    """The config with `changes` ({key: value or None}) made, and the keys that changed.  Every value is checked
    first, so a request with one bad value changes nothing (ValueError)."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError('send {"set": {"<key>": <value or null>, ...}}')
    checked = {k: check(k, v, cfg) for k, v in changes.items()}
    after = json.loads(json.dumps(cfg))                # what the file will say, for the rules between keys
    for key, v in checked.items():
        _put(after, key, v)
    if SPILL_DIR in checked and checked[SPILL_DIR] is None:
        for dep in SPILL_DEPENDENTS:                   # no folder: no SSD tier, the quota and the floor go with it
            checked[dep] = None
    for dep in SPILL_DEPENDENTS:
        if checked.get(dep) is not None and value_of(after, SPILL_DIR) is None:
            raise ValueError(f"{dep}: set conversation_cache_spill_dir first (the SSD tier is off without a folder)")
    if any(k in checked for k in LOOP_TAIL):           # a period as long as the tail reads every text as repeating
        chars, period = (value_of(after, k) or d for k, d in zip(LOOP_TAIL, LOOP_TAIL_DEFAULTS))
        if period * 2 > chars:
            raise ValueError(f"loop_tail_period: at most half of loop_tail_chars ({chars // 2}), not {period}")
    if checked.get("batch_mtp") and not (value_of(after, "parallel") or 0) >= 2:
        raise ValueError("batch_mtp: needs parallel 2 or more (one draft proposal per parallel request)")
    if "parallel" in checked and checked["parallel"] is None and "batch_mtp" not in checked and value_of(cfg, "batch_mtp"):
        checked["batch_mtp"] = None                    # parallel off: --batch-mtp has no meaning, so it does not stay in the args
    new = json.loads(json.dumps(cfg))                  # a deep copy: the caller's dict is left alone
    changed = []
    for key, v in checked.items():
        if value_of(new, key) == v:
            continue
        _put(new, key, v)
        changed.append(key)
    return new, changed


def _put(new: dict, key: str, v) -> None:
    """Writes `v` (None: remove) for `key` into `new`, in place."""
    kind = SPEC[key]
    if isinstance(kind, tuple) and kind[0] == "sampling":
        s = dict(new.get("sampling") or {}) if isinstance(new.get("sampling"), dict) else {}
        sub = key.split(".", 1)[1]
        if v is None:
            s.pop(sub, None)
        else:
            s[sub] = v
        if s:
            new["sampling"] = s
        else:
            new.pop("sampling", None)
    elif isinstance(kind, tuple) and kind[0] == "env":
        e = dict(new["env"]) if isinstance(new.get("env"), dict) else {}
        if v is None:
            e.pop(kind[1], None)
        else:
            e[kind[1]] = str(v)
        if e:
            new["env"] = e
        else:
            new.pop("env", None)
    elif isinstance(kind, tuple) and kind[0] in ("arg", "path", "flag", "argnum", "argenum", "argauto"):
        a = list(new.get("args") or [])
        flag = kind[1]
        if v is not None and kind[0] == "argnum":
            v = _num_text(v)                           # 1 not 1.0
        if kind[0] == "flag":
            if flag in a and v is None:
                a = [x for x in a if x != flag]
            elif flag not in a and v is not None:
                a.append(flag)
        elif flag in a[:-1]:
            i = a.index(flag)
            if v is None:
                del a[i:i + 2]
            else:
                a[i + 1] = str(v)
        elif v is not None:
            a += [flag, str(v)]
        new["args"] = a
        if kind[0] == "flag":                          # a bare flag: the group is the flag alone
            mine = [g for g in user_args.groups(user_args.capture(new)) if user_args.flag_of(g) != flag]
            mine = user_args.flat(mine + ([[flag]] if v is not None else []))
        else:
            mine = user_args.set_flag(user_args.capture(new), flag, v)   # a value set here is the person's: it survives setup and the app
        if mine:
            new[user_args.USER_KEY] = mine
        else:
            new.pop(user_args.USER_KEY, None)
        if user_args.WRITTEN_KEY in new:
            new[user_args.WRITTEN_KEY] = list(a)
    elif v is None:
        new.pop(key, None)
    else:
        new[key] = v


def save(path: str | Path, cfg: dict) -> Path:
    """The config written whole (a temporary file moved over the old one), the earlier one kept as <name>.bak."""
    path = Path(path)
    bak = path.with_name(path.name + ".bak")
    shutil.copyfile(path, bak)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    _atomic_mod.replace(tmp, path)
    return bak


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{Path(path).name} is not a JSON object")
    return cfg
