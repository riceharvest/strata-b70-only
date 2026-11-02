#!/usr/bin/env python3
"""tools/usage_profile.py - an expert-cache profile from the serve engine's device routing counts.

    python tools/usage_profile.py OUT.bin USAGE.bin[:W] [USAGE2.bin[:W] ...] [--base PROFILE.bin:W]
    python tools/usage_profile.py --stats PROFILE.bin USAGE.bin [...] [--pack PACKDIR]

A USAGE file is what the SYCL serve engine writes with STRATA_USAGE_DUMP=PATH (after every request: "STRU", n_layers,
n_expert, then n_layers x n_expert int32 counts of the routed entries of its decode windows, counted by the device-built
plan; a path inside the container, e.g. relative to the repo).  Each table becomes a distribution times its weight
(make_profile_mix.py's rule); --base adds an existing profile as a prior (rank r of N counts (N - r) / sum, times W), which
ranks the pairs the tables never routed and steadies the rest.  Pairs are ranked by the sum; unscored pairs follow in
make_profile.py's fill order.

--stats: the share of each table's routing a profile's cache holds, filled as the engine fills it (a budget of U
largest-blob slots, the pairs in profile order, each blob that still fits taken), for a few budgets U.  The blob sizes are
the pack's (--pack, default the Flash-Next IQ2_XS pack).

data/expert-profile-pi.bin was made this way: tools/pi_replay.py replayed 124 requests of six pi sessions (the 256K agent
task set, ~170-245K context) against the pibase server recipe with STRATA_USAGE_DUMP, then
    python tools/usage_profile.py data/expert-profile-pi.bin usage-replay6.bin --base data/expert-profile-sycl.bin:0.2
"""
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_profile as mp  # noqa: E402

DEFAULT_PACK = "/home/dario/models/qwen38-flash-next-gsq-rco/iq2_xs"


def read_usage(path):
    b = open(path, "rb").read()
    magic, L, E = struct.unpack_from("<3i", b, 0)
    if magic != 0x55525453:
        sys.exit(f"{path}: not a STRATA_USAGE_DUMP table")
    return L, E, struct.unpack_from("<%di" % (L * E), b, 12)


def spec(s):
    head, sep, tail = s.rpartition(":")
    if sep:
        try:
            return head, float(tail)
        except ValueError:
            pass
    return s, 1.0


def blob_sizes(pack):
    sizes = {}
    for line in open(Path(pack) / "native_experts.txt"):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        sizes[int(f[0])] = int(f[4])
    return sizes


def packed(ranked, units, sizes):
    budget, used, held = units * max(sizes.values()), 0, []
    for l, e in ranked:
        if used + sizes[l] <= budget:
            used += sizes[l]
            held.append((l, e))
    return held


def main():
    args = sys.argv[1:]
    pack = DEFAULT_PACK
    if "--pack" in args:
        i = args.index("--pack")
        pack = args[i + 1]
        del args[i:i + 2]
    if args and args[0] == "--stats":
        sizes = blob_sizes(pack)
        ranked = mp.read_profile(args[1])
        for u in args[2:]:
            L, E, c = read_usage(u)
            tot = float(sum(c))
            line = []
            for U in (12760, 13784, 14124, 14636, 15148):
                h = packed(ranked, U, sizes)
                line.append(f"{U}u({len(h)}): {100.0 * sum(c[l * E + e] for l, e in h) / tot:.2f}%")
            print(f"{Path(args[1]).name} on {Path(u).name} ({int(tot)} entries): " + "  ".join(line))
        return
    out, score, base, i = args[0], {}, None, 1
    while i < len(args):
        if args[i] == "--base":
            base = spec(args[i + 1])
            i += 2
            continue
        path, w = spec(args[i])
        i += 1
        L, E, c = read_usage(path)
        tot = float(sum(c))
        for l in range(L):
            for e in range(E):
                if c[l * E + e]:
                    score[(l, e)] = score.get((l, e), 0.0) + w * c[l * E + e] / tot
    if base is not None:
        ranked = mp.read_profile(base[0])
        n = len(ranked)
        s = n * (n + 1) / 2.0
        for r, p in enumerate(ranked):
            score[p] = score.get(p, 0.0) + base[1] * (n - r) / s
    ranked = [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    for e in range(mp.N_EXPERT):
        for l in range(mp.N_LAYER):
            if (l, e) not in seen:
                ranked.append((l, e))
    mp.write_profile(out, ranked)
    assert mp.read_profile(out) == ranked
    print(f"wrote {out}: {len(ranked)} pairs, {len(score)} scored")


if __name__ == "__main__":
    main()
