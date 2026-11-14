#!/usr/bin/env python3
"""Compare STRATA_LAYER_HASH files and name the first differing layer and buffer.

Usage:
  hashdiff.py <file_a> <file_b>
      Compare two runs' STRATA_LAYER_HASH outputs by (req, layer).
  hashdiff.py --half-split <file> [--pairs 36]
      Within a single 2-round run (e.g. 36 prompts sent twice = reqs 0..35 vs 36..71),
      compare round 0 vs round 1 for every pair and name the first differing layer/buffer.
  hashdiff.py --reqs <req_a> <req_b> <file>
      Compare two specific requests within one STRATA_LAYER_HASH file.
"""
import argparse
import re
import sys
from collections import defaultdict

LINE_RE = re.compile(
    r"req=(?P<req>-?\d+)\s+path=(?P<path>\S+)\s+pos0=(?P<pos0>-?\d+)\s+"
    r"T=(?P<T>\d+)\s+layer=(?P<layer>\d+)\s+hidden=(?P<hidden>[0-9a-fA-F]+)\s+stage=(?P<stage>[0-9a-fA-F]+)"
)


def load_hashes(path):
    """Return {req: {layer: {'hidden': str, 'stage': str, 'pos0': int, 'T': int, 'path': str}}}."""
    data = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = LINE_RE.search(line)
            if not m:
                raise ValueError(f"{path}:{lineno}: unrecognized line: {line!r}")
            req = int(m.group("req"))
            layer = int(m.group("layer"))
            data[req][layer] = {
                "path": m.group("path"),
                "pos0": int(m.group("pos0")),
                "T": int(m.group("T")),
                "hidden": m.group("hidden").lower(),
                "stage": m.group("stage").lower(),
            }
    return data


def diff_request_maps(map_a, map_b, pairs, label_a, label_b):
    """Compare ordered (id_a, id_b) pairs between map_a and map_b. Returns (ok, summary_lines)."""
    diffs = []
    compared_layers = 0
    compared_reqs = 0
    missing = []
    for req_a, req_b in pairs:
        la = map_a.get(req_a)
        lb = map_b.get(req_b)
        if la is None and lb is None:
            continue
        if la is None or lb is None:
            missing.append((req_a, req_b, "A" if la is None else "B"))
            continue
        compared_reqs += 1
        all_layers = sorted(set(la.keys()) | set(lb.keys()))
        first_diff = None
        for l in all_layers:
            ea = la.get(l)
            eb = lb.get(l)
            if ea is None or eb is None:
                missing.append((req_a, req_b, f"layer {l} missing in {'A' if ea is None else 'B'}"))
                continue
            compared_layers += 1
            bufs = []
            if ea["stage"] != eb["stage"]:
                bufs.append("stage")
            if ea["hidden"] != eb["hidden"]:
                bufs.append("hidden")
            if bufs:
                rec = (req_a, req_b, l, "+".join(bufs), ea, eb)
                diffs.append(rec)
                if first_diff is None:
                    first_diff = rec
        if first_diff is not None:
            # Keep scanning remaining requests only for count/completeness after recording first_diff.
            pass

    if missing:
        print(f"MISSING entries ({label_a} vs {label_b}): {missing[:5]}", file=sys.stderr)
    if not diffs and not missing:
        print(
            f"IDENTICAL: {compared_reqs} request pair(s), {compared_layers} layer entries matched ({label_a} vs {label_b})."
        )
        return True
    if diffs:
        req_a, req_b, l, which, ea, eb = diffs[0]
        print(
            f"DIFF: first divergence at req {req_a} ({label_a}, T={ea['T']}) vs req {req_b} ({label_b}, T={eb['T']}), "
            f"layer {l}, buffer={which}: "
            f"stage {ea['stage']} vs {eb['stage']}, hidden {ea['hidden']} vs {eb['hidden']}"
        )
        print(f"Total differing layer entries: {len(diffs)} across compared_reqs={compared_reqs}")
    return False


def main():
    ap = argparse.ArgumentParser(description="Compare STRATA_LAYER_HASH files.")
    ap.add_argument("files", nargs="+", help="One file (with --half-split/--reqs) or two files (A B)")
    ap.add_argument("--half-split", action="store_true", help="Compare req i vs req i+pairs inside one file")
    ap.add_argument("--pairs", type=int, default=36, help="Number of prompts in first round for --half-split (default: 36)")
    ap.add_argument("--reqs", nargs=2, type=int, metavar=("REQ_A", "REQ_B"), help="Compare two request ids within one file")
    args = ap.parse_args()

    if args.half_split or args.reqs:
        if len(args.files) != 1:
            ap.error("--half-split / --reqs take exactly 1 file")
        data = load_hashes(args.files[0])
        if args.reqs:
            pairs = [(args.reqs[0], args.reqs[1])]
        else:
            pairs = [(i, i + args.pairs) for i in range(args.pairs)]
        ok = diff_request_maps(data, data, pairs, "round0", "round1")
        sys.exit(0 if ok else 1)

    if len(args.files) != 2:
        ap.error("Provide two hash files <file_a> <file_b>, or use --half-split / --reqs on one file")
    data_a = load_hashes(args.files[0])
    data_b = load_hashes(args.files[1])
    reqs = sorted(set(data_a.keys()) | set(data_b.keys()))
    if not reqs:
        print("EMPTY: no hash entries found in either file.", file=sys.stderr)
        sys.exit(2)
    pairs = [(r, r) for r in reqs]
    ok = diff_request_maps(data_a, data_b, pairs, args.files[0], args.files[1])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
