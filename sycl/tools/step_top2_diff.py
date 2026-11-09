#!/usr/bin/env python3
"""step_top2_diff.py FILE: where does a batch slot's decode leave the solo decode of the same conversation?

FILE is what the engine writes with STRATA_STEP_TOP2=FILE (generate.cpp, step_top2_line), one line per kept decode row:

    S <request> <pos> <fed> <pick> <id1> <logit1> <id2> <logit2>     a solo window's row
    B <slot>    <pos> <fed> <pick> <id1> <logit1> <id2> <logit2>     a batch window's row

Run the conversation alone first, then in a slot, on one engine (agent-test/batch2.py does: solo A, solo B, A and B
together).  A batch row is matched to the solo row with the same <pos> and <fed>; the first solo line of a key is kept,
so the solo run that came before the batch windows is the reference.  For each slot the first row whose <pick> differs
is printed with both margins (logit1 - logit2).  A margin near zero on both sides is a tie that arithmetic can flip; a
clear margin on both sides means the two paths computed different logits from the same context (state or inputs).
Rows after that one feed different tokens and no longer match; they are counted, not compared.
"""
import sys


def parse(line):
    f = line.split()
    if len(f) != 9 or f[0] not in ("S", "B"):
        return None
    try:
        return {"kind": f[0], "who": int(f[1]), "pos": int(f[2]), "fed": int(f[3]), "pick": int(f[4]),
                "id1": int(f[5]), "l1": float(f[6]), "id2": int(f[7]), "l2": float(f[8])}
    except ValueError:
        return None


def show(tag, r):
    return (f"  {tag}: pick {r['pick']}  top1 {r['id1']} ({r['l1']:.6f})  top2 {r['id2']} ({r['l2']:.6f})  "
            f"margin {r['l1'] - r['l2']:.6f}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    solo, slots, skipped = {}, {}, 0
    with open(sys.argv[1]) as fh:
        for line in fh:
            r = parse(line)
            if r is None:
                skipped += bool(line.strip())
                continue
            if r["kind"] == "S":
                solo.setdefault((r["pos"], r["fed"]), r)
            else:
                slots.setdefault(r["who"], []).append(r)
    if skipped:
        print(f"{skipped} line(s) not understood (nologits or cut off): left out")
    if not slots:
        print("no batch rows (B lines) in the file")
        return 1
    worst = 0
    for slot in sorted(slots):
        rows = slots[slot]
        same = unmatched = 0
        first_delta = None
        diverged = None
        for i, r in enumerate(rows):
            s = solo.get((r["pos"], r["fed"]))
            if s is None:
                unmatched += 1
                continue
            if first_delta is None:
                first_delta = (r["pos"], r["l1"] - s["l1"], r["id1"] == s["id1"])
            if s["pick"] != r["pick"]:
                diverged = (i, s, r)
                break
            same += 1
        print(f"slot {slot}: {len(rows)} rows, {same} matched a solo row and picked the same token, "
              f"{unmatched} had no solo row")
        if first_delta is not None:
            print(f"  first matched row (pos {first_delta[0]}): top-1 logit batch - solo = {first_delta[1]:+.6f}"
                  f"{'' if first_delta[2] else ' (a different top-1 token)'}; 0 means the same arithmetic so far")
        if diverged is None:
            print("  no matched row picked a different token")
            continue
        worst = 1
        i, s, r = diverged
        print(f"  first different pick: batch row {i} of the slot, pos {r['pos']}, fed {r['fed']} "
              f"({len(rows) - i - 1} rows after it)")
        print(show("solo ", s))
        print(show("batch", r))
    return worst


if __name__ == "__main__":
    sys.exit(main())
