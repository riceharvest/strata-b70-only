#!/usr/bin/env python3
"""qsa_ref.py - numpy reference of one QSA layer at one decode step, against what the engine dumped.

    strata ... --no-graph (STRATA_Q36_QSA=<file.qsaidx> STRATA_Q36_QSA_DUMP=<dir> [STRATA_Q36_QSA_DUMP_LAYER=3 ..._MIN=2200 ..._COUNT=1])
    python3 -I tools/qwen35/qsa_ref.py --dump DIR --idx FILE.qsaidx

It reads DIR (meta.txt, h.f32, q.f32, gate.f32, att.f32, qidx.f32, pooled.f16, score.f32, sel.i32, nsel.i32, kcache.bin, vcache.bin) and the
indexer file and recomputes, in float64:
  1. the indexer's query:  rope(rms_head(q_proj . h) * q_norm)                      against qidx.f32
  2. the scores of the engine's pooled block keys                                  against score.f32
  3. the selection: top 512 by (score desc, block index asc), from the engine's scores AND from the recomputed ones
  4. the sparse attention over the selected blocks + the own block, from the engine's 8-bit K / V cache, times sigmoid(gate)  against att.f32
  5. (info) the dense attention over every cell, and how far the sparse output is from it.
  6. with --pool: the pooled block keys of every complete block from the layer's input rows of the WHOLE sequence (STRATA_Q36_QSA_DUMP_H=1 wrote
     hall.f32 / hall.txt; --no-graph), recomputed in float64 (raw key = rms(k_proj . h) * k_pre; block = rms(mean of 4) * k_norm, rotated to cell 4b),
     against pooled.f16; and the selection from those scores against the engine's.
Exit status 1 when a check fails.  The pooled keys themselves are not recomputed here (that needs every token's raw key): qsa_parity checks the
pooling on the card against float64.
"""
import argparse
import json
import struct
import sys

import numpy as np

KD, KH, RATIO, TOPB, HD = 128, 4, 4, 512, 256


def load_idx(path, layer):
    f = open(path, "rb")
    assert f.read(8) == b"QSAIDX\0\1"
    jl = struct.unpack("<Q", f.read(8))[0]
    js = json.loads(f.read(jl))
    data0 = (16 + jl + 4095) // 4096 * 4096
    out = {}
    for k in ("q_proj", "k_proj", "q_norm", "k_pre", "k_norm"):
        e = js["tensors"]["%d.%s" % (layer, k)]
        f.seek(data0 + e["off"])
        dt = "<f2" if e["dtype"] == "f16" else "<f4"
        out[k] = np.frombuffer(f.read(e["bytes"]), dtype=dt).astype(np.float64).reshape(e["shape"])
    return out


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(v, pos, base, n_rot):
    v = v.copy()
    h = n_rot // 2
    p = np.arange(h)
    th = pos * base ** (-2.0 * p / n_rot)
    c, s = np.cos(th), np.sin(th)
    a, b = v[..., :h].copy(), v[..., h:n_rot].copy()
    v[..., :h] = a * c - b * s
    v[..., h:n_rot] = a * s + b * c
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--pool", action="store_true", help="recompute the pooled keys from hall.f32 (needs STRATA_Q36_QSA_DUMP_H=1)")
    a = ap.parse_args()
    d = a.dump.rstrip("/") + "/"
    meta = dict(line.strip().split("=") for line in open(d + "meta.txt") if "=" in line)
    pos, ctx, vt, H, HKV = int(meta["pos"]), int(meta["ctx"]), int(meta["vt"]), int(meta["H"]), int(meta["HKV"])
    layer, nb = int(meta["layer"]), int(meta["blocks"])
    base, n_rot = float(meta["rope_base"]), int(meta["n_rot"])
    G = H // HKV
    w = load_idx(a.idx, layer)
    h = np.fromfile(d + "h.f32", dtype="<f4").astype(np.float64)
    bad = 0

    def report(name, ok, info):
        nonlocal bad
        print("%-60s %s  %s" % (name, "ok  " if ok else "FAIL", info))
        bad += 0 if ok else 1

    # 1. the indexer query
    qi = (w["q_proj"] @ h).reshape(KH, KD)
    qi = rope(rms(qi, w["q_norm"], 1e-6), pos, base, n_rot)
    qe = np.fromfile(d + "qidx.f32", dtype="<f4").astype(np.float64).reshape(KH, KD)
    e = np.abs(qi - qe).max()
    report("indexer query vs engine", e <= 1e-4 * max(1e-9, np.abs(qi).max()), "max |d| %.2e (max |q| %.2f)" % (e, np.abs(qi).max()))

    # 2. scores of the engine's pooled keys
    pooled = np.fromfile(d + "pooled.f16", dtype="<f2").astype(np.float64).reshape(nb, KD)
    sc = np.maximum(pooled @ qi.T, 0).sum(-1) / np.sqrt(KD)
    se = np.fromfile(d + "score.f32", dtype="<f4").astype(np.float64)
    e = np.abs(sc - se).max()
    report("scores of the pooled keys vs engine", e <= 4e-5 * max(1e-9, se.max()) + 1e-6, "max |d| %.2e (max score %.3f)" % (e, se.max()))

    # 3. selection
    sel = np.fromfile(d + "sel.i32", dtype="<i4")
    nsel = int(np.fromfile(d + "nsel.i32", dtype="<i4")[0])
    sel = sel[:nsel]
    want = min(nb, TOPB)
    order = np.lexsort((np.arange(nb), -se))[:want]   # the engine's scores: score desc, index asc
    ref_sel = np.sort(order)
    report("selection = top %d of the engine's scores, ties to the lower index" % want, nsel == want and np.array_equal(sel, ref_sel),
           "engine picked %d, differ in %d" % (nsel, len(set(sel.tolist()) ^ set(ref_sel.tolist()))))
    order2 = np.sort(np.lexsort((np.arange(nb), -sc))[:want])
    ov = len(set(sel.tolist()) & set(order2.tolist()))
    report("selection vs the recomputed scores (fp16 keys: ties at the boundary may flip)", ov >= want - 2, "%d of %d blocks in common" % (ov, want))
    if nb > want:
        kth = np.sort(se)[::-1][want - 1]
        print("    512th score %.6f, 513th %.6f, selected-score min %.6f" % (kth, np.sort(se)[::-1][want], se[sel].min()))

    # 6. (--pool) the pooled keys from the whole sequence's input rows
    if a.pool:
        lines = [ln.split() for ln in open(d + "hall.txt")]
        hb = np.fromfile(d + "hall.f32", dtype="<f4")
        nemb = hb.size // sum(int(t) for _, t in lines)
        hall = np.zeros((pos + 1, nemb))
        off = 0
        for p0s, ts in lines:   # (a cell written twice: the last forward wins)
            p0i, ti = int(p0s), int(ts)
            hall[p0i:p0i + ti] = hb[off * nemb:(off + ti) * nemb].reshape(ti, nemb)
            off += ti
        nbc = (pos + 1) // RATIO
        kraw = rms(hall[:nbc * RATIO] @ w["k_proj"].T, w["k_pre"], 1e-6)               # [cells][128]
        pm = rms(kraw.reshape(nbc, RATIO, KD).mean(1), w["k_norm"], 1e-6)
        pm = rope(pm, (np.arange(nbc) * RATIO)[:, None].astype(np.float64), base, n_rot)
        e = np.abs(pm - pooled[:nbc])
        tol = np.abs(pm) / 1024.0 + 5e-4
        nbad = int((e > tol).sum())
        report("pooled keys vs float64 from the sequence (%d blocks)" % nbc, nbad == 0, "max |d| %.2e, %d of %d values over the fp16 bound" % (e.max(), nbad, e.size))
        if nbad:
            bb = np.nonzero((e > tol).any(1))[0]
            print("    first bad blocks:", bb[:8].tolist(), " (of %d bad)" % len(bb))
        sc2 = np.maximum(pm @ qi.T, 0).sum(-1) / np.sqrt(KD)
        ord3 = np.sort(np.lexsort((np.arange(nbc), -sc2))[:want])
        ov3 = len(set(sel.tolist()) & set(ord3.tolist()))
        report("engine's selection vs the top %d of the float64 pooled scores" % want, ov3 >= want - 8, "%d of %d blocks in common" % (ov3, want))

    # 4. the sparse attention from the 8-bit cache
    kraw = np.fromfile(d + "kcache.bin", dtype=np.uint8)
    kcodes = kraw[: HKV * ctx * HD].view(np.int8).reshape(HKV, ctx, HD)
    kscale = kraw[HKV * ctx * HD:].view("<f2").astype(np.float64).reshape(HKV, ctx, 4)
    vraw = np.fromfile(d + "vcache.bin", dtype=np.uint8)
    vb = HKV * (vt // 32) * HD * 32
    vcodes = vraw[:vb].view(np.int8).reshape(HKV, vt // 32, HD, 32)
    vscale = vraw[vb:].view("<f4").astype(np.float64).reshape(HKV, vt)
    q = np.fromfile(d + "q.f32", dtype="<f4").astype(np.float64).reshape(H, HD)
    gate = np.fromfile(d + "gate.f32", dtype="<f4").astype(np.float64).reshape(H, HD)
    att = np.fromfile(d + "att.f32", dtype="<f4").astype(np.float64).reshape(H, HD)
    own = pos // RATIO
    blocks = [int(b) for b in sel if int(b) != own]
    cells = np.array([4 * b + j for b in blocks for j in range(4)] + list(range(own * 4, pos + 1)), dtype=np.int64)

    def attend(cells):
        out = np.zeros((H, HD))
        for kvh in range(HKV):
            kk = kcodes[kvh, cells].astype(np.float64).reshape(len(cells), 4, 64) * kscale[kvh, cells][:, :, None]
            kk = kk.reshape(len(cells), HD)
            vv = vcodes[kvh][cells // 32, :, cells % 32].astype(np.float64) * vscale[kvh, cells][:, None]   # [cells, HD]
            for g in range(G):
                hh = kvh * G + g
                s = kk @ q[hh] / 16.0
                p = np.exp(s - s.max())
                out[hh] = (p @ vv) / p.sum()
        return out / (1.0 + np.exp(-gate))

    ref = attend(cells)
    mx = max(1e-9, np.abs(ref).max())
    e = np.abs(ref - att).max()
    report("sparse attention (%d cells) vs engine" % len(cells), e <= 3e-4 * mx + 1e-6, "max |d| %.2e (max |out| %.3f)" % (e, mx))
    dense = attend(np.arange(0, pos + 1))
    print("    dense attention over %d cells: |sparse - dense| max %.3e = %.2f%% of max |dense|" % (pos + 1, np.abs(ref - dense).max(), 100 * np.abs(ref - dense).max() / max(1e-9, np.abs(dense).max())))
    print("PASS" if not bad else "FAIL: %d check(s)" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
