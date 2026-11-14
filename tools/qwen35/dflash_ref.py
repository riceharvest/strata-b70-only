#!/usr/bin/env python3
"""dflash_ref.py - CPU reference of the DFlash 2 block drafter for Qwen3.8-27B (numpy only, float32).

It is the spec of sycl/src/qwen35/q36_dflash.dp.cpp, written from vLLM's qwen3_dflash2.py / qwen3_dflash.py and checked against
the repo's earlier plain-torch port (tools/dflash2/dflash2_ref.py, same maths).  The decisions it encodes (PLAN.md D1-D9):

  * context rows: the full residual stream after target layers 5 19 33 47 61 (0-based, after the layer's MLP add), concatenated
    in that order, stored as bf16.  fc (25600 -> 5120), then ONE hidden_norm; per layer only k_proj / v_proj run on them; K gets
    the per-head k_norm and RoPE at the row's own position, V is as projected.  No q, no MLP on context rows.
  * block: 8 rows at positions P .. P+7, P = last committed position + 1.  Row 0 = the token the target just chose (the anchor),
    rows 1-7 = mask token 248070; embeddings are the TARGET's.  Every layer: rms -> grouped 2-tap conv -> q/k/v ->
    q/k norm + RoPE (NeoX, theta 1e7, all 128 dims) -> attention -> o -> conv -> +residual -> rms -> conv -> MLP -> conv -> +residual.
  * attention is NOT causal: row i (position P+i) sees context position j when |P+i-j| < 2048 and every block row.
  * the conv does not see the context: y[r] = (b0 + d0[r]) x[r] + (r >= 1) (b1 + d1[r]) x[r-1], r = row inside the block of 8.
  * final norm; rows 1-7 go through the target's head (top-16 by raw logit) and the selector walks them left to right.

Weights come from the BF16 checkpoint (--ckpt DIR) or from a pack made by dflash_pack.py (--pack FILE: the exact int4 the engine
streams, dequantized).  --act16 rounds every product's input to fp16, as the engine's W4A16 products do.

Subcommands
  selftest                      checks that need no data (conv, window, pack round trip); exit code 0 = pass
  replay DUMP [--ckpt D | --pack F] [--gguf G] [--act16] [--steps N]
                                replays an engine dump (STRATA_Q36_DFLASH_DUMP) and prints the agreement with the engine
  synth --ckpt D --pack F       the BF16 weights against the pack's on a synthetic context: the quantization's own cost (hidden
                                relative error, no head needed)

Dump format (written by the engine, one folder):
  meta.json     {"n_embd", "taps", "mask_token", "block", "window", "head_subset", "n_steps", ...}
  ctx.bin       records [i32 pos][i32 0][5 x 5120 u16 bf16]: every row flushed into the drafter's context, in order (a later record
                of a position replaces the earlier)
  steps.jsonl   one json a draft call: step, P, anchor, ctx_records (records flushed before it), valid_lo, cand [7][16], unary [7][16], path [7]
  blocks.bin    per step 10 x 5120 float32: embed(anchor), embed(mask), then the 8 final-normed block rows
  subset.bin    (if head_subset) int32 token ids of the draft head's rows
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dflash_pack as DP   # noqa: E402

H, NH, NKV, HD, FF, LAYERS, WIN = 5120, 32, 8, 128, 17408, 5, 2048
BLOCK, GROUP, TOPK, RANK = 8, 16, 16, 256


def rms(x, w, eps=1e-6):
    return (x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * w


def rope_tables(pos, theta=1.0e7, hd=HD):
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.asarray(pos, dtype=np.float64)[:, None] * inv[None, :]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)   # [n][64]


def rope_neox(x, cos, sin):
    """x [n][heads][128]; rotate_half over the whole head: (x1, x2) -> (x1 c - x2 s, x2 c + x1 s)."""
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


def conv_block(x, dyn, base):
    """x [8][H]; dyn [8][2 taps][H/16 groups]; base [2][H] -> y[r] = (b0 + d0[r]) x[r] + (r >= 1) (b1 + d1[r]) x[r-1]."""
    g = H // GROUP
    coef = base[None, :, :] + np.repeat(dyn, GROUP, axis=2)        # [8][2][H]
    y = coef[:, 0, :] * x
    y[1:] += coef[1:, 1, :] * x[:-1]
    return y


def silu(x):
    return x / (1.0 + np.exp(-x))


class Weights:
    def __init__(self, ckpt=None, pack=None, layers=LAYERS):
        self.layers = layers
        self.cache = {}
        if pack:
            self.pack, (self.js, self.base) = pack, DP.read_pack_index(pack)
            self.st = None
        else:
            self.st = DP.SafeTensors(os.path.join(ckpt, "model.safetensors"))
            self.pack = None

    def _ck(self, name):
        if name == "fc":
            return "fc.weight"
        if name == "hproj":
            return "candidate_selector.hidden_projection.weight"
        if name in ("hidden_norm", "norm"):
            return name + ".weight"
        l, kind = name.split(".", 1)
        p = "layers.%s." % l[1:]
        return p + {"q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
                    "o": "self_attn.o_proj.weight", "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight",
                    "down": "mlp.down_proj.weight", "akp": "attention_conv.kernel_projection.weight",
                    "mkp": "mlp_conv.kernel_projection.weight", "in_ln": "input_layernorm.weight",
                    "post_ln": "post_attention_layernorm.weight", "qn": "self_attn.q_norm.weight", "kn": "self_attn.k_norm.weight",
                    "abase": "attention_conv.base_kernel", "mbase": "mlp_conv.base_kernel"}[kind]

    def get(self, name):
        if name in self.cache:
            return self.cache[name]
        if self.pack:
            if name[0] == "l" and name.split(".", 1)[1] in ("k", "v"):
                l = int(name.split(".")[0][1:])
                kv = self.get("kv_all")
                a = 2048 * l + (0 if name.endswith(".k") else 1024)
                out = kv[a:a + 1024]
            else:
                out = DP.read_pack_tensor(self.pack, self.js, self.base, name)
                if name in ("pred_cb", "succ_cb"):
                    out = DP.bf16_to_f32(out)
        else:
            if name in ("pred_cb", "succ_cb"):
                out = self.st.get("candidate_selector." + ("predecessor_codebook" if name == "pred_cb" else "successor_codebook"))
            else:
                out = self.st.get(self._ck(name))
        self.cache[name] = out
        return out

    def vec(self, name):
        return self.get(name).reshape(-1)

    def codebook_rows(self, name, ids):
        """rows of a codebook without loading all of it (the checkpoint) or from the pack."""
        if self.pack:
            t = self.js["tensors"][name]
            out = np.empty((len(ids), t["n_in"]), np.float32)
            with open(self.pack, "rb") as f:
                for k, i in enumerate(ids):
                    f.seek(self.base + t["off"] + int(i) * t["n_in"] * 2)
                    out[k] = DP.bf16_to_f32(np.frombuffer(f.read(t["n_in"] * 2), dtype="<u2"))
            return out
        key = "candidate_selector." + ("predecessor_codebook" if name == "pred_cb" else "successor_codebook")
        return np.stack([self.st.get(key, rows=(int(i), int(i) + 1))[0] for i in ids])


class Drafter:
    def __init__(self, w, theta=1.0e7, eps=1e-6, mask_token=248070, act16=False, max_pos=8192):
        self.w, self.theta, self.eps, self.mask, self.act16 = w, theta, eps, mask_token, act16
        self.L = w.layers
        self.K = np.zeros((self.L, max_pos + BLOCK, NKV, HD), np.float32)
        self.V = np.zeros_like(self.K)
        self.have = np.zeros(max_pos + BLOCK, bool)

    def _a(self, x):   # the input of a W4A16 product
        return x.astype(np.float16).astype(np.float32) if self.act16 else x

    def mm(self, x, name):
        return self._a(x) @ self.w.get(name).T

    # ---- context
    def add_context(self, taps, pos):
        """taps [n][5H] float32 (bf16 values), pos [n] int."""
        w, n = self.w, len(pos)
        h = rms(self.mm(taps, "fc"), w.vec("hidden_norm"), self.eps)
        cos, sin = rope_tables(pos, self.theta)
        for l in range(self.L):
            k = self.mm(h, "l%d.k" % l).reshape(n, NKV, HD)
            v = self.mm(h, "l%d.v" % l).reshape(n, NKV, HD)
            k = rope_neox(rms(k, w.vec("l%d.kn" % l), self.eps), cos, sin)
            self.K[l, pos] = k
            self.V[l, pos] = v
        self.have[pos] = True

    # ---- one block
    def block(self, P, embed_anchor, embed_mask, valid_lo=0):
        """-> final-normed rows [8][H]."""
        w, eps = self.w, self.eps
        x = np.concatenate([embed_anchor[None], np.repeat(embed_mask[None], BLOCK - 1, axis=0)]).astype(np.float32)
        pos = np.arange(P, P + BLOCK)
        cos, sin = rope_tables(pos, self.theta)
        lo = max(valid_lo, P - WIN + 1, 0)
        cpos = np.array([j for j in range(lo, P) if self.have[j]], dtype=np.int64)
        for l in range(self.L):
            p = "l%d." % l
            r = x
            h = rms(x, w.vec(p + "in_ln"), eps)
            dyn = self.mm(h, p + "akp").reshape(BLOCK, 2, 2, H // GROUP)       # row, side, tap, group
            h = conv_block(h, dyn[:, 0], w.get(p + "abase").reshape(2, 2, H)[0])
            q = self.mm(h, p + "q").reshape(BLOCK, NH, HD)
            k = self.mm(h, p + "k").reshape(BLOCK, NKV, HD)
            v = self.mm(h, p + "v").reshape(BLOCK, NKV, HD)
            q = rope_neox(rms(q, w.vec(p + "qn"), eps), cos, sin)
            k = rope_neox(rms(k, w.vec(p + "kn"), eps), cos, sin)
            Kk = np.concatenate([self.K[l, cpos], k], axis=0)                   # [c + 8][8][128]
            Vv = np.concatenate([self.V[l, cpos], v], axis=0)
            kpos = np.concatenate([cpos, pos])
            vis = np.abs(pos[:, None] - kpos[None, :]) < WIN                     # [8][c + 8]
            o = np.empty((BLOCK, NH, HD), np.float32)
            for hh in range(NH):
                kv = hh // (NH // NKV)
                s = (q[:, hh, :] @ Kk[:, kv, :].T) * (HD ** -0.5)
                s = np.where(vis, s, -np.inf)
                s = s - s.max(axis=1, keepdims=True)
                a = np.exp(s)
                a /= a.sum(axis=1, keepdims=True)
                o[:, hh, :] = a @ Vv[:, kv, :]
            o = self.mm(o.reshape(BLOCK, NH * HD), p + "o")
            o = conv_block(o, dyn[:, 1], w.get(p + "abase").reshape(2, 2, H)[1])
            x = r + o
            r = x
            h = rms(x, w.vec(p + "post_ln"), eps)
            dyn = self.mm(h, p + "mkp").reshape(BLOCK, 2, 2, H // GROUP)
            h = conv_block(h, dyn[:, 0], w.get(p + "mbase").reshape(2, 2, H)[0])
            m = silu(self.mm(h, p + "gate")) * self.mm(h, p + "up")
            m = self.mm(m, p + "down")
            m = conv_block(m, dyn[:, 1], w.get(p + "mbase").reshape(2, 2, H)[1])
            x = r + m
        return rms(x, w.vec("norm"), eps)

    def select(self, hrows, cand, unary, anchor):
        """hrows [7][H] (the final-normed rows 1..7); cand [7][16] token ids; unary [7][16] -> the walked path (7 token ids)."""
        w = self.w
        hp = self.mm(hrows, "hproj")                                              # [7][256]
        path, prev = [], int(anchor)
        for i in range(len(hrows)):
            pr = w.codebook_rows("pred_cb", [prev])[0]
            su = w.codebook_rows("succ_cb", cand[i])
            s = np.asarray(unary[i], np.float32) + su @ (pr * hp[i])
            j = int(np.argmax(s))
            prev = int(cand[i][j])
            path.append(prev)
        return path


# ---------------------------------------------------------------------------------------------------- the target's head (GGUF)
class Head:
    def __init__(self, path, name="output.weight"):
        import gguf
        from gguf.quants import dequantize
        self.dq = dequantize
        rd = gguf.GGUFReader(path)
        t = {x.name: x for x in rd.tensors}
        self.t = t[name]
        self.rows = int(self.t.shape[1])
        self.raw = np.asarray(self.t.data).reshape(self.rows, -1)

    def rows_of(self, ids):
        return self.dq(self.raw[np.asarray(ids)], self.t.tensor_type).reshape(len(ids), -1).astype(np.float32)

    def logits_cand(self, h, ids):
        return self.rows_of(ids) @ h

    def top1_full(self, h, chunk=8192):
        best, bi = -np.inf, -1
        for a in range(0, self.rows, chunk):
            r = self.dq(self.raw[a:a + chunk], self.t.tensor_type).reshape(-1, int(self.t.shape[0])).astype(np.float32)
            l = r @ h
            j = int(np.argmax(l))
            if l[j] > best:
                best, bi = float(l[j]), a + j
        return bi


# ---------------------------------------------------------------------------------------------------- dump replay
def read_ctx(path):
    rec = 8 + 5 * H * 2
    raw = np.fromfile(path, dtype=np.uint8)
    n = len(raw) // rec
    raw = raw[:n * rec].reshape(n, rec)
    pos = raw[:, :4].copy().view("<i4")[:, 0]
    taps = DP.bf16_to_f32(raw[:, 8:].copy().view("<u2").reshape(n, 5 * H))
    return pos, taps


def cmd_replay(a):
    meta = json.load(open(os.path.join(a.dump, "meta.json")))
    steps = [json.loads(l) for l in open(os.path.join(a.dump, "steps.jsonl")) if l.strip()]
    if a.steps:
        steps = steps[:a.steps]
    cpos, ctaps = read_ctx(os.path.join(a.dump, "ctx.bin"))
    blocks = np.fromfile(os.path.join(a.dump, "blocks.bin"), dtype=np.float32).reshape(-1, 10, H)
    w = Weights(ckpt=a.ckpt, pack=a.pack)
    max_pos = int(max(cpos.max(), max(s["P"] for s in steps))) + 16
    d = Drafter(w, mask_token=meta.get("mask_token", 248070), act16=a.act16, max_pos=max_pos)
    head = Head(a.gguf) if a.gguf else None
    done, agree1, pathok, n1, hdiff, hcos, ldiff = 0, 0, 0, 0, [], [], []
    per_pos = np.zeros(7)
    for s in steps:
        t0 = time.time()
        upto = s["ctx_records"]
        if upto > done:
            for a0 in range(done, upto, 256):
                b0 = min(upto, a0 + 256)
                d.add_context(ctaps[a0:b0], cpos[a0:b0].astype(np.int64))
            done = upto
        b = blocks[s["step"]]
        out = d.block(s["P"], b[0], b[1], s.get("valid_lo", 0))
        cand = np.array(s["cand"]); un = np.array(s["unary"], np.float32)
        eng = b[2:]
        dd = np.abs(out - eng)
        hdiff.append(float(dd.max()))
        hcos.append(float(((out * eng).sum(1) / (np.linalg.norm(out, axis=1) * np.linalg.norm(eng, axis=1) + 1e-30)).min()))
        path = d.select(out[1:], cand, un, s["anchor"])
        pathok += int(path == s["path"])
        for i in range(7):
            e1 = int(cand[i][int(np.argmax(un[i]))])
            if head is not None:
                lg = head.logits_cand(out[1 + i], cand[i])
                r1 = int(cand[i][int(np.argmax(lg))])
                ldiff.append(float(np.abs(lg - un[i]).max()))
            else:
                r1 = e1   # no head: only the hidden rows and the path are compared
            agree1 += int(e1 == r1)
            per_pos[i] += int(e1 == r1)
            n1 += 1
        print("step %3d P %6d: hidden max|d| %.4f min cos %.5f path %s (%.1fs)" % (s["step"], s["P"], hdiff[-1], hcos[-1], "same" if path == s["path"] else "DIFF", time.time() - t0), flush=True)
    n = len(steps)
    print("steps %d: hidden max abs diff worst %.4f, min cosine %.5f, selector path identical in %d (%.1f%%)" % (n, max(hdiff), min(hcos), pathok, 100.0 * pathok / max(n, 1)))
    if head is not None:
        print("top-1 over the engine's candidates agrees %d / %d = %.2f%% (per position %s), candidate logit max abs diff %.3f" %
              (agree1, n1, 100.0 * agree1 / max(n1, 1), " ".join("%.0f%%" % (100 * x / max(n, 1)) for x in per_pos), max(ldiff)))
    else:
        print("no --gguf: the top-1 check was skipped (hidden rows and path only)")
    ok = (min(hcos) > 0.99 and pathok >= 0.99 * n) if head is None else (agree1 >= 0.99 * n1)
    print("RESULT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_synth(a):
    rng = np.random.default_rng(3)
    n, P = a.rows, a.rows
    taps = DP.bf16_to_f32(DP.f32_to_bf16(rng.standard_normal((n, 5 * H)).astype(np.float32) * a.scale))
    emb = (rng.standard_normal((2, H)) * 0.02).astype(np.float32)
    res = {}
    for tag, kw in (("bf16", dict(ckpt=a.ckpt)), ("pack", dict(pack=a.pack)), ("pack+act16", dict(pack=a.pack))):
        t0 = time.time()
        d = Drafter(Weights(**kw), act16=tag.endswith("act16"), max_pos=P + 16)
        d.add_context(taps, np.arange(n))
        res[tag] = d.block(P, emb[0], emb[1])
        print("%s: %.1fs" % (tag, time.time() - t0), flush=True)
    for tag in ("pack", "pack+act16"):
        dd = res[tag] - res["bf16"]
        print("%-11s vs bf16: relative error of the 8 final rows %.4f, min row cosine %.5f" %
              (tag, np.linalg.norm(dd) / np.linalg.norm(res["bf16"]), float(((res[tag] * res["bf16"]).sum(1) / (np.linalg.norm(res[tag], axis=1) * np.linalg.norm(res["bf16"], axis=1))).min())))
    return 0


def cmd_selftest(_):
    rng = np.random.default_rng(5)
    ok = True

    def check(name, cond, info=""):
        nonlocal ok
        print("%-46s %s %s" % (name, "ok" if cond else "FAIL", info))
        ok &= bool(cond)

    # conv against a loop
    x = rng.standard_normal((BLOCK, H)).astype(np.float32)
    dyn = rng.standard_normal((BLOCK, 2, H // GROUP)).astype(np.float32)
    base = rng.standard_normal((2, H)).astype(np.float32)
    y = conv_block(x, dyn, base)
    ref = np.zeros_like(x)
    for r in range(BLOCK):
        for c in range(H):
            g = c // GROUP
            ref[r, c] = (base[0, c] + dyn[r, 0, g]) * x[r, c] + ((base[1, c] + dyn[r, 1, g]) * x[r - 1, c] if r >= 1 else 0.0)
    check("conv_block = loop (taps do not cross blocks)", np.abs(y - ref).max() < 1e-4, "%.2e" % np.abs(y - ref).max())
    # rope: norm preserving, position 0 identity, the (x1, x2) pairing is rotate_half
    v = rng.standard_normal((3, 2, HD)).astype(np.float32)
    c, s = rope_tables([0, 5, 123456])
    r = rope_neox(v, c, s)
    check("rope: position 0 is the identity", np.abs(r[0] - v[0]).max() < 1e-6)
    check("rope: preserves the norm", np.abs(np.linalg.norm(r, axis=-1) - np.linalg.norm(v, axis=-1)).max() < 1e-3)
    i = 7
    ang = 5 * 1.0e7 ** (-2 * i / HD)
    exp1 = v[1, 0, i] * np.cos(ang) - v[1, 0, i + 64] * np.sin(ang)
    check("rope: pair (i, i + 64), theta 1e7", abs(r[1, 0, i] - exp1) < 1e-5)
    # pack round trip (needs b70w4_pack)
    import b70w4_pack as BP
    wq = (rng.standard_normal((48, 768)) * 0.02).astype(np.float32)
    q, z, sc = DP.quantize_rtn(wq, 5)
    q2, z2, sc2 = DP.unpack_records(BP.pack(q, z, sc), 48, 768)
    check("pack records round trip bit-exact", (q == q2).all() and (z == z2).all() and (sc == sc2).all())
    check("zero in 1 .. 16", z.min() >= 1 and z.max() <= 16)
    err = np.linalg.norm(DP.dequant_qzs(q, z, sc) - wq) / np.linalg.norm(wq)
    check("int4 g128 relative error < 0.12", err < 0.12, "%.4f" % err)
    # bf16 rounding
    t = rng.standard_normal(1000).astype(np.float32)
    b = DP.bf16_to_f32(DP.f32_to_bf16(t))
    check("bf16 round to nearest even", np.abs(b - t).max() <= np.abs(t).max() * 2 ** -8)
    # the attention window on a tiny block: a row sees |P+i-j| < WIN and every block row (checked through Drafter.block with 1 layer is
    # too big for a self-test; the window rule is exercised here directly)
    P = 5000
    pos = np.arange(P, P + BLOCK)
    kpos = np.concatenate([np.arange(max(0, P - WIN + 1), P), pos])
    vis = np.abs(pos[:, None] - kpos[None, :]) < WIN
    check("window: row 0 sees 2047 context + 8 block keys", vis[0].sum() == 2047 + 8)
    check("window: row 7 sees 2040 context + 8 block keys", vis[7].sum() == 2040 + 8)
    check("window: not causal (row 0 sees block row 7)", bool(vis[0, -1]))
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("replay")
    r.add_argument("dump")
    r.add_argument("--ckpt")
    r.add_argument("--pack")
    r.add_argument("--gguf", help="the target's GGUF (output.weight): logits of the candidates")
    r.add_argument("--act16", action="store_true")
    r.add_argument("--steps", type=int, default=0)
    s = sub.add_parser("synth")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--pack", required=True)
    s.add_argument("--rows", type=int, default=300)
    s.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()
    if a.cmd == "replay" and not (a.ckpt or a.pack):
        ap.error("replay needs --ckpt or --pack")
    return {"selftest": cmd_selftest, "replay": cmd_replay, "synth": cmd_synth}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
