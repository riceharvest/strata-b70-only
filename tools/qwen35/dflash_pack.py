#!/usr/bin/env python3
"""dflash_pack.py - the DFlash 2 block drafter of Qwen3.8-27B as a DFLASH2 pack for strata-q36 (sycl/src/qwen35/q36_dflash.*).

    python3 tools/qwen35/dflash_pack.py --ckpt /mnt/ssd/b70-prep/dflash2 --out /mnt/ssd/models/qwen38-27b/dflash2.pack \
        [--clip-search 5] [--layers N] [--report]

The checkpoint is incoai/Qwen3.8-27B-DFlash2 (config.json + model.safetensors, BF16).  Its 5 layers hold ~1.6 B weights in
matrices; they are quantized here, round to nearest (RTN) with an MSE search over a few range clips a group, to the SAME
int4 g128 asymmetric record the engine already streams for the target (tools/qwen35/b70w4_pack.py: records of 16 rows x 256
inputs, 2,176 bytes; weight = scale x (q - zero), scale fp16, zero 1 .. 16).  The pack code of b70w4_pack.py is reused, so the
layout is the one q36xr::mmvq_w4h reads.

File: "DFL2W4\\0\\1", u64 length of a JSON index, the index, the blobs at 4096-byte aligned offsets.  The index is
{"format": "DFLASH2", "config": {...}, "tensors": {name: {"off", "bytes", "kind", "n_in", "n_out"}}} with exactly that key
order (the C++ reader scans it).  kind "w4": records, n_out x n_in.  kind "f32": float32 [n_out][n_in] (n_out = leading dims
folded, n_in the last dim).  kind "bf16": raw 16-bit brain floats, the same shape rule.

Tensors (n_out x n_in):
    fc          5120 x 25600   the 5 tapped hidden states (concatenated in target_layer_ids order) -> hidden
    kv_all      10240 x 5120   layer-major: layer l rows [2048 l, 2048 l + 1024) = k_proj, the next 1024 = v_proj.  One weight
                               pass serves a batch of context rows (all 5 layers); a block pass reads one layer's slice
    l<i>.q      4096 x 5120    q_proj
    l<i>.o      5120 x 4096    o_proj
    l<i>.gate   17408 x 5120   mlp.gate_proj
    l<i>.up     17408 x 5120   mlp.up_proj
    l<i>.down   5120 x 17408   mlp.down_proj
    l<i>.akp    1280 x 5120    attention_conv.kernel_projection  (rows = side x tap x group: side*640 + tap*320 + group)
    l<i>.mkp    1280 x 5120    mlp_conv.kernel_projection
    hproj       256 x 5120     candidate_selector.hidden_projection
  f32: hidden_norm [5120], norm [5120], l<i>.in_ln, l<i>.post_ln [5120], l<i>.qn, l<i>.kn [128],
       l<i>.abase, l<i>.mbase [2 sides][2 taps][5120]
  bf16: pred_cb, succ_cb [248320][256] (the selector's codebooks; only gathered rows are read)

Memory: a row block of 256 rows at a time (< 6 GB).  Run with `systemd-run --user --scope -p MemoryMax=12G nice -n 10`.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b70w4_pack as BP   # noqa: E402  (pack(): the record layout; REC_DW)

MAGIC = b"DFL2W4\0\1"
REC_BYTES = 4 * BP.REC_DW
GROUP = 128


# ---------------------------------------------------------------------------------------------------- bf16 / safetensors
def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


class SafeTensors:
    """.safetensors read with numpy alone: u64 header length, a JSON header {name: {dtype, shape, data_offsets}}, the data."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > (64 << 20):
                raise SystemExit(path + ": not a safetensors file")
            self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.h.pop("__metadata__", None)

    def keys(self):
        return list(self.h)

    def shape(self, name):
        return self.h[name]["shape"]

    def get(self, name, rows=None):
        """float32 array of the tensor (BF16 / F16 / F32), optionally only rows [lo, hi)."""
        t = self.h[name]
        a, b = t["data_offsets"]
        item = {"BF16": 2, "F16": 2, "F32": 4}[t["dtype"]]
        shp = list(t["shape"])
        row_elems = int(np.prod(shp[1:])) if len(shp) > 1 else 1
        if rows is not None:
            lo, hi = rows
            off, cnt, shp = self.base + a + lo * row_elems * item, (hi - lo) * row_elems, [hi - lo] + shp[1:]
        else:
            off, cnt = self.base + a, (b - a) // item
        dt = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}[t["dtype"]]
        raw = np.fromfile(self.path, dtype=dt, count=cnt, offset=off).reshape(shp)
        if t["dtype"] == "BF16":
            return bf16_to_f32(raw)
        return raw.astype(np.float32)

    def raw_u16(self, name):
        t = self.h[name]
        a, b = t["data_offsets"]
        assert t["dtype"] == "BF16"
        return np.fromfile(self.path, dtype=np.uint16, count=(b - a) // 2, offset=self.base + a).reshape(t["shape"])


# ---------------------------------------------------------------------------------------------------- quantization
def quantize_rtn(w, clip_search=5):
    """w float32 [n_out][n_in] -> q uint8 [n_out][n_in] (0 .. 15), z uint8 [G][n_out] (1 .. 16), sc uint16 [G][n_out] (fp16 bits).

    Groups of 128 inputs, asymmetric: weight ~ s x (q - z), s rounded to fp16, z an integer in 1 .. 15 (the record cannot store 0).
    clip_search = number of candidate range shrinks (1.0, 0.95, ...), the one with the lowest squared error a group wins (1: plain RTN).
    """
    n_out, n_in = w.shape
    if n_in % GROUP:
        raise ValueError("n_in %d is not a multiple of %d" % (n_in, GROUP))
    G = n_in // GROUP
    wg = np.ascontiguousarray(w, dtype=np.float32).reshape(n_out, G, GROUP)
    lo0 = np.minimum(wg.min(axis=2), 0.0)
    hi0 = np.maximum(wg.max(axis=2), 0.0)
    best_err = np.full((n_out, G), np.inf, dtype=np.float64)
    best_q = np.zeros(wg.shape, dtype=np.uint8)
    best_z = np.ones((n_out, G), dtype=np.uint8)
    best_s = np.ones((n_out, G), dtype=np.float16)
    for k in range(max(1, clip_search)):
        t = 1.0 - 0.05 * k
        lo, hi = lo0 * t, hi0 * t
        s0 = (hi - lo) / 15.0
        s0 = np.where(s0 > 0, s0, 1.0)
        z = np.clip(np.rint(-lo / s0), 1, 15)
        # a zero of at least 1 means the grid reaches only 15 - z steps up and z steps down: widen the scale to cover both ends
        up = np.where(15 - z > 0, hi / np.maximum(15 - z, 1), 0.0)
        dn = -lo / z
        s = np.maximum(np.maximum(up, dn), 1e-30)
        s = np.where(np.maximum(up, dn) > 0, s, 1.0)
        s16 = s.astype(np.float16)
        sf = s16.astype(np.float32)
        sf = np.where(sf > 0, sf, 1.0)
        q = np.clip(np.rint(wg / sf[:, :, None]) + z[:, :, None], 0, 15)
        err = (((q - z[:, :, None]) * sf[:, :, None] - wg) ** 2).sum(axis=2, dtype=np.float64)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_q = np.where(better[:, :, None], q.astype(np.uint8), best_q)
        best_z = np.where(better, z.astype(np.uint8), best_z)
        best_s = np.where(better, s16, best_s)
    q = best_q.reshape(n_out, n_in)
    z = np.ascontiguousarray(best_z.T)                                  # [G][n_out]
    sc = np.ascontiguousarray(best_s.T).view(np.uint16)                 # [G][n_out]
    return q, z, sc


def dequant_qzs(q, z, sc):
    """The float32 matrix the engine's product sees: scale x (q - zero), the scale in fp16."""
    n_out, n_in = q.shape
    g = np.arange(n_in) // GROUP
    s = sc.view(np.float16).astype(np.float32)             # [G][n_out]
    zz = z.astype(np.float32)
    return s.T[:, g] * (q.astype(np.float32) - zz.T[:, g])


def unpack_records(rec, n_out, n_in):
    """The inverse of b70w4_pack.pack: records [tile][chunk][544] u32 -> (q [n_out][n_in] u8, z [G][n_out] u8, sc [G][n_out] u16)."""
    T, C = n_out // 16, n_in // 256
    rec = np.asarray(rec, dtype="<u4").reshape(T, C, BP.REC_DW)
    qs = rec[:, :, :512].reshape(T, C, 4, 8, 16)               # tile, chunk, pair, dword, row
    by = np.ascontiguousarray(qs.transpose(0, 4, 1, 2, 3)).view(np.uint8).reshape(T, 16, C, 4, 32)   # tile,row,chunk,pair,byte
    q = np.empty((T, 16, C, 4, 2, 32), dtype=np.uint8)
    q[:, :, :, :, 0, :] = by & 15
    q[:, :, :, :, 1, :] = by >> 4
    q = q.reshape(n_out, n_in)
    m = rec[:, :, 512:].reshape(T, C, 2, 16)                   # tile, chunk, group-in-chunk, row
    m = np.ascontiguousarray(m.transpose(0, 3, 1, 2)).reshape(n_out, C * 2)   # [n_out][G]
    sc = np.ascontiguousarray((m & 0xFFFF).astype(np.uint16).T)
    z = np.ascontiguousarray((m >> 16).astype(np.uint8).T)
    return q, z, sc


def pack_matrix_blocks(w_rows, clip_search):
    """Quantize + pack a float32 [n_out][n_in] matrix in row blocks; returns (records u32 array [T][C][544], rel_err, n_weights)."""
    n_out, n_in = w_rows.shape
    blocks, err2, ref2 = [], 0.0, 0.0
    for r0 in range(0, n_out, 256):
        w = w_rows[r0:r0 + 256]
        q, z, sc = quantize_rtn(w, clip_search)
        d = dequant_qzs(q, z, sc)
        err2 += float(((d - w) ** 2).sum(dtype=np.float64))
        ref2 += float((w ** 2).sum(dtype=np.float64))
        blocks.append(BP.pack(q, z, sc))
    return np.concatenate(blocks, axis=0), (err2 / max(ref2, 1e-30)) ** 0.5, n_out * n_in


# ---------------------------------------------------------------------------------------------------- the pack file
class PackWriter:
    def __init__(self, path):
        self.path, self.tmp = path, path + ".blobs"
        self.f = open(self.tmp, "wb")
        self.off = 0
        self.index = {}

    def add(self, name, arr, kind, n_in, n_out):
        pad = (-self.off) % 4096
        self.f.write(b"\0" * pad)
        self.off += pad
        data = arr.tobytes()
        self.f.write(data)
        self.index[name] = {"off": self.off, "bytes": len(data), "kind": kind, "n_in": int(n_in), "n_out": int(n_out)}
        self.off += len(data)

    def finish(self, cfg):
        self.f.close()
        js = json.dumps({"format": "DFLASH2", "version": 1, "record_bytes": REC_BYTES, "config": cfg, "tensors": self.index}).encode()
        head = MAGIC + struct.pack("<Q", len(js)) + js
        head += b"\0" * ((-len(head)) % 4096)
        with open(self.path, "wb") as fo, open(self.tmp, "rb") as fb:
            fo.write(head)
            while True:
                buf = fb.read(64 << 20)
                if not buf:
                    break
                fo.write(buf)
        os.remove(self.tmp)
        return len(head) + self.off


def read_pack_index(path):
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise SystemExit(path + " is not a DFLASH2 pack")
        n = struct.unpack("<Q", f.read(8))[0]
        js = json.loads(f.read(n))
    return js, (16 + n + 4095) // 4096 * 4096


def read_pack_tensor(path, js, base, name):
    """float32 [n_out][n_in] (or the leading dims folded) of a pack tensor; bf16 kinds as raw uint16."""
    t = js["tensors"][name]
    with open(path, "rb") as f:
        f.seek(base + t["off"])
        raw = f.read(t["bytes"])
    if t["kind"] == "w4":
        rec = np.frombuffer(raw, dtype="<u4")
        q, z, sc = unpack_records(rec, t["n_out"], t["n_in"])
        return dequant_qzs(q, z, sc)
    if t["kind"] == "f32":
        return np.frombuffer(raw, dtype="<f4").reshape(t["n_out"], t["n_in"]).copy()
    if t["kind"] == "bf16":
        return np.frombuffer(raw, dtype="<u2").reshape(t["n_out"], t["n_in"]).copy()
    raise SystemExit("unknown kind " + t["kind"])


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="folder with config.json and model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-search", type=int, default=5, help="range clips tried a group (1 = plain RTN)")
    ap.add_argument("--layers", type=int, default=-1, help="only the first N layers (a test; the pack is then not loadable)")
    ap.add_argument("--report", action="store_true", help="print the relative error of every matrix")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    dc = cfg["dflash_config"]
    st = SafeTensors(os.path.join(a.ckpt, "model.safetensors"))
    L = cfg["num_hidden_layers"] if a.layers < 0 else min(a.layers, cfg["num_hidden_layers"])
    H, FF, NH, NKV, HD = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
    if (H, FF, NH, NKV, HD) != (5120, 17408, 32, 8, 128) or cfg.get("is_causal", True) or dc["block_size"] != 8:
        raise SystemExit("this pack is for the 27B drafter: hidden 5120, ffn 17408, 32 / 8 heads of 128, non-causal, block 8")
    if dc["conv_kernel_size"] != 2 or dc["conv_group_size"] != 16 or dc["selector_rank"] != 256 or dc["selector_top_k"] != 16:
        raise SystemExit("unexpected conv / selector sizes: " + json.dumps(dc))
    out_cfg = {"hidden": H, "ffn": FF, "layers": L, "heads": NH, "kv_heads": NKV, "head_dim": HD, "window": cfg["sliding_window"],
               "rope_theta": float(cfg["rope_parameters"]["rope_theta"]), "eps": float(cfg["rms_norm_eps"]), "block": dc["block_size"],
               "mask_token": dc["mask_token_id"], "taps": dc["target_layer_ids"], "conv_group": dc["conv_group_size"],
               "conv_taps": dc["conv_kernel_size"], "rank": dc["selector_rank"], "top_k": dc["selector_top_k"],
               "vocab": cfg["vocab_size"], "clip_search": a.clip_search}
    w = PackWriter(a.out)
    rep, tot_bytes = [], 0

    def mat(name, ckpt_names, n_in_expect):
        nonlocal tot_bytes
        parts = [st.get(k) for k in ckpt_names]
        wm = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        n_out, n_in = wm.shape
        if n_in != n_in_expect:
            raise SystemExit("%s: %d inputs, expected %d" % (name, n_in, n_in_expect))
        rec, rel, nw = pack_matrix_blocks(wm, a.clip_search)
        w.add(name, rec.astype("<u4"), "w4", n_in, n_out)
        tot_bytes += rec.nbytes
        rep.append((name, rel))
        if a.report:
            print("%-14s %6d x %-6d relative error %.4f" % (name, n_out, n_in, rel), flush=True)
        return rel

    def vec(name, key, kind="f32"):
        t = st.get(key)
        w.add(name, t.astype("<f4"), "f32", t.shape[-1], int(np.prod(t.shape[:-1])) if t.ndim > 1 else 1)

    mat("fc", ["fc.weight"], 5 * H)
    # kv_all: layer-major, [k | v] a layer
    kv_names = []
    for l in range(L):
        kv_names += ["layers.%d.self_attn.k_proj.weight" % l, "layers.%d.self_attn.v_proj.weight" % l]
    mat("kv_all", kv_names, H)
    for l in range(L):
        p = "layers.%d." % l
        mat("l%d.q" % l, [p + "self_attn.q_proj.weight"], H)
        mat("l%d.o" % l, [p + "self_attn.o_proj.weight"], NH * HD)
        mat("l%d.gate" % l, [p + "mlp.gate_proj.weight"], H)
        mat("l%d.up" % l, [p + "mlp.up_proj.weight"], H)
        mat("l%d.down" % l, [p + "mlp.down_proj.weight"], FF)
        mat("l%d.akp" % l, [p + "attention_conv.kernel_projection.weight"], H)
        mat("l%d.mkp" % l, [p + "mlp_conv.kernel_projection.weight"], H)
        vec("l%d.in_ln" % l, p + "input_layernorm.weight")
        vec("l%d.post_ln" % l, p + "post_attention_layernorm.weight")
        vec("l%d.qn" % l, p + "self_attn.q_norm.weight")
        vec("l%d.kn" % l, p + "self_attn.k_norm.weight")
        vec("l%d.abase" % l, p + "attention_conv.base_kernel")
        vec("l%d.mbase" % l, p + "mlp_conv.base_kernel")
        print("layer %d packed (%.2f GiB so far)" % (l, w.off / 2 ** 30), flush=True)
    mat("hproj", ["candidate_selector.hidden_projection.weight"], H)
    vec("hidden_norm", "hidden_norm.weight")
    vec("norm", "norm.weight")
    for nm, key in (("pred_cb", "candidate_selector.predecessor_codebook"), ("succ_cb", "candidate_selector.successor_codebook")):
        u = st.raw_u16(key)
        w.add(nm, np.ascontiguousarray(u, dtype="<u2"), "bf16", u.shape[1], u.shape[0])
    size = w.finish(out_cfg)
    rel = np.array([r for _, r in rep])
    print("%s: %d tensors, %.2f GiB (w4 records %.2f GiB), relative error of the matrices: mean %.4f, worst %.4f (%s)" %
          (a.out, len(w.index), size / 2 ** 30, tot_bytes / 2 ** 30, rel.mean(), rel.max(), max(rep, key=lambda x: x[1])[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
