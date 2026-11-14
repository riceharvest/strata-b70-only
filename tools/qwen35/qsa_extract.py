#!/usr/bin/env python3
"""qsa_extract.py - the 80 QSA indexer tensors of Qwen3.8-27B as one small file for strata-q36 (STRATA_Q36_QSA=<file>).

    python3 -I tools/qwen35/qsa_extract.py --src MODEL.gguf|qsa-indexers.safetensors --out FILE.qsaidx [--dtype f16|f32] [--check]

Sources (the two published forms of the same indexers):
  * a QSA GGUF (e.g. pentacoxian-dev/Qwen3.8-27B-QSA-GSQ-RCO-GGUF, the RETRAINED indexers): tensors blk.<L>.indexer.{q_proj,k_proj}.weight,
    blk.<L>.indexer.{q_norm,k_norm}.weight, blk.<L>.indexer.k_norm.pre_weight, all F32; the norms are already stored as 1 + w
    (the converter folded it), so they are plain multipliers.
  * qsa-indexers.safetensors (higa2515/Qwen3.8-27B-QSA-BF16, the original indexers): "...layers.<L>._qsa_indexer.{q_proj.weight,
    k_proj.weight,q_norm,k_norm,k_block_norm}" F32; the norms are w and become 1 + w here (as the converter does), exactly once.

The file (little endian):   "QSAIDX\\0\\1", u64 length of a JSON index, the JSON, then the blobs at 4096-byte aligned offsets.
  JSON: {"meta": {n_embd, idx_dim, idx_heads, ratio, top_k, layers:[...], wdtype, norm_eps}, "tensors": {name: {off, bytes, shape, dtype}}}
  per attention layer L (3, 7, ..., 63):
     L.q_proj   [idx_heads * idx_dim][n_embd]   (rows = outputs; head h is rows h*128 .. h*128+127)   wdtype
     L.k_proj   [idx_dim][n_embd]                                                                  wdtype
     L.q_norm   [idx_dim]   f32   multiplier of the per-head RMS norm of the query
     L.k_pre    [idx_dim]   f32   multiplier of the RMS norm of each token's raw key   (GGUF: k_norm.pre_weight)
     L.k_norm   [idx_dim]   f32   multiplier of the RMS norm of the pooled block key    (GGUF: k_norm.weight)

WHY fp16 weights by default: the projections are read every step (52.4 M parameters = 105 MB in fp16, 210 MB in fp32 = 0.36 ms against
0.18 ms of the card's 590 GB/s) and their inputs are already rounded (the normed hidden state): fp16 keeps 11 bits per weight, a relative
error of 4.9e-4 per weight that averages out over 5120 terms (about 1e-5 of the output).  The selection only changes where two scores lie
closer than that to the 512th; `--check` prints how many of the 512 blocks of a random query would change.  --dtype f32 keeps the file
exact (the engine reads either).
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np

N_EMBD, IDX_DIM, IDX_HEADS, RATIO, TOP_K = 5120, 128, 4, 4, 2048
N_PARAMS = 52_434_944

GGUF_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
GGUF_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_gguf_indexers(path):
    """{layer: {name: float32 ndarray}} from a GGUF's blk.<L>.indexer.* tensors (F32 only)."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF", "not a GGUF"
    ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
    assert ver >= 2, "GGUF version %d" % ver

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def skip_value(t):
        if t == 8:
            rstr()
        elif t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            if et == 8:
                for _ in range(n):
                    rstr()
            elif et == 9:
                raise SystemExit("nested GGUF arrays are not handled")
            else:
                f.seek(GGUF_SZ[et] * n, 1)
        else:
            f.seek(GGUF_SZ[t], 1)

    align = 32
    for _ in range(nkv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        if k == "general.alignment" and t == 4:
            align = struct.unpack("<I", f.read(4))[0]
        else:
            skip_value(t)
    tensors = []
    for _ in range(nt):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        ty, off = struct.unpack("<IQ", f.read(12))
        tensors.append((name, dims, ty, off))
    base = (f.tell() + align - 1) // align * align
    out = {}
    pat = re.compile(r"^blk\.(\d+)\.indexer\.(q_proj\.weight|k_proj\.weight|q_norm\.weight|k_norm\.weight|k_norm\.pre_weight)$")
    for name, dims, ty, off in tensors:
        m = pat.match(name)
        if not m:
            continue
        assert ty == 0, "%s is not F32 (type %d)" % (name, ty)
        n = int(np.prod([int(d) for d in dims]))
        f.seek(base + off)
        a = np.frombuffer(f.read(4 * n), dtype="<f4").astype(np.float32)
        shape = tuple(int(d) for d in reversed(dims))   # ggml ne[0] is the fastest dimension
        a = a.reshape(shape)
        L = int(m.group(1))
        key = {"q_proj.weight": "q_proj", "k_proj.weight": "k_proj", "q_norm.weight": "q_norm",
               "k_norm.weight": "k_norm", "k_norm.pre_weight": "k_pre"}[m.group(2)]
        out.setdefault(L, {})[key] = a
    return out


def read_safetensors_indexers(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    suffix = {"q_proj.weight": "q_proj", "k_proj.weight": "k_proj", "q_norm": "q_norm", "k_norm": "k_pre", "k_block_norm": "k_norm"}
    for name, v in hdr.items():
        if name == "__metadata__" or "._qsa_indexer." not in name:
            continue
        pre, suf = name.split("._qsa_indexer.")
        L = int(re.search(r"layers\.(\d+)", pre).group(1))
        assert v["dtype"] == "F32", "%s: %s (only F32 is handled)" % (name, v["dtype"])
        a0, a1 = v["data_offsets"]
        f.seek(base + a0)
        a = np.frombuffer(f.read(a1 - a0), dtype="<f4").astype(np.float32).reshape(v["shape"])
        if suf.endswith("norm"):
            a = a + np.float32(1.0)   # the converter's fold: the norms are stored as w, the engine multiplies by 1 + w
        out.setdefault(L, {})[suffix[suf]] = a
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", choices=("f16", "f32"), default="f16", help="weights of q_proj / k_proj (norms are always f32)")
    ap.add_argument("--check", action="store_true", help="count the parameters and the selection change of the fp16 rounding")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("refusing to overwrite " + a.out)
    with open(a.src, "rb") as f:
        magic = f.read(4)
    idx = read_gguf_indexers(a.src) if magic == b"GGUF" else read_safetensors_indexers(a.src)
    layers = sorted(idx)
    assert len(layers) == 16 and all((L + 1) % 4 == 0 for L in layers), "expected 16 full-attention layers, got %s" % layers
    total = 0
    for L in layers:
        d = idx[L]
        assert d["q_proj"].shape == (IDX_HEADS * IDX_DIM, N_EMBD), (L, d["q_proj"].shape)
        assert d["k_proj"].shape == (IDX_DIM, N_EMBD), (L, d["k_proj"].shape)
        for k in ("q_norm", "k_pre", "k_norm"):
            assert d[k].shape == (IDX_DIM,), (L, k, d[k].shape)
            assert np.isfinite(d[k]).all()
        total += sum(int(v.size) for v in d.values())
    assert total == N_PARAMS, "parameters %d, expected %d" % (total, N_PARAMS)
    wdt = np.float16 if a.dtype == "f16" else np.float32
    index, blobs, off = {}, [], 0
    for L in layers:
        for key in ("q_proj", "k_proj", "q_norm", "k_pre", "k_norm"):
            arr = idx[L][key]
            if key.endswith("proj"):
                arr = arr.astype(wdt)
                ty = a.dtype
            else:
                ty = "f32"
            raw = np.ascontiguousarray(arr).astype("<f2" if ty == "f16" else "<f4").tobytes()
            index["%d.%s" % (L, key)] = {"off": off, "bytes": len(raw), "shape": list(arr.shape), "dtype": ty}
            blobs.append((off, raw))
            off += (len(raw) + 4095) // 4096 * 4096
    meta = {"n_embd": N_EMBD, "idx_dim": IDX_DIM, "idx_heads": IDX_HEADS, "ratio": RATIO, "top_k": TOP_K,
            "layers": layers, "wdtype": a.dtype, "norm_eps": 1e-6}
    # the blob offsets are relative to the start of the data section; the JSON is padded so that section starts on 4096
    js = json.dumps({"meta": meta, "tensors": index}, separators=(",", ":")).encode()
    head = 8 + 8 + len(js)
    data0 = (head + 4095) // 4096 * 4096
    with open(a.out, "wb") as o:
        o.write(b"QSAIDX\0\1")
        o.write(struct.pack("<Q", len(js)))
        o.write(js)
        o.write(b"\0" * (data0 - head))
        for boff, raw in blobs:
            o.seek(data0 + boff)
            o.write(raw)
        o.truncate(data0 + off)
    print("%s: %d layers %s, %d parameters, %s weights, %.1f MB" % (a.out, len(layers), layers, total, a.dtype, (data0 + off) / 1e6))
    if a.check:
        rng = np.random.default_rng(1)
        L = layers[0]
        wq, wk = idx[L]["q_proj"], idx[L]["k_proj"]
        wq16, wk16 = wq.astype(np.float16).astype(np.float32), wk.astype(np.float16).astype(np.float32)
        nb = 4096
        h = rng.standard_normal((nb * 4, N_EMBD)).astype(np.float32)   # random normed hidden rows

        def rms(x, w):
            return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * w

        def block_keys(w):
            k = rms(h @ w.T, idx[L]["k_pre"])
            return rms(k.reshape(nb, 4, IDX_DIM).mean(1), idx[L]["k_norm"])

        def scores(w_q, kb, x):
            q = rms((w_q @ x).reshape(IDX_HEADS, IDX_DIM), idx[L]["q_norm"])
            return np.maximum(kb @ q.T, 0).sum(-1) / np.sqrt(IDX_DIM)

        kb32 = block_keys(wk)
        kb16 = block_keys(wk16).astype(np.float16).astype(np.float32)   # the engine stores the pooled keys as fp16
        diffs = []
        for trial in range(4):
            x = rng.standard_normal(N_EMBD).astype(np.float32)
            s32, s16 = scores(wq, kb32, x), scores(wq16, kb16, x)
            t32 = set(np.argsort(-s32, kind="stable")[:512].tolist())
            t16 = set(np.argsort(-s16, kind="stable")[:512].tolist())
            diffs.append(512 - len(t32 & t16))
        print("check: fp16 weights + fp16 block keys against fp32, layer %d, %d random blocks, 4 random queries: %s of 512 selected "
              "blocks differ (random data has no structure; real scores are more peaked)" % (L, nb, diffs))


if __name__ == "__main__":
    main()
