#!/usr/bin/env python3
"""b70w4_pack.py - a GPTQ int4 checkpoint of Qwen3.8-27B as a B70W4 pack for strata-q36 (docs/Q36.md).

    python3 tools/qwen35/b70w4_pack.py --ckpt DIR --gguf FILE.gguf --out FILE.b70w4 [--layers N] [--check ROWS]

The checkpoint: GPTQ v1, 4 bits, asymmetric, groups of 128 inputs, no act-order (qweight int32 [n_in / 8][n_out], eight
4-bit values an int32 along the input; qzeros int32 [groups][n_out / 8]; scales fp16 [groups][n_out]; a weight is
scale x (q - (zero + 1))).  The GGUF of the same model names the tensors and gives the engine everything else (norms,
embeddings, the head, the MTP block, the small GDN tensors).  Each packed matrix is checked against the GGUF's tensor of
that name on a sample of rows (the two are the same weights quantized twice: their rows correlate above 0.9).

The pack: every matrix as records of 16 output rows (a tile) x 256 inputs (a chunk), tile after tile, chunk after chunk:
    4 x 128 dwords   the weights: pair p of the chunk's 32-weight sub-blocks (2p in the low nibbles, 2p + 1 in the high
                     ones), dword d (4 of the 32 bytes) of row i at [128 p + 16 d + i]
    2 x 16 dwords    group g of the chunk (128 inputs), row i at [512 + 16 g + i]: the fp16 scale in the low half, the
                     zero (1 .. 16) in the high half
2,176 bytes a record (4.25 bits a weight).  A hardware thread of the XMX rows form owns a tile and reads its records in
order with plain block loads; a sub-block's 32 nibbles are one operand of an s8 DPAS as they are.

File: "B70W4\\0\\0\\1", u64 length of a JSON index {name: {off, bytes, n_in, n_out}}, the index, the blobs at 4096-byte
aligned offsets.  The GGUF conversion regroups the 48 GDN heads (GGUF head g is checkpoint head 3 (g % 16) + g // 16):
the rows of in_proj_z, the last 6144 rows of in_proj_qkv and the columns of out_proj move with it.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

REC_DW = 544   # dwords a record


def st_open(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def st_get(path, hdr, base, name):
    t = hdr[name]
    a, b = t["data_offsets"]
    dt = {"F16": np.uint16, "I32": np.uint32}[t["dtype"]]
    return np.fromfile(path, dtype=dt, count=(b - a) // np.dtype(dt).itemsize, offset=base + a).reshape(t["shape"])


def head_perm():
    g = np.arange(48)
    return 3 * (g % 16) + g // 16   # GGUF head -> checkpoint head


def block_perm(n, block, heads):
    return (heads[:, None] * block + np.arange(block)[None, :]).reshape(-1) if n == len(heads) * block else None


def unpack(path, hdr, base, key):
    qw = st_get(path, hdr, base, key + ".qweight")
    qz = st_get(path, hdr, base, key + ".qzeros")
    sc = st_get(path, hdr, base, key + ".scales")
    n_in, n_out, groups = qw.shape[0] * 8, qw.shape[1], sc.shape[0]
    sh = (4 * np.arange(8, dtype=np.uint32))
    q = ((qw[:, None, :] >> sh[None, :, None]) & 15).astype(np.uint8).reshape(n_in, n_out)
    z = (((qz[:, :, None] >> sh[None, None, :]) & 15).astype(np.uint8).reshape(groups, n_out) + 1)
    if n_in != groups * 128:
        raise SystemExit(f"{key}: {n_in} inputs in {groups} groups: not groups of 128")
    return np.ascontiguousarray(q.T), z, sc   # q [n_out][n_in]; z, sc [groups][n_out]


def pack(q, z, sc):
    n_out, n_in = q.shape
    if n_out % 16 or n_in % 256:
        raise SystemExit(f"a matrix of {n_in} x {n_out}: rows must be a multiple of 16, inputs of 256")
    T, C = n_out // 16, n_in // 256
    q5 = q.reshape(T, 16, C, 4, 2, 32)                       # tile, row, chunk, pair, half (low / high nibble), byte
    by = (q5[:, :, :, :, 0, :] | (q5[:, :, :, :, 1, :] << 4)).astype(np.uint8)
    dw = np.ascontiguousarray(by).view("<u4")                # tile, row, chunk, pair, dword (8)
    qs = np.ascontiguousarray(dw.transpose(0, 2, 3, 4, 1)).reshape(T, C, 512)
    m = sc.astype(np.uint32) | (z.astype(np.uint32) << 16)   # [groups][n_out]
    m = np.ascontiguousarray(m.T.reshape(T, 16, C, 2).transpose(0, 2, 3, 1)).reshape(T, C, 32)
    return np.concatenate([qs, m], axis=2).astype("<u4")     # [tile][chunk][544]


def deq_rows(q, z, sc, rows):
    g = np.arange(q.shape[1]) // 128
    s = sc[:, rows].view(np.float16).astype(np.float32)   # [groups][rows]
    return s[g].T * (q[rows].astype(np.float32) - z[:, rows][g].T.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=-1, help="only the first N layers (a test)")
    ap.add_argument("--check", type=int, default=48, help="rows of each matrix compared with the GGUF's tensor (0: none)")
    a = ap.parse_args()
    import gguf
    from gguf.quants import dequantize
    rd = gguf.GGUFReader(a.gguf)
    gt = {t.name: t for t in rd.tensors}
    cfg = json.load(open(os.path.join(a.ckpt, "config.json")))
    qc = cfg.get("quantization_config", {})
    if qc.get("bits") != 4 or qc.get("group_size") != 128 or qc.get("sym") or qc.get("desc_act"):
        raise SystemExit("the checkpoint is not GPTQ 4-bit asymmetric with groups of 128 and no act-order")
    n_layers = len([f for f in os.listdir(a.ckpt) if f.startswith("model-layer")])
    if a.layers > 0:
        n_layers = min(n_layers, a.layers)
    heads = head_perm()
    names = [("linear_attn.in_proj_qkv", "attn_qkv"), ("linear_attn.in_proj_z", "attn_gate"), ("linear_attn.out_proj", "ssm_out"),
             ("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
             ("self_attn.o_proj", "attn_output"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"), ("mlp.down_proj", "ffn_down")]
    index, off, worst = {}, 0, 1.0
    rng = np.random.default_rng(7)
    tmp = a.out + ".blobs"
    with open(tmp, "wb") as fb:
        for layer in range(n_layers):
            path = os.path.join(a.ckpt, "model-layer%02d.safetensors" % layer)
            hdr, base = st_open(path)
            pre = next(k for k in hdr if ".layers.%d." % layer in k).split(".layers.%d." % layer)[0] + ".layers.%d." % layer
            for ck, gn in names:
                if pre + ck + ".qweight" not in hdr:
                    continue
                name = "blk.%d.%s.weight" % (layer, gn)
                if name not in gt:
                    raise SystemExit(f"{name}: not in the GGUF")
                q, z, sc = unpack(path, hdr, base, pre + ck)
                n_out, n_in = q.shape
                if ck == "linear_attn.in_proj_z":
                    p = block_perm(n_out, 128, heads)
                    q, z, sc = q[p], z[:, p], sc[:, p]
                elif ck == "linear_attn.in_proj_qkv":
                    v0 = n_out - 48 * 128
                    p = np.concatenate([np.arange(v0), v0 + block_perm(48 * 128, 128, heads)])
                    q, z, sc = q[p], z[:, p], sc[:, p]
                elif ck == "linear_attn.out_proj":
                    p = block_perm(n_in, 128, heads)
                    q, z, sc = np.ascontiguousarray(q[:, p]), z[heads], sc[heads]
                gs = [int(x) for x in gt[name].shape]
                if gs != [n_in, n_out]:
                    raise SystemExit(f"{name}: the GGUF has {gs}, the checkpoint {n_in} x {n_out}")
                corr = float("nan")
                if a.check > 0:
                    rows = np.sort(rng.choice(n_out, size=min(a.check, n_out), replace=False))
                    ref = dequantize(np.asarray(gt[name].data).reshape(n_out, -1)[rows], gt[name].tensor_type).reshape(len(rows), n_in)
                    w = deq_rows(q, z, sc, rows)
                    wc, rc = w - w.mean(1, keepdims=True), ref - ref.mean(1, keepdims=True)
                    c = (wc * rc).sum(1) / np.sqrt((wc ** 2).sum(1) * (rc ** 2).sum(1) + 1e-30)
                    corr = float(c.min())
                    worst = min(worst, corr)
                    if corr < 0.8:
                        raise SystemExit(f"{name}: a row correlates {corr:.3f} with the GGUF's tensor: not the same matrix")
                blob = pack(q, z, sc)
                pad = (-off) % 4096
                fb.write(b"\0" * pad)
                off += pad
                fb.write(blob.tobytes())
                index[name] = {"off": off, "bytes": int(blob.nbytes), "n_in": n_in, "n_out": n_out}
                off += blob.nbytes
            print("layer %2d: %d matrices so far, %.2f GiB, lowest row correlation with the GGUF %.3f" % (layer, len(index), off / 2 ** 30, worst), flush=True)
    js = json.dumps({"format": "B70W4", "version": 1, "record_bytes": 4 * REC_DW, "source": os.path.basename(os.path.normpath(a.ckpt)),
                     "tensors": index}).encode()
    head = b"B70W4\0\0\1" + struct.pack("<Q", len(js)) + js
    head += b"\0" * ((-len(head)) % 4096)
    with open(a.out, "wb") as fo, open(tmp, "rb") as fb:
        fo.write(head)
        while True:
            buf = fb.read(64 << 20)
            if not buf:
                break
            fo.write(buf)
    os.remove(tmp)
    print("%s: %d matrices, %.2f GiB, blobs from byte %d, lowest row correlation %.3f" % (a.out, len(index), (len(head) + off) / 2 ** 30, len(head), worst))


if __name__ == "__main__":
    sys.exit(main())
