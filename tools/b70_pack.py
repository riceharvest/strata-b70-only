#!/usr/bin/env python3
"""b70_pack.py - a model's matrices as a pack, the Arc Pro B70's own weight format (docs/B70_LIB.md).

    python3 tools/b70_pack.py --gguf FILE.gguf [FILE-00002-of-N.gguf ...] --out FILE.b70w4
                              [--match REGEX] [--layers N] [--search STEPS] [--check ROWS] [--list]
                              [--calib FOLDER [--holdout N]]

Any model.  Every 2-D tensor of the GGUF (all the shards given) whose name matches --match and whose shape the pack takes
(inputs a multiple of 256, rows a multiple of 16) is dequantized, quantized again to 4 bits (asymmetric, a scale and a zero
per output row and 128 inputs, the range found by a clip search) and written as a pack matrix under the GGUF's own name.
An engine then loads those matrices from the pack and everything else from the GGUF:

    Flash-Next family    strata --b70-pack FILE.b70w4 ...     (Settings: "b70_pack"; or STRATA_B70_PACK=FILE)
    dense family         strata --w4 FILE.b70w4 ...           (--b70-pack is the same flag)

--calib FOLDER quantizes with calibration (GPTQ) on the projections' own inputs, which the engine writes in a calibration
run (STRATA_B70_CALIB_DIR; docs/B70_LIB.md, "Making one"): the error of each matrix's products on held-out inputs is printed
beside the uncalibrated one.

A pack made here is a second quantization of the GGUF's values: it is as good as its source at best.  The lowest error
comes from a higher-precision GGUF (F16, BF16, Q8_0).  A checkpoint that was quantized to int4 with calibration (GPTQ,
AutoRound, AWQ) keeps its own codes instead: tools/qwen35/b70w4_pack.py packs one for the dense family.  Check a new pack
on the card before serving it (KL against the GGUF on a few prompts; docs/B70_LIB.md).

The format (the C++ reader and the kernels' view of it: sycl/include/strata/b70/pack.hpp).  A matrix is records of 16
output rows (a tile) x 256 inputs (a chunk), tile after tile, chunk after chunk:
    4 x 128 dwords   the weights: pair p of the chunk's 32-weight sub-blocks (2p in the low nibbles, 2p + 1 in the high
                     ones), dword d (4 of the 32 bytes) of row i at [128 p + 16 d + i]
    2 x 16 dwords    group g of the chunk (128 inputs), row i at [512 + 16 g + i]: the fp16 scale in the low half, the
                     zero (1 .. 16) in the high half;  weight = scale x (q - zero)
2,176 bytes a record (4.25 bits a weight).  A hardware thread of the card's stream product owns a tile and reads its
records in order with plain block loads; a sub-block's 32 nibbles are one operand of a DPAS as they are.

File: "B70W4\\0\\0\\1", u64 length of a JSON index {"tensors": {name: {off, bytes, n_in, n_out}}}, the index, the blobs at
4096-byte aligned offsets.
"""
import argparse
import json
import os
import re
import struct
import sys

import numpy as np

REC_DW = 544          # dwords a record
REC_BYTES = 4 * REC_DW
MAGIC = b"B70W4\0\0\1"
# the projections both model families of this repository reach through the shared products; an engine takes from a pack
# only the names it serves that way (the Flash-Next loader: the attention and recurrence projections)
DEFAULT_MATCH = r"\.(attn_qkv|attn_gate|ssm_out|attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down)\.weight$"


def shape_ok(n_in, n_out):
    return n_in > 0 and n_out > 0 and n_in % 256 == 0 and n_out % 16 == 0


def pack_bytes(n_in, n_out):
    return (n_out // 16) * (n_in // 256) * REC_BYTES


def pack(q, z, sc):
    """q uint8 [n_out][n_in] (0 .. 15), z [groups][n_out] (1 .. 16), sc uint16 fp16 bits [groups][n_out] -> the records,
    uint32 [tiles][chunks][544]."""
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


def unpack_records(rec, n_in, n_out):
    """The inverse of pack(): records -> (q [n_out][n_in], z [groups][n_out], sc fp16 bits [groups][n_out])."""
    T, C = n_out // 16, n_in // 256
    rec = np.asarray(rec, dtype="<u4").reshape(T, C, REC_DW)
    dw = rec[:, :, :512].reshape(T, C, 4, 8, 16).transpose(0, 4, 1, 2, 3)        # tile, row, chunk, pair, dword
    by = np.ascontiguousarray(dw).view(np.uint8).reshape(T, 16, C, 4, 32)
    q5 = np.stack([by & 15, by >> 4], axis=4)                                     # tile, row, chunk, pair, half, byte
    q = np.ascontiguousarray(q5).reshape(n_out, n_in)
    m = rec[:, :, 512:].reshape(T, C, 2, 16).transpose(0, 3, 1, 2).reshape(n_out, 2 * C).T   # [groups][n_out]
    return q, (m >> 16).astype(np.uint8), (m & 0xffff).astype(np.uint16)


def deq_rows(q, z, sc, rows):
    """Rows `rows` of the matrix as float32: scale x (q - zero)."""
    g = np.arange(q.shape[1]) // 128
    s = sc[:, rows].view(np.float16).astype(np.float32)   # [groups][rows]
    return s[g].T * (q[rows].astype(np.float32) - z[:, rows][g].T.astype(np.float32))


def _quantize_range(w, wmin, wmax):
    """w float32 [m][groups][128] to 4 bits over the range [wmin, wmax] of each group ([m][groups]; values outside it clip):
    (q int32 [m][groups][128], z int32 [m][groups] in 1 .. 16, sc_h float16 [m][groups])."""
    sc = np.maximum((wmax - wmin) / 15.0, 0.0)
    nz = sc > 0.0
    z = np.ones(wmin.shape, dtype=np.int32)
    z[nz] = np.clip(np.rint(-wmin[nz] / sc[nz]), 1, 16).astype(np.int32)
    z = np.where(wmax > 0.0, np.minimum(z, 14), z)
    z = np.where(wmin >= 0.0, 1, z)
    sc = np.where(wmin < 0.0, np.maximum(sc, -wmin / z), sc)
    sc = np.where((wmax > 0.0) & (z < 15), np.maximum(sc, wmax / np.maximum(15 - z, 1)), sc)
    # (a group of one repeated value has no range: the two lines above give it the step that puts the value on the grid.
    # Only a group of zeros keeps a zero scale.)
    sc_h = np.where(sc > 0.0, sc, 0.0).astype(np.float16)
    sc_f = sc_h.astype(np.float32)
    inv = np.where(sc_f > 0.0, 1.0 / np.where(sc_f > 0.0, sc_f, 1.0), 0.0)
    q = np.clip(np.rint(w * inv[:, :, None] + z[:, :, None]), 0, 15).astype(np.int32)
    return q, z, sc_h


def _group_params(w, search=0):
    """w float32 [m][groups][128] -> (q int32 [m][groups][128], z int32 [m][groups], sc_h float16 [m][groups]): each group's
    range is its minimum and maximum, or with search = N the best of that and N narrower ranges (down to 0.6 of it) by
    squared error."""
    wmin = w.min(axis=2)
    wmax = w.max(axis=2)
    q, z, sc_h = _quantize_range(w, wmin, wmax)
    if search > 0:
        def err(q_, z_, s_):
            d = s_.astype(np.float32)[:, :, None] * (q_ - z_[:, :, None]).astype(np.float32) - w
            return (d * d).sum(axis=2)
        best = err(q, z, sc_h)
        for p in np.linspace(1.0, 0.6, search + 1)[1:]:
            q2, z2, s2 = _quantize_range(w, wmin * np.float32(p), wmax * np.float32(p))
            e2 = err(q2, z2, s2)
            take = e2 < best
            if take.any():
                best = np.where(take, e2, best)
                q = np.where(take[:, :, None], q2, q)
                z = np.where(take, z2, z)
                sc_h = np.where(take, s2, sc_h)
    return q, z, sc_h


def quantize_rows_w4(w_f32, search=0):
    """Rows of float32 [m][n_in] (n_in a multiple of 128) to the pack's 4 bits: q int32 [m][n_in], z [groups][m], sc fp16 bits
    [groups][m].  search = 0: each group's range is its minimum and maximum.  search = N: N narrower ranges are tried too
    (down to 0.6 of the range) and each group keeps the one with the smallest squared error."""
    m, n_in = w_f32.shape
    assert n_in % 128 == 0
    ng = n_in // 128
    w = np.asarray(w_f32, dtype=np.float32).reshape(m, ng, 128)
    q, z, sc_h = _group_params(w, search)
    return q.reshape(m, n_in), np.ascontiguousarray(z.T), np.ascontiguousarray(sc_h.T.view(np.uint16))


def gptq_factor(H, damp=0.01):
    """The inputs' second moments H [n_in][n_in] (the sum of x x^T over calibration tokens) as what gptq_rows_w4 needs: the
    upper Cholesky factor U of the damped H's inverse (H^-1 = U^T U) and the inputs that were never active."""
    H = np.array(H, dtype=np.float64)
    n = H.shape[0]
    d = np.diag(H).copy()
    dead = d <= 0.0
    idx = np.where(dead)[0]
    H[idx, idx] = 1.0
    live_mean = float(d[~dead].mean()) if (~dead).any() else 1.0
    H[np.diag_indices(n)] += damp * live_mean
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    return np.ascontiguousarray(U), dead


def gptq_rows_w4(w_f32, U, dead=None, search=0):
    """quantize_rows_w4 with calibration (GPTQ): the inputs are quantized in order, and the error each one leaves is moved
    onto the inputs not yet quantized, weighted by how the calibration inputs vary together (U from gptq_factor), so the
    products with real inputs change least instead of the weights.  A group's scale and zero are taken from its weights as
    they stand when the group is reached.  Rows are independent: any subset of a matrix's rows gives those rows' result."""
    m, n_in = w_f32.shape
    assert n_in % 128 == 0 and U.shape == (n_in, n_in)
    ng = n_in // 128
    W = np.array(w_f32, dtype=np.float64)
    if dead is not None and dead.any():
        W[:, dead] = 0.0
    Q = np.zeros((m, n_in), dtype=np.int32)
    Z = np.zeros((m, ng), dtype=np.int32)
    SC = np.zeros((m, ng), dtype=np.float16)
    for g in range(ng):
        i1, i2 = 128 * g, 128 * g + 128
        W1 = W[:, i1:i2].copy()
        _, z, sc_h = _group_params(W1.astype(np.float32).reshape(m, 1, 128), search)
        z = z[:, 0].astype(np.float64)
        scf = sc_h[:, 0].astype(np.float64)
        inv = np.where(scf > 0.0, 1.0 / np.where(scf > 0.0, scf, 1.0), 0.0)
        U1 = U[i1:i2, i1:i2]
        Err = np.zeros_like(W1)
        for i in range(128):
            w = W1[:, i]
            qi = np.clip(np.rint(w * inv + z), 0, 15)
            e = (w - scf * (qi - z)) / U1[i, i]
            Q[:, i1 + i] = qi
            Err[:, i] = e
            if i < 127:
                W1[:, i + 1:] -= np.outer(e, U1[i, i + 1:])
        if i2 < n_in:
            W[:, i2:] -= Err @ U[i1:i2, i2:]
        Z[:, g] = z
        SC[:, g] = sc_h[:, 0]
    return Q, np.ascontiguousarray(Z.T), np.ascontiguousarray(SC.T.view(np.uint16))


def read_header(f):
    """(the index's JSON, the file offset of the blob area) of an open pack."""
    magic = f.read(8)
    if magic != MAGIC:
        raise SystemExit(f"bad B70W4 magic: {magic!r}")
    (n,) = struct.unpack("<Q", f.read(8))
    meta = json.loads(f.read(n).decode())
    hdr_end = 16 + n
    hdr_end += (-hdr_end) % 4096
    return meta, hdr_end


def write_pack(out, blobs_path, index, source="", extra=None):
    """The pack file: the header with `index` ({name: {off, bytes, n_in, n_out[, layout]}}), then the blob area (the file
    blobs_path, whose offsets the index names; it is removed)."""
    meta = {"format": "B70W4", "version": 1, "record_bytes": REC_BYTES, "source": source}
    if extra:
        meta.update(extra)
    meta["tensors"] = index            # last: the C++ reader takes the tensors object to be the end of the index
    js = json.dumps(meta).encode()
    head = MAGIC + struct.pack("<Q", len(js)) + js
    head += b"\0" * ((-len(head)) % 4096)
    with open(out, "wb") as fo, open(blobs_path, "rb") as fb:
        fo.write(head)
        while True:
            buf = fb.read(64 << 20)
            if not buf:
                break
            fo.write(buf)
    os.remove(blobs_path)
    return len(head)


def pack_float_matrix(rows_of, n_in, n_out, fb, off, search=4, check=48, rng=None, chunk_rows=1024, quantizer=None,
                      h_eval=None, out_err=None):
    """A float matrix as a pack matrix appended to the blob file fb at offset `off` (padded to 4096 first).
    rows_of(r0, r1) -> float32 [r1 - r0][n_in].  quantizer(rows) -> (q, z, sc) replaces the plain one (quantize_rows_w4 with
    `search`).  Returns (the index entry, the next offset, the relative RMS error of the weights, the lowest row correlation
    with the source over `check` sampled rows).  With h_eval (the second moments of inputs the quantizer has not seen) and a
    dict out_err, out_err gets "plain" and "used": the relative RMS error of the matrix's products with those inputs, for the
    plain quantizer and for the one used."""
    if not shape_ok(n_in, n_out):
        raise SystemExit(f"a matrix of {n_in} x {n_out}: rows must be a multiple of 16, inputs of 256")
    rng = rng or np.random.default_rng(7)
    pad = (-off) % 4096
    if pad:
        fb.write(b"\0" * pad)
        off += pad
    start = off
    chunk_rows = max(16, chunk_rows // 16 * 16)
    se = ss = 0.0
    worst = 1.0
    oe = {"plain": 0.0, "used": 0.0, "ref": 0.0}
    for r0 in range(0, n_out, chunk_rows):
        r1 = min(n_out, r0 + chunk_rows)
        ref = np.asarray(rows_of(r0, r1), dtype=np.float32).reshape(r1 - r0, n_in)
        q, z, sc = quantizer(ref) if quantizer else quantize_rows_w4(ref, search)
        qb = q.astype(np.uint8)
        w = deq_rows(qb, z, sc, np.arange(r1 - r0))
        if h_eval is not None and out_err is not None:
            r64 = ref.astype(np.float64)
            oe["ref"] += float(((r64 @ h_eval) * r64).sum())
            d = w.astype(np.float64) - r64
            oe["used"] += float(((d @ h_eval) * d).sum())
            if quantizer:
                q0, z0, s0 = quantize_rows_w4(ref, search)
                d = deq_rows(q0.astype(np.uint8), z0, s0, np.arange(r1 - r0)).astype(np.float64) - r64
            oe["plain"] += float(((d @ h_eval) * d).sum())
        se += float(((w - ref).astype(np.float64) ** 2).sum())
        ss += float((ref.astype(np.float64) ** 2).sum())
        if check > 0:
            pick = np.sort(rng.choice(r1 - r0, size=min(max(1, check * (r1 - r0) // n_out), r1 - r0), replace=False))
            rc = ref[pick] - ref[pick].mean(1, keepdims=True)
            wc = w[pick] - w[pick].mean(1, keepdims=True)
            c = (wc * rc).sum(1) / np.sqrt((wc ** 2).sum(1) * (rc ** 2).sum(1) + 1e-30)
            c = np.where((rc ** 2).sum(1) > 0.0, c, 1.0)   # (a row of one repeated value has no correlation to report)
            worst = min(worst, float(c.min()))
        blob = pack(qb, z, sc)
        fb.write(blob.tobytes())
        off += int(blob.nbytes)
    if out_err is not None and oe["ref"] > 0.0:
        out_err["plain"] = (max(oe["plain"], 0.0) / oe["ref"]) ** 0.5
        out_err["used"] = (max(oe["used"], 0.0) / oe["ref"]) ** 0.5
    ent = {"off": start, "bytes": off - start, "n_in": int(n_in), "n_out": int(n_out)}
    return ent, off, (se / ss) ** 0.5 if ss > 0 else 0.0, worst


class Calibration:
    """A folder the engine wrote in a calibration run (STRATA_B70_CALIB_DIR, docs/B70_LIB.md): <tensor name>.x16 (the FP16
    inputs of that projection, rows of n_in), aliases.txt ("name source": a projection that read another one's input),
    calls.txt ("name rows n_in" per write, in order).  The last `holdout` writes of each file are kept apart to measure on."""

    def __init__(self, folder, holdout=1):
        self.folder, self.holdout = folder, holdout
        self.alias, self.calls = {}, {}
        ap = os.path.join(folder, "aliases.txt")
        if os.path.exists(ap):
            for line in open(ap):
                f = line.split()
                if len(f) == 2:
                    self.alias[f[0]] = f[1]
        cp = os.path.join(folder, "calls.txt")
        if os.path.exists(cp):
            for line in open(cp):
                f = line.split()
                if len(f) == 3:
                    self.calls.setdefault(f[0], []).append(int(f[1]))
        self._key, self._val = None, None

    def source(self, name):
        src = self.alias.get(name, name)
        return src if os.path.exists(os.path.join(self.folder, src + ".x16")) else None

    def moments(self, name, n_in):
        """(H of the calibration rows, H of the held-out rows or None, their row counts) for a tensor, or None."""
        src = self.source(name)
        if src is None:
            return None
        if self._key != (src, n_in):
            x = np.memmap(os.path.join(self.folder, src + ".x16"), dtype="<f2", mode="r")
            if x.size % n_in:
                raise SystemExit(f"{src}.x16: {x.size} values are not rows of {n_in}")
            x = x.reshape(-1, n_in)
            calls = self.calls.get(src, [])
            n_val = sum(calls[-self.holdout:]) if self.holdout > 0 and len(calls) > self.holdout else 0
            n_cal = x.shape[0] - n_val

            def second_moments(r0, r1):
                h = np.zeros((n_in, n_in), dtype=np.float64)
                for a in range(r0, r1, 4096):
                    c = np.asarray(x[a:min(r1, a + 4096)], dtype=np.float32).astype(np.float64)
                    h += c.T @ c
                return h

            h_cal = second_moments(0, n_cal)
            h_val = second_moments(n_cal, x.shape[0]) if n_val > 0 else None
            u, dead = gptq_factor(h_cal)
            self._key, self._val = (src, n_in), (u, dead, h_val, n_cal, n_val)
        return self._val


def _dequantizer(type_id, type_name):
    """rows(raw uint8 [m][row bytes], n_in) -> float32 [m][n_in] for a GGUF tensor type, or None when nothing here can read
    it.  The float types are read as they are; the quantized ones go through gguf-py's dequantize (llama.cpp's package: an
    installed one, or the copy $STRATA_GGUF_PY / setup.py's clone holds, which knows the newer types)."""
    if type_name == "F32":
        return lambda raw, n_in: np.ascontiguousarray(raw).view("<f4").reshape(-1, n_in)
    if type_name == "F16":
        return lambda raw, n_in: np.ascontiguousarray(raw).view("<f2").reshape(-1, n_in).astype(np.float32)
    if type_name == "BF16":
        return lambda raw, n_in: (np.ascontiguousarray(raw).view("<u2").astype(np.uint32) << 16).view("<f4").reshape(-1, n_in)
    try:
        try:
            from _paths import add_gguf_py
            add_gguf_py()
        except (ImportError, SystemExit):
            pass                              # an installed gguf package serves as well
        import gguf
        from gguf.quants import dequantize
        qt = gguf.GGMLQuantizationType(type_id)
    except (ImportError, ValueError):
        return None
    return lambda raw, n_in: dequantize(np.ascontiguousarray(raw), qt).reshape(-1, n_in).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description="a model's matrices as a pack, the Arc Pro B70's own weight format")
    ap.add_argument("--gguf", nargs="+", required=True, help="the model's GGUF (every shard of a split file)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default=DEFAULT_MATCH, help="regular expression a tensor's name must match (search)")
    ap.add_argument("--layers", type=int, default=-1, help="only blk.0 .. blk.N-1 (a test)")
    ap.add_argument("--search", type=int, default=4, help="narrower ranges tried per group of 128 inputs (0: minimum and maximum)")
    ap.add_argument("--check", type=int, default=48, help="rows of each matrix whose correlation with the source is reported (0: none)")
    ap.add_argument("--list", action="store_true", help="print what would be packed and what is left out, write nothing")
    ap.add_argument("--calib", default=None, help="a calibration folder the engine wrote (STRATA_B70_CALIB_DIR): quantize with GPTQ")
    ap.add_argument("--holdout", type=int, default=1, help="with --calib: the last N writes of each input are held out and measured on")
    a = ap.parse_args()
    calib = Calibration(a.calib, a.holdout) if a.calib else None
    worst_plain, worst_used, n_calibrated = 0.0, 0.0, 0
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gguf_reader import GGUFFile          # this repository's reader: the directory of any GGUF, whatever its tensor types
    pat = re.compile(a.match)
    index, off = {}, 0
    worst_err, worst_corr, src_bytes = 0.0, 1.0, 0
    rng = np.random.default_rng(7)
    tmp = a.out + ".blobs"
    fb = None if a.list else open(tmp, "wb")
    try:
        for path in a.gguf:
            g = GGUFFile(path)
            for t in g.tensors:
                name = t.name
                if not pat.search(name):
                    continue
                if a.layers > 0 and name.startswith("blk."):
                    if int(name.split(".")[1]) >= a.layers:
                        continue
                shape = [int(x) for x in t.shape]
                nbytes = t.expected_bytes()
                if len(shape) != 2 or not shape_ok(shape[0], shape[1]):
                    print(f"left out: {name} {shape} ({t.type_name}): not a matrix of inputs % 256 == 0 and rows % 16 == 0")
                    continue
                n_in, n_out = shape
                deq = _dequantizer(t.type_id, t.type_name)
                if deq is None or nbytes is None or nbytes % n_out:
                    print(f"left out: {name} {shape} ({t.type_name}): no reader here for this type (a newer gguf-py: STRATA_GGUF_PY)")
                    continue
                if name in index:
                    raise SystemExit(f"{name}: in more than one shard")
                if a.list:
                    index[name] = {"n_in": n_in, "n_out": n_out}
                    print(f"pack: {name} {n_in} x {n_out} ({t.type_name}) -> {pack_bytes(n_in, n_out) / 2 ** 20:.2f} MiB")
                    continue
                raw = np.memmap(path, dtype=np.uint8, mode="r", offset=g.data_start + t.offset, shape=(n_out, nbytes // n_out))

                def rows_of(r0, r1, raw=raw, deq=deq, n_in=n_in):
                    return deq(raw[r0:r1], n_in)

                cal = calib.moments(name, n_in) if calib else None
                note = ""
                if cal is not None:
                    u, dead, h_val, n_cal, n_val = cal
                    oe = {}
                    ent, off, rel, corr = pack_float_matrix(
                        rows_of, n_in, n_out, fb, off, a.search, a.check, rng,
                        quantizer=lambda ref, u=u, dead=dead: gptq_rows_w4(ref, u, dead, a.search), h_eval=h_val, out_err=oe)
                    n_calibrated += 1
                    note = ", calibrated on %d tokens" % n_cal
                    if oe:
                        worst_plain, worst_used = max(worst_plain, oe["plain"]), max(worst_used, oe["used"])
                        note += "; error of its products on %d held-out tokens %.4f (uncalibrated %.4f)" % (n_val, oe["used"], oe["plain"])
                else:
                    if calib:
                        note = ", NOT calibrated (no inputs for it in the calibration folder)"
                    ent, off, rel, corr = pack_float_matrix(rows_of, n_in, n_out, fb, off, a.search, a.check, rng)
                index[name] = ent
                src_bytes += nbytes
                worst_err, worst_corr = max(worst_err, rel), min(worst_corr, corr)
                print("%s: %d x %d (%s) -> %.2f MiB, relative error %.4f, lowest row correlation %.4f%s"
                      % (name, n_in, n_out, t.type_name, ent["bytes"] / 2 ** 20, rel, corr, note), flush=True)
    finally:
        if fb:
            fb.close()
    if a.list:
        print(f"{len(index)} matrices would be packed")
        return 0
    if not index:
        os.remove(tmp)
        raise SystemExit("no tensor matched: nothing written (--list shows what the GGUF holds for a --match)")
    head = write_pack(a.out, tmp, index, source=os.path.basename(a.gguf[0]),
                      extra={"made_by": "tools/b70_pack.py",
                             "quantizer": (f"gptq-asym-g128-search{a.search}" if n_calibrated else f"rtn-asym-g128-search{a.search}"),
                             "calibrated": n_calibrated})
    print("%s: %d matrices, %.2f GiB (their GGUF blocks: %.2f GiB), blobs from byte %d, highest relative error %.4f, lowest row "
          "correlation %.4f" % (a.out, len(index), (head + off) / 2 ** 30, src_bytes / 2 ** 30, head, worst_err, worst_corr))
    if calib:
        print("calibrated %d of %d matrices; highest error of the products on held-out inputs %.4f (uncalibrated %.4f)"
              % (n_calibrated, len(index), worst_used, worst_plain))
    return 0


if __name__ == "__main__":
    sys.exit(main())
