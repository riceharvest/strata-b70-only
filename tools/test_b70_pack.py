"""The pack (the B70's own weight format): tools/b70_pack.py against the layout sycl/include/strata/b70/pack.hpp documents,
and the C++ reader against a file the tool wrote.  No GPU.

    python3 -m pytest tools/test_b70_pack.py -q
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import b70_pack as bp   # noqa: E402


def random_matrix(rng, n_out, n_in):
    w = (rng.standard_normal((n_out, n_in)) * rng.uniform(0.01, 2.0, (n_out, 1))).astype(np.float32)
    w[0, :128] = 0.0                       # a group of zeros
    w[1, :128] = np.abs(w[1, :128])        # one sign only
    w[2, 128:256] = -np.abs(w[2, 128:256])
    w[3, :] = 0.5                          # a constant row
    return w


class Records(unittest.TestCase):
    def test_layout_is_the_documented_one(self):
        """pack() against the record written value by value from the header's description."""
        rng = np.random.default_rng(3)
        n_out, n_in = 32, 512
        q = rng.integers(0, 16, (n_out, n_in), dtype=np.uint8)
        z = rng.integers(1, 17, (n_in // 128, n_out), dtype=np.uint8)
        sc = rng.integers(0, 0x7c00, (n_in // 128, n_out), dtype=np.uint16)
        rec = bp.pack(q, z, sc)
        self.assertEqual(rec.shape, (n_out // 16, n_in // 256, bp.REC_DW))
        self.assertEqual(rec.nbytes, bp.pack_bytes(n_in, n_out))
        for t in range(n_out // 16):
            for c in range(n_in // 256):
                want = np.zeros(bp.REC_DW, dtype=np.uint32)
                for p in range(4):                      # pair p: sub-blocks 2p (low nibbles) and 2p + 1 (high)
                    for i in range(16):                 # row i of the tile
                        lo = q[16 * t + i, 256 * c + 64 * p: 256 * c + 64 * p + 32].astype(np.uint32)
                        hi = q[16 * t + i, 256 * c + 64 * p + 32: 256 * c + 64 * p + 64].astype(np.uint32)
                        by = lo | (hi << 4)
                        for d in range(8):              # dword d: 4 of the sub-block's 32 bytes
                            b = by[4 * d: 4 * d + 4]
                            want[128 * p + 16 * d + i] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)
                for g in range(2):                      # group g of the chunk: scale | zero << 16
                    for i in range(16):
                        want[512 + 16 * g + i] = int(sc[2 * c + g, 16 * t + i]) | (int(z[2 * c + g, 16 * t + i]) << 16)
                np.testing.assert_array_equal(rec[t, c], want)

    def test_round_trip(self):
        rng = np.random.default_rng(4)
        for n_out, n_in in ((16, 256), (48, 1280), (64, 768)):
            q = rng.integers(0, 16, (n_out, n_in), dtype=np.uint8)
            z = rng.integers(1, 17, (n_in // 128, n_out), dtype=np.uint8)
            sc = rng.integers(0, 0x7c00, (n_in // 128, n_out), dtype=np.uint16)
            q2, z2, s2 = bp.unpack_records(bp.pack(q, z, sc), n_in, n_out)
            np.testing.assert_array_equal(q2, q)
            np.testing.assert_array_equal(z2, z)
            np.testing.assert_array_equal(s2, sc)

    def test_shapes_refused(self):
        self.assertTrue(bp.shape_ok(5120, 17408))
        self.assertFalse(bp.shape_ok(5000, 16))
        self.assertFalse(bp.shape_ok(256, 24))
        with self.assertRaises(SystemExit):
            bp.pack(np.zeros((24, 256), np.uint8), np.ones((2, 24), np.uint8), np.zeros((2, 24), np.uint16))


class Quantizer(unittest.TestCase):
    @staticmethod
    def constant(value):
        q, z, sc = bp.quantize_rows_w4(np.full((16, 256), value, np.float32), 0)
        return q.astype(np.uint8), z, sc

    def test_values_and_error(self):
        rng = np.random.default_rng(5)
        w = random_matrix(rng, 48, 1024)
        rows = np.arange(48)
        errs = []
        for search in (0, 4, 8):
            q, z, sc = bp.quantize_rows_w4(w, search)
            self.assertTrue(((q >= 0) & (q <= 15)).all())
            self.assertTrue(((z >= 1) & (z <= 16)).all())
            d = bp.deq_rows(q.astype(np.uint8), z, sc, rows)
            np.testing.assert_array_equal(d[0, :128], 0.0)                 # zeros stay zeros
            np.testing.assert_allclose(d[3], 0.5, rtol=2e-3)              # a repeated value is on the grid (it was written as zeros once)
            np.testing.assert_allclose(bp.deq_rows(*self.constant(-0.25), np.arange(16)), -0.25, rtol=2e-3)
            errs.append(float(((d - w) ** 2).sum()))
            rel = (errs[-1] / float((w ** 2).sum())) ** 0.5
            self.assertLess(rel, 0.2, f"search {search}: 4 bits over a group of 128 Gaussian values")
        self.assertLessEqual(errs[1], errs[0])                             # a search never loses to minimum / maximum
        self.assertLessEqual(errs[2], errs[1] * 1.0001)

    def test_without_search_the_range_is_kept(self):
        """search 0: no value clips, so the largest error is half a step."""
        rng = np.random.default_rng(6)
        w = random_matrix(rng, 16, 512)
        q, z, sc = bp.quantize_rows_w4(w, 0)
        d = bp.deq_rows(q.astype(np.uint8), z, sc, np.arange(16))
        step = sc.view(np.float16).astype(np.float32)[np.arange(512) // 128].T
        self.assertTrue((np.abs(d - w) <= 0.51 * step + 1e-7).all())   # (half a step, and the scale's FP16 rounding)


class File(unittest.TestCase):
    def write(self, folder, index_extra=None):
        rng = np.random.default_rng(8)
        mats = {"blk.0.attn_q.weight": (256, 32), "blk.11.ffn_down.weight": (768, 16), "output.weight": (512, 48)}
        index, off, blobs = {}, 0, {}
        tmp = folder / "t.blobs"
        with open(tmp, "wb") as fb:
            for name, (n_in, n_out) in mats.items():
                w = random_matrix(rng, n_out, n_in)
                ent, off, rel, corr = bp.pack_float_matrix(lambda r0, r1, w=w: w[r0:r1], n_in, n_out, fb, off, 4, 8, rng)
                self.assertLess(rel, 0.2)
                self.assertGreater(corr, 0.97)
                index[name] = ent
                q, z, sc = bp.quantize_rows_w4(w, 4)
                blobs[name] = bp.pack(q.astype(np.uint8), z, sc).tobytes()
        if index_extra:
            index.update(index_extra)
        out = folder / "t.b70w4"
        head = bp.write_pack(str(out), str(tmp), index, source="unit", extra={"made_by": "test"})
        self.assertFalse(tmp.exists())
        return out, index, blobs, head

    def test_header_and_blobs(self):
        with tempfile.TemporaryDirectory() as d:
            out, index, blobs, head = self.write(Path(d))
            with open(out, "rb") as f:
                meta, base = bp.read_header(f)
                self.assertEqual(base, head)
                self.assertEqual(base % 4096, 0)
                self.assertEqual(meta["format"], "B70W4")
                self.assertEqual(meta["record_bytes"], 2176)
                self.assertEqual(list(meta)[-1], "tensors")       # the C++ reader ends its read with this object
                for name, ent in meta["tensors"].items():
                    self.assertEqual(ent["off"] % 4096, 0)
                    self.assertEqual(ent["bytes"], bp.pack_bytes(ent["n_in"], ent["n_out"]))
                    f.seek(base + ent["off"])
                    self.assertEqual(f.read(ent["bytes"]), blobs[name])

    @unittest.skipUnless(shutil.which("g++"), "no g++")
    def test_cpp_reader(self):
        """sycl/include/strata/b70/pack.hpp opens the tool's file: the same index, the same bytes, find() by shape."""
        prog = r'''
#include "strata/b70/pack.hpp"
#include <cstdio>
#include <vector>
int main(int argc, char** argv) {
    strata::b70::pack::File f;
    std::string err;
    if (!f.open(argv[1], err)) { std::printf("ERR %s\n", err.c_str()); return 1; }
    std::printf("blobs %llu n %zu\n", (unsigned long long) f.blobs, f.t.size());
    for (const auto& [name, e] : f.t) {
        std::vector<unsigned char> b((size_t) e.bytes);
        if (!f.read(e, b.data(), err)) { std::printf("ERR %s\n", err.c_str()); return 1; }
        unsigned long long h = 1469598103934665603ull;
        for (unsigned char c : b) h = (h ^ c) * 1099511628211ull;
        std::printf("T %s %llu %llu %d %d [%s] %llu %d %d\n", name.c_str(), (unsigned long long) e.off, (unsigned long long) e.bytes,
                    e.n_in, e.n_out, e.layout.c_str(), h, f.find(name, e.n_in, e.n_out) != nullptr,
                    f.find(name, e.n_in, e.n_out + 16) != nullptr);
    }
    std::printf("K %d %zu %d %d\n", strata::b70::kTypeW4, strata::b70::pack::bytes(5120, 17408),
                (int) strata::b70::pack::shape_ok(5120, 17408), (int) strata::b70::pack::shape_ok(5000, 16));
    return 0;
}
'''
        def fnv(b):
            h = 1469598103934665603
            for c in b:
                h = ((h ^ c) * 1099511628211) & 0xffffffffffffffff
            return h

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            first = {"off": 0, "bytes": 2176, "n_in": 256, "n_out": 16, "layout": "gu_merged"}
            out, index, blobs, head = self.write(d, {"blk.0.ffn_gu.merged": first})
            (d / "r.cpp").write_text(prog)
            subprocess.run(["g++", "-std=c++20", "-O1", "-I", str(ROOT / "sycl/include"), str(d / "r.cpp"), "-o", str(d / "r")], check=True)
            res = subprocess.run([str(d / "r"), str(out)], capture_output=True, text=True, check=True).stdout.splitlines()
            self.assertEqual(res[0], f"blobs {head} n {len(index)}")
            seen = {}
            for line in res[1:]:
                if line.startswith("T "):
                    _, name, off, nb, n_in, n_out, layout, h, found, found_other = line.split(" ")
                    seen[name] = (int(off), int(nb), int(n_in), int(n_out), layout, int(h), found, found_other)
            self.assertEqual(set(seen), set(index))
            for name, ent in index.items():
                off, nb, n_in, n_out, layout, h, found, found_other = seen[name]
                self.assertEqual((off, nb, n_in, n_out), (ent["off"], ent["bytes"], ent["n_in"], ent["n_out"]))
                self.assertEqual(layout, "[" + ent.get("layout", "") + "]")
                self.assertEqual(found, "0" if "layout" in ent else "1")     # a derived blob is not a plain matrix
                self.assertEqual(found_other, "0")                           # another shape: not this matrix
                if name in blobs:
                    self.assertEqual(h, fnv(blobs[name]))
            self.assertEqual(res[-1], f"K 1004 {bp.pack_bytes(5120, 17408)} 1 0")

    @unittest.skipUnless(shutil.which("g++"), "no g++")
    def test_cpp_reader_refuses_other_files(self):
        prog = r'''
#include "strata/b70/pack.hpp"
#include <cstdio>
int main(int argc, char** argv) {
    strata::b70::pack::File f;
    std::string err;
    const bool ok = f.open(argv[1], err);
    std::printf("%d %zu\n", (int) ok, f.t.size());
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "r.cpp").write_text(prog)
            subprocess.run(["g++", "-std=c++20", "-I", str(ROOT / "sycl/include"), str(d / "r.cpp"), "-o", str(d / "r")], check=True)
            (d / "a").write_bytes(b"GGUF" + b"\0" * 64)
            js = json.dumps({"format": "B70W4", "tensors": {}}).encode()
            (d / "b").write_bytes(bp.MAGIC + len(js).to_bytes(8, "little") + js)
            for name in ("a", "b", "missing"):
                out = subprocess.run([str(d / "r"), str(d / name)], capture_output=True, text=True, check=True).stdout
                self.assertEqual(out.strip(), "0 0", name)


class Calibrated(unittest.TestCase):
    """GPTQ: the same format, the error moved to where the calibration inputs do not look."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        cls.n_in, cls.n_out, r = 512, 96, 48
        basis = rng.standard_normal((r, cls.n_in))

        def inputs(n):
            return (rng.standard_normal((n, r)) * np.linspace(3, 0.2, r)) @ basis + 0.15 * rng.standard_normal((n, cls.n_in))

        cls.xc, cls.xv = inputs(4000), inputs(1500)
        cls.w = (rng.standard_normal((cls.n_out, cls.n_in)) * rng.uniform(0.02, 1, (cls.n_out, 1))).astype(np.float32)
        cls.u, cls.dead = bp.gptq_factor(cls.xc.T @ cls.xc)

    def products_error(self, q, z, sc, x):
        d = bp.deq_rows(q.astype(np.uint8), z, sc, np.arange(self.n_out)).astype(np.float64) - self.w
        return float(np.sqrt(((d @ x.T) ** 2).sum() / ((self.w.astype(np.float64) @ x.T) ** 2).sum()))

    def test_products_on_held_out_inputs(self):
        for search in (0, 4):
            g = bp.gptq_rows_w4(self.w, self.u, self.dead, search)
            self.assertTrue(((g[0] >= 0) & (g[0] <= 15)).all())
            self.assertTrue(((g[1] >= 1) & (g[1] <= 16)).all())
            plain = self.products_error(*bp.quantize_rows_w4(self.w, search), self.xv)
            used = self.products_error(*g, self.xv)
            self.assertLess(used, 0.5 * plain, f"search {search}: {used:.4f} against {plain:.4f} uncalibrated")

    def test_rows_are_independent(self):
        a = bp.gptq_rows_w4(self.w[:32], self.u, self.dead, 4)
        b = bp.gptq_rows_w4(self.w, self.u, self.dead, 4)
        np.testing.assert_array_equal(a[0], b[0][:32])
        np.testing.assert_array_equal(a[1], b[1][:, :32])
        np.testing.assert_array_equal(a[2], b[2][:, :32])

    def test_uncorrelated_inputs_give_the_plain_quantizer(self):
        u, dead = bp.gptq_factor(np.eye(self.n_in) * 1000.0, damp=0.0)
        for x, y in zip(bp.gptq_rows_w4(self.w, u, dead, 0), bp.quantize_rows_w4(self.w, 0)):
            np.testing.assert_array_equal(x, y)

    def test_an_input_never_active_gets_zero_weights(self):
        h = self.xc.T @ self.xc
        h[:, 7] = 0.0
        h[7, :] = 0.0
        u, dead = bp.gptq_factor(h)
        self.assertTrue(dead[7])
        q, z, sc = bp.gptq_rows_w4(self.w, u, dead, 0)
        d = bp.deq_rows(q.astype(np.uint8), z, sc, np.arange(self.n_out))
        self.assertLess(float(np.abs(d[:, 7]).max()), 0.5 * float(np.abs(self.w[:, 7]).max()))

    def test_the_calibration_folder_through_the_tool(self):
        """The engine's folder layout (x16 files, aliases, calls) read back: the second matrix takes the first one's inputs."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            x = np.concatenate([self.xc, self.xv]).astype(np.float16)
            x.tofile(d / "blk.0.attn_qkv.weight.x16")
            (d / "aliases.txt").write_text("blk.0.attn_gate.weight blk.0.attn_qkv.weight\n")
            (d / "calls.txt").write_text("blk.0.attn_qkv.weight %d %d\nblk.0.attn_qkv.weight %d %d\n"
                                         % (len(self.xc), self.n_in, len(self.xv), self.n_in))
            cal = bp.Calibration(str(d), holdout=1)
            self.assertEqual(cal.source("blk.0.attn_gate.weight"), "blk.0.attn_qkv.weight")
            self.assertIsNone(cal.source("blk.0.ssm_out.weight"))
            u, dead, h_val, n_cal, n_val = cal.moments("blk.0.attn_gate.weight", self.n_in)
            self.assertEqual((n_cal, n_val), (len(self.xc), len(self.xv)))
            oe = {}
            with open(d / "blobs", "wb") as fb:
                ent, off, rel, corr = bp.pack_float_matrix(
                    lambda r0, r1: self.w[r0:r1], self.n_in, self.n_out, fb, 0, 4, 8, np.random.default_rng(1),
                    quantizer=lambda ref: bp.gptq_rows_w4(ref, u, dead, 4), h_eval=h_val, out_err=oe)
            self.assertEqual(ent["bytes"], bp.pack_bytes(self.n_in, self.n_out))
            self.assertLess(oe["used"], 0.5 * oe["plain"])


class Readers(unittest.TestCase):
    def test_float_types_need_no_gguf_package(self):
        w = np.array([[1.5, -2.25, 0.0, 3.0]], dtype=np.float32)
        f32 = bp._dequantizer(0, "F32")(w.view(np.uint8), 4)
        np.testing.assert_array_equal(f32, w)
        f16 = bp._dequantizer(1, "F16")(w.astype(np.float16).view(np.uint8), 4)
        np.testing.assert_array_equal(f16, w)
        bf16 = bp._dequantizer(30, "BF16")((w.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8), 4)
        np.testing.assert_array_equal(bf16, w)

    def test_a_type_nothing_reads_is_left_out_not_fatal(self):
        """A tensor type the installed gguf package does not know (the routed experts' Q2_0 was one) must not stop the tool:
        the directory comes from this repository's reader, and only the tensors that are packed are dequantized."""
        self.assertIsNone(bp._dequantizer(9999, "type9999"))


class Tool(unittest.TestCase):
    def test_a_gguf_end_to_end(self):
        """A small GGUF through the command line: the matching matrices of a packable shape, under the GGUF's names."""
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            try:
                from _paths import add_gguf_py
                add_gguf_py()
            except (ImportError, SystemExit):
                pass
            import gguf
        except ImportError:
            self.skipTest("no gguf package")
        rng = np.random.default_rng(9)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "m.gguf"
            wq = random_matrix(rng, 32, 512)
            wd = random_matrix(rng, 16, 256)
            w = gguf.GGUFWriter(str(src), "test")
            w.add_tensor("blk.0.attn_q.weight", wq)                                   # numpy [n_out][n_in]: GGUF shape [n_in, n_out]
            w.add_tensor("blk.1.ffn_down.weight", wd.astype(np.float16))
            w.add_tensor("blk.0.attn_k.weight", random_matrix(rng, 24, 256))          # rows not a multiple of 16
            w.add_tensor("blk.0.ffn_gate_shexp.weight", random_matrix(rng, 16, 256))  # not a default name
            w.add_tensor("blk.0.attn_norm.weight", np.ones(256, np.float32))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            out = d / "m.b70w4"
            r = subprocess.run([sys.executable, str(ROOT / "tools/b70_pack.py"), "--gguf", str(src), "--out", str(out)],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("left out: blk.0.attn_k.weight", r.stdout)
            with open(out, "rb") as f:
                meta, base = bp.read_header(f)
                self.assertEqual(set(meta["tensors"]), {"blk.0.attn_q.weight", "blk.1.ffn_down.weight"})
                for name, ref in (("blk.0.attn_q.weight", wq), ("blk.1.ffn_down.weight", wd.astype(np.float16).astype(np.float32))):
                    ent = meta["tensors"][name]
                    self.assertEqual((ent["n_in"], ent["n_out"]), (ref.shape[1], ref.shape[0]))
                    f.seek(base + ent["off"])
                    rec = np.frombuffer(f.read(ent["bytes"]), dtype="<u4")
                    q, z, sc = bp.unpack_records(rec, ent["n_in"], ent["n_out"])
                    got = bp.deq_rows(q, z, sc, np.arange(ent["n_out"]))
                    rel = float(((got - ref) ** 2).sum() / (ref ** 2).sum()) ** 0.5
                    self.assertLess(rel, 0.2, name)
            r = subprocess.run([sys.executable, str(ROOT / "tools/b70_pack.py"), "--gguf", str(src), "--out", str(d / "n.b70w4"),
                                "--match", "nothing_has_this_name"], capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertFalse((d / "n.b70w4").exists())


if __name__ == "__main__":
    unittest.main()
