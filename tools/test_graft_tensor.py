"""Tests for tools/graft_tensor.py on tiny synthetic GGUF files. No GPU, no network (the URL case uses a local server).

    python3 tools/test_graft_tensor.py
"""
from __future__ import annotations

import hashlib
import http.server
import re
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import graft_tensor as G  # noqa: E402
G.BACKOFF = 0.0
from gguf_reader import GGUFFile  # noqa: E402

IQ2_XXS, IQ2_S, Q2_0 = 16, 22, 42
ALIGN = 32
NAME = "blk.1.ffn_gate_exps.weight"


def blob(n: int, seed: int) -> bytes:
    return bytes((seed * 31 + i * 7) % 251 + 1 for i in range(n))     # never a zero byte


def gguf(tensors, align=ALIGN) -> bytes:
    """tensors: [(name, shape, type_id, data)]; each tensor sits in a slot padded to the alignment, in the order given."""
    def s(x):
        return struct.pack("<Q", len(x.encode())) + x.encode()
    head = struct.pack("<I", 0x46554747) + struct.pack("<IQQ", 3, len(tensors), 1)
    head += s("general.alignment") + struct.pack("<I", 4) + struct.pack("<I", align)
    off, body = 0, b""
    for name, shape, tid, data in tensors:
        head += s(name) + struct.pack("<I", len(shape)) + struct.pack("<%dQ" % len(shape), *shape) + struct.pack("<IQ", tid, off)
        slot = (len(data) + align - 1) // align * align
        body += data + bytes(slot - len(data))
        off += slot
    head += bytes((-len(head)) % align)
    return head + body


# 256 elements per block: IQ2_S 82 bytes, IQ2_XXS 66 bytes. shape [256, 1, 2] = two blocks.
SHAPE = [256, 1, 2]
OLD = blob(2 * 82, 1)
NEW = blob(2 * 66, 2)
BEFORE = blob(512, 3)
AFTER = blob(132, 4)


def target_bytes():
    return gguf([("token_embd.weight", [32, 4], 0, BEFORE), (NAME, SHAPE, IQ2_S, OLD), ("blk.1.ffn_up_exps.weight", SHAPE, IQ2_XXS, AFTER)])


def source_bytes(shape=SHAPE, tid=IQ2_XXS, data=NEW, lead=False):
    ts = [(NAME, shape, tid, data), ("other.weight", [32, 2], 0, blob(256, 9))]
    return gguf(([("lead.weight", [32, 1], 0, blob(128, 8))] if lead else []) + ts)


class Files(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.target = self.dir / "target.gguf"
        self.source = self.dir / "source.gguf"
        self.target.write_bytes(target_bytes())
        self.source.write_bytes(source_bytes(lead=True))
        self.log = []

    def tearDown(self):
        self.tmp.cleanup()

    def tensor(self, path, name=NAME):
        g = GGUFFile(path)
        t = next(x for x in g.tensors if x.name == name)
        return g, t

    def data(self, path, name=NAME, n=None):
        g, t = self.tensor(path, name)
        n = t.expected_bytes() if n is None else n
        with open(path, "rb") as f:
            f.seek(g.data_start + t.offset)
            return f.read(n)

    def run_graft(self, **kw):
        return G.graft(kw.pop("target", self.target), kw.pop("tensor", NAME), kw.pop("source", self.source), log=self.log.append, **kw)


class InPlace(Files):
    def test_the_tensor_is_taken_over_and_nothing_else_changes(self):
        before = self.target.read_bytes()
        self.assertEqual(self.run_graft(), "grafted")
        after = self.target.read_bytes()
        self.assertEqual(len(before), len(after))
        g, t = self.tensor(self.target)
        self.assertEqual(t.type_name, "IQ2_XXS")
        self.assertEqual(t.shape, SHAPE)
        self.assertEqual(self.data(self.target), NEW)
        old_slot = (len(OLD) + ALIGN - 1) // ALIGN * ALIGN
        self.assertEqual(self.data(self.target, n=old_slot)[len(NEW):], bytes(old_slot - len(NEW)))   # the rest is zeros
        self.assertEqual(self.data(self.target, "token_embd.weight", 512), BEFORE)
        self.assertEqual(self.data(self.target, "blk.1.ffn_up_exps.weight"), AFTER)
        changed = [i for i in range(len(before)) if before[i] != after[i]]
        head_changed = [i for i in changed if i < g.data_start]
        self.assertEqual(len(head_changed), 1)                         # one byte of the header: the type id 22 -> 16
        self.assertTrue(all(i >= g.data_start + t.offset for i in changed if i >= g.data_start))
        self.assertLess(max(changed), g.data_start + t.offset + old_slot)

    def test_a_second_run_changes_nothing(self):
        self.run_graft()
        once = self.target.read_bytes()
        self.assertEqual(self.run_graft(), "already")
        self.assertEqual(self.target.read_bytes(), once)

    def test_the_right_source_hash_is_accepted_and_a_wrong_one_leaves_the_file_alone(self):
        before = self.target.read_bytes()
        with self.assertRaisesRegex(G.Refused, "SHA-256"):
            self.run_graft(source_sha256="0" * 64)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(self.run_graft(source_sha256=hashlib.sha256(NEW).hexdigest()), "grafted")

    def test_out_writes_a_copy_and_leaves_the_target(self):
        before = self.target.read_bytes()
        out = self.dir / "copy.gguf"
        self.assertEqual(self.run_graft(out=out), "grafted")
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(self.tensor(out)[1].type_name, "IQ2_XXS")
        self.assertEqual(self.data(out), NEW)

    def test_an_interrupted_run_is_finished_by_the_next(self):
        """The data is written but the type is not changed yet: the file still reads as the original, and a new run completes it."""
        g, t = self.tensor(self.target)
        with open(self.target, "r+b") as f:
            f.seek(g.data_start + t.offset)
            f.write(NEW)
        self.assertEqual(self.tensor(self.target)[1].type_name, "IQ2_S")
        self.assertEqual(self.run_graft(), "grafted")
        self.assertEqual(self.data(self.target), NEW)
        self.assertEqual(self.tensor(self.target)[1].type_name, "IQ2_XXS")


class Refusals(Files):
    def refuse(self, pattern, **kw):
        before = self.target.read_bytes()
        with self.assertRaisesRegex(G.Refused, pattern):
            self.run_graft(**kw)
        self.assertEqual(self.target.read_bytes(), before)             # a refusal never changes the target

    def test_a_tensor_the_target_lacks(self):
        self.refuse("0 tensors named", tensor="blk.9.ffn_gate_exps.weight")

    def test_a_tensor_the_source_lacks(self):
        self.source.write_bytes(gguf([("other.weight", [32, 2], 0, blob(256, 9))]))
        self.refuse("the source has 0 tensors")

    def test_another_shape(self):
        self.source.write_bytes(source_bytes(shape=[256, 1, 3], data=blob(3 * 66, 5)))
        self.refuse("shape")

    def test_a_source_bigger_than_the_slot(self):
        # the target's slot is two IQ2_XXS blocks (132 bytes, padded to 160); two IQ2_S blocks (164) do not fit
        self.target.write_bytes(gguf([(NAME, SHAPE, IQ2_XXS, NEW), ("blk.1.ffn_up_exps.weight", SHAPE, IQ2_XXS, AFTER)]))
        self.source.write_bytes(source_bytes(tid=IQ2_S, data=OLD))
        self.refuse("bigger than the place")

    def test_a_type_whose_size_is_not_known(self):
        self.source.write_bytes(source_bytes(tid=999))
        self.refuse("not known")

    def test_a_source_that_is_too_short(self):
        raw = source_bytes()
        self.source.write_bytes(raw[:-300])
        self.refuse("too short")

    def test_not_a_gguf(self):
        self.source.write_bytes(b"nope" * 100)
        self.refuse("source.gguf")


class Handler(http.server.BaseHTTPRequestHandler):
    body = b""
    honour_range = True

    def log_message(self, *a):
        pass

    def do_GET(self):
        m = re.match(r"bytes=(\d+)-(\d+)", self.headers.get("Range", ""))
        if not m or not self.honour_range:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.body)))
            self.end_headers()
            self.wfile.write(self.body)
            return
        a, b = int(m.group(1)), min(int(m.group(2)), len(self.body) - 1)
        self.send_response(206)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (a, b, len(self.body)))
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(self.body[a:b + 1])


class Remote(Files):
    def serve(self, honour):
        h = type("H", (Handler,), {"body": self.source.read_bytes(), "honour_range": honour})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return "http://127.0.0.1:%d/UD-IQ1_M/source.gguf" % srv.server_address[1]

    def test_the_tensor_comes_over_range_requests(self):
        url = self.serve(True)
        self.assertEqual(self.run_graft(source=url, source_sha256=hashlib.sha256(NEW).hexdigest()), "grafted")
        self.assertEqual(self.data(self.target), NEW)
        self.assertEqual(self.tensor(self.target)[1].type_name, "IQ2_XXS")

    def test_a_server_that_ignores_range_is_refused(self):
        before = self.target.read_bytes()
        url = self.serve(False)
        with self.assertRaisesRegex(G.Refused, "ignored the range"):
            self.run_graft(source=url)
        self.assertEqual(self.target.read_bytes(), before)


class Command(Files):
    def test_exit_status(self):
        args = ["--target", str(self.target), "--tensor", NAME, "--source", str(self.source)]
        self.assertEqual(G.main(args), 0)
        self.assertEqual(G.main(args), 0)                              # already done
        self.assertEqual(G.main(["--target", str(self.target), "--tensor", "nope", "--source", str(self.source)]), 1)


class Compact(Files):
    """--compact: the hole a smaller tensor leaves is closed, every later tensor moves down, offsets stay contiguous."""

    def grafted(self):
        self.run_graft()
        return self.target

    def contiguous(self, path):
        g = GGUFFile(path)
        end = 0
        for t in sorted(g.tensors, key=lambda x: x.offset):
            self.assertEqual(t.offset, (end + g.alignment - 1) // g.alignment * g.alignment, t.name)
            end = t.offset + t.expected_bytes()
        # the file ends with the last tensor, or with the padding the original had after it (less than one alignment)
        self.assertTrue(g.data_start + end <= Path(path).stat().st_size < g.data_start + end + g.alignment)

    def test_the_hole_is_closed_and_the_data_follows(self):
        self.grafted()
        before = {t.name: self.data(self.target, t.name) for t in GGUFFile(self.target).tensors}
        size = self.target.stat().st_size
        self.assertEqual(G.compact(self.target, log=self.log.append), "compacted")
        self.contiguous(self.target)
        self.assertLess(self.target.stat().st_size, size)
        for name, d in before.items():
            self.assertEqual(self.data(self.target, name), d, name)
        self.assertFalse(self.target.with_name("target.gguf.compact").exists())
        self.assertEqual(G.compact(self.target, log=self.log.append), "already")

    def test_the_file_is_not_touched_by_default(self):
        self.grafted()
        self.assertRaises(AssertionError, self.contiguous, self.target)    # the hole is still there without --compact

    def test_wrong_expected_size_is_refused_and_changes_nothing(self):
        self.grafted()
        raw = self.target.read_bytes()
        with self.assertRaisesRegex(G.Refused, "expected"):
            G.compact(self.target, expect_size=1)
        self.assertEqual(self.target.read_bytes(), raw)
        self.assertFalse(self.target.with_name("target.gguf.compact").exists())

    def test_a_run_stopped_in_the_middle_is_finished_by_the_next(self):
        self.grafted()
        before = {t.name: self.data(self.target, t.name) for t in GGUFFile(self.target).tensors}
        real = G._save
        calls = []

        def crash(journal, plan):
            real(journal, plan)
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt

        old_chunk = G.CHUNK
        G.CHUNK = 16                                    # many small chunks, so it stops inside a tensor
        try:
            with self.assertRaises(KeyboardInterrupt):
                G_save = G._save
                G._save = crash
                try:
                    G.compact(self.target, log=self.log.append)
                finally:
                    G._save = G_save
            self.assertTrue(self.target.with_name("target.gguf.compact").exists())
            self.assertEqual(G.compact(self.target, log=self.log.append), "compacted")
        finally:
            G.CHUNK = old_chunk
        self.contiguous(self.target)
        for name, d in before.items():
            self.assertEqual(self.data(self.target, name), d, name)

    def test_cli_compact_after_the_graft(self):
        args = ["--target", str(self.target), "--tensor", NAME, "--source", str(self.source), "--compact"]
        self.assertEqual(G.main(args), 0)
        self.contiguous(self.target)
        self.assertEqual(G.main(["--target", str(self.target), "--compact"]), 0)


class Safety(Files):
    def test_a_hard_link_or_symlink_is_refused_and_the_other_path_is_untouched(self):
        raw = self.target.read_bytes()
        link = self.dir / "hard.gguf"
        link.hardlink_to(self.target)
        with self.assertRaisesRegex(G.Refused, "hard links"):
            self.run_graft()
        with self.assertRaisesRegex(G.Refused, "hard links"):
            G.compact(self.target)
        self.assertEqual(link.read_bytes(), raw)
        link.unlink()
        sym = self.dir / "sym.gguf"
        sym.symlink_to(self.target)
        with self.assertRaisesRegex(G.Refused, "symlink"):
            self.run_graft(target=sym)
        self.assertEqual(self.target.read_bytes(), raw)
        out = self.dir / "copy.gguf"                                  # --out works on a linked file: it writes a copy
        self.assertEqual(self.run_graft(target=sym, out=out), "grafted")

    def test_a_non_gguf_or_cut_target_is_a_refusal_everywhere(self):
        for junk in (b"<html>nope</html>" * 10, target_bytes()[:100]):
            self.target.write_bytes(junk)
            with self.assertRaisesRegex(G.Refused, "not a whole GGUF"):
                self.run_graft()
            with self.assertRaises(G.Refused):
                G.grafted(self.target, NAME, "IQ2_XXS")
            with self.assertRaises(G.Refused):
                G.compact(self.target)
            self.assertEqual(G.main(["--target", str(self.target), "--tensor", NAME, "--source", str(self.source)]), 1)
            self.assertEqual(self.target.read_bytes(), junk)

    def test_a_pending_journal_means_main_only_finishes_the_compact(self):
        (self.dir / "target.gguf.compact").write_text("{}")
        with self.assertRaises(Exception):                            # the journal here is empty: compact stops on it
            G.main(["--target", str(self.target), "--tensor", NAME, "--source", str(self.source), "--compact"])
        self.assertEqual(self.tensor(self.target)[1].type_name, "IQ2_S")      # and no graft was done first

    def test_is_compact(self):
        self.run_graft()
        self.assertFalse(G.is_compact(self.target))
        G.compact(self.target, log=self.log.append)
        self.assertTrue(G.is_compact(self.target))


class ShortBody(http.server.BaseHTTPRequestHandler):
    hits = 0

    def log_message(self, *a):
        pass

    def do_GET(self):
        type(self).hits += 1
        self.send_response(206)
        self.send_header("Content-Range", "bytes 0-99/1000")
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.write(b"x" * 40)                                   # the connection closes with 60 bytes missing


class Truncated(unittest.TestCase):
    def test_a_body_cut_short_is_retried_a_bounded_number_of_times_and_then_refused(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), ShortBody)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        ShortBody.hits = 0
        with self.assertRaisesRegex(G.Refused, "cannot read"):
            G._http_range("http://127.0.0.1:%d/x.gguf" % srv.server_address[1], 0, 100)
        self.assertEqual(ShortBody.hits, 3)


if __name__ == "__main__":
    unittest.main()
