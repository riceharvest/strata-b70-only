"""ignore_eos on a --batch engine's solo path: an end token that is not the end must not end the request in generate_batched."""
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest

from serve.server import StrataEngine

FAKE = r'''
import sys
print('INFO batch_slots=2 slot_cache=1', flush=True)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith('GEN '):
        if ' ignore_eos=1 ' in line:
            print('T 65\nT 248046\nT 66\nDONE 3 1 0.0 1.0 length 0 0 0 0 0 0 0 0.0 1 0', flush=True)
        else:
            print('T 65\nT 248046\nDONE 2 1 0.0 1.0 stop 0 0 0 0 0 0 0 0.0 1 0', flush=True)
'''


class IgnoreEosBatched(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        py = Path(self.tmp.name) / "native.py"
        py.write_text("#!" + sys.executable + "\n" + FAKE)
        py.chmod(0o700)
        self.script = py

    @unittest.skipIf(os.name == "nt", "the fake engine is a script with a shebang")
    def test_end_token_ends_the_solo_request_unless_ignore_eos(self):
        e = StrataEngine(str(self.script), ["--batch", "2"])
        self.addCleanup(e.close)
        self.assertEqual(e.batch, 2)
        self.assertEqual(list(e.generate_batched([1], 8, {}, threading.Event())), [65, 248046])
        self.assertEqual(list(e.generate_batched([1], 8, {"ignore_eos": True}, threading.Event())), [65, 248046, 66])


if __name__ == "__main__":
    unittest.main()
