"""The dense qwen35 family on synthetic GGUFs, through the host headers only (g++, no SYCL, no GPU, no model): `strata` built with
the dense family (STRATA_Q36_BUILD, the default) accepts a qwen35 file (the architecture guard, the 27B's geometry, the layout table
of a dense layer and of the MTP block) and its family switch (include/strata/qwen35/dispatch.hpp) sends exactly such a file to the
dense engine; a `strata` built with `Q36=OFF` compiles the same headers without STRATA_Q36_BUILD and still refuses it by name.
Skipped without g++.

    python3 -m pytest tools/test_q36_family.py -q
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GXX = shutil.which("g++")
SOURCE = ROOT / "tests" / "core" / "q36_dense_family_test.cpp"
DISPATCH = ROOT / "tests" / "core" / "q36_dispatch_test.cpp"


@unittest.skipUnless(GXX, "needs g++")
class DenseFamily(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_test(self, name: str, *defs: str, source: Path = SOURCE):
        exe = Path(self.tmp.name) / name
        # the CUDA-tree copy of the headers (include/): the same text as sycl/include's, without the SYCL includes
        cc = subprocess.run([GXX, "-std=c++20", "-O0", "-w", *defs, "-I", str(ROOT / "include"), "-I", str(ROOT / "tests" / "core"),
                             str(source), "-o", str(exe)], capture_output=True, text=True, timeout=300)
        self.assertEqual(cc.returncode, 0, cc.stderr[-2000:])
        return subprocess.run([str(exe)], capture_output=True, text=True, timeout=120)

    def test_strata_with_the_dense_family_accepts_the_dense_file(self):
        r = self.run_test("as_q36", "-DSTRATA_Q36_BUILD")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALL PASSED", r.stdout)

    def test_the_family_switch_sends_only_a_qwen35_file_to_the_dense_engine(self):
        r = self.run_test("dispatch", "-DSTRATA_Q36_BUILD", source=DISPATCH)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALL PASSED", r.stdout)

    def test_strata_without_the_dense_family_still_refuses_it_by_name(self):
        r = self.run_test("as_strata")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("ALL PASSED", r.stdout)
        self.assertIn("this engine requires 'qwen4exp'", r.stdout)


if __name__ == "__main__":
    unittest.main()
