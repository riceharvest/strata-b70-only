"""strata-q36's drafting for requests the device argmax cannot serve (a penalty, a temperature above 0), host side only: g++, no SYCL,
no GPU, no model.  Builds and runs tests/core/q36_spec_sampling_test.cpp (the keys the kernels order logits by, the kernels' two-stage
top-M run on the host, a window row against the plain sampler, a whole answer through windows against the plain loop, the accept / reject
rule against the target distribution, the Flash-Next engine's point-mass rule, the plain sampler against the code it replaced, and which
requests draft) and src/qwen35/sampling_test.cpp.  Skipped without g++.

    python3 -m pytest tools/test_q36_spec_sampling.py -q
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GXX = shutil.which("g++")


@unittest.skipUnless(GXX, "needs g++")
class SpecSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def build_run(self, source: Path, name: str, timeout: int = 600):
        exe = Path(self.tmp.name) / name
        cc = subprocess.run([GXX, "-std=c++20", "-O2", "-Wall", "-Wextra", "-I", str(ROOT / "include"),
                             "-I", str(ROOT / "sycl" / "src" / "qwen35"), str(source), "-o", str(exe)],
                            capture_output=True, text=True, timeout=timeout)
        self.assertEqual(cc.returncode, 0, cc.stderr[-3000:])
        return subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout)

    def test_window_sampling(self):
        r = self.build_run(ROOT / "tests" / "core" / "q36_spec_sampling_test.cpp", "q36_spec_sampling_test")
        self.assertEqual(r.returncode, 0, r.stdout[-4000:] + r.stderr[-2000:])
        self.assertIn("ALL PASSED", r.stdout)

    def test_host_penalties(self):
        r = self.build_run(ROOT / "src" / "qwen35" / "sampling_test.cpp", "qwen35_sampling_test")
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        self.assertIn("checks passed", r.stdout)


if __name__ == "__main__":
    unittest.main()
