"""strata-q36 is optional: off unless asked for, and nothing of it in the default build's graph.  No compiler, no GPU: a fake cmake
stands in for the oneAPI image, and the CMake file is read as text.  Skipped without bash.

    python3 -m pytest tools/test_q36_build_wiring.py -q
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_SH = ROOT / "sycl" / "tools" / "build.sh"
CMAKE = ROOT / "sycl" / "CMakeLists.txt"
BASH = shutil.which("bash")


def q36_block(text: str) -> str:
    """The text of `if(STRATA_BUILD_Q36) ... endif()` (the block's own nesting followed)."""
    start = text.index("if(STRATA_BUILD_Q36)")
    depth, pos = 0, start
    for m in re.finditer(r"^\s*(if|endif)\(", text[start:], re.M):
        depth += 1 if m.group(1) == "if" else -1
        pos = start + m.end() + 1   # (past the closing parenthesis of endif())
        if depth == 0:
            return text[start:pos]
    raise AssertionError("no matching endif() for if(STRATA_BUILD_Q36)")


@unittest.skipUnless(BASH, "needs bash")
class BuildScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.bin = Path(self.tmp.name) / "bin"
        self.bin.mkdir()
        self.log = Path(self.tmp.name) / "cmake.log"
        cmake = self.bin / "cmake"
        cmake.write_text('#!/bin/sh\necho "cmake $*" >> "%s"\nexit 0\n' % self.log.as_posix(), encoding="utf-8", newline="\n")
        cmake.chmod(0o755)
        self.build = Path(self.tmp.name) / "b"

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self, *targets, **env):
        merged = dict(os.environ, PATH=self.bin.as_posix() + os.pathsep + os.environ["PATH"], REPO=ROOT.as_posix(),
                      BUILD_DIR=self.build.as_posix())
        for key in ("AOT", "Q36", "DSV41", "PARITY", "I8_ESIMD", "HOST_MARCH"):
            merged.pop(key, None)
        merged.update(env)
        return subprocess.run([BASH, BUILD_SH.as_posix(), *targets], env=merged, capture_output=True, text=True, timeout=120)

    def test_off_by_default(self):
        r = self.run_build()
        self.assertEqual(r.returncode, 0, r.stderr)
        log = self.log.read_text()
        self.assertIn("-DSTRATA_BUILD_Q36=OFF", log)
        self.assertNotIn("-DSTRATA_BUILD_Q36=ON", log)
        self.assertIn("--target strata ", log + " ")

    def test_q36_on_reaches_cmake_and_names_its_target(self):
        r = self.run_build("strata-q36", Q36="ON")
        self.assertEqual(r.returncode, 0, r.stderr)
        log = self.log.read_text()
        self.assertIn("-DSTRATA_BUILD_Q36=ON", log)
        self.assertIn("--target strata-q36", log)

    def test_a_battlemage_build_without_the_fetched_projects_stops_before_cmake(self):
        # (a REPO of its own: the checkout's third_party/ may hold the fetched folders)
        with tempfile.TemporaryDirectory() as repo:
            r = self.run_build("strata-q36", Q36="ON", AOT="bmg-g31", REPO=repo)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("get-q36-tla.sh", r.stderr)
            self.assertFalse(self.log.exists(), "cmake never started")
            third = Path(repo) / "third_party"
            (third / "sycl-tla" / "include" / "cutlass").mkdir(parents=True)
            (third / "vllm-xpu-kernels" / "csrc" / "xpu" / "attn" / "xe_2").mkdir(parents=True)
            (third / "vllm-xpu-kernels" / "csrc" / "xpu" / "attn" / "xe_2" / "chunk_prefill.hpp").write_text("", encoding="utf-8")
            r = self.run_build("strata-q36", Q36="ON", AOT="bmg-g31", REPO=repo)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("-DSTRATA_BUILD_Q36=ON", self.log.read_text())

    def test_without_an_aot_target_there_is_nothing_to_fetch(self):
        with tempfile.TemporaryDirectory() as repo:
            self.assertEqual(self.run_build("strata-q36", Q36="ON", REPO=repo).returncode, 0)
            self.assertEqual(self.run_build("strata", AOT="bmg-g31", REPO=repo).returncode, 0)    # Q36 off: not asked for

    def test_an_existing_build_folder_takes_the_switch(self):
        self.build.mkdir()
        (self.build / "build.ninja").write_text("", encoding="utf-8")
        r = self.run_build("strata-q36", Q36="ON")
        self.assertEqual(r.returncode, 0, r.stderr)
        log = self.log.read_text()
        self.assertEqual(log.count("-DSTRATA_BUILD_Q36=ON"), 1)       # the reconfigure only: no fresh configure of a folder that has one
        self.assertNotIn("-G Ninja", log)


class CMakeFile(unittest.TestCase):
    def setUp(self):
        self.text = CMAKE.read_text(encoding="utf-8")

    def test_the_option_is_off(self):
        self.assertRegex(self.text, r'option\(STRATA_BUILD_Q36 "[^"]*" OFF\)')

    def test_nothing_of_it_before_its_section(self):
        # the section is the last thing in the file: its banner, the option and the block
        banner = self.text.index("# Qwen3.8-27B (dense, GGUF architecture qwen35)")
        before = self.text[:self.text.rindex("# ----", 0, banner)]
        for word in ("q36", "sycl-tla", "vllm-xpu", "STRATA_BUILD_Q36", "src/qwen35"):
            self.assertNotIn(word, before, word)
        block = q36_block(self.text)
        self.assertEqual(self.text[self.text.index(block) + len(block):].strip(), "", "nothing after the block")

    def test_the_fetch_is_named_where_the_third_party_code_is_missing(self):
        block = q36_block(self.text)
        self.assertIn("get-q36-tla.sh", block)
        self.assertIn("FATAL_ERROR", block)
        self.assertTrue((ROOT / "sycl" / "tools" / "get-q36-tla.sh").is_file())

    def test_every_source_of_the_target_exists(self):
        block = q36_block(self.text)
        for rel in re.findall(r"\$\{(?:CMAKE_CURRENT_SOURCE_DIR|STRATA_ROOT)\}/(src/qwen35/[\w.]+\.(?:cpp|hpp))", block):
            base = ROOT / "sycl" if "CMAKE_CURRENT_SOURCE_DIR}/" + rel in block else ROOT
            self.assertTrue((base / rel).is_file(), rel)


def without_q36_branches(text: str) -> str:
    """The text as a compile without STRATA_Q36_BUILD sees it: its `#if defined(...)` branches dropped, `#else` kept (other
    conditionals stay and are tracked, so their `#endif` is not taken for ours)."""
    out, stack = [], []
    for line in text.split("\n"):
        m = re.match(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", line)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ("if", "ifdef", "ifndef"):
                if "STRATA_Q36_BUILD" in rest:
                    stack.append(kind == "ifndef" or "!" in rest.split("STRATA_Q36_BUILD")[0])
                    continue
                stack.append(None)
            elif stack and stack[-1] is not None:
                if kind == "else":
                    stack[-1] = not stack[-1]
                elif kind == "endif":
                    stack.pop()
                continue
            elif kind == "endif" and stack:
                stack.pop()
        if all(x for x in stack if x is not None):
            out.append(line)
    return "\n".join(out)


class StrataIsKeptAsItWas(unittest.TestCase):
    """`strata` compiles the same kernel sources and headers as before: the 27B's forms are sources of strata-q36 only, and the
    few lines of shared headers that know the dense family are behind STRATA_Q36_BUILD, which only strata-q36 defines."""

    def test_the_kernel_files_say_nothing_of_it(self):
        for rel in ("sycl/src/kernels/cuda/native_mmvq.dp.cpp", "sycl/src/kernels/cuda/dequant_bf16.dp.cpp",
                    "sycl/include/strata/kernels/native_mmvq.hpp", "include/strata/kernels/dequant_bf16.hpp",
                    "sycl/src/program/generate.cpp"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            for word in ("q36", "STRATA_Q36", "dense_mode", "Qwen35)"):
                self.assertNotIn(word, text, f"{rel}: {word}")

    def test_the_family_headers_know_the_dense_family_only_behind_the_definition(self):
        for rel in ("include/strata/artifact/model_family.hpp", "include/strata/artifact/gguf_reader.hpp",
                    "include/strata/core/family_layout.hpp", "sycl/include/strata/core/family_layout.hpp"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("STRATA_Q36_BUILD", text, rel)
            self.assertNotRegex(without_q36_branches(text), r"\bQwen35\b(?!Moe)", rel)

    def test_only_strata_q36_defines_it(self):
        text = CMAKE.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"STRATA_Q36_BUILD=1", text)), 1)
        block = q36_block(text)
        self.assertRegex(block, r"target_compile_definitions\(strata-q36 PRIVATE [^)]*STRATA_Q36_BUILD=1")
        self.assertIn("src/qwen35/q36_mmvq_dense.dp.cpp", block)
        self.assertNotIn("q36_mmvq_dense", text.replace(block, ""), "the dense forms are not a source of any other target")

    def test_no_other_build_file_lists_the_dense_source(self):
        for rel in ("CMakeLists.txt", "sycl/CMakeLists.txt"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            if rel == "sycl/CMakeLists.txt":
                text = text.replace(q36_block(text), "")
            self.assertNotIn("q36_mmvq_dense", text, rel)


class ThirdParty(unittest.TestCase):
    PINS = {"sycl-tla": "87f6850680a580654b9ea2c80dbc01aeb36ad231", "vllm-xpu-kernels": "ddf336d86e3c8602888572a3502f951abd51df12"}

    def test_the_pins_agree_in_the_script_and_the_notices(self):
        script = (ROOT / "sycl" / "tools" / "get-q36-tla.sh").read_text(encoding="utf-8")
        notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for name, sha in self.PINS.items():
            self.assertIn(sha, script, name)
            self.assertIn(sha, notices, name)
        self.assertIn("SYCL_TLA_COMMIT=" + self.PINS["sycl-tla"], script)
        self.assertIn("VLLM_COMMIT=" + self.PINS["vllm-xpu-kernels"], script)

    def test_fetched_folders_are_ignored_and_the_licences_are_kept(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/third_party/sycl-tla/", ignore)
        self.assertIn("/third_party/vllm-xpu-kernels/", ignore)
        for f in ("vllm-xpu-kernels-LICENSE", "sycl-tla-LICENSE.txt"):
            self.assertTrue((ROOT / "third_party" / "licenses" / f).is_file(), f)

    def test_the_patch_and_the_modified_copy_say_what_they_are(self):
        patch = (ROOT / "sycl" / "third_party" / "q36-tla" / "vllm-fmha_utils-no-torch.patch").read_text(encoding="utf-8")
        self.assertIn("MODIFIES a file of vllm-xpu-kernels", patch)
        self.assertIn("diff --git a/csrc/xpu/attn/fmha_utils.hpp", patch)
        head = (ROOT / "sycl" / "src" / "qwen35" / "merge_attn_states.hpp").read_text(encoding="utf-8")[:900]
        self.assertIn("MODIFIED copy", head)
        self.assertIn("Apache", head)


if __name__ == "__main__":
    unittest.main()
