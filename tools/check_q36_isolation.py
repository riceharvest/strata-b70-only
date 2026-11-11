"""tools/check_q36_isolation.py [BASE] - `strata` must compile the sources it compiles on BASE (default: b70); strata-q36 and the dense
family are the only differences.  No compiler, no GPU: it reads git.

  1. every file that differs from BASE under the kernel, core, host and header trees that `strata` compiles must, with
     STRATA_Q36_BUILD undefined (the `#if defined(STRATA_Q36_BUILD)` branches dropped, `#else` kept), be BASE's text (comments and
     white space aside); the diff is printed, so the guards can be read;
  2. sycl/CMakeLists.txt must be BASE's text with the strata-q36 section appended (BASE's text is a prefix);
  3. nothing else of a tree `strata` compiles may differ (the list is printed);
  4. with g++ at hand, each changed header run through the preprocessor (SYCL headers stubbed, STRATA_Q36_BUILD undefined) gives
     byte for byte BASE's output.

    python3 tools/check_q36_isolation.py            # against b70
Exit status 0: all four hold.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "b70"
# the trees `strata` and its libraries are compiled from (sycl/CMakeLists.txt's source lists and what they include)
TREES = ("include/", "sycl/include/", "sycl/src/kernels/", "sycl/src/core/", "sycl/src/prefill/", "sycl/src/program/",
         "sycl/src/platform/", "sycl/src/ngram/", "src/core/", "src/kernels/", "src/prefill/", "src/artifact/", "src/platform/",
         "src/spec/", "src/ngram/")


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout


def resolve(text: str) -> str:
    """The text as a compile without STRATA_Q36_BUILD sees it."""
    out, stack = [], []
    for line in text.split("\n"):
        m = re.match(r"\s*#\s*(if|else|endif)\b(.*)", line)
        if m and (m.group(1) != "if" or "STRATA_Q36_BUILD" in m.group(2)):
            if m.group(1) == "if":
                stack.append("!" in m.group(2).split("STRATA_Q36_BUILD")[0])   # defined(X) is false: only `!defined` keeps its branch
            elif m.group(1) == "else":
                stack[-1] = not stack[-1]
            else:
                stack.pop()
            continue
        if all(stack):
            out.append(line)
    return "\n".join(out)


def norm(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocessed_identical(files: list[str]) -> bool:
    """g++ -E -P of each header in this tree and in BASE's, <sycl/sycl.hpp> and <dpct/dpct.hpp> stubbed: the same bytes."""
    gxx = shutil.which("g++")
    headers = [f for f in files if f.endswith((".hpp", ".h"))]
    if not gxx or not headers:
        print("\n(no g++, or no header changed: the preprocessor check is skipped)")
        return True
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for stub in ("sycl/sycl.hpp", "dpct/dpct.hpp"):
            (tmp / "stub" / stub).parent.mkdir(parents=True, exist_ok=True)
            (tmp / "stub" / stub).write_text("", encoding="utf-8")
        base = tmp / "base"
        base.mkdir()
        archive = subprocess.run(["git", "-C", str(ROOT), "archive", BASE, "include", "sycl/include"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(base)], input=archive, check=True)
        for f in headers:
            incs = ["include"] if not f.startswith("sycl/") else ["sycl/include", "include"]
            out = {}
            for name, root in (("this tree", ROOT), (BASE, base)):
                cmd = [gxx, "-std=c++20", "-x", "c++", "-E", "-P", "-w", "-I", str(tmp / "stub"), *sum((["-I", str(root / d)] for d in incs), []),
                       str(root / f)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                out[name] = r.stdout if r.returncode == 0 else f"error: {r.stderr[:200]}"
            same = out["this tree"] == out[BASE] and not out[BASE].startswith("error")
            ok &= same
            print(f"  g++ -E -P {f}: {len(out['this tree'].splitlines())} lines, byte for byte {BASE}'s: {same}")
    return ok


def main() -> int:
    ok = True
    changed = [f for f in git("diff", BASE, "--name-only").split() if f.startswith(TREES)]
    print(f"files under the trees strata compiles that differ from {BASE}: {changed or 'none'}")
    for f in changed:
        new = (ROOT / f).read_text(encoding="utf-8")
        old = git("show", f"{BASE}:{f}")
        same = norm(resolve(new)) == norm(old)
        ok &= same
        print(f"\n=== {f}: without STRATA_Q36_BUILD it is {BASE}'s text: {same}")
        print(git("diff", BASE, "--", f))
    cm_new = (ROOT / "sycl/CMakeLists.txt").read_text(encoding="utf-8")
    cm_old = git("show", f"{BASE}:sycl/CMakeLists.txt")
    prefix = cm_new.startswith(cm_old)
    ok &= prefix
    print("\nthe preprocessor, STRATA_Q36_BUILD undefined:")
    ok &= preprocessed_identical(changed)
    print(f"\nsycl/CMakeLists.txt: {BASE}'s text is a prefix of this one (the strata-q36 section is appended after line "
          f"{cm_old.count(chr(10))}): {prefix}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
