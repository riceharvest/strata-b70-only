"""tools/check_q36_isolation.py [BASE] - the merged layout's guard: one `strata` binary holds two model families, and the Flash-Next
family must be exactly what it was on BASE (default: 86f64dc2, the commit before the dense family became part of `strata`).
No compiler, no GPU: it reads git.

Before the merge the dense family was a second binary, `strata-q36`, and this tool proved that `strata` was compiled from BASE's
sources.  Now the dense family is built into `strata` (`STRATA_BUILD_Q36`, build.sh `Q36`, default ON), so the proof is: with
the dense family compiled out - the `#if defined(STRATA_Q36_BUILD)` branches dropped (`#else` kept) in the sources, and the
dense section of sycl/CMakeLists.txt (everything from its "Qwen3.8-27B (dense" banner to the end) cut off - what is left is BASE's
text.  `Q36=OFF` builds that.  Checks:

  1. every file that differs from BASE under the kernel, core, host and header trees that `strata` compiles must, with
     STRATA_Q36_BUILD undefined, be BASE's text (comments and white space aside); the diff is printed, so the guards can be read;
     a file new in this tree must be empty without it; a deleted one is a difference;
  2. sycl/CMakeLists.txt before the dense section is BASE's text; the dense section is the tail of the file;
  3. the merged layout: no `strata-q36` target, no second `main` in the dense sources, every dense source is listed only in the
     dense section and only as a source of `strata`, STRATA_Q36_BUILD is defined once (for every target, so an inline function has
     one definition in the binary), and only the allowed files mention STRATA_Q36_BUILD;
  4. the family switch is the first statement of `main()` in sycl/src/program/generate.cpp, behind the guard, before the
     buffering, the environment and the B70 defaults (so a dense start is the one the old binary made, and a Flash-Next start never
     reaches the dense code);
  5. with g++ at hand, each changed header run through the preprocessor (SYCL headers stubbed, STRATA_Q36_BUILD undefined) gives
     byte for byte BASE's output (nothing, for a header new in this tree).

The B70 card library (sycl/src/b70, sycl/include/strata/b70; docs/B70_LIB.md) is a second switch of the same kind:
`STRATA_BUILD_B70LIB` builds it and defines STRATA_B70_LIB for every target, and the shared sources reach it only behind
`#if defined(STRATA_B70_LIB)`.  So "compiled out" means both macros undefined here, the library's own trees are not compared
with BASE (they are new), the CMake text is compared up to the library's section, and check 6 holds the library's layout:
its section sits before the dense one, names only src/b70 sources, and nothing outside the allowed files names the macro.

    python3 tools/check_q36_isolation.py            # against 86f64dc2
Exit status 0: all six hold.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = sys.argv[1] if len(sys.argv) > 1 else "86f64dc2"
# the trees `strata` and its libraries are compiled from (sycl/CMakeLists.txt's source lists and what they include)
TREES = ("include/", "sycl/include/", "sycl/src/kernels/", "sycl/src/core/", "sycl/src/prefill/", "sycl/src/program/",
         "sycl/src/platform/", "sycl/src/ngram/", "src/core/", "src/kernels/", "src/prefill/", "src/artifact/", "src/platform/",
         "src/spec/", "src/ngram/")
# the files that may mention STRATA_Q36_BUILD (the family headers and the family switch)
ALLOWED = {"include/strata/artifact/model_family.hpp", "include/strata/artifact/gguf_reader.hpp",
           "include/strata/core/family_layout.hpp", "sycl/include/strata/core/family_layout.hpp",
           "include/strata/qwen35/dispatch.hpp", "sycl/src/program/generate.cpp"}
DENSE_BANNER = re.compile(r"^# -+\n# Qwen3\.8-27B \(dense", re.M)
B70_BANNER = re.compile(r"^# -+\n# The B70 card library", re.M)
# the two family / library switches: a branch behind either is not part of the Flash-Next-only text
GUARDS = ("STRATA_Q36_BUILD", "STRATA_B70_LIB")
# the card library's own trees (new, built by STRATA_BUILD_B70LIB): not compared with BASE
B70_TREES = ("sycl/include/strata/b70/", "sycl/src/b70/")
# the shared files that may reach the library (behind STRATA_B70_LIB)
B70_ALLOWED = {"sycl/src/kernels/cuda/native_mmvq.dp.cpp", "sycl/src/prefill/gemm.dp.cpp", "sycl/src/core/native_dense.cpp",
               "sycl/src/program/generate.cpp"}
DENSE_SOURCES = ("q36_engine.dp.cpp", "q36_mmvq_dense.dp.cpp", "q36_xr_dense.dp.cpp", "q36_requant.cpp")


def git(*args: str, check: bool = True) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=check).stdout


def in_base(path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def resolve(text: str) -> str:
    """The text as a compile without STRATA_Q36_BUILD sees it: its `#if defined(...)` branches dropped, `#else` kept.  Other
    conditionals stay as they are (and are tracked, so their `#endif` is not taken for ours)."""
    out, stack = [], []   # an entry per open conditional: None (not ours), or whether the branch we are in is kept
    for line in text.split("\n"):
        m = re.match(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b(.*)", line)
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ("if", "ifdef", "ifndef"):
                guard = next((g for g in GUARDS if g in rest), None)
                if guard:
                    stack.append(kind == "ifndef" or "!" in rest.split(guard)[0])   # defined(X) is false here
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


def norm(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def norm_cmake(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"#[^\n]*", "", text)).strip()


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
                if not (root / f).exists():   # a header new in this tree: BASE has none, so nothing is what it must preprocess to
                    out[name] = ""
                    continue
                cmd = [gxx, "-std=c++20", "-x", "c++", "-E", "-P", "-w", "-I", str(tmp / "stub"), *sum((["-I", str(root / d)] for d in incs), []),
                       str(root / f)]
                r = subprocess.run(cmd, capture_output=True, text=True)
                # (`#pragma once` of a header compiled as the main file is passed through by the preprocessor: not text of the header's)
                out[name] = "".join(ln for ln in r.stdout.splitlines(True) if ln.strip() not in ("", "#pragma once")) if r.returncode == 0 else f"error: {r.stderr[:200]}"
            same = out["this tree"] == out[BASE] and not out[BASE].startswith("error")
            ok &= same
            print(f"  g++ -E -P {f}: {len(out['this tree'].splitlines())} lines, byte for byte {BASE}'s: {same}")
    return ok


def check_files() -> bool:
    ok = True
    changed = [f for f in git("diff", BASE, "--name-only").split("\n") if f and f.startswith(TREES) and not f.startswith(B70_TREES)]
    print(f"files under the trees strata compiles that differ from {BASE}: {changed or 'none'}")
    for f in changed:
        old = in_base(f)
        path = ROOT / f
        if not path.exists():
            ok = False
            print(f"\n=== {f}: DELETED (a source of `strata` on {BASE})")
            continue
        new = path.read_text(encoding="utf-8")
        if old is None:
            body = norm(resolve(new).replace("#pragma once", ""))
            same = body == ""
            print(f"\n=== {f}: new in this tree; without STRATA_Q36_BUILD it is empty: {same}")
        else:
            same = norm(resolve(new)) == norm(resolve(old))   # (BASE has the header guards too, for the second binary)
            print(f"\n=== {f}: without STRATA_Q36_BUILD it is {BASE}'s text: {same}")
            print(git("diff", BASE, "--", f))
        ok &= same
    print("\nthe preprocessor, STRATA_Q36_BUILD undefined:")
    ok &= preprocessed_identical(changed)
    return ok


def check_cmake() -> bool:
    new = (ROOT / "sycl/CMakeLists.txt").read_text(encoding="utf-8")
    old = in_base("sycl/CMakeLists.txt") or ""
    mn, mo = DENSE_BANNER.search(new), DENSE_BANNER.search(old)
    if not mn or not mo:
        print("\nsycl/CMakeLists.txt: the dense section's banner is missing (here or on BASE): False")
        return False
    mb = B70_BANNER.search(new)   # the card library's section, when there is one, comes first and is not BASE's either
    cut = mb.start() if mb and mb.start() < mn.start() else mn.start()
    head_same = norm_cmake(new[:cut]) == norm_cmake(old[:mo.start()])
    print(f"\nsycl/CMakeLists.txt: before the library and dense sections (line {new[:cut].count(chr(10)) + 1}) it is {BASE}'s text: {head_same}")
    return head_same


def check_b70_lib() -> bool:
    """6. The card library is a switch of its own, before the dense section, and the shared sources reach it behind its macro."""
    ok = True

    def verdict(what: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  {what}: {cond}")

    new = (ROOT / "sycl/CMakeLists.txt").read_text(encoding="utf-8")
    mb, md = B70_BANNER.search(new), DENSE_BANNER.search(new)
    print("\nthe card library:")
    if not mb:
        print("  no card library section in sycl/CMakeLists.txt (a tree from before it): nothing to check")
        return True
    verdict("its section sits before the dense section", bool(md) and mb.start() < md.start())
    sect = re.sub(r"#[^\n]*", "", new[mb.start():md.start()] if md else new[mb.start():])
    rest = re.sub(r"#[^\n]*", "", new[:mb.start()] + (new[md.start():] if md else ""))
    verdict("it is a switch: option(STRATA_BUILD_B70LIB ...) and every library line inside if(STRATA_BUILD_B70LIB)",
            "option(STRATA_BUILD_B70LIB" in sect and re.search(r"^if\(STRATA_BUILD_B70LIB\)", sect, re.M) is not None
            and "add_library(strata_b70 " in sect.split("if(STRATA_BUILD_B70LIB)", 1)[-1])
    verdict("it names no model family's sources (no src/qwen35 path)", "qwen35/" not in sect)
    verdict("no src/b70 source is listed outside it", "src/b70/" not in rest and "_b70_src}" not in rest)
    listed = set(re.findall(r"\$\{_b70_src\}/([\w.]+)", sect))
    on_disk = {q.name for q in (ROOT / "sycl/src/b70").glob("*.cpp")}
    verdict(f"every source in sycl/src/b70 is built by it ({len(on_disk)} files)", listed == on_disk)
    verdict("STRATA_B70_LIB is defined once, for every target (add_compile_definitions), inside it",
            len(re.findall(r"STRATA_B70_LIB\b", sect + rest)) == 1 and "add_compile_definitions(STRATA_B70_LIB=1)" in sect)
    verdict("the dense family refuses a build without it", "if(NOT STRATA_BUILD_B70LIB)" in rest)
    hits = git("grep", "-l", "STRATA_B70_LIB", "--", "include", "src", "sycl/include", "sycl/src", check=False).split("\n")
    stray = sorted(f for f in hits if f and f not in B70_ALLOWED and not f.startswith(B70_TREES))
    verdict(f"only the shared dispatch, the prompt GEMM, the dense loader and the flag parser name STRATA_B70_LIB (stray: {stray or 'none'})",
            not stray)
    unguarded = []
    for f in sorted(B70_ALLOWED):
        text = (ROOT / f).read_text(encoding="utf-8")
        if "strata/b70/" in resolve(text) or "strata::b70" in resolve(text):
            unguarded.append(f)
    verdict(f"those files name the library only behind the macro (outside it: {unguarded or 'none'})", not unguarded)
    return ok


def check_layout() -> bool:
    ok = True

    def verdict(what: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  {what}: {cond}")

    new = (ROOT / "sycl/CMakeLists.txt").read_text(encoding="utf-8")
    banner = DENSE_BANNER.search(new)
    head, tail = (new[:banner.start()], new[banner.start():]) if banner else (new, "")
    code_tail = re.sub(r"#[^\n]*", "", tail)
    code_head = re.sub(r"#[^\n]*", "", head)
    print("\nthe merged layout:")
    verdict("the dense section exists and runs to the end of the file", bool(banner))
    verdict("no strata-q36 target (add_executable / target_* / a build line) is left", "strata-q36" not in code_head + code_tail)
    verdict("no src/qwen35 path is named before the dense section", "qwen35/" not in code_head)
    sources = re.search(r"target_sources\(strata PRIVATE(.*?)\)", code_tail, re.S)
    listed = sources.group(1) if sources else ""
    verdict("the dense sources are listed once as sources of `strata`: " + ", ".join(DENSE_SOURCES),
            all(listed.count(s) == 1 for s in DENSE_SOURCES))
    verdict("they are sources of no other target (the PARITY=ON tool q36_mmvq_dense_parity also lists the two dense decode forms)",
            code_tail.count("qwen35/q36_engine.dp.cpp") == 1 and code_tail.count("qwen35/q36_requant.cpp") == 1
            and code_tail.count("qwen35/q36_mmvq_dense.dp.cpp") == 2 and code_tail.count("qwen35/q36_xr_dense.dp.cpp") == 2)
    verdict("STRATA_Q36_BUILD is defined once, for every target (add_compile_definitions), inside the dense section",
            len(re.findall(r"STRATA_Q36_BUILD", code_head + code_tail)) == 1
            and re.search(r"add_compile_definitions\(STRATA_Q36_BUILD=1\)", code_tail) is not None)
    verdict("the dense family is a switch: option(STRATA_BUILD_Q36 ...) and every dense line inside if(STRATA_BUILD_Q36)",
            "option(STRATA_BUILD_Q36" in code_tail and re.search(r"^if\(STRATA_BUILD_Q36\)", code_tail, re.M) is not None
            and "target_sources(strata" in code_tail.split("if(STRATA_BUILD_Q36)", 1)[-1])
    verdict("src/qwen35/q36_main.cpp (the second binary's entry point) is gone", not (ROOT / "src/qwen35/q36_main.cpp").exists())
    for rel in ("sycl/src/qwen35/q36_engine.dp.cpp", "sycl/src/qwen35/q36_mmvq_dense.dp.cpp", "sycl/src/qwen35/q36_xr_dense.dp.cpp",
                "src/qwen35/q36_requant.cpp"):
        verdict(f"{rel} has no main()", not re.search(r"^int\s+main\s*\(", (ROOT / rel).read_text(encoding="utf-8"), re.M))
    users = [f for f in git("grep", "-l", "STRATA_Q36_BUILD", "--", "include", "sycl", "src", "CMakeLists.txt", check=False).split("\n")
             if f and f.startswith(TREES)]
    stray = sorted(set(users) - ALLOWED)
    verdict(f"only the family headers and the family switch mention STRATA_Q36_BUILD (stray: {stray or 'none'})", not stray)

    gen = (ROOT / "sycl/src/program/generate.cpp").read_text(encoding="utf-8")
    m = re.search(r"^int main\(int argc, char \*\*argv\) try \{\n(.*?)\n    // \*\*UNBUFFERED", gen, re.S | re.M)
    body = m.group(1) if m else ""
    code = [ln.strip() for ln in body.split("\n") if ln.strip() and not ln.strip().startswith("//")]
    verdict("the family switch is the first statement of main(), behind the guard, before the stdout mode and the environment",
            bool(m) and code[:1] == ["#if defined(STRATA_Q36_BUILD)"] and code[-1:] == ["#endif"]
            and any("serves_dense(argc, argv)" in ln and "strata::qwen35::run(argc, argv)" in ln for ln in code))
    return ok


def main() -> int:
    if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", BASE + "^{commit}"], capture_output=True).returncode != 0:
        print(f"BASE {BASE} is not a commit of this repository (give one: tools/check_q36_isolation.py <commit>)")
        return 2
    print(f"base: {BASE} = {git('rev-parse', BASE).strip()}")
    a = check_files()
    b = check_cmake()
    c = check_layout()
    d = check_b70_lib()
    print("\nALL HOLD" if a and b and c and d else "\nFAILED")
    return 0 if a and b and c and d else 1


if __name__ == "__main__":
    sys.exit(main())
