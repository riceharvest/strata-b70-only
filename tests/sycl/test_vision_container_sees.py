"""Pictures in the engine's container: which embeddings file it can open (sycl/serve/server_intel.py, sycl/serve/strata-sycl.sh)."""
import importlib.util
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _server_intel():
    sys.path.insert(0, str(ROOT / "sycl" / "serve"))
    spec = importlib.util.spec_from_file_location("server_intel_vision_test", ROOT / "sycl" / "serve" / "server_intel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_container_sees_shm_and_the_vision_dir_only():
    mod = _server_intel()
    assert mod.container_sees("/dev/shm/strata-vision-1-req-a.sve", {})
    assert not mod.container_sees("/dev/shm/x.sve", {"STRATA_SYCL_SHM": "0"})
    assert not mod.container_sees("/tmp/strata-vision-ab/req-a.sve", {})
    assert mod.container_sees("/data/vis/strata-vision-ab/req-a.sve", {"STRATA_VISION_DIR": "/data/vis"})
    assert not mod.container_sees("/data/vis2/req-a.sve", {"STRATA_VISION_DIR": "/data/vis"})
    assert mod.container_sees("/data/vis/q.sve", {"STRATA_VISION_DIR": "/data/vis/", "STRATA_SYCL_SHM": "0"})


def test_the_log_line_is_said_once_and_only_for_a_file_the_container_cannot_open(monkeypatch, capsys):
    mod = _server_intel()
    paths = iter(["/dev/shm/a.sve", "/tmp/x/b.sve", "/tmp/x/c.sve"])
    monkeypatch.setattr(mod.S, "combined_embeddings_path", lambda d, n: next(paths))
    monkeypatch.delenv("STRATA_VISION_DIR", raising=False)
    monkeypatch.delenv("STRATA_SYCL_SHM", raising=False)
    mod.install_shared_embeddings()
    for _ in range(3):
        mod.S.combined_embeddings_path(Path("/tmp/x"), 1)
    out = capsys.readouterr().out
    assert out.count("is not where the engine's container can read it") == 1
    assert "/tmp/x/b.sve" in out and "STRATA_VISION_DIR" in out


def _docker_args(tmp_path, env):
    fake = tmp_path / "docker"
    fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$OUT"\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    out = tmp_path / "args.txt"
    e = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "OUT": str(out), "HOME": str(tmp_path), "STRATA_SYCL_SECCOMP": "", **env}
    subprocess.run(["bash", str(ROOT / "sycl" / "serve" / "strata-sycl.sh"), "--max-context", "8"], env=e, check=True,
                   capture_output=True)
    return out.read_text().split("\n")


def test_the_wrapper_mounts_STRATA_VISION_DIR_only_when_set(tmp_path):
    vis = tmp_path / "vis"
    with_dir = _docker_args(tmp_path, {"STRATA_VISION_DIR": str(vis)})
    assert f"{vis}:{vis}" in with_dir and vis.is_dir()
    assert not any(a.endswith(":/nonexistent") for a in with_dir)
    without = _docker_args(tmp_path, {})
    assert not any(str(vis) in a for a in without)
