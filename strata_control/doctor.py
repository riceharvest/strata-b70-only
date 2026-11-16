"""Read-only inventory of the components a saved run config needs."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from importlib import metadata, util
from pathlib import Path

from strata_control.profiles import read_run_config


PATH_FLAGS = {
    "--pack": "directory",
    "--native": "file",
    "--ple-gguf": "file",
    "--expert-profile": "file",
    "--mtp": "directory",
    "--control-vector-scaled": "file_with_scale",
}
MONITOR_TEXTUAL_VERSION = "8.2.8"


def _location(raw: str, cfg: dict, config_path: Path) -> Path:
    from strata_control import engine_component
    if engine_component.is_reference(raw):
        try:
            return Path(engine_component.expand(raw, cfg, full=False))
        except engine_component.EngineMissing:
            return engine_component.engine_dir() / "(no engine for this PC)"     # reported as missing
    value = Path(raw).expanduser()
    if value.is_absolute():
        return value
    cwd = cfg.get("cwd")
    return (Path(cwd) if isinstance(cwd, str) and cwd else config_path.parent) / value


def _item(name: str, path: Path, kind: str, *, required: bool = True) -> dict:
    try:
        if kind == "file":
            valid = path.is_file() and path.stat().st_size > 0
        else:
            valid = path.is_dir()
    except OSError:
        valid = False
    return {"name": name, "status": "ok" if valid else ("missing" if required else "optional_missing"),
            "path": str(path), "kind": kind, "required": required}


def _selected_indices(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts: object = value.split(",")
    elif isinstance(value, list):
        parts = value
    else:
        parts = [value]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 64:
        raise ValueError("GPU selection must contain 1..64 numeric device indices")
    result = []
    for part in parts:
        if type(part) is int and 0 <= part <= 1023:
            result.append(part)
        elif isinstance(part, str) and re.fullmatch(r"[0-9]{1,4}", part.strip()) and int(part) <= 1023:
            result.append(int(part))
        else:
            raise ValueError("GPU selection must contain numeric device indices")
    if len(set(result)) != len(result):
        raise ValueError("GPU selection repeats a device index")
    return result


def _live_probe(cfg: dict, backend: str) -> tuple[dict, list[dict]]:
    try:
        selected = _selected_indices(cfg.get("gpu"))
    except ValueError as exc:
        return ({"status": "invalid_selection", "selected_indices": None, "cards": []},
                [{"code": "GPU_SELECTION_INVALID", "severity": "error", "message": str(exc)}])
    if backend == "cpu" and selected is None:
        return {"status": "not_applicable", "selected_indices": None, "cards": []}, []
    return ({"status": "unverified", "selected_indices": selected, "cards": [],
             "reason": f"{backend} device numbering needs a qualified live probe"},
            [{"code": "LIVE_GPU_PROBE_UNAVAILABLE", "severity": "error", "message":
              f"{backend} device/driver verification is not implemented for this command"}])


def inspect(config_path: Path | None, *, profile_id: str | None = None, live: bool = False) -> dict:
    report = {"schema_version": 1, "profile": profile_id, "source_config": str(config_path) if config_path else None,
              "components": [], "issues": [], "backend": None, "devices_selected": None,
              "live_gpu_probe": None}
    components: list[dict] = report["components"]
    if config_path is None:
        report["issues"].append({"code": "PROFILE_NOT_SELECTED", "severity": "info",
                                 "message": "select a run config or saved profile for model-specific checks"})
        components.append({"name": "terminal_cli", "status": "ok", "kind": "python_module", "required": True})
        components.append(_monitor_component())
        if live:
            report["issues"].append({"code": "PROFILE_NOT_SELECTED_FOR_LIVE_PROBE", "severity": "error",
                                     "message": "select a profile to compare its chosen devices with the host"})
        report["ok"] = not live
        return report

    cfg = read_run_config(config_path)
    report["backend"] = cfg.get("backend") or "sycl"
    report["devices_selected"] = cfg.get("gpu")
    exe = _location(cfg["exe"], cfg, config_path)
    components.append(_item("engine", exe, "file"))
    if components[-1]["status"] == "ok" and os.name != "nt" and not os.access(exe, os.X_OK):
        components[-1]["status"] = "invalid"
        components[-1]["reason"] = "engine file is not executable"
    build = exe.parent / "BUILD.json"
    if build.is_file():
        try:
            value = json.loads(build.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                components[-1]["version"] = value.get("version")
        except (OSError, ValueError):
            components[-1]["build_metadata"] = "invalid"
    else:
        components[-1]["build_metadata"] = "unavailable"

    tokenizer = _location(cfg["tokenizer"], cfg, config_path)
    for child in ("vocab.json", "merges.txt", "token_type.json"):
        components.append(_item(f"tokenizer/{child}", tokenizer / child, "file"))

    args = cfg["args"]
    for index, arg in enumerate(args):
        kind = PATH_FLAGS.get(arg)
        if kind is None:
            continue
        if index + 1 >= len(args):
            components.append({"name": arg, "status": "invalid", "kind": kind,
                               "required": True, "reason": "missing argument value"})
            continue
        raw = args[index + 1]
        if arg == "--mtp" and str(raw).isdigit():
            continue                                    # Qwen3.6: --mtp 2 is the draft depth, not a folder (#3298)
        if kind == "file_with_scale":
            candidate, separator, scale = raw.rpartition(":")
            if separator:
                try:
                    float(scale)
                    raw = candidate
                except ValueError:
                    pass
            kind = "file"
        components.append(_item(arg.lstrip("-"), _location(raw, cfg, config_path), kind))

    vision = cfg.get("vision")
    if isinstance(vision, dict):
        for key in ("exe", "mmproj", "model"):
            raw = vision.get(key)
            if isinstance(raw, str) and raw:
                components.append(_item(f"vision/{key}", _location(raw, cfg, config_path), "file"))
            else:
                components.append({"name": f"vision/{key}", "status": "invalid", "kind": "file",
                                   "required": True, "reason": "missing path"})

    components.append(_monitor_component())
    components.append({"name": "gpu_driver_probe", "status": "unverified", "kind": "external_tool",
                       "required": False, "tool": shutil.which("sycl-ls"),
                       "reason": "device/driver compatibility needs a live probe on the target host"})
    for item in components:
        if item["status"] in ("missing", "invalid") and item["required"]:
            report["issues"].append({"code": "COMPONENT_" + item["status"].upper(), "severity": "error",
                                     "component": item["name"], "message": item.get("reason") or "required component unavailable"})
    access = _device_access_issue(cfg.get("backend"))
    if access:
        report["issues"].append(access)
    cpu_pack = _cpu_pack_issue(cfg, config_path)
    if cpu_pack:
        report["issues"].append(cpu_pack)
    if live:
        report["live_gpu_probe"], live_issues = _live_probe(cfg, report["backend"])
        report["issues"].extend(live_issues)
    report["ok"] = not any(issue["severity"] == "error" for issue in report["issues"])
    report["integrity"] = "unverified_no_manifest_hashes"
    return report


def _device_access_issue(backend: str | None, dev_root: str = "/dev") -> dict | None:
    """Linux Intel (SYCL) opens the card through /dev/dri/renderD*: a user outside the `render`
    group gets "No device of requested type available" from the engine, which names nothing to fix. None when it is fine,
    cannot apply (other OS, other backend) or there is no such device node at all."""
    if backend != "sycl" or not sys.platform.startswith("linux"):
        return None
    nodes = sorted(Path(dev_root, "dri").glob("renderD*"))
    if not nodes or any(os.access(node, os.R_OK | os.W_OK) for node in nodes):
        return None
    return {"code": "GPU_DEVICE_ACCESS", "severity": "warning",
            "message": f"this user cannot open {', '.join(str(n) for n in nodes)}; the {backend} engine will not see the card. "
                       "Add the user to the `render` group (and `video` for some distributions), then log in again"}


AVX512_FULL = {"avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx512vbmi"}   # what the Q2_0 pack's CPU kernel uses


def _cpu_flags(cpuinfo: str = "/proc/cpuinfo") -> set[str] | None:
    """The CPU feature flags on Linux; None where they cannot be read (other OS, no /proc)."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        for line in Path(cpuinfo).read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("flags"):
                return set(line.partition(":")[2].split())
    except OSError:
        return None
    return None


def _cpu_pack_issue(cfg: dict, config_path: Path, cpuinfo: str = "/proc/cpuinfo") -> dict | None:
    """The canonical Q2_0 pack (a --pack folder with index.txt and no native_experts.txt) has an AVX-512 CPU kernel; the engine stops at
    start on any other CPU with "this CPU cannot run the expert kernel". An i-quant, NVFP4 or Unsloth pack (native_experts.txt) runs on AVX2
    CPUs. None when it is fine, cannot apply (a pack of another engine, a CPU whose flags cannot be read) or there is no such pack."""
    args = cfg.get("args") or []
    if "--pack" not in args or args.index("--pack") + 1 >= len(args):
        return None
    pack = _location(args[args.index("--pack") + 1], cfg, config_path)
    if not (pack / "index.txt").is_file() or (pack / "native_experts.txt").exists() or (pack / "ds4.txt").exists():
        return None
    flags = _cpu_flags(cpuinfo)
    if flags is None or AVX512_FULL <= flags:
        return None
    return {"code": "CPU_PACK_NEEDS_AVX512", "severity": "warning",
            "message": f"{pack} is the pack for CPUs with AVX-512 (it has no native_experts.txt), and this CPU has no AVX-512 (Ryzen 3000-5000 and "
                       "Intel Core 12th-14th generation do not): the engine will stop at start. Run `strata setup` again; on this CPU it builds "
                       "the pack that runs on AVX2"}


def _monitor_component() -> dict:
    component = {"name": "monitor_tui", "kind": "python_module", "required": False,
                 "expected_version": MONITOR_TEXTUAL_VERSION}
    try:
        found = util.find_spec("textual")
    except (ImportError, ValueError):
        found = None
    if found is None:
        return {**component, "status": "optional_missing",
                "reason": "install the optional monitor dependency from requirements-monitor.txt"}
    try:
        version = metadata.version("textual")
    except metadata.PackageNotFoundError:
        return {**component, "status": "unverified",
                "reason": "Textual module is present but its installed version is unavailable"}
    if version != MONITOR_TEXTUAL_VERSION:
        return {**component, "status": "unverified", "version": version,
                "reason": f"interactive monitor was qualified with Textual {MONITOR_TEXTUAL_VERSION}"}
    return {**component, "status": "ok", "version": version,
            "check": "package_version_only"}
