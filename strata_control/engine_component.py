"""Finding the engine package for this machine in a local component folder. There is no download path.

The engine (the C++ `strata` binary and its GPU runtime libraries) is a separate, independently versioned
component. The desktop app and the terminal start whatever is installed in the component folder; nothing here, and
nothing on the start path, fetches an engine. A missing or unusable engine is a typed error that says what to do
(`EngineMissing`, exit code 5, the same family as `acquisition.AcquisitionRefused`) and never an attempt to get one.

Layout (``components_dir()``)::

    <components>/engine/<version>-<platform>-<backend>/
        engine-component.json        the manifest (below)
        strata.exe | strata          the entrypoint (path from the manifest)
        *.dll | *.so                 the GPU runtime libraries, hashed in the manifest too
        BUILD.json                   optional, the engine's own build record

Manifest (``engine-component.json``, schema 2; schema 1 packages still load). packaging/README.md has the full table::

    {"schema_version": 2, "id": "strata-engine", "name": "strata-engine", "version": "0.1.45",
     "platform": "linux-x86_64", "backend": "sycl", "archs": ["intel-arc"],
     "min_driver": {}, "build_id": "be885026f9-sycl", "min_runtime": "0.1.45",
     "entrypoint": "strata", "files": [{"path": "strata", "sha256": "...", "bytes": 1}, ...],
     "signature": null}                      # S6 fills it; verify_signature() is the hook

Checks, in order, and each one names what failed:

* the manifest parses strictly; file paths are relative, plain, and stay inside the component folder;
* the platform and backend are combinations the project commits to (`components.BACKEND_MATRIX`);
* the backend is the one this machine's GPU needs (detected, or asked for);
* the signature (none yet; a verifier registered by S6 is called, and `STRATA_REQUIRE_SIGNED_COMPONENTS=1` refuses
  an unsigned or unverifiable package);
* every listed file exists with the listed size, and every file's SHA-256 matches (`verify(full=True)`, which is what
  a start does; the quick listing/status path hashes only the entrypoint);
* the runtime that is asking is new enough for `min_runtime`. A package that fails an integrity check is
  `EngineIntegrityError`: nothing is started.

Nothing here imports anything outside the standard library and `strata_control`.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath

from strata_control import acquisition, longpath
from strata_control.targets import BACKEND_MATRIX, backend_supported, host_platform

SCHEMA_VERSION = 2
READABLE_SCHEMAS = (1, 2)           # schema 1 (S3's first packages: no archs/min_driver/build_id/signature) still loads
MANIFEST = "engine-component.json"
COMPONENT_ID = "strata-engine"
ENV_DIR = "STRATA_COMPONENTS_DIR"
ENV_BACKEND = "STRATA_ENGINE_BACKEND"
ENV_REQUIRE_SIGNED = "STRATA_REQUIRE_SIGNED_COMPONENTS"
#: The updater's state (``strata update``, ``strata components rollback``) lives in the component folder; the only member the
#: start path reads is ``engine.pinned``: the folder a manual rollback chose, which wins over a newer installed package.
UPDATE_STATE = Path(".updates") / "state.json"

#: the version of the control/runtime that is asking; the app reads it from VERSION when it has one.
RUNTIME_VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"


class EngineMissing(acquisition.AcquisitionRefused):
    """No usable engine component for this machine. Nothing was fetched and nothing will be."""

    code = "engine_component_missing"
    exit_code = 5

    def __init__(self, reason: str, root: Path | None = None, *, problems: list[dict] | None = None, notice: str | None = None):
        self.reason = reason
        self.root = root
        self.problems = problems or []
        self.plain_notice = notice
        where = f" in {root}" if root else ""
        RuntimeError.__init__(
            self,
            (f"{notice} (" if notice else "") + f"No engine for this PC was found{where}: {reason}" + (")" if notice else "") + ". The app never downloads an engine while it "
            "starts or serves. Download the engine for your GPU with Update engine in the app's Settings, or with "
            "`strata update` (or `strata install`) in the Terminal, or copy an engine package into the component folder.")
        self.what = "the engine component"
        self.command = "strata install"


class EngineIntegrityError(EngineMissing):
    """A package is installed but is not what its manifest says (a changed or missing file, a bad signature). Nothing
    was started. The message names the file; the fix is to repair or reinstall the component, never to run it anyway."""

    code = "engine_component_integrity"

    def __init__(self, reason: str, root: Path | None = None, *, problems: list[dict] | None = None):
        self.reason = reason
        self.root = root
        self.problems = problems or []
        RuntimeError.__init__(
            self,
            f"The engine package failed its integrity check, so it was not started: {reason}. Repair or reinstall "
            f"the engine component with Update engine in the app's Settings or `strata update` in the Terminal "
            f"(`strata install` also works); the app never downloads one while it runs.")
        self.what = "the engine component"
        self.command = "strata install"


class ComponentProblem(ValueError):
    """One engine package is unusable. The message says why; the caller decides what to do next."""


class IntegrityProblem(ComponentProblem):
    """The package's files or signature do not match its manifest (as opposed to a package for another PC)."""


@dataclass
class EngineComponent:
    id: str
    version: str
    platform: str
    backend: str
    entrypoint: str
    files: tuple[tuple[str, str, int], ...]      # (relative path, sha256, bytes)
    root: Path
    min_runtime: str | None = None
    archs: tuple[str, ...] = ()
    notes: str = ""
    min_driver: dict = field(default_factory=dict)       # {vendor: version}: the oldest driver the build runs on (read, not checked)
    build_id: str = ""                                    # which engine build this is (source commit + toolkit)
    variant: str = ""                                     # optional tag when one backend has several builds
    signature: dict | None = None                         # S6: {"alg", "key_id", "value"} over signing_payload()
    raw: dict = field(default_factory=dict, repr=False)   # the manifest as read, for the signature check

    @property
    def exe(self) -> Path:
        return self.root / self.entrypoint

    def to_dict(self) -> dict:
        return {"id": self.id, "version": self.version, "platform": self.platform, "backend": self.backend,
                "entrypoint": self.entrypoint, "exe": str(self.exe), "root": str(self.root),
                "min_runtime": self.min_runtime, "archs": list(self.archs), "min_driver": dict(self.min_driver),
                "build_id": self.build_id, "variant": self.variant, "signed": bool(self.signature), "file_count": len(self.files), "bytes": sum(item[2] for item in self.files)}


def package_lib_dirs(folder) -> list[str]:
    """The folders an engine package needs on the library path (``PATH`` on Windows, ``LD_LIBRARY_PATH`` elsewhere): its own folder, then
    the ones its BUILD.json names in ``lib_dirs`` (relative). Only folders that exist inside the package."""
    folder = Path(folder)
    dirs = [str(folder)]
    try:
        names = json.loads((folder / "BUILD.json").read_text(encoding="utf-8-sig")).get("lib_dirs") or []
    except (OSError, ValueError, AttributeError):
        names = []
    for name in names:
        if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
            continue
        path = folder / name
        if path.is_dir() and str(path) not in dirs:
            dirs.append(str(path))
    return dirs


# ----------------------------------------------------------------------------------------------- where

def components_dir(environ=None, *, windows: bool | None = None, home: Path | None = None) -> Path:
    """The per-user component folder. ``STRATA_COMPONENTS_DIR`` wins (the app and the tests set it)."""
    environ = os.environ if environ is None else environ
    explicit = environ.get(ENV_DIR)
    if explicit:
        return Path(explicit)
    windows = os.name == "nt" if windows is None else windows
    home = Path.home() if home is None else home
    if windows:
        return Path(environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")) / "Strata" / "components"
    return Path(environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")) / "strata" / "components"


def engine_dir(root: Path | None = None) -> Path:
    return (root or components_dir()) / "engine"


def newest_installed_version(root: Path | None = None) -> str | None:
    """The highest engine version in the component folder, or None. A quick scan; nothing is hashed."""
    try:
        found, _ = scan(root)
    except Exception:                                   # noqa: BLE001 - informational only
        return None
    return max((item.version for item in found), key=_version_key, default=None)


def cached_notice(root: Path | None = None) -> str | None:
    """"update available (x.y.z): run `strata update`" from the last explicit check, or None. Reads one small local file; it never
    touches the network, so showing it at a start does not break the rule that only an update step downloads.

    The notice belongs to the versions the check was made for: it is dropped when Control or the installed engine is not the one
    recorded then (an app install, an update, a rollback), and an engine that is not newer than the installed one is never offered."""
    base = root or components_dir()
    try:
        last = (json.loads((base / UPDATE_STATE).read_text(encoding="utf-8-sig")).get("last_check") or {})
    except (OSError, ValueError, AttributeError):
        return None
    if not isinstance(last, dict):
        return None
    control = runtime_version()
    if "control" in last and last["control"] != control:
        return None
    engine, app = last.get("engine"), last.get("app")
    installed = newest_installed_version(base)
    if "engine_installed" in last and last["engine_installed"] != installed:
        engine = None
    if engine and installed and _version_key(str(engine)) <= _version_key(installed):
        engine = None
    parts = ([f"engine {engine}"] if engine else []) + ([f"app {app}"] if app else [])
    return f"update available ({', '.join(parts)}): run `strata update`" if parts else None


def previous_folder(root: Path | None = None) -> str | None:
    """The package folder the last update kept as the version to roll back to, if it is still installed."""
    base = root or components_dir()
    try:
        value = json.loads((base / UPDATE_STATE).read_text(encoding="utf-8-sig"))["engine"]["previous"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if isinstance(value, str) and value and "/" not in value and "\\" not in value and (engine_dir(base) / value / MANIFEST).is_file():
        return value
    return None


def pinned_folder(root: Path | None = None) -> str | None:
    """The package folder a manual rollback pinned, or None. Read-only; a missing or damaged state file means no pin."""
    try:
        data = json.loads(((root or components_dir()) / UPDATE_STATE).read_text(encoding="utf-8-sig"))
        value = data["engine"]["pinned"]
        return value if isinstance(value, str) and value and "/" not in value and "\\" not in value else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ----------------------------------------------------------------------------------------------- manifest

def _version_key(value: str) -> tuple:
    return tuple(int(part) if part.isdigit() else 0 for part in re.split(r"[.\-+]", value))


def _plain_relative(path: str) -> str:
    if not isinstance(path, str) or not path or "\0" in path:
        raise ComponentProblem("a file path is empty")
    for flavour in (PurePosixPath(path), PureWindowsPath(path)):
        if flavour.is_absolute() or flavour.drive or flavour.root:
            raise ComponentProblem(f"file path {path!r} is absolute")
        if ".." in flavour.parts:
            raise ComponentProblem(f"file path {path!r} leaves the component folder")
    return path.replace("\\", "/")


def read_manifest(folder: Path) -> EngineComponent:
    path = folder / MANIFEST
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise ComponentProblem(f"{MANIFEST} is missing") from None
    except (OSError, ValueError) as exc:
        raise ComponentProblem(f"{MANIFEST} is not valid JSON ({type(exc).__name__})") from None
    if not isinstance(data, dict) or data.get("schema_version") not in READABLE_SCHEMAS:
        raise ComponentProblem(f"{MANIFEST} has an unsupported schema_version")
    schema = data["schema_version"]
    if data.get("id") != COMPONENT_ID or data.get("name", COMPONENT_ID) != COMPONENT_ID:
        raise ComponentProblem(f"{MANIFEST} is not an engine component (id {data.get('id')!r})")
    for key in ("version", "platform", "backend", "entrypoint"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise ComponentProblem(f"{MANIFEST} has no {key}")
    if schema >= 2 and (not isinstance(data.get("build_id"), str) or not data["build_id"].strip()):
        raise ComponentProblem(f"{MANIFEST} has no build_id")
    if data["backend"] not in BACKEND_MATRIX:
        raise ComponentProblem(f"unknown backend {data['backend']!r}")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ComponentProblem(f"{MANIFEST} lists no files")
    parsed = []
    for item in files:
        if not isinstance(item, dict):
            raise ComponentProblem("a file entry is not an object")
        rel = _plain_relative(item.get("path"))
        digest = str(item.get("sha256", "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ComponentProblem(f"{rel} has no valid sha256")
        size = item.get("bytes")
        if type(size) is not int or size < 0:
            raise ComponentProblem(f"{rel} has no valid size")
        parsed.append((rel, digest, size))
    entry = _plain_relative(data["entrypoint"])
    if entry not in {item[0] for item in parsed}:
        raise ComponentProblem(f"the entrypoint {entry} is not in the file list")
    if len({item[0] for item in parsed}) != len(parsed):
        raise ComponentProblem(f"{MANIFEST} lists a file twice")
    archs = data.get("archs") or []
    if not isinstance(archs, list) or not all(isinstance(a, str) and a for a in archs):
        raise ComponentProblem("archs must be a list of strings")
    if schema >= 2 and data["backend"] != "cpu" and not archs:
        raise ComponentProblem(f"{MANIFEST} lists no archs for a {data['backend']} engine")
    driver = data.get("min_driver") or {}
    if not isinstance(driver, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in driver.items()):
        raise ComponentProblem("min_driver must map a vendor to a version string")
    signature = data.get("signature")
    if signature is not None and not (isinstance(signature, dict) and isinstance(signature.get("alg"), str)
                                      and isinstance(signature.get("value"), str)):
        raise ComponentProblem("signature must be null or {alg, key_id, value}")
    return EngineComponent(
        id=data["id"], version=data["version"], platform=data["platform"], backend=data["backend"],
        entrypoint=entry, files=tuple(parsed), root=folder, min_runtime=data.get("min_runtime"),
        archs=tuple(archs), notes=str(data.get("notes") or ""), min_driver=dict(driver),
        build_id=str(data.get("build_id") or ""), variant=str(data.get("variant") or ""), signature=signature,
        raw=data)


# ----------------------------------------------------------------------------------------------- signature (S6)

#: A signature verifier is ``fn(payload: bytes, signature: dict, component) -> None``; it raises
#: ``IntegrityProblem`` (or any exception) when the signature does not hold. S6 registers one that checks the project's
#: key; until then there is none, unsigned packages load (unless ``STRATA_REQUIRE_SIGNED_COMPONENTS=1``), and a package
#: that carries a signature nobody can check is reported as "signature not verified", never as verified.
_verifier: Callable[[bytes, dict, "EngineComponent"], None] | None = None


def register_signature_verifier(fn: Callable[[bytes, dict, "EngineComponent"], None] | None) -> None:
    global _verifier
    _verifier = fn


def signing_payload(manifest: dict) -> bytes:
    """The bytes a signature covers: the manifest without its ``signature`` member, as canonical JSON (sorted keys,
    no spaces, UTF-8). The file hashes are inside it, so one signature covers every file."""
    body = {key: value for key, value in manifest.items() if key != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify_signature(component: "EngineComponent", environ=None) -> str:
    """"verified", "unsigned" or "unverified" (signed, no verifier yet). Raises IntegrityProblem for a bad signature,
    and for an unsigned or unverifiable package when ``STRATA_REQUIRE_SIGNED_COMPONENTS`` is set."""
    environ = os.environ if environ is None else environ
    required = environ.get(ENV_REQUIRE_SIGNED, "").strip().lower() in ("1", "true", "yes", "on")
    if not component.signature:
        if required:
            raise IntegrityProblem("the package is not signed and this PC requires signed components")
        return "unsigned"
    if _verifier is None:
        # the keys this build trusts (update_keys.json, plus the explicit test-key file) become the verifier on first use, so
        # a signed package is verified on every start without anything having to register it first
        from strata_control import signing
        try:
            signing.ensure_component_verifier()
        except signing.SignatureError as exc:
            raise IntegrityProblem(f"this PC's update keys cannot be read ({exc})") from None
    if _verifier is None:
        if required:
            raise IntegrityProblem("the package is signed but this build cannot check signatures yet")
        return "unverified"
    try:
        _verifier(signing_payload(component.raw), component.signature, component)
    except IntegrityProblem:
        raise
    except Exception as exc:
        raise IntegrityProblem(f"the signature does not verify ({type(exc).__name__}: {exc})") from None
    return "verified"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_settled(path: Path, digest: str) -> str | None:
    """The file's SHA-256, read again once when the first read fails or differs: a file that was just unpacked or is
    being scanned by a virus checker can read short or locked for a moment, and a failure must not be kept (#3488)."""
    actual = None
    for attempt in range(2):
        try:
            actual = _sha256(path)
        except OSError:
            actual = None
        if actual == digest:
            break
        if attempt == 0:
            time.sleep(0.5)
    return actual


def verify(component: EngineComponent, *, full: bool = False) -> None:
    """Raise IntegrityProblem unless the files are what the manifest says. `full` hashes every file; without it only
    the entrypoint is hashed (sizes are always checked)."""
    # extended-length paths: under a long user name an engine's files can pass 259 characters, which Windows (long paths off)
    # reports as "missing" for an ordinary path
    folder = longpath.long_path(component.root)
    root = folder.resolve()
    for rel, digest, size in component.files:
        target = folder / rel
        try:
            info = target.lstat()
        except OSError:
            raise IntegrityProblem(f"{rel} is missing") from None
        if target.is_symlink() or not target.is_file():
            raise IntegrityProblem(f"{rel} is not a regular file")
        if not longpath.same_tree(target.resolve(), root):
            raise IntegrityProblem(f"{rel} resolves outside the component folder")
        if info.st_size != size:
            raise IntegrityProblem(f"{rel} is {info.st_size} bytes, the manifest says {size}")
        if full or rel == component.entrypoint:
            actual = _hash_settled(target, digest)
            if actual != digest:
                raise IntegrityProblem(f"{rel} does not match its recorded SHA-256 (expected {digest[:12]}, "
                                       f"found {actual[:12] if actual else 'unreadable'})")
    if os.name != "nt" and not os.access(component.exe, os.X_OK):
        raise IntegrityProblem(f"{component.entrypoint} is not executable")


def runtime_version() -> str | None:
    try:
        return RUNTIME_VERSION_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


# ----------------------------------------------------------------------------------------------- the GPU

def split_backend(value) -> tuple[str, str]:
    """(backend, variant) for a name the person typed: "hip-rdna2" is the hip family and its rdna2 package (the way the feed labels an AMD
    package, #4013), "hip" is the family alone, any other name is returned as it is with no variant."""
    name = str(value or "").strip().lower()
    if name not in BACKEND_MATRIX and name.startswith("hip-") and len(name) > 4:
        return "hip", name[4:]
    return name, ""


def detect_backend(environ=None) -> tuple[str | None, str]:
    """(backend, how) for this machine: sycl when an Intel GPU is there. Bounded, read-only probes;
    ``STRATA_ENGINE_BACKEND`` overrides (the tests and the support desk use it)."""
    environ = os.environ if environ is None else environ
    forced = split_backend(environ.get(ENV_BACKEND, ""))[0]      # "hip-rdna2" starts the hip family (the package choice reads the variant)
    if forced:
        if forced not in BACKEND_MATRIX:
            raise ComponentProblem(f"{ENV_BACKEND}={forced!r} is not one of {', '.join(sorted(BACKEND_MATRIX))}")
        return forced, "environment"
    tool = shutil.which("sycl-ls")
    if tool:
        try:
            done = subprocess.run([tool], capture_output=True, timeout=8, text=True)
        except (OSError, subprocess.SubprocessError):
            done = None
        if done is not None and done.returncode == 0 and ("level_zero" in done.stdout.lower() or "gpu" in done.stdout.lower()):
            return "sycl", "sycl-ls"
    found = _drm_vendor_backend() if os.name != "nt" else None
    if found:
        return found, "sysfs pci vendor"
    return None, "no GPU tool answered"


def _drm_vendors(root: str) -> set[str]:
    vendors = set()
    try:
        for card in sorted(Path(root).glob("card[0-9]*")):
            if "-" in card.name:
                continue
            try:
                vendors.add((card / "device" / "vendor").read_text().strip().lower())
            except OSError:
                continue
    except OSError:
        return set()
    return vendors


def _drm_vendor_backend(root: str = "/sys/class/drm") -> str | None:
    """Linux without sycl-ls on PATH (the engine runs in a container): an Intel display adapter (PCI vendor 0x8086)
    means sycl. Read-only sysfs, nothing is run."""
    if not os.path.isdir(root):
        return None
    return "sycl" if "0x8086" in _drm_vendors(root) else None


def arch_native(archs, present: list[str]) -> bool:
    """Whether a build with `archs` holds machine code made for one of the `present` cards itself (``sm_89`` on an ``sm_89`` card),
    not only code that runs there through the same-major rule of `arch_covers` (``sm_86`` on an RTX 40) or as PTX. Only a tie-breaker
    between packages that fit equally: a package with the card's own image is taken first (#3935)."""
    cards = {item[3:] for item in present if item.startswith("sm_") and item[3:].isdigit()}
    return any(arch.startswith("sm_") and arch[3:] in cards for arch in archs)


# ----------------------------------------------------------------------------------------------- resolve

def scan(root: Path | None = None) -> tuple[list[EngineComponent], list[dict]]:
    """Every readable engine package in the folder, and the ones that were not (with the reason)."""
    folder = engine_dir(root)
    found, problems = [], []
    try:
        entries = sorted(item for item in folder.iterdir() if item.is_dir() and not item.name.startswith("."))
    except OSError:
        return [], []
    for entry in entries[:64]:
        try:
            found.append(read_manifest(entry))
        except ComponentProblem as exc:
            problems.append({"folder": str(entry), "problem": str(exc)})
    return found, problems


def no_package_note(archs) -> str | None:
    """One plain sentence when some of these architectures have no ready-made package in this release. Every card this
    fork runs on (the Intel Arc) has one, so: None."""
    return None


def resolve(root: Path | None = None, *, backend: str | None = None, target: str | None = None,
            full: bool = False, runtime: str | None = None) -> EngineComponent:
    """The engine to start on this machine, or `EngineMissing` / `EngineIntegrityError`. Never downloads, never creates
    the folder. `full=True` hashes every file (a start does); the default is the quick check the listings use."""
    base = root or components_dir()
    target = target or host_platform()
    found, problems = scan(base)
    if not found and not problems:
        raise EngineMissing("the component folder holds no engine package", base)
    wanted = backend
    how = "requested"
    if wanted is None:
        wanted, how = detect_backend()
    if wanted is None:
        # no GPU tool answered: only a CPU engine can be started
        wanted = "cpu"
    usable, reasons, broken = [], [], []
    have_runtime = runtime if runtime is not None else runtime_version()
    for item in found:
        label = f"{item.version} {item.platform}-{item.backend}"
        if item.platform != target:
            reasons.append(f"{label} is for {item.platform}, this PC is {target}")
            continue
        ok, why = backend_supported(item.backend, target)
        if not ok:
            reasons.append(f"{label}: {why}")
            continue
        if item.backend != wanted:
            reasons.append(f"{label} is a {item.backend} engine, this PC needs {wanted} ({how})")
            continue
        if item.min_runtime and have_runtime and _version_key(have_runtime) < _version_key(item.min_runtime):
            reasons.append(f"{label} needs control {item.min_runtime} or newer, this is {have_runtime}")
            continue
        try:
            verify_signature(item)
            verify(item, full=full)
        except ComponentProblem as exc:
            reasons.append(f"{label}: {exc}")
            broken.append(f"{label}: {exc}")
            continue
        usable.append(item)
    if not usable:
        everything = reasons + [p["folder"] + ": " + p["problem"] for p in problems]
        if broken and len(broken) == len(reasons):       # every package that fits this PC failed its integrity check
            raise EngineIntegrityError("; ".join(broken), base, problems=problems)
        raise EngineMissing("; ".join(everything) or "nothing matched", base, problems=problems)
    pin = pinned_folder(base)
    if pin:                                   # a manual rollback: the pinned package wins while it is usable
        for item in usable:
            if item.root.name == pin:
                return item
    return max(usable, key=lambda item: _version_key(item.version))      # newest version first


#: A run config can name its engine as "component:strata-engine" instead of a path: the server then asks the
#: component folder for the engine that fits this PC when it starts (a config written before an engine update keeps
#: working after it, and a config copied to another PC finds that PC's engine).
REFERENCE_PREFIX = "component:"


def is_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith(REFERENCE_PREFIX)


def expand(value: str, cfg: dict | None = None, *, root: Path | None = None, full: bool = True) -> str:
    """The engine path for `value`: unchanged when it is a path, resolved when it is a component reference. This is the
    start path, so every file's SHA-256 is checked (`full`): a changed file refuses the start instead of running."""
    if not is_reference(value):
        return value
    name = value[len(REFERENCE_PREFIX):]
    if name not in ("", COMPONENT_ID):
        raise EngineMissing(f"the run config names an unknown component {name!r}", root or components_dir())
    backend = (cfg or {}).get("backend")
    return str(resolve(root, backend=backend if backend in BACKEND_MATRIX else None, full=full).exe)


def status(root: Path | None = None, *, backend: str | None = None, full: bool = False) -> dict:
    """What `strata components engine` and the app's host report. Always answers; never raises for a missing engine."""
    base = root or components_dir()
    found, problems = scan(base)
    try:
        detected, how = detect_backend()
    except ComponentProblem as exc:
        detected, how = None, str(exc)
    report = {"schema_version": 1, "components_dir": str(base), "platform": host_platform(),
              "backend": backend or detected, "backend_source": "requested" if backend else how,
              "installed": [item.to_dict() for item in found], "problems": problems,
              "engine": None, "usable": False, "notice": None, "acquires": False,
              "pinned": pinned_folder(base), "previous": previous_folder(base), "update_notice": cached_notice(base)}
    try:
        chosen = resolve(base, backend=backend, full=full)
        report["engine"] = chosen.to_dict()
        report["usable"] = True
    except EngineMissing as exc:
        report["notice"] = getattr(exc, "plain_notice", None) or str(exc)
    return report
