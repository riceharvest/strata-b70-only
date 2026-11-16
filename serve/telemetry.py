"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: the Intel Arc (xe) reader, sycl/serve/xe_telemetry.py, which sycl/serve/server_intel.py installs as gpu_reader;
  without it there are no GPU readings.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
import platform
import sys
import threading
import time
from collections.abc import Callable, Sequence

HISTORY = 60


# ------------------------------------------------------------------------------------------------ GPU
class _NoGpu:
    """No GPU readings: the shape of a card reader (ok/name/read) with nothing to read.  The Intel Arc reader
    (sycl/serve/xe_telemetry.py) is installed as gpu_reader by sycl/serve/server_intel.py."""
    note = None

    def __init__(self, index: int = 0) -> None:
        self.index = index

    def ok(self) -> bool:
        return False

    def name(self) -> str | None:
        return None

    def read(self) -> dict:
        return {}


def gpu_reader(index: int = 0):
    """The card's readings; sycl/serve/server_intel.py replaces this with the Intel Arc (xe) reader."""
    return _NoGpu(index)


def free_vram_mib(index: int = 0) -> int | None:
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_topology(root: str = "/sys/devices/system") -> tuple[int | None, int | None]:
    """(sockets, NUMA nodes) of this computer: how many CPU packages it has and how many memory nodes (None where it cannot be read).
    Linux reads /sys, Windows asks GetLogicalProcessorInformationEx / GetNumaHighestNodeNumber, macOS asks sysctl (Apple silicon and every Mac
    sold so far: one package, no NUMA)."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            needed = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformationEx(3, None, ctypes.byref(needed))          # RelationProcessorPackage
            buf = ctypes.create_string_buffer(needed.value)
            sockets = None
            if needed.value and kernel32.GetLogicalProcessorInformationEx(3, buf, ctypes.byref(needed)):
                sockets, offset = 0, 0
                while offset + 8 <= needed.value:
                    relationship = int.from_bytes(buf.raw[offset:offset + 4], "little")
                    size = int.from_bytes(buf.raw[offset + 4:offset + 8], "little")
                    if size <= 0:
                        break
                    sockets += 1 if relationship == 3 else 0
                    offset += size
            high = ctypes.c_ulong(0)
            numa = high.value + 1 if kernel32.GetNumaHighestNodeNumber(ctypes.byref(high)) else None
            return (sockets or None), numa
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.packages"], capture_output=True, text=True, timeout=5).stdout.strip()
            return (int(out) if out.isdigit() else 1), None
        import glob
        packages = set()
        for f in glob.glob(os.path.join(root, "cpu", "cpu[0-9]*", "topology", "physical_package_id")):
            try:
                packages.add(open(f, encoding="utf-8").read().strip())
            except OSError:
                continue
        nodes = len(glob.glob(os.path.join(root, "node", "node[0-9]*")))
        return (len(packages) or None), (nodes or None)
    except Exception:                                        # noqa: BLE001 - a missing figure is shown as missing, never guessed
        return None, None


def _cpu_name() -> str | None:
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self) -> None:
        self.prev = self._times()

    def _times(self) -> tuple[int, int] | None:
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self) -> float | None:
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra: Callable[[], dict] | None = None, gpu_index: int = 0,
                 gpu_indices: Sequence[int] | None = None) -> None:
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as setup numbers them; `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), and "gpus" has each card's own."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, gpu_reader(i)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        gpu_name = " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None
        self.static = {
            "gpu_name": gpu_name,
            "gpu_vram_total_mib": None,
            "gpu_count": len(self.gpus),
            "gpu_note": getattr(self.gpu, "note", None),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "sockets": _cpu_topology()[0],
            "numa_nodes": _cpu_topology()[1],
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _disk(self) -> tuple[float | None, float | None]:
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self) -> dict:
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "name": reader.name(), "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power")}
                             for (i, r), (_, reader) in zip(reads, self.gpus)]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def close(self) -> None:
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self) -> None:
        """One sample a second, for as long as the server lives.

        A failing sample must not end the thread: on a machine where the GPU/psutil source is not readable
        (a CI runner, a container without NVML) the first exception used to kill the sampler outright, so
        `GET /metrics` silently stopped updating and `test_responses.NoLeakedSampler` saw "the server did
        not start its hardware sampler" - the thread had started and died between two readings.
        """
        complained = False
        while not self._stop.is_set():
            try:
                s = self.sample()
            except Exception as e:                    # noqa: BLE001 - a monitor must outlive its source
                if not complained:                    # once, not once a second
                    print(f"[strata] metrics: a sample failed ({e}); the sampler keeps trying", flush=True)
                    complained = True
                self._stop.wait(1.0)
                continue
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self) -> dict:
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
