"""Low-impact execution helpers for the long CPU-bound experiment scripts.

None of this changes modelling behaviour. It only reduces the heat a run
produces, which matters because this machine is a thin Ryzen 5 7530U laptop
(6 cores / 12 threads) that reached ~96 C during a previous blocking run.

Four independent controls, cheapest first:

1. ``cap_blas_threads`` - numpy/BLAS otherwise fan out to all 12 logical
   processors for the logistic-regression fit, which is the sharpest thermal
   spike in the whole experiment. Must be called *before* numpy is imported.
2. ``set_low_priority`` / ``pin_to_cpus`` - put the process at below-normal
   scheduling priority and on a handful of logical processors, so no library
   can light up the whole package.
3. ``ThermalGuard`` - thins the CPU-bound loops out with periodic sleeps, and
   pauses the run when a background temperature sampler reports the CPU is too
   hot. Pausing keeps partial work on disk; it does not throw it away.
4. Nothing here restarts a stage. Progress safety comes from the caller's
   on-disk checkpoints.

A real limitation worth stating: the guard can only act between Python-level
checkpoints. It cannot interrupt a single long C call, so during the
``LogisticRegression.fit`` call the only protection is the thread cap and the
CPU pinning. That is why those two are the controls that matter most there.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

DEFAULT_PAUSE_ABOVE_C = 82.0
DEFAULT_RESUME_BELOW_C = 72.0
DEFAULT_ABORT_ABOVE_C = 93.0
DEFAULT_SAMPLE_SECONDS = 20
MAX_PAUSE_SECONDS = 20 * 60


class ThermalAbort(RuntimeError):
    """Raised when the CPU stays too hot for too long."""


# --------------------------------------------------------------------------
# 1. BLAS thread cap. Call before importing numpy.
# --------------------------------------------------------------------------
def cap_blas_threads(n: int = 1) -> None:
    """Stop BLAS/OpenMP from using every core. Must precede ``import numpy``."""
    value = str(int(n))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS"):
        os.environ[var] = value


def preparse_blas_threads(argv: list[str], default: int = 1) -> int:
    """Read ``--threads N`` out of argv before numpy exists, then cap BLAS."""
    for i, arg in enumerate(argv):
        if arg == "--threads" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                break
        if arg.startswith("--threads="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                break
    return default


# --------------------------------------------------------------------------
# 2. Priority and CPU pinning
# --------------------------------------------------------------------------
def _kernel32():
    """kernel32 with explicit signatures.

    Without these, ``GetCurrentProcess`` returns the pseudo-handle -1, which
    gets truncated when passed as a default-width int, and both calls below
    fail silently while reporting success-looking defaults.
    """
    dll = ctypes.WinDLL("kernel32", use_last_error=True)
    dll.GetCurrentProcess.argtypes = []
    dll.GetCurrentProcess.restype = ctypes.c_void_p
    dll.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    dll.SetPriorityClass.restype = ctypes.c_int
    dll.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    dll.SetProcessAffinityMask.restype = ctypes.c_int
    return dll


def set_low_priority(below_normal: bool = True) -> str:
    """Drop this process to below-normal scheduling priority."""
    dll = _kernel32()
    priority = BELOW_NORMAL_PRIORITY_CLASS if below_normal else 0x00000020
    if dll.SetPriorityClass(dll.GetCurrentProcess(), priority):
        return "below-normal" if below_normal else "normal"
    return f"unchanged (SetPriorityClass failed, err={ctypes.get_last_error()})"


def pin_to_cpus(mask: int) -> int:
    """Restrict this process to the logical processors in ``mask``.

    Returns the number of processors actually allowed, or 0 on failure.
    """
    if mask <= 0:
        return 0
    dll = _kernel32()
    if dll.SetProcessAffinityMask(dll.GetCurrentProcess(), ctypes.c_size_t(mask)):
        return bin(mask).count("1")
    return 0


def default_cpu_mask(n_cpus: int, total: int | None = None) -> int:
    """Mask for the first ``n_cpus`` logical processors."""
    if total is None:
        total = os.cpu_count() or 1
    n_cpus = max(1, min(n_cpus, total))
    return (1 << n_cpus) - 1


# --------------------------------------------------------------------------
# 3. Background temperature sampler
# --------------------------------------------------------------------------
_SAMPLER_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$pattern = $env:OPENCODE_TZ_PATTERN
while ($true) {
    $samples = (Get-Counter '\Thermal Zone Information(*)\Temperature').CounterSamples
    $cpu = $samples | Where-Object { $_.InstanceName -match $pattern } | Select-Object -First 1
    if (-not $cpu) { $cpu = $samples | Sort-Object CookedValue -Descending | Select-Object -First 1 }
    if ($cpu) {
        $c = [math]::Round($cpu.CookedValue - 273.15, 1)
        [Console]::Out.WriteLine("$c")
        [Console]::Out.Flush()
    }
    Start-Sleep -Seconds $env:OPENCODE_TZ_INTERVAL
}
"""


class _Sampler(threading.Thread):
    """Owns one long-lived PowerShell that samples CPU temp and prints it.

    One process, mostly asleep, so it costs almost no CPU itself. Reading the
    temperature from Python by spawning a fresh process per sample would be
    exactly the wrong thing to do on a machine that is already too hot.
    """

    def __init__(self, pattern: str, interval: int) -> None:
        super().__init__(daemon=True)
        self.script = Path(tempfile.gettempdir()) / "opencode_tz_sampler.ps1"
        self.script.write_text(_SAMPLER_PS, encoding="utf-8")
        env = dict(os.environ)
        env["OPENCODE_TZ_PATTERN"] = pattern
        env["OPENCODE_TZ_INTERVAL"] = str(int(interval))
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File", str(self.script)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env, encoding="utf-8", errors="replace",
        )
        self.readings: deque = deque(maxlen=4096)
        self.available = threading.Event()

    def run(self) -> None:
        for line in self.proc.stdout:          # type: ignore[union-attr]
            line = line.strip()
            try:
                self.readings.append(float(line))
            except ValueError:
                continue
            self.available.set()

    def latest(self) -> float | None:
        return self.readings[-1] if self.readings else None

    def stop(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass


# --------------------------------------------------------------------------
# 4. The guard itself
# --------------------------------------------------------------------------
class ThermalGuard:
    """Pause work when hot, thin out the hot loops, and report what happened.

    Usage inside a CPU-bound loop::

        guard = ThermalGuard(pause_above=82, resume_below=72)
        for rec in stream:
            ...
            guard.tick()          # cheap; really acts every ``tick_every`` rows
        guard.stop()
    """

    def __init__(
        self,
        pause_above: float = DEFAULT_PAUSE_ABOVE_C,
        resume_below: float = DEFAULT_RESUME_BELOW_C,
        abort_above: float = DEFAULT_ABORT_ABOVE_C,
        *,
        sample_seconds: int = DEFAULT_SAMPLE_SECONDS,
        pattern: str = "cpuz",
        tick_every: int = 2000,
        duty_sleep: float = 0.0,
        duty_every: int = 2000,
        label: str = "",
        verbose: bool = True,
    ) -> None:
        if resume_below >= pause_above:
            raise ValueError("resume_below must be below pause_above")
        self.pause_above = float(pause_above)
        self.resume_below = float(resume_below)
        self.abort_above = float(abort_above)
        self.tick_every = max(1, int(tick_every))
        self.duty_sleep = float(duty_sleep)
        self.duty_every = max(1, int(duty_every))
        self.label = label
        self.verbose = verbose

        self.max_temp: float | None = None
        self.paused_seconds = 0.0
        self.pause_count = 0
        self.temperature_available = False

        self._n = 0
        self._sampler: _Sampler | None = None
        try:
            self._sampler = _Sampler(pattern, sample_seconds)
            self._sampler.start()
            self._sampler.available.wait(timeout=5.0)
            self.temperature_available = bool(self._sampler.readings)
        except Exception:
            self.temperature_available = False
        if self.verbose and not self.temperature_available:
            print(f"  [thermal] temperature unavailable on '{pattern}'; "
                  f"running with priority/pinning/thread caps only", flush=True)

    # -- reading ---------------------------------------------------------
    def temp(self) -> float | None:
        if self._sampler is None:
            return None
        value = self._sampler.latest()
        if value is None:
            return None
        self.temperature_available = True
        if self.max_temp is None or value > self.max_temp:
            self.max_temp = value
        return value

    def _say(self, message: str) -> None:
        if self.verbose:
            prefix = f"[thermal{':' + self.label if self.label else ''}]"
            print(f"  {prefix} {message}", flush=True)

    # -- acting ----------------------------------------------------------
    def check(self, force: bool = False) -> None:
        """Read the temperature and pause if it is too high.

        Fail-open: if no temperature is available the run continues, because
        the thread cap, pinning and duty cycle are still doing their job.
        """
        value = self.temp()
        if value is None:
            return
        if value < self.pause_above and not force:
            return
        if value >= self.abort_above:
            raise ThermalAbort(
                f"CPU {value:.1f} C reached the abort ceiling "
                f"{self.abort_above:.1f} C; stopping to protect the machine. "
                f"Partial work already checkpointed is kept.")

        self.pause_count += 1
        self._say(f"CPU {value:.1f} C >= {self.pause_above:.1f} C; pausing "
                  f"until <= {self.resume_below:.1f} C")
        started = time.perf_counter()
        while True:
            if time.perf_counter() - started > MAX_PAUSE_SECONDS:
                raise ThermalAbort(
                    f"still at/above the pause threshold after "
                    f"{MAX_PAUSE_SECONDS // 60} min of cooling; stopping. "
                    f"Partial work already checkpointed is kept.")
            time.sleep(5.0)
            value = self.temp()
            if value is None:
                break
            if value < self.resume_below:
                break
        waited = time.perf_counter() - started
        self.paused_seconds += waited
        self._say(f"resumed after {waited:.0f}s cooling "
                  f"(now {value if value is None else f'{value:.1f} C'})")

    def tick(self) -> None:
        """Cheap per-row hook: really acts only every ``tick_every`` calls."""
        self._n += 1
        if self.duty_sleep and self._n % self.duty_every == 0:
            time.sleep(self.duty_sleep)
        if self._n % self.tick_every == 0:
            self.check()

    def stop(self) -> None:
        if self._sampler is not None:
            self._sampler.stop()

    def report(self) -> dict:
        return {
            "temperature_monitoring": self.temperature_available,
            "max_cpu_temp_c": self.max_temp,
            "pause_threshold_c": self.pause_above,
            "resume_threshold_c": self.resume_below,
            "abort_threshold_c": self.abort_above,
            "pause_events": self.pause_count,
            "total_paused_seconds": round(self.paused_seconds, 1),
            "duty_sleep_seconds_per": self.duty_sleep,
            "duty_every_rows": self.duty_every,
        }


def describe_threads() -> dict:
    """The BLAS-related env vars, for the run report."""
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys if k in os.environ}
