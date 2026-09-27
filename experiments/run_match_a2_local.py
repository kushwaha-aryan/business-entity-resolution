"""Business Entity Resolution - 20k S1 matching A/B (production vs production+A2).

FINAL self-contained local experiment runner for Windows / PyCharm.

Run with no arguments to do everything:

    python experiments/run_match_a2_local.py

or pick stages:   --stage 3        (dataset|subset|candidates|features|model|report|all)

Methodology is fixed by the existing project source in ``src/`` and is imported
from there, not copied: ``preprocessing.py``, ``blocking.py``, ``features.py``
and ``matching.py`` are used exactly as they are. Nothing in ``src/`` is
modified. No reimplementation, no alternate model, no external data, no
embeddings/APIs/geocoding/LLM calls, and the competition test split is never
read.

    20,000 S1 subset        blake2b(entity_id, key=b'exp20k-s1', 8) % 110 == 0
    production blocking     name_2tok + addr_ht0 + addr_ht1, max_group_size 1000
    A2 blocking             ("A2_name_2tok_order", country,
                             name_core[0][:4], name_core[1][:4]), cap 1000
    features                the existing 24 pairwise features, float32
    model                   LogisticMatcher: lbfgs, C=1.0, class_weight=None,
                            max_iter=1000, random_state=0, unscaled features
    split                   S1-level, validation_fraction=0.2, salt
                            'exp20k-split-v1', so an S1 entity is entirely in
                            train or entirely in validation
    metric                  F0.5, threshold chosen on validation

Thermal safety
--------------
This machine is a Ryzen 5 7530U laptop that reached ~96 C during an earlier
blocking run. Four independent controls are applied:

  1. BLAS/OpenMP capped to 1 thread, set *before* numpy is imported.
  2. Process dropped to below-normal priority and pinned to 3 logical CPUs.
  3. A background PowerShell temperature sampler: pause at 84 C, resume at
     79 C, abort at 90 C. Pausing keeps everything already on disk; it never
     throws work away.
  4. A duty-cycle sleep in the CPU-bound loops.

A known limit: the guard can only act between Python-level checkpoints, so it
cannot interrupt a single long C call. During the logistic-regression fit the
only protection is the thread cap and CPU pinning, which is exactly why those
are the defaults.

Checkpoints
-----------
Everything durable lives in ``experiments/checkpoints/match_a2/``. Writes are
atomic: a ``.part`` temporary in the same directory, ``flush`` + ``fsync``, then
``os.replace``. A reader therefore never sees a half-written file.

A stage is skipped ONLY when every file it owns is present AND passes
validation. A single surviving file is never enough. An incomplete or corrupt
checkpoint is discarded and that stage alone is regenerated; the expensive
10.3M-row S2/S3 blocking index is never rebuilt just because a metadata file
went missing.
"""
from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# BLAS thread cap. This MUST happen before numpy is imported anywhere, or the
# logistic-regression fit fans out across all 12 logical processors and spikes
# the CPU temperature.
# ---------------------------------------------------------------------------
def preparse_int_flag(argv: list[str], name: str, default: int) -> int:
    """Read ``--name N`` / ``--name=N`` out of argv before numpy exists."""
    for i, arg in enumerate(argv):
        if arg == name and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return default
        if arg.startswith(name + "="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return default
    return default


def cap_blas_threads(n: int = 1) -> None:
    value = str(max(1, int(n)))
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS"):
        os.environ[var] = value


BLAS_THREADS = preparse_int_flag(sys.argv[1:], "--threads", 1)
cap_blas_threads(BLAS_THREADS)

import argparse                                          # noqa: E402
import ctypes                                            # noqa: E402
import dataclasses                                       # noqa: E402
import gc                                                # noqa: E402
import hashlib                                          # noqa: E402
import json                                             # noqa: E402
import pickle                                           # noqa: E402
import shutil                                           # noqa: E402
import subprocess                                        # noqa: E402
import tempfile                                         # noqa: E402
import threading                                        # noqa: E402
import time                                             # noqa: E402
import zipfile                                          # noqa: E402
from collections import deque                           # noqa: E402
from pathlib import Path                                # noqa: E402

import numpy as np                                       # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import matching as M                           # noqa: E402
from src.blocking import (BlockingConfig, BlockingIndex,  # noqa: E402
                          CandidatePair, RecordRef, generate_candidates)
from src.features import FEATURE_NAMES, featurize       # noqa: E402

# ===========================================================================
# EXPERIMENT CONSTANTS - these define the experiment. Do not tune them.
# ===========================================================================
N_S1 = 20_000
S1_SALT, S1_MOD = b"exp20k-s1", 110
CFG = BlockingConfig()
CAP = CFG.max_group_size
PREFIX4 = CFG.name_prefix_length
SPLIT_FRACTION = 0.2
SPLIT_SALT = "exp20k-split-v1"
BETA = 0.5

# Sanity checks from the completed local blocking A/B. A mismatch is LOGGED,
# never fatal: it means the implementation drifted, not that the run must die.
EXPECT_PROD_CANDIDATES = 6_373_030
EXPECT_UNION_CANDIDATES = 7_580_413
EXPECT_A2_ONLY_CANDIDATES = 1_207_383
EXPECT_PROD_TRUE = 55_325
EXPECT_A2_ONLY_TRUE = 2_995
EXPECT_TOTAL_TRUE = 69_301

N_FEATURES = len(FEATURE_NAMES)
SOURCES = (("train_source2.tsv", "S2"), ("train_source3.tsv", "S3"))
REQUIRED_TRAIN_FILES = ("train_source1.tsv", "train_source2.tsv",
                        "train_source3.tsv", "train_ground_truth.tsv")

# ===========================================================================
# PATHS
# ===========================================================================
CKPT = REPO / "experiments" / "checkpoints" / "match_a2"
WORK = REPO / "experiments" / "work_match_a2"
DATA_ROOT = WORK / "dataset"
ZIP_HINT_ENV = "MATCH_A2_ZIP"

T0 = time.perf_counter()
CHECKS: list[tuple[str, bool, str]] = []
GUARD = None
PRIORITY = "not set"
CPUS_ALLOWED = "all"


def step(msg: str) -> None:
    print(f"[{time.perf_counter() - T0:8.1f}s] {msg}", flush=True)


def sub(msg: str) -> None:
    print(f"           {msg}", flush=True)


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    print(f"           [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def expect(name: str, got: int, want: int) -> None:
    """Sanity control: log loudly, never abort."""
    if got == want:
        check(name, True, f"{got:,} == {want:,}")
    else:
        CHECKS.append((name, False, f"{got:,} != {want:,}"))
        print(f"           [DIFF] {name}: got {got:,}, expected {want:,} "
              f"(delta {got - want:+,}). Logged, not fatal.", flush=True)


def guard_tick() -> None:
    if GUARD is not None:
        GUARD.tick()


def human(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


# ===========================================================================
# THERMAL / CPU CONTROLS  (Windows-compatible, no third-party dependency)
# ===========================================================================
class ThermalAbort(RuntimeError):
    """Raised when the CPU is too hot to continue safely."""


BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
DEFAULT_PAUSE_ABOVE_C = 84.0
DEFAULT_RESUME_BELOW_C = 79.0
DEFAULT_ABORT_ABOVE_C = 90.0
DEFAULT_SAMPLE_SECONDS = 20
MAX_PAUSE_SECONDS = 20 * 60


def _kernel32():
    """kernel32 with explicit signatures.

    Without them ``GetCurrentProcess`` returns the pseudo-handle -1, which gets
    truncated when passed as a default-width int, and both calls fail silently
    while reporting a success-looking default.
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
    dll = _kernel32()
    priority = BELOW_NORMAL_PRIORITY_CLASS if below_normal else 0x00000020
    if dll.SetPriorityClass(dll.GetCurrentProcess(), priority):
        return "below-normal" if below_normal else "normal"
    return f"unchanged (SetPriorityClass failed, err={ctypes.get_last_error()})"


def pin_to_cpus(mask: int) -> int:
    if mask <= 0:
        return 0
    dll = _kernel32()
    if dll.SetProcessAffinityMask(dll.GetCurrentProcess(), ctypes.c_size_t(mask)):
        return bin(mask).count("1")
    return 0


def default_cpu_mask(n_cpus: int) -> int:
    total = os.cpu_count() or 1
    return (1 << max(1, min(n_cpus, total))) - 1


_SAMPLER_PS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$pattern = $env:MATCH_A2_TZ_PATTERN
while ($true) {
    $samples = (Get-Counter '\Thermal Zone Information(*)\Temperature').CounterSamples
    $cpu = $samples | Where-Object { $_.InstanceName -match $pattern } | Select-Object -First 1
    if (-not $cpu) { $cpu = $samples | Sort-Object CookedValue -Descending | Select-Object -First 1 }
    if ($cpu) {
        [Console]::Out.WriteLine([string][math]::Round($cpu.CookedValue - 273.15, 1))
        [Console]::Out.Flush()
    }
    Start-Sleep -Seconds $env:MATCH_A2_TZ_INTERVAL
}
"""


class _TempSampler(threading.Thread):
    """One long-lived, mostly-asleep PowerShell that prints CPU temperatures.

    Reading the temperature by spawning a fresh process per sample would be
    exactly the wrong thing to do on an already-hot machine, so one process is
    kept alive and its stdout is drained here.
    """

    def __init__(self, pattern: str, interval: int) -> None:
        super().__init__(daemon=True)
        script = Path(tempfile.gettempdir()) / "match_a2_tz_sampler.ps1"
        script.write_text(_SAMPLER_PS, encoding="utf-8")
        env = dict(os.environ)
        env["MATCH_A2_TZ_PATTERN"] = pattern
        env["MATCH_A2_TZ_INTERVAL"] = str(int(interval))
        self.proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File", str(script)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, env=env, encoding="utf-8", errors="replace",
        )
        self.readings: deque[float] = deque(maxlen=4096)
        self.available = threading.Event()

    def run(self) -> None:
        for line in self.proc.stdout:            # type: ignore[union-attr]
            try:
                self.readings.append(float(line.strip()))
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


class ThermalGuard:
    """Pause when hot, thin out the hot loops, and report what happened."""

    def __init__(self, pause_above: float, resume_below: float, abort_above: float,
                 *, sample_seconds: int = DEFAULT_SAMPLE_SECONDS,
                 pattern: str = "cpuz", duty_sleep: float = 0.0,
                 duty_every: int = 2000, tick_every: int = 1000) -> None:
        if resume_below >= pause_above:
            raise ValueError("resume_below must be below pause_above")
        if abort_above <= pause_above:
            raise ValueError("abort_above must be above pause_above")
        self.pause_above = float(pause_above)
        self.resume_below = float(resume_below)
        self.abort_above = float(abort_above)
        self.duty_sleep = float(duty_sleep)
        self.duty_every = max(1, int(duty_every))
        self.tick_every = max(1, int(tick_every))
        self.max_temp: float | None = None
        self.paused_seconds = 0.0
        self.pause_count = 0
        self.available = False
        self._n = 0
        self._sampler: _TempSampler | None = None
        try:
            self._sampler = _TempSampler(pattern, sample_seconds)
            self._sampler.start()
            self._sampler.available.wait(timeout=6.0)
            self.available = bool(self._sampler.readings)
        except Exception:
            self.available = False
        if not self.available:
            print(f"           [thermal] temperature unavailable for "
                  f"'{pattern}'; running on priority/pinning/thread caps only",
                  flush=True)

    def temp(self) -> float | None:
        if self._sampler is None:
            return None
        value = self._sampler.latest()
        if value is not None:
            self.available = True
            if self.max_temp is None or value > self.max_temp:
                self.max_temp = value
        return value

    def _say(self, message: str) -> None:
        print(f"           [thermal] {message}", flush=True)

    def check(self, force: bool = False) -> None:
        """Pause if too hot. Fail-open when no temperature is readable."""
        value = self.temp()
        if value is None:
            return
        if value < self.pause_above and not force:
            return
        if value >= self.abort_above:
            raise ThermalAbort(
                f"CPU {value:.1f} C reached the abort ceiling "
                f"{self.abort_above:.1f} C. Stopping to protect the machine. "
                f"Everything already checkpointed on disk is kept; rerun to "
                f"resume.")
        self.pause_count += 1
        self._say(f"CPU {value:.1f} C >= {self.pause_above:.1f} C; pausing until "
                  f"<= {self.resume_below:.1f} C")
        started = time.perf_counter()
        while True:
            if time.perf_counter() - started > MAX_PAUSE_SECONDS:
                raise ThermalAbort(
                    f"still at/above {self.pause_above:.1f} C after "
                    f"{MAX_PAUSE_SECONDS // 60} min of cooling. Stopping. "
                    f"Checkpointed work is kept.")
            time.sleep(5.0)
            value = self.temp()
            if value is None or value < self.resume_below:
                break
        waited = time.perf_counter() - started
        self.paused_seconds += waited
        self._say(f"resumed after {human(waited)} cooling"
                  + ("" if value is None else f" (now {value:.1f} C)"))

    def tick(self) -> None:
        """Cheap per-row hook; only acts every ``tick_every`` calls."""
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
            "temperature_monitoring": self.available,
            "max_cpu_temp_c": self.max_temp,
            "pause_above_c": self.pause_above,
            "resume_below_c": self.resume_below,
            "abort_above_c": self.abort_above,
            "pause_events": self.pause_count,
            "total_paused_seconds": round(self.paused_seconds, 1),
            "duty_sleep_seconds": self.duty_sleep,
            "duty_every_rows": self.duty_every,
        }


def describe_threads() -> dict:
    keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys if k in os.environ}


# ===========================================================================
# ATOMIC CHECKPOINT WRITES
# ===========================================================================
def _fsync_dir(directory: Path) -> None:
    """Best-effort directory fsync. Windows cannot open a directory handle."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)          # atomic within one filesystem
    _fsync_dir(path.parent)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def _json_default(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    return str(obj)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default)
                      + "\n")


def atomic_write_pickle(path: Path, obj) -> None:
    atomic_write_bytes(path, pickle.dumps(obj, protocol=4))


def atomic_save_npy(path: Path, array, allow_pickle: bool = False) -> None:
    """Atomic .npy write via a file handle.

    np.save() would otherwise append a second '.npy' to the '.part' temporary
    name, so the replace below would find nothing at the path it expects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.save(fh, array, allow_pickle=allow_pickle)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def discard(path: Path, why: str) -> None:
    """Remove a checkpoint that failed validation, and say so."""
    for candidate in (path, path.with_name(path.name + ".part")):
        try:
            if candidate.is_file():
                candidate.unlink()
                print(f"           [discard] {candidate.name} ({why})", flush=True)
        except OSError:
            pass


# ===========================================================================
# DATASET ACCESS
# ===========================================================================
def find_dataset_zip() -> Path:
    """Locate the competition zip relative to the project, then the script."""
    override = os.environ.get(ZIP_HINT_ENV, "").strip()
    searched: list[Path] = []
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"{ZIP_HINT_ENV}={override} is not a file. Point it at "
            f"6ab10eb3b23ba_student_resource.zip or unset it.")

    roots = [REPO / "dataset", REPO / "data", REPO,
             Path(__file__).resolve().parent, Path.cwd()]
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        searched.append(resolved)
        zips = sorted(p for p in resolved.glob("*.zip") if p.is_file())
        if len(zips) == 1:
            return zips[0]
    # two or more candidates somewhere: prefer the one that names the dataset
    for root in seen:
        matches = [p for p in sorted(root.glob("*student_resource*.zip"))
                   if p.is_file()]
        if len(matches) == 1:
            return matches[0]
    listing = {str(r): sorted(p.name for p in r.glob("*.zip")) for r in searched}
    raise FileNotFoundError(
        f"could not find 6ab10eb3b23ba_student_resource.zip.\n"
        f"Put the zip in {REPO / 'dataset'} (that folder is scanned first), or "
        f"set the {ZIP_HINT_ENV} environment variable to its full path.\n"
        f"Zips found: {listing}")


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract, refusing any member that would escape ``dest``.

    ZipFile.extractall already strips leading separators and drops '..'
    components, but the archive is treated as untrusted input: every member is
    resolved first and rejected if it lands outside dest.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(
                    f"refusing zip member {member.filename!r}: it resolves to "
                    f"{target}, outside the extraction root {root}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            written += 1
    return written


def _iter_dirs(root: Path, max_depth: int = 4):
    yield root
    frontier = [root]
    for _ in range(max_depth):
        nxt = []
        for parent in frontier:
            try:
                children = [p for p in sorted(parent.iterdir()) if p.is_dir()]
            except OSError:
                continue
            for child in children:
                yield child
                nxt.append(child)
        frontier = nxt


def find_train_dir(root: Path) -> Path | None:
    """First folder under root that holds all four required training TSVs."""
    for candidate in (root, root / "train"):
        if all((candidate / n).is_file() for n in REQUIRED_TRAIN_FILES):
            return candidate
    for candidate in _iter_dirs(root):
        if all((candidate / n).is_file() for n in REQUIRED_TRAIN_FILES):
            return candidate
    return None


def train_path(name: str) -> Path:
    """Resolve a training TSV, refusing anything that is not a training file."""
    if "test" in name.lower():
        raise AssertionError(f"refusing non-train file: {name}")
    path = (TRAIN_DIR / name).resolve()
    if TRAIN_DIR.resolve() not in path.parents:
        raise AssertionError(f"refusing path outside train/: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return path


TRAIN_DIR: Path | None = None


# ===========================================================================
# STAGE 1 - dataset preparation
# ===========================================================================
def stage1_dataset(args) -> None:
    global TRAIN_DIR
    started = time.perf_counter()
    step("STAGE 1/6  DATASET PREPARATION")
    CKPT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    if args.zip:
        archive = Path(args.zip).expanduser().resolve()
        if not archive.is_file():
            raise FileNotFoundError(f"--zip {archive} is not a file")
    else:
        archive = find_dataset_zip()
    sub(f"zip: {archive} ({archive.stat().st_size / 1e9:.2f} GB)")

    found = find_train_dir(DATA_ROOT)
    if found is not None and not args.force:
        sub(f"already extracted, reusing: {found}")
    else:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        sub(f"extracting to {DATA_ROOT} - a few minutes")
        t = time.perf_counter()
        n = safe_extract(archive, DATA_ROOT)
        sub(f"extracted {n:,} files in {human(time.perf_counter() - t)}")
        found = find_train_dir(DATA_ROOT)

    if found is None:
        listing = sorted(p.name for p in DATA_ROOT.glob("*"))[:20] \
            if DATA_ROOT.is_dir() else []
        raise FileNotFoundError(
            f"after extraction, no folder under {DATA_ROOT} contains all of "
            f"{list(REQUIRED_TRAIN_FILES)}.\n"
            f"Expected something like "
            f"{DATA_ROOT}/student_resource/dataset/train/.\n"
            f"Found: {listing}")

    TRAIN_DIR = found
    for name in REQUIRED_TRAIN_FILES:
        path = train_path(name)
        size = path.stat().st_size
        if size <= 0:
            raise FileNotFoundError(f"dataset file is empty: {path}")
        sub(f"  {name:<26} {size / 1e6:>10.1f} MB")
    check("all four training files present and non-empty", True,
          f"{TRAIN_DIR}")
    step(f"  STAGE 1 done in {human(time.perf_counter() - started)}")


# ===========================================================================
# STAGE 2 - S1 subset + ground truth
# ===========================================================================
def hashed(entity_id: str, salt: bytes = S1_SALT, mod: int = S1_MOD) -> bool:
    digest = hashlib.blake2b(entity_id.encode("utf-8"), key=salt,
                             digest_size=8).digest()
    return int.from_bytes(digest, "big") % mod == 0


def a2_key(record):
    """A2 blocking key: original-order first two core name tokens, prefix 4."""
    if len(record.name_core) < 2:
        return None
    return ("A2_name_2tok_order", record.country,
            record.name_core[0][:PREFIX4], record.name_core[1][:PREFIX4])


def load_subset() -> tuple[dict, dict] | None:
    """Return validated (s1_records, truth), or None if not trustworthy."""
    subset_path = CKPT / "s1_records.pkl"
    truth_path = CKPT / "truth.pkl"
    if not (subset_path.is_file() and truth_path.is_file()):
        for p in (subset_path, truth_path):
            if not p.is_file():
                continue
            try:
                payload = pickle.loads(p.read_bytes())
            except Exception as exc:
                discard(p, f"unpicklable: {exc}")
                return None
            if p is subset_path and len(payload) != N_S1:
                discard(p, f"subset has {len(payload):,} records, need {N_S1:,}")
                return None
            if p is truth_path and not payload:
                discard(p, "ground truth is empty")
                return None
        return None
    try:
        s1_records = pickle.loads(subset_path.read_bytes())
        truth = pickle.loads(truth_path.read_bytes())
    except Exception as exc:
        discard(subset_path, f"unpicklable: {exc}")
        discard(truth_path, f"unpicklable: {exc}")
        return None
    if len(s1_records) != N_S1:
        discard(subset_path, f"subset has {len(s1_records):,}, need {N_S1:,}")
        return None
    if not truth:
        discard(truth_path, "ground truth is empty")
        return None
    return s1_records, truth


def stage2_subset(args) -> tuple[dict, dict]:
    started = time.perf_counter()
    step("STAGE 2/6  S1 SUBSET + GROUND TRUTH")
    loaded = None if args.force else load_subset()
    if loaded is not None:
        s1_records, truth = loaded
        sub(f"reusing validated checkpoints: {len(s1_records):,} S1 records, "
            f"{len(truth):,} GT rows")
        check("S1 subset size", len(s1_records) == N_S1, f"{len(s1_records):,}")
        expect("total true pairs", sum(len(v) for v in truth.values()),
               EXPECT_TOTAL_TRUE)
        step(f"  STAGE 2 done in {human(time.perf_counter() - started)} (cached)")
        return s1_records, truth

    sub(f"streaming S1, selecting on blake2b(key={S1_SALT!r}) % {S1_MOD} == 0")
    s1_records: dict[str, object] = {}
    scanned = 0
    for rec in M.iter_records(train_path("train_source1.tsv")):
        scanned += 1
        if len(s1_records) < N_S1 and hashed(rec.entity_id):
            s1_records[rec.entity_id] = rec
        if scanned % 200_000 == 0:
            guard_tick()
            sub(f"  scanned {scanned:,}, selected {len(s1_records):,}")
        if len(s1_records) >= N_S1:
            break
    if len(s1_records) != N_S1:
        raise AssertionError(
            f"selected {len(s1_records):,} S1 records, expected {N_S1:,}")
    sub(f"  S1 scanned {scanned:,}, selected {len(s1_records):,}")
    atomic_write_pickle(CKPT / "s1_records.pkl", s1_records)
    check("S1 subset size", len(s1_records) == N_S1, f"{len(s1_records):,}")

    sub("streaming train_ground_truth.tsv, keeping only the subset's S1 ids")
    wanted = set(s1_records)
    truth: dict[str, frozenset[str]] = {}
    with train_path("train_ground_truth.tsv").open(
            "r", encoding="utf-8", errors="replace", newline="") as fh:
        fh.readline()                              # header
        rows = 0
        for line in fh:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2:
                continue
            entity = parts[0].strip()
            if entity not in wanted:
                continue
            truth[entity] = frozenset(p.strip() for p in parts[1].split(",")
                                      if p.strip())
            rows += 1
            if rows % 20_000 == 0:
                guard_tick()
    total_true = sum(len(v) for v in truth.values())
    sub(f"  {len(truth):,} GT rows, {total_true:,} true pairs")
    expect("total true pairs", total_true, EXPECT_TOTAL_TRUE)
    atomic_write_pickle(CKPT / "truth.pkl", truth)
    step(f"  STAGE 2 done in {human(time.perf_counter() - started)}")
    return s1_records, truth


# ===========================================================================
# STAGE 3 - candidate generation
# ===========================================================================
def cand_paths(src: str) -> tuple[Path, Path]:
    return CKPT / f"cand_{src}_ids.npy", CKPT / f"cand_{src}_packed.npy"


def read_candidate_meta() -> dict | None:
    """Load and validate the stage-3 metadata, rebuilding it if derivable.

    The rule: candidate generation is complete only when BOTH arrays for BOTH
    sources AND the metadata exist and agree. If the arrays are intact but the
    metadata is gone, the metadata is reconstructed from the arrays themselves,
    because bit 0 of every packed row is the production flag by construction.
    Nothing here ever raises for a missing file; it returns None instead, and
    the caller regenerates.
    """
    meta_path = CKPT / "stage3_meta.json"
    arrays_ok = True
    rows_per_src: dict[str, int] = {}
    for _, src in SOURCES:
        ids_path, packed_path = cand_paths(src)
        if not (ids_path.is_file() and packed_path.is_file()):
            arrays_ok = False
            break
        try:
            ids = np.load(ids_path, allow_pickle=True, mmap_mode=None)
            packed = np.load(packed_path, mmap_mode=None)
        except Exception as exc:
            discard(ids_path, f"unreadable: {exc}")
            discard(packed_path, f"unreadable: {exc}")
            arrays_ok = False
            break
        if ids.shape[0] != packed.shape[0]:
            discard(packed_path, f"length {packed.shape[0]} != ids "
                                 f"{ids.shape[0]}")
            arrays_ok = False
            break
        rows_per_src[src] = int(ids.shape[0])

    if not arrays_ok:
        return None

    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            discard(meta_path, f"unparseable: {exc}")
            meta = None
        if meta is not None:
            needed = ("production_candidates", "union_candidates",
                      "n_s1", "true_pairs", "n_prod_true", "n_a2_only_true",
                      "rows_per_src")
            if all(k in meta for k in needed):
                if (meta.get("rows_per_src") == rows_per_src
                        and sum(rows_per_src.values())
                        == meta.get("union_candidates")):
                    return meta
                sub("stage3_meta.json disagrees with the candidate arrays; "
                    "rebuilding from the arrays")
            else:
                sub("stage3_meta.json is missing fields; rebuilding from the "
                    "arrays")

    # ---- reconstruct the metadata from the arrays themselves -------------
    n_union = 0
    n_prod = 0
    for _, src in SOURCES:
        _, packed_path = cand_paths(src)
        packed = np.load(packed_path, mmap_mode="r")
        n_union += int(packed.shape[0])
        n_prod += int(np.count_nonzero(packed & 1))
        del packed
    s1_path = CKPT / "s1_records.pkl"
    truth_path = CKPT / "truth.pkl"
    n_s1 = len(pickle.loads(s1_path.read_bytes())) if s1_path.is_file() else None
    true_pairs = None
    if truth_path.is_file():
        try:
            true_pairs = sum(len(v) for v in
                             pickle.loads(truth_path.read_bytes()).values())
        except Exception:
            true_pairs = None
    meta = {
        "n_s1": n_s1,
        "true_pairs": true_pairs,
        "production_candidates": n_prod,
        "union_candidates": n_union,
        "n_prod_true": None,
        "n_a2_only_true": None,
        "a2_only_candidates": n_union - n_prod,
        "rows_per_src": rows_per_src,
        "reconstructed": True,
    }
    atomic_write_json(meta_path, meta)
    sub("reconstructed stage3_meta.json from the candidate arrays")
    return meta


def stage3_candidates(args, s1_records: dict) -> dict:
    started = time.perf_counter()
    step("STAGE 3/6  CANDIDATE GENERATION")
    if not args.force:
        meta = read_candidate_meta()
        if meta is not None:
            sub("candidate arrays + metadata validated; skipping the 10.3M-row "
                "S2/S3 index rebuild")
            report_candidate_counts(meta)
            step(f"  STAGE 3 done in {human(time.perf_counter() - started)} "
                 f"(cached)")
            return meta

    s1_order = sorted(s1_records)
    s1_a2_keys = {a2_key(r) for r in s1_records.values()}
    s1_a2_keys.discard(None)
    sub(f"  {len(s1_a2_keys):,} distinct A2 keys across the subset")

    sub("streaming the FULL S2 + S3 into the production index and A2 buckets")
    index = BlockingIndex(CFG)
    a2_buckets: dict[tuple, list] = {}
    for fname, src in SOURCES:
        n = 0
        for rec in M.iter_records(train_path(fname)):
            n += 1
            index.add(rec, src)
            key = a2_key(rec)
            if key is not None and key in s1_a2_keys:
                bucket = a2_buckets.get(key)
                if bucket is None:
                    a2_buckets[key] = [RecordRef(src, rec.entity_id)]
                elif len(bucket) < CAP:
                    bucket.append(RecordRef(src, rec.entity_id))
            if n % 200_000 == 0:
                guard_tick()
                sub(f"    {src}: {n:,} rows indexed")
        sub(f"    {src}: {n:,} rows indexed")
    n_buckets = len(a2_buckets)
    sub(f"  {n_buckets:,} A2 buckets within cap {CAP}")

    sub("querying both arms for all 20,000 S1 records")
    truth = pickle.loads((CKPT / "truth.pkl").read_bytes())
    want: dict[str, dict] = {"S2": {}, "S3": {}}
    n_prod_seen = 0
    n_prod_true = 0
    n_a2_true = 0

    def _add(store: dict, key: str, packed: int) -> None:
        current = store.get(key)
        if current is None:
            store[key] = packed
        elif type(current) is int:
            store[key] = [current, packed]
        else:
            current.append(packed)

    s1_index = {eid: i for i, eid in enumerate(s1_order)}
    for done, eid in enumerate(s1_order, 1):
        rec = s1_records[eid]
        base = s1_index[eid] * 2
        truth_set = truth.get(eid, frozenset())
        prod = {(c.candidate.source, c.candidate.entity_id)
                for c in generate_candidates(rec, index, "S1")}
        n_prod_seen += len(prod)
        for src, cid in prod:
            _add(want[src], cid, base | 1)
            if cid in truth_set:
                n_prod_true += 1
        key = a2_key(rec)
        if key is not None:
            for ref in a2_buckets.get(key, ()):
                if (ref.source, ref.entity_id) not in prod:
                    _add(want[ref.source], ref.entity_id, base)
                    if ref.entity_id in truth_set:
                        n_a2_true += 1
        guard_tick()
        if done % 2_000 == 0:
            sub(f"    queried {done:,}/{N_S1:,} S1")

    n_union = 0
    rows_per_src: dict[str, int] = {}
    for _, src in SOURCES:
        store = want[src]
        total = sum(1 if type(v) is int else len(v) for v in store.values())
        ids = np.empty(total, dtype=object)
        packed = np.empty(total, dtype=np.int64)
        i = 0
        for cid, value in store.items():
            if type(value) is int:
                ids[i] = cid
                packed[i] = value
                i += 1
            else:
                for pk in value:
                    ids[i] = cid
                    packed[i] = pk
                    i += 1
        ids_path, packed_path = cand_paths(src)
        atomic_save_npy(ids_path, ids, allow_pickle=True)
        atomic_save_npy(packed_path, packed)
        rows_per_src[src] = int(ids.shape[0])
        n_union += ids.shape[0]
        sub(f"    {src}: {ids.shape[0]:,} candidate rows over {len(store):,} "
            f"distinct ids")

    truth_pairs_total = sum(len(v) for v in truth.values())
    meta = {
        "n_s1": len(s1_records),
        "true_pairs": truth_pairs_total,
        "production_candidates": n_prod_seen,
        "union_candidates": n_union,
        "a2_only_candidates": n_union - n_prod_seen,
        "n_prod_true": n_prod_true,
        "n_a2_only_true": n_a2_true,
        "a2_buckets": n_buckets,
        "rows_per_src": rows_per_src,
        "max_group_size": CAP,
        "name_prefix_length": PREFIX4,
        "blas_threads": BLAS_THREADS,
        "thermal": GUARD.report() if GUARD is not None else None,
        "reconstructed": False,
    }
    del index, a2_buckets, want
    gc.collect()
    atomic_write_json(CKPT / "stage3_meta.json", meta)
    report_candidate_counts(meta)
    step(f"  STAGE 3 done in {human(time.perf_counter() - started)}")
    return meta


def report_candidate_counts(meta: dict) -> None:
    prod = int(meta.get("production_candidates") or 0)
    union = int(meta.get("union_candidates") or 0)
    n_s1 = int(meta.get("n_s1") or N_S1)
    expect("production candidate count", prod, EXPECT_PROD_CANDIDATES)
    expect("union candidate count", union, EXPECT_UNION_CANDIDATES)
    expect("A2-only additions", union - prod, EXPECT_A2_ONLY_CANDIDATES)
    sub(f"  average candidates per S1: production {prod / max(1, n_s1):.1f}, "
        f"production+A2 {union / max(1, n_s1):.1f}")
    if meta.get("n_prod_true") is not None:
        expect("production true pairs", int(meta["n_prod_true"]),
               EXPECT_PROD_TRUE)
        expect("A2-only true pairs", int(meta["n_a2_only_true"]),
               EXPECT_A2_ONLY_TRUE)


# ===========================================================================
# STAGE 4 - feature generation
# ===========================================================================
FEATURE_STATE = CKPT / "stage4_state.json"
FEATURE_ROWS = CKPT / "stage4_rows.npz"
X_FINAL = CKPT / "X.npy"
X_PARTIAL = CKPT / "X.partial.npy"


def load_feature_state() -> dict | None:
    """Validated stage-4 state, or None. Never raises on a missing file."""
    if not (FEATURE_STATE.is_file() and FEATURE_ROWS.is_file()):
        # Half a stage-4 checkpoint is not a checkpoint. Drop the orphan so a
        # later run cannot mistake it for current work.
        for p in (FEATURE_STATE, FEATURE_ROWS):
            if p.is_file():
                discard(p, "the other half of this checkpoint is missing")
        return None
    try:
        state = json.loads(FEATURE_STATE.read_text(encoding="utf-8"))
    except Exception as exc:
        discard(FEATURE_STATE, f"unparseable: {exc}")
        return None
    needed = ("pos", "pos_in_file", "n_prod_true", "done_src", "n_union",
              "n_features")
    if not all(k in state for k in needed):
        discard(FEATURE_STATE, f"missing keys {[k for k in needed if k not in state]}")
        return None
    if int(state["n_features"]) != N_FEATURES:
        discard(FEATURE_STATE, f"built with {state['n_features']} features, "
                               f"need {N_FEATURES}")
        return None
    if not 0 <= int(state["pos"]) <= int(state["n_union"]):
        discard(FEATURE_STATE, f"pos {state['pos']} outside 0..{state['n_union']}")
        return None
    if 0 <= int(state["pos"]) < int(state["n_union"]) \
            and int(state["pos_in_file"]) == 0 and not state["done_src"]:
        discard(FEATURE_STATE, "resume point is inconsistent")
        return None
    try:
        with np.load(FEATURE_ROWS) as saved:
            if any(k not in saved for k in ("y", "is_prod", "is_a2only_true",
                                            "s1_of_row")):
                discard(FEATURE_ROWS, "missing label arrays")
                return None
            if any(saved[k].shape[0] != int(state["n_union"])
                   for k in ("y", "is_prod", "is_a2only_true", "s1_of_row")):
                discard(FEATURE_ROWS, "array length != n_union")
                return None
    except Exception as exc:
        discard(FEATURE_ROWS, f"unreadable: {exc}")
        return None
    return state


def open_feature_matrix(n_union: int, resuming: bool):
    """Return (memmap, is_usable).

    Only a matrix that a validated state file vouches for may be continued. A
    matrix with no state is discarded rather than guessed at, because a wrong
    ``pos`` would silently mislabel every remaining row.
    """
    want = (n_union, N_FEATURES)
    if resuming:
        for path in (X_FINAL, X_PARTIAL):
            if not path.is_file():
                continue
            try:
                array = np.load(path, mmap_mode="r+")
            except Exception as exc:
                discard(path, f"unreadable memmap: {exc}")
                continue
            if array.shape != want:
                discard(path, f"shape {array.shape} != {want}")
                continue
            return array, True
    for path in (X_FINAL, X_PARTIAL):
        if path.is_file():
            discard(path, "no validated state; rebuilding the matrix from zero")
    array = np.lib.format.open_memmap(X_PARTIAL, mode="w+", dtype=np.float32,
                                      shape=want)
    return array, False


def load_want(n_union: int) -> dict[str, dict]:
    """Rebuild the candidate-id -> packed-rows map, straight off disk."""
    want: dict[str, dict] = {}
    for _, src in SOURCES:
        ids = np.load(cand_paths(src)[0], allow_pickle=True)
        packed = np.load(cand_paths(src)[1])
        store: dict = {}
        for cid, pk in zip(ids.tolist(), packed.tolist()):
            current = store.get(cid)
            if current is None:
                store[cid] = pk
            elif type(current) is int:
                store[cid] = [current, pk]
            else:
                current.append(pk)
        want[src] = store
        del ids, packed
    return want


def stage4_features(args, meta: dict, s1_records: dict) -> None:
    started = time.perf_counter()
    step("STAGE 4/6  FEATURE GENERATION")
    n_union = int(meta["union_candidates"])
    if n_union <= 0:
        raise AssertionError(f"stage 3 reported {n_union} candidate rows")

    state = None if args.force else load_feature_state()
    if state is not None and int(state["n_union"]) != n_union:
        discard(FEATURE_STATE, f"built for {state['n_union']:,} rows, stage 3 "
                               f"now says {n_union:,}")
        state = None
    if state is None and not args.force:
        for path in (X_FINAL, X_PARTIAL):
            if path.is_file():
                discard(path, "no validated state accompanies this matrix")

    resuming = state is not None
    X, matrix_ok = open_feature_matrix(n_union, resuming)

    if not matrix_ok or not resuming:
        pos = 0
        pos_in_file = 0
        n_prod_true = 0
        done_src: list[str] = []
        y = np.zeros(n_union, dtype=np.int8)
        is_prod = np.zeros(n_union, dtype=bool)
        is_a2only_true = np.zeros(n_union, dtype=bool)
        s1_of_row = np.zeros(n_union, dtype=np.int32)
        sub(f"  starting fresh: {n_union:,} rows x {N_FEATURES} float32 "
            f"({n_union * N_FEATURES * 4 / 1e6:.0f} MB memmap)")
    else:
        pos = int(state["pos"])
        pos_in_file = int(state["pos_in_file"])
        n_prod_true = int(state["n_prod_true"])
        done_src = list(state["done_src"])
        with np.load(FEATURE_ROWS) as saved:
            y = saved["y"].copy()
            is_prod = saved["is_prod"].copy()
            is_a2only_true = saved["is_a2only_true"].copy()
            s1_of_row = saved["s1_of_row"].copy()
        sub(f"  resuming at {pos:,} of {n_union:,} rows "
            f"(pos_in_file={pos_in_file:,}, completed={done_src})")

    if pos < n_union:
        want = load_want(n_union)
        s1_order = sorted(s1_records)
        truth = pickle.loads((CKPT / "truth.pkl").read_bytes())
        last_checkpoint = pos

        def write_checkpoint(src: str) -> None:
            """Flush, then write labels, then commit the state file.

            Label arrays are written BEFORE the state, so a crash in between
            leaves the state pointing at older-but-consistent work rather than
            claiming rows whose labels were never saved.
            """
            X.flush()
            atomic_save_npz(FEATURE_ROWS, y=y, is_prod=is_prod,
                           is_a2only_true=is_a2only_true, s1_of_row=s1_of_row)
            atomic_write_json(FEATURE_STATE, {
                "pos": int(pos), "pos_in_file": int(pos_in_file),
                "n_prod_true": int(n_prod_true), "done_src": list(done_src),
                "n_union": int(n_union), "n_features": int(N_FEATURES),
            })
            sub(f"  checkpoint: {pos:,} rows after {src} "
                f"(done={done_src})")

        for fname, src in SOURCES:
            if pos >= n_union:
                break
            if src in done_src:
                sub(f"  {src} already complete, skipping")
                continue
            sub(f"  featurising {src}")
            store = want[src]
            skip = pos_in_file
            seen = 0
            joined = 0
            file_started = pos
            for rec in M.iter_records(train_path(fname)):
                value = store.get(rec.entity_id)
                if value is None:
                    guard_tick()
                    continue
                seen += 1
                pks = (value,) if type(value) is int else value
                if skip > 0:                       # already done before the crash
                    skip -= len(pks)
                    guard_tick()
                    continue
                for pk in pks:
                    si = pk >> 1
                    prod_row = bool(pk & 1)
                    eid = s1_order[si]
                    X[pos] = featurize(s1_records[eid], rec)
                    s1_of_row[pos] = si
                    is_prod[pos] = prod_row
                    hit = rec.entity_id in truth.get(eid, frozenset())
                    y[pos] = 1 if hit else 0
                    is_a2only_true[pos] = (not prod_row) and hit
                    if hit and prod_row:
                        n_prod_true += 1
                    pos += 1
                    pos_in_file += 1
                    joined += 1
                    guard_tick()
                if pos - last_checkpoint >= args.feature_checkpoint_every:
                    write_checkpoint(f"{src} (partial)")
                    last_checkpoint = pos
            sub(f"    {src}: joined {seen:,}/{len(store):,} ids, "
                f"{pos - file_started:,} rows this pass, {pos:,} total")
            if skip != 0:
                raise AssertionError(
                    f"{src}: resume skipped {skip:,} rows but the file ran out")
            if seen != len(store):
                raise AssertionError(
                    f"{src}: {len(store) - seen:,} wanted candidate ids were "
                    f"never found in the file")
            pos_in_file = 0
            if src not in done_src:
                done_src.append(src)
            write_checkpoint(src)
            last_checkpoint = pos
            if GUARD is not None:
                GUARD.check(force=True)

        del want
        gc.collect()

    if pos != n_union:
        raise AssertionError(
            f"featurised {pos:,} of {n_union:,} expected rows")

    X.flush()
    del X
    gc.collect()                      # close the mmap before replacing it
    if X_PARTIAL.is_file():
        os.replace(X_PARTIAL, X_FINAL)           # atomic promotion
        _fsync_dir(CKPT)
    check("all rows featurised", True, f"{pos:,}")
    check("production-arm true pairs", n_prod_true == EXPECT_PROD_TRUE,
          f"{n_prod_true:,} == {EXPECT_PROD_TRUE:,}")
    check("A2-only true pairs", int(is_a2only_true.sum()) == EXPECT_A2_ONLY_TRUE,
          f"{int(is_a2only_true.sum()):,} == {EXPECT_A2_ONLY_TRUE:,}")
    check("total true pairs",
          int(y.sum()) == EXPECT_PROD_TRUE + EXPECT_A2_ONLY_TRUE,
          f"{int(y.sum()):,}")
    step(f"  STAGE 4 done in {human(time.perf_counter() - started)}")


# =========================================================================--
# STAGE 5 - model fit and evaluation
# ===========================================================================
MODEL_RESULTS = CKPT / "stage5_results.json"


def load_model_results() -> dict | None:
    if not MODEL_RESULTS.is_file():
        return None
    try:
        payload = json.loads(MODEL_RESULTS.read_text(encoding="utf-8"))
    except Exception as exc:
        discard(MODEL_RESULTS, f"unparseable: {exc}")
        return None
    arms = payload.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        discard(MODEL_RESULTS, "expected exactly two arms")
        return None
    for arm in arms:
        if not isinstance(arm, dict):
            discard(MODEL_RESULTS, "malformed arm")
            return None
        if "best" not in arm or "reports" not in arm or "name" not in arm:
            discard(MODEL_RESULTS, f"arm {arm.get('name')!r} is incomplete")
            return None
        best = arm["best"]
        for key in ("threshold", "precision", "recall", "f_beta",
                    "true_positives", "false_positives", "false_negatives",
                    "predicted_positives"):
            if key not in best:
                discard(MODEL_RESULTS, f"arm {arm['name']} best missing {key}")
                return None
    return payload


def stage5_model(args) -> dict:
    started = time.perf_counter()
    step("STAGE 5/6  LOGISTIC REGRESSION")
    if not args.force:
        cached = load_model_results()
        if cached is not None:
            sub("reusing validated stage5_results.json")
            for arm in cached["arms"]:
                sub(f"  {arm['name']}: F0.5={arm['best']['f_beta']:.4f} "
                    f"@ threshold {arm['best']['threshold']}")
            step(f"  STAGE 5 done in {human(time.perf_counter() - started)} "
                 f"(cached)")
            return cached

    for needed in (X_FINAL, FEATURE_ROWS, CKPT / "s1_records.pkl"):
        if not needed.is_file():
            raise FileNotFoundError(
                f"stage 5 needs {needed.name}, which stage 4 did not produce")

    s1_order = sorted(pickle.loads((CKPT / "s1_records.pkl").read_bytes()))
    X = np.load(X_FINAL, mmap_mode="r")
    with np.load(FEATURE_ROWS) as saved:
        y = saved["y"].copy()
        is_prod = saved["is_prod"].copy()
        is_a2only_true = saved["is_a2only_true"].copy()
        s1_of_row = saved["s1_of_row"].copy()
    n_union = int(X.shape[0])
    sub(f"  loaded {n_union:,} x {X.shape[1]} features "
        f"(positives {int(y.sum()):,})")

    # -- S1-level split, one call per unique S1 id --------------------------
    probe = [M.LabelledPair(
        CandidatePair(RecordRef("S1", eid), RecordRef("S2", "S2-0")), 0)
        for eid in s1_order]
    train_probe, val_probe = M.split_by_reference(
        probe, validation_fraction=SPLIT_FRACTION, salt=SPLIT_SALT)
    val_s1 = {i.pair.reference.entity_id for i in val_probe}
    train_s1 = {i.pair.reference.entity_id for i in train_probe}
    del probe, train_probe, val_probe
    s1_index = {eid: i for i, eid in enumerate(s1_order)}
    is_val_s1 = np.zeros(len(s1_order), dtype=bool)
    for eid in val_s1:
        is_val_s1[s1_index[eid]] = True
    row_is_val = is_val_s1[s1_of_row]
    sub(f"  split: {len(train_s1):,} train S1, {len(val_s1):,} validation S1 "
        f"(salt {SPLIT_SALT!r}, fraction {SPLIT_FRACTION})")
    check("every S1 entity on exactly one side",
          len(train_s1) + len(val_s1) == len(s1_order),
          f"{len(train_s1):,} + {len(val_s1):,} = {len(s1_order):,}")
    check("no S1 entity on both sides", not (train_s1 & val_s1))

    def run_arm(name: str, mask) -> dict:
        if mask is None:
            rows = np.arange(y.size)
        else:
            rows = np.flatnonzero(mask)
        ya, a2t, row_val = y[rows], is_a2only_true[rows], row_is_val[rows]
        val_idx = np.flatnonzero(row_val)
        trn_idx = np.flatnonzero(~row_val)
        ytr = ya[trn_idx].tolist()
        yva = ya[val_idx].tolist()
        Xtr = X if trn_idx.size == ya.size else X[rows[trn_idx]]
        Xva = np.asarray(X[rows[val_idx]])
        sub(f"  [{name}] {ya.size:,} rows (train {trn_idx.size:,}, "
            f"valid {val_idx.size:,}), positives {int(ya.sum()):,} "
            f"({int(ya.sum()) / max(1, ya.size):.4%})")
        guard_tick()
        t = time.perf_counter()
        model = M.LogisticMatcher()
        model.fit(Xtr, ytr)
        summary = model.summary
        coefficients = model.coefficients()
        sub(f"  [{name}] fit in {human(time.perf_counter() - t)}: "
            f"n_iter={summary.n_iter}, converged={summary.converged}, "
            f"class_weight={summary.class_weight}")
        if GUARD is not None:
            GUARD.check(force=True)
        scores = model.predict_proba(Xva)
        reports = M.evaluate_thresholds(yva, scores, beta=BETA)
        best = M.best_validation_threshold(reports)
        predicted = [score >= best.threshold for score in scores]
        a2_val = a2t[val_idx].tolist()
        out = {
            "name": name,
            "rows": int(ya.size),
            "positives": int(ya.sum()),
            "pos_rate": float(ya.sum()) / max(1, int(ya.size)),
            "train_rows": int(trn_idx.size),
            "valid_rows": int(val_idx.size),
            "valid_positives": int(ya[val_idx].sum()),
            "candidates_per_s1": float(ya.size) / max(1, len(s1_order)),
            "n_iter": summary.n_iter,
            "converged": summary.converged,
            "class_weight": summary.class_weight,
            "C": model.C, "solver": model.solver,
            "max_iter": model.max_iter, "random_state": model.random_state,
            "coefficients": coefficients,
            "reports": [dataclasses.asdict(r) for r in reports],
            "best": dataclasses.asdict(best),
            "tp": best.true_positives, "fp": best.false_positives,
            "fn": best.false_negatives, "precision": best.precision,
            "recall": best.recall, "f_beta": best.f_beta,
            "pred_pos": best.predicted_positives,
            "a2_only_true_total": int(a2t.sum()),
            "a2_only_true_valid": int(sum(a2_val)),
            "a2_only_true_caught": int(sum(1 for a, p in zip(a2_val, predicted)
                                           if a and p)),
        }
        del Xtr, Xva, scores, predicted, ya, a2t, row_val, rows, val_idx, trn_idx
        gc.collect()
        return out

    arms = [run_arm("production", is_prod), run_arm("production+A2", None)]

    check("no NaN/inf anywhere in the feature matrix",
          bool(np.isfinite(np.asarray(X)).all()), f"all {n_union:,} rows")
    payload = {
        "n_union": n_union,
        "n_features": int(X.shape[1]),
        "n_s1": len(s1_order),
        "split": {"validation_fraction": SPLIT_FRACTION, "salt": SPLIT_SALT,
                  "train_s1": len(train_s1), "validation_s1": len(val_s1)},
        "beta": BETA,
        "arms": arms,
        "thermal": GUARD.report() if GUARD is not None else None,
    }
    atomic_write_json(MODEL_RESULTS, payload)
    for arm in arms:
        sub(f"  {arm['name']}: P={arm['precision']:.4f} R={arm['recall']:.4f} "
            f"F0.5={arm['f_beta']:.4f} @ {arm['best']['threshold']}")
    step(f"  STAGE 5 done in {human(time.perf_counter() - started)}")
    return payload


# ===========================================================================
# STAGE 6 - final comparison report
# ===========================================================================
REPORT_TXT = CKPT / "match_a2_report.txt"
REPORT_JSON = CKPT / "match_a2_summary.json"


def stage6_report(args, meta: dict, model: dict, s1_records: dict) -> None:
    started = time.perf_counter()
    step("STAGE 6/6  FINAL COMPARISON REPORT")
    if not args.force and REPORT_TXT.is_file() and REPORT_JSON.is_file() \
            and not args.rerun_report:
        sub("report already written; use --rerun-report to rebuild it")
        step(f"  STAGE 6 done in {human(time.perf_counter() - started)} (cached)")
        print()
        print(REPORT_TXT.read_text(encoding="utf-8"))
        return

    truth = pickle.loads((CKPT / "truth.pkl").read_bytes())
    total_true = sum(len(v) for v in truth.values())
    n_s1 = len(s1_records)
    prod_cand = int(meta["production_candidates"])
    union_cand = int(meta["union_candidates"])
    prod_true = int(meta.get("n_prod_true") or EXPECT_PROD_TRUE)
    a2_true = int(meta.get("n_a2_only_true") or EXPECT_A2_ONLY_TRUE)
    arm1, arm2 = model["arms"]
    blocking = {
        "production": {
            "candidates": prod_cand,
            "candidates_per_s1": prod_cand / max(1, n_s1),
            "true_pairs": prod_true,
            "recall": prod_true / max(1, total_true),
        },
        "production+A2": {
            "candidates": union_cand,
            "candidates_per_s1": union_cand / max(1, n_s1),
            "true_pairs": prod_true + a2_true,
            "recall": (prod_true + a2_true) / max(1, total_true),
        },
    }
    summary = {
        "n_s1": n_s1,
        "n_features": N_FEATURES,
        "total_true_pairs": total_true,
        "candidate_counts": {"production": prod_cand, "production+A2": union_cand,
                             "a2_only_additions": union_cand - prod_cand},
        "blocking": blocking,
        "arms": model["arms"],
        "split": model["split"],
        "beta": BETA,
        "resources": {
            "blas_threads": BLAS_THREADS,
            "blas_env": describe_threads(),
            "priority": PRIORITY,
            "cpus_allowed": CPUS_ALLOWED,
            "thermal": GUARD.report() if GUARD is not None else None,
            "total_elapsed_seconds": round(time.perf_counter() - T0, 1),
        },
        "checks": [{"name": n, "ok": o, "detail": d} for n, o, d in CHECKS],
    }
    atomic_write_json(REPORT_JSON, summary)

    lines: list[str] = []
    add = lines.append
    rule = "=" * 78

    add(rule)
    add("BUSINESS ENTITY RESOLUTION - 20k S1 MATCHING A/B")
    add("Arm A = production blocking    Arm B = production blocking + A2")
    add(rule)
    add("")
    add("1. CONTROLS")
    add(f"   S1 records in the subset            {n_s1:,}")
    add(f"   features per pair                   {N_FEATURES}")
    add(f"   production blocking                 name_2tok + addr_ht0 + "
        f"addr_ht1, cap {CAP}")
    add(f"   A2 blocking                         original-order first 2 core "
        f"name tokens, prefix {PREFIX4}, country, cap {CAP}")
    add(f"   model                               LogisticRegression "
        f"(solver={arm1['solver']}, C={arm1['C']}, "
        f"class_weight={arm1['class_weight']}, max_iter={arm1['max_iter']}, "
        f"random_state={arm1['random_state']}, features unscaled)")
    add(f"   split                               S1-level, "
        f"validation_fraction={SPLIT_FRACTION}, salt {SPLIT_SALT!r} "
        f"({model['split']['train_s1']:,} train / "
        f"{model['split']['validation_s1']:,} validation S1)")
    add(f"   metric                              F{BETA}")
    add(f"   total true pairs for these S1      {total_true:,}")
    add("")
    add("2. BLOCKING")
    add(f"   {'arm':<18}{'candidates':>14}{'per S1':>10}{'true pairs':>13}"
        f"{'recall':>10}")
    for name, stats in blocking.items():
        add(f"   {name:<18}{stats['candidates']:>14,}"
            f"{stats['candidates_per_s1']:>10.1f}{stats['true_pairs']:>13,}"
            f"{stats['recall']:>10.4%}")
    add(f"   A2-only additions                  {union_cand - prod_cand:,}")
    add(f"   A2-only true pairs recovered       {a2_true:,}")
    add(f"   A2 share of the candidate pool     "
        f"{(union_cand - prod_cand) / max(1, union_cand):.2%} of union rows are "
        f"A2-only")
    add("")
    add("3. VALIDATION RESULT AT EACH ARM'S BEST F0.5 THRESHOLD")
    add(f"   {'arm':<18}{'rows':>12}{'positives':>12}{'prec':>9}{'recall':>9}"
        f"{'F0.5':>9}{'thr':>6}{'TP':>9}{'FP':>11}{'FN':>10}")
    for arm in (arm1, arm2):
        best = arm["best"]
        add(f"   {arm['name']:<18}{arm['rows']:>12,}{arm['positives']:>12,}"
            f"{best['precision']:>9.4f}{best['recall']:>9.4f}"
            f"{best['f_beta']:>9.4f}{best['threshold']:>6.1f}"
            f"{best['true_positives']:>9,}{best['false_positives']:>11,}"
            f"{best['false_negatives']:>10,}")
    add("")
    add(f"   F0.5   (+A2 - production)          "
        f"{arm2['f_beta'] - arm1['f_beta']:+.4f}")
    add(f"   precision change                    "
        f"{arm2['precision'] - arm1['precision']:+.4f}")
    add(f"   recall change                       "
        f"{arm2['recall'] - arm1['recall']:+.4f}")
    add(f"   true positives                      {arm2['tp'] - arm1['tp']:+,}")
    add(f"   false positives                     {arm2['fp'] - arm1['fp']:+,}")
    add("")
    add("4. THE A2-RECOVERED TRUE PAIRS")
    add(f"   in the candidate set                "
        f"{arm2['a2_only_true_total']:,}")
    add(f"   in the validation split             "
        f"{arm2['a2_only_true_valid']:,}")
    add(f"   predicted positive at the +A2 best  "
        f"{arm2['a2_only_true_caught']:,} "
        f"(threshold {arm2['best']['threshold']})")
    add(f"   recall on that subset               "
        f"{arm2['a2_only_true_caught'] / max(1, arm2['a2_only_true_valid']):.4%}")
    add("")
    add("   These pairs do not exist in the production arm at all: production")
    add("   blocking never retrieved them, so the production model never had the")
    add("   chance to score them. The comparison above is therefore the whole")
    add("   effect of A2: the extra candidates it brings, and what the matcher")
    add("   does with them.")
    add("")
    add("5. THRESHOLD SWEEPS")
    for arm in (arm1, arm2):
        add("")
        add(f"   {arm['name']}")
        add(f"     {'thr':>5}{'prec':>9}{'rec':>9}{'F0.5':>9}{'TP':>9}"
            f"{'FP':>10}{'FN':>9}{'pred+':>10}")
        for r in arm["reports"]:
            add(f"     {r['threshold']:>5.1f}{r['precision']:>9.4f}"
                f"{r['recall']:>9.4f}{r['f_beta']:>9.4f}"
                f"{r['true_positives']:>9,}{r['false_positives']:>10,}"
                f"{r['false_negatives']:>9,}{r['predicted_positives']:>10,}")
    add("")
    add("6. FITTED COEFFICIENTS (unscaled features)")
    add(f"   {'feature':<28}{'production':>13}{'production+A2':>15}")
    for feature in FEATURE_NAMES:
        add(f"   {feature:<28}{arm1['coefficients'][feature]:>13.4f}"
            f"{arm2['coefficients'][feature]:>15.4f}")
    add("")
    add("7. RESOURCES AND CHECKS")
    res = summary["resources"]
    add(f"   BLAS/OpenMP threads                 {BLAS_THREADS} "
        f"({res['blas_env']})")
    add(f"   process priority                    {res['priority']}")
    add(f"   logical CPUs allowed                {res['cpus_allowed']}")
    thermal = res["thermal"]
    if thermal:
        hottest = thermal["max_cpu_temp_c"]
        add(f"   temperature monitoring              "
            f"{'active' if thermal['temperature_monitoring'] else 'unavailable'}")
        add(f"   max CPU temp observed               "
            f"{'unavailable' if hottest is None else f'{hottest:.1f} C'}")
        add(f"   pause / resume / abort             "
            f"{thermal['pause_above_c']:.0f} C / {thermal['resume_below_c']:.0f} C "
            f"/ {thermal['abort_above_c']:.0f} C")
        add(f"   pause events                        {thermal['pause_events']} "
            f"({human(thermal['total_paused_seconds'])} total)")
    add(f"   total elapsed                       "
        f"{human(res['total_elapsed_seconds'])}")
    add("")
    for name, ok, detail in CHECKS:
        add(f"   [{'PASS' if ok else 'FAIL'}] {name}"
            + (f" -- {detail}" if detail else ""))
    add("")
    add(rule)
    add("Validation F0.5 on a bounded S1 subset against the full S2/S3 pool.")
    add("This is NOT competition performance and NOT a submitted result. The 20k")
    add("S1 subset is a sample, and the competition metric is macro-averaged over")
    add("S1 entities, whereas the numbers above are pair-level on pooled")
    add("candidates.")
    add(rule)

    text = "\n".join(lines) + "\n"
    atomic_write_text(REPORT_TXT, text)
    step(f"  wrote {REPORT_TXT}")
    step(f"  wrote {REPORT_JSON}")
    step(f"  STAGE 6 done in {human(time.perf_counter() - started)}")
    print()
    print(text)


def print_finisher(meta: dict, model: dict) -> None:
    arm1, arm2 = model["arms"]
    prod = int(meta["production_candidates"])
    union = int(meta["union_candidates"])
    n_s1 = int(meta.get("n_s1") or N_S1)
    total_true = int(meta.get("true_pairs") or EXPECT_TOTAL_TRUE)
    prod_true = int(meta.get("n_prod_true") or EXPECT_PROD_TRUE)
    a2_true = int(meta.get("n_a2_only_true") or EXPECT_A2_ONLY_TRUE)
    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Production candidates:      {prod:,} ({prod / max(1, n_s1):.1f} per S1)")
    print(f"Union candidates:           {union:,} "
          f"({union / max(1, n_s1):.1f} per S1)")
    print(f"A2-only candidates:         {union - prod:,}")
    print(f"Production blocking recall: "
          f"{prod_true / max(1, total_true):.4%} "
          f"({prod_true:,}/{total_true:,})")
    print(f"Production+A2 recall:       "
          f"{(prod_true + a2_true) / max(1, total_true):.4%} "
          f"({prod_true + a2_true:,}/{total_true:,})")
    print(f"A2-only true pairs caught: {arm2['a2_only_true_caught']:,} of "
          f"{arm2['a2_only_true_valid']:,} in validation")
    print(f"Production F0.5:            {arm1['f_beta']:.4f} "
          f"(P {arm1['precision']:.4f} / R {arm1['recall']:.4f} "
          f"@ threshold {arm1['best']['threshold']})")
    print(f"Production+A2 F0.5:         {arm2['f_beta']:.4f} "
          f"(P {arm2['precision']:.4f} / R {arm2['recall']:.4f} "
          f"@ threshold {arm2['best']['threshold']})")
    print(f"F0.5 delta (+A2):           "
          f"{arm2['f_beta'] - arm1['f_beta']:+.4f}")
    print("=" * 60)
    print(f"Report : {REPORT_TXT}")
    print(f"Metrics: {REPORT_JSON}")
    print(f"Checkpoints: {CKPT}")
    print("=" * 60)


# ===========================================================================
# DRIVER
# ===========================================================================
STAGES = ("dataset", "subset", "candidates", "features", "model", "report",
          "all")


def main(argv: list[str] | None = None) -> int:
    global GUARD, PRIORITY, CPUS_ALLOWED, TRAIN_DIR

    parser = argparse.ArgumentParser(
        description="Business Entity Resolution 20k matching A/B, "
                    "staged for thermal safety. Run with no arguments for all.")
    parser.add_argument("--stage", default="all", choices=STAGES,
                        help="run a single stage, or 'all' (default)")
    parser.add_argument("--zip", default="",
                        help="explicit path to the dataset zip")
    parser.add_argument("--threads", type=int, default=1,
                        help="BLAS/OpenMP threads; must precede numpy import")
    parser.add_argument("--cpus", type=int, default=3,
                        help="logical processors this process may use")
    parser.add_argument("--pause-above", type=float,
                        default=DEFAULT_PAUSE_ABOVE_C)
    parser.add_argument("--resume-below", type=float,
                        default=DEFAULT_RESUME_BELOW_C)
    parser.add_argument("--abort-above", type=float,
                        default=DEFAULT_ABORT_ABOVE_C)
    parser.add_argument("--duty-sleep", type=float, default=0.05,
                        help="seconds of sleep per --duty-every rows")
    parser.add_argument("--duty-every", type=int, default=2000)
    parser.add_argument("--sample-seconds", type=int, default=20,
                        help="thermal sampling interval; lower reacts faster "
                             "but costs a little CPU")
    parser.add_argument("--feature-checkpoint-every", type=int, default=400_000,
                        help="rows between mid-file feature checkpoints")
    parser.add_argument("--no-thermal-guard", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="ignore validated checkpoints and recompute")
    parser.add_argument("--rerun-report", action="store_true",
                        help="rebuild the report from saved results")
    args = parser.parse_args(argv)

    step("BUSINESS ENTITY RESOLUTION - 20k matching A/B")
    step(f"python {sys.version.split()[0]}   numpy {np.__version__}")
    try:
        import sklearn
        step(f"scikit-learn {sklearn.__version__}")
    except Exception:
        step("scikit-learn not importable yet")
    step(f"project {REPO}")
    step(f"checkpoints {CKPT}")
    step(f"BLAS/OpenMP threads capped to {BLAS_THREADS} before numpy import")

    CKPT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    PRIORITY = set_low_priority(below_normal=True)
    CPUS_ALLOWED = pin_to_cpus(default_cpu_mask(args.cpus)) or "failed"
    time.sleep(1.0)                      # let priority/affinity settle
    step(f"priority={PRIORITY}  cpus={CPUS_ALLOWED} of {os.cpu_count()} logical")

    if not args.no_thermal_guard:
        GUARD = ThermalGuard(
            pause_above=args.pause_above, resume_below=args.resume_below,
            abort_above=args.abort_above, duty_sleep=args.duty_sleep,
            duty_every=args.duty_every,
            sample_seconds=max(1, int(args.sample_seconds)),
        )
        now = GUARD.temp()
        step(f"thermal guard active, current CPU "
             f"{'unavailable' if now is None else f'{now:.1f} C'}; "
             f"pause >= {args.pause_above:.0f} C, resume <= "
             f"{args.resume_below:.0f} C, abort >= {args.abort_above:.0f} C")

    order = list(STAGES[:-1]) if args.stage == "all" else [args.stage]
    try:
        s1_records: dict | None = None
        meta: dict | None = None
        model: dict | None = None
        for stage in order:
            if stage == "dataset":
                if TRAIN_DIR is None:
                    stage1_dataset(args)
                continue
            # Everything below reads the TSVs and the earlier checkpoints, so
            # make sure the dataset and subset are resolved exactly once.
            if TRAIN_DIR is None:
                stage1_dataset(args)
            if s1_records is None:
                s1_records, _ = stage2_subset(args)
            if stage == "subset":
                continue
            if meta is None:
                meta = read_candidate_meta()
                if meta is None:
                    meta = stage3_candidates(args, s1_records)
            if stage == "candidates":
                continue
            if stage == "features":
                stage4_features(args, meta, s1_records)
                continue
            if model is None:
                model = stage5_model(args)
            if stage == "model":
                continue
            stage6_report(args, meta, model, s1_records)

        if args.stage == "all" and meta is not None and model is not None:
            print_finisher(meta, model)
    except ThermalAbort as exc:
        step(f"THERMAL STOP: {exc}")
        step(f"everything completed so far is checkpointed in {CKPT}; "
             f"rerun the same command to resume")
        return 2
    except KeyboardInterrupt:
        step("interrupted by the user; checkpointed work is kept, rerun to resume")
        return 130
    finally:
        if GUARD is not None:
            GUARD.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
