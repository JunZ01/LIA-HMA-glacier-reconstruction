"""Isolated, strictly serial child processes; imports only the standard library.

No model or command is selected here. The production entry point must supply a
fresh authorization check for each phase and a frozen GPU-result verifier.
Workers must not spawn model descendants: this component owns only its direct
child and never enumerates, terminates, or alters any external process.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Callable, Mapping, Sequence
import uuid

SS4_PYTHON = Path("D:/Anaconda/envs/SS4/python.exe")
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
    "OMP_THREAD_LIMIT", "OPENBLAS_DEFAULT_NUM_THREADS",
)


def _base_environment() -> dict[str, str]:
    env = dict(os.environ)
    for name in list(env):
        upper = name.upper()
        if upper == "KMP_DUPLICATE_LIB_OK" and env[name].strip():
            raise RuntimeError("KMP_DUPLICATE_LIB_OK is forbidden, not a recovery option")
        if upper in {"KMP_DUPLICATE_LIB_OK", "PYTHONHOME", "PYTHONPATH",
                     "PYTHONUSERBASE", "PYTHONSTARTUP"}:
            del env[name]
        elif upper in set(THREAD_VARIABLES) | {"PATH"}:
            del env[name]
    env.update({name: "1" for name in THREAD_VARIABLES})
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    return env


def build_gpu_environment(project_root: str | Path) -> dict[str, str]:
    """The verified SS4/project-Torch DLL search path; no imports or launch.

    The GPU worker must separately perform the frozen import-time stubs filter,
    verify the single loaded OpenMP DLL, and set Torch intra/inter-op threads to
    one. An environment dictionary alone cannot establish those properties.
    """
    env = _base_environment()
    root = Path(project_root).resolve()
    site = root / ".runtime/torch-cu128-v1/site-packages"
    windows = Path(os.environ.get("SystemRoot", "C:/Windows"))
    env["PATH"] = ";".join(map(str, (site / "torch/lib", windows / "System32", windows)))
    env["PYTHONPATH"] = str(site)
    # Retain the existing frozen GPU worker's required environment marker.
    env["LIA_191_GPU_SPINUP_V1_CHILD"] = "1"
    return env


def build_cpu_environment(project_root: str | Path) -> dict[str, str]:
    """A fresh SS4 CPU scorer: CUDA hidden and no project-Torch PYTHONPATH."""
    env = _base_environment()
    windows = Path(os.environ.get("SystemRoot", "C:/Windows"))
    env["PATH"] = ";".join(map(str, (SS4_PYTHON.parent / "Library/bin",
                                       windows / "System32", windows)))
    env["PYTHONPATH"] = str(Path(project_root).resolve() /
        "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers")
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.pop("LIA_191_GPU_SPINUP_V1_CHILD", None)
    return env


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _exclusive_json(path: Path, value: dict) -> None:
    """Publish a complete, fsynced immutable record without replacing a file."""
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # An existing target is an error, including a competing publisher.
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _authorize(callback: Callable[[str], bool], phase: str) -> None:
    if not callable(callback) or callback(phase) is not True:
        raise PermissionError("fresh explicit phase authorization required: " + phase)


def _stop_owned_child(child: subprocess.Popen) -> None:
    """Only operate on the Popen object this invocation itself created."""
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
    child.wait()


def run_phase(command: Sequence[str], env: Mapping[str, str],
              phase_root: str | Path, timeout_seconds: float, *,
              authorize: Callable[[str], bool], phase: str = "gpu",
              cwd: str | Path | None = None) -> dict:
    """Wait for one owned child and durably record exit, timing and log hashes.

    A failure is returned as evidence, never retried. A denied authorization or
    existing output directory raises before Popen. Standard streams go to disk,
    avoiding pipe deadlock and unbounded in-memory logging. No shell is used.
    """
    if (isinstance(command, (str, bytes)) or not command or
            any(not isinstance(arg, str) or not arg for arg in command)):
        raise ValueError("command must be a nonempty argument sequence")
    if isinstance(timeout_seconds, bool) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("a positive finite timeout is required")
    if phase not in {"gpu", "score"}:
        raise ValueError("phase must be gpu or score")
    _authorize(authorize, phase)
    root = Path(phase_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    command = list(command)
    child_env = dict(env)
    started = time.monotonic()
    record = dict(schema="lia-gpu-production-phase-v1", phase=phase,
                  command=command, pid=None, exit_code=None,
                  started_at_utc=_now(), started_monotonic_ns=time.monotonic_ns(),
                  status="launch-failure", timed_out=False, child_exited=False,
                  automatic_retry=False, output_root=str(root))
    _exclusive_json(root / "attempt.json", record)
    child = None
    interrupted = None
    with (root / "stdout.log").open("xb", buffering=0) as out, \
            (root / "stderr.log").open("xb", buffering=0) as err:
        try:
            # Revalidate immediately before process creation, after reserving
            # the immutable attempt directory. Never consume a stale guard.
            _authorize(authorize, phase)
            child = subprocess.Popen(command, cwd=cwd, env=child_env,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err, shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            record["pid"] = child.pid
            _exclusive_json(root / "started.json", dict(record, status="running"))
            try:
                child.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                record["timed_out"] = True
                _stop_owned_child(child)
            record["status"] = ("timeout" if record["timed_out"] else
                                "completed" if child.returncode == 0 else "technical-failure")
        except BaseException as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
            record["status"] = "technical-failure" if child else "launch-failure"
            if child is not None:
                _stop_owned_child(child)
            if not isinstance(exc, Exception):
                interrupted = exc
        finally:
            if child is not None:
                record["exit_code"] = child.wait()
                record["child_exited"] = True
            os.fsync(out.fileno())
            os.fsync(err.fileno())
    record.update(finished_at_utc=_now(), finished_monotonic_ns=time.monotonic_ns(),
                  wall_seconds=time.monotonic() - started,
                  stdout_sha256=_sha256(root / "stdout.log"),
                  stderr_sha256=_sha256(root / "stderr.log"))
    _exclusive_json(root / "exit.json", record)
    record["exit_evidence_path"] = str(root / "exit.json")
    record["exit_evidence_sha256"] = _sha256(root / "exit.json")
    if interrupted is not None:
        raise interrupted
    return record


def run_serial(*, gpu_command: Sequence[str], gpu_env: Mapping[str, str],
               score_command: Sequence[str], score_env: Mapping[str, str],
               output_root: str | Path, timeout_seconds: float,
               authorize: Callable[[str], bool],
               verify_gpu_result: Callable[[dict], bool],
               cwd: str | Path | None = None) -> dict:
    """GPU exit -> frozen-result verification -> separately authorized scorer.

    timeout_seconds is a total budget, not a renewed budget for each phase.
    The caller's verifier must validate the complete frozen GPU manifest. Exit
    zero alone never authorizes scoring. This function does not import a scorer.
    """
    if not callable(verify_gpu_result):
        raise TypeError("a frozen GPU result-manifest verifier is required")
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError("serial output root already exists: " + str(root))
    started = time.monotonic()
    gpu = run_phase(gpu_command, gpu_env, root / "gpu", timeout_seconds,
                    authorize=authorize, phase="gpu", cwd=cwd)
    report = dict(schema="lia-gpu-production-serial-v1", status="gpu-failed",
                  gpu=gpu, score=None, scoring_started=False,
                  gpu_score_overlap=False, automatic_retry=False)
    try:
        if gpu["status"] != "completed" or not gpu["child_exited"] or gpu["exit_code"] != 0:
            return report
        if _sha256(Path(gpu["exit_evidence_path"])) != gpu["exit_evidence_sha256"]:
            raise RuntimeError("GPU exit evidence changed before scoring")
        if verify_gpu_result(gpu) is not True:
            raise RuntimeError("frozen GPU result manifest did not pass")
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            report["status"] = "timeout-before-score"
            return report
        score = run_phase(score_command, score_env, root / "score", remaining,
                          authorize=authorize, phase="score", cwd=cwd)
        report.update(score=score, scoring_started=score["pid"] is not None,
                      status="completed" if score["status"] == "completed" else "score-failed")
        if score["started_monotonic_ns"] < gpu["finished_monotonic_ns"]:
            raise RuntimeError("serial process ordering invariant violated")
        return report
    except Exception as exc:
        report.update(status="score-gate-failed", error=type(exc).__name__ + ": " + str(exc))
        return report
    finally:
        report.update(wall_seconds=time.monotonic() - started, finished_at_utc=_now())
        _exclusive_json(root / "serial-exit.json", report)
