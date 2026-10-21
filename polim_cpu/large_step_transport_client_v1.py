"""GPU-safe client for the isolated large-step SciPy transport service."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import numpy as np


class IsolatedTransportService:
    def __init__(
        self,
        static_arrays,
        work_root: str | Path,
        *,
        helper_python: str | Path = "d:/ProgramData/Anaconda3/python.exe",
    ):
        self.root = Path(work_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.static_path = self.root / "transport-static.npz"
        if not self.static_path.exists():
            with self.static_path.open("xb") as stream:
                np.savez(stream, **static_arrays)
        project = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment.pop("POLIM_TORCH_SITE", None)
        environment.pop("KMP_DUPLICATE_LIB_OK", None)
        environment["PYTHONPATH"] = str(project)
        self.stderr = (self.root / "transport-service-stderr.log").open(
            "a", encoding="utf-8"
        )
        self.process = subprocess.Popen(
            [
                str(helper_python),
                "-m",
                "polim_cpu.large_step_transport_service_v1",
                str(self.static_path),
            ],
            cwd=project,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
        )

    def _call(self, operation: str, **arrays) -> dict[str, np.ndarray]:
        token = uuid.uuid4().hex
        request = self.root / f"request-{token}.npz"
        output = self.root / f"response-{token}.npz"
        try:
            with request.open("xb") as stream:
                np.savez(stream, **arrays)
            command = {"operation": operation, "request": str(request), "output": str(output)}
            self.process.stdin.write(json.dumps(command) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("isolated transport service exited without a reply")
            answer = json.loads(line)
            if not answer["ok"]:
                raise RuntimeError(
                    f"isolated transport service failed: {answer['error']}\n"
                    f"{answer['traceback']}"
                )
            with np.load(output, allow_pickle=False) as z:
                return {name: z[name] for name in z.files}
        finally:
            request.unlink(missing_ok=True)
            output.unlink(missing_ok=True)

    def predict(self, thickness_old, velocity_old, bed, smb_requested, dt_years):
        return self._call(
            "predict",
            thickness_old=thickness_old,
            velocity_old=velocity_old,
            bed=bed,
            smb_requested=smb_requested,
            dt_years=np.array(dt_years),
        )["thickness"]

    def correct(self, thickness_old, candidate_thickness, velocity, smb_requested, dt_years):
        return self._call(
            "correct",
            thickness_old=thickness_old,
            candidate_thickness=candidate_thickness,
            velocity=velocity,
            smb_requested=smb_requested,
            dt_years=np.array(dt_years),
        )

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            self.process.wait(timeout=30)
        self.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
