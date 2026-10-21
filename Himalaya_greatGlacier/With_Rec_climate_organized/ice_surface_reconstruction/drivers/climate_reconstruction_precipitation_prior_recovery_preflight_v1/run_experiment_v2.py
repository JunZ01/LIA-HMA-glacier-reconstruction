"""Run the precipitation-prior preflight with the corrected MATLAB launch path."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    source = Path(__file__).with_name("run_experiment.py")
    spec = importlib.util.spec_from_file_location("precipitation_prior_preflight_v1", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-load-failed:{source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    experiment = load_module()
    original_run = experiment.subprocess.run

    def corrected_run(command, *args, **kwargs):
        corrected = [
            item.replace("launch.m", "launch_v2.m") if isinstance(item, str) else item
            for item in command
        ]
        return original_run(corrected, *args, **kwargs)

    experiment.subprocess.run = corrected_run
    return experiment.main()


if __name__ == "__main__":
    raise SystemExit(main())

