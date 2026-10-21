from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = _project_root()
    config_default = Path(__file__).with_name("spatial_export_audit_config.json")
    output_default = (
        root
        / "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_increment_attribution_v1/inputs"
    )
    parser.add_argument("--config", type=Path, default=config_default)
    parser.add_argument("--output-dir", type=Path, default=output_default)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(config["fixed_result"])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as handle:
        data = handle["Xa_m_all"]
        status = np.asarray(handle["experiment_status"]).ravel()
        active = np.flatnonzero(status == 1)
        if data.shape != (50, 1001, 336) or active.size != 50:
            raise ValueError(
                f"unexpected-fixed-ensemble:{data.shape}:{active.size}-active"
            )
        total = np.zeros((1001, 336), dtype=float)
        for index in active:
            total += data[index]
        ensemble_mean = (total / active.size).T
        experiment_1 = np.asarray(data[0], dtype=float).T

    output_path = output / "fixed_o6_50_experiment_mean.npz"
    np.savez_compressed(
        output_path,
        years=np.arange(1000, 2001, dtype=int),
        ensemble_mean=ensemble_mean,
        experiment_1=experiment_1,
        active_experiment_indices_1based=active + 1,
    )
    manifest = {
        "source_mode": "read-only",
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_size_bytes": source.stat().st_size,
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "source_shape": list(data.shape),
        "active_experiment_count": int(active.size),
    }
    (output / "fixed_o6_export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
