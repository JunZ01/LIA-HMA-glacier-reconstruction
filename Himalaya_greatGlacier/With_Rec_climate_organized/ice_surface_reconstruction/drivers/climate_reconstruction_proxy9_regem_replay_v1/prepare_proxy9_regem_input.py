"""Extract archived proxy9 residuals to a new RegEM input MAT file."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.io


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
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = _project_root()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source = Path(config["archive_root"]) / config["proxy_path"]
    output = root / config["result_root"]
    output.mkdir(parents=True, exist_ok=True)

    with h5py.File(source, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        residuals = np.vstack(
            [np.asarray(archive[reference]["Rvec"], dtype=float).reshape(-1, order="F") for reference in references]
        )
    gates = config["gates"]
    expected = (int(gates["expected_proxy_count"]), int(gates["expected_residual_years"]))
    if residuals.shape != expected:
        raise ValueError(f"unexpected-residual-shape:{residuals.shape};expected:{expected}")

    input_path = output / "regem_input.mat"
    scipy.io.savemat(input_path, {"R0v": residuals}, do_compression=False)
    manifest = {
        "experiment_id": config["experiment_id"],
        "source_mode": "read-only",
        "source": {"path": str(source), "sha256": _sha256(source), "size_bytes": source.stat().st_size},
        "regem_source": config["regem_source"],
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "shape": list(residuals.shape),
        "input_nan_count": int(np.isnan(residuals).sum()),
        "observed_entry_count": int(np.isfinite(residuals).sum()),
    }
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
