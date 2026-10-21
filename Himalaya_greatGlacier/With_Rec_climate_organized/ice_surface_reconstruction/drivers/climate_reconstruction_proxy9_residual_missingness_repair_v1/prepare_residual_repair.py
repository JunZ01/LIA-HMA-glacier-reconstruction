"""Prepare a controlled RegEM input with structural residual zeros restored to NaN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("experiment_config.json"))
    args = parser.parse_args()
    root = project_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    proxy_path = Path(cfg["archive_root"]) / cfg["sources"]["proxy9"]
    output = root / cfg["result_root"]
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)

    start, end = cfg["periods"]["calibration"]
    calibration_years = np.arange(start, end + 1)
    rows: list[dict[str, object]] = []
    with h5py.File(proxy_path, "r") as archive:
        references = np.asarray(archive["proxy_data/full_proxy"]).reshape(-1, order="F")
        raw = np.full((len(references), len(calibration_years)), np.nan)
        repaired = np.full_like(raw, np.nan)
        availability = np.zeros_like(raw, dtype=bool)
        for index, reference in enumerate(references):
            proxy = archive[reference]
            residual = np.asarray(proxy["Rvec"], dtype=float).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"], dtype=int).reshape(-1, order="F")
            values = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            if residual.size != calibration_years.size:
                raise ValueError(f"record-{index + 1}-residual-size:{residual.size}")
            by_year = {int(year): bool(np.isfinite(value)) for year, value in zip(ages, values)}
            valid = np.asarray([by_year.get(int(year), False) for year in calibration_years])
            raw[index] = residual
            repaired[index] = residual
            repaired[index, ~valid] = np.nan
            availability[index] = valid
            structural = ~valid
            unexpected_nonzero = structural & np.isfinite(residual) & ~np.isclose(residual, 0.0, atol=1e-15)
            observed_changed = valid & ~np.isclose(repaired[index], residual, equal_nan=True)
            rows.append({
                "proxy9_record_1based": index + 1,
                "effective_calibration_count": int(valid.sum()),
                "missing_calibration_count": int(structural.sum()),
                "raw_zero_count": int(np.isfinite(residual).sum() - np.count_nonzero(residual)),
                "structural_zero_count": int(np.sum(structural & np.isclose(residual, 0.0, atol=1e-15))),
                "structural_nonzero_count": int(unexpected_nonzero.sum()),
                "observed_entry_changed_count": int(observed_changed.sum()),
                "raw_residual_variance": float(np.nanvar(residual, ddof=1)),
                "observed_only_residual_variance": float(np.nanvar(residual[valid], ddof=1)) if valid.sum() > 1 else np.nan,
            })

    frame = pd.DataFrame(rows)
    expected_shape = (int(cfg["gates"]["proxy_count"]), int(cfg["gates"]["calibration_year_count"]))
    if raw.shape != expected_shape:
        raise ValueError(f"unexpected-shape:{raw.shape};expected:{expected_shape}")
    if int(frame.structural_nonzero_count.sum()) != 0:
        raise RuntimeError("missing-observation years contain nonzero archived residuals")
    if int(frame.observed_entry_changed_count.sum()) != 0:
        raise RuntimeError("observed residual changed during preparation")

    scipy.io.savemat(
        output / "regem_input.mat",
        {"R0v": repaired, "R0v_archived": raw, "observation_available": availability.astype(np.uint8), "calibration_years": calibration_years},
        do_compression=False,
    )
    frame.to_csv(output / "residual_missingness_preflight.csv", index=False)
    manifest = {
        "experiment_id": cfg["experiment_id"],
        "proxy9": {"path": str(proxy_path), "sha256": sha256(proxy_path), "size_bytes": proxy_path.stat().st_size},
        "shape": list(raw.shape),
        "structural_zero_entries_replaced_with_nan": int((~availability & np.isclose(raw, 0.0, atol=1e-15)).sum()),
        "preexisting_nan_entries": int(np.isnan(raw).sum()),
        "repaired_nan_entries": int(np.isnan(repaired).sum()),
        "records_affected": int((frame.missing_calibration_count > 0).sum()),
        "records_with_variance_dilution_below_0p8": int(((frame.raw_residual_variance / frame.observed_only_residual_variance) < 0.8).sum()),
        "input_mat_sha256": sha256(output / "regem_input.mat"),
    }
    (output / "input_manifest_before.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
