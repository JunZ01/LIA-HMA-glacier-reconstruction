from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import decompose_temperature_candidate_b2_stage_c_increment_v1 as core


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
STATIC = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2/candidate_b2_stage_c_static_input_v2.npz"
REAL = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v3/candidate_b2_stage_c_real_proxy_output_v1.npz"
OUTPUT = LOOP / "temp_b2_increment_group_audit_v1"
PERIODS = ((1500, 1849), (1850, 1920), (1940, 2000), (1961, 1990))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def zone(longitude: float) -> str:
    if longitude < 80.0:
        return "West"
    if longitude < 90.0:
        return "Central"
    return "East"


def record_key(location: np.ndarray, coefficients: np.ndarray, ptype: str, method: str) -> tuple[object, ...]:
    return tuple(np.round(location, 8)) + tuple(np.round(coefficients, 10)) + (ptype, method)


def contribution_matrix(static: np.lib.npyio.NpzFile, real: np.lib.npyio.NpzFile, network: int, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observations = np.asarray(real[f"proxy{network}_centered_observations"], dtype=float)
    operator_mean = np.asarray(static[f"proxy{network}_balanced_operator_forced_mean"], dtype=float)
    operator_residual = np.asarray(static[f"proxy{network}_pooled_operator_residual_library"], dtype=float)
    variance = np.asarray(static[f"proxy{network}_residual_variance"], dtype=float)
    state_residual = np.asarray(static["pooled_state_residual_library"], dtype=float)
    innovation = observations - operator_mean
    contributions = np.zeros(observations.shape, dtype=float)
    masks: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for index, row in enumerate(observations):
        active = np.isfinite(row)
        key = np.packbits(active, bitorder="little").tobytes()
        if key not in masks:
            masks[key] = (active, [])
        masks[key][1].append(index)
    for active, indices in masks.values():
        if not active.any():
            continue
        gain = core.hma_gain(state_residual, operator_residual[:, active], variance[active], weights)
        rows = np.asarray(indices, dtype=int)
        contributions[np.ix_(rows, active)] = innovation[np.ix_(rows, active)] * gain
    return contributions, innovation


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    static = np.load(STATIC)
    real = np.load(REAL)
    years = np.asarray(real["years"], dtype=int)
    if np.any((years >= 1921) & (years <= 1939)):
        raise RuntimeError("sealed years present")
    weights = np.asarray(static["hma_weights"], dtype=float)
    weights /= weights.sum()

    metadata = {}
    matrices = {}
    innovations = {}
    keys = {}
    for network in (6, 9):
        locations = np.asarray(static[f"proxy{network}_locations"], dtype=float)
        coefficients = np.asarray(static[f"proxy{network}_psm_b0_b1_b2"], dtype=float)
        ptypes = np.asarray(static[f"proxy{network}_ptype"]).astype(str)
        methods = np.asarray(static[f"proxy{network}_psm_method"]).astype(str)
        metadata[network] = pd.DataFrame(
            {
                "proxy_position": np.arange(len(locations)),
                "latitude": locations[:, 0],
                "longitude": locations[:, 1],
                "zone": [zone(value) for value in locations[:, 1]],
                "b1_sign": np.where(coefficients[:, 1] >= 0, "b1_nonnegative", "b1_negative"),
                "b2_sign": np.where(coefficients[:, 2] >= 0, "b2_nonnegative", "b2_negative"),
                "ptype": ptypes,
                "psm_method": methods,
            }
        )
        matrices[network], innovations[network] = contribution_matrix(static, real, network, weights)
        keys[network] = [record_key(locations[i], coefficients[i], ptypes[i], methods[i]) for i in range(len(locations))]

    overlap = sorted(set(keys[6]) & set(keys[9]), key=str)
    overlap6 = np.asarray([key in set(overlap) for key in keys[6]])
    overlap9 = np.asarray([key in set(overlap) for key in keys[9]])
    metadata[6]["network_overlap"] = np.where(overlap6, "shared_exact", "unique")
    metadata[9]["network_overlap"] = np.where(overlap9, "shared_exact", "unique")

    rows = []
    dimensions = ("zone", "b1_sign", "b2_sign", "network_overlap")
    for network in (6, 9):
        for start, end in PERIODS:
            period = (years >= start) & (years <= end)
            for dimension in dimensions:
                for group in sorted(metadata[network][dimension].unique()):
                    members = metadata[network][dimension].eq(group).to_numpy()
                    annual = matrices[network][period][:, members].sum(axis=1)
                    active = np.isfinite(innovations[network][period][:, members])
                    rows.append(
                        {
                            "period": f"{start}-{end}",
                            "network": f"proxy{network}",
                            "dimension": dimension,
                            "group": group,
                            "proxy_count": int(members.sum()),
                            "mean_hma_contribution_degC": float(annual.mean()),
                            "mean_active_count": float(active.sum(axis=1).mean()),
                        }
                    )
    groups = pd.DataFrame(rows)
    groups.to_csv(OUTPUT / "increment_group_contributions.csv", index=False)
    metadata[6].to_csv(OUTPUT / "proxy6_group_metadata.csv", index=False)
    metadata[9].to_csv(OUTPUT / "proxy9_group_metadata.csv", index=False)

    modern = groups[groups["period"].eq("1961-1990")]
    dominant = modern.loc[modern.groupby(["network", "dimension"])["mean_hma_contribution_degC"].apply(lambda x: x.abs().idxmax())]
    dominant.to_csv(OUTPUT / "modern_anchor_dominant_groups.csv", index=False)

    summary = {
        "experiment": OUTPUT.name,
        "status": "complete",
        "proxy6_count": len(keys[6]),
        "proxy9_count": len(keys[9]),
        "exact_shared_record_count": len(overlap),
        "proxy6_shared_fraction": float(overlap6.mean()),
        "proxy9_shared_fraction": float(overlap9.mean()),
        "networks_are_independent_proxy_sets": False if overlap else True,
        "dominant_1961_1990_groups": dominant[["network", "dimension", "group", "mean_hma_contribution_degC"]].to_dict(orient="records"),
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "enkf_rerun": False,
        "oggm_run": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = f"""# Candidate B2 Stage C grouped increment audit

- Exact proxy identity overlap: {len(overlap)} records; {overlap6.mean():.1%} of proxy6 and {overlap9.mean():.1%} of proxy9.
- The two network results are therefore not independent confirmations.
- Grouped 1961-1990 contributions are reported by longitude zone, b1 sign, b2 sign, and exact network overlap status.

This audit diagnoses the shared modern-anchor update without deleting records or changing the frozen Stage C result.
"""
    (OUTPUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")
    manifest = {
        "inputs": {str(STATIC): sha256(STATIC), str(REAL): sha256(REAL)},
        "outputs": {str(path): sha256(path) for path in sorted(OUTPUT.iterdir()) if path.name != "source_manifest.json"},
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
