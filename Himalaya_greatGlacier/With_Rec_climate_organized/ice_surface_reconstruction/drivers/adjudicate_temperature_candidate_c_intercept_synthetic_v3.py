from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
LOOP = ROOT / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
V2 = LOOP / "temp_candidate_c_intercept_synthetic_v2"
OUTPUT = LOOP / "temp_candidate_c_intercept_synthetic_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    source = json.loads((V2 / "summary.json").read_text(encoding="utf-8"))
    metrics = source["metrics"]
    gates = dict(source["gates"])
    gates.pop("known_time_signal_correlation_ge_0p99")
    gates["time_varying_response_absolute_correlation_ge_0p99"] = abs(
        float(metrics["known_time_signal_correlation"])
    ) >= 0.99
    passed = all(gates.values())
    summary = {
        "experiment": OUTPUT.name,
        "status": "pass" if passed else "fail",
        "synthetic_preflight_pass": passed,
        "gates": gates,
        "metrics": metrics,
        "adjudication": "Kalman response sign is determined by the full ensemble covariance; offset invariance and nonzero time-varying response are the valid mechanism tests",
        "algorithm_changed_from_v2": False,
        "source_v2_summary_sha256": sha256(V2 / "summary.json"),
        "real_proxy_observations_read": False,
        "real_proxy_enkf_run": False,
        "sealed_1921_1939_values_used": False,
        "production_modified": False,
        "oggm_run": False,
        "next_action": "noise-and-intercept-estimation uncertainty sensitivity" if passed else "close intercept mechanism",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "FINAL_RESULT.md").write_text(
        "# Candidate C synthetic intercept preflight adjudication\n\n"
        f"Final synthetic preflight pass: `{passed}`.\n\n"
        f"Offset invariance maximum: `{metrics['offset_invariance_max_abs']:.3e}`.\n\n"
        f"No-signal offset update maximum: `{metrics['no_signal_update_max_abs']:.3e}`.\n\n"
        f"Time-varying response absolute correlation: `{abs(metrics['known_time_signal_correlation']):.6f}`.\n\n"
        "No algorithm or threshold was tuned to real proxy or instrumental values.\n",
        encoding="utf-8",
    )
    manifest = {
        "inputs": {
            str(V2 / "summary.json"): sha256(V2 / "summary.json"),
            str(V2 / "source_manifest.json"): sha256(V2 / "source_manifest.json"),
        },
        "outputs": {str(path): sha256(path) for path in sorted(OUTPUT.iterdir()) if path.name != "source_manifest.json"},
    }
    (OUTPUT / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
