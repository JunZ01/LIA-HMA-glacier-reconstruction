from __future__ import annotations

import json
from pathlib import Path

import prepare_temperature_candidate_b2_stage_c_static_package_v1 as v1


DRIVER = Path(__file__).resolve()
V1_DRIVER = Path(v1.__file__).resolve()
v1.OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v2"
v1.PACKAGE = v1.OUT / "candidate_b2_stage_c_static_input_v2.npz"


def main() -> None:
    v1.main()
    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gates = summary["gates"]
    required_true = (
        "prior_hash_exact",
        "allowed_year_axis_exact",
        "sealed_year_absent",
        "proxy6_selected_count_201",
        "proxy9_selected_count_290",
        "operator_centers_finite",
    )
    summary.update(
        {
            "experiment": v1.OUT.name,
            "static_package_pass": all(gates[name] for name in required_true)
            and gates["real_proxy_data_dataset_read"] is False
            and gates["stage_c_enkf_run"] is False,
            "supersedes_technical_failure": (
                "temperature_balanced_pooled_candidate_b2_stage_c_static_package_v1"
            ),
            "v2_change": (
                "treat false real-data-read and false EnKF-run safety states as required passes"
            ),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    manifest_path = v1.OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_v1"] = {
        "path": str(V1_DRIVER),
        "sha256": v1.sha256(V1_DRIVER),
    }
    manifest["active_wrapper_v2"] = {
        "path": str(DRIVER),
        "sha256": v1.sha256(DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not summary["static_package_pass"]:
        raise RuntimeError("static-package-v2-gate-failed")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
