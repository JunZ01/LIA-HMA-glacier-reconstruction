from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
LOOP = (
    ROOT
    / "validation_data/results/paleo_surface_validation_v0"
    / "climate_reconstruction_improvement_loop_v2"
)
SOURCE = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v2"
OUT = LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_launch_preflight_v3"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    source_summary_path = SOURCE / "summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    if not source_summary.get("preauthorization_launch_ready"):
        raise RuntimeError("source-preflight-not-ready")
    for name in (
        "launch_manifest.json",
        "round277_full_value_closure.json",
        "AUTHORIZATION_RECEIPT_SCHEMA.md",
    ):
        shutil.copy2(SOURCE / name, OUT / name)

    age_axis = source_summary.pop("proxy_availability_mask_audit")
    for record in age_axis.values():
        record["interpretation"] = (
            "age-axis support only; finite real-observation availability was not read"
        )
        record["finite_observation_mask_known"] = False
    source_summary.update(
        {
            "experiment": OUT.name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "supersedes": SOURCE.name,
            "v3_correction": (
                "age_yng support cannot establish finite data availability; real active masks remain unknown until authorized value read"
            ),
            "proxy_age_axis_support_audit": age_axis,
            "finite_real_proxy_active_masks_known": False,
            "real_proxy_observations_read": False,
            "authorization_missing": True,
        }
    )
    (OUT / "summary.json").write_text(
        json.dumps(source_summary, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "FINAL_PREAUTHORIZATION_REPORT.md").write_text(
        "# Candidate B2 Stage C preauthorization report v3\n\n"
        f"Launch preparation pass: `{source_summary['preauthorization_launch_ready']}`.\n\n"
        f"Frozen launch fingerprint: `{source_summary['launch_fingerprint_sha256']}`.\n\n"
        "Static-input closure, synthetic EnKF closure, Round 277 `/100` value "
        "closure, sealed-period protection, and evaluation inputs pass. `age_yng` "
        "confirms only age-axis support; finite real-proxy active masks remain "
        "unknown because no proxy `data` value has been read.\n\n"
        "No real proxy value has been read and no real-data EnKF has run. Explicit "
        "user authorization remains the only launch boundary.\n",
        encoding="utf-8",
    )
    manifest = {
        "source_v2_summary": {
            "path": str(source_summary_path),
            "sha256": core.sha256(source_summary_path),
        },
        "source_v2_launch_manifest": {
            "path": str(SOURCE / "launch_manifest.json"),
            "sha256": core.sha256(SOURCE / "launch_manifest.json"),
        },
        "correction_driver": {"path": str(DRIVER), "sha256": core.sha256(DRIVER)},
        "real_proxy_observations_read": False,
        "stage_c_enkf_run": False,
        "sealed_period_values_used": False,
        "prohibited_source_accessed": False,
    }
    (OUT / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(source_summary, indent=2))


if __name__ == "__main__":
    main()
