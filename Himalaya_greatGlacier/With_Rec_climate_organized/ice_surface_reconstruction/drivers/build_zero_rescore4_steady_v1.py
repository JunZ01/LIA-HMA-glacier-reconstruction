"""Register four corrected zero-MB summaries and build their 50-year steady configs.

Preparation only: no OGGM model is imported or advanced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


CASES = {
    2328: ("w228zero2328v1", "w227inv2328v1"),
    2547: ("w274zero2547mindt30v1", "w264inv2547v1"),
    2580: ("w273zero2580v1", "w261inv2580v1"),
    5443: ("w274zero5443mindt30v1", "w270inv5443v1"),
}
RESCORE_SUB = (
    "validation_data/results/paleo_surface_validation_v0/"
    "lia_191_zero_mb_legacy_four_formal_rescore_v1"
)
CONFIG_SUB = (
    "Himalaya_greatGlacier/With_Rec_climate_organized/"
    "ice_surface_reconstruction/config/"
    "synthetic_lia_system_standard_inversion_v1"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    rescore_root = root / RESCORE_SUB
    config_root = root / CONFIG_SUB
    manifest_path = root / "w725zero_rescore4_steady_manifest_v1.json"
    require(not manifest_path.exists(), f"collision:{manifest_path}")
    records = []

    for sid, (source_zero_name, inversion_name) in CASES.items():
        source_zero = root / source_zero_name
        inversion_root = root / inversion_name
        rescore_path = rescore_root / f"system_{sid}_formal_rescore.json"
        source_summary_path = source_zero / "zero_mb_summary_formal.json"
        source_config_path = source_zero / "config_snapshot.json"
        inversion_summary_path = inversion_root / "inversion_summary_formal.json"
        inversion_config_path = inversion_root / "config_snapshot.json"
        for path in (
            rescore_path,
            source_summary_path,
            source_config_path,
            inversion_summary_path,
            inversion_config_path,
        ):
            require(path.is_file(), f"missing:{path}")

        rescore = read_json(rescore_path)
        source_summary = read_json(source_summary_path)
        source_config = read_json(source_config_path)
        inversion_summary = read_json(inversion_summary_path)
        inversion_config = read_json(inversion_config_path)
        require(int(rescore["system_id"]) == sid, f"rescore-id:{sid}")
        require(rescore["formal_status"] == "zero-mb-pass", f"rescore-status:{sid}")
        require(not rescore["formal_review_flags"], f"rescore-flags:{sid}")
        require(rescore["source_summary_sha256"] == sha(source_summary_path), f"source-hash:{sid}")
        require(inversion_summary["status"] == "inversion-pass", f"inversion-status:{sid}")
        require(int(inversion_config["system_id"]) == sid, f"inversion-id:{sid}")
        require(list(inversion_config["spinup_window"]) == [1441, 1470], f"window:{sid}")
        require(inversion_config["stage2"]["prcp_anomaly_mode"] == "off", f"prcp:{sid}")

        overlay_name = f"w725zero_rescore_overlay_{sid}_v1"
        steady_name = f"w725steady_parent_rescore4_{sid}_v1"
        overlay_root = root / overlay_name
        steady_root = root / steady_name
        config_path = config_root / f"steady_parent_rescore4_{sid}_v1.json"
        for path in (overlay_root, steady_root, config_path):
            require(not path.exists(), f"collision:{path}")

        corrected = dict(source_summary)
        corrected["status"] = "zero-mb-pass"
        corrected["review_flags"] = []
        corrected["formal_rescore"] = {
            "version": "legacy-four-per-step-closure-v1",
            "source_root": source_zero_name,
            "source_summary_sha256": rescore["source_summary_sha256"],
            "criterion": rescore["closure_gate"],
            "maximum_absolute_step_closure_residual_m3": rescore[
                "maximum_absolute_step_closure_residual_m3"
            ],
            "per_step_closure_limit_m3": rescore["per_step_closure_limit_m3"],
            "rescore_evidence": str(rescore_path.relative_to(root)),
        }
        write_new(overlay_root / "zero_mb_summary_formal.json", corrected)
        write_new(overlay_root / "config_snapshot.json", source_config)

        steady_config = {
            "experiment": "conservative-one-way-tributary-fluxbased-v1",
            "stage": "remaining-route-zero-rescore4-parent-steady-50y-v1",
            "system_id": sid,
            "synthetic_rgi_id": inversion_config["synthetic_rgi_id"],
            "inversion_root": {"anchor": "project", "sub": inversion_name},
            "zero_mb_root": {"anchor": "project", "sub": overlay_name},
            "output_root": {"anchor": "project", "sub": steady_name},
            "climate_filesuffix": "_batch1_1441_1470_prcp_off",
            "spinup_window": [1441, 1470],
            "registered_lia_length_m": float(
                inversion_config["registered_lia_length_m"]
            ),
            "stage2": inversion_config["stage2"],
            "gates": {
                "volume_change_fraction_max": 0.02,
                "lia_length_relative_error_max": 0.10,
                "minimum_free_downstream_m": 5000.0,
            },
            "authorization": {"historical": False},
        }
        write_new(config_path, steady_config)
        records.append(
            {
                "system_id": sid,
                "source_zero_root": source_zero_name,
                "zero_rescore_overlay_root": overlay_name,
                "inversion_root": inversion_name,
                "steady_config": str(config_path.relative_to(root)),
                "steady_output_root": steady_name,
                "delta_t_c": float(inversion_config["stage2"]["delta_t_c"]),
                "rescore_sha256": sha(rescore_path),
                "corrected_zero_summary_sha256": sha(
                    overlay_root / "zero_mb_summary_formal.json"
                ),
            }
        )

    manifest = {
        "status": "configs-built-not-run",
        "count": len(records),
        "historical_authorized": False,
        "oggm_started": False,
        "records": records,
    }
    write_new(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
