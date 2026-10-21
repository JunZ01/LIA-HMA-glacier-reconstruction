"""Synthesize the verified Round 161-165 temperature-family mechanism results."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_ID = "round_166_temperature_family_decision_synthesis_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    driver = Path(__file__).resolve().parent
    for parent in driver.parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            root = parent
            break
    else:
        raise RuntimeError("project-root-not-found")
    base = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = base / EXPERIMENT_ID
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    contract = driver / "round166_temperature_family_decision_synthesis_v1_contract.md"
    shutil.copy2(contract, output / "ROUND_CONTRACT.md")
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "contract": contract,
        "driver": Path(__file__).resolve(),
        "round161": base / "round_161_temperature_perfect_model_twin_v2/summary.json",
        "round162": base / "round_162_temperature_twin_information_geometry_v1/summary.json",
        "round163": base / "round_163_temperature_psm_mode_observability_v1/summary.json",
        "round164": base / "round_164_temperature_twin_residual_variance_ablation_v1/summary.json",
        "round165": base / "round_165_temperature_twin_joint_network_fusion_v1/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing:" + ";".join(missing))
    before = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    results = {name: json.loads(paths[name].read_text(encoding="utf-8")) for name in ("round161", "round162", "round163", "round164", "round165")}

    r162 = results["round162"]["primary_noise_free_equal_network_hma"]
    r164 = results["round164"]
    r165 = results["round165"]
    rows = [
        {"mechanism": "sampled_prior_representation", "evidence": f"prior99 oracle skill={r162['prior_99_oracle_skill']:.3f}; full={r162['prior_full_oracle_skill']:.3f}", "decision": "not_primary_limit"},
        {"mechanism": "uniform_increment_or_gain_scaling", "evidence": f"actual={r162['actual_state_skill']:.3f}; oracle scalar={r162['oracle_scalar_state_skill']:.3f}", "decision": "closed"},
        {"mechanism": "top100_adaptive_prior_selection", "evidence": "Round161 adaptive recovery is worse than ordinary", "decision": "closed"},
        {"mechanism": "residual_variance_reduction", "evidence": f"archived={r164['archived_R_equal_network_hma_ratio_of_median_rmse_skill']:.3f}; near-zero={r164['near_zero_R_equal_network_hma_ratio_of_median_rmse_skill']:.3f}", "decision": "diagnostic_only_closed_for_production"},
        {"mechanism": "joint_proxy6_proxy9_assimilation", "evidence": f"archived={r165['archived_R']['joint_skill']:.3f}; near-zero={r165['near_zero_R']['joint_skill']:.5f}", "decision": "diagnostic_only_closed_for_production"},
        {"mechanism": "current_archived_temperature_family", "evidence": "perfect-model gate and verified modern OOF family gates fail", "decision": "closed"},
        {"mechanism": "new_raw_temperature_evidence_or_chronology", "evidence": "required by canonical Sections 289 and 301", "decision": "admissible_next_route"},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "mechanism_decision_matrix.csv", index=False)

    next_contract = "# Next Temperature Evidence Route\n\n"
    next_contract += "## Priority\n\nAcquire genuinely new raw temperature-sensitive evidence in central (`80-90 E`) or eastern (`>=90 E`) HMA; central HMA is first priority because it remains the weakest perfect-model region.\n\n"
    next_contract += "## Minimum identity and chronology gate\n\n- Raw chronology or measurement series, not a same-source published reconstruction.\n- No exact/near-exact alias with proxy6 or proxy9.\n- Documented temperature interpretation and season.\n- LIA coverage reaching at least 1500 and at least 20 allowed modern observations in 1940-2000.\n- Locked 1921-1939 values skipped before numeric decoding.\n\n"
    next_contract += "## Modern gate before paleo inspection\n\n- Blocked 1940-2000 OOF against both Berkeley Earth and CRU.\n- Positive documented orientation and split-period sign stability.\n- OOF correlation at least 0.30 for both products, positive skill, and no material cross-product regression.\n- Do not inspect the candidate paleo trajectory unless every frozen gate passes.\n\n"
    next_contract += "## Consequence\n\nUntil such evidence passes, retain the current six-member reconstruction as legacy experimental forcing and the four-family product only as an HMA regional 50-year comparison curve. Do not create a corrected gridded production temperature field.\n"
    (output / "NEXT_ROUTE_CONTRACT.md").write_text(next_contract, encoding="utf-8")

    after = {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()}
    unchanged = before == after
    if not unchanged:
        raise RuntimeError("source-hash-changed")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": "close_archived_temperature_family_advance_only_new_evidence_or_chronology",
        "instrumental_truth_twin_authorized": False,
        "production_reconstruction_authorized": False,
        "oggm_authorized": False,
        "next_route": "new_raw_central_or_eastern_hma_temperature_evidence",
        "source_hashes_unchanged": unchanged,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps({"before": before, "after": after, "unchanged": unchanged}, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 166 Temperature Family Decision\n\n"
        "- Close the archived prior/PSM/operator family for further repair.\n"
        "- Do not run an instrumental-truth twin or production reconstruction from this family.\n"
        "- Advance only genuinely new raw central/eastern HMA temperature evidence or chronology through the frozen modern gate.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
