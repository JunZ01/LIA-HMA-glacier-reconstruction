"""Build a read-only, population-level extension design for the LIA 191 study.

The design replaces single-system ranking with stratified discovery,
confirmation, and expansion cohorts.  It starts no model process and grants no
launch authority.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DRIVER = Path(__file__).resolve()
ROOT = next(parent for parent in DRIVER.parents if (parent / "LIA_PROJECT_FACTS.md").exists())
BASE = ROOT / "validation_data/results/paleo_surface_validation_v0"
FEATURES = BASE / "lia_191_retained32_success_factor_extension_v1/system_feature_table_191.csv"
OUT = BASE / "lia_191_population_extension_cohorts_v1"


def stable_key(system_id: int) -> str:
    return hashlib.sha256(f"lia-population-cohort-v1:{system_id}".encode()).hexdigest()


def allocate_quotas(counts: pd.Series, total: int) -> dict[str, int]:
    exact = counts / counts.sum() * total
    quotas = np.floor(exact).astype(int)
    nonempty = counts[counts > 0].index
    quotas.loc[nonempty] = np.maximum(quotas.loc[nonempty], 1)
    while quotas.sum() > total:
        candidates = quotas[quotas > 1]
        key = min(candidates.index, key=lambda value: (exact[value] - quotas[value], value))
        quotas[key] -= 1
    while quotas.sum() < total:
        key = max(counts.index, key=lambda value: (exact[value] - quotas[value], counts[value], value))
        quotas[key] += 1
    return {str(key): int(value) for key, value in quotas.items()}


def main() -> int:
    if OUT.exists():
        raise FileExistsError(f"versioned-output-exists:{OUT}")
    OUT.mkdir(parents=True)

    table = pd.read_csv(FEATURES)
    pool = table[
        table["shape_group"].eq("remaining133")
        & table["target_direction"].eq("retreat")
        & table["outline_state"].eq("valid-single-polygon")
        & ~table["response_group"].eq("technical-floor-exhausted-not-retryable")
    ].copy()
    if len(pool) != 93:
        raise RuntimeError(f"eligible-pool-count:{len(pool)}")

    pool["size_class"] = pd.cut(
        pool["lia_length_km"], [-np.inf, 10.0, 14.0, np.inf],
        labels=["small_lt10", "medium_10_14", "large_ge14"], right=False,
    ).astype(str)
    pool["retreat_class"] = pd.cut(
        pool["target_retreat_fraction"], [-np.inf, 0.13, 0.18, np.inf],
        labels=["low_lt13pct", "core_13_18pct", "high_ge18pct"], right=False,
    ).astype(str)
    pool["primary_stratum"] = pool["rgi_region"].astype(str) + "_" + pool["retreat_class"]
    pool["stable_selection_key"] = pool["system_id"].astype(int).map(stable_key)

    counts = pool.groupby("primary_stratum").size().sort_index()
    quotas = allocate_quotas(counts, 32)
    selected_ids: list[int] = []
    for stratum, group in pool.groupby("primary_stratum", sort=True):
        chosen = group.sort_values("stable_selection_key").head(quotas[stratum])
        selected_ids.extend(chosen["system_id"].astype(int).tolist())
    if len(selected_ids) != 32 or len(set(selected_ids)) != 32:
        raise RuntimeError("pilot-selection-count-mismatch")

    pilot = pool[pool["system_id"].isin(selected_ids)].copy()
    pilot["balance_stratum"] = pilot["primary_stratum"] + "_" + pilot["size_class"]
    assignments: dict[int, str] = {}
    totals = {"discovery": 0, "confirmation": 0}
    for _, group in pilot.groupby("balance_stratum", sort=True):
        local = {"discovery": 0, "confirmation": 0}
        for row in group.sort_values("stable_selection_key").itertuples():
            if local["discovery"] < local["confirmation"]:
                choice = "discovery"
            elif local["confirmation"] < local["discovery"]:
                choice = "confirmation"
            else:
                choice = "discovery" if totals["discovery"] <= totals["confirmation"] else "confirmation"
            assignments[int(row.system_id)] = choice
            local[choice] += 1
            totals[choice] += 1
    if totals != {"discovery": 16, "confirmation": 16}:
        raise RuntimeError(f"pilot-balance-count-mismatch:{totals}")
    discovery_ids = {system_id for system_id, value in assignments.items() if value == "discovery"}
    confirmation_ids = {system_id for system_id, value in assignments.items() if value == "confirmation"}
    pool["cohort"] = "expansion_after_confirmation"
    pool.loc[pool["system_id"].isin(discovery_ids), "cohort"] = "discovery_global_mechanism"
    pool.loc[pool["system_id"].isin(confirmation_ids), "cohort"] = "heldout_independent_confirmation"
    pool["cohort_order"] = pool.groupby("cohort")["stable_selection_key"].rank(method="first").astype(int)

    cohort_order = {
        "discovery_global_mechanism": 1,
        "heldout_independent_confirmation": 2,
        "expansion_after_confirmation": 3,
    }
    pool["cohort_stage"] = pool["cohort"].map(cohort_order)
    pool = pool.sort_values(["cohort_stage", "cohort_order", "system_id"])

    columns = [
        "system_id", "rgi_id", "rgi_region", "cohort", "cohort_stage", "cohort_order",
        "primary_stratum", "size_class", "retreat_class", "lia_length_km", "rgi_length_km",
        "target_retreat_km", "target_retreat_fraction", "outline_state", "mechanism_class",
        "response_group", "stable_selection_key",
    ]
    pool[columns].to_csv(OUT / "population_extension_cohort_assignment.csv", index=False)

    summary_rows = []
    for cohort, group in pool.groupby("cohort", sort=False):
        summary_rows.append({
            "cohort": cohort,
            "count": len(group),
            "region13": int(group["rgi_region"].astype(str).eq("13").sum()),
            "region14": int(group["rgi_region"].astype(str).eq("14").sum()),
            "region15": int(group["rgi_region"].astype(str).eq("15").sum()),
            "small_lt10": int(group["size_class"].eq("small_lt10").sum()),
            "medium_10_14": int(group["size_class"].eq("medium_10_14").sum()),
            "large_ge14": int(group["size_class"].eq("large_ge14").sum()),
            "low_retreat": int(group["retreat_class"].eq("low_lt13pct").sum()),
            "core_retreat": int(group["retreat_class"].eq("core_13_18pct").sum()),
            "high_retreat": int(group["retreat_class"].eq("high_ge18pct").sum()),
            "median_lia_length_km": float(group["lia_length_km"].median()),
            "median_retreat_fraction": float(group["target_retreat_fraction"].median()),
        })
    cohort_summary = pd.DataFrame(summary_rows)
    cohort_summary.to_csv(OUT / "cohort_balance_summary.csv", index=False)

    protocol = {
        "schema": "lia-191-population-extension-cohorts-v1",
        "status": "read-only-batch-design-not-authorized-for-launch",
        "supersedes_decision_use_of": "lia_191_retained32_success_factor_extension_v1/top20_new_mechanism_shortlist.csv",
        "accepted_now": 58,
        "additional_accepted_needed_for_100": 42,
        "eligible_population_pool": 93,
        "cohorts": {
            "discovery_global_mechanism": 16,
            "heldout_independent_confirmation": 16,
            "expansion_after_confirmation": 61,
        },
        "selection": {
            "type": "deterministic proportional stratification",
            "strata": ["RGI region", "target retreat class"],
            "balance_diagnostics": ["LIA length class", "mechanism class", "existing response group"],
            "individual_priority_score_used": False,
            "single_representative_used": False,
        },
        "anti_overfitting_contract": [
            "No glacier-specific search range, forcing shape, solver, gate, or stopping rule is allowed.",
            "Only already registered glacier-specific upstream and Stage-2 inputs may differ across systems.",
            "Discovery decisions use cohort aggregate evidence and failure classes, never rescue of one named glacier.",
            "After the global mechanism is frozen, the held-out 16-system confirmation cohort is run with zero retuning.",
            "A technical failure may receive only the execution-contract exact-mechanism retry; scientific failures are retained in the denominator.",
            "The 61-system expansion cohort remains sealed until an explicit controller transaction promotes the class.",
        ],
        "practical_population_gate": {
            "confirmation_minimum_joint_passes": 8,
            "confirmation_cohort_size": 16,
            "minimum_confirmation_yield_fraction": 0.50,
            "reason": "At least 42 additional accepted systems are needed; a held-out yield materially below 50% does not support scaling 58 to approximately 100 from the 93-system pool.",
            "joint_pass_definition": "formal endpoint and trajectory gates plus required independent reproduction; the non-monotonic shape screen is reported separately and cannot replace formal acceptance",
            "authority": "planning diagnostic only; not a controller or scientific gate until separately approved",
        },
        "model_run_performed": False,
        "launch_authorized": False,
        "old_output_modified": False,
        "controller_state_modified": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )

    report = [
        "# Population-Level Extension Design",
        "",
        "This design replaces single-glacier ranking with fixed-rule cohort testing.",
        "",
        "- Discovery cohort: 16 systems, global mechanism decisions only.",
        "- Held-out confirmation cohort: 16 systems, zero retuning.",
        "- Expansion cohort: 61 systems, sealed until class promotion.",
        "- No individual priority score or single representative controls the route.",
        "- The current 58 accepted systems need 42 additions to reach 100.",
        "",
        "No model was run and no launch is authorized by this artifact.",
    ]
    (OUT / "FINAL_RESULT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
