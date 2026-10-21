"""Route saved states with distinct melt-f values for split-forward testing."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import build_lia_191_multistate_split_forward_router_v1 as common
from .build_lia_191_iterative_loop import project_root


ROOT = project_root()
PALEO = ROOT / "validation_data/results/paleo_surface_validation_v0"
OUTPUT = (
    PALEO
    / "lia_191_controller_v1/diagnostics/"
    "multistate_melt_split_forward_router_v1"
)


def _all_split_tested_sources() -> set[str]:
    sources: set[str] = set()
    for path in PALEO.glob("lia_191_*split_forward*/**/summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = payload.get("source_dir")
        if source:
            sources.add(str(Path(source).resolve()).lower())
    return sources


def main() -> None:
    chains = pd.read_csv(common.INVENTORY)
    frozen_ids = set(pd.read_csv(common.FROZEN)["lia_unique_id"].astype(int))
    if len(frozen_ids) != 21:
        raise RuntimeError(f"Expected 21 frozen systems, found {len(frozen_ids)}")
    chains["lia_unique_id"] = chains["lia_unique_id"].astype(int)
    subset = chains.loc[
        common._bool(chains["stable_chain"])
        & common._bool(chains["audit_length_observation_valid"])
        & ~common._bool(chains["technical_failure_bool"])
        & ~chains["lia_unique_id"].isin(frozen_ids)
    ].copy()
    tested_sources = _all_split_tested_sources()
    rows = []
    for _, row in subset.iterrows():
        parsed = common._read_state(row, tested_sources)
        if parsed is None:
            continue
        parsed["already_any_split_forward_tested"] = (
            str(Path(parsed["output_dir"]).resolve()).lower() in tested_sources
        )
        rows.append(parsed)
    states = pd.DataFrame(rows)
    signature = [
        "lia_unique_id",
        "delta_t_c",
        "melt_f",
        "prcp_fac",
        "glen_a_pa_3_s_1",
        "temp_melt_c",
        "geometry_variant",
    ]
    states = (
        states.sort_values(["objective_sep", "spinup_stability_fraction", "task_id"])
        .drop_duplicates(signature, keep="first")
        .reset_index(drop=True)
    )

    source_rows = []
    state_rows = []
    for system_id, group in states.groupby("lia_unique_id", sort=True):
        work = group.copy()
        work["melt_key"] = work["melt_f"].map(lambda value: f"{value:.10f}")
        melt_states = (
            work.sort_values(["objective_sep", "spinup_stability_fraction"])
            .groupby("melt_key", as_index=False, sort=True)
            .first()
            .sort_values("objective_sep")
        )
        if len(melt_states) < 2:
            continue
        untested = melt_states.loc[
            ~melt_states["already_any_split_forward_tested"].astype(bool)
        ].copy()
        if untested.empty:
            continue
        source = untested.sort_values("objective_sep").iloc[0]
        volumes = pd.to_numeric(melt_states["spinup_volume_m3"], errors="coerce")
        source_rows.append(
            {
                "lia_unique_id": int(system_id),
                "rgi_id": source["rgi_id"],
                "distinct_melt_state_count": int(len(melt_states)),
                "source_task_id": source["task_id"],
                "source_output_dir": source["output_dir"],
                "source_delta_t": source["delta_t_c"],
                "source_melt_f": source["melt_f"],
                "source_melt_multiplier": source["melt_f_stage2_multiplier"],
                "source_glen_multiplier": source["glen_a_stage2_multiplier"],
                "source_signed_lia_sep": source["signed_lia_sep"],
                "source_signed_rgi_sep": source["signed_rgi_sep"],
                "source_objective_sep": source["objective_sep"],
                "source_endpoint_class": source["endpoint_class"],
                "spinup_volume_spread_fraction": (
                    (volumes.max() - volumes.min()) / volumes.median()
                ),
            }
        )
        for rank, row in enumerate(melt_states.itertuples(index=False), start=1):
            state_rows.append(
                {
                    "lia_unique_id": int(system_id),
                    "state_rank": rank,
                    "task_id": row.task_id,
                    "output_dir": row.output_dir,
                    "delta_t_c": row.delta_t_c,
                    "melt_f": row.melt_f,
                    "melt_f_stage2_multiplier": row.melt_f_stage2_multiplier,
                    "glen_a_stage2_multiplier": row.glen_a_stage2_multiplier,
                    "spinup_volume_m3": row.spinup_volume_m3,
                    "signed_lia_sep": row.signed_lia_sep,
                    "signed_rgi_sep": row.signed_rgi_sep,
                    "objective_sep": row.objective_sep,
                    "endpoint_class": row.endpoint_class,
                    "already_any_split_forward_tested": row.already_any_split_forward_tested,
                }
            )

    router = pd.DataFrame(source_rows).sort_values(
        ["source_objective_sep", "spinup_volume_spread_fraction"],
        ascending=[True, False],
    ).reset_index(drop=True)
    if router.empty:
        raise RuntimeError("No untested distinct-melt saved states found")
    router["router_rank"] = range(1, len(router) + 1)
    selected: list[int] = []
    classes: set[str] = set()
    for idx, row in router.iterrows():
        endpoint_class = str(row["source_endpoint_class"])
        if endpoint_class not in classes:
            selected.append(idx)
            classes.add(endpoint_class)
        if len(selected) == 3:
            break
    if len(selected) < 3:
        selected.extend(idx for idx in router.index if idx not in selected)
        selected = selected[:3]
    router["selected_representative"] = router.index.isin(selected)
    queue = pd.DataFrame(state_rows)
    selected_ids = set(
        router.loc[router["selected_representative"], "lia_unique_id"].astype(int)
    )
    queue["selected_system"] = queue["lia_unique_id"].isin(selected_ids)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    states.to_csv(OUTPUT / "reusable_saved_states.csv", index=False)
    router.to_csv(OUTPUT / "system_router.csv", index=False)
    queue.to_csv(OUTPUT / "melt_state_queue.csv", index=False)
    summary = {
        "schema_version": 1,
        "question": (
            "Can a different already-converged melt-f state change transient "
            "amplitude enough for independent forward delta-T to close both endpoints?"
        ),
        "frozen_system_count_excluded": len(frozen_ids),
        "eligible_distinct_melt_system_count": int(len(router)),
        "selected_representatives": router.loc[
            router["selected_representative"], "lia_unique_id"
        ].astype(int).tolist(),
        "selected_classes": router.loc[
            router["selected_representative"], "source_endpoint_class"
        ].astype(str).tolist(),
        "launched_oggm": False,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(
        router.loc[
            router["selected_representative"],
            [
                "lia_unique_id",
                "source_melt_multiplier",
                "source_delta_t",
                "source_endpoint_class",
                "source_objective_sep",
                "spinup_volume_spread_fraction",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
