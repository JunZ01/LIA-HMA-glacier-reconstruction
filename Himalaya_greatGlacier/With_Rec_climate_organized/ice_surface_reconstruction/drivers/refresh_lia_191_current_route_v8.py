"""Accepted-53 route after the formal 5443 historical pass.

Read-only route collection plus queue-file generation.  No queue command is
executed by this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import refresh_lia_191_current_route_v6 as v6
from .refresh_lia_191_current_route_v7 import _active_process_snapshot_v7
from .build_lia_191_current_route_v4 import ROOT, VALIDATION, _set


OUT = VALIDATION / "lia_191_current_route_v8"
PYTHON = Path(r"D:\Anaconda\envs\oggm_env2\python.exe")
CONFIG_5443 = (
    ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config"
    / "synthetic_lia_system_standard_inversion_v1/historical_junction_coupled_isotonic_5443_endp0p5_v1.json"
)
ORIGINAL_5443 = ROOT / "w739hist_isotonic_5443_endp0p5_v1"
REPRO_5443 = ROOT / "w739hist_isotonic_5443_endp0p5_v1_independent_reproduction_v1"
COMPARE_5443 = (
    VALIDATION
    / "lia_191_current_route_v8/reproduction_comparison_5443_v1.json"
)


_base_overlay = v6._overlay_terminal_evidence


def _overlay_terminal_evidence_v8(route: pd.DataFrame) -> None:
    _base_overlay(route)
    _set(
        route,
        5443,
        accepted=False,
        current_route="historical-reproduction-pending",
        current_gate="formal-historical-pass-awaiting-independent-reproduction",
        controller_state="queued",
        mechanism_class="junction-coupled-isotonic-historical-pass",
        next_legal_action="run one exact independent reproduction; register only if full trajectory and final state reproduce exactly",
        latest_evidence_stage="historical",
        latest_evidence_status="historical-pass",
        latest_flags="e_lia_sep=0.0851159;e_rgi_sep=0.0447487",
        latest_evidence="w739hist_isotonic_5443_endp0p5_v1/historical_summary.json",
        update_source="junction-coupled-isotonic-5443-historical-v1",
        active_process_count=0,
        active_process_ids="",
    )


def _queue_v8(route: pd.DataFrame, active_ids: set[int]) -> pd.DataFrame:
    rows: list[dict] = []
    sid = 5443
    row = route.loc[route["lia_unique_id"].eq(sid)].iloc[0]
    collision = REPRO_5443.exists()
    if not bool(row.accepted) and sid not in active_ids:
        run_command = (
            f"& '{PYTHON}' -m "
            "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
            f"run_conservative_historical_ramp_overlay_v1 --config '{CONFIG_5443}' "
            f"--end-delta-t 0.5 --output-root '{REPRO_5443}'"
        )
        compare_command = (
            f"& '{PYTHON}' -m "
            "Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers."
            f"compare_historical_reproduction_v1 --original '{ORIGINAL_5443}' "
            f"--reproduction '{REPRO_5443}' --output-json '{COMPARE_5443}'"
        )
        rows.append(
            {
                "priority": 1,
                "system_id": sid,
                "queue_state": "run-ready" if not collision else "output-collision-review",
                "action_type": "exact-independent-historical-reproduction",
                "next_legal_action": row.next_legal_action,
                "config_path": str(CONFIG_5443),
                "module": "run_conservative_historical_ramp_overlay_v1",
                "output_root": str(REPRO_5443),
                "dependencies": "formal 5443 historical pass; isotonic static/zero-MB/steady pass; original config exists",
                "launchable_now": bool(CONFIG_5443.exists() and ORIGINAL_5443.exists() and not collision),
                "blocking_reason": "" if not collision else "versioned reproduction output already exists",
                "command": run_command,
                "post_run_command": compare_command,
            }
        )

    # The finite 3144 bracket is scientifically legal but has no preregistered
    # midpoint configuration, so it cannot be launched from this snapshot.
    sid = 3144
    row = route.loc[route["lia_unique_id"].eq(sid)].iloc[0]
    if not bool(row.accepted) and sid not in active_ids:
        rows.append(
            {
                "priority": 2,
                "system_id": sid,
                "queue_state": "config-audit-required",
                "action_type": "generate-one-preregistered-midpoint-config",
                "next_legal_action": row.next_legal_action,
                "config_path": "",
                "module": "",
                "output_root": "",
                "dependencies": "zero-MB30 pass; +0.375/-0.125 steady summaries; unchanged Stage2/climate/geometry",
                "launchable_now": False,
                "blocking_reason": "midpoint config/output are not yet preregistered",
                "command": "",
                "post_run_command": "",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    v6.OUT = OUT
    v6._active_process_snapshot = _active_process_snapshot_v7
    v6._overlay_terminal_evidence = _overlay_terminal_evidence_v8
    v6._queue = _queue_v8
    result = v6.main()

    queue = pd.read_csv(OUT / "next_legal_queue.csv")
    launchable = queue.loc[queue["launchable_now"].astype(str).str.lower().eq("true")]
    lines = [
        "# accepted53 current-route v8 legal queue",
        "# Generated only; this file was not executed.",
    ]
    for row in launchable.itertuples(index=False):
        lines.append(str(row.command))
        if pd.notna(row.post_run_command) and str(row.post_run_command):
            lines.append(str(row.post_run_command))
    (OUT / "launch_queue.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["supersedes"] = "lia_191_current_route_v7"
    summary["new_formal_pass_awaiting_reproduction"] = [5443]
    summary["driver"] = str(Path(__file__).resolve())
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    raise SystemExit(main())
