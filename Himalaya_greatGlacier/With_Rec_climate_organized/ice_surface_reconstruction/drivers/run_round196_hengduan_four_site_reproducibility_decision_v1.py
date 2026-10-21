from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output = loop / "round_196_hengduan_four_site_reproducibility_decision_v1"
    output.mkdir(parents=True, exist_ok=False)
    r195 = loop / "round_195_sigfree_toolchain_preflight_v3/summary.json"
    r193 = loop / "round_193_east_hma_dual_index_wang_shape_v1/summary.json"
    inputs = {"round195": sha256(r195), "round193": sha256(r193)}
    tool = json.loads(r195.read_text(encoding="ascii"))
    if tool["selected_executable_count"] != 1 or tool["selected_documentation_count"] != 1:
        raise RuntimeError("unexpected-round195-tool-state")
    result = {
        "experiment_id": output.name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "published_method": {
            "software_family": "RCSigFree",
            "power_transform": True,
            "detrending": "Friedman_variable_span_super_smoother",
            "alpha": 7,
            "signal_free_iteration": True,
            "mean": "biweight_robust",
            "variance_stabilization": "Osborn_et_al_1997_style",
            "eps_rbar_window_years": 51,
            "eps_threshold": 0.85,
            "minimum_core_tree_depth": "6_cores_3_trees",
        },
        "official_tool": {
            "archive_sha256": tool["archive_sha256"],
            "executable_count": tool["selected_executable_count"],
            "documentation_count": tool["selected_documentation_count"],
            "example_or_control_file_present": False,
            "version_resource_present": False,
            "static_interface_class": "interactive_Fortran_with_graphics_prompts",
            "binary_executed": False,
        },
        "exact_five_site_reproduction_demonstrated": False,
        "four_site_chronology_authorized": False,
        "ad_hoc_open_source_approximation_authorized": False,
        "locked_1921_1939_numeric_values_accessed": False,
        "jionglacuo_opened_or_hashed": False,
        "enkf_run": False,
        "oggm_run": False,
        "decision": "close_four_site_branch_pending_explicit_binary_execution_or_author_supplied_control_or_chronology",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="ascii")
    (output / "source_manifest.json").write_text(json.dumps({"inputs": inputs}, indent=2), encoding="ascii")
    (output / "FINAL_RESULT.md").write_text(
        "# Round 196 Hengduan Four-Site Reproducibility Decision\n\n"
        "- The exact published chronology method requires RCSigFree-specific processing.\n"
        "- The official suite contains one interactive executable and the method paper, but no example/control file.\n"
        "- Exact five-site reproduction has not been demonstrated.\n"
        "- A hand-written approximation is not authorized because its low-frequency differences could be mistaken for climate.\n\n"
        "Decision: close the four-site branch pending explicit binary execution or an author-supplied control file/chronology.\n",
        encoding="ascii",
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

