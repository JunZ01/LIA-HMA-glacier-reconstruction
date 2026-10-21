"""Test two new HMA temperature chronologies in the frozen Round 116 OOF model."""

from __future__ import annotations

import importlib.util
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
LOCKED_START = 1921
LOCKED_END = 1939


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module-spec-failed:{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return [
        "".join(node.text or "" for node in item.iter(f"{{{NS_MAIN}}}t"))
        for item in root.findall(f"{{{NS_MAIN}}}si")
    ]


def sheet_targets(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall(f"{{{NS_PKG_REL}}}Relationship")
    }
    result: dict[str, str] = {}
    for sheet in workbook.findall(f".//{{{NS_MAIN}}}sheet"):
        target = targets[sheet.attrib[f"{{{NS_REL}}}id"]].replace("\\", "/")
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = f"xl/{target}"
        result[sheet.attrib["name"]] = target
    return result


def cell_text(cell: ET.Element, strings: list[str]) -> str:
    cell_type = cell.attrib.get("t", "n")
    value_node = cell.find(f"{{{NS_MAIN}}}v")
    if cell_type == "s" and value_node is not None:
        return strings[int(value_node.text or "0")]
    inline_node = cell.find(f"{{{NS_MAIN}}}is")
    if cell_type == "inlineStr" and inline_node is not None:
        return "".join(node.text or "" for node in inline_node.iter(f"{{{NS_MAIN}}}t"))
    return "" if value_node is None else value_node.text or ""


def chronology_1940_2000(path: Path, sheet_name: str, years: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode chronology values only for the explicitly allowed modern years."""
    wanted = set(int(year) for year in years)
    values: dict[int, float] = {}
    selected_rows: dict[int, int] = {}
    locked_value_cells_decoded = 0
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        targets = sheet_targets(archive)
        if sheet_name not in targets:
            raise RuntimeError(f"missing-sheet:{path.name}:{sheet_name}")
        with archive.open(targets[sheet_name]) as stream:
            for _, row in ET.iterparse(stream, events=("end",)):
                if row.tag != f"{{{NS_MAIN}}}row":
                    continue
                cells = {
                    cell.attrib.get("r", "")[:1].upper(): cell
                    for cell in row.findall(f"{{{NS_MAIN}}}c")
                }
                year_cell = cells.get("A")
                if year_cell is None:
                    row.clear()
                    continue
                raw_year = cell_text(year_cell, strings).strip()
                try:
                    year = int(round(float(raw_year)))
                except ValueError:
                    row.clear()
                    continue
                if LOCKED_START <= year <= LOCKED_END:
                    # Deliberately do not call cell_text on column B.
                    row.clear()
                    continue
                if year in wanted:
                    value_cell = cells.get("B")
                    if value_cell is None:
                        raise RuntimeError(f"missing-value-cell:{path.name}:{sheet_name}:{year}")
                    raw_value = cell_text(value_cell, strings).strip()
                    values[year] = float(raw_value) if raw_value else np.nan
                    selected_rows[year] = int(row.attrib.get("r", "0"))
                row.clear()
    missing = sorted(wanted.difference(values))
    if missing:
        raise RuntimeError(f"missing-modern-years:{path.name}:{missing}")
    result = np.asarray([values[int(year)] for year in years], dtype=float)
    if not np.all(np.isfinite(result)) or np.nanstd(result, ddof=1) <= 1e-10:
        raise RuntimeError(f"invalid-modern-chronology:{path.name}")
    return result, {
        "source": str(path),
        "sheet": sheet_name,
        "value_column": "B",
        "first_selected_year": int(years[0]),
        "last_selected_year": int(years[-1]),
        "selected_year_count": int(len(years)),
        "first_selected_workbook_row": int(selected_rows[int(years[0])]),
        "last_selected_workbook_row": int(selected_rows[int(years[-1])]),
        "locked_1921_1939_value_cells_decoded": int(locked_value_cells_decoded),
    }


def run_branch(
    branch: str,
    proxy: np.ndarray,
    base,
    round116,
    common,
    residual_target: np.ndarray,
    legacy: np.ndarray,
    observations: dict[str, np.ndarray],
    weights: dict[str, np.ndarray],
    support: np.ndarray,
    area_weights: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    residual_oof = np.full_like(legacy, np.nan)
    inner_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for outer_fold, (start, end) in enumerate(base.BLOCKS, start=1):
        test = (base.YEARS >= start) & (base.YEARS <= end)
        train = ~test
        rank, ridge, rows = round116.select_residual_hyperparameters(
            base,
            common,
            proxy,
            residual_target,
            legacy,
            train,
            observations,
            weights,
            support,
            area_weights,
        )
        for row in rows:
            row["branch"] = branch
            row["outer_fold"] = int(outer_fold)
            row["outer_period"] = f"{start}-{end}"
        inner_rows.extend(rows)
        prediction, diagnostics = base.fit_low_rank(
            proxy,
            residual_target,
            train,
            test,
            area_weights,
            rank,
            ridge,
        )
        residual_oof[np.ix_(support, test)] = prediction
        selection_rows.append(
            {
                "branch": branch,
                "outer_fold": int(outer_fold),
                "outer_period": f"{start}-{end}",
                "selected_rank": int(rank),
                "selected_ridge": float(ridge),
                **diagnostics,
            }
        )
    if not np.all(np.isfinite(residual_oof[support])):
        raise RuntimeError(f"residual-oof-stitch-failed:{branch}")
    candidate = legacy.copy()
    candidate[support] += residual_oof[support]
    return base.recenter(candidate), selection_rows, inner_rows


def branch_metrics(
    branch: str,
    pairs: pd.DataFrame,
    blocks: pd.DataFrame,
    spatial: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, bool]]:
    cp = pairs[pairs.branch.eq(branch)]
    cb = blocks[blocks.branch.eq(branch)]
    cs = spatial[spatial.branch.eq(branch)]
    relative = cp.improvement_vs_equal_six.to_numpy(float)
    paths_relative = cb.improvement_vs_equal_six.to_numpy(float)
    absolute = cp.skill_vs_zero.to_numpy(float)
    metrics = {
        "median_rmse_improvement_over_equal_six": float(np.nanmedian(relative)),
        "worst_pair_rmse_improvement_over_equal_six": float(np.nanmin(relative)),
        "fraction_block_region_product_paths_improving": float(np.mean(paths_relative > 0)),
        "median_skill_over_zero": float(np.nanmedian(absolute)),
        "worst_skill_over_zero": float(np.nanmin(absolute)),
        "fraction_positive_regional_skill": float(np.mean(absolute > 0)),
        "median_annual_regional_correlation": float(np.nanmedian(cp.annual_correlation)),
        "median_15year_regional_correlation": float(np.nanmedian(cp.correlation_15year)),
        "median_spatial_residual_skill": float(np.nanmedian(cs.spatial_residual_skill)),
        "worst_spatial_residual_skill": float(np.nanmin(cs.spatial_residual_skill)),
        "fraction_positive_spatial_skill": float(np.mean(cs.spatial_residual_skill > 0)),
        "median_spatial_pattern_correlation": float(
            np.nanmedian(cs.median_annual_spatial_pattern_correlation)
        ),
    }
    gates = {
        "M_REL_1_median_rmse_improvement_over_equal_six_at_least_0_05": bool(np.nanmedian(relative) >= 0.05),
        "M_REL_2_worst_pair_regression_at_most_0_05": bool(np.nanmin(relative) >= -0.05 - 1e-12),
        "M_REL_3_fraction_block_region_product_paths_improving_at_least_0_75": bool(np.mean(paths_relative > 0) >= 0.75),
        "M_ABS_1_median_rmse_skill_over_zero_at_least_0_05": bool(np.nanmedian(absolute) >= 0.05),
        "M_ABS_2_worst_rmse_skill_over_zero_at_least_minus_0_05": bool(np.nanmin(absolute) >= -0.05 - 1e-12),
        "M_ABS_3_fraction_positive_regional_skill_at_least_0_75": bool(np.mean(absolute > 0) >= 0.75),
        "M_ABS_4_median_annual_regional_correlation_at_least_0_30": bool(np.nanmedian(cp.annual_correlation) >= 0.30),
        "M_SPAT_1_median_spatial_skill_positive": bool(np.nanmedian(cs.spatial_residual_skill) > 0),
        "M_SPAT_2_worst_spatial_skill_at_least_minus_0_05": bool(np.nanmin(cs.spatial_residual_skill) >= -0.05 - 1e-12),
        "M_SPAT_3_fraction_positive_spatial_skill_at_least_0_75": bool(np.mean(cs.spatial_residual_skill > 0) >= 0.75),
        "M_SPAT_4_median_spatial_pattern_correlation_at_least_0_20": bool(
            np.nanmedian(cs.median_annual_spatial_pattern_correlation) >= 0.20
        ),
    }
    return metrics, gates


def main() -> int:
    driver = Path(__file__).resolve().parent
    round116 = load_module(
        "round120_round116",
        driver / "run_climate_reconstruction_loop_v2_round116_temperature_low_rank_residual_oof_v1.py",
    )
    base = load_module(
        "round120_base",
        driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    )
    common = base.load_common(driver)
    root = base.project_root()
    loop = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    identity = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_external_proxy_identity_closure_20260818_v1"
    raw = root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_external_proxy_discovery_20260818_v1/raw_unopened"
    round115_dir = loop / "round_115_temperature_low_rank_oof_v2"
    round116_dir = loop / "round_116_temperature_low_rank_residual_oof_v1"
    output = loop / "round_120_external_temperature_incremental_oof_v1"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    paths = {
        "facts": root / "LIA_PROJECT_FACTS.md",
        "proxy9": root / "validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v1/round_03_nested_ensemble_selection/proxy9_allowed_1940_2000_official_baseline_v2.mat",
        "ledger": loop / "round_110_unresolved_proxy_metadata_recovery_v4/proxy_evidence_master_ledger_metadata_recovered.csv",
        "round115_fields": round115_dir / "temperature_low_rank_oof_fields.npz",
        "round116_fields": round116_dir / "temperature_low_rank_residual_oof_fields.npz",
        "round116_summary": round116_dir / "summary.json",
        "round113_gates": loop / "round_113_frozen_multiscale_validation_gates_v1/acceptance_gates.csv",
        "identity_result": identity / "FINAL_RESULT.md",
        "hengduan": raw / "hengduan_upload_information.xlsx",
        "haizi": raw / "haizi_temperature_supplement.xlsx",
        "contract": driver / "round120_external_temperature_incremental_oof_v1_contract.md",
        "driver": Path(__file__).resolve(),
        "round116_driver": driver / "run_climate_reconstruction_loop_v2_round116_temperature_low_rank_residual_oof_v1.py",
        "base_driver": driver / "run_climate_reconstruction_loop_v2_round115_temperature_low_rank_oof_v1.py",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prior = json.loads(paths["round116_summary"].read_text(encoding="utf-8"))
    if prior["passes_all_required_modern_gates"] or not prior["source_hashes_unchanged"]:
        raise RuntimeError("round116-boundary-contract-failed")
    if prior["locked_1921_1939_accessed"] or prior["external_reserve_opened"]:
        raise RuntimeError("round116-data-boundary-failed")
    shutil.copy2(paths["contract"], output / "ROUND_CONTRACT.md")
    before = {name: base.file_state(path) for name, path in paths.items()}

    hengduan, hengduan_meta = chronology_1940_2000(
        paths["hengduan"], "Tree-ring width chronology", base.YEARS
    )
    haizi, haizi_meta = chronology_1940_2000(paths["haizi"], "Sheet1", base.YEARS)
    chronology = pd.DataFrame(
        {"year": base.YEARS, "hengduan_regional_trw": hengduan, "haizi_std": haizi}
    )
    chronology.to_csv(output / "external_temperature_chronologies_1940_2000.csv", index=False)
    extraction = {"hengduan": hengduan_meta, "haizi": haizi_meta}
    (output / "chronology_extraction_audit.json").write_text(
        json.dumps(extraction, indent=2), encoding="utf-8"
    )

    fields = np.load(paths["round115_fields"])
    years = np.asarray(fields["years"], dtype=int)
    if not np.array_equal(years, base.YEARS):
        raise RuntimeError("round115-year-closure-failed")
    support = np.asarray(fields["observed_support"], dtype=bool)
    hma_mask = np.asarray(fields["hma_mask"], dtype=bool)
    grid_lat = np.asarray(fields["grid_lat"], dtype=float)
    grid_lon = np.asarray(fields["grid_lon"], dtype=float)
    legacy = np.asarray(fields["equal_six_legacy"], dtype=float)
    observations = {
        name: np.asarray(fields[f"observation_{name}"], dtype=float)
        for name in base.OBSERVATION_VARIABLES
    }
    if int(support.sum()) != 99 or legacy.shape != (336, 61):
        raise RuntimeError("round115-field-support-closure-failed")
    weights = common.region_weights(grid_lat, grid_lon, hma_mask)
    weights = {name: values * support for name, values in weights.items()}
    area_weights = weights["hma"][support]
    consensus = np.mean(np.stack([values[support] for values in observations.values()]), axis=0)
    residual_target = consensus - legacy[support]

    loaded = scipy.io.loadmat(paths["proxy9"], variable_names=["years", "observations"])
    proxy_years = np.asarray(loaded["years"], dtype=int).reshape(-1)
    proxy_all = np.asarray(loaded["observations"], dtype=float)
    if not np.array_equal(proxy_years, base.YEARS) or proxy_all.shape != (61, 297):
        raise RuntimeError("proxy9-contract-failed")
    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    proxy_ledger = ledger[ledger.network.eq("proxy9")].copy()
    proxy_ledger["network_record_1based"] = proxy_ledger.network_record_1based.astype(int)
    proxy_ledger = proxy_ledger.sort_values("network_record_1based")
    selected = (
        proxy_ledger.documented_network_role_v4.eq("documented_temperature")
        & proxy_ledger.operator_metadata_ready_v4.astype(str).str.lower().eq("true")
    ).to_numpy()
    if len(proxy_ledger) != 297 or int(selected.sum()) != 226:
        raise RuntimeError("temperature-proxy-selection-closure-failed")
    proxy = proxy_all[:, selected]

    matrices = {
        "plus_hengduan": np.column_stack([proxy, hengduan]),
        "plus_haizi": np.column_stack([proxy, haizi]),
        "plus_both": np.column_stack([proxy, hengduan, haizi]),
    }
    candidates: dict[str, np.ndarray] = {}
    selections: list[dict[str, Any]] = []
    inner_scores: list[dict[str, Any]] = []
    for branch, matrix in matrices.items():
        candidate, branch_selections, branch_inner = run_branch(
            branch,
            matrix,
            base,
            round116,
            common,
            residual_target,
            legacy,
            observations,
            weights,
            support,
            area_weights,
        )
        candidates[branch] = candidate
        selections.extend(branch_selections)
        inner_scores.extend(branch_inner)

    old_fields = np.load(paths["round116_fields"])
    round116_candidate = np.asarray(old_fields["candidate"], dtype=float)
    branches = {
        **candidates,
        "round116_226_proxy_control": round116_candidate,
        "equal_six_legacy": legacy,
        "zero_anomaly": np.zeros_like(legacy),
    }
    pairs, blocks, curves, spatial = base.score_all(branches, observations, weights, common)
    evaluated = ["round116_226_proxy_control", "plus_hengduan", "plus_haizi", "plus_both"]
    metrics: dict[str, dict[str, float]] = {}
    gates: dict[str, dict[str, bool]] = {}
    for branch in evaluated:
        metrics[branch], gates[branch] = branch_metrics(branch, pairs, blocks, spatial)
    baseline = metrics["round116_226_proxy_control"]
    metric_deltas = {
        branch: {name: value - baseline[name] for name, value in current.items()}
        for branch, current in metrics.items()
        if branch != "round116_226_proxy_control"
    }
    combined_passes = bool(all(gates["plus_both"].values()))

    pd.DataFrame(selections).to_csv(output / "outer_fold_hyperparameter_selection.csv", index=False)
    pd.DataFrame(inner_scores).to_csv(output / "inner_hyperparameter_scores.csv", index=False)
    pairs.to_csv(output / "regional_product_scores.csv", index=False)
    blocks.to_csv(output / "block_region_product_scores.csv", index=False)
    spatial.to_csv(output / "spatial_product_scores.csv", index=False)
    curves.to_csv(output / "regional_curves.csv", index=False)
    np.savez_compressed(
        output / "external_temperature_incremental_oof_fields.npz",
        years=base.YEARS,
        grid_lat=grid_lat,
        grid_lon=grid_lon,
        hma_mask=hma_mask,
        observed_support=support,
        plus_hengduan=candidates["plus_hengduan"],
        plus_haizi=candidates["plus_haizi"],
        plus_both=candidates["plus_both"],
        round116_226_proxy_control=round116_candidate,
        equal_six_legacy=legacy,
    )

    hma = curves[curves.region.eq("hma")]
    colors = {
        "round116_226_proxy_control": "#777777",
        "plus_hengduan": "#0072B2",
        "plus_haizi": "#009E73",
        "plus_both": "#D55E00",
        "equal_six_legacy": "#CC79A7",
    }
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    for branch in ["equal_six_legacy", *evaluated]:
        values = hma[(hma.kind == "reconstruction") & (hma.series == branch)]
        axes[0].plot(values.year, values.value_degC, color=colors[branch], linewidth=1.7, label=branch)
    obs = hma[hma.kind.eq("observation")].groupby("year", as_index=False).value_degC.mean()
    axes[0].plot(obs.year, obs.value_degC, color="#111111", linewidth=1.8, label="five-product mean")
    axes[0].axhline(0, color="#888888", linewidth=0.7)
    axes[0].set_ylabel("HMA anomaly (degC)")
    axes[0].set_title("External chronology incremental OOF comparison")
    axes[0].legend(frameon=False, ncol=3, fontsize=8)
    metric_names = [
        "worst_pair_rmse_improvement_over_equal_six",
        "fraction_block_region_product_paths_improving",
        "median_spatial_residual_skill",
        "worst_spatial_residual_skill",
        "fraction_positive_spatial_skill",
    ]
    labels = ["worst pair", "improving paths", "median spatial", "worst spatial", "positive spatial"]
    x = np.arange(len(metric_names))
    width = 0.19
    for index, branch in enumerate(evaluated):
        axes[1].bar(
            x + (index - 1.5) * width,
            [metrics[branch][name] for name in metric_names],
            width=width,
            color=colors[branch],
            label=branch,
        )
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("fraction / skill")
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(output / "round120_external_temperature_incremental_oof.png", dpi=220)
    plt.close(fig)

    after = {name: base.file_state(path) for name, path in paths.items()}
    unchanged = bool(before == after)
    if not unchanged:
        raise RuntimeError("source-changed-during-round120")
    (output / "source_manifest.json").write_text(
        json.dumps({"before": before, "after": after, "source_hashes_unchanged": unchanged}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "experiment_id": "round_120_external_temperature_incremental_oof_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "decision": "advance_combined_external_temperature_branch" if combined_passes else "close_two_chronology_repair_of_annual_low_rank_residual_method",
        "combined_branch_passes_all_required_modern_gates": combined_passes,
        "locked_1921_1939_accessed": False,
        "locked_1921_1939_candidate_values_decoded": 0,
        "external_validation_reserve_opened": False,
        "source_hashes_unchanged": unchanged,
        "existing_temperature_proxy_count": 226,
        "branch_proxy_counts": {name: int(matrix.shape[1]) for name, matrix in matrices.items()},
        "chronology_extraction": extraction,
        "metrics": metrics,
        "metric_deltas_vs_round116": metric_deltas,
        "gates": gates,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Round 120 External Temperature Incremental OOF",
        "",
        f"- Decision: `{summary['decision']}`.",
        f"- Combined branch passes every frozen gate: `{str(combined_passes).lower()}`.",
        "- Locked 1921-1939 candidate values decoded: `0`.",
        "- External validation reserve opened: `false`.",
        f"- Source hashes unchanged: `{str(unchanged).lower()}`.",
        "",
        "## Branch results",
        "",
    ]
    for branch in evaluated:
        current = metrics[branch]
        passed = sum(gates[branch].values())
        lines.extend(
            [
                f"### {branch}",
                "",
                f"- Gates passed: `{passed}/{len(gates[branch])}`.",
                f"- Median improvement over equal-six: `{current['median_rmse_improvement_over_equal_six']:.3%}`.",
                f"- Worst pair improvement: `{current['worst_pair_rmse_improvement_over_equal_six']:.3%}`.",
                f"- Improving block-region-product paths: `{current['fraction_block_region_product_paths_improving']:.3%}`.",
                f"- Median spatial residual skill: `{current['median_spatial_residual_skill']:.3%}`.",
                f"- Worst spatial residual skill: `{current['worst_spatial_residual_skill']:.3%}`.",
                f"- Positive spatial skill fraction: `{current['fraction_positive_spatial_skill']:.3%}`.",
                "",
            ]
        )
    lines.extend(["## Combined branch gates", ""])
    lines.extend(f"- `{name}`: `{str(value).lower()}`" for name, value in gates["plus_both"].items())
    (output / "FINAL_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
