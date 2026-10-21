"""Run and summarize the 1901-1920 proxy6 EnKF increment attribution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import scipy.io

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MATLAB_EXE = Path(r"E:\Program Files\MATLAB\R2024b\bin\matlab.exe")


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def archive_root() -> Path:
    archive_folder = "".join(chr(value) for value in (25991, 20214, 25972, 29702))
    paper_folder = "1 " + "".join(chr(value) for value in (35770, 25991))
    return Path(r"D:\Users\JUNZ\Desktop") / archive_folder / paper_folder / "work1" / "compared_result_all"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matlab_text(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1, order="F")
    return "".join(chr(int(value)) for value in values if int(value) != 0)


def scalar(dataset: h5py.Dataset) -> float:
    return float(np.asarray(dataset).reshape(-1, order="F")[0])


def proxy_metadata(path: Path) -> pd.DataFrame:
    rows = []
    with h5py.File(path, "r") as handle:
        references = np.asarray(handle["proxy_data/full_proxy"]).reshape(-1, order="F")
        for index, reference in enumerate(references, start=1):
            proxy = handle[reference]
            location = np.asarray(proxy["location"], dtype=float).reshape(-1, order="F")
            coefficients = np.asarray(proxy["b"], dtype=float).reshape(-1, order="F")
            ages = np.asarray(proxy["age_yng"], dtype=float).reshape(-1, order="F")
            data = np.asarray(proxy["data"], dtype=float).reshape(-1, order="F")
            valid = np.isfinite(ages) & np.isfinite(data)
            rows.append(
                {
                    "proxy_index": index,
                    "latitude": float(location[0]),
                    "longitude": float(location[1]),
                    "proxy_type": matlab_text(proxy["ptype"]),
                    "psm_method": matlab_text(proxy["psm_mthd"]),
                    "coefficient_intercept": float(coefficients[0]),
                    "coefficient_temperature": float(coefficients[1]),
                    "coefficient_precipitation": float(coefficients[2]),
                    "psm_correlation": scalar(proxy["pcr"]),
                    "temperature_correlation": scalar(proxy["pcr_t"]),
                    "precipitation_correlation": scalar(proxy["pcr_p"]),
                    "psm_pvalue": scalar(proxy["pval"]),
                    "first_observed_year": int(np.nanmin(ages[valid])),
                    "last_observed_year": int(np.nanmax(ages[valid])),
                    "observed_year_count": int(valid.sum()),
                }
            )
    metadata = pd.DataFrame(rows)
    if len(metadata) != 202:
        raise RuntimeError(f"unexpected-proxy-count:{len(metadata)}")
    return metadata


def vector(handle: h5py.File, name: str, dtype=float) -> np.ndarray:
    return np.asarray(handle[name], dtype=dtype).reshape(-1)


def orient_proxy_year(values: np.ndarray, proxy_count: int, year_count: int) -> np.ndarray:
    if values.shape == (year_count, proxy_count):
        return values
    if values.shape == (proxy_count, year_count):
        return values.T
    raise ValueError(f"unexpected-proxy-year-shape:{values.shape}")


def main() -> int:
    root = project_root()
    driver = Path(__file__).resolve().parent
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v1"
    )
    if output.exists():
        raise FileExistsError(f"independent-result-directory-already-exists:{output}")
    output.mkdir(parents=True)

    paths = {
        "proxy6": archive_root() / "python_data" / "proxy_data_6.mat",
        "replay_input": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_precipitation_enkf_exp1_replay_v1/replay_input.mat"
        ),
        "prior": root / (
            "validation_data/results/paleo_surface_validation_v0/"
            "climate_reconstruction_precipitation_prior_recovery_preflight_v1/"
            "precipitation_prior_336x848_mm_day.mat"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    hashes_before = {name: sha256(path) for name, path in paths.items()}
    metadata = proxy_metadata(paths["proxy6"])
    metadata.to_csv(output / "proxy6_metadata.csv", index=False)

    prefdir = output / "matlab_prefdir"
    prefdir.mkdir()
    environment = os.environ.copy()
    environment["MATLAB_PREFDIR"] = str(prefdir)
    launch = driver / "launch.m"
    completed = subprocess.run(
        [str(MATLAB_EXE), "-batch", f"run('{launch.as_posix()}')"],
        cwd=driver,
        env=environment,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    (output / "matlab_r2024b_stdout.log").write_text(completed.stdout, encoding="utf-8", errors="replace")
    (output / "matlab_r2024b_stderr.log").write_text(completed.stderr, encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        raise RuntimeError(f"matlab-attribution-failed:{completed.returncode}")

    with h5py.File(output / "attribution_output.mat", "r") as handle:
        years = vector(handle, "run_years", int)
        prior = float(vector(handle, "prior_hma_mean")[0])
        posterior = vector(handle, "posterior_hma_mean")
        total_increment = vector(handle, "total_increment")
        closure = vector(handle, "closure_error")
        active = vector(handle, "active_proxy_count", int)
        state_weights = vector(handle, "state_weights")
        contribution = orient_proxy_year(np.asarray(handle["contribution"], dtype=float), 202, 20)
        innovation = orient_proxy_year(np.asarray(handle["innovation"], dtype=float), 202, 20)
        gain_weight = orient_proxy_year(np.asarray(handle["gain_weight"], dtype=float), 202, 20)

    replay = scipy.io.loadmat(paths["replay_input"], variable_names=["target_mean"])
    target_mean = np.asarray(replay["target_mean"], dtype=float)
    target_indices = years - 1000
    archive_hma_mean = state_weights @ target_mean[:, target_indices]
    archive_closure = posterior - archive_hma_mean
    max_archive_closure = float(np.max(np.abs(archive_closure)))
    max_contribution_closure = float(np.max(np.abs(closure)))
    if max_contribution_closure > 1e-12:
        raise RuntimeError("proxy-contribution-closure-failed")
    if max_archive_closure > 0.01:
        raise RuntimeError("archive-posterior-closure-failed")

    rows = []
    for year_index, year in enumerate(years):
        for proxy_index in range(202):
            if not np.isfinite(innovation[year_index, proxy_index]):
                continue
            rows.append(
                {
                    "year": int(year),
                    "proxy_index": proxy_index + 1,
                    "innovation": float(innovation[year_index, proxy_index]),
                    "hma_gain_weight": float(gain_weight[year_index, proxy_index]),
                    "hma_increment_contribution_mm_day": float(contribution[year_index, proxy_index]),
                }
            )
    annual_proxy = pd.DataFrame(rows).merge(metadata, on="proxy_index", how="left", validate="many_to_one")
    annual_proxy.to_csv(output / "annual_proxy_contributions_1901_1920.csv", index=False)

    proxy_summary = (
        annual_proxy.groupby("proxy_index", as_index=False)
        .agg(
            active_year_count=("year", "count"),
            net_contribution_mm_day=("hma_increment_contribution_mm_day", "sum"),
            mean_contribution_mm_day=("hma_increment_contribution_mm_day", "mean"),
            mean_abs_contribution_mm_day=("hma_increment_contribution_mm_day", lambda value: float(np.mean(np.abs(value)))),
            maximum_abs_contribution_mm_day=("hma_increment_contribution_mm_day", lambda value: float(np.max(np.abs(value)))),
            mean_innovation=("innovation", "mean"),
            mean_abs_innovation=("innovation", lambda value: float(np.mean(np.abs(value)))),
            mean_hma_gain_weight=("hma_gain_weight", "mean"),
        )
        .merge(metadata, on="proxy_index", how="left", validate="one_to_one")
    )
    proxy_summary["absolute_net_contribution_mm_day"] = proxy_summary["net_contribution_mm_day"].abs()
    proxy_summary = proxy_summary.sort_values("absolute_net_contribution_mm_day", ascending=False)
    proxy_summary.to_csv(output / "proxy_contribution_summary_1901_1920.csv", index=False)

    yearly = pd.DataFrame(
        {
            "year": years,
            "prior_hma_mean_mm_day": prior,
            "posterior_hma_mean_mm_day": posterior,
            "archive_hma_mean_mm_day": archive_hma_mean,
            "total_increment_mm_day": total_increment,
            "active_proxy_count": active,
            "contribution_closure_error_mm_day": closure,
            "archive_closure_error_mm_day": archive_closure,
        }
    )
    yearly.to_csv(output / "yearly_hma_increment_1901_1920.csv", index=False)

    ranked = proxy_summary.head(15).sort_values("net_contribution_mm_day")
    figure, axes = plt.subplots(2, 1, figsize=(13.2, 9.0), constrained_layout=True)
    axes[0].plot(years, np.full_like(posterior, prior), color="#666666", lw=1.5, label="prior mean")
    axes[0].plot(years, posterior, color="#d95f02", lw=2.2, label="posterior mean")
    axes[0].bar(years, total_increment, color="#4c78a8", alpha=0.4, label="analysis increment")
    axes[0].set_ylabel("HMA precipitation (mm/day)")
    axes[0].set_title("Proxy6 EnKF experiment-1: prior and posterior HMA mean")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.2)
    colors = np.where(ranked["net_contribution_mm_day"] < 0, "#2166ac", "#b2182b")
    axes[1].barh([f"#{value}" for value in ranked["proxy_index"]], ranked["net_contribution_mm_day"], color=colors)
    axes[1].axvline(0.0, color="#333333", lw=0.8)
    axes[1].set_xlabel("Net 1901-1920 HMA contribution (mm/day summed over years)")
    axes[1].set_title("Largest absolute proxy contributions")
    axes[1].grid(axis="x", alpha=0.2)
    figure.savefig(output / "figure_increment_attribution.png", dpi=220)
    plt.close(figure)

    negative_total = float(proxy_summary.loc[proxy_summary["net_contribution_mm_day"] < 0, "net_contribution_mm_day"].sum())
    positive_total = float(proxy_summary.loc[proxy_summary["net_contribution_mm_day"] > 0, "net_contribution_mm_day"].sum())
    absolute_total = float(proxy_summary["absolute_net_contribution_mm_day"].sum())
    top5_fraction = float(proxy_summary.head(5)["absolute_net_contribution_mm_day"].sum() / absolute_total)
    posterior_trend = float(100.0 * np.polyfit(years - years.mean(), posterior, 1)[0])
    hashes_after = {name: sha256(path) for name, path in paths.items()}
    summary = {
        "experiment_id": "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v1",
        "date": "2026-08-13",
        "case": "proxy6 / precipitation / EnKF / experiment 1",
        "matlab": "R2024b",
        "years": [1901, 1920],
        "locked_final_period_accessed": False,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "prior_hma_mean_mm_day": prior,
        "posterior_hma_trend_mm_day_per_century": posterior_trend,
        "maximum_proxy_contribution_closure_error_mm_day": max_contribution_closure,
        "maximum_archive_posterior_closure_error_mm_day": max_archive_closure,
        "active_proxy_range": [int(active.min()), int(active.max())],
        "negative_contribution_total_mm_day_yearsum": negative_total,
        "positive_contribution_total_mm_day_yearsum": positive_total,
        "top5_absolute_net_contribution_fraction": top5_fraction,
        "top10_proxies": proxy_summary.head(10).to_dict(orient="records"),
        "interpretation_boundary": (
            "This attributes one reproducible proxy6 EnKF experiment. It does not authorize proxy removal "
            "or automatically generalize to proxy9, AOEnKF, or HGAOEnKF."
        ),
    }
    (output / "source_manifest.json").write_text(json.dumps({"before": hashes_before, "after": hashes_after}, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "FINAL_RESULT.md").write_text(
        "\n".join(
            [
                "# Precipitation EnKF Increment Attribution V1",
                "",
                f"- Proxy-contribution closure: `{max_contribution_closure:.3g} mm/day`",
                f"- Archive-posterior closure: `{max_archive_closure:.6g} mm/day`",
                f"- Posterior HMA trend, 1901-1920: `{posterior_trend:.6g} mm/day/century`",
                f"- Top-five fraction of absolute net contributions: `{top5_fraction:.3f}`",
                f"- Source hashes unchanged: `{hashes_before == hashes_after}`",
                "",
                summary["interpretation_boundary"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
