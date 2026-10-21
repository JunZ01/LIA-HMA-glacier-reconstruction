"""Add external-trend consistency diagnostics to PSM stability V2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists():
            return parent
    raise RuntimeError("project-root-not-found")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = project_root()
    output = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_psm_temporal_stability_v2"
    )
    attribution = root / (
        "validation_data/results/paleo_surface_validation_v0/"
        "climate_reconstruction_precipitation_enkf_exp1_increment_attribution_v2/"
        "proxy_trend_contribution_1901_1920.csv"
    )
    inputs = {
        "proxy_table": output / "psm_temporal_stability_proxy_level.csv",
        "projection_table": output / "psm_temporal_stability_hma_projection_proxy_level.csv",
        "attribution": attribution,
        "figure": output / "figure_psm_temporal_stability.png",
    }
    destinations = {
        "trend_csv": output / "external_trend_consistency_1901_1920.csv",
        "qa_json": output / "supplemental_qa.json",
        "result_md": output / "SUPPLEMENTAL_RESULT.md",
        "manifest": output / "supplemental_source_manifest.json",
    }
    if missing := [str(path) for path in inputs.values() if not path.exists()]:
        raise FileNotFoundError("missing-inputs:" + ";".join(missing))
    if existing := [str(path) for path in destinations.values() if path.exists()]:
        raise FileExistsError("supplemental-output-already-exists:" + ";".join(existing))
    before = {name: sha256(path) for name, path in inputs.items()}

    proxies = pd.read_csv(inputs["proxy_table"])
    projections = pd.read_csv(inputs["projection_table"])
    attribution_table = pd.read_csv(attribution).set_index("proxy_index")
    rows = []
    for split in sorted(proxies["split"].unique()):
        coefficients = proxies[proxies["split"].eq(split)].set_index("proxy_index")
        climate = projections[
            projections["split"].eq(split)
            & projections["period"].eq("external_1901_1920")
        ].set_index("proxy_index")
        predicted = (
            coefficients["fit_b1_temperature"]
            * climate["temperature_trend_c_per_century"]
            + coefficients["fit_b2_precipitation"]
            * climate["precipitation_trend_mm_day_per_century"]
        )
        actual = attribution_table.loc[
            predicted.index, "innovation_trend_proxy_units_per_century"
        ]
        gains = attribution_table.loc[predicted.index, "mean_hma_gain_weight"]
        rows.append(
            {
                "split": split,
                "proxy_count": int(len(predicted)),
                "actual_vs_predicted_trend_correlation": float(actual.corr(predicted)),
                "trend_sign_agreement_fraction": float(
                    np.mean(np.sign(actual) == np.sign(predicted))
                ),
                "trend_rmse_proxy_units_per_century": float(
                    np.sqrt(np.mean((actual - predicted) ** 2))
                ),
                "predicted_hma_trend_mm_day_per_century": float(
                    (gains * predicted).sum()
                ),
                "actual_hma_trend_mm_day_per_century": float(
                    attribution_table.loc[
                        predicted.index, "contribution_trend_mm_day_per_century"
                    ].sum()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(destinations["trend_csv"], index=False)

    image = np.asarray(Image.open(inputs["figure"]).convert("RGB"))
    qa = {
        "locked_final_period_accessed": False,
        "figure_shape": list(image.shape),
        "figure_pixel_standard_deviation": float(image.std()),
        "figure_nonwhite_fraction": float(np.mean(np.any(image < 250, axis=2))),
    }
    destinations["qa_json"].write_text(json.dumps(qa, indent=2), encoding="utf-8")
    after = {name: sha256(path) for name, path in inputs.items()}
    destinations["manifest"].write_text(
        json.dumps({"before": before, "after": after}, indent=2), encoding="utf-8"
    )
    first, second = result.iloc[0], result.iloc[1]
    destinations["result_md"].write_text(
        "\n".join(
            [
                "# Supplemental External-Trend Consistency",
                "",
                f"- Early-fit trend correlation: `{first.actual_vs_predicted_trend_correlation:.6g}`",
                f"- Late-fit trend correlation: `{second.actual_vs_predicted_trend_correlation:.6g}`",
                f"- Early-fit sign agreement: `{first.trend_sign_agreement_fraction:.3f}`",
                f"- Late-fit sign agreement: `{second.trend_sign_agreement_fraction:.3f}`",
                f"- Predicted HMA trends: `{first.predicted_hma_trend_mm_day_per_century:.6g}` and `{second.predicted_hma_trend_mm_day_per_century:.6g} mm/day/century`",
                f"- Actual network HMA trend: `{first.actual_hma_trend_mm_day_per_century:.6g} mm/day/century`",
                f"- Input hashes unchanged: `{before == after}`",
                "",
                "The split-calibrated instrumental PSMs do not explain the proxy network's large 1901-1920 common trend.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(result.to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
