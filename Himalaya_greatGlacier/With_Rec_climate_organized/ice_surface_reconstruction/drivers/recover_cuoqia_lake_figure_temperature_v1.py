from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
OUT = ROOT / (
    "validation_data/results/paleo_surface_validation_v0/"
    "climate_reconstruction_improvement_loop_v2/"
    "temperature_cuoqia_lake_numeric_recovery_v1"
)
FIGURE = OUT / "pdf_page_5_Im0.png"
ARTICLE_PDF = OUT / "fevo-11-1119869.pdf"
SUPPLEMENT = OUT / "Table_1.DOCX"

LOCK_START = 1921
LOCK_END = 1939

# Pixel centers read from the published axis ticks in the embedded Figure 2.
X_TICKS_PX = np.array([192, 374, 556, 738, 920, 1102, 1284], dtype=float)
X_TICKS_YEAR = np.array([1700, 1750, 1800, 1850, 1900, 1950, 2000], dtype=float)
RED_Y_TICKS_PX = np.array([73, 133, 192.5, 252, 312, 371], dtype=float)
RED_Y_TICKS_C = np.array([7.0, 6.5, 6.0, 5.5, 5.0, 4.5], dtype=float)
MAGENTA_Y_TICKS_PX = np.array([166, 226, 285, 345, 405, 464], dtype=float)
MAGENTA_Y_TICKS_C = np.array([7.0, 6.5, 6.0, 5.5, 5.0, 4.5], dtype=float)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


def linear_map(x: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    slope, intercept = np.polyfit(source, target, 1)
    return slope * x + intercept


def square_marker_centers(mask: np.ndarray, threshold: int = 65) -> list[tuple[int, int, int]]:
    """Locate dense 9x9 square markers while suppressing nearby duplicate windows."""
    integral = np.pad(mask.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    k = 9
    scores = (
        integral[k:, k:]
        - integral[:-k, k:]
        - integral[k:, :-k]
        + integral[:-k, :-k]
    )
    ys, xs = np.where(scores >= threshold)
    candidates = sorted(
        ((int(scores[y, x]), int(x + k // 2), int(y + k // 2)) for y, x in zip(ys, xs)),
        reverse=True,
    )
    kept: list[tuple[int, int, int]] = []
    for score, x, y in candidates:
        if all((x - px) ** 2 + (y - py) ** 2 > 9**2 for _, px, py in kept):
            kept.append((score, x, y))
    return sorted(kept, key=lambda item: (item[1], item[2]))


def extract_series(image: np.ndarray, series: str) -> pd.DataFrame:
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    if series == "CQ1":
        mask = (red > 245) & (green < 25) & (blue < 25)
        y_max = 375
        y_ticks_px = RED_Y_TICKS_PX
        y_ticks_c = RED_Y_TICKS_C
    elif series == "CQ2":
        mask = (red > 245) & (green < 25) & (blue > 245)
        y_max = 470
        y_ticks_px = MAGENTA_Y_TICKS_PX
        y_ticks_c = MAGENTA_Y_TICKS_C
    else:
        raise ValueError(series)

    # Restrict the search to the temperature panel and remove the plotted legend.
    mask[:20, :] = False
    mask[y_max:, :] = False
    mask[:, :190] = False
    mask[:, 1112:] = False  # Only recover through approximately 1953.
    mask[225:330, 225:560] = False

    # Remove the sealed interval before any marker detection or value conversion.
    lock_x0 = int(np.floor(linear_map(np.array([LOCK_START]), X_TICKS_YEAR, X_TICKS_PX)[0]))
    lock_x1 = int(np.ceil(linear_map(np.array([LOCK_END]), X_TICKS_YEAR, X_TICKS_PX)[0]))
    mask[:, lock_x0 : lock_x1 + 1] = False

    rows = []
    for score, x, y in square_marker_centers(mask):
        year_float = float(linear_map(np.array([x]), X_TICKS_PX, X_TICKS_YEAR)[0])
        year = int(round(year_float))
        if LOCK_START <= year <= LOCK_END:
            raise RuntimeError("Sealed-period marker escaped the pixel mask")
        if year < 1700 or year > 1953:
            continue
        temperature = float(linear_map(np.array([y]), y_ticks_px, y_ticks_c)[0])
        rows.append(
            {
                "series": series,
                "year_figure_estimate": year_float,
                "year_rounded": year,
                "temperature_march_october_c_figure_derived": temperature,
                "marker_x_px": x,
                "marker_y_px": y,
                "marker_score_9x9": score,
                "source_kind": "published_figure_digitization",
            }
        )
    frame = pd.DataFrame(rows)
    frame["same_rounded_year_index"] = frame.groupby("year_rounded").cumcount() + 1
    return frame


def trend_summary(frame: pd.DataFrame, start: int, end: int) -> dict[str, float | int | str | None]:
    subset = frame[(frame.year_rounded >= start) & (frame.year_rounded <= end)].copy()
    if len(subset) < 3:
        return {
            "series": str(frame.series.iloc[0]),
            "start": start,
            "end": end,
            "n": int(len(subset)),
            "slope_c_per_century": None,
        }
    slope = np.polyfit(
        subset.year_figure_estimate.to_numpy(),
        subset.temperature_march_october_c_figure_derived.to_numpy(),
        1,
    )[0]
    return {
        "series": str(frame.series.iloc[0]),
        "start": start,
        "end": end,
        "n": int(len(subset)),
        "slope_c_per_century": float(slope * 100.0),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (FIGURE, ARTICLE_PDF, SUPPLEMENT):
        if not path.exists():
            raise FileNotFoundError(path)

    image = np.array(Image.open(FIGURE).convert("RGB"))
    cq1 = extract_series(image.copy(), "CQ1")
    cq2 = extract_series(image.copy(), "CQ2")
    points = pd.concat([cq1, cq2], ignore_index=True)
    if points.year_rounded.between(LOCK_START, LOCK_END).any():
        raise RuntimeError("Sealed 1921-1939 values are present in the extracted table")
    points.to_csv(OUT / "cuoqia_temperature_figure_derived_safe_points.csv", index=False)

    trends = [
        trend_summary(cq1, 1700, 1920),
        trend_summary(cq1, 1850, 1920),
        trend_summary(cq2, 1870, 1920),
        trend_summary(cq1, 1940, 1950),
        trend_summary(cq2, 1940, 1950),
    ]
    pd.DataFrame(trends).to_csv(OUT / "cuoqia_safe_period_trends.csv", index=False)

    safe = Image.open(FIGURE).convert("RGB")
    draw = ImageDraw.Draw(safe)
    x0 = int(np.floor(linear_map(np.array([LOCK_START]), X_TICKS_YEAR, X_TICKS_PX)[0]))
    x1 = int(np.ceil(linear_map(np.array([LOCK_END]), X_TICKS_YEAR, X_TICKS_PX)[0]))
    draw.rectangle((x0, 0, x1, safe.height), fill=(230, 230, 230))
    draw.text((x0 + 4, 8), "SEALED 1921-1939", fill=(0, 0, 0))
    safe.save(OUT / "published_figure2_sealed_period_masked.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for label, frame, color in (("CQ1", cq1, "#c62828"), ("CQ2", cq2, "#b000b5")):
        ax.plot(
            frame.year_figure_estimate,
            frame.temperature_march_october_c_figure_derived,
            marker="s",
            markersize=4,
            linewidth=1.2,
            label=f"{label} (figure-derived)",
            color=color,
        )
    ax.axvspan(LOCK_START, LOCK_END, color="0.85", zorder=5, label="sealed: not extracted")
    ax.set_xlim(1700, 1951)
    ax.set_xlabel("Year")
    ax.set_ylabel("March-October temperature (degC)")
    ax.set_title("Lake Cuoqia quantitative temperature recovered from published Figure 2")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cuoqia_safe_temperature_curve.png", dpi=180)
    plt.close(fig)

    hashes = {path.name: sha256(path) for path in (FIGURE, ARTICLE_PDF, SUPPLEMENT)}
    summary = {
        "experiment": "temperature_cuoqia_lake_numeric_recovery_v1",
        "site": {
            "name": "Lake Cuoqia",
            "latitude": 27.4052,
            "longitude": 99.7722,
            "elevation_m": 3960,
            "region": "southeastern Tibetan Plateau / eastern HMA margin",
        },
        "paper_doi": "10.3389/fevo.2023.1119869",
        "official_article_url": "https://www.frontiersin.org/journals/ecology-and-evolution/articles/10.3389/fevo.2023.1119869/full",
        "official_supplement_object_url": "https://public-pages-files-2025.frontiersin.org/articles/1119869/file/table_1.docx/1119869_supplementary-materials_tables_1_docx/1",
        "target": "March-October quantitative temperature from site-specific brGDGT calibration",
        "source_numeric_status": "supplement_has_figures_only; temperatures recovered from embedded published Figure 2",
        "point_counts": {"CQ1": int(len(cq1)), "CQ2": int(len(cq2)), "total": int(len(points))},
        "trends": trends,
        "digitization_limits": [
            "figure-derived values are approximate and are not an author-supplied numeric table",
            "pixel-to-axis conversion uncertainty is small relative to unresolved proxy calibration and age-model uncertainty",
            "CQ1 and CQ2 have different plotted axes and must not be pooled as one absolute-temperature series without alignment",
        ],
        "independence": "physical lake-sediment brGDGT evidence; independent of project tree-ring PSM and gridded instrumental products",
        "sealed_period": {
            "years": "1921-1939",
            "values_extracted": False,
            "derived_image_masked": True,
        },
        "prohibited_source_accessed": False,
        "production_modified": False,
        "enkf_run": False,
        "oggm_run": False,
        "source_sha256": hashes,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    trend_lines = []
    for item in trends:
        slope = item["slope_c_per_century"]
        value = "not estimable" if slope is None else f"{slope:+.3f} degC/century"
        trend_lines.append(
            f"- {item['series']} {item['start']}-{item['end']}: n={item['n']}, {value}."
        )
    report = "\n".join(
        [
            "# Lake Cuoqia numeric recovery v1",
            "",
            "## Result",
            "",
            "The official Frontiers supplement was recovered without registration, but it contains only two supplementary figures and no age-temperature table. The quantitative March-October temperature curves were therefore recovered from the original-resolution Figure 2 embedded in the official article PDF.",
            "",
            "The recovered points are independent physical lake-sediment brGDGT evidence, not an additional tree-ring reconstruction or gridded-product vote. They are suitable for testing low-frequency direction and approximate amplitude at the southeastern HMA margin, but not for exact annual RMSE scoring.",
            "",
            "## Safe-period trends",
            "",
            *trend_lines,
            "",
            "## Restrictions",
            "",
            "- Values for 1921-1939 were not extracted; the derived review image masks that interval.",
            "- Figure-derived values remain approximate and do not replace an author-supplied table.",
            "- CQ1 and CQ2 use different plotted temperature axes and are reported separately.",
            "- No production file, EnKF, or OGGM run was touched.",
            "- prohibited_source_accessed=false.",
            "",
        ]
    )
    (OUT / "FINAL_RESULT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
