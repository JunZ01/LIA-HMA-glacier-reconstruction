from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(r"E:\3spyderProject\1_LIA_himalaya_oggm")
SOURCE = PROJECT / (
    r"validation_data\sources\external_hma_temperature_kalam_summer_v1"
    r"\springer_vor_fig5.png"
)
OUTPUT = PROJECT / (
    r"validation_data\results\paleo_surface_validation_v0"
    r"\climate_reconstruction_temperature_west_b1_alpha0p25_kalam_blind_test_v1"
)

EXPECTED_SIZE = (2056, 1575)
X_LEFT = 176
X_RIGHT = 2001
YEAR_LEFT = 1790
YEAR_RIGHT = 2018
YEAR_STOP = 1900
PANEL_TOP = 730
PANEL_BOTTOM = 1470
PLOT_TOP = 764
PLOT_BOTTOM = 1401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(SOURCE).convert("RGB")
    if image.size != EXPECTED_SIZE:
        raise RuntimeError(f"Unexpected source size: {image.size}")

    gray = np.asarray(image.convert("L"))
    # The published frame is three pixels thick at both endpoints. This check
    # protects the year mapping without inspecting the plotted time series.
    left_dark = int(np.count_nonzero(gray[:, X_LEFT] < 90))
    right_dark = int(np.count_nonzero(gray[:, X_RIGHT] < 90))
    if min(left_dark, right_dark) < 1200:
        raise RuntimeError(
            f"Expected frame not found: left={left_dark}, right={right_dark}"
        )

    x_stop_float = X_LEFT + (
        (YEAR_STOP - YEAR_LEFT) / (YEAR_RIGHT - YEAR_LEFT)
    ) * (X_RIGHT - X_LEFT)
    x_stop = int(np.floor(x_stop_float))
    if x_stop >= X_RIGHT:
        raise RuntimeError("Invalid blind-crop boundary")

    context_path = OUTPUT / "springer_vor_fig5b_through_1900_context.png"
    plot_path = OUTPUT / "springer_vor_fig5b_through_1900_plot.png"
    image.crop((0, PANEL_TOP, x_stop + 1, PANEL_BOTTOM)).save(context_path)
    image.crop((X_LEFT, PLOT_TOP, x_stop + 1, PLOT_BOTTOM)).save(plot_path)

    manifest = {
        "source": str(SOURCE),
        "source_url": (
            "https://media.springernature.com/full/springer-static/image/"
            "art%3A10.1007%2Fs00484-024-02683-9/MediaObjects/"
            "484_2024_2683_Fig5_HTML.png"
        ),
        "source_sha256": sha256(SOURCE),
        "source_size": list(image.size),
        "axis_mapping": {
            "x_left": X_LEFT,
            "year_left": YEAR_LEFT,
            "x_right": X_RIGHT,
            "year_right": YEAR_RIGHT,
            "x_stop_float": x_stop_float,
            "x_stop_inclusive": x_stop,
            "year_stop": YEAR_STOP,
        },
        "frame_dark_pixel_counts": {
            "left": left_dark,
            "right": right_dark,
        },
        "outputs": {
            "context": str(context_path),
            "context_sha256": sha256(context_path),
            "plot": str(plot_path),
            "plot_sha256": sha256(plot_path),
        },
        "locked_pixels_exposed": False,
    }
    (OUTPUT / "blind_crop_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
