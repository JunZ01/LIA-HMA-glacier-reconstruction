"""Overflow-safe execution wrapper for Bamu digitization blind validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

import digitize_validate_temperature_bamu_v1 as impl


impl.OUTPUT = impl.base.RESULTS / "climate_reconstruction_temperature_bamu_digitized_blind_validation_v2"
impl.CONTRACT = impl.OUTPUT / "ROUND_CONTRACT.md"


def digitize(image: np.ndarray, mask_name: str) -> pd.DataFrame:
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)
    if mask_name == "strict":
        red = (r >= 240) & (g <= 25) & (b <= 25)
    elif mask_name == "medium":
        red = (r >= 200) & (g <= 80) & (b <= 80) & (r >= 2 * np.maximum(g, b))
    elif mask_name == "broad":
        red = (r >= 150) & (r >= 1.7 * np.maximum(g, b))
    else:
        raise ValueError(mask_name)
    x_transform = impl.fit_axis(impl.X_PIXELS, impl.X_YEARS)
    y_transform = impl.fit_axis(impl.Y_PIXELS, impl.Y_TEMPERATURE)
    rows = []
    for x in np.where(np.any(red, axis=0))[0]:
        year = float(np.polyval(x_transform, x))
        if 1921 <= year < 1940:
            continue
        ys = np.where(red[:, x])[0]
        ys = ys[(ys >= 10600) & (ys <= 11800)]
        if len(ys):
            y_center = float(np.median(ys))
            rows.append({
                "pixel_x": int(x), "pixel_y_center": y_center, "year": year,
                "temperature_degC": float(np.polyval(y_transform, y_center)),
            })
    frame = pd.DataFrame(rows).sort_values("year")
    if frame.empty:
        raise ValueError(f"No red centerline extracted for {mask_name}")
    return frame


impl.digitize = digitize


if __name__ == "__main__":
    impl.main()
