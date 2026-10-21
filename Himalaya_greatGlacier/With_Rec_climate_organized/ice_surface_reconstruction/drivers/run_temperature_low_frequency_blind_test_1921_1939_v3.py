"""Second compatibility launch for the frozen v1 blind-test source."""

from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name(
    "run_temperature_low_frequency_blind_test_1921_1939_v1.py"
)
START = "REFERENCE_SOURCE = Path("
END = "\nHMA_MASK ="
REPLACEMENT = (
    'REFERENCE_SOURCE = Path(r"E:\\3spyderProject\\HMA_500\\Down_scale_new\\'
    'out_data\\climate_anomaly_05deg_with_ensemble_1000-2000.nc")'
)

source = SOURCE.read_text(encoding="utf-8")
left = source.index(START)
right = source.index(END, left)
source = source[:left] + REPLACEMENT + source[right:]
exec(compile(source, str(SOURCE), "exec"), globals(), globals())
