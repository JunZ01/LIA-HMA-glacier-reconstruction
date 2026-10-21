"""Compatibility launch for the frozen v1 blind test.

V1 was sealed before unlock but contains one non-scientific Python parse error:
its Windows reference-directory raw string ends in a backslash. This launcher
changes only that literal in memory and executes the otherwise unchanged v1
source. The frozen correction, formulas, periods, and gates are unchanged.
"""

from __future__ import annotations

from pathlib import Path


SOURCE = Path(__file__).with_name(
    "run_temperature_low_frequency_blind_test_1921_1939_v1.py"
)
BROKEN = 'r"E:\\3spyderProject\\HMA_500\\Down_scale_new\\out_data\\"'
FIXED = 'r"E:\\3spyderProject\\HMA_500\\Down_scale_new\\out_data"'

source = SOURCE.read_text(encoding="utf-8")
if source.count(BROKEN) != 1:
    raise RuntimeError("unexpected-v1-reference-path-literal")
source = source.replace(BROKEN, FIXED)
exec(compile(source, str(SOURCE), "exec"), globals(), globals())
