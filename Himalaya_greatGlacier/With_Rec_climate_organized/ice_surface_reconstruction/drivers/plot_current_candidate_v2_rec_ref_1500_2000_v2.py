"""Compatibility launcher correcting the v1 external-reference path literal."""

from pathlib import Path


driver = Path(__file__).with_name("plot_current_candidate_v2_rec_ref_1500_2000_v1.py")
source = driver.read_text(encoding="utf-8")
start = source.index("REFERENCE = Path(")
end = source.index("CANDIDATE = PROJECT", start)
replacement = (
    'REFERENCE = Path(r"E:\\3spyderProject\\HMA_500\\Down_scale_new\\out_data") '
    '/ "climate_anomaly_05deg_with_ensemble_1000-2000.nc"\n\n'
)
source = source[:start] + replacement + source[end:]
namespace = {"__name__": "__main__", "__file__": str(driver)}
exec(compile(source, str(driver), "exec"), namespace)
