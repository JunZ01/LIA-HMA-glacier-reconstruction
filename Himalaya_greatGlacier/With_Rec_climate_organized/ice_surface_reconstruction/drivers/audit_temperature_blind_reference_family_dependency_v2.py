"""Execute the v1 audit with the Windows reference path corrected."""

from pathlib import Path


SOURCE = Path(__file__).with_name(
    "audit_temperature_blind_reference_family_dependency_v1.py"
)
START = "REFERENCE = Path("
END = "\nASSIGNMENT ="
REPLACEMENT = (
    'REFERENCE = Path("E:/3spyderProject/HMA_500/Down_scale_new/out_data/'
    'climate_anomaly_05deg_with_ensemble_1000-2000.nc")'
)

source = SOURCE.read_text(encoding="utf-8")
left = source.index(START)
right = source.index(END, left)
source = source[:left] + REPLACEMENT + source[right:]
source = source.replace(
    "climate_reconstruction_temperature_blind_reference_family_dependency_audit_v1",
    "climate_reconstruction_temperature_blind_reference_family_dependency_audit_v2",
)
namespace = {
    "__name__": "__main__",
    "__file__": str(Path(__file__).resolve()),
}
exec(compile(source, str(SOURCE), "exec"), namespace)
