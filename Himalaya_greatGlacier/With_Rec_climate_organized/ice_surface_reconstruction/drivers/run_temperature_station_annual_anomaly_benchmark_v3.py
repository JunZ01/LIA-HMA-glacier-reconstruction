from pathlib import Path

import rasterio  # Load GDAL before h5py on this Windows environment.

import run_temperature_station_annual_anomaly_benchmark_v2 as experiment


experiment.OUTPUT = (
    experiment.base.RESULT_ROOT / "temperature_station_annual_anomaly_benchmark_v3"
)
experiment.CONTRACT = (
    experiment.base.DRIVERS / "temperature_station_annual_anomaly_benchmark_v3_contract.md"
)
experiment.DRIVER = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(experiment.main())
