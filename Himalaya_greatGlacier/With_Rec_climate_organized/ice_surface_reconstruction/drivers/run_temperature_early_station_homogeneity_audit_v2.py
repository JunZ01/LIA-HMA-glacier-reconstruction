from pathlib import Path

import pandas as pd


pd.DataFrame.product = property(lambda frame: frame["product"])

import run_temperature_early_station_homogeneity_audit_v1 as experiment


experiment.OUTPUT = experiment.LOOP / "temperature_early_station_homogeneity_audit_v2"
experiment.CONTRACT = (
    experiment.DRIVERS / "temperature_early_station_homogeneity_audit_v2_contract.md"
)
experiment.DRIVER = Path(__file__).resolve()


if __name__ == "__main__":
    raise SystemExit(experiment.main())
