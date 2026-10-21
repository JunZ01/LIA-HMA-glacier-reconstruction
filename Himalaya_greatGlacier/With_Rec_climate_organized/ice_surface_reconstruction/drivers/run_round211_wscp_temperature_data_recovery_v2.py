from __future__ import annotations

import run_round211_wscp_temperature_data_recovery_v1 as audit


audit.OUT = (
    audit.LOOP / "round_211_wscp_temperature_data_recovery_v2"
)
audit.DRIVER = (
    audit.ROOT
    / "Himalaya_greatGlacier/With_Rec_climate_organized"
    / "ice_surface_reconstruction/drivers"
    / "run_round211_wscp_temperature_data_recovery_v2.py"
)
audit.ARCHIVE_URL = "https://www.geodoi.ac.cn/WebEn/down.aspx?fileID=6140"


if __name__ == "__main__":
    audit.main()
