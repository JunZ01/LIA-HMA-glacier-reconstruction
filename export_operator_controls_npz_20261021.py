"""Export the three frozen modern controls as plain .npz (numpy only) for offline operator work."""
from pathlib import Path
import numpy as np
import diagnose_extension_operators_20261021 as diag

EXPORT = diag.BASE / "extension_operator_controls_20261021_v1" / "npz_export"
EXPORT.mkdir(parents=True, exist_ok=True)

for case in diag.select_cases():
    d = diag.load_case(case)
    uid = case["lia_unique_id"]
    np.savez_compressed(
        EXPORT / f"modern_control_{uid}.npz",
        reference_H_m=d["hw"].astype("float32"),
        topo_m=d["topo"].astype("float32"),
        modern_mask=d["modern"],
        constraint_rows=d["rr"].astype("int32"),
        constraint_cols=d["cc"].astype("int32"),
        dx_m=d["dx"],
        native_volume_km3=d["native_volume"],
        inversion_1d_volume_km3=d["inversion_1d_volume"],
        region=case["region"], rgi_id=case["rgi_id"], lia_unique_id=uid,
    )
    print(uid, case["region"], d["modern"].sum(), "cells", len(d["rr"]), "constraints")
print("written to", EXPORT)
