"""Build a no-dynamics OGGM MonthlyTI reference for the frozen 24 lanes."""
from __future__ import annotations
import copy, gzip, json, pickle, sys
from pathlib import Path
import numpy as np

DRIVERS=Path(__file__).resolve().parent
sys.path.insert(0,str(DRIVERS))
from lia_1d_gpu_preflight_guard_v2 import FIXTURE_MANIFEST_PATH, PROJECT_ROOT, load_json, sha256

def main():
    from oggm import cfg, utils
    from oggm.core.massbalance import MonthlyTIModel
    cfg.initialize(logging_level="CRITICAL")
    fixture=load_json(FIXTURE_MANIFEST_PATH)
    npz_path=PROJECT_ROOT/fixture["npz_path"]
    if sha256(npz_path)!=fixture["npz_sha256"]: raise RuntimeError("fixture hash mismatch")
    out=FIXTURE_MANIFEST_PATH.parent/"mb_reference_cpu_v1.npz"
    if out.exists(): raise FileExistsError(out)
    reference=np.zeros((24,int(fixture["max_nx"])),dtype=np.float64)
    for lane in fixture["lanes"]:
        with gzip.open(lane["flowline_path"],"rb") as handle: fl=pickle.load(handle)[0]
        climate=Path(lane["climate_path"]); gdir=utils.GlacierDirectory(lane["rgi_id"],base_dir=str(climate.parents[3]))
        cfg.PARAMS["temp_melt"]=float(lane["parameters"]["temp_melt"])
        mb=MonthlyTIModel(gdir,filename="gcm_data",input_filesuffix=climate.stem[len("gcm_data"):],ys=1836,ye=2000,repeat=False,
                          melt_f=float(lane["parameters"]["melt_f"]),prcp_fac=float(lane["parameters"]["prcp_fac"]),
                          temp_bias=float(lane["parameters"]["delta_t"]),check_calib_params=False)
        values=mb.get_annual_mb(np.asarray(fl.surface_h,dtype=np.float64),year=1836.0)
        reference[int(lane["lane"]),:len(values)]=values
    temporary=out.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle: np.savez(handle,annual_mb_1836=reference)
    temporary.replace(out)
    print(json.dumps({"status":"pass","path":str(out.relative_to(PROJECT_ROOT)),"sha256":sha256(out)}))
    return 0
if __name__=="__main__": raise SystemExit(main())
