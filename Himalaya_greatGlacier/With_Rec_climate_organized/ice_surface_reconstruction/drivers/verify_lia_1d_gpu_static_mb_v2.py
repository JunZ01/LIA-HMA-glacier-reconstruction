"""Compare the CUDA MonthlyTI formula with OGGM without advancing dynamics."""
from __future__ import annotations
import json, sys, tempfile
from pathlib import Path
import torch
import numpy as np

DRIVERS=Path(__file__).resolve().parent
sys.path.insert(0,str(DRIVERS))
from lia_1d_gpu_preflight_guard_v2 import FIXTURE_MANIFEST_PATH, PROJECT_ROOT, load_json, sha256
from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState

def main():
    fixture=load_json(FIXTURE_MANIFEST_PATH); bridge=PROJECT_ROOT/fixture["npz_path"]
    reference=FIXTURE_MANIFEST_PATH.parent/"mb_reference_cpu_v1.npz"
    output=FIXTURE_MANIFEST_PATH.parent/"mb_equivalence_cuda_v2.json"
    if output.exists(): raise FileExistsError(output)
    with np.load(bridge) as z: a={k:z[k] for k in z.files}
    device=torch.device("cuda:0")
    def t(name,dtype=torch.float64): return torch.as_tensor(a[name],dtype=dtype,device=device)
    state=BatchedFlowlineState(bed_h=t("bed_h"),widths_m=t("widths_m"),thick=t("thick"),valid_mask=t("valid_mask",torch.bool),
        nx=t("nx",torch.int64),dx_meter=t("dx_meter"),glen_a=t("glen_a"),melt_f=t("melt_f"),prcp_fac=t("prcp_fac"),
        temp_bias=t("temp_bias"),temp_melt=t("temp_melt"),climate_temp=t("climate_temp"),climate_prcp=t("climate_prcp"),
        climate_gradient=t("climate_gradient"),climate_ref_hgt=t("climate_ref_hgt"),climate_years=t("climate_years",torch.int64))
    got=state.annual_mass_balance().cpu().numpy()
    with np.load(reference) as z: expected=z["annual_mb_1836"]
    mask=a["valid_mask"]; difference=np.abs(got[mask]-expected[mask]); denominator=np.maximum(np.abs(expected[mask]),1e-30)
    payload={"schema":"lia-1d-gpu-static-mb-equivalence-v2","status":"pass" if float(difference.max())<=1e-15 else "fail",
             "dynamics_performed":False,"lanes":24,"max_absolute_m_ice_s":float(difference.max()),
             "max_relative":float((difference/denominator).max()),"bridge_sha256":sha256(bridge),
             "cpu_reference_sha256":sha256(reference),"torch_version":torch.__version__,"device":torch.cuda.get_device_name(0)}
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=output.parent,delete=False) as handle:
        json.dump(payload,handle,indent=2);handle.write("\n");temporary=Path(handle.name)
    temporary.replace(output); print(json.dumps(payload,indent=2)); return 0 if payload["status"]=="pass" else 2
if __name__=="__main__": raise SystemExit(main())
