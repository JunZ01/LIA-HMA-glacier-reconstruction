"""Prepare the 214 new source-native LMR temperature identities."""

from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "LIA_PROJECT_FACTS.md").exists(): return parent
    raise RuntimeError("project-root-not-found")


def state(path: Path) -> dict[str, object]:
    stat=path.stat(); digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""): digest.update(block)
    return {"path":str(path),"size_bytes":int(stat.st_size),"mtime_ns":int(stat.st_mtime_ns),"sha256":digest.hexdigest()}


def season_code(value: str) -> str:
    text=str(value).replace("[","").replace("]","").replace(","," "); numbers=text.split()
    if numbers==["6","7","8"]: return "JJA"
    if numbers==[str(month) for month in range(1,13)]: return "ANN"
    if numbers==["4","5","6"]: return "AMJ"
    if numbers==["4"]: return "APR"
    raise RuntimeError(f"unsupported-season:{value}")


def main() -> int:
    project=root(); base=project/"validation_data/results/paleo_surface_validation_v0/climate_reconstruction_improvement_loop_v2"
    output=base/"round_13_new_lmr_temperature_network_validation_v1"
    lmr_root=Path(r"D:\Users\JUNZ\MATLAB\Projects\untitled\PHYDA-v1-master\proxy-data\LMR_NCDC_database_v.0.2.0")
    paths={
        "round11":base/"round_11_full_pages_lmr_hma_source_inventory_v2/full_pages_lmr_hma_candidate_inventory.csv",
        "lmr_data":lmr_root/"LMR_data.mat","lmr_metadata":lmr_root/"lmr_meta.csv",
        "contract":output/"ROUND_CONTRACT.md","driver":Path(__file__).resolve(),
    }
    for path in paths.values():
        if not path.exists(): raise FileNotFoundError(path)
    target=output/"new_lmr_temperature_network_v1.mat"
    if target.exists(): raise FileExistsError(target)
    before={name:state(path) for name,path in paths.items()}
    inventory=pd.read_csv(paths["round11"])
    selected=inventory[(inventory["eligible_before_sequence_dedup"]==True)&(inventory["network_role"]=="documented_temperature")&(inventory["source_database"]=="LMR_v0_2_0")].copy()
    selected=selected.sort_values("formal_identity")
    if len(selected)!=214 or selected["formal_identity"].nunique()!=214: raise RuntimeError("identity-contract-failed")
    if set(selected["documented_direction"])!={"positive"}: raise RuntimeError("direction-contract-failed")
    metadata=pd.read_csv(paths["lmr_metadata"]); metadata["NCDC ID"]=metadata["NCDC ID"].astype(str).str.strip(); metadata=metadata.set_index("NCDC ID",drop=False)
    raw=loadmat(paths["lmr_data"],squeeze_me=True,struct_as_record=False,variable_names=["lmr2k_data","lmr_data_names","year"])
    years_all=np.asarray(raw["year"],dtype=int).reshape(-1); matrix=np.asarray(raw["lmr2k_data"],dtype=float)
    names=[str(value).strip() for value in np.asarray(raw["lmr_data_names"]).reshape(-1)]
    missing=sorted(set(metadata.index)-set(names)); unknown=sorted(set(names)-set(metadata.index))
    if len(missing)==1 and unknown==["nan"] and names.count("nan")==1: names[names.index("nan")]=missing[0]
    if set(names)!=set(metadata.index): raise RuntimeError("lmr-name-closure-failed")
    lookup={name:index for index,name in enumerate(names)}; identities=selected["formal_identity"].tolist()
    mask=(years_all>=1940)&(years_all<=2000); years=years_all[mask]
    if not np.array_equal(years,np.arange(1940,2001)) or np.any((years>=1921)&(years<=1939)): raise RuntimeError("year-contract-failed")
    observations=np.column_stack([matrix[mask,lookup[identity]] for identity in identities])
    locations=np.asarray([[float(metadata.loc[identity,"Lat (N)"]),float(metadata.loc[identity,"Lon (E)"])] for identity in identities])
    seasons=[season_code(metadata.loc[identity,"Seasonality"]) for identity in identities]
    counts=np.sum(np.isfinite(observations),axis=0)
    if np.any(counts<20): raise RuntimeError("coverage-contract-failed")
    savemat(target,{"years":years.reshape(-1,1),"observations":observations,"locations":locations,"identity_ids":np.asarray(identities,dtype=object).reshape(-1,1)},do_compression=True)
    rows=[]
    for index,identity in enumerate(identities):
        meta=metadata.loc[identity]; rows.append({"network_index_1based":index+1,"formal_identity":identity,"latitude":locations[index,0],"longitude":locations[index,1],"season_code":seasons[index],"calibration_finite_count":int(counts[index]),"measurement":meta["Proxy measurement"]})
    with (output/"network_manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={"status":"complete","identity_count":214,"unique_site_count":int(len({(round(x,6),round(y,6)) for x,y in locations})),"season_counts":dict(pd.Series(seasons).value_counts()),"locked_1921_1939_accessed":False}
    (output/"preparation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    after={name:state(path) for name,path in paths.items()}; (output/"preparation_source_manifest.json").write_text(json.dumps({"sources_before":before,"sources_after":after,"source_states_unchanged":before==after},indent=2),encoding="utf-8")
    if before!=after: raise RuntimeError("source-state-changed")
    print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
