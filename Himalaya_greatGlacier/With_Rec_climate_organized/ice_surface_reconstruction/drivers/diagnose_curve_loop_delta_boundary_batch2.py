"""Consolidate and plot the second three-system local delta-T batch.

No OGGM task is launched. The diagnostic combines the initial controller
profile, the nine-chain boundary batch, and the 1077 refinement.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYSTEMS=(1077,2226,2928)
ROUTING_TOLERANCE=0.20
STABILITY_LIMIT=0.02


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def as_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value,(bool,np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true","1","yes"}


def load_scores(root: Path) -> pd.DataFrame:
    base=root/"validation_data"/"results"/"paleo_surface_validation_v0"
    sources=[
        ("initial_delta_grid",base/"lia_191_controller_v1"/"scores.csv"),
        ("batch2_boundary",base/"lia_191_curve_loop_delta_boundary_batch2_v1"/"scores.csv"),
        ("1077_refine",base/"lia_191_curve_loop_delta_refine_1077_r1_v1"/"scores.csv"),
    ]
    frames=[]
    for source,path in sources:
        frame=pd.read_csv(path)
        frame=frame[pd.to_numeric(frame["lia_unique_id"],errors="coerce").isin(SYSTEMS)].copy()
        frame["experiment_source"]=source
        frames.append(frame)
    out=pd.concat(frames,ignore_index=True,sort=False)
    numeric=[
        "lia_unique_id","delta_t","melt_f","spinup_stability_fraction",
        "lia_target_m","rgi_target_m","lia_rgi_separation_m",
        "forward_peak_m","forward_2000_m","e_lia_sep","e_rgi_sep",
        "objective_sep","peak_to_2000_amplitude_m","amplitude_ratio_to_target",
    ]
    for column in numeric:
        out[column]=pd.to_numeric(out[column],errors="coerce")
    out["lia_unique_id"]=out["lia_unique_id"].astype(int)
    out["technical_failure_bool"]=out["technical_failure"].map(as_bool)
    out["spinup_converged_bool"]=out["spinup_converged"].map(as_bool)
    out["stable_chain"]=(
        ~out["technical_failure_bool"]
        & out["summary_status"].eq("ok")
        & out["spinup_converged_bool"]
        & out["spinup_stability_fraction"].le(STABILITY_LIMIT)
    )
    out["routing_joint_pass"]=(
        out["stable_chain"]
        & out["e_lia_sep"].le(ROUTING_TOLERANCE)
        & out["e_rgi_sep"].le(ROUTING_TOLERANCE)
    )
    out["e_lia_total"]=abs(out["forward_peak_m"]-out["lia_target_m"])/out["lia_target_m"]
    out["e_rgi_total"]=abs(out["forward_2000_m"]-out["rgi_target_m"])/out["rgi_target_m"]
    out["both_endpoints_within_10pct_total"]=(
        out["stable_chain"]
        & out["e_lia_total"].le(0.10)
        & out["e_rgi_total"].le(0.10)
    )
    out=out.drop_duplicates(["lia_unique_id","delta_t","melt_f"],keep="last")
    return out.sort_values(["lia_unique_id","delta_t"]).reset_index(drop=True)


def load_forward(output_dir: str) -> pd.DataFrame:
    frame=pd.read_csv(Path(output_dir)/"timeseries.csv")
    frame=frame[frame["phase"].eq("forward")].copy()
    frame["year"]=pd.to_numeric(frame["year"],errors="coerce")
    frame["length_km"]=pd.to_numeric(frame["length_m"],errors="coerce")/1000
    return frame.dropna(subset=["year","length_km"]).sort_values("year")


def plot_system(group: pd.DataFrame,plot_dir: Path) -> Path:
    uid=int(group.iloc[0]["lia_unique_id"])
    stable=group[group["stable_chain"]].copy().sort_values("delta_t")
    best=stable.sort_values(["objective_sep","e_lia_total","e_rgi_total"]).iloc[0]
    lia=float(group.iloc[0]["lia_target_m"])/1000
    rgi=float(group.iloc[0]["rgi_target_m"])/1000
    separation=abs(lia-rgi)
    corridor=(min(lia,rgi)-0.2*separation,max(lia,rgi)+0.2*separation)

    fig,axes=plt.subplots(1,2,figsize=(13.2,5.8))
    colors=plt.cm.viridis(np.linspace(0.08,0.90,max(len(stable),2)))
    for color,row in zip(colors,stable.itertuples(index=False)):
        forward=load_forward(str(row.output_dir))
        is_best=np.isclose(float(row.delta_t),float(best["delta_t"]))
        label=(
            f"dT={float(row.delta_t):+.5f} K | "
            f"Esep=({float(row.e_lia_sep):.2f},{float(row.e_rgi_sep):.2f})"
        )
        for ax in axes:
            ax.plot(
                forward["year"],forward["length_km"],color=color,
                lw=3.0 if is_best else 1.7,label=label if ax is axes[0] else None,
            )

    for ax in axes:
        ax.axhspan(corridor[0],corridor[1],color="0.7",alpha=0.14)
        ax.axhline(lia,color="#d95f02",lw=1.7)
        ax.axhline(rgi,color="#1b9e77",lw=1.7)
        ax.set_xlim(1470,2000)
        ax.set_xlabel("Year")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("Main-flowline length (km)")
    axes[0].set_title("All stable branches")
    axes[0].legend(frameon=False,fontsize=7,ncol=1,loc="best")
    pad=max(0.35*separation,0.35)
    axes[1].set_ylim(min(lia,rgi)-pad,max(lia,rgi)+pad)
    axes[1].set_title("Endpoint-target zoom")

    fig.suptitle(
        f"LIA system {uid} | best dT={float(best['delta_t']):+.5f} K | "
        f"LIA total error={100*float(best['e_lia_total']):.1f}% | "
        f"RGI total error={100*float(best['e_rgi_total']):.1f}% | "
        f"amplitude={100*float(best['amplitude_ratio_to_target']):.1f}%"
    )
    fig.subplots_adjust(left=0.07,right=0.99,bottom=0.12,top=0.86,wspace=0.18)
    plot_dir.mkdir(parents=True,exist_ok=True)
    path=plot_dir/f"system_{uid}_delta_boundary_batch2.png"
    fig.savefig(path,dpi=180)
    plt.close(fig)
    return path


def main() -> int:
    root=project_root()
    output=(
        root/"validation_data"/"results"/"paleo_surface_validation_v0"
        /"lia_191_curve_loop_delta_boundary_batch2_v1"/"diagnostics"/"consolidated_v1"
    )
    output.mkdir(parents=True,exist_ok=True)
    scores=load_scores(root)
    scores.to_csv(output/"combined_delta_diagnostics.csv",index=False)
    paths={}
    rows=[]
    for uid,group in scores.groupby("lia_unique_id",sort=True):
        path=plot_system(group,output/"plots")
        paths[int(uid)]=str(path)
        stable=group[group["stable_chain"]]
        best=stable.sort_values(["objective_sep","e_lia_total","e_rgi_total"]).iloc[0]
        rows.append({
            "lia_unique_id":int(uid),
            "stable_chain_count":int(group["stable_chain"].sum()),
            "technical_failure_count":int(group["technical_failure_bool"].sum()),
            "routing_joint_pass_count":int(group["routing_joint_pass"].sum()),
            "within_10pct_total_count":int(group["both_endpoints_within_10pct_total"].sum()),
            "best_delta_t":float(best["delta_t"]),
            "best_e_lia_sep":float(best["e_lia_sep"]),
            "best_e_rgi_sep":float(best["e_rgi_sep"]),
            "best_e_lia_total":float(best["e_lia_total"]),
            "best_e_rgi_total":float(best["e_rgi_total"]),
            "best_amplitude_ratio":float(best["amplitude_ratio_to_target"]),
            "plot_path":str(path),
        })
    summary=pd.DataFrame(rows)
    summary.to_csv(output/"system_summary.csv",index=False)
    record={
        "launched_oggm":False,
        "new_batch2_chain_count":9,
        "new_1077_refinement_chain_count":3,
        "technical_failure_count":int(scores["technical_failure_bool"].sum()),
        "routing_joint_pass_systems":summary.loc[summary.routing_joint_pass_count.gt(0),"lia_unique_id"].astype(int).tolist(),
        "systems_with_total_length_endpoint_errors_below_10pct":summary.loc[summary.within_10pct_total_count.gt(0),"lia_unique_id"].astype(int).tolist(),
        "routing_tolerance_is_final_acceptance":False,
        "system_summary":summary.to_dict(orient="records"),
    }
    (output/"diagnosis.json").write_text(
        json.dumps(record,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
