"""One-shot response-corrected uniform-thickness candidate after v2 formal50."""

from __future__ import annotations

import copy, gzip, json, pickle
from pathlib import Path
import numpy as np

from . import network_connected_front_system2005_climate_v2 as climate
from . import network_coupled_semiimplicit_system2005_zero_mb_v1 as base
from . import network_topology_adaptive_split_system2005_zero_mb_v1 as topology
from .run_system2005_policy_v6_joint_candidate_v1 import hypsometry_cdf, reverse_counts

ROOT=Path(__file__).resolve().parents[4]
RESULTS=ROOT/'validation_data/results/paleo_surface_validation_v0'
SOURCE=RESULTS/'lia_191_system2005_policy_v6_year9_materialization_v1/lia_natural_state_year9.pkl.gz'
V2=RESULTS/'lia_191_system2005_policy_v6_joint_candidate_v2/summary.json'
OUT=RESULTS/'lia_191_system2005_policy_v6_uniform_thickness_v3_design'
TOL=1e-6
TARGET_DV=0.0125
REGISTERED=25985.52229946727

def lifted(source,height):
    candidate=copy.deepcopy(source)
    for fl in candidate:
        thick=np.asarray(fl.thick,dtype=float).copy(); active=thick>TOL
        thick[active]+=float(height); fl.thick=thick
    return candidate

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    v2=json.loads(V2.read_text(encoding='utf-8'))
    if v2.get('status')!='formal-50y-fail-mechanism-closed': raise RuntimeError('v2 formal result not closed')
    formal=v2['formal']
    if formal['checks']['connected_length'] is not True or formal['checks']['volume_change'] is not False: raise RuntimeError('v2 is not the expected volume-only failure')
    observed=float(formal['volume_change_fraction']); old_scale=float(v2['static']['volume_scale'])
    scale=old_scale*(1.0+observed)/(1.0+TARGET_DV)
    with gzip.open(SOURCE,'rb') as h: source=pickle.load(h)
    source_volume=base.total_volume(source); target=source_volume*scale
    low,high=0.0,100.0
    for _ in range(100):
        mid=0.5*(low+high)
        if base.total_volume(lifted(source,mid))<target: low=mid
        else: high=mid
    height=0.5*(low+high); candidate=lifted(source,height)
    volume=base.total_volume(candidate); source_area=climate.ice_area(source); area=climate.ice_area(candidate)
    source_root=climate.root_metrics(source,REGISTERED); root=climate.root_metrics(candidate,REGISTERED)
    source_support=topology.support_state(source,TOL); support=topology.support_state(candidate,TOL)
    source_reverse=reverse_counts(source); reverse=reverse_counts(candidate); cdf=hypsometry_cdf(source,candidate)
    checks={'response_source_is_complete_50y_volume_only_failure':True,
            'derived_scale_below_3p5pct_cap':scale<=1.035,
            'volume_scale_exact':abs(volume/source_volume-scale)<=1e-12,
            'area_change_within_0p5pct':abs(area/source_area-1.0)<=0.005,
            'support_unchanged':support==source_support,
            'connected_length_unchanged':root['length_m']==source_root['length_m'],
            'free_domain_unchanged':root['free_downstream_m']==source_root['free_downstream_m'],
            'within_reverse_unchanged':reverse['within']==source_reverse['within'],
            'junction_reverse_unchanged':reverse['junction']==source_reverse['junction'],
            'hypsometry_cdf_within_2pct':cdf<=0.02,
            'finite_nonnegative':all(np.isfinite(fl.section).all() and np.all(np.asarray(fl.section)>=0) for fl in candidate)}
    summary={'schema':'lia-191-system2005-policy-v6-uniform-thickness-v3-design','status':'one-shot-v3-static-feasible' if all(checks.values()) else 'v3-static-infeasible','mechanism':'response-corrected-uniform-active-thickness-lift','post_result_adjustment_disclosed':True,'target_formal_volume_change_fraction':TARGET_DV,'v2_observed_volume_change_fraction':observed,'v2_volume_scale':old_scale,'derived_volume_scale':scale,'derived_uniform_thickness_lift_m':height,'candidate_volume_m3':volume,'candidate_area_m2':area,'area_change_fraction':area/source_area-1.0,'candidate_root':root,'candidate_reverse':reverse,'hypsometry_cdf_maximum':cdf,'checks':checks,'candidate_state_written':False,'simulation_launched':False,'historical_run':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False)); return 0 if all(checks.values()) else 1

if __name__=='__main__': raise SystemExit(main())
