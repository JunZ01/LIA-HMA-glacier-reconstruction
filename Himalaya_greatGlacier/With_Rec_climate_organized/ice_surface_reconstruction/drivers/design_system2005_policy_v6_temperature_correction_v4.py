"""Select one warmer temperature for the already static-passing v2 state."""

from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]; R=ROOT/'validation_data/results/paleo_surface_validation_v0'
V2=R/'lia_191_system2005_policy_v6_joint_candidate_v2/summary.json'; V3=R/'lia_191_system2005_policy_v6_joint_candidate_v3/summary.json'
COLD=R/'lia_191_network_topology_adaptive_split_system2005_steady_50y_coldm0p125_6hour_v1/summary.json'; WARM=R/'lia_191_network_topology_adaptive_split_system2005_steady_50y_6hour_v1/summary.json'
STATE=R/'lia_191_system2005_policy_v6_joint_candidate_v2/joint_candidate_initial_state.pkl.gz'; OUT=R/'lia_191_system2005_policy_v6_temperature_correction_v4_design'
def sha(p):
 d=hashlib.sha256()
 with Path(p).open('rb') as h:
  for b in iter(lambda:h.read(1048576),b''): d.update(b)
 return d.hexdigest().upper()
def main():
 OUT.mkdir(parents=True,exist_ok=False)
 v2=json.loads(V2.read_text(encoding='utf-8'));v3=json.loads(V3.read_text(encoding='utf-8'));cold=json.loads(COLD.read_text(encoding='utf-8'));warm=json.loads(WARM.read_text(encoding='utf-8'))
 if v2.get('status')!='formal-50y-fail-mechanism-closed' or v3.get('status')!='formal-50y-fail-mechanism-closed': raise RuntimeError('required full responses missing')
 if not STATE.exists(): raise FileNotFoundError(STATE)
 t0=-0.25;selected=-0.21875;next_warm=-0.1875
 volume_slope=(float(warm['volume_change_fraction'])-float(cold['volume_change_fraction']))/(float(warm['stage2']['delta_t_c'])-float(cold['stage2']['delta_t_c']))
 length_slope=(float(warm['root']['length_m'])-float(cold['root']['length_m']))/(float(warm['stage2']['delta_t_c'])-float(cold['stage2']['delta_t_c']))
 dv0=float(v2['formal']['volume_change_fraction']);l0=float(v2['formal']['final_length_m'])
 pred_dv=dv0+volume_slope*(selected-t0);pred_l=l0+length_slope*(selected-t0)
 pred_next_dv=dv0+volume_slope*(next_warm-t0);pred_next_l=l0+length_slope*(next_warm-t0)
 checks={'v2_complete_volume_only_failure':v2['formal']['checks']['volume_change'] is False and v2['formal']['checks']['connected_length'] is True,
         'v3_complete_and_uniform_lift_family_closed':v3['formal']['checks']['volume_change'] is False and v3['formal']['checks']['connected_length'] is True,
         'selected_half_grid':abs(selected/0.03125-round(selected/0.03125))<1e-12,
         'selected_predicted_volume_inside_2pct':abs(pred_dv)<=0.02,
         'selected_predicted_length_above_continuous_gate':pred_l>=25450.125918037134,
         'next_warmer_predicted_length_below_continuous_gate':pred_next_l<25450.125918037134,
         'initial_state_exists':True}
 summary={'schema':'lia-191-system2005-policy-v6-temperature-correction-v4-design','status':'unique-temperature-candidate-selected' if all(checks.values()) else 'design-fail','system_id':2005,'candidate':{'initial_state':str(STATE.relative_to(ROOT)).replace('\\','/'),'initial_state_sha256':sha(STATE),'delta_t_c':selected,'state_modified':False,'other_stage2_modified':False},'response':{'source_delta_t_c':t0,'source_volume_change_fraction':dv0,'source_final_length_m':l0,'volume_change_slope_per_k':volume_slope,'length_slope_m_per_k':length_slope,'predicted_volume_change_fraction':pred_dv,'predicted_final_length_m':pred_l,'next_warmer_delta_t_c':next_warm,'next_warmer_predicted_volume_change_fraction':pred_next_dv,'next_warmer_predicted_final_length_m':pred_next_l,'prediction_is_design_not_acceptance':True},'checks':checks,'stages':{'preflight_years':10,'formal_years':50,'candidate_count_max':1,'historical':False},'simulation_launched':False}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
