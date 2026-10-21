"""Resolve disclosed old-climate matches and truncated static-only files."""
from . import lia_cpu_melt_factor_v10_279 as r
from pathlib import Path
def main():
    src=r.TX/'legacy_cache_reconciliation.json';a=r.read(src);resolved=[]
    assert not a['exact_physics_matches']
    for x in a['possible_partial_matches']:
        r.check(x['legacy_source']);o=r.read(r.ROOT/x['legacy_source']['path'])
        assert o['spinup_window']==[1807,1836] and o['spinup_years']==500 and o['start_year']==1836
        resolved.append({**x,'resolution':'not-equivalent: old forcing1807-1836/full500/forward1836-2000 versus new1441-1470/full900/forward1470-2000. No cached state can be relabelled as the new equilibrium.'})
    excluded=[]
    for x in a['unreadable']:
        p=r.ROOT/x['file'];text=p.read_text(encoding='utf-8-sig')
        assert 'lia_single_multi_initial_state_static_v2/outputs/' in p.as_posix()
        assert 'completed-static-lossless-pair-not-a-dynamical-restart' in text and '"model_years": 0' in text
        assert '"parameters"' not in text and '"dynamical_initial_state_ready": false' in text
        excluded.append({'file':r.ref(p),'resolution':'truncated existing static geometry report, explicitly zero model years and not a dynamical restart; not a completed parameter-run cache. Preserved without repair.'})
    assert len(resolved)==4 and len(excluded)==3
    out={'status':'pass','prior_audit':r.ref(src),'resolved_different_climate_partial_matches':resolved,'excluded_non_dynamic_truncated_reports':excluded,'remaining_exact_or_unresolved_duplicates':0,'closed_configurations':'Named old B-window scalar closures retain their original historical b+delta forcing. Current approved physical-axis experiment retains only spinup delta, fixed historical b and q0, full900 and all stable parents. Old closed bracket reproduction is not authorized.','source_old_forward_implementation':r.ref(r.DRIVERS/'lia_adaptive_spinup_chain.py'),'resolution_code':r.ref(Path(__file__)),'new_model_steps':0}
    r.write(r.TX/'legacy_cache_resolution.json',out);print({'status':'pass','different_climate_matches':4,'static_reports_excluded':3},flush=True)
if __name__=='__main__':main()
