"""Read-only forensic ledger analysis; no BP or transport solve, no replay."""
from pathlib import Path
import hashlib,json
import numpy as np
from polim_cpu import packed_storage_v1 as st
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter

PROJECT=Path(__file__).resolve().parents[2]
RESULTS=PROJECT/'validation_data/results/paleo_surface_validation_v0'
SOURCE=PROJECT/'polim_cpu/runs/v32r4-historical-1470-1570-v1'
OLD=PROJECT/'polim_cpu/runs/v38-large-step-phase2-v1'
NEW=PROJECT/'polim_cpu/runs/v38r1-large-step-phase2-v1'
P1=PROJECT/'polim_cpu/runs/v37r1-large-step-phase1-v1/system2387-1482-quarter'
TRACKED={}


def track(p):
    p=Path(p);TRACKED[p.relative_to(PROJECT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();return p


def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def arrays(p):return st.arrays(track(p))
def restore(p):
    for n in ('state.npz','metadata.json','seal.json'):
        if (p/n).exists():track(p/n)
    contract=read(p.parent/'contract.json')['contract']
    return st.restore(p,contract)


def ref(year):
    return restore(SOURCE/f"frame-{read(SOURCE/f'year-{year}.json')['step']:06d}")


def provenance():
    scoped=['large_step_transport_v1.py','large_step_coupling_v1.py','large_step_gpu_bridge_v1.py',
        'large_step_phase2_worker_v1.py','large_step_phase2_worker_v2.py','large_step_phase2_service_v1.py',
        'large_step_phase2_solve_v1.py','large_step_phase2_report_v1.py','large_step_phase2_common_v1.py',
        'large_step_transport_service_v1.py','large_step_transport_client_v1.py',
        'diffusion_predictor_v2.py','diffusion_predictor_service_v2.py','predictor_repair_worker_v1.py',
        'gpu_allocator_safe_v1.py','gpu_chunked_v1.py','gpu_bp_v1.py','projected_amg_v1.py','v13r1/bp.py']
    paths=['polim_cpu/'+n for n in scoped]
    records=[]
    for branch in ['polim_v38_large_step_phase2_v1','polim_v38r1_large_step_phase2_v1','polim_v40_predictor_repair_v1']:
        plan=read(RESULTS/branch/'plan.frozen.json');checks=[]
        for p in paths:
            expected=plan.get('code_hashes',{}).get(p)
            if expected:
                actual=hashlib.sha256(track(PROJECT/p).read_bytes()).hexdigest()
                checks.append(dict(path=p,expected=expected,actual=actual,match=expected.lower()==actual.lower()))
        records.append(dict(branch=branch,checks=checks,all_scoped_hashes_match=all(x['match'] for x in checks)))
        read(RESULTS/branch/'completion.json')
    for p in ['polim_v38r1_large_step_phase2_v1/ledger-audit.json','polim_v38r1_large_step_phase2_v1/failure-review.json',
              'polim_v39_transport_budget_audit_v1/summary.json','polim_v39_transport_budget_audit_v1/plan.frozen.json']:
        read(RESULTS/p)
    return records


def rejection_audit():
    rows=[]
    for base in (OLD,NEW):
        for p in sorted(base.glob('*/attempt-*.json')):
            d=read(p)
            if d['status']=='accepted':continue
            last=d.get('history',[{}])[-1];fail=[]
            for key,gate in [('bp',1e-6),('thickness_equation_relative',1e-8),('mass_relative',1e-10),('fixed_point_relative',1e-8),('cumulative_mass_relative',1e-10)]:
                if key in last and last[key]>gate:fail.append(key)
            if last.get('active_set_equal') is False:fail.append('active_set_equal')
            rows.append(dict(case=p.parent.name,window=[int(d['start']),int(d['start'])+2] if d['start'] in (1482,1552) else [1552,1554],
                start=d['start'],dt=d['dt'],bp_calls=d['bp_calls'],last=last,
                failed_gates=fail,termination=d.get('error'),category='outer-budget-exhausted' if 'eight complete' in d.get('error','') else 'inspect-explicit-error',
                source=p.relative_to(PROJECT).as_posix()))
    review=read(RESULTS/'polim_v38r1_large_step_phase2_v1/failure-review.json')
    for c in review['cases']:
        rows.append(dict(case=c['case'],window=[1552,1554],start=c['last_recorded_year'],
            dt=c['failed_candidate_year']-c['last_recorded_year'],bp_calls=None,bp_calls_bounds=c['missing_BP_calls_bounds'],
            last=None,last_available_previous_step_cumulative=c['last_cumulative_mass'],
            failed_gates=['cumulative_mass_relative'],category='post-commit-cumulative-mass-assertion',
            termination='AssertionError; failing candidate arrays and outer telemetry not saved',
            source='validation_data/results/paleo_surface_validation_v0/polim_v38r1_large_step_phase2_v1/failure-review.json'))
    return rows


def budget_audit():
    z=arrays(SOURCE/'input.npz');adapter=ExistingMeshFluxAdapter(z)
    area=z['geo_dual_area'];edges=z['geo_edges'];boundary=adapter.topology.boundary_donors;n=len(area)
    nodes=np.array([10190,680,681,732,733]);touch=np.flatnonzero(np.any(np.isin(edges,nodes),axis=1))
    def div(q):
        a=np.zeros(n);np.add.at(a,edges[:,0],q);np.add.at(a,edges[:,1],-q);return a
    def acc(start):return dict(initial=ref(start)[0]['h'],potential=np.zeros(n),unmet=np.zeros(n),net=np.zeros(n),
                              boundary=np.zeros(n),residual=np.zeros(n),endpoint=None,steps=[],signs=[],annual_boundary={},annual_nodes={})
    def add(b,t0,t1,h0,h1,pot,unmet,q,bo,res,**extra):
        b['potential']+=pot;b['unmet']+=unmet;b['net']+=div(q)
        bv=np.bincount(boundary,weights=bo,minlength=n);b['boundary']+=bv;b['residual']+=res;b['endpoint']=h1
        b['annual_boundary'].setdefault(int(np.floor(t0+1e-8)),np.zeros(len(boundary)))[:] += bo
        b['steps'].append(dict(start=t0,end=t1,H_before=h0[nodes].tolist(),H_after=h1[nodes].tolist(),
            unmet_m3=unmet[nodes].tolist(),**extra))
        b['signs'].append(np.sign(q[touch]).astype(int).tolist())
        if t1==int(t1):b['annual_nodes'][int(t1)]={k:b[k][nodes].copy() for k in ['potential','unmet','net','boundary','residual']}
    def baseline(start):
        b=acc(start);old,om=ref(start);_,end=ref(start+2)
        for step in range(om['step']+1,end['step']+1):
            current,m=restore(SOURCE/f'frame-{step:06d}');dt=m['year']-om['year'];year=int(np.floor(om['year']+1e-8))
            smb=arrays(SOURCE/f'forcing-{year}.npz')['smb'];forced=np.maximum(old['h']+dt*smb,0.)
            unmet=area*(forced-old['h']-dt*smb);rates=adapter.rates(old['u']);donor=rates.donor_receiver_rate()[0]
            q=dt*rates.internal_signed*forced[donor];bo=dt*rates.boundary_outward*forced[boundary]
            residual=area*(current['h']-old['h'])-dt*area*smb-unmet+div(q)+np.bincount(boundary,weights=bo,minlength=n)
            outr=np.bincount(donor,weights=abs(rates.internal_signed),minlength=n)+np.bincount(boundary,weights=rates.boundary_outward,minlength=n)
            cfl=float(np.max((dt*outr/area)[forced>0],initial=0.));err=float(np.max(abs(residual)/area))
            exporterr=abs(float(bo.sum())-(m['exported']-om['exported']))
            if err>1e-8 or exporterr>1e-5 or cfl>.35+1e-10:
                raise RuntimeError(f'Cannot identify baseline single-substep flux: year {m["year"]}; no dynamics replay permitted')
            add(b,om['year'],m['year'],old['h'],current['h'],dt*area*smb,unmet,q,bo,residual,
                pretransport_H=forced[nodes].tolist(),endpoint_error_m=err,CFL=cfl,export_error_m3=exporterr)
            old,om=current,m
        return b
    def candidate(case):
        start=int(case[:4]);b=acc(start);h=b['initial'].copy()
        if case=='1482-quarter':
            state,meta=restore(P1/'frame-000001');smb=arrays(SOURCE/'forcing-1482.npz')['smb'];dt=.25
            rates=adapter.rates(state['u']);q=dt*rates.internal_signed*state['h'][rates.donor_receiver_rate()[0]]
            bo=dt*rates.boundary_outward*state['h'][boundary]
            raw=area*(state['h']-h)-dt*area*smb+div(q)+np.bincount(boundary,weights=bo,minlength=n)
            unmet=np.maximum(raw,0.)
            audit=read(RESULTS/'polim_v38_large_step_phase2_v1/phase1-audit.json')
            # Reuse its actual-SMB hash and total; only new per-outlet algebra here.
            assert abs(bo.sum()-audit['boundary_export_volume'])<1e-6
            add(b,1482.,1482.25,h,state['h'],dt*area*smb,unmet,q,bo,raw-unmet,prefix='existing phase1 final arrays')
            h=state['h']
        for root,limit in ((OLD/case,1553. if start==1552 else 1484.),(NEW/case,1554.)):
            if not root.exists():continue
            for p in sorted(root.glob('attempt-*.json')):
                row=read(p)
                if row['status']!='accepted' or row['end']>limit:continue
                lp=root/row['ledger'];a=arrays(lp);assert TRACKED[lp.relative_to(PROJECT).as_posix()]==row['ledger_sha256'].lower()
                dt=row['dt'];q=dt*a['internal_flux'];bo=dt*a['boundary_flux']
                pot=dt*area*a['potential_smb'];unmet=a['unmet_ablation_volume'];res=a['local_residual']
                h1=h+(pot+unmet-div(q)-np.bincount(boundary,weights=bo,minlength=n)+res)/area
                check=None
                if row['end']==int(row['end']):
                    saved,_=restore(root/f"frame-{int(row['end']):06d}");check=float(abs(h1-saved['h']).max());assert check<1e-8
                add(b,row['start'],row['end'],h,h1,pot,unmet,q,bo,res,annual_reconstruction_error_m=check,
                    transport_active=a['transport_active'][nodes].astype(bool).tolist())
                h=h1
        return b
    old1552=baseline(1552);new1552=candidate('1552-quarter');local=[]
    for j,node in enumerate(nodes):
        terms={k:float(new1552[k][node]-old1552[k][node]) for k in ['potential','unmet','net','boundary','residual']}
        lhs=float(area[node]*(new1552['endpoint'][node]-old1552['endpoint'][node]));rhs=terms['potential']+terms['unmet']-terms['net']-terms['boundary']+terms['residual']
        local.append(dict(node=int(node),area_m2=float(area[node]),thickness_difference_m=lhs/area[node],
                          delta_volume_m3=lhs,**terms,identity_error_m3=lhs-rhs,
                          interpretation='budget attribution only; no causal separation of timestep, velocity timing, source ordering'))
    events={}
    for label,b in [('v32',old1552),('v38_retained',new1552)]:
        signs=np.array(b['signs']);nodeevents=[]
        for j,node in enumerate(nodes):
            hvals=np.array([r['H_before'][j] for r in b['steps']]+[b['steps'][-1]['H_after'][j]])
            pre=np.array([r.get('pretransport_H',r['H_before'])[j] for r in b['steps']])
            nodeevents.append(dict(node=int(node),minimum_saved_or_algebraic_H=float(hvals.min()),
                endpoint_wet_dry_transitions=int(np.count_nonzero((hvals[1:]>1e-10)!=(hvals[:-1]>1e-10))),
                source_stage_zero_count=int(np.count_nonzero(pre<=1e-10)) if label=='v32' else None,
                limited_steps=int(sum(r['unmet_m3'][j]>1e-6 for r in b['steps']))))
        edgeevents=[]
        for k,ei in enumerate(touch):
            seq=signs[:,k];nz=seq[seq!=0]
            edgeevents.append(dict(edge=int(ei),nodes=edges[ei].tolist(),positive_steps=int(sum(seq>0)),negative_steps=int(sum(seq<0)),
                zero_steps=int(sum(seq==0)),nonzero_direction_switches=int(sum(nz[1:]!=nz[:-1]))))
        events[label]=dict(nodes=nodeevents,edges=edgeevents,steps=b['steps'])
    old1482=baseline(1482);outlets=[]
    nb=len(z['geo_outer_a'])
    for case in ['1482-quarter','1482-half','1482-annual']:
        b=candidate(case);annual=[]
        for year in [1482,1483]:
            refbo=old1482['annual_boundary'][year];bo=b['annual_boundary'][year];delta=bo-refbo
            # Two quadrature donor pieces per physical boundary edge.
            grouped=delta[:nb]+delta[nb:];order=np.argsort(-abs(grouped))[:12]
            annual.append(dict(start=year,end=year+1,reference_m3=float(refbo.sum()),candidate_m3=float(bo.sum()),
                difference_m3=float(delta.sum()),top_edges=[dict(edge=int(e),nodes=[int(z['geo_outer_a'][e]),int(z['geo_outer_b'][e])],
                    xy=z['xy'][[z['geo_outer_a'][e],z['geo_outer_b'][e]]].tolist(),delta_m3=float(grouped[e]),
                    reference_m3=float(refbo[e]+refbo[nb+e]),candidate_m3=float(bo[e]+bo[nb+e])) for e in order],
                remainder_outside_top12_m3=float(grouped.sum()-grouped[order].sum())))
        outlets.append(dict(case=case,years=annual,total_difference_m3=sum(x['difference_m3'] for x in annual)))
    return dict(local1554=local,events1552=events,outlets1482=outlets,
        baseline_step_counts=dict(window1552=len(old1552['steps']),window1482=len(old1482['steps'])),
        method='Independent evaluation on saved states or recorded face fluxes; no new BP/implicit transport solve or trajectory replay. Candidate H reconstructed from recorded balance, checked at sealed annual endpoints.',
        direction_limit='Signs of transported volumes, not unrecorded candidate velocities on zero-donor faces. Zero flux does not determine donor direction. Within-step switches unobservable.',
        representative1552='quarter cap retained 1552-1553 plus repaired 1553-1554; other caps annual errors retained in original completion')


def run():
    p=provenance();r=rejection_audit();b=budget_audit()
    return dict(provenance=p,rejections=r,budgets=b,input_hashes=TRACKED.copy(),
                source_hashes_pass=all(x['all_scoped_hashes_match'] for x in p))
