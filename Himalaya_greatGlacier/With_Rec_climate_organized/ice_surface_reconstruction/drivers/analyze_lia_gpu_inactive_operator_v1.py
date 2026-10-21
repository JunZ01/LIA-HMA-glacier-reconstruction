"""CPU-only algebra tests on immutable states, never a glacier trajectory."""

import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import solve_banded

from lia_gpu_four_group_search_v2 import ROOT, Candidate, load_search, file_hash
from lia_gpu_matrix_io_v2 import Store, atomic_json
from prepare_lia_gpu_four_group_v2 import OUTPUT


def active_prefix(h, mb):
    active = np.flatnonzero((h > 0) | (mb > 0))
    return min(len(h), max(3, int(active[-1]) + 2 if len(active) else 3))


def assemble(h, bed, width, dx, glen, mb, dt):
    n = len(h)
    hs = .5 * (h[:-1] + h[1:])
    ws = .5 * (width[:-1] + width[1:])
    slope = np.diff(bed+h)/dx
    diffusivity = np.zeros(n+1)
    diffusivity[1:-1] = .4*glen*hs**5*(900*9.80665)**3*ws*np.abs(slope)**2
    lo = -dt/dx**2 * diffusivity[:-1]/width
    up = -dt/dx**2 * diffusivity[1:]/width
    diag = 1+dt/dx**2 * (diffusivity[:-1]+diffusivity[1:])/width
    bed_gradient = np.diff(np.r_[bed[0],bed,bed[-1]])/dx
    correction = -diffusivity*bed_gradient
    rhs = h+mb*dt+dt/width*(correction[:-1]-correction[1:])/dx
    return lo,diag,up,rhs


def solve(coefficients):
    lo,diag,up,rhs = coefficients
    band = np.zeros((3,len(rhs)))
    band[0,1:] = up[:-1]
    band[1] = diag
    band[2,:-1] = lo[1:]
    return solve_banded((1,1),band,rhs,check_finite=True)


def pcr_numpy(coefficients):
    # Algebraic replay of the existing PCR sequence, not a CUDA timing proxy.
    a,b,c,d = [x.copy() for x in coefficients]
    n = len(d)
    idx = np.arange(n)
    stride = 1
    while stride < n:
        left,right = np.maximum(idx-stride,0),np.minimum(idx+stride,n-1)
        alpha = np.where(idx>=stride,-a/b[left],0)
        beta = np.where(idx+stride<n,-c/b[right],0)
        a,b,c,d = (np.where(idx>=stride,alpha*a[left],0),
                   b+alpha*c[left]+beta*a[right],
                   np.where(idx+stride<n,beta*c[right],0),
                   d+alpha*d[left]+beta*d[right])
        stride *= 2
    return d/b


def annual_mb(data, i, base, candidate, h, phase):
    n = len(h)
    temp = data['system_climate_temp'][i,phase,:,None] + float(base['base_temp_bias_degC']) + candidate.delta
    gradient = data['system_climate_gradient'][i,phase,:,None]
    temp = temp+gradient*(data['system_bed_h'][i,:n]+h-data['system_climate_ref_hgt'][i])[None,:]
    prcp = data['system_climate_prcp'][i,phase,:,None]*float(base['prcp_fac'])*candidate.prcp
    solid = prcp*np.clip(1-temp/2,0,1)
    melt = float(base['melt_f'])*candidate.melt*(365/12)*np.maximum(temp-float(base['temp_melt_degC']),0)
    return (solid-melt).sum(axis=0)/31536000/900


def metrics(h,width,dx):
    return np.array([(h>0).sum()*dx,width[h>0].sum()*dx,(h*width).sum()*dx])


def check_case(data,index,base,candidate,h,phase,dt,with_pcr=False):
    i = index[candidate.system]
    n = len(h)
    bed,width = data['system_bed_h'][i,:n],data['system_widths_m'][i,:n]
    dx = float(data['system_dx_meter'][i])
    mb = annual_mb(data,i,base,candidate,h,phase)
    coeff = assemble(h,bed,width,dx,float(base['glen_a_pa_3_s_1'])*candidate.glen,mb,dt)
    cut = active_prefix(h,mb)
    lo,diag,up,rhs = coeff
    decoupled = bool(cut==n or (up[cut-1]==0 and lo[cut]==0 and
        np.all(diag[cut:]==1) and np.all(lo[cut:]==0) and np.all(up[cut:]==0) and np.all(rhs[cut:]<=0)))
    full_linear = solve(coeff)
    full = np.maximum(full_linear,0)
    reduced = np.zeros(n)
    reduced[:cut] = np.maximum(solve(tuple(x[:cut] for x in coeff)),0)
    m1,m2 = metrics(full,width,dx),metrics(reduced,width,dx)
    rel = np.abs(m1-m2)/np.maximum(np.maximum(np.abs(m1),np.abs(m2)),1e-30)
    error = float(np.max(np.abs(full-reduced)))
    result = dict(system=candidate.system,delta=candidate.delta,phase=phase,dt_seconds=dt,
        full_nodes=n,solve_nodes=cut,node_fraction=cut/n,decoupled=decoupled,
        max_thickness_error_m=error,length_error_m=float(abs(m1[0]-m2[0])),
        area_relative_error=float(rel[1]),volume_relative_error=float(rel[2]),
        diagonal_dominance_margin=float(np.min(diag-np.abs(lo)-np.abs(up))),
        would_lose_positive_mb_if_h_only=bool(np.any(mb[active_prefix(h,np.zeros(n)):] > 0)),
        passed=bool(decoupled and error<=1e-6 and m1[0]==m2[0] and rel[1]<=1e-10 and rel[2]<=1e-8))
    if with_pcr:
        pcr = pcr_numpy(coeff)
        pcr_short = np.zeros(n)
        pcr_short[:cut] = np.maximum(pcr_numpy(tuple(x[:cut] for x in coeff)),0)
        result['numpy_pcr_prefix_error_m'] = float(np.max(np.abs(np.maximum(pcr,0)-pcr_short)))
        result['numpy_pcr_reference_max_error_m'] = float(np.max(np.abs(np.maximum(pcr,0)-full)))
        residual = diag*pcr-rhs
        residual[1:] += lo[1:]*pcr[:-1]
        residual[:-1] += up[:-1]*pcr[1:]
        scale = np.max(np.abs(diag)+np.abs(lo)+np.abs(up))*np.max(np.abs(pcr))+np.max(np.abs(rhs))
        result['numpy_pcr_backward_error'] = float(np.max(np.abs(residual))/max(scale,1e-30))
    return result,coeff


def main():
    started = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT/'validation_data/results/paleo_surface_validation_v0'/('lia_gpu_operator_offline_'+stamp)
    out.mkdir()
    shared = file_hash(ROOT/'LIA_CURRENT_STATE.json')
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    search = load_search(manifest['contract'])
    index = {row['system']:i for i,row in enumerate(manifest['systems'])}
    needed = ('system_nx','system_dx_meter','system_thick','system_bed_h','system_widths_m',
              'system_climate_temp','system_climate_prcp','system_climate_gradient','system_climate_ref_hgt')
    with np.load(OUTPUT/'inputs.npz',allow_pickle=False) as blob:
        data = {k:blob[k] for k in needed}
    rows = []
    for system,i in index.items():
        h = data['system_thick'][i,:data['system_nx'][i]]
        for delta in (-1.,0.,1.):
            c = Candidate(system,delta=delta)
            for phase in (0,15,29):
                for dt in (108000.,36000.):
                    result,_ = check_case(data,index,search.baselines[system],c,h,phase,dt)
                    result['source']='initial-immutable-geometry'
                    rows.append(result)
    print(json.dumps(dict(event='initial-algebra-complete',cases=len(rows),passed=sum(r['passed'] for r in rows))),flush=True)
    store = Store(ROOT/'wg2g1')
    first = search.pack(search.candidates(1,[]),{r['system']:r['nx'] for r in manifest['systems']})[0]
    chosen = [first[int(i)] for i in np.linspace(0,len(first)-1,24)]
    fixture,source_rows = {},[]
    for j,c in enumerate(chosen):
        key = search.spin_key(c)
        record = store.read('spin',key)
        if record:
            arrays=store.arrays('spin',key)
            source = store.directory('spin',key)/'result.json'
        else:
            record,arrays = store.latest('spin',key)
            source = ROOT/'wg2g1/resume/spin'/key[:16]/('y%04d'%record['year'])/'checkpoint'/key[:16]/'result.json'
        source_rows.append(dict(candidate=asdict(c),year=record['year'],source=str(source),sha256=file_hash(source)))
        for phase in (0,15,29):
            for dt in (108000.,36000.):
                result,coeff = check_case(data,index,search.baselines[c.system],c,arrays['thick'],phase,dt,True)
                result.update(source='saved-production-state',source_year=record['year'])
                rows.append(result)
                if phase==0 and dt==108000:
                    for name,value in zip(('lower','diagonal','upper','rhs'),coeff):
                        fixture['case%02d_%s'%(j,name)]=value
                    fixture['case%02d_cut'%j]=np.array(result['solve_nodes'])
    with (out/'fixed_equations.npz').open('xb') as stream:
        np.savez_compressed(stream,**fixture)
    fractions=np.array([r['node_fraction'] for r in rows])
    pcr=[r for r in rows if 'numpy_pcr_reference_max_error_m' in r]
    summary=dict(status='offline-algebra-pass' if all(r['passed'] for r in rows) else 'offline-algebra-fail',
        cases=len(rows),passed=sum(r['passed'] for r in rows),initial_cases=3438,saved_state_cases=len(rows)-3438,
        no_torch_or_cuda_import=True,no_trajectory_advanced=True,production_unchanged=True,
        source_manifest_sha256=file_hash(OUTPUT/'manifest.json'),contract=manifest['contract'],
        solve_node_fraction=dict(min=float(fractions.min()),median=float(np.median(fractions)),max=float(fractions.max())),
        median_solve_nodes=float(np.median([r['solve_nodes'] for r in rows])),
        thickness_error_max=max(r['max_thickness_error_m'] for r in rows),
        volume_relative_error_max=max(r['volume_relative_error'] for r in rows),
        min_diagonal_dominance_margin=min(r['diagonal_dominance_margin'] for r in rows),
        positive_mb_missed_by_thickness_only=sum(r['would_lose_positive_mb_if_h_only'] for r in rows),
        numpy_pcr_reference_error_max=max(r['numpy_pcr_reference_max_error_m'] for r in pcr),
        numpy_pcr_prefix_error_max=max(r['numpy_pcr_prefix_error_m'] for r in pcr),
        numpy_pcr_backward_error_max=max(r['numpy_pcr_backward_error'] for r in pcr),
        fixed_equations_sha256=file_hash(out/'fixed_equations.npz'),
        source_states=source_rows,wall_seconds=time.perf_counter()-started,
        shared_state_unchanged=file_hash(ROOT/'LIA_CURRENT_STATE.json')==shared,
        limitations=['One-operator algebra only, not long-run equivalence or measured GPU speedup.',
          'Recompute active prefix every step and positive MB on the full original domain at each annual update.',
          'Keep one downstream zero-thickness neighbor and validate zero coupling before omitting tail solves.',
          'Preserve full-grid climatic mass-balance and residual sums, geometry, diagnostic domain and full-state checkpoints.',
          'CUDA PCR versus CPU arithmetic differs; saved CPU fixture does not qualify a GPU replacement.',
          'No vendor GPU solver called. Existing production scope and implementation hashes unchanged.'])
    atomic_json(out/'cases.json',rows)
    atomic_json(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_states'},indent=2),flush=True)
    print(str(out),flush=True)


if __name__=='__main__':
    main()
