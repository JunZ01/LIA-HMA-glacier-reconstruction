"""Read-only v10.275 terminal audit, frozen CPU curve scoring, and figures.

No OGGM/Torch import, model creation, parameter change or acceptance promotion.
Data analysis uses the bundled Python; the separate plot command uses the
existing matplotlib environment because matplotlib is absent from the bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
from score_lia_all191_forward_essential_standard_v3 import score_one

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'validation_data/results/paleo_surface_validation_v0'
TX = BASE / 'lia_cpu_fixed_climate_forward_v10_275_transaction_v1'
OUT = BASE / 'lia_cpu_fixed_climate_forward_analysis_v10_276_v1'
TARGET = BASE / 'lia_191_controller_v1/core191_length_references.csv'
CPU_SCORER = Path(__file__).with_name('score_lia_all191_forward_essential_standard_v3.py')
DIRECTION_PRECEDENT = Path(__file__).with_name('lia_gpu_matrix_scoring_v2.py')
CP_YEARS = list(range(1470, 1981, 30)) + [2000]
METRICS = ['length_m', 'area_m2', 'volume_m3']


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest().upper()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(clean(obj), f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def ref(path):
    path = Path(path)
    return {'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(path)}


def verify(path, expected):
    actual = sha(path)
    if actual != expected.upper():
        raise RuntimeError(f'hash mismatch: {path}')
    return actual


def csv(path, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        pd.DataFrame(rows).to_csv(f, index=False)


def audit_score():
    if OUT.exists():
        raise RuntimeError(f'Output already exists; no overwrite: {OUT}')
    OUT.mkdir()
    inputs = [TX / x for x in ('transaction_manifest.json', 'matrix.json',
              'population_191.json', 'curve_standard.json')]
    inputs += [TARGET, CPU_SCORER, DIRECTION_PRECEDENT, Path(__file__), ROOT / 'w275/r/summary.json']
    write(OUT / 'analysis_transaction.json', {
        'schema': 'lia-cpu-postprocess-v10.276', 'created_at': now(),
        'authorization': 'User: 开始做吧; analyze all completed fixed-climate forward curves.',
        'model_launch_authorized': False, 'accepted_ledger_effect': 0,
        'inputs': [ref(p) for p in inputs],
        'operators': {
            'peak': 'raw maximum length over all 1470..2000',
            'endpoint': 'raw length at 2000',
            'shape': 'unchanged CPU v3 functions, 1500..2000 with 1499 volume/area for B',
            'direction': 'sign(L2000-L1470) == sign(RGI-LIA), required if absolute target separation >=300 m',
            'amplitude': 'max(L1470..2000)-min(L1470..2000) >=0.5*abs(RGI-LIA)',
            'direction_amplitude_provenance': 'v10.260 frozen rules; same explicit arithmetic already implemented in lia_gpu_matrix_scoring_v2.py. No GPU model or GPU results used.',
            'initial_length': 'reported only; never an eligibility gate',
            'first30_first60': 'descriptive changes only; no new exclusion threshold',
            'sensitivity': 'within-system max-minus-min across ALL saved stable parents under identical forward climate and physics; this diagnoses initial-state dependence, not independent validation of climate',
            'zero_ice': 'retain as disappearance outcome; log-length shape not evaluable, not a technical failure',
            'formal_acceptance': 'not performed; independent reproduction still required'},
        'population': 191, 'members': 1393, 'forward_systems': 173})
    matrix = read(TX / 'matrix.json')
    population = read(TX / 'population_191.json')
    targets = pd.read_csv(TARGET).set_index('lia_unique_id')
    manifest = read(TX / 'transaction_manifest.json')
    def check_refs(x):
        if isinstance(x, dict):
            if 'path' in x and 'sha256' in x:
                verify(ROOT / x['path'], x['sha256'])
            for v in x.values():
                check_refs(v)
        elif isinstance(x, list):
            for v in x:
                check_refs(v)
    check_refs(manifest)
    terminal = read(ROOT / 'w275/r/summary.json')
    records = {x['key']: x for x in terminal['records']}
    assert len(matrix) == len(records) == 1393
    assert len({r['key'] for r in matrix}) == len({r['fingerprint'] for r in matrix}) == 1393
    assert sorted({r['lia_unique_id'] for r in matrix}) == population['forward_ids']
    assert set(population['forward_ids']).isdisjoint(population['no_stable_ids'])
    assert sorted(population['forward_ids'] + population['no_stable_ids']) == population['all_ids']
    rows, audits, histories, warning_counts = [], [], {}, {'margin': 0, 'disappearance': 0}
    for i, m in enumerate(matrix):
        key, sid = m['key'], int(m['lia_unique_id'])
        record = records[key]
        assert record['status'] == 'completed' and record['returncode'] == 0 and not record['technical_failure']
        rp = ROOT / record['result']['path']
        verify(rp, record['result']['sha256'])
        result = read(rp)
        assert result['fingerprint'] == m['fingerprint'] and result['lia_unique_id'] == sid
        assert result['rgi_id'] == m['rgi_id'] and result['calendar_end'] == 2000
        assert result['checkpoints'] == CP_YEARS and result['checkpoint_contract_pass']
        assert result['forward_temp_bias_degC'] == m['base_temp_bias_degC'] and result['q'] == 0
        verify(ROOT / m['source_state'], m['source_state_sha256'])
        verify(ROOT / m['source_checkpoint'], m['source_checkpoint_sha256'])
        ap = ROOT / result['annual']['path']
        verify(ap, result['annual']['sha256'])
        a = pd.read_csv(ap, float_precision='round_trip')
        assert list(a.year) == list(range(1470, 2001))
        assert np.isfinite(a[METRICS].to_numpy()).all() and (a[METRICS].to_numpy() >= 0).all()
        cp_refs, topology = [], None
        for y in CP_YEARS:
            folder = rp.parent / 'cp' / str(y)
            cp = read(folder / 'c.json')
            assert cp['fingerprint'] == m['fingerprint'] and cp['lia_unique_id'] == sid and cp['rgi_id'] == m['rgi_id']
            assert cp['calendar_year'] == y and cp['history_rows'] == y-1470+1
            assert cp['forward_temp_bias_degC'] == m['base_temp_bias_degC']
            assert cp['source_state_sha256'].upper() == m['source_state_sha256'].upper()
            for p, v in cp['parameters'].items():
                assert v == m[p], (key, y, p)
            topology = topology or cp['topology']
            assert topology == cp['topology']
            verify(folder / 's.gz', cp['state_sha256'])
            verify(folder / 'h.json', cp['history_sha256'])
            for p in METRICS:
                assert cp['state'][p] == a.iloc[y-1470][p], (key,y,p)
            cp_refs.append(ref(folder / 'c.json'))
        h = read(rp.parent / 'cp/2000/h.json')
        for p in METRICS + ['year']:
            assert np.array_equal(np.array([x[p] for x in h]), a[p].to_numpy()), (key,p)
        for p in METRICS:
            assert float(m[p]) == a.iloc[0][p]
        histories[key] = a
        positive = bool((a[METRICS].to_numpy() > 0).all())
        t = targets.loc[sid]
        tl, tr = float(t.lia_registered_length_m), float(t.rgi_registered_length_m)
        L, V = a.length_m.to_numpy(), a.volume_m3.to_numpy()
        separation = abs(tl-tr)
        peak, end, amp = float(L.max()), float(L[-1]), float(np.ptp(L))
        r = {p: m[p] for p in ('key', 'lia_unique_id', 'rgi_id', 'fingerprint', 'delta_t_eq_degC',
             'base_temp_bias_degC','melt_f','prcp_fac','glen_a_pa_3_s_1','fs','temp_melt_degC')}
        r.update({'initial_length_m': L[0], 'lia_reference_m': tl, 'rgi_reference_m': tr,
                  'peak_length_m': peak, 'peak_year_first': 1470+int(np.argmax(L)),
                  'end_length_m': end, 'amplitude_m': amp,
                  'peak_error_relative': abs(peak-tl)/tl, 'end_error_relative': abs(end-tr)/tr,
                  'peak_error_signed_relative': (peak-tl)/tl, 'end_error_signed_relative': (end-tr)/tr,
                  'initial_length_error_diagnostic_relative': abs(L[0]-tl)/tl,
                  'peak_separation_error_diagnostic': abs(peak-tl)/separation if separation else None,
                  'end_separation_error_diagnostic': abs(end-tr)/separation if separation else None,
                  'lia_gate': abs(peak-tl)/tl <= .1, 'rgi_gate': abs(end-tr)/tr <= .1,
                  'direction_gate': bool(separation < 300 or np.sign(L[-1]-L[0]) == np.sign(tr-tl)),
                  'amplitude_gate': bool(amp >= .5*separation), 'shape_evaluable': positive,
                  'forward_temp_bias_degC': m['base_temp_bias_degC'], 'forward_q_degC': 0.,
                  'handoff_temperature_step_degC': -m['delta_t_eq_degC'],
                  'first30_length_change_m': L[30]-L[0], 'first60_length_change_m': L[60]-L[0],
                  'first30_volume_change_relative': (V[30]-V[0])/V[0] if V[0] else None,
                  'first60_volume_change_relative': (V[60]-V[0])/V[0] if V[0] else None,
                  'total_length_change_m': L[-1]-L[0],
                  'max_thickness_m': float(a.maximum_thickness_m.max()),
                  'end_max_thickness_m': float(a.maximum_thickness_m.iloc[-1]),
                  'margin_warning_years': len(result['domain_margin_warning_years']),
                  'ice_disappeared_years': len(result['ice_disappeared_years']),
                  'runtime_seconds': result['runtime_seconds'],
                  'integration_seconds': result['timing']['integration_seconds'],
                  'checkpoint_seconds': result['timing']['checkpoint_seconds'],
                  'source_annual': str(ap.relative_to(ROOT)), 'source_annual_sha256': sha(ap)})
        if positive and separation:
            old = score_one(pd.Series({**m,'production_member_key':key}),t,a,sha(ap),
                            {**result,'forward_fingerprint':m['fingerprint']},rp)
            for p in ('length_best_lag_years', 'length_best_lag_correlation', 'length_lag_sign_consistency',
                      'lag_at_upper_boundary', 'lag_gate_pass', 'persistent_growth_phase_count',
                      'persistent_retreat_phase_count', 'persistent_phase_count_total', 'sustained_variation_share',
                      'bidirectional_multistage_gate_pass', 'terminal_20yr_downward_variation_share',
                      'terminal_20yr_robust_drop_z', 'flag_terminal_endpoint_drop',
                      'high_frequency_residual_rms_range_ratio','high_frequency_residual_tv_share',
                      'high_frequency_residual_zero_crossings','flag_high_frequency','artifact_gate_pass'):
                r[p] = old[p]
            r['shape_gate'] = bool(old['lag_gate_pass'] and old['bidirectional_multistage_gate_pass'] and old['artifact_gate_pass'])
        else:
            r.update({'shape_evaluable':False,'shape_gate':False,'lag_gate_pass':None,
                      'bidirectional_multistage_gate_pass':None,'artifact_gate_pass':None})
        r['endpoint_pair_gate'] = bool(r['lia_gate'] and r['rgi_gate'])
        r['provisional_curve_pass'] = bool(r['endpoint_pair_gate'] and r['direction_gate'] and r['amplitude_gate'] and r['shape_gate'])
        failures = [p for p in ('lia_gate','rgi_gate','direction_gate','amplitude_gate','lag_gate_pass',
                               'bidirectional_multistage_gate_pass','artifact_gate_pass') if r[p] is not None and not r[p]]
        if not r['shape_evaluable']:
            failures.append('shape-not-evaluable')
        r['failed_checks_nonexclusive'] = ';'.join(failures) or 'none'
        r['acceptance_effect'] = 0
        warning_counts['margin'] += int(r['margin_warning_years']>0)
        warning_counts['disappearance'] += int(r['ice_disappeared_years']>0)
        rows.append(clean(r))
        audits.append({'key':key,'lia_unique_id':sid,'result':ref(rp),'annual':ref(ap),
                       'checkpoints':cp_refs,'checks':'all-source-hash-identity-parameters-topology-annual-and-checkpoint-passed'})
        if (i+1)%100 == 0:
            print(json.dumps({'audited_and_scored':i+1,'total':1393}),flush=True)
    frame = pd.DataFrame(rows)
    glacier_rows, spreads = [], []
    for sid in population['all_ids']:
        group = frame[frame.lia_unique_id == sid]
        t = targets.loc[sid]
        g = {'lia_unique_id':sid,'rgi_id':str(t.rgi_id),'stable_parent_count':len(group),
             'lia_reference_m':float(t.lia_registered_length_m),'rgi_reference_m':float(t.rgi_registered_length_m),
             'status':'no-stable-parent-retained-for-later-work' if group.empty else 'forward-completed-not-final-acceptance'}
        for p in ('lia_gate','rgi_gate','endpoint_pair_gate','direction_gate','amplitude_gate','shape_gate','provisional_curve_pass'):
            g[p+'_count'] = int(group[p].sum())
        g['provisional_delta_t_eq_degC'] = ', '.join(f'{x:+.1f}' for x in group.loc[group.provisional_curve_pass,'delta_t_eq_degC'])
        if not group.empty:
            assert group[['base_temp_bias_degC','melt_f','prcp_fac','glen_a_pa_3_s_1','fs','temp_melt_degC']].nunique().max() == 1
            g.update({'delta_t_eq_min_degC':group.delta_t_eq_degC.min(),'delta_t_eq_max_degC':group.delta_t_eq_degC.max(),
                      'best_peak_error_relative':group.peak_error_relative.min(),
                      'best_end_error_relative':group.end_error_relative.min(),
                      'max_thickness_m':group.max_thickness_m.max()})
            stack = np.stack([histories[k][METRICS].to_numpy() for k in group.key])
            span = np.ptp(stack,axis=0)
            for y in range(1470,2001):
                s = {'lia_unique_id':sid,'year':y,'parent_count':len(group)}
                for j,p in enumerate(METRICS):
                    s[p+'_spread'] = span[y-1470,j]
                    s[p+'_spread_fraction_of_initial'] = span[y-1470,j]/span[0,j] if len(group)>1 and span[0,j]>0 else None
                spreads.append(s)
            g.update({'initial_length_spread_m':span[0,0],'year1500_length_spread_m':span[30,0],
                      'year1530_length_spread_m':span[60,0],'end_length_spread_m':span[-1,0],
                      'end_length_spread_fraction_of_initial':span[-1,0]/span[0,0] if len(group)>1 and span[0,0]>0 else None,
                      'initial_volume_spread_m3':span[0,2],'end_volume_spread_m3':span[-1,2],
                      'end_volume_spread_fraction_of_initial':span[-1,2]/span[0,2] if len(group)>1 and span[0,2]>0 else None})
        glacier_rows.append(clean(g))
    csv(OUT/'cell_scores_1393.csv',rows)
    csv(OUT/'glacier_summary_191.csv',glacier_rows)
    csv(OUT/'initial_state_spread_annual.csv',spreads)
    write(OUT/'scored_cells.json',rows)
    write(OUT/'glacier_summary.json',glacier_rows)
    write(OUT/'member_integrity_audit.json',audits)
    gates = {}
    for p in ('lia_gate','rgi_gate','endpoint_pair_gate','direction_gate','amplitude_gate','lag_gate_pass',
              'bidirectional_multistage_gate_pass','artifact_gate_pass','shape_gate','provisional_curve_pass'):
        mask = frame[p].eq(True)
        gates[p] = {'cells':int(mask.sum()),'systems':int(frame.loc[mask,'lia_unique_id'].nunique())}
    gg = pd.DataFrame(glacier_rows)
    ratio = gg.end_length_spread_fraction_of_initial.dropna()
    summary = {'schema':'lia-cpu-forward-analysis-summary-v10.276','created_at':now(),
        'population':191,'completed_cells':1393,'forward_systems':173,'no_stable_systems':18,
        'no_stable_ids':population['no_stable_ids'],'technical_failures':0,'pending_cells':0,
        'annual_rows_verified':1393*531,'checkpoint_commits_verified':1393*19,
        'source_and_member_integrity':'pass','parent_pid':terminal['parent_pid'],
        'finished_at':terminal['finished_at'],'controller_summary':ref(ROOT/'w275/r/summary.json'),
        'gates':gates,'warning_cells':warning_counts,'accepted_ledger_effect':0,'new_model_runs':0,
        'first30_length_change_m_median':float(frame.first30_length_change_m.median()),
        'first60_length_change_m_median':float(frame.first60_length_change_m.median()),
        'first30_volume_change_relative_median':float(frame.first30_volume_change_relative.median()),
        'first60_volume_change_relative_median':float(frame.first60_volume_change_relative.median()),
        'initial_state_sensitivity':{'systems_with_nonzero_initial_length_spread':len(ratio),
            'median_final_over_initial_length_spread':float(ratio.median()),
            'systems_smaller_final_length_spread':int((ratio<1).sum()),
            'systems_larger_final_length_spread':int((ratio>1).sum()),
            'median_initial_length_spread_m':float(gg.initial_length_spread_m.median()),
            'median_end_length_spread_m':float(gg.end_length_spread_m.median())},
        'observation_direction':{'modeled_net_growth_cells':int((frame.total_length_change_m>0).sum()),
            'modeled_net_retreat_cells':int((frame.total_length_change_m<0).sum()),
            'modeled_zero_net_change_cells':int((frame.total_length_change_m==0).sum())},
        'limitations':['Curve passes are provisional, not newly accepted glaciers.',
            'B(t) is derived from volume changes, not an independent climate-response validation.',
            'Initial-condition spread does not isolate a climate-truth error or prove causation.',
            'All initial transient years are retained; original shape period1500..2000 is unchanged.',
            'No single glacier parameter tuning, further physics search or rerun is authorized.'],
        'outputs':[ref(OUT/p) for p in ('cell_scores_1393.csv','glacier_summary_191.csv',
                  'initial_state_spread_annual.csv','member_integrity_audit.json')]}
    write(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,
        'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    cells = pd.DataFrame(read(OUT/'scored_cells.json'))
    glaciers = read(OUT/'glacier_summary.json')
    summary = read(OUT/'summary.json')
    pdest = OUT/'figures'
    if pdest.exists():
        raise RuntimeError('No overwrite of figures')
    pdest.mkdir()
    cmap, norm = plt.get_cmap('coolwarm'), Normalize(-1,1)
    paths = []
    groups = []
    for g in glaciers:
        sid = g['lia_unique_id']
        group = cells[cells.lia_unique_id==sid]
        if group.empty:
            continue
        arrays = [(r,pd.read_csv(ROOT/r['source_annual'])) for r in group.to_dict('records')]
        groups.append((g,arrays))
        fig, axes = plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
        for r,a in arrays:
            for ax,col,scale in zip(axes,METRICS,[1000,1e6,1e9]):
                ax.plot(a.year,a[col]/scale,color=cmap(norm(r['delta_t_eq_degC'])),lw=1.1,alpha=.85)
        axes[0].axhline(g['lia_reference_m']/1000,color='#252525',ls='--',lw=1.2,label='LIA 长度参考（非逐年观测）')
        axes[0].errorbar([2000],[g['rgi_reference_m']/1000],yerr=[g['rgi_reference_m']/10000],fmt='*',color='#167642',markersize=11,capsize=4,label='RGI 2000 参考及 ±10%')
        axes[0].legend(loc='best',fontsize=9,frameon=False)
        for ax,label in zip(axes,['长度 (km)','面积 (km²)','体积 (km³)']):
            ax.set_ylabel(label)
            ax.set_xlim(1470,2006)
            ax.axvspan(1470,1500,color='#dddddd',alpha=.3)
            ax.axvline(1530,color='#888888',lw=.7,ls=':')
            ax.grid(axis='y',alpha=.15)
        axes[-1].set_xlabel('年份；灰带：前 30 年，点线：前 60 年')
        fig.suptitle(f"冰川 {sid}   {g['rgi_id']}\n{len(group)} 个稳态初始状态 · 同一历史气候 · 暂通过 {g['provisional_curve_pass_count']} 个",fontsize=15)
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes,label='仅 spin-up 的温度偏差 ΔT_eq (℃)',shrink=.7,pad=.015)
        name=f'glacier_{sid}.png'
        fig.savefig(pdest/name,dpi=130)
        plt.close(fig)
        paths.append(name)
        if len(paths)%25==0:
            print(json.dumps({'glacier_figures':len(paths),'total':173}),flush=True)
    # Every glacier is shown in deterministic ID order, without choosing a winner.
    for first in range(0,len(groups),16):
        subset=groups[first:first+16]
        fig,axes=plt.subplots(4,4,figsize=(17,12),layout='constrained')
        for ax,(g,arrays) in zip(axes.flat,subset):
            for r,a in arrays:
                ax.plot(a.year,a.length_m/1000,color=cmap(norm(r['delta_t_eq_degC'])),lw=.7,alpha=.8)
            ax.axhline(g['lia_reference_m']/1000,color='#333333',ls='--',lw=.8)
            ax.plot([2000],[g['rgi_reference_m']/1000],'*',color='#13743e',ms=7)
            ax.set_title(f"{g['lia_unique_id']}   n={len(arrays)}   暂通过 {g['provisional_curve_pass_count']}",fontsize=10)
            ax.tick_params(labelsize=8)
            ax.grid(axis='y',alpha=.15)
            ax.set_xlim(1470,2008)
            ax.set_ylabel('km',fontsize=8)
        for ax in list(axes.flat)[len(subset):]:
            ax.set_visible(False)
        fig.suptitle(f'173 条冰川的全部长度曲线   第 {first//16+1}/11 页\n虚线：LIA 参考；绿星：RGI 2000；蓝至红：ΔT_eq = −1 至 +1 ℃',fontsize=14)
        fig.savefig(pdest/f'overview_{first//16+1:02d}.png',dpi=125)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.7),layout='constrained')
    labels=['LIA 峰值 ≤10%','RGI 末端 ≤10%','峰值、末端都符合','曲线形态符合','全部现行条件符合']
    keys=['lia_gate','rgi_gate','endpoint_pair_gate','shape_gate','provisional_curve_pass']
    values=[summary['gates'][k]['cells'] for k in keys]
    axes[0].barh(labels[::-1],values[::-1],color='#315c7b')
    for i,v in enumerate(values[::-1]):
        axes[0].text(v+10,i,str(v),va='center',fontsize=11)
    axes[0].set_xlim(0,1393)
    axes[0].set_xlabel('参数组合数量（不同项目并非依次筛选）')
    axes[0].set_title('1393 个组合：各项条件分别统计')
    spread=pd.read_csv(OUT/'initial_state_spread_annual.csv')
    multi=spread[spread.parent_count>1]
    for col,label,color in [('length_m_spread_fraction_of_initial','长度','#315c7b'),
                            ('volume_m3_spread_fraction_of_initial','体积','#a7502e')]:
        by=multi.groupby('year')[col]
        axes[1].plot(by.median().index,by.median(),color=color,label=label+'中位数')
        axes[1].fill_between(by.median().index,by.quantile(.25),by.quantile(.75),color=color,alpha=.12)
    axes[1].axhline(1,color='#666666',lw=.8,ls='--')
    axes[1].set_xlabel('年份')
    axes[1].set_ylabel('同冰川初态间差异 / 1470 年差异')
    axes[1].set_title('初始状态差异如何保留\n阴影为冰川间四分位范围；初始差异为零者不算比值')
    axes[1].legend(frameon=False)
    fig.suptitle('固定历史气候的 forward 结果 · 没有新增模拟',fontsize=15)
    fig.savefig(pdest/'summary.png',dpi=145)
    plt.close(fig)
    sections=[]
    for g in glaciers:
        sid=g['lia_unique_id']
        n=g['stable_parent_count']
        content=(f'<a href="figures/glacier_{sid}.png"><img loading="lazy" src="figures/glacier_{sid}.png" alt="冰川{sid}全部曲线"></a>' if n else '<p>900 年网格没有稳态初态；保留在 191 条冰川名单中，未启动 forward，也未作最终失败判断。</p>')
        sections.append(f'<details data-id="{sid} {g["rgi_id"]}"><summary>冰川 {sid} · {g["rgi_id"]} · {n} 个组合 · 暂通过 {g["provisional_curve_pass_count"]} 个</summary>{content}</details>')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定历史气候冰川曲线</title><style>body{max-width:1180px;margin:30px auto;padding:0 20px;font:16px/1.6 system-ui;color:#253442;background:#f7f8fa}h1{font-size:27px}img{width:100%;height:auto}summary{padding:12px;cursor:pointer}details{background:white;margin:8px 0;border:1px solid #dce2e6}input{padding:10px;font:inherit;width:340px}a{color:#245d86}.nav a{margin-right:14px}p{padding:0 10px}</style>
    <h1>固定历史气候：全部冰川曲线</h1><p>1393 个组合已运行至 2000 年，覆盖 173 条冰川；另外 18 条没有稳态初态，名单完整保留。每条冰川的颜色只表示 spin-up 温度偏差；forward 均使用同一继承气候与参数，未携带这个偏差。所有曲线均保留，不选一条代表其余。</p>
    <p>峰值与末端分别允许总长度误差 10%；保留已定方向、振幅与 v3 形态条件。初始长度误差和前 30/60 年变化只作诊断。曲线形态的原计算时段为 1500–2000，图中仍展示全部 1470–2000。暂通过不等于正式接受，尚需独立复现。</p><img src="figures/summary.png" alt="总体统计与初态敏感性"><div class="nav">'''
    html+=' '.join(f'<a href="figures/overview_{i:02d}.png">总览 {i}</a>' for i in range(1,12))
    html+='</div><p><a href="cell_scores_1393.csv">全部组合指标</a>　<a href="glacier_summary_191.csv">191 条冰川汇总</a></p><input id="search" placeholder="输入冰川 ID 或 RGI 编号">'+''.join(sections)
    html+='''<script>document.getElementById('search').addEventListener('input',e=>{for(const d of document.querySelectorAll('details'))d.hidden=!d.dataset.id.toLowerCase().includes(e.target.value.toLowerCase())});</script></html>'''
    with (OUT/'index.html').open('x',encoding='utf-8') as f:
        f.write(html)
    write(OUT/'figures_manifest.json',{'created_at':now(),'glacier_figures':173,'overview_pages':11,
                                     'outputs':[ref(p) for p in sorted(pdest.glob('*.png'))]+[ref(OUT/'index.html')]})
    print(json.dumps({'plots_complete':True,'gallery':str(OUT/'index.html')}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['audit-score','plots'])
    args=parser.parse_args()
    if args.mode=='audit-score':
        audit_score()
    else:
        plots()
