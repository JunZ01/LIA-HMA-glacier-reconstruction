"""Saved-output analysis only. No OGGM, forcing reader, solver or model launch."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import html
import importlib.util
import itertools
import json

import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'validation_data/results/paleo_surface_validation_v0'
PREV = ROOT / 'analysis/lia_51_systematic_analysis_v1'
PARAM = ROOT / 'analysis/lia_51_climate_parameter_analysis_v1'
LOADER = ROOT / 'analysis/lia_51_terminus_elevation_curves_v1/build_grouped_curves.py'
SOURCES = {}
METRICS = [('length_km', '长度', 'km'), ('area_km2', '面积', 'km²'), ('volume_km3', '体积', 'km³')]
COLORS = {'stable': '#218573', 'timing_sensitive': '#bf7026', 'amplitude_sensitive': '#b74657',
          'boundary': '#81509d', 'untested': '#72818e'}
LABELS = {'stable': '已有组合内较稳定', 'timing_sensitive': '幅度较稳、时序敏感',
          'amplitude_sensitive': '幅度也敏感', 'boundary': '边界控制、时序受限', 'untested': '备选证据不足'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def reg(p, expected=None):
    p = Path(p)
    if not p.is_absolute():
        p = ROOT / p
    lowered = p.as_posix().lower()
    assert not any(s in lowered for s in ['20cr', 'jionglacuo', 's7t4zc34sr', 'temperature reconstruction.xlsx'])
    assert p.name != 'inputs.npz', 'Raw forcing bundle is outside scope'
    digest = sha(p)
    if expected is not None:
        assert digest == expected.upper(), f'Hash mismatch: {p}'
    SOURCES[p.relative_to(ROOT).as_posix()] = digest
    return p


def read(p, expected=None):
    return json.loads(reg(p, expected).read_text(encoding='utf-8-sig'))


def save(name, data):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def segments(years, mask):
    ix = np.flatnonzero(mask)
    if not len(ix):
        return []
    blocks = np.split(ix, np.flatnonzero(np.diff(ix) > 1) + 1)
    return [[float(years[b[0]]), float(years[b[-1]])] for b in blocks]


def hausdorff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if not len(a) or not len(b):
        return None
    distance = np.abs(a[:, None] - b[None, :])
    return float(max(distance.min(axis=0).max(), distance.min(axis=1).max()))


def peak(years, x, start=1500):
    use = (years >= start) & (years <= 2000)
    y, v = years[use], x[use]
    value = float(v.max())
    mask = np.isclose(v, value, rtol=1e-10, atol=1e-12)
    return dict(value=value, first_year=float(y[mask][0]), last_year=float(y[mask][-1]),
                segments=segments(y, mask), tied_points=int(mask.sum()),
                left_boundary=bool(mask[0]), right_boundary=bool(mask[-1]),
                source_interval_max_years=float(np.diff(y).max()))


def near(years, values, frac):
    mask = values >= values.max() * (1 - frac) - 1e-12 * max(1, values.max())
    return dict(fraction=frac, years=years[mask].tolist(), segments=segments(years, mask),
                grid_points=int(mask.sum()), left_boundary=bool(mask[0]), right_boundary=bool(mask[-1]))


def grid(years, values, start=1500, step=10):
    # Include 2000 even if a start-anchored 20yr grid has a final 10yr interval.
    y = np.unique(np.r_[np.arange(start, 2001, step), 2000]).astype(float)
    return y, np.interp(y, years, values)


def interval_states(years, values):
    threshold = float(values.max() * METHODS['interval_state_fraction'])
    changes = np.diff(values)
    states = np.where(changes > threshold, 1, np.where(changes < -threshold, -1, 0))
    cuts = np.r_[0, np.flatnonzero(np.diff(states)) + 1, len(states)]
    runs = [dict(start=float(years[a]), end=float(years[b]), duration_years=float(years[b]-years[a]),
                 state=int(states[a]), change=float(values[b]-values[a]),
                 change_pct_of_start=float(100*(values[b]/values[a]-1)))
            for a, b in zip(cuts[:-1], cuts[1:])]
    growing = np.flatnonzero(states == 1)
    last = int(growing[-1] + 1) if len(growing) else 0
    decline = years[-1]-years[last] >= 30 and values[last]-values[-1] >= .01*values.max()
    return dict(years=years.tolist(), values=values.tolist(), states=states.tolist(), runs=runs,
                sustained_runs=[r for r in runs if r['duration_years'] >= 20 and r['state'] != 0],
                final_decline_start=float(years[last]) if decline else None,
                final_decline_left_censored=bool(decline and last == 0))


def diagnose(years, values):
    p = peak(years, values)
    primary_y, primary_x = grid(years, values)
    sensitivity = []
    for start, step, fraction in itertools.product(METHODS['window_starts'], METHODS['common_steps_years'], METHODS['near_maximum_fractions']):
        y, x = grid(years, values, start, step)
        indices, props = find_peaks(x, prominence=fraction*x.max(), distance=max(1, int(np.ceil(20/step))), plateau_size=1)
        local = [dict(year=float(y[k]), value=float(x[k]), prominence_pct=float(100*props['prominences'][i]/x.max()),
                      plateau=[float(y[props['left_edges'][i]]), float(y[props['right_edges'][i]])])
                 for i, k in enumerate(indices)]
        sensitivity.append(dict(start=start, step=step, fraction=fraction, near=near(y, x, fraction),
                                local_peaks=local, common_grid_peak=peak(y, x, start)))
    base = next(d for d in sensitivity if d['start']==1500 and d['step']==10 and d['fraction']==.01)
    periods = [dict(start=s, end=e, absolute=float(np.interp(e, years, values)-np.interp(s, years, values)),
                    change_pct=float(100*(np.interp(e, years, values)/np.interp(s, years, values)-1)))
               for s,e in METHODS['fixed_periods']]
    # Compare window choices on the actual shared domain, not removed early years.
    window_distances = []
    for d in sensitivity:
        if d['step']==10 and d['fraction']==.01:
            shared = max(1500, d['start'])
            distance = hausdorff([t for t in base['near']['years'] if t>=shared], [t for t in d['near']['years'] if t>=shared])
            window_distances.append(dict(start=d['start'], shared_start=shared, hausdorff_years=distance))
    coarse = next(d for d in sensitivity if d['start']==1500 and d['step']==20 and d['fraction']==.01)
    return dict(native_peak=p, native_window_peaks=[dict(start=s, **peak(years, values,s)) for s in METHODS['window_starts']],
                near=base['near'], local_peaks=base['local_peaks'], interval=interval_states(primary_y, primary_x),
                interval20=interval_states(*grid(years,values,1500,20)), sensitivity=sensitivity,
                window_distances=window_distances, resolution_hausdorff_years=hausdorff(base['near']['years'], coarse['near']['years']),
                periods=periods, loss_peak_to_2000_pct=float(100*(1-values[-1]/p['value'])),
                loss_1900_2000_pct=float(100*(1-values[-1]/np.interp(1900,years,values))),
                smallest_saved_positive_increment=float(np.min(np.diff(np.unique(np.round(values,10))))) if len(np.unique(np.round(values,10)))>1 else None,
                increment_note='Observed distinct-value spacing only; not proof of physical/model grid resolution')


def load_data():
    spec = importlib.util.spec_from_file_location('saved_curve_loader', LOADER)
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    reg(LOADER)
    curves, members, sampling = loader.load_sources()
    SOURCES.update(loader.SOURCES)
    parameter_manifest = read(PARAM/'output_manifest.json')
    parameters = read(PARAM/'parameters_51.json', parameter_manifest['parameters_51.json'])
    old_manifest = read(PREV/'output_manifest.json')
    old_alts = read(PREV/'saved_alternative_sensitivity.json', old_manifest['saved_alternative_sensitivity.json'])
    gallery = read(BASE/'lia_new38_legacy_review_recalculation_v1/data.json')
    targets = {d['system'] for d in old_alts}
    selected = {m['system']:m for m in members}
    catalogue_cache = {}
    alternatives = defaultdict(list)
    for g in gallery:
        sid = g['system']
        if sid not in targets:
            continue
        for c in g['cases']:
            proof = c['proof']
            rp = reg(proof['result']['path'], proof['result']['sha256'])
            sp = reg(proof['state']['path'], proof['state']['sha256'])
            result = read(rp)
            assert result['candidate']==c['candidate']
            with np.load(sp,allow_pickle=False) as z:
                years = z['years'].copy().astype(float)
                values = z['history'].copy()
            assert np.array_equal(years,np.arange(1470,2001)) and values.shape[0]==531 and values.shape[1]>=3
            # Bind parent identity from the saved catalog, not merely equal delta values.
            cp = rp.parents[3]/'catalog.json'
            if cp not in catalogue_cache:
                catalogue_cache[cp] = read(cp) if cp.exists() else None
            catalogue = catalogue_cache[cp]
            matches = [] if catalogue is None else [r for r in catalogue['rows'] if ((r.get('forward') or {}).get('result') or {}).get('path')==proof['result']['path']]
            parent = matches[0].get('spin') if len(matches)==1 else None
            if parent:
                reg(parent['result']['path'],parent['result']['sha256'])
            case = dict(system=sid, case_id=rp.parent.name, result_source=proof['result']['path'],
                        state_source=proof['state']['path'], selected=proof['result']['path']==selected[sid].get('result_source'),
                        parameters=c['candidate'], parent_state_sha256=parent['state']['sha256'] if parent else None,
                        parent_identity_basis='saved catalog, result hash checked; parent array not reopened' if parent else 'not established',
                        years=years, values={'length_km':values[:,0]/1000,'area_km2':values[:,1]/1e6,'volume_km3':values[:,2]/1e9})
            alternatives[sid].append(case)
    assert len(alternatives)==15 and sum(map(len,alternatives.values()))==113
    for sid,cases in alternatives.items():
        assert sum(c['selected'] for c in cases)==1
        ref = next(c for c in curves if c['system']==sid)
        chosen = next(c for c in cases if c['selected'])
        for key,_,_ in METRICS:
            assert np.array_equal(chosen['values'][key],ref['native_'+key])
    return curves,members,parameters,alternatives,old_alts


def reference_scale_distances(reference, fraction):
    base=next(v['near']['years'] for v in reference['sensitivity'] if v['start']==1500 and v['step']==10 and v['fraction']==fraction)
    coarse=next(v['near']['years'] for v in reference['sensitivity'] if v['start']==1500 and v['step']==20 and v['fraction']==fraction)
    distances=[hausdorff(base,coarse)]
    for v in reference['sensitivity']:
        if v['step']==10 and v['fraction']==fraction:
            shared=max(1500,v['start'])
            distances.append(hausdorff([t for t in base if t>=shared],[t for t in v['near']['years'] if t>=shared]))
    return distances


def classify(rel, reference, timing=30, amplitude=2, fraction=.01):
    if rel['case_count']<2:
        return 'untested'
    boundary = any(rel[k]['boundary_any'] for k in ['length_km','volume_km3'])
    if boundary:
        return 'boundary'
    amp = all(v<=amplitude for v in rel['amplitude_ranges_pp'].values())
    if not amp:
        return 'amplitude_sensitive'
    stable = True
    for key in ['length_km','volume_km3']:
        r=rel[key]
        stable &= r['near_set_pairwise_distances'][str(fraction)]<=timing
        stable &= all(d is not None and d<=timing for d in reference_scale_distances(reference[key],fraction))
    return 'stable' if stable else 'timing_sensitive'


def evaluate_alternatives(alternatives, references):
    diagnostics, reliability = [], []
    pair_tests = []
    for sid, cases in alternatives.items():
        records=[]
        for c in cases:
            d={k:v for k,v in c.items() if k not in ['values','years']}
            d['metrics']={key:diagnose(c['years'],c['values'][key]) for key,_,_ in METRICS}
            diagnostics.append(d); records.append(d)
        rel=dict(system=sid,case_count=len(cases),distinct_parent_count=len({c['parent_state_sha256'] for c in cases if c['parent_state_sha256']}),
                 initial_offsets=sorted({c['parameters']['delta'] for c in cases}),
                 forward_offsets=sorted({c['parameters']['constant_offset_degC'] for c in cases}))
        for key,_,_ in METRICS:
            peaks=[c['metrics'][key]['native_peak'] for c in records]
            distances={}
            for fraction in METHODS['near_maximum_fractions']:
                sets=[next(v['near']['years'] for v in c['metrics'][key]['sensitivity'] if v['start']==1500 and v['step']==10 and v['fraction']==fraction) for c in records]
                distances[str(fraction)]=max([hausdorff(a,b) for a,b in itertools.combinations(sets,2)]+[0.])
            sets=[set(c['metrics'][key]['near']['years']) for c in records]
            rel[key]=dict(exact_first_peak_year_range=[min(p['first_year'] for p in peaks),max(p['first_year'] for p in peaks)],
                          boundary_any=any(p['left_boundary'] or p['right_boundary'] for p in peaks),
                          boundary_cases=sum(p['left_boundary'] or p['right_boundary'] for p in peaks),
                          near_set_pairwise_distances=distances, common_near_years=sorted(set.intersection(*sets)),
                          union_near_years=sorted(set.union(*sets)),
                          local_peak_counts=sorted({len(c['metrics'][key]['local_peaks']) for c in records}))
        rel['amplitude_ranges_pp']={key+'_peak_loss':float(np.ptp([c['metrics'][key]['loss_peak_to_2000_pct'] for c in records])) for key in ['length_km','volume_km3']}
        rel['amplitude_ranges_pp']['volume_loss_1900']=float(np.ptp([c['metrics']['volume_km3']['loss_1900_2000_pct'] for c in records]))
        rel['amplitude_stable_at_2pp']=all(v<=2 for v in rel['amplitude_ranges_pp'].values()) if len(cases)>=2 else None
        rel['metric_timing_evidence']={key:dict(case_distance_years=rel[key]['near_set_pairwise_distances']['0.01'],
            reference_window_and_resolution_distances_years=reference_scale_distances(references[sid]['metrics'][key],.01),
            boundary_any=rel[key]['boundary_any'],
            interpretation='conditional case/scale diagnostics, not independent chronology') for key in ['length_km','volume_km3']}
        rel['label']=classify(rel,references[sid]['metrics'])
        rel['label_zh']=LABELS[rel['label']]
        rel['threshold_sensitivity']=[dict(timing_years=t,amplitude_pp=a,fraction=f,label=classify(rel,references[sid]['metrics'],t,a,f)) for t,a,f in itertools.product(METHODS['timing_distance_sensitivity_years'],METHODS['amplitude_spread_sensitivity_percentage_points'],METHODS['near_maximum_fractions'])]
        # Pairwise historical evidence only: parameter equality and exact saved-parent identity are distinct.
        for a,b in itertools.combinations(records,2):
            pa,pb=a['parameters'],b['parameters']
            except_delta = all(pa[k]==pb[k] for k in pa if k!='delta')
            same_parent = a['parent_state_sha256'] is not None and a['parent_state_sha256']==b['parent_state_sha256']
            except_q = all(pa[k]==pb[k] for k in pa if k!='constant_offset_degC')
            kind='same_forward_different_initial_setting' if except_delta and pa['delta']!=pb['delta'] else ('same_saved_parent_different_forward_q' if same_parent and except_q and pa['constant_offset_degC']!=pb['constant_offset_degC'] else None)
            if kind:
                pair_tests.append(dict(system=sid,a=a['case_id'],b=b['case_id'],kind=kind,
                                       length_near_hausdorff_years=hausdorff(a['metrics']['length_km']['near']['years'],b['metrics']['length_km']['near']['years']),
                                       volume_near_hausdorff_years=hausdorff(a['metrics']['volume_km3']['near']['years'],b['metrics']['volume_km3']['near']['years']),
                                       volume_1900_loss_difference_pp=abs(a['metrics']['volume_km3']['loss_1900_2000_pct']-b['metrics']['volume_km3']['loss_1900_2000_pct'])))
        reliability.append(rel)
    return diagnostics,reliability,pair_tests


def regional(atlas, case_diags):
    records=[]
    for key,_,_ in METRICS:
        for group in ['all','low','middle','high']:
            rows=[d for d in atlas if group=='all' or d['elevation_group']==group]
            years=np.arange(1500,2001,10)
            occ=np.array([[t in d['metrics'][key]['near']['years'] for t in years] for d in rows],float)
            states=np.array([d['metrics'][key]['interval']['states'] for d in rows])
            pixels=sorted({d['climate_pixel'] for d in rows})
            def weighted(a):
                return np.mean([a[[i for i,d in enumerate(rows) if d['climate_pixel']==p]].mean(axis=0) for p in pixels],axis=0).tolist()
            lower=occ.copy(); upper=occ.copy()
            for i,d in enumerate(rows):
                alts=[c for c in case_diags if c['system']==d['system']]
                if len(alts)>=2:
                    a=np.array([[t in c['metrics'][key]['near']['years'] for t in years] for c in alts],float)
                    lower[i]=a.min(axis=0); upper[i]=a.max(axis=0)
            rec=dict(metric=key,group=group,n=len(rows),pixel_n=len(pixels),years=years.tolist(),
                     near_equal_system=occ.mean(axis=0).tolist(),near_equal_pixel=weighted(occ),
                     conditional_lower=lower.mean(axis=0).tolist(),conditional_upper=upper.mean(axis=0).tolist(),
                     near_threshold_sensitivity=[])
            for f in METHODS['near_maximum_fractions']:
                o=np.array([[t in next(v['near']['years'] for v in d['metrics'][key]['sensitivity'] if v['start']==1500 and v['step']==10 and v['fraction']==f) for t in years] for d in rows],float)
                rec['near_threshold_sensitivity'].append(dict(fraction=f,equal_system=o.mean(axis=0).tolist()))
            for value,name in [(-1,'shrinking'),(0,'small'),(1,'growing')]:
                indicator=states==value
                rec[name+'_equal_system']=indicator.mean(axis=0).tolist()
                rec[name+'_equal_pixel']=weighted(indicator)
            records.append(rec)
    return records


def figsave(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150)
    fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)


def timelines(atlas):
    cmap=ListedColormap(['#477cab','#eeeeee','#bd7a38']); norm=BoundaryNorm([-1.5,-.5,.5,1.5],3)
    for order,field in [('elevation','terminus_elevation_m'),('longitude','longitude')]:
        rows=sorted(atlas,key=lambda d:d[field])
        fig,axes=plt.subplots(1,3,figsize=(17,14),sharey=True)
        for ax,(key,label,_) in zip(axes,METRICS):
            a=np.array([d['metrics'][key]['interval']['states'] for d in rows])
            ax.pcolormesh(np.arange(1500,2001,10),np.arange(len(rows)+1)-.5,a,cmap=cmap,norm=norm,shading='flat')
            for i,d in enumerate(rows):
                for lo,hi in d['metrics'][key]['near']['segments']:
                    ax.plot([lo,hi] if lo!=hi else [lo-1,hi+1],[i,i],color='#222c31',lw=2.2)
                p=d['metrics'][key]['native_peak']
                if p['left_boundary'] or p['right_boundary']:
                    ax.scatter(p['first_year'],i,marker='>',s=35,color='#b02549',clip_on=False)
            ax.set_title(label,fontsize=14); ax.set_xlim(1500,2000); ax.set_xlabel('年份')
            ax.set_xticks(np.arange(1500,2001,100)); ax.grid(axis='x',alpha=.2)
        axes[0].set_yticks(range(len(rows)))
        axes[0].set_yticklabels([f"{d['system']} · {d[field]:.0f} m" if order=='elevation' else f"{d['system']} · {d[field]:.2f}°E" for d in rows],fontsize=8)
        axes[0].invert_yaxis()
        fig.suptitle('51条冰川历史阶段｜按'+('LIA末端高程' if order=='elevation' else '经度')+'排列',fontsize=18,y=.984)
        handles=[Line2D([],[],color='#477cab',lw=6,label='退缩/减小'),Line2D([],[],color='#bd7a38',lw=6,label='推进/增大'),Line2D([],[],color='#ddd',lw=6,label='变化小于阈值'),Line2D([],[],color='#222c31',lw=2,label='距窗口最大值≤1%')]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.96),ncol=4,frameon=False)
        fig.text(.5,.015,'颜色：十年端点变化超过窗口最大值的±0.1%；黑线可分段，不代表独立定年。红标记：精确峰值触及窗口边界。',ha='center',fontsize=10)
        fig.subplots_adjust(left=.15,right=.97,top=.91,bottom=.065,wspace=.08)
        figsave(fig,'01_timeline_'+order)


def plot_region(records):
    fig,axes=plt.subplots(2,3,figsize=(16,8.8),sharex=True)
    for j,(key,label,_) in enumerate(METRICS):
        d=next(r for r in records if r['metric']==key and r['group']=='all'); y=d['years']
        ax=axes[0,j]
        ax.fill_between(y,np.array(d['conditional_lower'])*100,np.array(d['conditional_upper'])*100,color='#879da8',alpha=.25,label='已有备选的条件性范围')
        ax.plot(y,np.array(d['near_equal_system'])*100,color='#254d68',lw=2,label='等冰川权重')
        ax.plot(y,np.array(d['near_equal_pixel'])*100,color='#ab6d2c',ls='--',label='等气候像元权重')
        ax.set_title(label+'：接近各自窗口最大值'); ax.set_ylabel('样本占比 / %');ax.legend(fontsize=8,frameon=False)
        ax=axes[1,j]
        for key2,lab,col in [('growing','推进/增大','#bd7a38'),('shrinking','退缩/减小','#477cab'),('small','变化小','#888888')]:
            ax.plot(np.array(y[:-1])+5,np.array(d[key2+'_equal_system'])*100,color=col,label=lab,lw=1.7)
        ax.set_title(label+'：十年变化状态');ax.set_ylabel('样本占比 / %');ax.set_xlabel('年份')
    for ax in axes.ravel():ax.set_xlim(1500,2000);ax.set_ylim(0,100);ax.grid(alpha=.15)
    fig.suptitle('样本内的共同演变：不将区域平均峰值当作统一LIA年代',fontsize=17,y=.98)
    fig.legend(*axes[1,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.525),ncol=3,frameon=False,fontsize=9)
    fig.text(.5,.015,'51条选定系统、33个气候像元；阴影只改变有备选的系统，其余固定，不是完整不确定性或概率区间。',ha='center',fontsize=10)
    fig.subplots_adjust(top=.88,bottom=.10,wspace=.28,hspace=.36)
    figsave(fig,'02_regional_activity')


def plot_phase_sensitivity(atlas,records):
    fig,axes=plt.subplots(3,2,figsize=(14,11),sharex=True)
    for j,(key,label) in enumerate([('length_km','长度'),('volume_km3','体积')]):
        allrow=next(r for r in records if r['metric']==key and r['group']=='all')
        for s,color in zip(allrow['near_threshold_sensitivity'],['#648995','#203d59','#bd803b']):
            axes[0,j].plot(allrow['years'],100*np.array(s['equal_system']),label=f"距最大值≤{s['fraction']*100:g}%",color=color,lw=1.7)
        for group,lab,color in [('low','<4000 m','#286c9f'),('middle','4000—<4500 m','#b78032'),('high','≥4500 m','#8b5598')]:
            d=next(r for r in records if r['metric']==key and r['group']==group)
            axes[1,j].plot(d['years'],100*np.array(d['near_equal_system']),label=f"{lab} · n={d['n']}",color=color,lw=1.7)
        for step,color in [(10,'#203d59'),(20,'#bd803b')]:
            y=np.arange(1500,2001,step)
            a=np.array([[t in next(v['near']['years'] for v in d['metrics'][key]['sensitivity'] if v['start']==1500 and v['step']==step and v['fraction']==.01) for t in y] for d in atlas])
            axes[2,j].plot(y,100*a.mean(axis=0),label=f'{step}年比较网格',color=color,lw=1.7)
        axes[0,j].set_title(label+'：阈值敏感性（全部51条）')
        axes[1,j].set_title(label+'：末端高程组（1%阈值）')
        axes[2,j].set_title(label+'：时间尺度敏感性（1%阈值）')
    for ax in axes.ravel():
        ax.set_xlim(1500,2000);ax.set_ylim(0,100);ax.set_ylabel('接近各自窗口最大值的占比 / %');ax.grid(alpha=.15)
        ax.legend(frameon=False,fontsize=8,loc='upper left')
    for ax in axes[-1]:ax.set_xlabel('年份')
    fig.suptitle('共同阶段是否稳定？阈值、高程组与时间尺度并列检查',fontsize=17,y=.985)
    fig.text(.5,.014,'阈值改变会改变最集中年份；面积平台更宽。上述差异是样本描述，不证明高程或气候参数的因果效应。',ha='center',fontsize=10)
    fig.subplots_adjust(left=.09,right=.97,top=.91,bottom=.065,hspace=.36,wspace=.20)
    figsave(fig,'05_phase_sensitivity')


def plot_reliability(atlas,rels):
    fig,axes=plt.subplots(1,2,figsize=(15,7),gridspec_kw={'width_ratios':[1,1.15]})
    for label,color in COLORS.items():
        rows=[d for d in atlas if d['reliability_label']==label]
        if rows:
            axes[0].scatter([d['longitude'] for d in rows],[d['latitude'] for d in rows],c=color,s=[25+5*np.sqrt(d['static_area_km2']) for d in rows],label=f'{LABELS[label]} · {len(rows)}',alpha=.85)
    axes[0].set_xlabel('经度 / °E');axes[0].set_ylabel('纬度 / °N');axes[0].set_title('空间分布（点大小表示静态面积）');axes[0].legend(fontsize=9,frameon=False)
    rows=sorted(rels,key=lambda d:d['system'])
    for i,d in enumerate(rows):
        for offset,key,col in [(-.14,'length_km','#477cab'),(.14,'volume_km3','#bd7a38')]:
            dist=d[key]['near_set_pairwise_distances']['0.01']
            axes[1].barh(i+offset,dist,height=.25,color=col,alpha=.8)
        if d['case_count']<2:axes[1].text(5,i,'仅1条：不可检验',va='center',fontsize=8)
    axes[1].set_yticks(range(len(rows)));axes[1].set_yticklabels([f"{d['system']} · n={d['case_count']}" for d in rows]);axes[1].invert_yaxis()
    axes[1].axvline(30,color='#555',ls='--',lw=1)
    axes[1].set_xlabel('各组合近最大时段的最大集合距离 / 年');axes[1].set_title('已有备选覆盖：长度（蓝）与体积（棕）')
    for ax in axes:ax.grid(alpha=.15)
    fig.suptitle('可靠性是历史结论的标签，不改变51条统一样本或正式接纳记录',fontsize=16,y=.98)
    fig.text(.5,.015,'“较稳定”仅限已保存组合及本次诊断规则；36条无该备选集，另1条仅保存单个组合，均不能据此判断稳健。',ha='center',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.12,wspace=.28)
    figsave(fig,'03_reliability')


def individual_plots(atlas, curves, alternatives):
    folder=OUT/'glaciers';folder.mkdir(exist_ok=True)
    byid={c['system']:c for c in curves}
    for d in atlas:
        sid=d['system'];c=byid[sid]
        fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
        for ax,(key,label,unit) in zip(axes,METRICS):
            for a in alternatives.get(sid,[]):
                ax.plot(a['years'],a['values'][key],color='#a9b3bc',alpha=.65,lw=.85)
            y=np.array(c['native_years']);x=np.array(c['native_'+key])
            ax.plot(y,x,color='#234e69',lw=1.6,label='原选定曲线')
            for lo,hi in d['metrics'][key]['near']['segments']:
                ax.axvspan(max(1500,lo-5),min(2000,hi+5),color='#d9aa63',alpha=.23)
            for p in d['metrics'][key]['local_peaks']:
                ax.scatter(p['year'],p['value'],s=25,color='#a24b48',zorder=4)
            ax.set_ylabel(f'{label} / {unit}');ax.grid(alpha=.16);ax.set_xlim(1500,2000)
        axes[-1].set_xlabel('年份')
        ncases=len(alternatives.get(sid,[]))
        coverage=f'审计集合 {ncases} 个（含原选定）' if ncases else '仅主曲线，未纳入跨方案检验'
        fig.suptitle(f"{sid}｜末端 {d['terminus_elevation_m']:.0f} m｜{d['longitude']:.2f}°E\n{LABELS[d['reliability_label']]} · {coverage}",fontsize=14,y=.985)
        fig.text(.5,.02,'深蓝：原选定历史曲线；灰：已有备选；浅棕：十年网格上距最大值≤1%（显示各端点±5年）；红点：满足突出度的局部峰。',ha='center',fontsize=8)
        fig.subplots_adjust(top=.89,bottom=.08,hspace=.18,left=.12,right=.97)
        fig.savefig(folder/f'{sid}.png',dpi=130);plt.close(fig)


def plot_representatives(atlas,curves,alternatives,rels):
    # Fixed diagnostic examples discussed before this analysis plus objective coverage rules.
    chosen=[1012,2580]
    stable=[d for d in rels if d['label']=='stable']
    if stable:chosen.append(max(stable,key=lambda d:d['case_count'])['system'])
    changing=[d for d in rels if len(d['forward_offsets'])>1 and d['system'] not in chosen]
    if changing:chosen.append(max(changing,key=lambda d:d['case_count'])['system'])
    chosen=list(dict.fromkeys(chosen))
    lookup={c['system']:c for c in curves}
    fig,axes=plt.subplots(len(chosen),2,figsize=(14,3.1*len(chosen)),squeeze=False)
    for i,sid in enumerate(chosen):
        for j,(key,label) in enumerate([('length_km','长度'),('volume_km3','体积')]):
            ax=axes[i,j];c=lookup[sid]
            for a in alternatives[sid]:
                # Shared selected-2000 denominator preserves absolute differences across cases.
                ax.plot(a['years'],100*(a['values'][key]/c['native_'+key][-1]-1),color='#9ca9b2',lw=.9,alpha=.6)
            ax.plot(c['native_years'],100*(np.array(c['native_'+key])/c['native_'+key][-1]-1),color='#244f6b',lw=1.8)
            ax.set_title(f'{sid} · {label} · {len(alternatives[sid])} 个已有组合',fontsize=12)
            ax.set_ylabel('相对原选定2000年值 / %');ax.grid(alpha=.15);ax.set_xlim(1500,2000)
    for ax in axes[-1]:ax.set_xlabel('年份')
    fig.suptitle('历史曲线的条件性差异：统一参照保留组合间幅度差别',fontsize=17,y=.985)
    fig.text(.5,.015,'灰线不是概率集合；深蓝为原选定曲线。代表选择规则：1012/2580诊断例＋较稳类最多组合＋额外含多q的最多组合。',ha='center',fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.06,hspace=.38,wspace=.22)
    figsave(fig,'04_representatives')
    return chosen


def segtext(segments_):
    return '；'.join(f'{a:g}' if a==b else f'{a:g}—{b:g}' for a,b in segments_) or '无'


def write_report(atlas,rels,summary):
    rows=[]
    for d in sorted(atlas,key=lambda x:x['terminus_elevation_m']):
        l=d['metrics']['length_km'];v=d['metrics']['volume_km3']
        rows.append('<tr>'+''.join('<td>'+str(x)+'</td>' for x in [
            f'<a href="glaciers/{d["system"]}.png">{d["system"]}</a>',f'{d["terminus_elevation_m"]:.0f}',f'{d["longitude"]:.2f}',
            d['native_step_years'],LABELS[d['reliability_label']],d['alternative_case_count'],
            segtext(l['native_peak']['segments'])+('＊' if l['native_peak']['left_boundary'] or l['native_peak']['right_boundary'] else ''),
            segtext(l['near']['segments']),len(l['local_peaks']),segtext(v['near']['segments']),
            f'{v["loss_peak_to_2000_pct"]:.1f}',f'{v["loss_1900_2000_pct"]:.1f}'])+'</tr>')
    counts='；'.join(f'{LABELS[k]} {v}条' for k,v in summary['label_counts'].items())
    sections=''
    for title,name,caption in [
        ('1 · 51条历史阶段：末端高程排序','01_timeline_elevation','长度、面积、体积并列，观察共同阶段和不同步。黑线是近最大时段，不是独立LIA定年。'),
        ('2 · 空间顺序：由西向东','01_timeline_longitude','改变排列顺序不改变样本或阶段定义；避免将高程与区域位置混为一谈。'),
        ('3 · 样本内共同变化','02_regional_activity','逐冰川与逐气候像元权重并列。备选范围只覆盖14条有多个保存组合的冰川，不能作为区域年代置信区间。'),
        ('4 · 阶段识别的阈值与尺度敏感性','05_phase_sensitivity','1%规则下，长度近最大范围占比在1880年最高（24/51），体积在1870年最高（34/51）。但体积阈值放宽到2%后最集中点移到1770年，不能把某个最高点直接写成统一LIA最大年代。'),
        ('5 · 可靠性与证据覆盖','03_reliability',counts+'。边界标签优先，但连续差异指标和阈值敏感性均保留。'),
        ('6 · 代表冰川：历史曲线的条件性差异','04_representatives','所有备选保持既有选择，不重打分。1012/2580的最早最大值年份跳变需要结合近最大平台解释，不能直接称为数百年定年误差。')]:
        sections+=f'<section><h2>{title}</h2><p>{caption}</p><a href="{name}.svg"><img src="{name}.png" loading="lazy"></a></section>'
    headers=['编号/曲线','末端高程 m','经度 °E','原生间隔 年','条件性标签','备选集记录数','精确长度峰值＊边界','长度近最大时段','长度局部峰数','体积近最大时段','体积峰后损失 %','体积1900后损失 %']
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>51条冰川 · LIA阶段与可靠性</title>
<style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;color:#243845;background:#f0f3f5;margin:0}main{max-width:1300px;margin:auto;padding:35px 25px}h1{font-size:30px}h2{font-size:22px}section{background:white;border-radius:12px;padding:24px;margin:22px 0}img{width:100%;height:auto}a{color:#216b8a}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}th{background:#edf2f5;position:sticky;top:0}.scroll{overflow:auto;max-height:680px}.note{border-left:5px solid #bc843a;padding:15px;background:#fff8eb}input{padding:10px;width:300px;max-width:90%;margin-bottom:15px}.stats{color:#446575;font-size:20px}</style><main>
<h1>51条冰川：LIA演变阶段与可靠性图谱</h1>
<p class="stats">51条统一样本 · 1500—2000主窗口 · 113个已有组合 / 15条覆盖 · 0次新模拟</p>
<div class="note">研究的是历史演变，不是spinup过程。这里只分析保存的长度、面积、体积。1470年前不在模拟范围内；登记1400年不等于独立最大期定年。51条来源标记保持原样，可靠性标签不改变正式接纳台账。</div>
<section><h2>先读结论边界</h2><p>同一冰川可以具有稳定的20世纪损失，却有不稳定的“最早最大值年份”。本报告把精确最大值、近最大平台、多次局部推进分开，保留所有不连续时段。1%近最大范围、30年时序差异、2个百分点幅度差异均为本次描述规则，不是物理误差或既有接纳标准。</p>
<p>本次具体发现：36/51条的精确最大长度在多个保存时点重复出现，其中6条有不连续的并列最大时段；1%阈值下，21/51条具有多段近最大长度时期。因此“单个峰值年”通常不足以概括曲线。1012号原选定曲线的精确长度峰值同时出现在1679—1694年和1788—1797年；2580号部分备选的早期差异则同时影响长度与体积最大期。</p>
<p>原生1年/10年数据均保留；比较采用10年与20年网格。无备选和仅单个方案都不等于稳定。备选范围不是随机集合，不计算“成功概率”。参数与气候的机制归因及独立观测验证尚未完成。</p></section>'''
    page+=sections+'<section><h2>7 · 51条完整总表</h2><p>点击编号查看该冰川的三项历史曲线与全部已有备选。近最大区间只表示满足阈值的网格点跨度；＊标记精确峰值触及窗口边界。</p><input id="filter" placeholder="搜索编号、标签或年份"><div class="scroll"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div></section>'
    page+='''<section><h2>8 · 方法、数据与下一步</h2><p>先确认多阶段推进和峰顶平台是否真实可区分，再解释空间差异。降水校准因子不等于实际降水；长度目标已参与拟合，不能重复作为独立验证。体积曲线是模型量，不能据此声称已验证古气候或历史质量平衡。</p>
<p>已有同历史参数、不同初态设置的组合可检查初态依赖；同一保存父态、不同历史q的组合可检查历史校正敏感性。身份核查与逐对结果保留，但这些有限比较不覆盖完整物理不确定性。</p>
<p><a href="atlas.json">51条诊断</a> · <a href="conditional_reliability.json">条件性可靠性与阈值敏感性</a> · <a href="pair_comparisons.json">配对比较</a> · <a href="regional_activity.json">样本共同阶段</a> · <a href="methods.json">完整定义</a> · <a href="summary.json">摘要</a> · <a href="validation.json">验证记录</a> · <a href="source_manifest.json">来源哈希</a></p></section>
</main><script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))})</script></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')


def unit_tests():
    y=np.arange(1500,1561,10,dtype=float)
    assert segments(y,np.array([1,1,0,1,0,0,1],bool))==[[1500.,1510.],[1530.,1530.],[1560.,1560.]]
    assert hausdorff([1500,1510,1800],[1500,1510,1790])==10
    assert hausdorff([1500,1800],[1600])==200
    assert hausdorff([], [1500]) is None
    p=peak(y,np.array([5,4,5,3,2,1,5],float))
    assert p['left_boundary'] and p['right_boundary'] and len(p['segments'])==3
    n=near(y,np.array([100,99,98,99,97,96,100],float),.01)
    assert n['segments']==[[1500.,1510.],[1530.,1530.],[1560.,1560.]]
    d=interval_states(y,np.array([10,9,8,7,6,5,4],float))
    assert d['states']==[-1]*6 and d['final_decline_left_censored']
    d=interval_states(y,np.ones(7))
    assert d['states']==[0]*6 and d['final_decline_start'] is None
    y2,x2=grid(np.array([1470.,2000.]),np.array([1.,2.]),1470,20)
    assert y2[-1]==2000 and np.diff(y2).min()==10 and x2[-1]==2
    return 9


def main():
    global METHODS
    METHODS=read(OUT/'methods.json')
    reg(Path(__file__))
    before={name:sha(ROOT/name) for name in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']}
    tests=unit_tests()
    curves,members,parameters,alternatives,old_alts=load_data()
    print('Sources and selected curves verified; analyzing 51 references and 113 saved cases.',flush=True)
    pars={d['system']:d for d in parameters}; ms={d['system']:d for d in members}
    atlas=[]
    for c in curves:
        sid=c['system'];p=pars[sid];m=ms[sid]
        d={k:p[k] for k in ['system','status','longitude','latitude','terminus_elevation_m','elevation_group','static_area_km2','gradient_deg','azimuth_deg','climate_pixel','policy_group','route_family']}
        d['source']=m['source'];d['length_operator']=m['length_operator'];d['native_step_years']=m['native_step_years_max']
        d['metrics']={key:diagnose(np.array(c['native_years']),np.array(c['native_'+key])) for key,_,_ in METRICS}
        atlas.append(d)
    references={d['system']:d for d in atlas}
    case_diags,rels,pairs=evaluate_alternatives(alternatives,references)
    rlookup={d['system']:d for d in rels}
    for d in atlas:
        rel=rlookup.get(d['system'])
        d['reliability_label']=rel['label'] if rel else 'untested'
        d['alternative_case_count']=rel['case_count'] if rel else 0
        d['evidence_scope']='conditional saved cases' if rel and rel['case_count']>=2 else 'no multi-case robustness established'
    old_count=0
    for g in old_alts:
        for old in g['cases']:
            new=next(c for c in case_diags if c['result_source']==old['result_source'])
            assert np.isclose(new['metrics']['volume_km3']['loss_1900_2000_pct'],old['volume_loss_1900_pct'],atol=1e-10,rtol=0)
            # Old argmax chose strictly largest; new tie-aware peak may legitimately differ.
            old_count+=1
    activity=regional(atlas,case_diags)
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),systems=51,formal_acceptance_unchanged=True,
                 simulation_launches=0,raw_climate_access=False,alternative_systems=15,alternative_cases=113,
                 multiple_case_systems=sum(r['case_count']>=2 for r in rels),climate_pixels=len({d['climate_pixel'] for d in atlas}),
                 label_counts=dict(Counter(d['reliability_label'] for d in atlas)),
                 repeated_exact_length_peak_systems=sum(d['metrics']['length_km']['native_peak']['tied_points']>1 for d in atlas),
                 disjoint_exact_length_peak_systems=sum(len(d['metrics']['length_km']['native_peak']['segments'])>1 for d in atlas),
                 multi_near_length_interval_systems=sum(len(d['metrics']['length_km']['near']['segments'])>1 for d in atlas),
                 exact_peak_boundary_counts={key:sum(d['metrics'][key]['native_peak']['left_boundary'] or d['metrics'][key]['native_peak']['right_boundary'] for d in atlas) for key,_,_ in METRICS},
                 local_peak_count_distribution={key:dict(Counter(str(len(d['metrics'][key]['local_peaks'])) for d in atlas)) for key,_,_ in METRICS},
                 near_segment_count_distribution={key:dict(Counter(str(len(d['metrics'][key]['near']['segments'])) for d in atlas)) for key,_,_ in METRICS},
                 pair_counts=dict(Counter(p['kind'] for p in pairs)),
                 classification_threshold_counts=[dict(timing_years=t,amplitude_pp=a,fraction=f,counts=dict(Counter(classify(r,references[r['system']]['metrics'],t,a,f) for r in rels))) for t,a,f in itertools.product([20,30,50],[1,2,5],[.005,.01,.02])])
    for name,obj in [('atlas.json',atlas),('case_diagnostics.json',case_diags),('conditional_reliability.json',rels),('pair_comparisons.json',pairs),('regional_activity.json',activity),('reference_curves.json',curves)]:save(name,obj)
    print('Diagnostics complete; rendering overview figures and 51 glacier panels.',flush=True)
    timelines(atlas);plot_region(activity);plot_phase_sensitivity(atlas,activity);plot_reliability(atlas,rels)
    summary['representative_systems']=plot_representatives(atlas,curves,alternatives,rels)
    individual_plots(atlas,curves,alternatives)
    save('summary.json',summary);write_report(atlas,rels,summary)
    for name,digest in before.items():assert sha(ROOT/name)==digest, f'Canonical file changed: {name}'
    for p,digest in SOURCES.items():assert sha(ROOT/p)==digest,f'Source changed: {p}'
    save('source_manifest.json',SOURCES)
    save('validation.json',dict(status='numerical-pass-visual-review-pending',unit_tests=tests,
         selected_curve_count=51,selected_alternative_exact_matches=15,old_case_volume_metric_closure=old_count,
         reference_diagnostic_variants=51*3*18,alternative_diagnostic_variants=113*3*18,
         checked_source_hashes=len(SOURCES),canonical_hashes=before,canonical_files_unchanged=True,
         raw_climate_access=False,simulations_launched=0,formal_acceptance_changed=False))
    save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps({k:v for k,v in summary.items() if k!='classification_threshold_counts'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
