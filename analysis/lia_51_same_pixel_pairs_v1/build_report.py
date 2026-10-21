"""Saved-output, within-registered-pixel comparisons. No glacier model imports."""
from __future__ import annotations
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib, html, itertools, json, sys
sys.dont_write_bytecode = True
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PREV = ROOT/'analysis/lia_51_stage_controls_v1'
ATLAS = ROOT/'analysis/lia_51_stage_reliability_v1'
METRICS = ['length_km', 'volume_km3']
NAMES = {'length_km':'长度', 'volume_km3':'体积'}
ROUTES = {'legacy_constant':'恒定·原分支', 'candidate_constant':'恒定·GPU', 'synthetic_timevarying':'时变·合成系统'}
SOURCES = {}
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(), 'font.size':10,
                     'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def reg(p, expected=None):
    p = Path(p); p = p if p.is_absolute() else ROOT/p
    assert not any(s in p.as_posix().lower() for s in ['20cr', 'jionglacuo', 's7t4zc34sr', 'temperature reconstruction.xlsx'])
    assert p.name != 'inputs.npz'
    digest = sha(p)
    if expected: assert digest == expected.upper(), str(p)
    SOURCES[p.relative_to(ROOT).as_posix()] = digest
    return p
def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x, (float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, (bool,np.bool_)): return bool(x)
    if isinstance(x, np.integer): return int(x)
    return x
def save(name, x): (OUT/name).write_text(json.dumps(clean(x), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def mean(t,x,a,b):
    y = np.r_[a, t[(t>a)&(t<b)], b]
    return float(np.trapz(np.interp(y,t,x),y)/(b-a))
def interpolate(c,key,step=10):
    t = np.arange(1500,2001,step,dtype=float)
    x = np.interp(t,c['native_years'],c['native_'+key])
    assert np.all(np.isfinite(x)) and min(x)>0
    return t,x
def centered(c,key,step=10,baseline=(1600,1700)):
    t,x = interpolate(c,key,step)
    maximum = float(x.max()); ref = mean(t,x,*baseline)
    return t,100*x/maximum,100*(x-ref)/maximum,maximum,ref
def bracket(year,step):
    lo = 1500+np.floor((year-1500+1e-9)/step)*step
    hi = 1500+np.ceil((year-1500-1e-9)/step)*step
    return [int(lo),int(hi)]


def episodes(t,g,threshold=2.,minimum=30.):
    """Exact threshold crossing on saved common-grid piecewise-linear curves."""
    out = []
    for sign in [1,-1]:
        intervals = []
        for i in range(len(t)-1):
            l,r = max(t[i],1700.),min(t[i+1],1900.)
            if r<=l: continue
            a,b = sign*np.interp([l,r],t,g)
            if max(a,b)<threshold: continue
            if min(a,b)>=threshold: left,right=l,r
            else:
                cross=l+(threshold-a)/(b-a)*(r-l)
                left,right=(cross,r) if b>a else (l,cross)
            if right-left<=1e-9: continue
            if intervals and abs(left-intervals[-1][1])<1e-8: intervals[-1][1]=right
            else: intervals.append([left,right])
        for left,right in intervals:
            if right-left+1e-8<minimum: continue
            out.append(dict(sign=sign,start_estimate=left,end_estimate=right,duration_years=right-left,
                start_bracket=bracket(left,t[1]-t[0]),end_bracket=bracket(right,t[1]-t[0]),
                left_censored=abs(left-1700)<1e-8,right_censored=abs(right-1900)<1e-8))
    return sorted(out,key=lambda d:(d['start_estimate'],d['sign']))


def describe(ca,cb,key,step=10,baseline=(1600,1700),threshold=2.,minimum=30.,full=True):
    t,na,xa,ma,ra = centered(ca,key,step,baseline)
    _,nb,xb,mb,rb = centered(cb,key,step,baseline)
    g = xb-xa; ep = episodes(t,g,threshold,minimum)
    ans = dict(step=step,baseline=baseline,threshold_pp=threshold,minimum_duration_years=minimum,
               late_contrast_pp=mean(t,g,1800,1900),modern_mean_gap_pp=mean(t,g,1900,2000),episodes=ep)
    if full:
        periods=[]
        for a in range(1500,2000,50):
            b=a+50
            av=np.interp([a,b],t,na); bv=np.interp([a,b],t,nb)
            periods.append(dict(start=a,end=b,mean_gap_pp=mean(t,g,a,b),a_net_change_pp=float(av[1]-av[0]),b_net_change_pp=float(bv[1]-bv[0])))
        largest=max(periods[:8],key=lambda p:abs(p['mean_gap_pp']))
        ans.update(years=t,normalized_a=na,normalized_b=nb,centered_a=xa,centered_b=xb,gap_pp=g,
                   normalization_a=ma,normalization_b=mb,reference_a=ra,reference_b=rb,periods=periods,
                   largest_premodern_absolute_bin=largest)
    return ans


def ratio(a,b): return max(a,b)/min(a,b)
def haversine(a,b):
    lon1,lat1,lon2,lat2=np.radians([a['longitude'],a['latitude'],b['longitude'],b['latitude']])
    h=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return float(6371*2*np.arcsin(np.sqrt(np.clip(h,0,1))))


def load_inputs():
    methods=read(reg(OUT/'methods.json'))
    authority=methods['authority']
    if (OUT/'authority_reconciliation_v1.json').exists():
        reconciliation=read(reg(OUT/'authority_reconciliation_v1.json'))
        assert reconciliation['initial_authority_hashes']==authority
        prefix=(ROOT/'LIA_PROJECT_FACTS.md').read_bytes()[:reconciliation['facts_previous_bytes']]
        assert hashlib.sha256(prefix).hexdigest().upper()==reconciliation['facts_previous_sha256']
        reg(reconciliation['current_route']['path'],reconciliation['current_route']['sha256'])
        authority=reconciliation['current_authority_hashes']
    for p,h in authority.items(): reg(p,h)
    inp={}
    for folder,names in [(PREV,['features_51.json','validation.json']), (ATLAS,['reference_curves.json','atlas.json','case_diagnostics.json'])]:
        manifest=read(reg(folder/'output_manifest.json'))
        for name in names: inp[name]=read(reg(folder/name,manifest[name]))
    assert inp['validation.json']['status']=='pass'
    # Check previously verified lineage. Earlier authority snapshots are provenance, not current facts.
    for p,h in read(reg(PREV/'source_manifest.json')).items():
        if p in methods['authority']: continue
        reg(p,h)
    rows=inp['features_51.json']; curves={c['system']:c for c in inp['reference_curves.json']}
    atlas={c['system']:c for c in inp['atlas.json']}
    assert len(rows)==51 and set(curves)=={r['system'] for r in rows}
    groups=defaultdict(list)
    for r in rows: groups[r['climate_pixel']].append(r)
    peer_ids={r['system'] for group in groups.values() if len(group)>1 for r in group}
    cases=defaultdict(list)
    for c in inp['case_diagnostics.json']:
        if c['system'] not in peer_ids: continue
        p=reg(c['state_source']); assert all(c['parameters'][k]==1 for k in ['melt','prcp','glen'])
        with np.load(p,allow_pickle=False) as f:
            y=f['years'].copy(); h=f['history'].copy()
        assert y[0]==1470 and y[-1]==2000 and np.all(np.diff(y)>0)
        d=dict(c); d['curve']=dict(native_years=y,native_length_km=h[:,0]/1000,native_volume_km3=h[:,2]/1e9)
        if c['selected']:
            for key in METRICS:
                t,x=interpolate(d['curve'],key); _,ref=interpolate(curves[c['system']],key)
                assert np.allclose(x,ref,atol=1e-10,rtol=1e-10)
        cases[c['system']].append(d)
    return methods,rows,curves,atlas,groups,cases


def metadata_pairs(rows,atlas,groups):
    scales={k:float(np.subtract(*np.percentile([r[k] if k!='log_melt' else np.log10(r['melt_f']) for r in rows],[75,25]))) for k in ['base_temp_bias_c','q_mean_1500_2000_c','log_melt','log_glen_a']}
    for k in scales:
        if scales[k]<=1e-10:scales[k]=float(np.std([r[k] if k!='log_melt' else np.log10(r['melt_f']) for r in rows]))
    pairs=[]
    for pixel,group in sorted(groups.items()):
        if len(group)<2: continue
        assert max(r['prcp_fac'] for r in group)-min(r['prcp_fac'] for r in group)<1e-9
        for a,b in itertools.combinations(sorted(group,key=lambda r:(r['terminus_elevation_m'],r['system'])),2):
            same=a['route_family']==b['route_family']; constant=not a['timevarying'] and not b['timevarying']
            diffs={k:float(b[k]-a[k]) for k in ['terminus_elevation_m','gradient_deg','base_temp_bias_c','q_mean_1500_2000_c','melt_f','log_glen_a']}
            score=np.mean([abs((b[k]-a[k]) if k!='log_melt' else np.log10(b['melt_f']/a['melt_f']))/s for k,s in scales.items() if s>1e-10])
            pairs.append(dict(pair_id=f"{a['system']}_{b['system']}",pixel=pixel,a=a['system'],b=b['system'],
                same_route=same,same_route_constant=same and constant,both_constant=constant,
                route_a=a['route_family'],route_b=b['route_family'],parameter_distance=float(score),differences=diffs,
                area_ratio=ratio(a['static_area_km2'],b['static_area_km2']),melt_ratio=ratio(a['melt_f'],b['melt_f']),
                glen_ratio=ratio(a['glen_a'],b['glen_a']),azimuth_separation_deg=abs((b['azimuth_deg']-a['azimuth_deg']+180)%360-180),
                centroid_distance_km=haversine(a,b),length_operator_a=atlas[a['system']]['length_operator'],length_operator_b=atlas[b['system']]['length_operator'],
                reliability_a=a['previous_reliability_label'],reliability_b=b['previous_reliability_label']))
    eligible=[p for p in pairs if p['same_route_constant']]
    spot=[]
    for kind,pool,key in [
        ('height',[p for p in eligible if abs(p['differences']['gradient_deg'])<=2 and p['area_ratio']<=2],lambda p:(-p['differences']['terminus_elevation_m'],p['pair_id'])),
        ('slope',[p for p in eligible if p['differences']['terminus_elevation_m']<=600 and p['area_ratio']<=2],lambda p:(-abs(p['differences']['gradient_deg']),p['pair_id']))]:
        if pool: spot.append(dict(reason=kind,pair_id=sorted(pool,key=key)[0]['pair_id']))
    pool=[p for p in eligible if p['pair_id'] not in {s['pair_id'] for s in spot} and (p['differences']['terminus_elevation_m']>=400 or abs(p['differences']['gradient_deg'])>=3)]
    if pool:spot.append(dict(reason='parameter_proximity',pair_id=sorted(pool,key=lambda p:(p['parameter_distance'],p['pair_id']))[0]['pair_id']))
    return pairs,spot,scales


def analyze(pairs,curves,cases,methods):
    sensitivity=[]; replacements=[]
    for p in pairs:
        p['metrics']={}; p['conditional']={}; p['definition_sensitivity']={}
        for key in METRICS:
            ca,cb=curves[p['a']],curves[p['b']]
            d=describe(ca,cb,key); p['metrics'][key]=d
            variants=[]
            for base,step,threshold,minimum in itertools.product(methods['sensitivity']['reference_windows'],methods['sensitivity']['grid_steps'],methods['sensitivity']['threshold_pp'],methods['sensitivity']['minimum_duration_years']):
                s=describe(ca,cb,key,step,base,threshold,minimum,full=False)
                s.update(pair_id=p['pair_id'],metric=key); sensitivity.append(s); variants.append(s)
            detected=[s for s in variants if s['episodes']]
            p['definition_sensitivity'][key]=dict(definitions=len(variants),detected=len(detected),
                first_start_bracket_span=[min(s['episodes'][0]['start_bracket'][0] for s in detected),max(s['episodes'][0]['start_bracket'][1] for s in detected)] if detected else None,
                first_signs=sorted({s['episodes'][0]['sign'] for s in detected}),left_censored_definitions=sum(s['episodes'][0]['left_censored'] for s in detected))
            paths=[np.asarray(d['gap_pp'])]; late=[d['late_contrast_pp']]; rs=[]
            for side in ['a','b']:
                for case in cases[p[side]]:
                    if case['selected']: continue
                    a=case['curve'] if side=='a' else ca; b=case['curve'] if side=='b' else cb
                    alt=describe(a,b,key)
                    rec=dict(pair_id=p['pair_id'],metric=key,replaced_system=p[side],case_id=case['case_id'],
                        parameters=case['parameters'],parent_state_sha256=case['parent_state_sha256'],state_source=case['state_source'],
                        late_contrast_pp=alt['late_contrast_pp'],gap_pp=alt['gap_pp'],episodes=alt['episodes'])
                    rs.append(rec); replacements.append(rec); paths.append(np.asarray(alt['gap_pp']));late.append(alt['late_contrast_pp'])
            p['conditional'][key]=dict(nonselected_replacements=len(rs),tested_endpoints=[p[side] for side in ['a','b'] if any(not c['selected'] for c in cases[p[side]])],
                gap_min_pp=np.min(paths,axis=0),gap_max_pp=np.max(paths,axis=0),late_min_pp=min(late),late_max_pp=max(late),
                replacement_detections=sum(bool(r['episodes']) for r in rs),reference_detection=bool(d['episodes']))
    return sensitivity,replacements


def aggregate(pairs):
    results=[]
    for cohort in ['all_pairs','same_route_constant']:
        subset=[p for p in pairs if cohort=='all_pairs' or p['same_route_constant']]
        for orientation in ['height','slope']:
            for key in METRICS:
                bypixel=defaultdict(list)
                for p in subset:
                    sign=1 if orientation=='height' else np.sign(p['differences']['gradient_deg'])
                    if sign==0:continue
                    bypixel[p['pixel']].append((p,sign))
                per_pixel=[]
                for pixel,entries in sorted(bypixel.items()):
                    gaps=np.array([np.array(p['metrics'][key]['gap_pp'])*s for p,s in entries])
                    late=[p['metrics'][key]['late_contrast_pp']*s for p,s in entries]
                    per_pixel.append(dict(pixel=pixel,pair_count=len(entries),median_gap_pp=np.median(gaps,axis=0),median_late_contrast_pp=float(np.median(late))))
                matrix=np.array([x['median_gap_pp'] for x in per_pixel]); late=np.array([x['median_late_contrast_pp'] for x in per_pixel])
                lo=[float(np.median(np.delete(late,i))) for i in range(len(late))]
                results.append(dict(cohort=cohort,orientation=orientation,metric=key,pair_count=len(subset),pixel_count=len(per_pixel),
                    years=pairs[0]['metrics'][key]['years'],pixel_results=per_pixel,median_gap_pp=np.median(matrix,axis=0),
                    q25_gap_pp=np.percentile(matrix,25,axis=0),q75_gap_pp=np.percentile(matrix,75,axis=0),
                    median_late_contrast_pp=float(np.median(late)),positive_pixels=int(sum(late>1e-8)),negative_pixels=int(sum(late<-1e-8)),zero_pixels=int(sum(abs(late)<=1e-8)),
                    leave_one_pixel_median_range_pp=[min(lo),max(lo)]))
    return results


def finish_figure(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=145,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def era(ax,baseline=True):
    ax.axvspan(1470,1500,color='#bdc4cc',alpha=.18)
    ax.axvspan(1900,2000,color='#bdc4cc',alpha=.12)
    if baseline:ax.axvspan(1600,1700,color='#d8b76b',alpha=.13)
    ax.axvline(1700,color='#a2a7ab',ls=':',lw=.7);ax.axvline(1900,color='#a2a7ab',ls=':',lw=.7)
    ax.set_xlim(1470,2000);ax.grid(alpha=.15)


def figures(rows,pairs,spots,curves,groups,aggregates):
    lookup={r['system']:r for r in rows}; peers={p[s] for p in pairs for s in ['a','b']}
    fig,axes=plt.subplots(1,2,figsize=(14,5.2))
    for flag,label,color in [(False,'无同像元对象（22）','#b6bfc6'),(True,'有同像元对象（29）','#377895')]:
        r=[r for r in rows if (r['system'] in peers)==flag]
        axes[0].scatter([x['longitude'] for x in r],[x['latitude'] for x in r],c=color,label=label,s=45)
    for pixel,group in groups.items():
        if len(group)>1:
            x=np.mean([r['longitude'] for r in group]); y=np.mean([r['latitude'] for r in group])
            offset={'30:25':(-38,28),'30:26':(-35,-24),'30:27':(24,20),'31:27':(25,-15),'34:36':(-40,14),'34:37':(14,-15)}.get(pixel,(3,5))
            axes[0].annotate(pixel,(x,y),xytext=offset,textcoords='offset points',fontsize=8,
                             arrowprops={'arrowstyle':'-','color':'#a5b0b8','lw':.6} if offset!=(3,5) else None)
    axes[0].set(xlabel='冰川中心经度 / °E',ylabel='冰川中心纬度 / °N',title='样本覆盖：11个多冰川像元');axes[0].legend(fontsize=9)
    for strict,label,color in [(False,'其它配对16','#b6bfc6'),(True,'同分支·恒定11','#377895')]:
        ps=[p for p in pairs if p['same_route_constant']==strict]
        axes[1].scatter([p['differences']['terminus_elevation_m'] for p in ps],[p['differences']['gradient_deg'] for p in ps],c=color,label=label,s=45)
    for s in spots:
        p=next(p for p in pairs if p['pair_id']==s['pair_id']);axes[1].annotate(p['pair_id'].replace('_','→'),(p['differences']['terminus_elevation_m'],p['differences']['gradient_deg']),xytext=(4,5),textcoords='offset points',fontsize=8)
    axes[1].axhline(0,color='#aaaaaa',lw=.7);axes[1].set(xlabel='高末端 B − 低末端 A：高程差 / m',ylabel='B − A：坡度差 / °',title='高程差与坡度差常同时出现，不能单独归因');axes[1].legend(fontsize=9)
    fig.suptitle('51条保留为总样本；全部27组同像元配对均展示',fontsize=16);fig.tight_layout();finish_figure(fig,'01_coverage_geometry')
    fig,axes=plt.subplots(1,2,figsize=(13,10),sharey=True)
    for ax,key in zip(axes,METRICS):
        matrix=np.array([[q['mean_gap_pp'] for q in p['metrics'][key]['periods']] for p in pairs]);vmax=max(3.,float(np.ceil(abs(matrix).max())))
        im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        ax.set_xticks(range(10),[f'{a}–{a+50}' for a in range(1500,2000,50)],rotation=60,ha='right',fontsize=8)
        ax.set_yticks(range(len(pairs)),[('* ' if p['same_route_constant'] else '')+p['pair_id'].replace('_','→') for p in pairs],fontsize=8)
        ax.set_title(NAMES[key]+'：高末端 B − 低末端 A')
        for i in range(len(pairs)):
            for j in range(10):ax.text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if abs(matrix[i,j])>vmax*.55 else '#333333')
        ax.axvline(7.5,color='#555555',lw=1);fig.colorbar(im,ax=ax,shrink=.65,label='相对参考世纪的差异 / 百分点')
    fig.suptitle('差异集中在哪些50年窗口？  * 同分支且恒定校正',fontsize=16)
    fig.text(.5,.01,'每条曲线先除以自身1500—2000最大值，再扣除自身1600—1700均值；红色表示B相对更大。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.96]);finish_figure(fig,'02_pair_periods')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for i,orient in enumerate(['height','slope']):
        for j,key in enumerate(METRICS):
            ax=axes[i,j]
            for cohort,color,label in [('all_pairs','#b4a5b9','全部配对：11像元'),('same_route_constant','#337b9c','同分支恒定：7像元')]:
                a=next(a for a in aggregates if (a['cohort'],a['orientation'],a['metric'])==(cohort,orient,key))
                ax.plot(a['years'],a['median_gap_pp'],color=color,label=label,lw=2)
                if cohort=='same_route_constant':ax.fill_between(a['years'],a['q25_gap_pp'],a['q75_gap_pp'],color=color,alpha=.14,label='像元间四分位范围')
            ax.axhline(0,color='#888888',lw=.8);era(ax);ax.set_xlim(1500,2000)
            ax.set_title(('高末端 − 低末端' if orient=='height' else '较陡 − 较缓')+' · '+NAMES[key]);ax.set_ylabel('参考世纪居中差异 / 百分点')
            if i==1:ax.set_xlabel('历史年份')
            if i==j==0:ax.legend(fontsize=8)
    fig.suptitle('先在像元内汇总，再给予每个像元同等权重',fontsize=16)
    fig.text(.5,.01,'配对共享冰川，不能把27组当成27个独立样本；曲线与区间均为描述统计，不是地形因果效应。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);finish_figure(fig,'03_pixel_balanced')
    colors=['#347e9e','#b96a36','#7d6996','#638a4b']
    for pixel,group in sorted(groups.items()):
        if len(group)<2:continue
        fig,axes=plt.subplots(2,2,figsize=(13,7.8),sharex=True)
        for color,r in zip(colors,sorted(group,key=lambda r:r['terminus_elevation_m'])):
            for j,key in enumerate(METRICS):
                c=curves[r['system']];t,n,x,m,ref=centered(c,key)
                ny=np.array(c['native_years']);nx=np.array(c['native_'+key])
                label=f"{r['system']} · {r['terminus_elevation_m']:.0f}m · {r['gradient_deg']:.1f}°"
                axes[0,j].plot(ny,nx/m*100,color=color,label=label,lw=1.5)
                axes[1,j].plot(t,x,color=color,label=label,lw=1.5)
        for i in range(2):
            for j,key in enumerate(METRICS):
                ax=axes[i,j];era(ax);ax.set_title(NAMES[key]+('：原始保存曲线 / 共同网格最大值' if i==0 else '：相对1600—1700平均值'))
                ax.set_ylabel('各自最大值的百分比' if i==0 else '变化 / 百分点')
                if i==1:ax.axhline(0,color='#888888',lw=.7);ax.set_xlabel('历史年份')
        axes[0,0].legend(fontsize=8,loc='best')
        fig.suptitle(f'气候像元 {pixel} | {len(group)}条冰川；同像元≠相同有效强迫',fontsize=16)
        fig.text(.5,.01,'浅黄：参考世纪；灰色：早期边界及20世纪。原始曲线保留本来时间分辨率，居中曲线采用共同10年网格。',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.04,1,.96]);finish_figure(fig,'pixel_'+pixel.replace(':','_'))
    for s in spots:
        p=next(p for p in pairs if p['pair_id']==s['pair_id']);fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
        for j,key in enumerate(METRICS):
            d=p['metrics'][key];cond=p['conditional'][key]
            for side,color in [('a','#347e9e'),('b','#b96a36')]:
                sid=p[side];c=curves[sid];r=lookup[sid]
                label=f"{sid} · {r['terminus_elevation_m']:.0f}m · {r['gradient_deg']:.1f}°"
                axes[0,j].plot(c['native_years'],c['native_'+key],color=color,label=label,lw=1.5)
                axes[1,j].plot(d['years'],d['normalized_'+side],color=color,lw=1.5)
            axes[0,j].set_ylabel(NAMES[key]+(' / km' if key=='length_km' else ' / km³'));axes[0,j].legend(fontsize=8)
            axes[1,j].set_ylabel('各自窗口最大值 / %')
            axes[2,j].plot(d['years'],d['gap_pp'],color='#374f68',label='参考配对')
            if cond['nonselected_replacements']:
                axes[2,j].fill_between(d['years'],cond['gap_min_pp'],cond['gap_max_pp'],color='#739da3',alpha=.25,label='单端备选条件范围')
            for v in [-2,0,2]:axes[2,j].axhline(v,color='#999999',ls='--' if v else '-',lw=.7)
            for e in d['episodes']:axes[2,j].axvspan(e['start_estimate'],e['end_estimate'],color='#bda259',alpha=.14)
            axes[2,j].set_ylabel('居中 B−A / 百分点');axes[2,j].set_xlabel('历史年份');axes[2,j].legend(fontsize=8)
            for i in range(3):era(axes[i,j])
        fig.suptitle(f"{p['a']} → {p['b']} | 像元 {p['pixel']} | 高末端B − 低末端A",fontsize=16)
        df=p['differences'];fig.text(.5,.01,f"Δ高程 {df['terminus_elevation_m']:.0f}m；Δ坡度 {df['gradient_deg']:+.2f}°；Δ基础温偏 {df['base_temp_bias_c']:+.2f}℃；Δq {df['q_mean_1500_2000_c']:+.2f}℃；融化因子比 {p['melt_ratio']:.2f}；冰流参数比 {p['glen_ratio']:.2f}\n金色区段为1700—1900内的描述性持续差异，不是实际分化起始年代；备选范围不是置信区间。",ha='center',fontsize=8.5)
        fig.tight_layout(rect=[0,.055,1,.96]);finish_figure(fig,'pair_'+p['pair_id'])


def event_text(events):
    if not events:return '未达到本次标准'
    e=events[0];a,b=e['start_bracket'];text='1700时已存在' if e['left_censored'] else (str(a) if a==b else f'{a}—{b}')
    return text+f"；持续约{e['duration_years']:.0f}年"+('；延伸至1900边界' if e['right_censored'] else '')
def fmt(x,n=2):return f'{x:.{n}f}'
def table(headers,body):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in body)+'</tbody></table></div>'


def report(rows,pairs,spots,groups,aggregates,summary):
    sections=[]
    sections.append('<h1>同气候像元，冰川何时出现不同演变？</h1><p class="lead">51条统一样本 · 29条有同像元对象 · 11个像元 · 全部27组配对</p>')
    sections.append('<div class="note">这里比较已经保存的历史长度和体积，不研究spinup、不启动模拟、不改变接纳记录。<strong>同像元不是相同有效强迫，更不是只改变地形的实验。</strong>模型分支、温度校正、融化因子、冰流参数及初始状态仍可能不同。</div>')
    sections.append('<h2>先读结果</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['findings'])+'</ul>')
    sections.append('<nav><a href="#pixels">11个像元完整曲线</a> · <a href="#pairs">全部27组配对</a> · <a href="#limits">解释边界与复核</a> · <a href="../lia_51_stage_controls_v1/index.html">上一轮全样本分析</a></nav>')
    sections.append('<h2>1. 配对覆盖和地形差异</h2><img src="01_coverage_geometry.png"><p>箭头始终从低末端A指向高末端B。同分支恒定子集共11组、7个像元；另有1组同分支时变配对。27组之间共享冰川，不能当成27个独立样本。</p>')
    sections.append('<h2>2. 差异出现在哪些时期？</h2><p>每条曲线先除以自身1500—2000最大值，再扣除自身1600—1700的平均值。以下差值为高末端B减低末端A，参考世纪平均差为零是计算定义，不表示当时状态或强迫相同。正值表示B相对参考世纪更大／缩小得更少，不等于B当时持续增长。</p><img src="02_pair_periods.png">')
    sections.append('<h2>3. 每个像元同等权重，方向是否一致？</h2><img src="03_pixel_balanced.png">')
    sections.append(table(['配对子集','比较方向','指标','像元数','19世纪差值中位数 / pp','正／负／零像元','逐像元剔除范围 / pp'],[[('全部' if a['cohort']=='all_pairs' else '同分支恒定'),('高末端−低末端' if a['orientation']=='height' else '较陡−较缓'),NAMES[a['metric']],a['pixel_count'],fmt(a['median_late_contrast_pp']),f"{a['positive_pixels']} / {a['negative_pixels']} / {a['zero_pixels']}",' — '.join(fmt(v) for v in a['leave_one_pixel_median_range_pp'])] for a in aggregates]))
    sections.append('<p>先求每个像元内配对中位数，再跨像元汇总；时间曲线的逐时中位数与世纪指标的中位数不必相同。范围是敏感性范围，不是概率区间。</p><h2>4. 按地形与参数元数据选出的重点配对</h2><p>选择规则在计算配对曲线差异前固定，未按结果强弱挑选。但这是已知全样本关联后的探索，不是独立确认。所有27组仍完整保留。</p>')
    spot_names={'height':'坡度相近、面积比≤2中，高程差最大','slope':'高程差≤600m、面积比≤2中，坡度差最大','parameter_proximity':'有明显地形差异的其余配对中，参数距离最小'}
    for s in spots:
        p=next(p for p in pairs if p['pair_id']==s['pair_id'])
        sections.append(f'<h3>{p["a"]} → {p["b"]}：{spot_names[s["reason"]]}</h3><img src="pair_{p["pair_id"]}.png">')
        sections.append('<p>'+ '；'.join(NAMES[k]+'持续差异：'+event_text(p['metrics'][k]['episodes']) for k in METRICS) +'。这里只报告1700—1900窗口内首次达到2个百分点、30年条件的位置，不称为历史分化的真实起点。</p>')
        sections.append('<div class="note">'+html.escape(summary['spotlight_interpretations'][p['pair_id']])+'</div>')
    sections.append('<h2 id="pixels">5. 11个像元：全部历史曲线及参数</h2>')
    for pixel,group in sorted(groups.items()):
        if len(group)<2:continue
        ident=pixel.replace(':','_')
        sections.append(f'<details id="pixel_{ident}"><summary>像元 {pixel} · '+', '.join(str(r['system']) for r in group)+'</summary><img loading="lazy" src="pixel_'+ident+'.png">')
        sections.append(table(['冰川','末端m','坡度°','面积km²','朝向°','分支','基础温偏℃','q均值℃','融化因子','降水因子','冰流A','已有备选诊断'],[[r['system'],fmt(r['terminus_elevation_m'],0),fmt(r['gradient_deg']),fmt(r['static_area_km2']),fmt(r['azimuth_deg'],1),ROUTES[r['route_family']],fmt(r['base_temp_bias_c']),fmt(r['q_mean_1500_2000_c']),fmt(r['melt_f']),fmt(r['prcp_fac']),f"{r['glen_a']:.2e}",r['previous_reliability_label']] for r in group]))
        sections.append('<p>q是已选的历史附加校正，不是古温度；降水因子不是实际降水量。时变分支的q均值掩盖时间结构。同像元内降水因子相同，无法用此对照识别它的作用。</p></details>')
    sections.append('<h2 id="pairs">6. 全部27组配对，不剔除不一致结果</h2><p>每组提供两项世纪差值、首次持续差异、36种定义中检测次数，以及已有单端备选的条件范围。检测频率不是事件概率；未检测也不表示曲线相同。</p>')
    body=[]
    for p in pairs:
        row=[f'<a href="#pixel_{p["pixel"].replace(":","_")}">{p["a"]} → {p["b"]}</a>',p['pixel'],'同分支恒定' if p['same_route_constant'] else ('同分支时变' if p['same_route'] else '跨分支'),fmt(p['differences']['terminus_elevation_m'],0),fmt(p['differences']['gradient_deg']),fmt(p['area_ratio'])]
        for key in METRICS:
            d=p['metrics'][key];s=p['definition_sensitivity'][key];c=p['conditional'][key]
            span=s['first_start_bracket_span']
            row.extend([fmt(d['late_contrast_pp']),event_text(d['episodes']),f"{s['detected']}/36"+('；首段范围'+str(span) if span else ''),f"{c['nonselected_replacements']}次；[{c['late_min_pp']:.2f}, {c['late_max_pp']:.2f}]" if c['nonselected_replacements'] else '未检验'])
        body.append(row)
    sections.append(table(['低→高末端','像元','配对类型','Δ高程m','Δ坡度°','面积比','长度ΔD pp','长度首次持续差异','长度定义敏感性','长度备选范围pp','体积ΔD pp','体积首次持续差异','体积定义敏感性','体积备选范围pp'],body))
    sections.append('<details><summary>22条没有同像元对象的冰川（没有从总样本删除）</summary><p>'+', '.join(str(x) for x in summary['without_peer_ids'])+'</p></details>')
    sections.append('<h2 id="limits">解释边界与复核</h2><ul><li>早期与现代背景完整展示，但无1470年前推断；1500年不代表初值影响已消失，1900年不是独立定年的LIA终点。</li><li>模型长度采用既有各自算子，不能把两条冰川的绝对长度差解释为地理末端迁移；归一化并不能消除算子差异。</li><li>备选只覆盖部分冰川，单次仅替换一个端点；条件范围不是所有合理参数的联合不确定性。</li><li>朝向按圆周最短角距比较，不代表实测辐射。关联不是地形、温度或模型方法的因果效应。</li><li>持续差异阈值是新设的描述规则，不是科学接纳门槛、统计显著性或年代误差。</li><li>参数距离采用全51条的四分位距缩放；基础温偏四分位距为零但并非常数，改用标准差，未将其忽略。此技术修正未改变三个重点配对。</li></ul>')
    files=['methods.json','summary.json','pairs.json','coverage_51.json','aggregates.json','definition_sensitivity.json','alternative_replacements.json','authority_reconciliation_v1.json','source_manifest.json','validation.json','output_manifest.json']
    sections.append('<p>'+ ' · '.join(f'<a href="{f}">{f}</a>' for f in files)+'</p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>51条冰川：同气候像元配对</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;background:#f5f7f9;color:#263947;margin:0}main{max-width:1320px;margin:auto;padding:32px}h1{font-size:32px}h2{margin-top:44px}h3{margin-top:30px}.lead{color:#496b7e;font-size:19px}.note{background:#e6edf2;border-left:5px solid #44748c;padding:18px}img{max-width:100%;height:auto;background:white;margin:12px 0;border:1px solid #e0e6ea}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{border:1px solid #dbe2e7;padding:8px;text-align:left}th{background:#e7edf1}.scroll{overflow:auto}details{background:white;border:1px solid #dde5e9;padding:16px;margin:12px 0}summary{cursor:pointer;font-weight:600}a{color:#236b90}li{margin:6px 0}nav{margin:24px 0}</style><main>'+''.join(sections)+'</main><script>function reveal(){let el=document.getElementById(location.hash.slice(1));if(el&&el.tagName==="DETAILS")el.open=true}addEventListener("hashchange",reveal);reveal();</script></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')


def main():
    if (OUT/'validation.json').exists(): assert read(OUT/'validation.json')['status']!='pass','Finalized report is immutable; create v2 for changes.'
    methods,rows,curves,atlas,groups,cases=load_inputs()
    pairs,spots,scales=metadata_pairs(rows,atlas,groups)
    assert len(pairs)==27 and sum(p['same_route_constant'] for p in pairs)==11
    assert [s['pair_id'] for s in spots]==['2622_2827','1012_1044','2939_2928'],'Zero-IQR repair changed spotlight identities; disclose before proceeding.'
    # Persist the outcome-independent selection before paired curve differences are computed.
    save('pair_selection.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),pairs=pairs,spotlights=spots,parameter_distance_scales=scales,scale_fallback='population standard deviation when nonconstant zero-IQR',methods_sha256=sha(OUT/'methods.json')))
    sensitivity,replacements=analyze(pairs,curves,cases,methods)
    aggregates=aggregate(pairs)
    peer_ids={p[s] for p in pairs for s in ['a','b']}
    coverage=[dict(system=r['system'],pixel=r['climate_pixel'],has_peer=r['system'] in peer_ids,peer_ids=[q['system'] for q in groups[r['climate_pixel']] if q['system']!=r['system']],route=r['route_family']) for r in rows]
    findings=[]
    for a in aggregates:
        if a['cohort']!='same_route_constant' or (a['orientation'],a['metric']) not in [('height','volume_km3'),('slope','length_km')]:continue
        direction='高末端−低末端' if a['orientation']=='height' else '较陡−较缓'
        findings.append(f"同分支恒定子集（7像元）中，{direction}的{NAMES[a['metric']]}19世纪差值中位数为{a['median_late_contrast_pp']:+.2f}个百分点；正/负像元为{a['positive_pixels']}/{a['negative_pixels']}。这是相对17世纪参考的曲线差异，不是孤立的地形效应。")
    findings.append('同像元配对仍有温度校正、融化因子和冰流参数差异；重点配对按元数据选择，不按曲线结果挑选。')
    for key in METRICS:
        strict=[p for p in pairs if p['same_route_constant']]
        n=sum(bool(p['metrics'][key]['episodes']) for p in strict)
        findings.append(f"11组同分支恒定配对中，{NAMES[key]}有{n}组在1700—1900内达到2个百分点且持续30年的描述标准；需同时查看36种定义及无检测情况，不能据此给出唯一分化年代。")
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),systems=51,with_peer=len(peer_ids),without_peer_ids=sorted(set(curves)-peer_ids),pixels=11,pairs=27,
        same_route_pairs=sum(p['same_route'] for p in pairs),same_route_constant_pairs=11,same_route_constant_pixels=len({p['pixel'] for p in pairs if p['same_route_constant']}),
        spotlights=spots,sensitivity_records=len(sensitivity),alternative_records=len(replacements),saved_cases_used=sum(map(len,cases.values())),
        paired_systems_with_alternatives=sum(any(not c['selected'] for c in cs) for cs in cases.values()),findings=findings,
        simulations_launched=0,raw_climate_access=False,formal_acceptance_unchanged=True)
    summary['spotlight_interpretations']={
        '2622_2827':'高程相差1277m、坡度仅差0.65°。18世纪起相对演变明显分离，36种定义均检测到长度和体积持续差异，首次区间总体为1700—1740。但基础温偏差4.04℃、附加q差1.38℃、冰流参数比2.13，且无已审查备选曲线，不能把分离单独归因于末端高程。',
        '1012_1044':'这是重要的对照结果：坡度相差4.73°，但1700年之后两条归一化曲线很接近，长度与体积均未达到主持续差异标准。较大的长度差异主要在1500—1600出现；其早期差异还受1012已有备选影响。融化因子相差约1.77倍，不能据此认定坡度无作用。',
        '2939_2928':'参数距离在预设候选中最小，不代表参数相同：融化因子比仍为1.54。长度差异反复变号，主标准仅在约1840—1850进入持续正差异，约1890—1900退出；变更参考期或阈值后，首次区间可落在1700—1860，方向也会改变。体积未达到主标准，不能用一个分化年份概括两项指标。'}
    save('coverage_51.json',coverage);save('pairs.json',pairs);save('aggregates.json',aggregates)
    save('definition_sensitivity.json',sensitivity);save('alternative_replacements.json',replacements);save('summary.json',summary)
    figures(rows,pairs,spots,curves,groups,aggregates);report(rows,pairs,spots,groups,aggregates,summary)
    for path,digest in SOURCES.items(): assert sha(ROOT/path)==digest,path
    save('source_manifest.json',SOURCES)
    save('validation.json',dict(status='built-independent-verification-pending',source_count=len(SOURCES),authority_hashes={p:SOURCES[p] for p in methods['authority']},initial_authority_hashes=methods['authority'],systems=51,pairs=len(pairs),definition_records=len(sensitivity),simulations_launched=0,raw_climate_access=False))
    save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__': main()
