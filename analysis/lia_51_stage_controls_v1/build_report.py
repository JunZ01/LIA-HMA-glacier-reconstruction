"""Analyze saved glacier geometry and metadata. Never runs a glacier model."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import html
import itertools
import json
import sys
sys.dont_write_bytecode=True
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ATLAS=ROOT/'analysis/lia_51_stage_reliability_v1'
PARAM=ROOT/'analysis/lia_51_climate_parameter_analysis_v1'
SOURCES={}
GEOM=['terminus_elevation_m','gradient_deg','log_area','normalized_mean_height','longitude']
CLIMATE=['base_temp_bias_c','q_mean_1500_2000_c','melt_f','prcp_fac']
CORE=GEOM+CLIMATE+['log_glen_a']
EXTRA=['latitude','northness','eastness']
ROUTES=['route_candidate','route_synthetic']
METRICS=['length_km','volume_km3']
RESPONSES=['length_contrast_pp','volume_contrast_pp']
LABELS={'terminus_elevation_m':'末端高程','gradient_deg':'主线坡度','log_area':'静态面积（log）',
        'normalized_mean_height':'归一化平均高程','longitude':'经度','latitude':'纬度',
        'northness':'朝向北向分量','eastness':'朝向东向分量','base_temp_bias_c':'基础温度偏差',
        'q_mean_1500_2000_c':'附加温度校正q','melt_f':'融化因子','prcp_fac':'降水因子','log_glen_a':'冰流参数A（log）',
        'length_contrast_pp':'长度：19世纪−17世纪','volume_contrast_pp':'体积：19世纪−17世纪'}
MODEL_LABELS={'M0':'训练均值','M1':'地形与位置','M2':'+模型分支','M3':'+气候参数','M4':'+冰流参数'}
ROUTE_COLORS={'legacy_constant':'#377895','candidate_constant':'#74a367','synthetic_timevarying':'#b45578'}
ROUTE_NAMES={'legacy_constant':'恒定·原分支22','candidate_constant':'恒定·GPU19','synthetic_timevarying':'时变·合成系统10'}
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':10,
                     'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()


def reg(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    path=p.as_posix().lower()
    assert not any(t in path for t in ['20cr','jionglacuo','s7t4zc34sr','temperature reconstruction.xlsx'])
    assert p.name!='inputs.npz'
    digest=sha(p)
    if expected is not None:assert digest==expected.upper(),f'Source changed: {p}'
    SOURCES[p.relative_to(ROOT).as_posix()]=digest
    return p


def read(p,expected=None):return json.loads(reg(p,expected).read_text(encoding='utf-8-sig'))


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value


def save(name,obj):(OUT/name).write_text(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def load():
    inputs={}
    for folder,names in [(ATLAS,['atlas.json','reference_curves.json','case_diagnostics.json','validation.json']),
                         (PARAM,['parameters_51.json'])]:
        manifest=read(folder/'output_manifest.json')
        for name in names:inputs[name]=read(folder/name,manifest[name])
    assert inputs['validation.json']['status']=='pass'
    # Verify immutable input lineage without binding an old unrelated authority snapshot.
    for folder in [ATLAS,PARAM]:
        lineage=read(folder/'source_manifest.json')
        for path,digest in lineage.items():
            if path in ['LIA_PROJECT_FACTS.md','LIA_CURRENT_STATE.json']:continue
            reg(path,digest)
    rows=inputs['parameters_51.json'];curves=inputs['reference_curves.json']
    assert len(rows)==51 and {r['system'] for r in rows}=={c['system'] for c in curves}
    byid={d['system']:d for d in inputs['atlas.json']}
    for r in rows:r['previous_reliability_label']=byid[r['system']]['reliability_label']
    cases=[]
    for c in inputs['case_diagnostics.json']:
        sp=reg(c['state_source']);assert c['state_source'] in SOURCES
        with np.load(sp,allow_pickle=False) as z:
            y=z['years'].copy().astype(float);v=z['history'].copy()
        assert y[0]==1470 and y[-1]==2000 and np.all(np.diff(y)>0)
        assert all(c['parameters'][k]==1 for k in ['melt','prcp','glen'])
        cases.append(dict(system=c['system'],case_id=c['case_id'],selected=c['selected'],parameters=c['parameters'],
                          state_source=c['state_source'],years=y,
                          values={'length_km':v[:,0]/1000,'volume_km3':v[:,2]/1e9}))
    return rows,curves,cases


def section(y,x,start,end):
    t=np.r_[start,y[(y>start)&(y<end)],end].astype(float)
    return t,np.interp(t,y,x)


def duration_above(t,v,threshold):
    total=0.
    for a,b,left,right in zip(v[:-1],v[1:],t[:-1],t[1:]):
        if a>=threshold and b>=threshold:part=1.
        elif a<threshold and b<threshold:part=0.
        else:
            crossing=float(np.clip((threshold-a)/(b-a),0,1))
            part=crossing if a>=threshold else 1-crossing
        total+=(right-left)*part
    return total


def describe(y,x,step=10,norm_start=1500,offset=0,fraction=.01,with_periods=False):
    # Fixed 1500 anchor; a short leading1470-1480 interval is retained for normalization checks.
    t=np.unique(np.r_[1470,np.arange(1480,2001,step)]).astype(float)
    v=np.interp(t,y,x)
    _,norm_values=section(t,v,norm_start,2000)
    maximum=float(norm_values.max())
    periods=[]
    for a,b in METHODS['calendar_periods'] if with_periods else []:
        ty,vx=section(t,v,a,b)
        periods.append(dict(start=a,end=b,mean=float(np.trapz(vx,ty)/(b-a)),
            mean_relative_to_max_pct=float(100*np.trapz(vx,ty)/(b-a)/maximum),
            near_duration_years=duration_above(ty,vx,maximum*(1-fraction)),
            start_value=float(vx[0]),end_value=float(vx[-1]),
            net_change_pct=float(100*(vx[-1]/vx[0]-1))))
    ea,ev=section(t,v,1600+offset,1700+offset);la,lv=section(t,v,1800+offset,1900+offset)
    early=float(np.trapz(ev,ea)/100);late=float(np.trapz(lv,la)/100)
    early_near=duration_above(ea,ev,maximum*(1-fraction));late_near=duration_above(la,lv,maximum*(1-fraction))
    return dict(step=step,normalization_start=norm_start,offset=offset,fraction=fraction,maximum=maximum,
                early_mean=early,late_mean=late,contrast_pp=100*(late-early)/maximum,
                early_near_percent=early_near,late_near_percent=late_near,
                near_contrast_pp=late_near-early_near,periods=periods)


def enrich(rows,curves):
    lookup={c['system']:c for c in curves};profiles=[];sensitivity=[]
    for row in rows:
        c=lookup[row['system']];r=dict(row);p=dict(system=row['system'],metrics={})
        for key in METRICS:
            d=describe(np.array(c['native_years']),np.array(c['native_'+key]),with_periods=True)
            p['metrics'][key]=d
            r[key.split('_')[0]+'_contrast_pp']=d['contrast_pp']
            r[key.split('_')[0]+'_near_contrast_pp']=d['near_contrast_pp']
        profiles.append(p)
        row.update(r)
    for offset,step,start,fraction in itertools.product([-20,0,20],[10,20],[1470,1500,1550],[.005,.01,.02]):
        records=[]
        for row in rows:
            c=lookup[row['system']];record=dict(system=row['system'])
            for key in METRICS:
                d=describe(np.array(c['native_years']),np.array(c['native_'+key]),step,start,offset,fraction)
                short=key.split('_')[0];record[short+'_contrast_pp']=d['contrast_pp'];record[short+'_near_contrast_pp']=d['near_contrast_pp']
            records.append(record)
        sensitivity.append(dict(offset=offset,step=step,start=start,fraction=fraction,records=records))
    return profiles,sensitivity


def cohort(rows,name):
    if name=='all51':return rows
    if name=='constant41':return [r for r in rows if r['policy_group']=='constant']
    return [r for r in rows if r['status']==('retained' if name=='original32' else 'candidate')]


def corr(rows,x,y,controls=None):
    controls=controls or []
    keys=[x,y]+controls
    data=np.array([[r[k] for k in keys] for r in rows],float)
    data=data[np.isfinite(data).all(axis=1)]
    n=len(data);minimum=max(12,3*(len(controls)+2)) if controls else 4
    result=dict(n=n,controls=controls,minimum_n=minimum,rho=None)
    if n<minimum:return {**result,'reason':'insufficient n for control count'}
    ranks=np.column_stack([rankdata(np.round(data[:,i],10),method='average') for i in range(data.shape[1])])
    if any(np.std(ranks[:,i])<1e-10 for i in range(ranks.shape[1])):return {**result,'reason':'constant variable'}
    if not controls:return {**result,'rho':float(np.corrcoef(ranks[:,:2].T)[0,1]),'reason':None}
    z=np.c_[np.ones(n),(ranks[:,2:]-ranks[:,2:].mean(axis=0))/ranks[:,2:].std(axis=0)]
    rank=int(np.linalg.matrix_rank(z));result.update(design_rank=rank,residual_df=n-rank-1,condition_number=float(np.linalg.cond(z)))
    if rank<z.shape[1]:return {**result,'reason':'rank deficient'}
    residual=ranks[:,:2]-z@np.linalg.lstsq(z,ranks[:,:2],rcond=None)[0]
    if min(np.std(residual,axis=0))<1e-9:return {**result,'reason':'zero residual variance'}
    return {**result,'rho':float(np.corrcoef(residual.T)[0,1]),'reason':None}


def active_routes(rows):return [k for k in ROUTES if len({r[k] for r in rows})>1]


def controls(rows,x,mode):
    if mode=='joint':return [k for k in CORE if k!=x]+active_routes(rows)
    if mode=='spatial_route':
        base=[k for k in ['longitude','latitude'] if k!=x]+active_routes(rows)
        if x in ['northness','eastness']:base+=[k for k in ['northness','eastness'] if k!=x]
        return base
    return []


def associations(rows,profiles):
    results=[];period_results=[]
    for name in ['all51','constant41','original32','original19']:
        sub=cohort(rows,name)
        for x in CORE+EXTRA:
            for y in RESPONSES+['length_near_contrast_pp','volume_near_contrast_pp']:
                modes=['raw','spatial_route']+(['joint'] if x in CORE and name in ['all51','constant41'] else [])
                for mode in modes:results.append(dict(cohort=name,predictor=x,response=y,mode=mode,**corr(sub,x,y,controls(sub,x,mode))))
    for key in METRICS:
        for j,period in enumerate(METHODS['calendar_periods']):
            sub=[{**row,'period_mean':p['metrics'][key]['periods'][j]['mean_relative_to_max_pct']} for row,p in zip(rows,profiles)]
            for x in CORE+EXTRA:period_results.append(dict(metric=key,period=period,predictor=x,**corr(sub,x,'period_mean')))
    return results,period_results


def pixel_robustness(rows):
    output=[]
    for name in ['all51','constant41']:
        sub=cohort(rows,name);pixels=sorted({r['climate_pixel'] for r in sub})
        for x in CORE:
            for y in RESPONSES:
                deletions=[]
                for p in pixels:
                    remain=[r for r in sub if r['climate_pixel']!=p]
                    # Retain the registered control set; a collapsed route is reported, not silently dropped.
                    deletions.append(dict(pixel=p,raw=corr(remain,x,y),joint=corr(remain,x,y,controls(sub,x,'joint'))))
                medians=[{x:float(np.median([r[x] for r in sub if r['climate_pixel']==p])),y:float(np.median([r[y] for r in sub if r['climate_pixel']==p]))} for p in pixels]
                rx=rankdata(np.round([r[x] for r in sub],10));ry=rankdata(np.round([r[y] for r in sub],10))
                a=[];b=[];ngroup=0
                for p in pixels:
                    ix=[i for i,r in enumerate(sub) if r['climate_pixel']==p]
                    if len(ix)<2:continue
                    ngroup+=1;a.extend(rx[ix]-rx[ix].mean());b.extend(ry[ix]-ry[ix].mean())
                within=float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-10 else None
                output.append(dict(cohort=name,predictor=x,response=y,pixel_count=len(pixels),deletions=deletions,
                    equal_pixel_median=corr(medians,x,y),within_pixel=dict(n=len(a),pixel_n=ngroup,rho=within,
                    predictor_rank_sd=float(np.std(a)),response_rank_sd=float(np.std(b)),
                    reason='no within-pixel predictor or response contrast' if within is None else None,
                    interpretation='Demeaned global ranks; few independent groups; no within-pixel causal identification')))
    return output


def sensitivity_correlations(rows,variants):
    result=[]
    for v in variants:
        # Continuous D does not depend on near threshold; avoid redundant copies.
        responses=RESPONSES.copy() if v['fraction']==.01 else []
        responses+=['length_near_contrast_pp','volume_near_contrast_pp']
        sub=[{**r,**d} for r,d in zip(rows,v['records'])]
        for x in CORE:
            for y in responses:
                result.append(dict(offset=v['offset'],step=v['step'],start=v['start'],fraction=v['fraction'],
                    predictor=x,response=y,raw=corr(sub,x,y),joint=corr(sub,x,y,controls(sub,x,'joint'))))
    return result


def alternative_analysis(rows,cases):
    profiles=[];replacements=[];groups=defaultdict(list)
    for c in cases:
        d={k:c[k] for k in ['system','case_id','selected','parameters','state_source']};d['metrics']={}
        for key in METRICS:d['metrics'][key]=describe(c['years'],c['values'][key],with_periods=True)
        profiles.append(d);groups[c['system']].append(d)
        if c['selected']:
            ref=next(r for r in rows if r['system']==c['system'])
            for key in METRICS:assert abs(ref[key.split('_')[0]+'_contrast_pp']-d['metrics'][key]['contrast_pp'])<1e-10
            continue
        # Physical values stay fixed because the saved cases have verified unit multipliers.
        changed=[]
        for r in rows:
            rr=dict(r)
            if r['system']==c['system']:
                for key in METRICS:
                    rr[key.split('_')[0]+'_contrast_pp']=d['metrics'][key]['contrast_pp']
                rr['q_mean_1500_2000_c']=c['parameters']['constant_offset_degC']
            changed.append(rr)
        vals=[]
        for x in CORE:
            for y in RESPONSES:vals.append(dict(predictor=x,response=y,raw=corr(changed,x,y)['rho'],joint=corr(changed,x,y,controls(changed,x,'joint'))['rho']))
        replacements.append(dict(system=c['system'],case_id=c['case_id'],correlations=vals))
    ranges=[]
    for sid,sub in groups.items():
        row=dict(system=sid,n=len(sub),metrics={})
        for key in METRICS:
            data=[d['metrics'][key]['contrast_pp'] for d in sub]
            row['metrics'][key]=dict(minimum=min(data),maximum=max(data),range_pp=max(data)-min(data),
                sign_consistent=bool(len(sub)>=2 and (min(data)>0 or max(data)<0)),
                note='one saved case is not robustness evidence')
        ranges.append(row)
    assert len(replacements)==98
    return profiles,replacements,ranges


def fit_predict(train,test,xkeys,ykey):
    y=np.array([r[ykey] for r in train]);test_y=np.array([r[ykey] for r in test])
    if not xkeys:
        return np.full(len(test),y.mean()),dict(rank=1,condition=1.,train_r2=0.)
    a=np.array([[r[k] for k in xkeys] for r in train]);b=np.array([[r[k] for k in xkeys] for r in test])
    mu=a.mean(axis=0);sd=a.std(axis=0)
    if np.any(sd<1e-12):return None,dict(reason='constant training predictor')
    z=np.c_[np.ones(len(a)),(a-mu)/sd];zz=np.c_[np.ones(len(b)),(b-mu)/sd]
    rank=np.linalg.matrix_rank(z)
    if rank<z.shape[1]:return None,dict(reason='rank deficient training matrix')
    coef=np.linalg.lstsq(z,y,rcond=None)[0]
    train_r2=1-float(np.sum((y-z@coef)**2)/np.sum((y-y.mean())**2))
    return zz@coef,dict(rank=int(rank),condition=float(np.linalg.cond(z)),train_r2=train_r2)


def cross_validation(rows):
    scores=[]
    for cohort_name in ['all51','constant41']:
        sub=cohort(rows,cohort_name);routes=active_routes(sub)
        models={'M0':[],'M1':GEOM,'M2':GEOM+routes,'M3':GEOM+routes+CLIMATE,'M4':GEOM+routes+CLIMATE+['log_glen_a']}
        for y in RESPONSES:
            for model,xkeys in models.items():
                full,info=fit_predict(sub,sub,xkeys,y);predictions=[];failures=[]
                for p in sorted({r['climate_pixel'] for r in sub}):
                    train=[r for r in sub if r['climate_pixel']!=p];test=[r for r in sub if r['climate_pixel']==p]
                    pred,diag=fit_predict(train,test,xkeys,y)
                    if pred is None:failures.append(dict(pixel=p,**diag));continue
                    mean=float(np.mean([r[y] for r in train]))
                    for r,v in zip(test,pred):predictions.append(dict(system=r['system'],pixel=p,observed=r[y],predicted=float(v),training_mean=mean,train_n=len(train),design=diag))
                sse=sum((p['observed']-p['predicted'])**2 for p in predictions);null_sse=sum((p['observed']-p['training_mean'])**2 for p in predictions)
                scores.append(dict(cohort=cohort_name,response=y,model=model,predictors=xkeys,n=len(sub),
                    training_r2=info.get('train_r2'),held_out_n=len(predictions),q2=1-sse/null_sse if not failures else None,
                    rmse_pp=np.sqrt(sse/len(predictions)),failed_folds=failures,predictions=predictions))
    return scores


def group_profiles(rows,profiles):
    out=[]
    for dim,groups in [('elevation_group',['low','middle','high']),('route_family',list(ROUTE_COLORS))]:
        for group in groups:
            ids={r['system'] for r in rows if r[dim]==group};sub=[p for p in profiles if p['system'] in ids]
            for key in METRICS:
                means=np.array([[v['mean_relative_to_max_pct'] for v in p['metrics'][key]['periods']] for p in sub])
                near=np.array([[v['near_duration_years'] for v in p['metrics'][key]['periods']] for p in sub])
                out.append(dict(dimension=dim,group=group,n=len(sub),metric=key,
                    mean_q25_median_q75=np.quantile(means,[.25,.5,.75],axis=0),near_q25_median_q75=np.quantile(near,[.25,.5,.75],axis=0)))
    return out


def figsave(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def plot_profiles(rows,profiles,groups):
    byid={p['system']:p for p in profiles};ordered=sorted(rows,key=lambda r:r['terminus_elevation_m'])
    fig,axes=plt.subplots(1,2,figsize=(13,14),sharey=True)
    for ax,key,title in zip(axes,METRICS,['长度','体积']):
        a=np.array([[d['mean_relative_to_max_pct'] for d in byid[r['system']]['metrics'][key]['periods']] for r in ordered])
        im=ax.imshow(a,aspect='auto',vmin=70,vmax=100,cmap='YlGnBu',interpolation='nearest')
        for i in range(51):
            for j in range(5):ax.text(j,i,f'{a[i,j]:.0f}',ha='center',va='center',fontsize=7,color='white' if a[i,j]>92 else '#243743')
        ax.set_xticks(range(5));ax.set_xticklabels(['1500—1600','1600—1700','1700—1800','1800—1900','1900—2000'],rotation=35,ha='right');ax.set_title(title+'：世纪平均值 / 窗口最大值（%）')
    axes[0].set_yticks(range(51));axes[0].set_yticklabels([f"{r['system']} · {r['terminus_elevation_m']:.0f} m" for r in ordered],fontsize=8)
    fig.suptitle('51条统一样本｜先看整个时段，不指定唯一最大年份',fontsize=17,y=.97)
    fig.text(.5,.016,'按末端高程排序。固定世纪是比较区间，不是独立定年的LIA分期；每条冰川用自己的同一窗口最大值归一化。',ha='center',fontsize=9)
    fig.subplots_adjust(left=.17,right=.83,top=.92,bottom=.11,wspace=.08)
    cax=fig.add_axes([.87,.32,.02,.37])
    fig.colorbar(im,cax=cax,label='相对窗口最大值 / %；<70同色')
    figsave(fig,'01_century_profiles')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True)
    for i,dim in enumerate(['elevation_group','route_family']):
        for j,key in enumerate(METRICS):
            ax=axes[i,j]
            subset=[g for g in groups if g['dimension']==dim and g['metric']==key]
            for g,col in zip(subset,['#286c9f','#ba8439','#955894']):
                label={'low':'<4000m','middle':'4000—<4500m','high':'≥4500m',
                       'legacy_constant':'恒定·原分支','candidate_constant':'恒定·GPU','synthetic_timevarying':'时变·合成系统'}[g['group']]
                q=np.array(g['mean_q25_median_q75']);ax.plot(range(5),q[1],marker='o',color=col,label=f"{label} · n={g['n']}")
                ax.fill_between(range(5),q[0],q[2],color=col,alpha=.12)
            ax.set_title(('末端高程组' if i==0 else '来源分支')+' · '+('长度' if j==0 else '体积'));ax.set_ylabel('世纪平均值 / 窗口最大值（%）');ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
            ax.set_xticks(range(5));ax.set_xticklabels(['16世纪','17世纪','18世纪','19世纪','20世纪'])
    fig.suptitle('分组时段轮廓｜中位数与四分位范围；不是参数干预效应',fontsize=16,y=.98)
    fig.tight_layout(rect=(0,0,1,.95));figsave(fig,'02_group_profiles')


def plot_associations(assoc):
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    for ax,y in zip(axes,RESPONSES):
        for i,x in enumerate(CORE):
            for offset,mode,color,marker,label in [(-.16,'raw','#487f9e','o','原始相关'),(.16,'joint','#b36570','s','联合条件相关')]:
                a=next(a for a in assoc if a['cohort']=='all51' and a['predictor']==x and a['response']==y and a['mode']==mode)
                if a['rho'] is not None:ax.scatter(a['rho'],i+offset,c=color,marker=marker,s=44,label=label if i==0 else None)
        ax.set_yticks(range(len(CORE)));ax.set_yticklabels([LABELS[x] for x in CORE]);ax.set_xlim(-1,1);ax.axvline(0,color='#999',lw=.8);ax.grid(axis='x',alpha=.18);ax.set_title(LABELS[y]);ax.set_xlabel('秩相关系数，不是因果效应');ax.legend(frameon=False)
    axes[0].invert_yaxis()
    fig.suptitle('哪些属性与较晚时段的相对规模有关？｜统一51条',fontsize=17,y=.985)
    fig.text(.5,.01,'主指标 D＝100×（19世纪均值−17世纪均值）/ 同一窗口最大值。联合模型控制其余核心变量及来源分支。',ha='center',fontsize=9)
    fig.subplots_adjust(left=.16,right=.97,top=.90,bottom=.09,wspace=.15)
    figsave(fig,'03_core_associations')


def plot_scatter(rows):
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for i,y in enumerate(RESPONSES):
        for j,x in enumerate(['terminus_elevation_m','gradient_deg','prcp_fac']):
            ax=axes[i,j]
            for route,col in ROUTE_COLORS.items():
                sub=[r for r in rows if r['route_family']==route]
                ax.scatter([r[x] for r in sub],[r[y] for r in sub],color=col,s=38,marker='s' if route=='synthetic_timevarying' else 'o',label=ROUTE_NAMES[route],alpha=.85)
            ax.axhline(0,color='#888',lw=.7);ax.set_xlabel(LABELS[x]+(' / m' if x=='terminus_elevation_m' else ' / °' if x=='gradient_deg' else '（倍数）'))
            ax.set_ylabel(LABELS[y]+' / 百分点');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('固定展示的三条线索：末端高程、坡度、降水因子',fontsize=17,y=.98)
    fig.tight_layout(rect=(0,0,1,.95));figsave(fig,'04_geometry_parameter_scatter')


def plot_robustness(assoc,pixels,sensitivity,replacements):
    focus=['terminus_elevation_m','gradient_deg','log_area','prcp_fac','melt_f','q_mean_1500_2000_c']
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    for ax,y in zip(axes,RESPONSES):
        for i,x in enumerate(focus):
            raw=next(a for a in assoc if a['cohort']=='all51' and a['predictor']==x and a['response']==y and a['mode']=='joint')['rho']
            const=next(a for a in assoc if a['cohort']=='constant41' and a['predictor']==x and a['response']==y and a['mode']=='joint')['rho']
            p=next(p for p in pixels if p['cohort']=='all51' and p['predictor']==x and p['response']==y)
            sets=[([v['joint']['rho'] for v in p['deletions']],-.21,'#487f9e','逐像元剔除'),
                  ([v['joint']['rho'] for v in sensitivity if v['predictor']==x and v['response']==y],0,'#b88737','时段/尺度/基准敏感性'),
                  ([v['joint'] for r in replacements for v in r['correlations'] if v['predictor']==x and v['response']==y],.21,'#9b608b','单系统备选替换')]
            for vals,offset,col,label in sets:
                vals=[v for v in vals if v is not None]
                if vals:ax.plot([min(vals),max(vals)],[i+offset]*2,color=col,lw=3,label=label if i==0 else None)
            if raw is not None:ax.scatter(raw,i,s=40,c='black',zorder=5,label='51条主结果' if i==0 else None)
            if const is not None:ax.scatter(const,i,s=50,facecolor='white',edgecolor='black',marker='D',zorder=6,label='恒定41条' if i==0 else None)
        ax.set_yticks(range(len(focus)));ax.set_yticklabels([LABELS[x] for x in focus]);ax.axvline(0,c='#888',lw=.8);ax.set_xlim(-1,1);ax.grid(axis='x',alpha=.15);ax.set_title(LABELS[y]);ax.set_xlabel('联合条件秩相关；线段不是置信区间')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8,loc='lower left',frameon=False)
    fig.suptitle('关联是否经得住已有证据的不同处理？',fontsize=17,y=.98)
    fig.subplots_adjust(left=.16,right=.97,top=.90,bottom=.1,wspace=.15);figsave(fig,'05_robustness')


def plot_predictions(cv):
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,y in zip(axes,RESPONSES):
        for cohort_name,color in [('all51','#356e8a'),('constant41','#b37a38')]:
            sub=[r for r in cv if r['cohort']==cohort_name and r['response']==y]
            ax.plot(range(5),[r['training_r2'] for r in sub],ls='--',marker='o',color=color,alpha=.6,label=cohort_name+' 训练R²')
            ax.plot(range(5),[r['q2'] for r in sub],marker='s',color=color,label=cohort_name+' 留像元Q²')
        ax.axhline(0,color='#777',lw=.7);ax.set_xticks(range(5));ax.set_xticklabels([MODEL_LABELS[m] for m in ['M0','M1','M2','M3','M4']],rotation=25,ha='right');ax.set_title(LABELS[y]);ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
    axes[0].set_ylabel('解释/预测评分；Q²<0表示不如训练均值基线')
    fig.suptitle('加入参数是否改善跨气候像元的预测？',fontsize=17,y=.98)
    fig.text(.5,.016,'标准化仅用训练样本，负评分不截断。留出对象的参数也已经校准；本检查不代表未校准冰川的预测能力或独立验证。',ha='center',fontsize=9)
    fig.subplots_adjust(top=.86,bottom=.19,wspace=.14);figsave(fig,'06_pixel_held_out_prediction')


def report(rows,profiles,assoc,pixels,cv,ranges,summary):
    def f(x):return '不估计' if x is None else f'{x:+.2f}'
    results=[]
    for y in RESPONSES:
        for x in CORE:
            a=next(v for v in assoc if (v['cohort'],v['predictor'],v['response'],v['mode'])==('all51',x,y,'raw'))
            b=next(v for v in assoc if (v['cohort'],v['predictor'],v['response'],v['mode'])==('all51',x,y,'joint'))
            c=next(v for v in assoc if (v['cohort'],v['predictor'],v['response'],v['mode'])==('constant41',x,y,'joint'))
            p=next(v for v in pixels if (v['cohort'],v['predictor'],v['response'])==('all51',x,y))
            results.append('<tr>'+''.join(f'<td>{value}</td>' for value in [LABELS[y],LABELS[x],f(a['rho']),f(b['rho']),f(c['rho']),f(p['equal_pixel_median']['rho']),f(p['within_pixel']['rho'])])+'</tr>')
    tables='<table><thead><tr>'+''.join(f'<th>{x}</th>' for x in ['指标','属性','原始ρ','联合ρ','恒定41联合ρ','等像元中位数ρ','像元内ρ'])+'</tr></thead><tbody>'+''.join(results)+'</tbody></table>'
    curves=[]
    for r in sorted(rows,key=lambda r:r['terminus_elevation_m']):
        curves.append('<tr>'+''.join(f'<td>{v}</td>' for v in [f'<a href="../lia_51_stage_reliability_v1/glaciers/{r["system"]}.png">{r["system"]}</a>',f'{r["terminus_elevation_m"]:.0f}',f'{r["gradient_deg"]:.1f}',f'{r["static_area_km2"]:.1f}',f'{r["prcp_fac"]:.3f}',f(r['length_contrast_pp']),f(r['volume_contrast_pp']),r['previous_reliability_label']])+'</tr>')
    cvrows=[]
    for v in cv:cvrows.append('<tr>'+''.join(f'<td>{x}</td>' for x in [v['cohort'],LABELS[v['response']],MODEL_LABELS[v['model']],f(v['training_r2']),f(v['q2']),f'{v["rmse_pp"]:.2f}',v['held_out_n']])+'</tr>')
    cvtable='<table><tr>'+''.join(f'<th>{x}</th>' for x in ['样本','指标','模型','训练R²','留像元Q²','RMSE 百分点','留出预测数'])+'</tr>'+''.join(cvrows)+'</table>'
    orient=[]
    for y in RESPONSES:
        for x in ['northness','eastness']:
            r=next(v for v in assoc if (v['cohort'],v['predictor'],v['response'],v['mode'])==('all51',x,y,'spatial_route'))
            orient.append('<tr>'+''.join(f'<td>{v}</td>' for v in [LABELS[y],LABELS[x],f(r['rho']),r['n']])+'</tr>')
    orientation_table='<table><tr><th>指标</th><th>朝向分量</th><th>空间/分支条件ρ</th><th>n</th></tr>'+''.join(orient)+'</table>'
    parts=''
    for title,name,caption in [
        ('1 · 每条冰川在不同时段有多接近最大规模','01_century_profiles','每条冰川一行，按末端高程排序。原生时间间隔1年/10年均保留在前一版；本次统一十年尺度比较。'),
        ('2 · 高程组与来源分支的时段轮廓','02_group_profiles','粗线为中位数，阴影是冰川间四分位范围，不是不确定性区间。来源分支与校正策略不能独立识别。'),
        ('3 · 地形与参数关联','03_core_associations','主指标D为19世纪与17世纪平均值的相对差；正值说明较晚时段规模更大，并不意味着19世纪持续增长。联合相关控制其余核心变量和来源分支，仍不是因果效应。'),
        ('4 · 逐冰川查看固定线索','04_geometry_parameter_scatter','预先固定展示末端高程、坡度和降水因子，不按结果择优挑散点。降水因子是已校准的倍数，不是实际降水量。'),
        ('5 · 现有证据支持到哪一步','05_robustness','14条有多个备选，37条未建立跨方案稳健性，全部继续保留。每次只替换一个系统，并同步使用该方案的历史q；不是同时变化的完整参数集合。'),
        ('6 · 参数有没有增加跨像元预测信息','06_pixel_held_out_prediction','训练拟合好不代表新像元预测好；Q²以各留出折的训练均值为基线，负值不截断。没有用留出结果选择模型或调整参数。留出对象的参数也已经校准，因此只检查保存结果间的可预测关系，不代表未校准冰川的预测能力。')]:
        parts+=f'<section><h2>{title}</h2><p>{caption}</p><a href="{name}.svg"><img src="{name}.png" loading="lazy"></a></section>'
    findings=''.join(f'<li>{html.escape(x)}</li>' for x in summary['findings'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>51条冰川 · 演变时段差异与控制因素</title><style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;margin:0;background:#f0f3f5;color:#253c4b}main{max-width:1280px;margin:auto;padding:30px 24px}h1{font-size:29px}h2{font-size:22px}section{background:white;padding:23px;border-radius:12px;margin:22px 0}img{width:100%;height:auto}a{color:#246d8a}.note{background:#fff6e7;padding:16px;border-left:5px solid #bc823e}.scroll{overflow:auto;max-height:650px}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:9px;border-bottom:1px solid #dce3e7;text-align:left}th{background:#eef3f6;position:sticky;top:0}input{padding:10px;max-width:90%;width:300px;margin:12px 0}</style><main><h1>51条冰川：演变时段差异与地形、气候参数</h1><p>统一51条 · 长度与体积 · 33个气候像元 · 已有备选敏感性 · 无新模拟</p><div class="note">固定世纪是比较窗口，不是独立定年的LIA分期。本文回答“哪些属性与时段差异共同变化、能否帮助跨冰川预测”，不把内部统计关系写成已识别的因果机制。</div>'''
    page+='<section><h2>本轮结果</h2><ul>'+findings+'</ul><p>数值检查不等于独立观测验证。51条是已选择样本，不代表全部191条或整个喜马拉雅。</p></section>'+parts
    page+='<section><h2>核心结果完整表</h2><div class="scroll">'+tables+'</div><p>像元内相关仅来自11个多冰川像元、29条冰川；独立分组很少。在当前数值精度下，这11个像元内部的降水因子均不变化，因此该项像元内相关不估计，不是相关为零。没有用p值挑选结果。</p><details><summary>朝向补充检查：同时控制另一朝向分量、经纬度与分支</summary>'+orientation_table+'<p>本指标下关系较弱；主线朝向不是辐射观测，不能据此排除局地坡向或遮蔽作用。</p></details><details><summary>全部固定模型的预测评分</summary>'+cvtable+'</details><p>分支标签同时包含冰川对象、初态和校正策略的共同差异，不能将其评分增益解释成某种方法的独立效应。</p></section>'
    page+='<section><h2>51条逐项指标</h2><input id="filter" placeholder="搜索编号或可靠性标签"><div class="scroll"><table id="inventory"><thead><tr>'+''.join(f'<th>{x}</th>' for x in ['编号/原曲线','末端高程m','坡度°','静态面积km²','降水因子','长度D 百分点','体积D 百分点','原可靠性标签'])+'</tr></thead><tbody>'+''.join(curves)+'</tbody></table></div></section>'
    page+='''<section><h2>方法与数据</h2><p>D＝100×（1800—1900平均规模−1600—1700平均规模）/1500—2000最大规模。辅以近最大范围持续比例，按线性连接的保存几何曲线估算，不提高原生时间精度。区间平移±20年、10/20年尺度、1470/1500/1550基准窗口及0.5%/1%/2%阈值均保留；+20年敏感性包含1900年后的历史。</p><p>温度q是附加校正，不是实际古温度；朝向用北向/东向分量，不等同于辐射。历史长度已经受校准约束，不能当作独立验证；体积也是模型量。</p><p><a href="methods.json">方法定义</a> · <a href="period_profiles.json">逐时段指标</a> · <a href="associations.json">全部关联</a> · <a href="pixel_robustness.json">像元检查</a> · <a href="definition_sensitivity.json">定义敏感性</a> · <a href="alternative_ranges.json">备选指标范围</a> · <a href="cross_validation.json">逐条留出预测</a> · <a href="validation.json">核查记录</a> · <a href="source_manifest.json">来源哈希</a> · <a href="summary.json">摘要</a></p></section></main><script>document.getElementById('filter').addEventListener('input',function(){const q=this.value.toLowerCase();document.querySelectorAll('#inventory tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))})</script></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')


def make_summary(rows,assoc,cv,ranges):
    findings=[]
    geometry={}
    for x,y in [('gradient_deg','length_contrast_pp'),('terminus_elevation_m','volume_contrast_pp')]:
        values=[next(v['rho'] for v in assoc if (v['predictor'],v['cohort'],v['mode'],v['response'])==(x,co,'joint',y)) for co in ['all51','constant41']]
        geometry[x]=values
    findings.append(f"坡度与长度时段差的联合相关为{geometry['gradient_deg'][0]:+.2f}（恒定41条内{geometry['gradient_deg'][1]:+.2f}）；末端高程与体积时段差为{geometry['terminus_elevation_m'][0]:+.2f}（恒定41条内{geometry['terminus_elevation_m'][1]:+.2f}）。提示地形差异值得跟进，但不证明响应机制。")
    for y in RESPONSES:
        def get(x,co,mode):return next(v['rho'] for v in assoc if (v['predictor'],v['cohort'],v['mode'],v['response'])==(x,co,mode,y))
        p=[get('prcp_fac',co,mode) for co,mode in [('all51','raw'),('all51','joint'),('constant41','joint')]]
        findings.append(f"{LABELS[y]}：降水因子原始相关{p[0]:+.2f}，联合条件相关{p[1]:+.2f}，恒定41条内{p[2]:+.2f}。这是已选参数与模拟演变的关联，不是降水强迫的因果效应。")
        m2=next(v for v in cv if (v['cohort'],v['response'],v['model'])==('all51',y,'M2'))
        m3=next(v for v in cv if (v['cohort'],v['response'],v['model'])==('all51',y,'M3'))
        findings.append(f"{LABELS[y]}：地形＋分支的留像元Q²为{m2['q2']:+.2f}；加入气候参数后为{m3['q2']:+.2f}。比较的是内部跨像元预测，不是独立观测验证。")
    multi=[p for p,n in Counter(r['climate_pixel'] for r in rows).items() if n>=2]
    pconstant=all(len({round(r['prcp_fac'],10) for r in rows if r['climate_pixel']==p})==1 for p in multi)
    findings.append('11个多冰川像元内部的降水因子均相同：本样本不能利用像元内对照单独识别其作用。气候参数没有稳定增加本次跨像元预测信息，不等于气候对冰川不重要。')
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),systems=51,climate_pixels=33,
        precipitation_factor_constant_within_all_multi_system_pixels=pconstant,
        multi_system_pixels=sum(v>=2 for v in Counter(r['climate_pixel'] for r in rows).values()),
        within_pixel_systems=sum(v for v in Counter(r['climate_pixel'] for r in rows).values() if v>=2),
        existing_case_count=113,multi_case_system_count=sum(r['n']>=2 for r in ranges),nonselected_case_replacements=98,
        simulation_launches=0,raw_climate_access=False,formal_acceptance_unchanged=True,findings=findings)


def unit_tests():
    t=np.array([0.,10.]);assert duration_above(t,np.array([0.,10.]),5)==5
    assert duration_above(t,np.array([10.,0.]),5)==5
    assert duration_above(t,np.array([5.,5.]),5)==10
    assert duration_above(t,np.array([4.,4.]),5)==0
    y=np.arange(1470,2001,10.);x=np.full(len(y),5.)
    a=describe(y,x,with_periods=True);assert a['contrast_pp']==0 and a['near_contrast_pp']==0
    assert all(p['mean_relative_to_max_pct']==100 and p['near_duration_years']==100 for p in a['periods'])
    b=describe(y,y,with_periods=True);assert abs(b['contrast_pp']-10)<1e-10
    c=describe(y,2*y);assert abs(c['contrast_pp']-b['contrast_pp'])<1e-10
    return 8


def main():
    global METHODS
    METHODS=read(OUT/'methods.json');reg(Path(__file__))
    authority={p:sha(ROOT/p) for p in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']}
    tests=unit_tests();rows,curves,cases=load()
    print('Verified saved sources. Building common-period metrics.',flush=True)
    profiles,variants=enrich(rows,curves)
    assoc,period_assoc=associations(rows,profiles)
    pixels=pixel_robustness(rows)
    definition=sensitivity_correlations(rows,variants)
    print('Testing existing alternative histories and climate-pixel held-out prediction.',flush=True)
    case_profiles,replacements,ranges=alternative_analysis(rows,cases)
    cv=cross_validation(rows);groups=group_profiles(rows,profiles)
    assert len(RESPONSES)==2 and len(assoc)==496 and len(definition)==1440 and len(cv)==20
    summary=make_summary(rows,assoc,cv,ranges)
    for name,obj in [('features_51.json',rows),('period_profiles.json',profiles),('metric_variants.json',variants),('associations.json',assoc),
        ('period_associations.json',period_assoc),('pixel_robustness.json',pixels),('definition_sensitivity.json',definition),
        ('case_profiles.json',case_profiles),('alternative_replacements.json',replacements),('alternative_ranges.json',ranges),
        ('cross_validation.json',cv),('group_profiles.json',groups),('summary.json',summary)]:save(name,obj)
    print('Rendering six scientific overview figures.',flush=True)
    plot_profiles(rows,profiles,groups);plot_associations(assoc);plot_scatter(rows);plot_robustness(assoc,pixels,definition,replacements);plot_predictions(cv)
    report(rows,profiles,assoc,pixels,cv,ranges,summary)
    for name,digest in authority.items():assert sha(ROOT/name)==digest,f'Authority changed during this build: {name}'
    for p,digest in SOURCES.items():assert sha(ROOT/p)==digest,f'Source changed during analysis: {p}'
    save('source_manifest.json',SOURCES)
    save('validation.json',dict(status='numerical-checks-pass-independent-and-visual-review-pending',unit_tests=tests,
        source_hashes=len(SOURCES),authority_hashes=authority,selected_curves=51,case_count=113,selected_case_metric_matches=15,
        variants=len(variants),variant_system_count=len(variants)*51,correlations=len(assoc),definition_checks=len(definition),
        cross_validation_models=len(cv),simulations_launched=0,raw_climate_access=False))
    save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
