"""Saved-output-only unified 51-system parameter analysis. No dynamics or raw climate reads."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json,hashlib,html,subprocess
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'analysis/lia_51_systematic_analysis_v1'
BASE=ROOT/'validation_data/results/paleo_surface_validation_v0'
SOURCES={}
COLORS={'legacy_constant':'#25799a','candidate_constant':'#6b9f58','synthetic_timevarying':'#b0496f'}
ROUTES={'legacy_constant':'恒定·原模型分支22条','candidate_constant':'恒定·GPU分支19条','synthetic_timevarying':'时变·合成系统分支10条'}
LABELS={'base_temp_bias_c':'基础温度偏差 / ℃','q_mean_1500_2000_c':'平均附加校正 q / ℃',
 'effective_bias_mean_c':'基础偏差 + 平均q / ℃','q_change_1900_2000_c':'1900—2000附加校正增量 / ℃',
 'melt_f':'融化因子 / kg/(m²·day·K)','prcp_fac':'降水因子（倍数）','log_glen_a':'log10 冰流参数A / Pa^-3 s^-1',
 'spin_delta_c':'初态构建校正 / ℃','terminus_elevation_m':'末端高程 / m','normalized_mean_height':'归一化平均高程',
 'longitude':'经度 / °E','log_area':'log10 静态面积 / km²',
 'volume_loss_1900_pct':'1900—2000体积损失 / %','length_loss_peak_pct':'峰值至2000长度损失 / %',
 'volume_loss_peak_pct':'峰值至2000体积损失 / %','length_interior_peak_year':'内部长度峰值年份','volume_interior_peak_year':'内部体积峰值年份'}
CLIMATE=['base_temp_bias_c','q_mean_1500_2000_c','melt_f','prcp_fac']
PARAMS=CLIMATE+['effective_bias_mean_c','q_change_1900_2000_c','log_glen_a','spin_delta_c']
RESPONSES=['volume_loss_1900_pct','length_loss_peak_pct','volume_loss_peak_pct','length_interior_peak_year','volume_interior_peak_year']
GEOM=['terminus_elevation_m','normalized_mean_height','longitude','log_area']
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':10,'figure.dpi':140})

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def reg(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    digest=sha(p)
    if expected is not None:assert digest==expected.upper(),f'source changed: {p}'
    SOURCES[p.relative_to(ROOT).as_posix()]=digest
    return p
def read(p,expected=None):return json.loads(reg(p,expected).read_text(encoding='utf-8-sig'))
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    return v
def save(name,data):(OUT/name).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def policy_values(p,years):
    years=np.asarray(years,float);fraction=np.clip((years-1470)/530,0,1)
    gamma=p['ramp_exponent'] if p['ramp_exponent'] is not None else 1.
    q=p['forward_start_c']+(p['forward_end_c']-p['forward_start_c'])*fraction**gamma
    if p.get('pulse'):
        pulse=p['pulse'];left=(years-pulse['start_year'])/(pulse['peak_year']-pulse['start_year'])
        right=(pulse['end_year']-years)/(pulse['end_year']-pulse['peak_year'])
        q=q+pulse['amplitude_c']*np.maximum(0,np.minimum(left,right))
    return q

def load():
    old_outputs=read(OLD/'output_manifest.json')
    inputs={n:read(OLD/n,old_outputs[n]) for n in ['features.json','members.json','parameters.json','curves.json']}
    lineage=read(OLD/'source_manifest.json')
    # Canonical files evolved in unrelated solver branches. Keep historical snapshot hashes,
    # check immutable data lineage, and bind current authority independently below.
    for path,digest in lineage.items():
        if path in ['LIA_PROJECT_FACTS.md','LIA_CURRENT_STATE.json']:continue
        reg(path,digest)
    for path in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:reg(ROOT/path)
    matrix=read(BASE/'lia_191_gpu_four_group_design_v10_262_v1/four_group_parameter_matrix.json')
    bp=matrix['baseline_source'];baselines=pd.read_csv(reg(bp['path'],bp['sha256']))
    columns=bp['per_system_fields'];years=np.arange(1470,2001);rows=[];policies=[];provenance=[]
    for f,m,p in zip(inputs['features.json'],inputs['members.json'],inputs['parameters.json']):
        assert f['system']==m['system']==p['system'];sid=f['system'];r={**f}
        evidence=list(p['source_files']);restored=False
        if m['status']=='candidate':
            plan_path=(ROOT/m['curve_page']).parent/'plan.json';plan=read(plan_path)
            ref=plan['inputs']['manifest'];read(ref['path'],ref['sha256'])
            assert ref['sha256']=='E1FCDC73CCD9ADADD57CD2929BB1A9DA4666CEB9BC232F376475C5BBF69AB6AA'
            result=read(m['result_source']);assert result['candidate']==m['parameters']
            b=baselines.loc[baselines.lia_unique_id.eq(sid),columns].drop_duplicates();assert len(b)==1
            b=b.iloc[0];q=m['parameters'];assert q['forward_end']==0
            for name,mult in [('melt_f','melt'),('prcp_fac','prcp'),('glen_a_pa_3_s_1','glen')]:
                r['glen_a' if name=='glen_a_pa_3_s_1' else name]=float(b[name])*q[mult]
            r['base_temp_bias_c']=float(b.base_temp_bias_degC);r['temp_melt_c']=float(b.temp_melt_degC)
            r.update(melt_multiplier=q['melt'],prcp_multiplier=q['prcp'],glen_multiplier=q['glen'])
            evidence.extend([plan_path.relative_to(ROOT).as_posix(),bp['path'],m['result_source']]);restored=True
        else:
            folder=(ROOT/m['source']).parent
            if m['model_route']=='frozen-legacy':s=read(folder/'summary.json')['parameters']
            else:s=read(folder/'config_snapshot.json')['stage2']
            r['temp_melt_c']=float(s['temp_melt_c'])
            for name,source in [('melt_f','melt_f'),('prcp_fac','prcp_fac'),('base_temp_bias_c','base_temp_bias_c')]:assert np.isclose(r[name],s[source])
        vals=policy_values(p,years);base=r['base_temp_bias_c']
        r.update(q_mean_1500_2000_c=float(vals[30:].mean()),q_1900_c=float(vals[430]),
            q_change_1900_2000_c=float(vals[-1]-vals[430]),effective_bias_mean_c=float(base+vals[30:].mean()),
            effective_bias_start_c=float(base+vals[0]),effective_bias_end_c=float(base+vals[-1]),
            log_glen_a=float(np.log10(r['glen_a'])),route_candidate=int(p['route_family']=='candidate_constant'),
            route_synthetic=int(p['route_family']=='synthetic_timevarying'),
            policy_group='timevarying' if p['route_family']=='synthetic_timevarying' else 'constant',
            parameters_restored_from_frozen_baseline=restored,parameter_source_files=evidence)
        assert all(np.isfinite(r[k]) for k in PARAMS)
        rows.append(r)
        policies.append(dict(system=sid,years=years.tolist(),q_c=vals.tolist(),base_plus_q_c=(base+vals).tolist(),
             ramp_exponent=p.get('ramp_exponent'),pulse=p.get('pulse'),policy=p['policy'],
             note='registered policy evaluated at integer years; excludes reconstructed temperature and lapse-rate adjustment'))
        provenance.append(dict(system=sid,restored=restored,source_files=evidence,origin_status=m['status'],
            combined_analysis_inclusion=True,formal_acceptance_unchanged=True))
    frame=pd.DataFrame(rows);assert len(frame)==51 and frame.system.nunique()==51
    return frame,inputs['curves.json'],policies,provenance

def corr(frame,x,y):
    d=frame[[x,y]].dropna().round(10);valid=len(d)>=4 and all(d[k].nunique()>1 for k in [x,y])
    return dict(n=len(d),rho=float(spearmanr(d[x],d[y]).statistic) if valid else None,
                reason=None if valid else 'insufficient sample or constant parameter')

def partial(frame,x,y,controls):
    keys=list(dict.fromkeys([x,y]+controls));d=frame[keys].dropna().round(10);n=len(d);min_n=max(12,3*(len(controls)+2))
    out=dict(n=n,controls=controls,rho=None,minimum_n=min_n)
    if n<min_n:return {**out,'reason':'small sample for this control set'}
    if any(d[k].nunique()<2 for k in keys):return {**out,'reason':'constant variable'}
    ranks=d.apply(lambda a:rankdata(a,method='average'));z=np.column_stack([np.ones(n),ranks[controls]])
    z[:,1:]=(z[:,1:]-z[:,1:].mean(axis=0))/z[:,1:].std(axis=0)
    rank=int(np.linalg.matrix_rank(z));out.update(design_rank=rank,condition_number=float(np.linalg.cond(z)),residual_df=n-rank-1)
    if rank<z.shape[1]:return {**out,'reason':'rank deficient'}
    res=[ranks[k].values-z@np.linalg.lstsq(z,ranks[k].values,rcond=None)[0] for k in [x,y]]
    if any(np.std(a)<1e-10 for a in res):return {**out,'reason':'near-zero residual variance'}
    return {**out,'rho':float(np.corrcoef(*res)[0,1]),'reason':None}

def cohorts(frame):
    return {'all51':frame,'constant41':frame[frame.policy_group.eq('constant')],
        'timevarying10':frame[frame.policy_group.eq('timevarying')],
        'original32_sensitivity':frame[frame.status.eq('retained')],'original19_sensitivity':frame[frame.status.eq('candidate')]}

def analyze(frame,curves):
    associations=[];adjusted=[];robust=[];summaries=[]
    for cohort,sub in cohorts(frame).items():
        for x in PARAMS:
            for y in RESPONSES:associations.append(dict(cohort=cohort,parameter=x,response=y,**corr(sub,x,y)))
        for x in PARAMS+['glen_a','forward_start_c','forward_end_c','temp_melt_c']:
            a=sub[x].dropna();summaries.append(dict(cohort=cohort,parameter=x,n=len(a),minimum=a.min(),q25=a.quantile(.25),median=a.median(),q75=a.quantile(.75),maximum=a.max()))
    for cohort in ['all51','constant41']:
        sub=cohorts(frame)[cohort];routes=['route_candidate','route_synthetic'] if cohort=='all51' else ['route_candidate']
        for x in CLIMATE:
            for y in RESPONSES:
                for model,controls in [('geometry_route',GEOM+routes),('joint_climate',[k for k in CLIMATE if k!=x]+GEOM+routes+['log_glen_a'])]:
                    adjusted.append(dict(cohort=cohort,parameter=x,response=y,model=model,**partial(sub,x,y,controls)))
            controls=[k for k in CLIMATE if k!=x]+GEOM+routes+['log_glen_a']
            deletions=[]
            for pix in sorted(sub.climate_pixel.unique()):
                keep=sub[sub.climate_pixel.ne(pix)]
                deletions.append(dict(omitted=pix,raw=corr(keep,x,RESPONSES[0]),joint=partial(keep,x,RESPONSES[0],controls)))
            pixel=sub.groupby('climate_pixel')[[x,RESPONSES[0]]].median().reset_index()
            robust.append(dict(cohort=cohort,parameter=x,equal_pixel_medians=corr(pixel,x,RESPONSES[0]),deletions=deletions))
    variables=PARAMS+GEOM
    matrix=frame[variables].round(10).corr(method='spearman')
    groups=[];byid={c['system']:c for c in curves}
    for x in ['q_mean_1500_2000_c','melt_f','prcp_fac']:
        grouping_values=frame[x].round(10)
        bounds=np.quantile(grouping_values,[1/3,2/3]);labels=np.digitize(grouping_values,bounds,right=False)
        for i in range(3):
            sub=frame[labels==i];assert len(sub)>0
            entry=dict(parameter=x,group=i+1,bounds=bounds.tolist(),n=len(sub),systems=sub.system.tolist(),
                timevarying_count=int(sub.route_synthetic.sum()),years=list(range(1500,2001)),
                median_loss_1900_pct=float(sub.volume_loss_1900_pct.median()))
            for key in ['length_km','volume_km3']:
                a=np.array([byid[s][key] for s in sub.system]);norm=100*(a/a[:,-1,None]-1)
                entry[key]=dict(q25_median_q75=np.quantile(norm,[.25,.5,.75],axis=0).tolist())
            groups.append(entry)
    return associations,adjusted,robust,summaries,dict(variables=variables,rho=matrix.values.tolist()),groups

def savefig(name):
    plt.savefig(OUT/(name+'.png'),bbox_inches='tight');plt.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close()

def scatter_routes(ax,frame,x,y):
    for route,col in COLORS.items():
        s=frame[frame.route_family.eq(route)];ax.scatter(s[x],s[y],c=col,s=36,marker='s' if route=='synthetic_timevarying' else 'o',label=ROUTES[route],alpha=.85)

def figures(f,policies,associations,adjusted,robust,matrix,groups):
    fig,axs=plt.subplots(2,3,figsize=(13,8));rng=np.random.default_rng(510920)
    for ax,key in zip(axs.flat,['base_temp_bias_c','q_mean_1500_2000_c','melt_f','prcp_fac','log_glen_a','q_change_1900_2000_c']):
        for j,(route,col) in enumerate(COLORS.items()):
            sub=f[f.route_family.eq(route)];ax.scatter(j+rng.uniform(-.16,.16,len(sub)),sub[key],c=col,s=30,alpha=.85)
            ax.plot([j-.23,j+.23],[sub[key].median()]*2,color='black',lw=2)
        ax.set(xticks=[0,1,2],xticklabels=['恒定·原分支\n22条','恒定·GPU\n19条','时变·合成系统\n10条'],ylabel=LABELS[key]);ax.grid(axis='y',alpha=.2)
    fig.suptitle('01 统一51条的实际参数｜黑线为组内中位数；分支仅作来源标记')
    fig.tight_layout();savefig('01_distributions')
    fig,axs=plt.subplots(2,2,figsize=(12,8));lookup=f.set_index('system');varying_index=0
    for p in policies:
        row=lookup.loc[p['system']];j=int(row.route_synthetic);color=COLORS[row.route_family]
        if j:color=plt.get_cmap('tab10')(varying_index);varying_index+=1
        for i,key in enumerate(['q_c','base_plus_q_c']):axs[i,j].plot(p['years'],p[key],c=color,alpha=.7 if j else .5,lw=1,label=str(p['system']) if j else None)
    axs[0,1].legend(loc='upper left',ncol=2,fontsize=7,title='系统编号',title_fontsize=8)
    for j,title in enumerate(['恒定校正41条','时变校正10条']):
        for i,lab in enumerate(['附加校正 q / ℃','基础偏差 + q / ℃']):axs[i,j].set(title=title,ylabel=lab,xlabel='历史年份',xlim=(1470,2045));axs[i,j].grid(alpha=.2)
    for i in range(2):
        lo=min(axs[i,0].get_ylim()[0],axs[i,1].get_ylim()[0]);hi=max(axs[i,0].get_ylim()[1],axs[i,1].get_ylim()[1])
        for j in range(2):axs[i,j].set_ylim(lo,hi)
    fig.suptitle('02 温度校正策略｜年整点策略示意，不是实际气候温度；不含重建距平与高程订正')
    fig.tight_layout();savefig('02_temperature_policies')
    vars=matrix['variables'];arr=np.asarray(matrix['rho'],float);fig,ax=plt.subplots(figsize=(12,10))
    im=ax.imshow(arr,vmin=-1,vmax=1,cmap='RdBu_r')
    for i in range(len(vars)):
        for j in range(len(vars)):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white' if abs(arr[i,j])>.65 else 'black')
    ax.set(xticks=range(len(vars)),yticks=range(len(vars)),xticklabels=[LABELS[x].split(' / ')[0] for x in vars],yticklabels=[LABELS[x].split(' / ')[0] for x in vars])
    plt.setp(ax.get_xticklabels(),rotation=55,ha='right');fig.colorbar(im,ax=ax,label='Spearman ρ')
    ax.set_title('03 参数与地形的共同变化｜全51条；相关提示补偿可能，不证明参数可互换')
    fig.tight_layout();savefig('03_parameter_dependence')
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for ax,key in zip(axs.flat,CLIMATE+['effective_bias_mean_c','log_glen_a']):
        scatter_routes(ax,f,key,RESPONSES[0]);r=corr(f,key,RESPONSES[0]);ax.set(xlabel=LABELS[key],ylabel=LABELS[RESPONSES[0]],title=f"ρ={r['rho']:.2f}，n={r['n']}");ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.suptitle('04 参数与历史体积损失｜全51条；各冰川参数已经过选择，不能解释为因果效应')
    fig.tight_layout();savefig('04_parameter_response')
    fig,axs=plt.subplots(1,2,figsize=(13,5.5),sharey=True)
    for ax,co,title in zip(axs,['all51','constant41'],['统一51条','恒定校正41条']):
        for i,x in enumerate(CLIMATE):
            raw=next(a for a in associations if a['cohort']==co and a['parameter']==x and a['response']==RESPONSES[0])
            joint=next(a for a in adjusted if a['cohort']==co and a['parameter']==x and a['response']==RESPONSES[0] and a['model']=='joint_climate')
            rob=next(a for a in robust if a['cohort']==co and a['parameter']==x)
            vals=[a['joint']['rho'] for a in rob['deletions'] if a['joint']['rho'] is not None]
            ax.scatter(raw['rho'],i+.12,marker='o',c='#25799a',s=45,label='原始相关' if i==0 else None)
            if joint['rho'] is not None:ax.scatter(joint['rho'],i-.12,marker='s',c='#b0496f',s=45,label='联合条件秩相关' if i==0 else None)
            if vals:ax.plot([min(vals),max(vals)],[i-.12]*2,c='#b0496f',lw=2,alpha=.65,label='逐像元剔除范围' if i==0 else None)
        ax.axvline(0,c='gray',lw=1);ax.set(xlim=(-1,1),yticks=range(4),yticklabels=[LABELS[x].split(' / ')[0] for x in CLIMATE],xlabel='相关系数（不是因果效应量）',title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('05 联合检验｜控制其余气候参数、地形、分支和冰流参数；线段不是置信区间')
    fig.tight_layout();savefig('05_joint_robustness')
    fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for ax,key in zip(axs.flat,CLIMATE+['effective_bias_mean_c','q_change_1900_2000_c']):
        sc=ax.scatter(f.longitude,f.latitude,c=f[key],cmap='viridis',s=55,edgecolors='white',linewidth=.5)
        ax.set(title=LABELS[key],xlabel='经度 / °E',ylabel='纬度 / °N');ax.grid(alpha=.2);fig.colorbar(sc,ax=ax,shrink=.7)
    fig.suptitle('06 参数的空间分布｜点为选定冰川系统；不是气候场或区域插值')
    fig.tight_layout();savefig('06_spatial_parameters')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for j,x in enumerate(['q_mean_1500_2000_c','melt_f','prcp_fac']):
        for g,col in zip([g for g in groups if g['parameter']==x],['#25799a','#d29339','#b0496f']):
            for i,key in enumerate(['length_km','volume_km3']):
                q=np.array(g[key]['q25_median_q75']);ax=axs[i,j];ax.plot(g['years'],q[1],c=col,label=f"第{g['group']}组 n={g['n']}，时变{g['timevarying_count']}条");ax.fill_between(g['years'],q[0],q[2],color=col,alpha=.12)
        for i,title in enumerate(['长度','体积']):
            axs[i,j].set(title=LABELS[x].split(' / ')[0]+'分组 · '+title,xlabel='年份',ylabel='相对各自模拟2000年 / %');axs[i,j].legend(fontsize=7);axs[i,j].grid(alpha=.2)
    fig.suptitle('07 按参数三分位分组的历史曲线｜统一51条；中位数与四分位范围；不是参数干预试验')
    fig.tight_layout();savefig('07_grouped_history')
    fig,axs=plt.subplots(1,2,figsize=(14,6))
    response_labels=['1900—2000\n体积损失','历史最大值→2000\n长度损失','历史最大值→2000\n体积损失','长度内部\n峰值年份','体积内部\n峰值年份']
    for ax,mode in zip(axs,['raw','joint']):
        source=associations if mode=='raw' else [a for a in adjusted if a['model']=='joint_climate']
        cells=[[next(a for a in source if a['cohort']=='all51' and a['parameter']==x and a['response']==y) for y in RESPONSES] for x in CLIMATE]
        vals=np.array([[np.nan if a['rho'] is None else a['rho'] for a in row] for row in cells])
        im=ax.imshow(vals,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
        for i,row in enumerate(cells):
            for j,a in enumerate(row):
                text=f"{a['rho']:+.2f}\nn={a['n']}" if a['rho'] is not None else f"不计算\nn={a['n']}"
                ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if a['rho'] is not None and abs(a['rho'])>.65 else 'black')
        ax.set(xticks=range(5),yticks=range(4),xticklabels=response_labels,yticklabels=[LABELS[x].split(' / ')[0] for x in CLIMATE],title='原始秩相关' if mode=='raw' else '联合条件秩相关')
        plt.setp(ax.get_xticklabels(),rotation=35,ha='right');fig.colorbar(im,ax=ax,shrink=.75,label='ρ')
    fig.suptitle('08 整段历史与固定阶段并列｜最大值取1500—2000原生曲线；边界峰值不参加年份相关')
    fig.tight_layout();savefig('08_full_history_responses')

def fmt(v):
    if v is None or (isinstance(v,float) and not np.isfinite(v)):return '不可计算'
    if isinstance(v,(float,np.floating)):return f'{v:.3g}' if 0<abs(v)<.001 else f'{v:.3f}'
    return str(v)
def table(records,columns=None,rename=None):
    df=pd.DataFrame(records)
    if columns:df=df[columns]
    if rename:df=df.rename(columns=rename)
    return df.to_html(index=False,escape=True,na_rep='未提供',float_format=fmt,border=0)

def report(f,associations,adjusted,robust,summaries,matrix,groups):
    results=[]
    for x in CLIMATE:
        raw=next(a for a in associations if a['cohort']=='all51' and a['parameter']==x and a['response']==RESPONSES[0])
        joint=next(a for a in adjusted if a['cohort']=='all51' and a['parameter']==x and a['response']==RESPONSES[0] and a['model']=='joint_climate')
        constant=next(a for a in adjusted if a['cohort']=='constant41' and a['parameter']==x and a['response']==RESPONSES[0] and a['model']=='joint_climate')
        rob=next(a for a in robust if a['cohort']=='all51' and a['parameter']==x);vals=[d['joint']['rho'] for d in rob['deletions'] if d['joint']['rho'] is not None]
        results.append(dict(parameter=x,label=LABELS[x].split(' / ')[0],raw_rho=raw['rho'],joint_rho=joint['rho'],constant_joint_rho=constant['rho'],
             joint_n=joint['n'],joint_controls=joint['controls'],deletion_min=min(vals) if vals else None,deletion_max=max(vals) if vals else None,
             computed_deletions=len(vals),total_pixels=f.climate_pixel.nunique()))
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),unified_sample=51,constant=41,timevarying=10,climate_pixels=f.climate_pixel.nunique(),
        baseline_parameters_restored=19,all_parameter_rows_complete=True,main_response=RESPONSES[0],main_results=results,
        simulations_launched=False,raw_climate_accessed=False,acceptance_ledger_modified=False,
        attribution='not identifiable from selected cross-glacier associations',historical_window=[1500,2000])
    save('summary.json',summary)
    paragraphs=''.join(f"<li>{r['label']}：原始相关 {fmt(r['raw_rho'])}；联合条件秩相关 {fmt(r['joint_rho'])}；恒定校正41条内 {fmt(r['constant_joint_rho'])}。</li>" for r in results)
    interpretation=(f"<p>值得优先关注的是：融化因子的原始相关为 {results[2]['raw_rho']:.2f}，联合控制后为 {results[2]['joint_rho']:.2f}；平均q的原始相关为 {results[1]['raw_rho']:.2f}，不能据此把附加校正大小当作退缩幅度的单一解释。基础温度偏差由正相关转为负的条件相关，提示原始关系依赖于样本和协变量构成；这不是‘升温减缓退缩’的证据。</p>"
       f"<p>共有 {int(f.base_temp_bias_c.eq(0).sum())} 条基础温度偏差恰为0，其余为非零校准值，不能把这组参数简单看作均匀连续的气候观测。10条时变策略在1900—2000年增加的q最多约 {f.q_change_1900_2000_c.max():.3f} ℃，与其1470—2000总增量不同；这里比较的是附加校正，不是这一时期真实变暖量。</p>")
    selected=[s for s in summaries if s['cohort']=='all51' and s['parameter'] in CLIMATE+['glen_a','temp_melt_c']]
    for s in selected:s['label']=LABELS.get(s['parameter'],{'glen_a':'冰流参数 A / Pa⁻³ s⁻¹','temp_melt_c':'融化阈值 / ℃'}.get(s['parameter']))
    blocks=[]
    def block(title,text,fig=None,extra=''):
        blocks.append(f'<section><h2>{title}</h2><p>{text}</p>'+('' if fig is None else f'<a href="{fig}.svg"><img src="{fig}.png" alt="{html.escape(title)}" loading="lazy"></a>')+extra+'</section>')
    block('1. 统一51条：实际参数已补齐','沿用同一批51条代表曲线；原32/19身份仅保留为来源标记和补充稳健性划分。19条的搜索倍率均为1，实际物理参数已从冻结基础参数表恢复，不再以倍率替代参数。正式接受名单没有改动。','01_distributions',table(selected,['label','n','minimum','median','maximum'],{'label':'参数','minimum':'最小','median':'中位数','maximum':'最大'}))
    block('2. 气候输入与校正不是同一个量','共同框架为GSWP3_W5E5的1961—1990年月气候态，加ensemble_rec_ref重建温度距平；降水异常关闭，再由各冰川降水因子缩放。温度校正示意包含基础偏差与历史q(t)，不包含实际重建距平或高程递减率项。恒定q不代表恒定历史气候；spinup校正只描述初态构建，不解释成真实古温度。','02_temperature_policies', '<p>这51条包含41条恒定q和10条时变q。时变策略与合成系统模型分支重合，因此不能从组间差别分离出“时变校正的独立作用”。具体年整点策略见 <a href="temperature_policies.json">校正策略数据</a>。本次没有读取原始气候数组或重新运行模型。</p>')
    block('3. 参数组合：先看是否相互关联','完整矩阵同时包含基础温度偏差、历史校正、融化因子、降水因子、冰流参数以及地形变量。基础偏差与平均q相加得到组合偏差，这是定义上的联系，不能当作独立证据。其他相关也可能来自空间分布、校准策略或择优过程，不能仅凭矩阵认定参数补偿机制。','03_parameter_dependence')
    block('4. 参数与历史变化','主响应为1900—2000年的相对体积损失，正值表示损失。所有51条进入主分析；峰值至2000年的长度/体积损失、窗口内部峰值年份作为补充。边界峰值不作为明确最大期参与峰值时间相关。','04_parameter_response','<p><a href="associations.json">全部参数—响应相关</a>同时保存弱相关、不可计算项和样本数，不按显著性筛选。</p>')
    block('4b. 从历史最大值到2000：不只看20世纪','并列检查1500—2000窗口内最大长度、最大体积到2000年的损失，以及峰值出现时间。固定1900—2000阶段便于比较同一时间段；最大值到2000则反映各冰川各自峰值后的变化，两者回答不同问题。最大值来自模型曲线，不等于独立定年的LIA最大期。','08_full_history_responses')
    block('5. 联合条件与删除稳健性','联合检验控制其余三项气候参数、末端高程、高程—面积分布、经度、面积、模型分支及冰流参数。结果为秩残差之间的相关，而不是每升温1℃的损失效应。统一51条与恒定41条并列；逐登记气候像元剔除用于检查单个共享气候区域的影响，范围不是置信区间。','05_joint_robustness','<ul>'+paragraphs+'</ul>'+table(results,['label','joint_n','raw_rho','joint_rho','constant_joint_rho','deletion_min','deletion_max'],{'label':'参数','joint_n':'n','raw_rho':'原始ρ','joint_rho':'联合ρ','constant_joint_rho':'恒定41条联合ρ','deletion_min':'剔除最小ρ','deletion_max':'剔除最大ρ'})+interpretation+'<p>参数是在现有拟合/选择流程后得到的。样本量有限、共享气候像元及校准目标约束仍会影响解释；条件控制并不构成独立气候归因。10条时变样本只报告描述性相关，不强行拟合多参数控制模型。</p>')
    block('6. 空间差异','将实际参数放回冰川系统位置，判断参数大小是否随空间位置共同变化。51条对应33个登记气候像元；地图上的点不代表区域连续气候场。','06_spatial_parameters')
    block('7. 参数分组的历史曲线','按平均q、融化因子和降水因子的样本三分位阈值分别分组。同值参数不因微小浮点误差而拆组，因此不强行保证每组数量相同。每组同时列出数量和时变分支数量，防止把组内模型构成差异解释为参数作用。曲线为各冰川相对模拟2000年的中位数与四分位范围；既有十年输出的年度连线为插值，不增加时间精度。','07_grouped_history',table([dict(参数=LABELS[g['parameter']].split(' / ')[0],组=g['group'],下阈值=g['bounds'][0],上阈值=g['bounds'][1],n=g['n'],时变条数=g['timevarying_count'],体积损失中位数=g['median_loss_1900_pct']) for g in groups]))
    cols=['system','policy_group','base_temp_bias_c','forward_start_c','forward_end_c','q_mean_1500_2000_c','effective_bias_mean_c','melt_f','prcp_fac','glen_a','temp_melt_c','terminus_elevation_m','volume_loss_1900_pct']
    names={'system':'系统','policy_group':'校正形式','base_temp_bias_c':'基础偏差 ℃','forward_start_c':'q起点 ℃','forward_end_c':'q终点 ℃','q_mean_1500_2000_c':'q均值 ℃','effective_bias_mean_c':'组合偏差均值 ℃','melt_f':'融化因子','prcp_fac':'降水因子','glen_a':'冰流A','temp_melt_c':'融化阈值 ℃','terminus_elevation_m':'末端高程 m','volume_loss_1900_pct':'1900—2000体积损失 %'}
    block('8. 逐条参数与来源','这里列的是实际参数值。下载完整数据可查看原来源身份、模型分支、参数恢复证据及全部历史诊断。',extra='<input id="filter" placeholder="搜索系统编号或数值"><div class="scroll" id="inventory">'+table(f.to_dict('records'),cols,names)+'</div><p><a href="parameters_51.json">51条完整参数</a> · <a href="provenance.json">来源追溯</a> · <a href="parameter_summary.json">参数统计</a> · <a href="adjusted_associations.json">全部联合检验</a> · <a href="robustness.json">删除检验</a> · <a href="methods.json">方法</a> · <a href="validation.json">验证记录</a></p>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>51条LIA冰川：气候参数联合分析</title><style>body{margin:0;background:#edf2f5;color:#183347;font-family:Microsoft YaHei,Arial,sans-serif;line-height:1.8}main{max-width:1250px;margin:auto;padding:24px}header,section{background:white;padding:26px;margin-bottom:20px;border-radius:12px}h1{font-size:29px}h2{font-size:22px}img{width:100%;height:auto}.note{background:#fff3dc;padding:14px}a{color:#146886}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left;white-space:nowrap}th{background:#edf3f5}.scroll{overflow:auto}input{padding:10px;min-width:280px;margin-bottom:12px}</style><main><header><h1>51条LIA冰川：气候参数联合分析</h1><p>统一51条主样本｜41条恒定校正 + 10条时变校正｜只分析已保存历史结果</p><p class="note">本报告研究参数配置与历史响应之间的关系，不把拟合温度校正当作古温度重建，也不把相关当作独立气候归因。</p></header>'''+''.join(blocks)+'''</main><script>document.getElementById('filter').addEventListener('input',function(){let q=this.value.toLowerCase();document.querySelectorAll('#inventory tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8');return summary

def main():
    command=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
    check=subprocess.run([sys.executable,'-X','utf8',str(command)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    assert check.returncode==0,check.stderr;controller=json.loads(check.stdout);assert controller['status'].lower()=='pass'
    save('controller_validation.json',controller);reg(OUT/'methods.json');reg(Path(__file__))
    f,curves,policies,provenance=load();print('Verified all51; actual baseline parameters restored for19.',flush=True)
    associations,adjusted,robust,summaries,matrix,groups=analyze(f,curves)
    for name,data in [('parameters_51.json',f.to_dict('records')),('temperature_policies.json',policies),('provenance.json',provenance),
        ('associations.json',associations),('adjusted_associations.json',adjusted),('robustness.json',robust),
        ('parameter_summary.json',summaries),('parameter_correlations.json',matrix),('group_statistics.json',groups)]:save(name,data)
    figures(f,policies,associations,adjusted,robust,matrix,groups)
    summary=report(f,associations,adjusted,robust,summaries,matrix,groups);save('source_manifest.json',SOURCES)
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
