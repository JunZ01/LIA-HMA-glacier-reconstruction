"""Frozen-output synthesis. No model, climate reader or optimisation imports."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib, html, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
ATLAS=ROOT/'analysis/lia_51_stage_reliability_v1'; FEATURES=ROOT/'analysis/lia_51_stage_controls_v1'
KEYS=['length_km','area_km2','volume_km3']; NAMES=['长度','面积','体积']; UNITS=['km','km²','km³']
PERIODS=[(1500,1850),(1850,1900),(1900,2000),(1500,2000)]; GRID=np.arange(1500,2001,10)
LABELS={'stable':'已有组合内较稳定','timing_sensitive':'幅度较稳、时序敏感','amplitude_sensitive':'幅度也敏感','boundary':'峰值触及边界','untested':'缺少多方案检验'}
COLORS={'stable':'#218573','timing_sensitive':'#bf7026','amplitude_sensitive':'#b74657','boundary':'#81509d','untested':'#84939e'}
COHORTS={'all51':'统一51条','constant41':'恒定修正41条','synthetic10':'时变修正10条','original32':'原32条（来源）','original19':'原19条（来源）'}
SOURCES={}
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def reg(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    assert p.name!='inputs.npz' and not any(s in p.as_posix().lower() for s in ['20cr','jionglacuo','s7t4zc34sr','temperature reconstruction.xlsx','s41612-025-01254-y'])
    h=sha(p)
    if expected:assert h==expected.upper(),str(p)
    SOURCES[p.relative_to(ROOT).as_posix()]=h;return p
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(name,x):(OUT/name).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def aligned(c,key,years=GRID):return np.interp(years,c['native_years'],c['native_'+key])
def period(t,x,a,b):
    start,end=np.interp([a,b],t,x);delta=float(end-start)
    return dict(start=a,end=b,value_start=float(start),value_end=float(end),delta=delta,change_pct=float(100*delta/start),net_rate_per_year=delta/(b-a))
def load_inputs():
    methods=read(reg(OUT/'methods.json'))
    authority=methods['authority']
    for reconciliation in sorted(OUT.glob('authority_reconciliation_v*.json')):
        rec=read(reg(reconciliation))
        assert hashlib.sha256((ROOT/'LIA_PROJECT_FACTS.md').read_bytes()[:rec['facts_previous_bytes']]).hexdigest().upper()==rec['facts_previous_sha256']
        assert authority['LIA_PROJECT_FACTS.md']==rec['facts_previous_sha256']
        authority=rec['current_authority_hashes']
        for p,h in rec['evidence'].items():reg(p,h)
    for p,h in authority.items():reg(p,h)
    data={}
    for folder,names in [(ATLAS,['reference_curves.json','atlas.json','case_diagnostics.json','conditional_reliability.json','regional_activity.json','summary.json','validation.json']), (FEATURES,['features_51.json'])]:
        manifest=read(reg(folder/'output_manifest.json'))
        for name in names:data[name]=read(reg(folder/name,manifest[name]))
    assert data['validation.json']['status']=='pass'
    lineage=read(reg(ATLAS/'source_manifest.json'))
    for a in data['atlas.json']:reg(a['source'],lineage[a['source']])
    for c in data['case_diagnostics.json']:
        for k in ['state_source','result_source']:reg(c[k],lineage[c[k]])
    curves={c['system']:c for c in data['reference_curves.json']};atlas={a['system']:a for a in data['atlas.json']};features={f['system']:f for f in data['features_51.json']}
    assert len(curves)==51 and set(curves)==set(atlas)==set(features)
    cases=defaultdict(list)
    for c in data['case_diagnostics.json']:
        row=dict(system=c['system'],case_id=c['case_id'],selected=c['selected'],native_years=GRID,peak={},periods={})
        for key in KEYS:
            m=c['metrics'][key];assert m['interval']['years']==GRID.tolist()
            row['native_'+key]=np.asarray(m['interval']['values']);row['peak'][key]=m['native_peak']
            row['periods'][key]=[period(GRID,row['native_'+key],a,b) for a,b in PERIODS]
            for p,old in zip(row['periods'][key],m['periods']):
                assert p['start']==old['start'] and np.isclose(p['delta'],old['absolute']) and np.isclose(p['change_pct'],old['change_pct'])
            if c['selected']:assert np.allclose(row['native_'+key],aligned(curves[c['system']],key),rtol=1e-11,atol=1e-10)
        cases[c['system']].append(row)
    assert len(cases)==15 and sum(map(len,cases.values()))==113 and sum(len(v)>1 for v in cases.values())==14
    assert all(sum(c['selected'] for c in cc)==1 for cc in cases.values())
    return methods,curves,atlas,features,cases,data

def members(features):
    return {'all51':sorted(features),'constant41':[s for s,f in features.items() if not f['timevarying']], 'synthetic10':[s for s,f in features.items() if f['timevarying']], 'original32':[s for s,f in features.items() if f['status']=='retained'], 'original19':[s for s,f in features.items() if f['status']=='candidate']}
def make_individual(curves,atlas,cases):
    rows=[]
    for sid,c in sorted(curves.items()):
        a=atlas[sid];row=dict(system=sid,source_status=a['status'],route_family=a['route_family'],climate_pixel=a['climate_pixel'],length_operator=a['length_operator'],native_step_years=a['native_step_years'],case_count=len(cases.get(sid,[])),reliability_label=a['reliability_label'],metrics={})
        for key in KEYS:
            x=np.asarray(c['native_'+key]);t=np.asarray(c['native_years']);peak=a['metrics'][key]['native_peak']
            assert np.isclose(peak['value'],max(x[(t>=1500)&(t<=2000)]))
            pp=[]
            for start,end in PERIODS:
                p=period(t,x,start,end);choices=cases.get(sid) or [c]
                changes=[period(v['native_years'],v['native_'+key],start,end) for v in choices]
                p['saved_case_delta_range']=[min(q['delta'] for q in changes),max(q['delta'] for q in changes)]
                p['saved_case_pct_range']=[min(q['change_pct'] for q in changes),max(q['change_pct'] for q in changes)]
                p['conditional_range_available']=len(choices)>1;pp.append(p)
            loss=peak['value']-float(x[-1])
            row['metrics'][key]=dict(periods=pp,native_peak=peak,peak_to_2000_loss=loss,peak_to_2000_loss_pct=100*loss/peak['value'],near_maximum_segments=a['metrics'][key]['near']['segments'])
        rows.append(row)
    return rows

def aggregate(curves,features,cases,individual):
    rows={r['system']:r for r in individual};result={}
    for group,ids in members(features).items():
        assert len(ids)==int(''.join(filter(str.isdigit,group)))
        rec=dict(systems=ids,n=len(ids),metrics={})
        for key in KEYS:
            values=np.array([aligned(curves[s],key) for s in ids]);periods=[]
            for j,(start,end) in enumerate(PERIODS):
                pp=[rows[s]['metrics'][key]['periods'][j] for s in ids];changes=[p['change_pct'] for p in pp]
                r=dict(start=start,end=end,individual_change_pct_quantiles=np.percentile(changes,[0,25,50,75,100]),shrinking_count=sum(p['delta']<-1e-10 for p in pp),growing_count=sum(p['delta']>1e-10 for p in pp),zero_count=sum(abs(p['delta'])<=1e-10 for p in pp))
                if key!='length_km':
                    total=values.sum(axis=0);r['total']=period(GRID,total,start,end)
                    r['total']['conditional_delta_range']=[sum(p['saved_case_delta_range'][i] for p in pp) for i in [0,1]]
                periods.append(r)
            rec['metrics'][key]=dict(periods=periods)
            if key!='length_km':
                total=values.sum(axis=0);lower=[];upper=[]
                for sid in ids:
                    xx=np.array([aligned(c,key) for c in (cases.get(sid) or [curves[sid]])]);lower.append(xx.min(axis=0));upper.append(xx.max(axis=0))
                maximum=total.max()
                rec['metrics'][key].update(years=GRID,totals=total,pointwise_conditional_lower=np.sum(lower,axis=0),pointwise_conditional_upper=np.sum(upper,axis=0),common_grid_maximum=float(maximum),common_grid_maximum_years=GRID[np.isclose(total,maximum,rtol=1e-10,atol=1e-12)],near_1pct_years=GRID[total>=.99*maximum],maximum_to_2000_loss_pct=float(100*(maximum-total[-1])/maximum))
        result[group]=rec
    return result

def timing(curves,features,prior):
    out=[]
    for group in ['all51','constant41','synthetic10']:
        ids=members(features)[group];pixels=sorted({features[s]['climate_pixel'] for s in ids})
        for key in KEYS:
            for step in [10,20]:
                t=np.arange(1500,2001,step);x=np.array([aligned(curves[s],key,t) for s in ids]);mx=x.max(axis=1)
                def weighted(ind):return np.mean([ind[[i for i,s in enumerate(ids) if features[s]['climate_pixel']==p]].mean(axis=0) for p in pixels],axis=0)
                variants=[]
                for f in [.005,.01,.02]:
                    mask=x>=mx[:,None]*(1-f)-1e-12*np.maximum(1,mx[:,None]);eq=mask.mean(axis=0);wp=weighted(mask)
                    variants.append(dict(fraction=f,equal_system=eq,equal_pixel=wp,max_fraction=float(eq.max()),max_fraction_years=t[np.isclose(eq,eq.max())],max_pixel_fraction=float(wp.max()),max_pixel_years=t[np.isclose(wp,wp.max())]))
                d=np.diff(x,axis=1);states=np.where(d>.001*mx[:,None],1,np.where(d<-.001*mx[:,None],-1,0));decline=states==-1
                rec=dict(group=group,metric=key,step=step,years=t,n=len(ids),pixel_n=len(pixels),near_variants=variants,shrinking_equal_system=decline.mean(axis=0),shrinking_equal_pixel=weighted(decline))
                if group=='all51' and step==10:
                    old=next(p for p in prior if p['group']=='all' and p['metric']==key)
                    for name in ['shrinking_equal_system','shrinking_equal_pixel']:assert np.allclose(rec[name],old[name])
                    for variant,oldv in zip(variants,old['near_threshold_sensitivity']):assert np.allclose(variant['equal_system'],oldv['equal_system'])
                    assert np.allclose(variants[1]['equal_pixel'],old['near_equal_pixel'])
                out.append(rec)
    return out

def summarize(individual,agg,time,data):
    allm=agg['all51']['metrics'];v=allm['volume_km3'];byid={r['system']:r for r in individual}
    contributions=sorted([dict(system=r['system'],loss=-r['metrics']['volume_km3']['periods'][2]['delta']) for r in individual],key=lambda r:r['loss'],reverse=True)
    total=-v['periods'][2]['total']['delta'];top5=sum(c['loss'] for c in contributions[:5])
    conditional={}
    for key in KEYS:
        rr=[r for r in individual if r['case_count']>=2];p=[r['metrics'][key]['periods'][2] for r in rr]
        conditional[key]=dict(multi_case_systems=len(rr),all_saved_cases_shrink=sum(v['saved_case_delta_range'][1]<-1e-10 for v in p),selected_shrink=allm[key]['periods'][2]['shrinking_count'])
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),systems=51,climate_pixels=33,simulation_launches=0,raw_climate_access=False,formal_acceptance_unchanged=True,
        native_sampling_counts=dict(Counter(str(r['native_step_years']) for r in individual)),multiple_case_systems=14,untested_systems=37,case_count=113,
        reliability_label_counts=dict(Counter(r['reliability_label'] for r in individual)),prior_classification_threshold_counts=data['summary.json']['classification_threshold_counts'],
        period_volume_totals=v['periods'],period_area_totals=allm['area_km2']['periods'],period_length_medians=allm['length_km']['periods'],conditional_1900_2000_direction=conditional,
        common_decadal_volume_maximum=v['common_grid_maximum'],common_decadal_volume_maximum_years=v['common_grid_maximum_years'],common_volume_near_1pct_years=v['near_1pct_years'],
        top5_volume_loss_contribution=dict(rows=contributions[:5],loss=top5,share_of_net_loss=top5/total,all_contributions=contributions),
        exact_repeated_length_maximum_count=sum(r['metrics']['length_km']['native_peak']['tied_points']>1 for r in individual),
        caveat='Conditional saved-case ranges and classifications are incomplete sensitivity evidence, not full reconstruction confidence or independent observations.')

def figsave(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def figures(individual,agg,timing):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,name,unit in zip(axes,KEYS[1:],NAMES[1:],UNITS[1:]):
        r=agg['all51']['metrics'][key];t=r['years'];x=r['totals'];p=r['periods'][2]['total']
        ax.fill_between(t,r['pointwise_conditional_lower'],r['pointwise_conditional_upper'],color='#9cafbd',alpha=.45,label='已有14条多方案的逐时范围')
        ax.plot(t,x,color='#226e8d',lw=2,label='51条固定选定结果');ax.axvline(1900,color='#888888',ls=':',lw=1)
        ax.set(title=f'同期总{name}：1900—2000减少{-p["delta"]:.2f} {unit}（{-p["change_pct"]:.1f}%）',xlabel='年份',ylabel=f'样本总{name} / {unit}');ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.suptitle('统一51条：同期模型总量，不是整个喜马拉雅估计',fontsize=15)
    fig.text(.5,.008,'灰带只改变有备选证据的14条，其余37条固定；不是置信区间，也不是一条可以连续实现的历史。10年共同网格。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.93]);figsave(fig,'01_sample_totals')
    rows=sorted(individual,key=lambda r:r['system']);fig,axes=plt.subplots(1,6,figsize=(19,14),sharey=True)
    for j,(key,name,unit) in enumerate(zip(KEYS,NAMES,UNITS)):
        for col,pct in enumerate([False,True]):
            ax=axes[2*j+col]
            for y,r in enumerate(rows):
                p=r['metrics'][key]['periods'][2];field='change_pct' if pct else 'delta';range_key='saved_case_pct_range' if pct else 'saved_case_delta_range';low,high=p[range_key]
                if p['conditional_range_available']:ax.plot([-high,-low],[y,y],c=COLORS[r['reliability_label']],lw=2.2,alpha=.75)
                ax.scatter(-p[field],y,c=COLORS[r['reliability_label']],s=19,zorder=3)
            ax.axvline(0,color='#bbbbbb',lw=.8);ax.grid(axis='x',alpha=.16);ax.set_xlabel('减少比例 / %' if pct else f'减少量 / {unit}');ax.set_title(name+(' · 相对量' if pct else ' · 绝对量'),fontsize=11)
    axes[0].set_yticks(range(51),[str(r['system'])+(' *' if r['case_count']>=2 else '') for r in rows],fontsize=8);axes[0].invert_yaxis()
    handles=[Line2D([],[],color=c,marker='o',ls='',label=LABELS[k]) for k,c in COLORS.items()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.966),ncol=5,fontsize=9)
    fig.suptitle('1900—2000：逐冰川变化幅度与已有备选范围',fontsize=17,y=.995)
    fig.text(.5,.005,'正值表示减少；横线为同一冰川已有案例的范围，星号表示有多个案例。无横线不等于没有不确定性。颜色沿用此前的条件性综合诊断。',ha='center',fontsize=10)
    fig.tight_layout(rect=[.015,.03,1,.94]);figsave(fig,'02_individual_changes')
    fig,axes=plt.subplots(3,2,figsize=(13,10),sharex=True)
    for i,(key,name) in enumerate(zip(KEYS,NAMES)):
        r=next(x for x in timing if x['group']=='all51' and x['metric']==key and x['step']==10);constant=next(x for x in timing if x['group']=='constant41' and x['metric']==key and x['step']==10)
        t=np.asarray(r['years']);ax=axes[i,0];vv=r['near_variants']
        ax.fill_between(t,100*np.array(vv[0]['equal_system']),100*np.array(vv[2]['equal_system']),color='#aec6d0',alpha=.35,label='0.5%—2%定义敏感性')
        ax.plot(t,100*np.array(vv[1]['equal_system']),c='#246b89',label='51条等权 · 1%')
        ax.plot(t,100*np.array(vv[1]['equal_pixel']),c='#945740',ls=':',label='33个像元等权 · 1%')
        ax.plot(t,100*np.array(constant['near_variants'][1]['equal_system']),c='#4d8873',ls='--',label='恒定修正41条 · 1%')
        ax.set_ylabel(name+'近最大值占比 / %');ax.set_ylim(0,100)
        ax=axes[i,1]
        # Explicit interval-right endpoint keeps the final1990-2000 bin visible.
        for values,color,ls,label in [(r['shrinking_equal_system'],'#246b89','-','51条等权'),(r['shrinking_equal_pixel'],'#945740',':','33个像元等权'),(constant['shrinking_equal_system'],'#4d8873','--','恒定修正41条')]:
            ax.stairs(100*np.array(values),t,color=color,ls=ls,label=label or None,baseline=None)
        ax.set_ylabel(name+'净减少的冰川占比 / %');ax.set_ylim(0,105)
    for ax in axes.flat:ax.grid(alpha=.15);ax.set_xlim(1500,2000)
    axes[0,0].set_title('接近各自1500—2000窗口最大值');axes[0,1].set_title('各10年区间的变化方向');axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('年份')
    fig.suptitle('共同阶段与不同步：最大值占比不是统一的LIA最大年代',fontsize=16)
    fig.text(.5,.01,'净减少：10年变化小于窗口最大值的−0.1%；不是质量平衡状态。阴影为定义敏感性，不是概率区间。模型分支与修正政策仍混杂。',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.94]);figsave(fig,'03_shared_timing')

def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def fmt(v,n=2):return f'{v:.{n}f}'
def report(individual,agg,time,summary):
    v=agg['all51']['metrics']['volume_km3'];p=v['periods'][2]['total'];lo,hi=p['conditional_delta_range'];med=agg['all51']['metrics']['length_km']['periods'][2]['individual_change_pct_quantiles'][2]
    s=['<h1>51条冰川：LIA历史变化的主结果</h1><p class="lead">变化幅度 · 共同阶段与个体差异 · 条件性不确定性</p>',
       '<div class="note">研究对象是1500—2000年的已有历史模拟，不是spinup过程。51条统一分析；未重新择优、未运行新模拟、未修改接纳台账。下面是选定模型结果的描述，独立观测验证尚未建立。</div>',
       f'<h2>先看最重要的结果</h2><ul><li>1900—2000年，51条同期模型总体积从{p["value_start"]:.2f}降至{p["value_end"]:.2f} km³，减少{-p["delta"]:.2f} km³（{-p["change_pct"]:.2f}%）。</li><li>同一时期，逐条长度减少比例的中位数为{-med:.2f}%；这与总体积的百分比回答的是不同问题。</li><li>只替换已有14条多方案冰川，其他37条固定时，总体积减少量为{-hi:.2f}—{-lo:.2f} km³。这不是全部重建不确定性或置信区间。</li><li>不能赋予51条冰川一个统一最大年代：{summary["exact_repeated_length_maximum_count"]}条存在多个保存时点并列长度最大值，且部分近最大时段不连续。</li></ul>',
       '<h2>1. 变化了多少：固定时期，而不是把各自最大值拼成一个状态</h2><img src="01_sample_totals.png">']
    rows=[]
    early=v['periods'][0]['total'];middle=v['periods'][1]['total']
    s.append(f'<p class="note">历史主线：1500—1850同期总体积净变化{early["change_pct"]:+.2f}%，1850—1900为{middle["change_pct"]:+.2f}%，1900—2000为{p["change_pct"]:+.2f}%。这些是固定端点之间的净变化，期间仍有多次波动，不等于单调增长或退缩；早段还可能保留初始状态的影响。</p>')
    for metric,name,unit in zip(KEYS,NAMES,UNITS):
        for r in agg['all51']['metrics'][metric]['periods']:
            q=r['individual_change_pct_quantiles'];total=r.get('total')
            rows.append([name,f"{r['start']}—{r['end']}",f'{q[2]:+.2f} [{q[1]:+.2f}, {q[3]:+.2f}]',f"{r['shrinking_count']}/51",f"{total['delta']:+.3f} {unit}" if total else '长度不求和',f"{total['change_pct']:+.2f}%" if total else '—',f"{total['net_rate_per_year']:+.4f} {unit}/年" if total else '—'])
    s.append(table(['指标','比较时期','个体变化%：中位数 [四分位范围]','净减少条数','同期总量变化','同期总量变化%','时期平均净变化率'],rows))
    s.append('<p>表中变化=end−start，负值为减少；相对量以该时期起点为分母。四分位范围描述冰川之间的差异，不是不确定性区间。1850和1900是沿用的比较界线，不是根据这些曲线独立确定的阶段边界。1500—2000净变化不能替代“先增后减”等完整轨迹。</p>')
    s+=['<h2>2. 每条冰川：绝对量和相对量都保留</h2><img src="02_individual_changes.png">',f'<p>1900—2000年，体积净减少最多的5条占样本总净减少量的{summary["top5_volume_loss_contribution"]["share_of_net_loss"]:.1%}。因此总体积曲线不代表“平均冰川”，需要与逐条变化并列阅读。</p>']
    s+=['<h2>3. 什么时候变化：共同阶段与个体不同步</h2><img src="03_shared_timing.png">']
    rows=[]
    for key,name in zip(KEYS,NAMES):
        r=next(x for x in time if x['group']=='all51' and x['metric']==key and x['step']==10);n=r['near_variants'][1]
        rows.append([name,', '.join(str(int(y)) for y in n['max_fraction_years']),f"{n['max_fraction']:.1%}",', '.join(str(int(y)) for y in n['max_pixel_years']),f"{n['max_pixel_fraction']:.1%}"])
    s.append(table(['指标','近最大占比最高的10年网格时点','51条等权占比','像元等权最高时点','33像元等权占比'],rows))
    s.append('<p>这里统计“同一时点有多少条接近各自最大值”，不是统计独立年代证据。近最大范围、精确并列峰值和实际年代误差不是同一个概念。22条原始曲线每10年保存，29条逐年保存；插值不提高年代精度。</p><p><a href="../lia_51_stage_reliability_v1/index.html">完整时序图谱、逐条曲线和阈值敏感性</a> · <a href="../lia_51_same_pixel_pairs_v1/index.html">同气候像元配对结果</a></p>')
    s+=['<h2>4. 哪些结论得到备选方案支持，哪些仍未知</h2>']
    rows=[]
    for key,name in zip(KEYS,NAMES):
        c=summary['conditional_1900_2000_direction'][key];rows.append([name,f"{c['selected_shrink']}/51",f"{c['all_saved_cases_shrink']}/{c['multi_case_systems']}",'37条缺少多方案检验'])
    s.append(table(['1900—2000变化方向','选定曲线净减少','多方案冰川的所有已有案例均净减少','证据缺口'],rows))
    rows=[]
    for key,name,unit in zip(KEYS[1:],NAMES[1:],UNITS[1:]):
        for r in agg['all51']['metrics'][key]['periods']:
            p=r['total'];lo,hi=p['conditional_delta_range'];rows.append([name,f"{r['start']}—{r['end']}",f"{p['delta']:+.3f}",f"[{lo:+.3f}, {hi:+.3f}] {unit}"])
    s.append(table(['指标','时期','选定总量变化','已有方案条件性范围：同一方案两端配对'],rows))
    s.append('<p>这些范围在现有案例集合内取极值：每条冰川始终使用同一方案计算两个端点的差，再相加；不是把独立端点包络相减。不同冰川的极端方案可分别取值，不意味着这些组合同时满足同一额外气候约束。</p>')
    counts=summary['reliability_label_counts'];s.append(table(['沿用的综合诊断','条数'],[[LABELS[k],counts.get(k,0)] for k in LABELS]))
    s.append('<p>“已有组合内较稳定”只在此前指定的时序和幅度规则下成立，不能称为已验证的真实历史。37条未检验者继续保留，不被删除或默认为稳定。完整不确定性还包括气候重建、结构、初态与观测约束；本轮没有量化这些来源。</p>')
    rows=[]
    for group,r in agg.items():
        p=r['metrics']['volume_km3']['periods'][2]['total'];rows.append([COHORTS[group],r['n'],fmt(p['value_start']),fmt(p['value_end']),fmt(-p['delta']),fmt(-p['change_pct'])+'%'])
    s.append('<h2>5. 来源与修正政策敏感性</h2>'+table(['子集','条数','1900体积km³','2000体积km³','减少km³','减少比例'],rows)+'<p>统一51条为主。41/10、32/19仅作敏感性拆分，不能把它们之间的差异归因为气候政策效应；冰川属性、模型分支和校准参数仍然混杂。</p>')
    s.append('<h2>51条结果明细</h2><p>下表为1900—2000减少量（正值表示减少），点击编号查看原有完整历史曲线。窗口最大值只作补充，未被当作独立定年的LIA状态。</p>')
    rows=[]
    for r in individual:
        fields=[]
        for key in KEYS:
            p=r['metrics'][key]['periods'][2];fields.extend([fmt(-p['delta'],3),fmt(-p['change_pct'])])
        rows.append([f'<a href="../lia_51_stage_reliability_v1/glaciers/{r["system"]}.png">{r["system"]}</a>',*fields,r['case_count'],LABELS[r['reliability_label']]])
    s.append(table(['编号','长度km','长度%','面积km²','面积%','体积km³','体积%','已有案例数','条件性诊断'],rows))
    s.append('<h2>本轮可以说什么，暂时不能说什么</h2><ul><li>可以描述这51条选定模拟的变化幅度、共同阶段和个体差异；可以报告现有备选案例对这些描述的影响。</li><li>不能外推至191条或整个喜马拉雅；不能将拟合长度端点、最大长度再当作独立验证。</li><li>不能把模型长度变化直接等同于地图上的末端位移，或把静态V3体积与某年模拟体积相减。</li><li>不能据此识别气候与几何各自的因果贡献，也不能给出唯一的LIA最大年代。</li></ul>')
    s.append('<p class="note">下一项真正增加证据强度的工作，是盘点这51条是否有未参与校准的历史长度、面积或高程变化观测，区分可检验、仅有端点约束和无独立观测三类。本轮未替代或启动这一独立验证。</p>')
    files=['methods.json','individual_changes.json','cohort_aggregates.json','timing_summary.json','summary.json','validation.json','source_manifest.json','output_manifest.json']
    s.append('<details><summary>定义、数据与核查记录</summary><p>'+' · '.join(f'<a href="{f}">{f}</a>' for f in files)+'</p></details>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>51条冰川历史变化主结果</title><style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#f2f5f7;color:#293f4d;margin:0}main{max-width:1350px;margin:auto;padding:32px}h1{font-size:32px}h2{margin-top:44px}.lead{color:#497b91;font-size:20px}.note{padding:18px;background:#e3edf2;border-left:5px solid #40809a}img{width:100%;height:auto;background:white;margin:16px 0}.scroll{overflow:auto}table{border-collapse:collapse;background:white;white-space:nowrap;font-size:13px}th,td{border:1px solid #d6e0e5;padding:8px}th{background:#e1ebf0}a{color:#226e8d}details{background:white;padding:14px}</style><main>'+''.join(s)+'</main></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')

def main():
    if (OUT/'validation.json').exists():assert read(OUT/'validation.json')['status']!='pass','Finalized outputs are immutable; create a new version.'
    methods,curves,atlas,features,cases,data=load_inputs()
    individual=make_individual(curves,atlas,cases);agg=aggregate(curves,features,cases,individual);times=timing(curves,features,data['regional_activity.json']);summary=summarize(individual,agg,times,data)
    for name,x in [('individual_changes.json',individual),('cohort_aggregates.json',agg),('timing_summary.json',times),('summary.json',summary)]:save(name,x)
    figures(individual,agg,times);report(individual,agg,times,summary)
    for p,h in SOURCES.items():assert sha(ROOT/p)==h,p
    save('source_manifest.json',SOURCES);save('validation.json',dict(status='built-independent-verification-pending',source_count=len(SOURCES),authority_hashes={p:SOURCES[p] for p in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']},simulation_launches=0,raw_climate_access=False))
    save('output_manifest.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='output_manifest.json'})
    print(json.dumps(clean({k:summary[k] for k in ['period_volume_totals','conditional_1900_2000_direction','common_decadal_volume_maximum_years','reliability_label_counts']}),ensure_ascii=False))

if __name__=='__main__':main()
