"""One preregistered fixed-A control; reuse the completed coupled run unchanged.

No new momentum/transport implementation, no parameter fitting, no locked climate.
prepare -> launch -> finish. All numerical results and transactions are immutable.
"""
import argparse
import base64
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import numpy as np
from .v3.runner import PROJECT, digest, seal, write_json, implementation

NAME = 'polim_lia_v3_system2387_thermal_attribution_v1'
EVIDENCE = PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
SOURCE = PROJECT/'polim_cpu/runs/lia-2387-v3-pilot-v1'
OUT = PROJECT/'polim_cpu/runs/lia-2387-v3-fixed-a-control-v1'
REPORT = PROJECT/'polim_cpu/reports/thermal-attribution-v1'
VALIDATOR = PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'


def now(): return datetime.now(timezone.utc).isoformat()
def ref(p): return {'path':Path(p).resolve().relative_to(PROJECT).as_posix(), 'sha256':digest(p).upper()}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def validate():
    check = subprocess.run([sys.executable,str(VALIDATOR)],cwd=PROJECT,capture_output=True,text=True)
    if check.returncode: raise RuntimeError(check.stdout+check.stderr)
    return json.loads(check.stdout)


def transition(stage, evidence):
    validate()
    path = PROJECT/'LIA_CURRENT_STATE.json'; original = path.read_bytes(); state = json.loads(original)
    if state['current_focus']['state']=='decision-required': raise RuntimeError('Mainline decision required')
    authorized = stage=='pilot'
    if authorized and NAME in state.get('research_branches',{}): raise RuntimeError('Branch already registered')
    if not authorized and not state['research_branches'][NAME]['launch_authorized']: raise RuntimeError('Branch already closed')
    record = {'schema':'polim-thermal-attribution-registration-v1','stage':stage,'created_at':now(),
              'source_state_sha256':digest(path),'source_route':state['authoritative_route'],
              'evidence':ref(evidence),'user_authorization':['采用这些假设，完成热力耦合试算','继续'],
              'scope':'Continue high-speed diagnosis with one fixed-A control, reusing original coupled result',
              'dynamics_performed_by_registration':False,'accepted_count_effect':0}
    record_path = EVIDENCE/f'{stage}-registration.json'; write_json(record_path,record)
    branch = {'status':'short-control-authorized' if authorized else 'terminal',
              'launch_authorized':authorized,'allowed_stages':['real-pilot','checkpoint-continuation'] if authorized else [],
              'system':2387,'evidence':ref(record_path),'plan':ref(EVIDENCE/'plan.frozen.json'),
              'accepted_count_effect':0,'production_replacement_authorized':False,'automatic_population_expansion':False,
              'next_legal_action':'Execute only the single frozen control' if authorized else 'Read-only completed attribution; no retry or expansion'}
    route = read(PROJECT/state['authoritative_route']['path']); route['version']+=f'+polim-thermal-attribution-{stage}'; route[NAME]=branch
    target = EVIDENCE/f'route-{stage}.json'; write_json(target,route)
    updated = copy.deepcopy(state); updated['research_branches'][NAME]=branch
    updated['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+=f'+polim-thermal-attribution-{stage}'; updated['updated_at']=now()
    for key in ('counts','population','current_focus','next_action','scientific_contract'):
        if updated[key]!=state[key]: raise RuntimeError('Unexpected mainline change')
    if path.read_bytes()!=original: raise RuntimeError('Concurrent state change')
    temporary=path.with_name(f'.LIA_CURRENT_STATE.thermal.{os.getpid()}.tmp')
    write_json(temporary,updated); os.replace(temporary,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {now()} — PoLIM thermal attribution {stage}\n\nUser requested continuation after the completed thermal pilot. One fixed-A diagnostic control is registered under `{NAME}`; original coupled results are reused. Temperature and basal melting remain active; only the rheological temperature/water feedback is disabled. No observational calibration or model promotion. Mainline fields, facts, and 58/191 accepted count remain unchanged. Evidence: `{ref(record_path)["path"]}`.\n')
    write_json(EVIDENCE/f'{stage}-controller-validation.json',validate())


def physical_fingerprint(config, code):
    c=copy.deepcopy(config)
    for k in ('case_id','research_branch','input_npz'): c.pop(k,None)
    return seal({'config':c,'implementation':code})


def state_at(folder):
    with np.load(folder/'state.npz',allow_pickle=False) as d: return {k:d[k].copy() for k in d.files}


def point_summary(d, scale):
    ij=np.unravel_index(abs(d['u']).argmax(),d['u'].shape); i=int(ij[0]); wet=d['h']>0
    return {'maximum_speed_m_year':float(abs(d['u'][ij])),'peak_column':i,'peak_layer':int(ij[1]),
            'peak_x_m':float(d['x'][i]),'peak_thickness_m':float(d['h'][i]),
            'peak_surface_slope':float(np.gradient(d['bed']+d['h'],d['x'])[i]),
            'peak_column_A_over_supplied_A':(d['A'][i]/scale).tolist(),
            'wet_A_over_supplied_A_minmax':[float(d['A'][wet].min()/scale),float(d['A'][wet].max()/scale)]}


def prepare():
    validate(); EVIDENCE.mkdir(parents=True,exist_ok=False)
    old=read(SOURCE/'run.json'); code=implementation()
    if code!=old['payload']['implementation']: raise RuntimeError('Existing implementation changed')
    fixture=PROJECT/'polim_cpu/runs/v3-fixture-final/verification.json'
    f=read(fixture)
    if f['passed']!=7 or f['failed']!=0: raise RuntimeError('Previous engineering evidence unavailable')
    fixture_runs=list(fixture.parent.rglob('run.json'))
    if not fixture_runs or any(read(p)['payload']['implementation']!=code for p in fixture_runs):
        raise RuntimeError('Fixture implementation differs')
    config=copy.deepcopy(old['payload']['config']); config['case_id']='lia-2387-v3-fixed-A-diagnostic'
    config['research_branch']=NAME; config['input_npz']=os.path.relpath(SOURCE/'inputs.npz',EVIDENCE)
    config['parameters']['thermal_feedback']=False
    fingerprint=physical_fingerprint(config,code)
    scanned=[]
    for p in (PROJECT/'polim_cpu/runs').rglob('run.json'):
        m=read(p)
        if 'payload' not in m or 'implementation' not in m['payload']: continue
        c=m['payload']['config']
        if c.get('system')!=2387: continue
        fp=physical_fingerprint(c,m['payload']['implementation']); scanned.append({'manifest':ref(p),'fingerprint':fp})
        if fp==fingerprint: raise RuntimeError('Duplicate numerical contract: '+str(p))
    scale=config['parameters']['glen_a_pa_3_s_1']*31536000
    checkpoints=sorted(SOURCE.glob('step-*'))
    audit={'schema':'polim-saved-speed-audit-v1','original_run':ref(SOURCE/'run.json'),
           'initial':point_summary(state_at(checkpoints[0]),scale),'final':point_summary(state_at(checkpoints[-1]),scale),
           'code_observation':'v3 maps supplied scalar A to pressure-corrected -10 C; original 1-D source uses scalar A directly with no temperature law',
           'interpretation':'The -10 C anchoring is an experimental assumption; high slope and A both enter dynamics. Spatial parameter ratios alone are not a causal velocity attribution.',
           'original_scalar_driver':ref(VALIDATOR.parent/'torch_flowline_1d_batched_semiimplicit_v1.py')}
    write_json(EVIDENCE/'saved-state-audit.json',audit); write_json(EVIDENCE/'case.json',config)
    plan={'schema':'polim-thermal-attribution-plan-v1','branch':NAME,'created_at':now(),
          'config':ref(EVIDENCE/'case.json'),'config_sha256':seal(config),'implementation':code,
          'driver':ref(Path(__file__)),'source_input':ref(SOURCE/'inputs.npz'),'original_run':ref(SOURCE/'run.json'),
          'engineering_evidence_reused':ref(fixture),'duplicate_scan':scanned,'physical_fingerprint':fingerprint,
          'user_authorization':['采用这些假设，完成热力耦合试算','继续'],
          'question':'How much does the experimental temperature/water-dependent A mapping change speed?',
          'sole_physical_change':'thermal_feedback=false; retain supplied A=7.538801981725612e-23 Pa^-3 s^-1 everywhere',
          'thermal_state':'Climate-derived initial temperature, 0.05 W/m2 geothermal, heat and basal melt still computed',
          'scope':'one system2387, year1441, duration0.1 year, all4094cells, same9layers and solver; no new coupled rerun',
          'comparison':'Initial identical-state speeds isolate A mapping; final differences include evolving geometry, enthalpy and melt; do not label final differences a pure direct effect',
          'acceptance':'numerical completion and integrity only; no target speed, observed calibration, equilibrium or historical acceptance',
          'resources':{'device':'cuda','cpu_threads':1,'estimated_seconds':[5,180]},
          'checkpoint_cadence':'every accepted step, max dt0.005 year, CFL may shorten',
          'stop_conditions':['nonfinite','negative thickness','Newton/thermal failure','checksum/controller mismatch'],
          'retry_policy':'none without a newly diagnosed technical repair transaction','scientific_acceptance':False,'accepted_count_effect':0}
    write_json(EVIDENCE/'plan.frozen.json',plan); transition('pilot',EVIDENCE/'plan.frozen.json')
    print(json.dumps({'status':'control-authorized','plan':ref(EVIDENCE/'plan.frozen.json'),'engineering_checks_reused':7}))


def launch():
    validate(); plan=read(EVIDENCE/'plan.frozen.json')
    if digest(__file__)!=plan['driver']['sha256'].lower(): raise RuntimeError('Experiment driver changed')
    if OUT.exists() or (EVIDENCE/'launch.json').exists(): raise RuntimeError('Control already launched')
    command=[sys.executable,'-m','polim_cpu.v3','run',str(EVIDENCE/'case.json'),'--out',str(OUT),'--device','cuda','--no-report']
    started=time.perf_counter()
    with (EVIDENCE/'stdout.log').open('xb') as out, (EVIDENCE/'stderr.log').open('xb') as err:
        process=subprocess.Popen(command,cwd=PROJECT,stdout=out,stderr=err)
        write_json(EVIDENCE/'launch.json',{'pid':process.pid,'command':command,'started_at':now(),'output_root':str(OUT),
                   'checkpoint_cadence':plan['checkpoint_cadence'],'estimated_seconds':[5,180],'stop_conditions':plan['stop_conditions']})
        rc=process.wait()
    write_json(EVIDENCE/'process-exit.json',{'exit_code':rc,'ended_at':now(),'wall_seconds':time.perf_counter()-started,
               'classification':'process-completed-pending-audit' if rc==0 else 'technical-failure'})
    if rc:
        transition('terminal',EVIDENCE/'process-exit.json'); raise RuntimeError('Control failed; inspect retained stderr')
    print(json.dumps({'exit_code':rc,'output':str(OUT)}))


def audit_run(root):
    manifest=read(root/'run.json'); contract=manifest['contract']
    if seal(manifest['payload'])!=contract: raise RuntimeError('Manifest seal mismatch')
    if digest(root/'inputs.npz')!=manifest['payload']['input_sha256']: raise RuntimeError('Input mismatch')
    folders=sorted(root.glob('step-*')); initial=state_at(folders[0]); refs=[]
    for step,folder in enumerate(folders):
        meta=read(folder/'checkpoint.json'); marker=read(folder/'commit.json')
        if digest(folder/'checkpoint.json')!=marker['metadata_sha256'] or digest(folder/'state.npz')!=meta['state_sha256'] or meta['contract']!=contract: raise RuntimeError('Checkpoint seal mismatch')
        if meta['step']!=step or len(meta['history'])!=step+1: raise RuntimeError('Checkpoint sequence mismatch')
        d=state_at(folder)
        if any(not np.isfinite(v).all() for v in d.values()) or (d['h']<0).any(): raise RuntimeError('Invalid state')
        if any(not np.array_equal(initial[k],d[k]) for k in ('x','bed','width','cell_width')): raise RuntimeError('Source geometry changed')
        if not manifest['payload']['config']['parameters']['thermal_feedback']:
            A=manifest['payload']['config']['parameters']['glen_a_pa_3_s_1']*31536000
            if not np.all(d['A']==A): raise RuntimeError('Control A is not constant')
        refs.append({'step':step,'state':ref(folder/'state.npz'),'metadata':ref(folder/'checkpoint.json')})
    h=meta['history']; mass=max(abs(v['mass_error_m3']) for v in h); residual=max(v['momentum_relative_residual'] for v in h)
    if abs(meta['clock_years']-.1)>1e-12 or mass/max(h[0]['volume_m3'],1)>1e-10 or residual>1e-6: raise RuntimeError('Completion/mass/residual check failed')
    return initial,d,h,{'manifest':ref(root/'run.json'),'checkpoints':refs,'steps':meta['step'],
             'initial':h[0],'final':h[-1],'max_mass_error_m3':mass,'max_momentum_relative_residual':residual,
             'full_domain_cells':len(d['h']),'all_checkpoint_integrity_pass':True}


def finish():
    if read(EVIDENCE/'process-exit.json')['exit_code']!=0: raise RuntimeError('No successful control')
    ai,af,ah,aa=audit_run(SOURCE); bi,bf,bh,ba=audit_run(OUT)
    equal=('x','bed','width','cell_width','h','E','smb','surface_c','melt')
    if not all(np.array_equal(ai[k],bi[k]) for k in equal): raise RuntimeError('Initial comparison is confounded')
    am=read(SOURCE/'run.json')['payload']; bm=read(OUT/'run.json')['payload']
    if am['implementation']!=bm['implementation'] or am['runtime']!=bm['runtime']: raise RuntimeError('Code/runtime comparison mismatch')
    ac=copy.deepcopy(am['config']); bc=copy.deepcopy(bm['config'])
    for c in (ac,bc):
        for k in ('case_id','research_branch','input_npz'): c.pop(k,None)
    bc['parameters']['thermal_feedback']=True
    if ac!=bc: raise RuntimeError('Additional physical config difference')
    r0=ah[0]['max_speed_m_year']/bh[0]['max_speed_m_year']; r1=ah[-1]['max_speed_m_year']/bh[-1]['max_speed_m_year']
    result={'schema':'polim-thermal-attribution-result-v1','created_at':now(),'status':'completed',
            'plan':ref(EVIDENCE/'plan.frozen.json'),'original':aa,'fixed_A_control':ba,
            'identical_initial_fields':list(equal),'same_code_and_runtime':True,'sole_physical_config_difference':'thermal_feedback',
            'initial_maximum_speed_ratio_coupled_over_control':r0,'final_maximum_speed_ratio_coupled_over_control':r1,
            'initial_peak_reduction_percent':100*(1-1/r0),'final_peak_reduction_percent':100*(1-1/r1),
            'interpretation':'Initial contrast isolates prescribed rheological mapping on identical geometry and thermal state. Final contrast is the total trajectory response, including geometry/thermal/melt changes. Maxima can occur at different nodes.',
            'conclusion':'Experimental temperature mapping amplifies speed; fixed-A control also retains high speeds. No physical calibration is established by reducing speed.',
            'scientific_acceptance':False,'accepted_count_effect':0}
    REPORT.mkdir(parents=True,exist_ok=False)
    render(result,ai,af,bi,bf,ah,bh)
    result['report']=ref(REPORT/'index.html'); result['figure']=ref(REPORT/'comparison.png')
    write_json(EVIDENCE/'terminal.json',result); transition('terminal',EVIDENCE/'terminal.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('original','fixed_A_control')},indent=2))


def render(r,ai,af,bi,bf,ah,bh):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors=['#bc552b','#206b9b']; labels=['Temperature-dependent A','Fixed supplied A']
    for history,color,label in zip((ah,bh),colors,labels):
        axes[0,0].plot([v['clock_years'] for v in history],[v['max_speed_m_year']/1000 for v in history],color=color,label=label)
        axes[1,1].plot([v['clock_years'] for v in history],[(v['volume_m3']-history[0]['volume_m3'])/1e6 for v in history],color=color,label=label)
    take=slice(0,110); x=ai['x'][take]/1000
    for initial,final,color,label in zip((ai,bi),(af,bf),colors,labels):
        for ax,d in ((axes[0,1],initial),(axes[1,0],final)):
            ax.plot(x,abs(d['u'][take]).max(axis=1)/1000,color=color,label=label)
    axes[0,0].set(title='Maximum speed through time',xlabel='Time (years)',ylabel='Speed (km/year)')
    axes[0,1].set(title='Initial state: identical geometry and temperature',xlabel='Distance (km)',ylabel='Column maximum speed (km/year)')
    axes[1,0].set(title='Final state: total evolving response',xlabel='Distance (km)',ylabel='Column maximum speed (km/year)')
    axes[1,1].set(title='Volume change from common initial state',xlabel='Time (years)',ylabel='Volume change (million m3)')
    for ax in axes.flat: ax.legend(); ax.grid(alpha=.15)
    fig.suptitle('System 2387 | one fixed-A control | 0.1 year | no calibration',fontsize=15)
    fig.savefig(REPORT/'comparison.png',dpi=150); plt.close(fig)
    a,b=r['original'],r['fixed_A_control']; png=base64.b64encode((REPORT/'comparison.png').read_bytes()).decode()
    rows=''.join(f'<tr><td>{label}</td><td>{v1}</td><td>{v2}</td></tr>' for label,v1,v2 in [
        ('初始最大流速（m/年）',f'{ah[0]["max_speed_m_year"]:,.1f}',f'{bh[0]["max_speed_m_year"]:,.1f}'),
        ('最终最大流速（m/年）',f'{ah[-1]["max_speed_m_year"]:,.1f}',f'{bh[-1]["max_speed_m_year"]:,.1f}'),
        ('时间步 / 断点',f'{a["steps"]} / {a["steps"]+1}',f'{b["steps"]} / {b["steps"]+1}'),
        ('最大累计质量误差（m³）',f'{a["max_mass_error_m3"]:.3e}',f'{b["max_mass_error_m3"]:.3e}'),
        ('最终体积变化（m³）',f'{ah[-1]["volume_m3"]-ah[0]["volume_m3"]:,.1f}',f'{bh[-1]["volume_m3"]-bh[0]["volume_m3"]:,.1f}')])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>系统 2387 · 热力反馈对照</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 22px;background:#f5f7fa;color:#183047;font:16px/1.75 system-ui}}.tag{{color:#8b4b29}}h1{{line-height:1.3}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px}}.card,section{{background:white;border-radius:14px;padding:22px;margin:14px 0}}b.metric{{display:block;font-size:30px}}img{{width:100%;border-radius:14px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #dce5ed}}a{{color:#206b9b}}</style>
<p class="tag">第三版 · 已完成一次固定参数对照 · 研究试算</p><h1>温度软化放大了流速，关闭反馈后仍偏大</h1>
<p>两次结果使用相同的 4094 格原始剖面、气候、初始冰温与数值方法。新增对照只把流动参数 A 固定为来源值；温度、含水率、应变热和底部融化仍参与计算。原热力耦合结果直接复用。</p>
<div class="cards"><div class="card"><b class="metric">{r['initial_maximum_speed_ratio_coupled_over_control']:.2f} 倍</b>相同初始状态下的最大流速比</div><div class="card"><b class="metric">{bh[-1]['max_speed_m_year']/1000:.2f} km/年</b>固定参数对照的最终最大流速</div><div class="card"><b class="metric">0.1 年</b>两次试算的共同时间范围</div></div>
<section><h2>结果对比</h2><table><tr><th>指标</th><th>温度反馈试算</th><th>固定 A 对照</th></tr>{rows}</table></section>
<img src="data:image/png;base64,{png}" alt="两种流变设置下的速度、沿程分布和体积变化对比">
<section><h2>这次能够确认什么</h2><p>原实现把来源 A 值锚定在压力修正后的 −10°C，再叠加温度和含水率软化。来源的一维模型则直接使用标量 A。这一额外映射能明显改变流速，不能视为已经完成跨模型等价校准。</p>
<p>初始状态完全相同，因此初始速度对照隔离了流变映射的影响。最终最大流速比为 {r['final_maximum_speed_ratio_coupled_over_control']:.2f}，包含了冰厚、温度与融化在各自轨迹上的变化；两组最大值也可能来自不同位置。不能把这个最终比值解释为纯温度的直接贡献。</p>
<p>固定 A 对照仍有较高流速。原始剖面局部很陡，流动参数也从不同动量模型中继承；目前未分离这两者的贡献。此结果不构成观测一致性、平衡态或 LIA 历史重建验收，也不据此自动选择新的默认参数。</p></section>
<section><h2>假设与保存</h2><p>继续使用已确认的气候冰温初始化和 0.05 W/m² 地热通量。完整原始物理域保留，所有断点通过哈希、有限值、非负冰厚和质量收支合并检查；复用已通过的 7 项工程检查。</p>
<p><a href="../../runs/lia-2387-v3-pilot-v1/index.html">原始热力试算</a> · <a href="../../../validation_data/results/paleo_surface_validation_v0/{NAME}/terminal.json">完整对照证据</a></p></section></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=('prepare','launch','finish'))
    globals()[parser.parse_args().stage]()
