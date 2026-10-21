"""Register, run and report four bounded adaptive GPU BP comparisons."""
from .solver_acceleration_v1 import *
from . import gpu_long_storage_v1 as st
from .historical_climate_v1 import annual
from .gpu_model_v5 import code_hashes as historical_hashes
from types import SimpleNamespace
import subprocess

NAME='polim_v33_adaptive_bp_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v33-adaptive-bp-v1'
SOURCE=R/'runs/v32r4-historical-1470-1570-v1'
REPORT=R/'reports/adaptive-bp-v1'


def code_hashes():
    return {**historical_hashes(),**{(R/n).relative_to(PROJECT).as_posix():digest(R/n) for n in ['adaptive_bp_comparison_v1.py','adaptive_bp_worker_v1.py','adaptive_bp_solve_v1.py']}}


def transition(stage,evidence,fact):
    validate()
    sp=PROJECT/'LIA_CURRENT_STATE.json'; raw=sp.read_bytes(); s=json.loads(raw)
    fp=PROJECT/'LIA_PROJECT_FACTS.md'; fb=fp.read_bytes()
    assert s['current_focus']['state']!='decision-required'
    assert digest(fp).upper()==s['scientific_contract']['sha256'].upper()
    authorized=stage=='authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b=dict(status=stage,launch_authorized=authorized,allowed_stages=['four-one-year-comparisons','checkpoint-restart'] if authorized else [],plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,production_replacement_authorized=False,full_stokes_authorized=False)
    number=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
    addition=f'\n\n## {number}. System2387 adaptive BP performance\n\n- {fact} Evidence `{ref(evidence)["path"]}`.\n'
    fh=hashlib.sha256(fb+addition.encode()).hexdigest().upper()
    route=read(PROJECT/s['authoritative_route']['path']);suffix=f'+{NAME}-{stage}'
    route['version']+=suffix;route[NAME]=b
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/f'route-{stage}.json';write_json(rp,route)
    n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['scientific_contract']['sha256']=fh;n['state_id']+=suffix;n['updated_at']=datetime.now(timezone.utc).isoformat()
    assert sp.read_bytes()==raw and fp.read_bytes()==fb
    temp=sp.with_name(f'.LIA_CURRENT_STATE.adaptive-{os.getpid()}.tmp');write_json(temp,n)
    with fp.open('ab') as f:f.write(addition.encode());f.flush();os.fsync(f.fileno())
    os.replace(temp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} — adaptive BP {stage}\n\n{fact}\n')
    write_json(E/f'{stage}-validation.json',validate())


def prepare():
    validate();assert not RUN.exists() and not E.exists()
    RUN.mkdir();E.mkdir()
    source_contract=read(SOURCE/'contract.json')['contract'];z=arrays(SOURCE/'input.npz');climate=arrays(SOURCE/'climate.npz');mesh=arrays(GRIDS/'grid-50m/mesh.npz')
    parameters=read(SOURCE/'config.json')['parameters'];sources=[SOURCE/n for n in ['input.npz','climate.npz','config.json','contract.json']]+[GRIDS/'grid-50m/mesh.npz']
    cases=[]
    for year in [1480,1550]:
        first=read(SOURCE/f'year-{year}.json')['step'];last=read(SOURCE/f'year-{year+1}.json')['step']
        frames={key:f'frame-{i:06d}' for key,i in [('start_frame',first),('previous_frame',first-1),('end_frame',last)]}
        for frame in frames.values():
            st.restore(SOURCE/frame,source_contract)
            sources.extend(SOURCE/frame/n for n in ['state.npz','metadata.json','seal.json'])
        state,_=st.restore(SOURCE/frames['start_frame'],source_contract)
        a=original.prepare({**mesh,'thickness':state['h']});rr,ee,k,force=original.evaluate(a,state['u'],True,A=z['A'])
        fixture=dict(hvp=k@z['direction'],smb=annual(z['bed']+state['h'],climate,parameters,year,SimpleNamespace(xp=np,array=np.asarray)))
        for mode in ['exact','lagged']:
            folder=RUN/f'{year}-{mode}';folder.mkdir()
            cfg=dict(case_id=folder.name,source_root=str(SOURCE),start_year=year,end_year=year+1,mode=mode,parameters=parameters,**frames)
            st.write(folder/'config.json',cfg)
            with (folder/'fixture.npz').open('xb') as f:np.savez(f,**fixture)
            cases.append(dict(config=ref(folder/'config.json'),fixture=ref(folder/'fixture.npz')))
    plan=dict(schema=NAME,user_authorization='User says start after proposal to optimize solver using saved windows; no reconstruction inverse run yet.',system=2387,grid_m=50,cases=cases,physics='Unchanged fixed-bed/fixed-A/no-slip BP, original annual historical TI climate, fixed footprint and conservative transport, float64, 30-day cap/CFL .35.',optimization='Reuse SPD AMG between wet-support changes with exact current tangent diagonal congruence scaling each Newton; rebuild once on 200CG/true residual failure. Energy-and-residual screened temporal secant and thickness predictors. Stop at accepted line-search residual instead of unused terminal tangent.',lagged_variant='Only lagged cases: hold velocity at most60days and only if current exact normalized BP residual<=1e-3; exact1e-6 at complete solves and year endpoints. Explicit experimental momentum imbalance; no default replacement.',gates=dict(exact_max_H_m=.001,exact_u_L2=.001,lagged_max_H_m=.25,lagged_H_weighted_RMS_m=.05,lagged_u_L2=.01,lagged_volume_relative=.001,lagged_export_relative=.01,mass_relative=1e-10,complete_BP=1e-6,operator_relative=1e-9),reference='Read original v32r4 saved states and timings; no reference dynamics replay. Compare full-year endpoints and same-start incremental export; historical times not simultaneous hardware-controlled.',resources=dict(GPU_workers=1,allocator_fraction=.10,case_seconds=900,estimate_total_minutes=[3,10]),recovery='Atomic accepted/pending and every5Newton; two-step intentional process restart in first exact case, injected unavailable status; no completed cases replayed. One identified technical retry in new version if required. Independent cases continue on failure.',implementation=code_hashes(),sources={p.relative_to(PROJECT).as_posix():digest(p) for p in sources})
    write_json(E/'plan.frozen.json',plan);contract=seal(plan)
    for case in cases:
        folder=(PROJECT/case['config']['path']).parent
        st.write(folder/'contract.json',dict(contract=contract,plan=ref(E/'plan.frozen.json')))
    transition('authorized',E/'plan.frozen.json','User authorizes four bounded50m windows1480-1481/1550-1551: exact adaptiveAMG+screenedpredictor and explicit residual-limited60dayvelocityreuse. Same BP1e-6 at full solves; lagged1e-3 only between solves, separate engineering comparison gates frozen. Original saved baseline retained, no completed baseline rerun, no default/mainline/count change. One combined fixture/storage/status test and firstcase two-step process restart; 3-10min estimate.')


def launch(folder,stop=None):
    validate();plan=read(E/'plan.frozen.json');assert plan['implementation']==code_hashes()
    assert read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for p,h in plan['sources'].items():assert digest(PROJECT/p)==h,p
    for case in plan['cases']:
        for v in case.values():assert digest(PROJECT/v['path']).upper()==v['sha256'].upper()
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'));env=os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=os.pathsep.join([str(site),str(PROJECT)]);env['POLIM_TORCH_SITE']=str(site)
    cmd=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.adaptive_bp_worker_v1',str(folder)]+([str(stop)] if stop else [])
    tag=time.time_ns()
    with (folder/f'worker-{tag}.log').open('x') as f:
        process=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
        st.write(folder/f'launch-{tag}.json',dict(pid=process.pid,command=cmd,start=datetime.now(timezone.utc).isoformat(),checkpoint='each pending/accepted and5Newton',estimate_minutes=[1,3],max_seconds=900))
        process.wait(timeout=960)
    files=sorted(folder.glob('result-*.json'))
    if not files: return dict(status='failed',error='Worker exited before result; see log')
    return st.read(files[-1])


def finish():
    mesh=arrays(GRIDS/'grid-50m/mesh.npz');z=arrays(SOURCE/'input.npz');source_contract=read(SOURCE/'contract.json')['contract'];out=[]
    historical=read(PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_v32r4_historical_1470_1570_v1/completion.json')['rows']
    for folder in sorted(RUN.iterdir()):
        if not folder.is_dir() or not (folder/'config.json').exists():continue
        cfg=read(folder/'config.json');contract=read(folder/'contract.json')['contract']
        phases=[read(p) for p in sorted(folder.glob('result-*.json'))]
        row=dict(case=folder.name,status=phases[-1]['status'] if phases else 'failed')
        if row['status']=='completed':
            checks=[st.restore(p,contract) for p in sorted(folder.glob('frame-*')) if p.is_dir()]
            final,meta=checks[-1];initial,im=st.restore(SOURCE/cfg['start_frame'],source_contract);reference,rm=st.restore(SOURCE/cfg['end_frame'],source_contract)
            a=original.prepare({**mesh,'thickness':final['h']});r,_,_,f=original.evaluate(a,final['u'],False,A=z['A'])
            dh=final['h']-reference['h'];dual=z['geo_dual_area'];volume=float(reference['h']@dual)
            export_ref=rm['exported']-im['exported'];rows=[m['row'] for _,m in checks if 'row' in m]
            row.update(H_max_m=float(abs(dh).max()),H_RMS_m=float(np.sqrt((dh*dh)@dual/dual.sum())),u_relative_L2=float(np.linalg.norm(final['u']-reference['u'])/np.linalg.norm(reference['u'])),volume_relative=abs(float(dh@dual))/volume,export_relative=abs(meta['exported']-export_ref)/max(abs(export_ref),1.),CPU_BP_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.)),max_mass_relative=max(x['mass_relative'] for x in rows),max_held_BP=max(x['BP_residual'] for x in rows),advance_seconds=sum(p.get('advance_seconds',0) for p in phases),setup_seconds=sum(p.get('setup_seconds',0) for p in phases),hierarchy_seconds=sum(p.get('hierarchy_seconds',0) for p in phases),checkpoint_seconds=sum(p.get('checkpoint_seconds',0) for p in phases),rebuilds=sum(p.get('rebuilds',0) for p in phases),Newton=sum(x['Newton'] for x in rows),CG=sum(x['CG'] for x in rows),skipped=sum(x['skipped'] for x in rows),steps=len(rows),predictor_selections={k:sum(x['predictor']==k for x in rows) for k in ['held','time-secant','thickness3','thickness4']})
            baseline=[r for r in historical if cfg['start_year']<r['year']<=cfg['end_year']+1e-9]
            row.update(reference_seconds=sum(r['seconds'] for r in baseline),reference_Newton=sum(r['Newton'] for r in baseline),reference_CG=sum(r['CG'] for r in baseline))
            row['speedup']=row['reference_seconds']/row['advance_seconds']
            common=row['CPU_BP_residual']<=1e-6 and row['max_mass_relative']<=1e-10
            row['passed']=common and (row['H_max_m']<=.001 and row['u_relative_L2']<=.001 and row['max_held_BP']<=1e-6 if cfg['mode']=='exact' else row['H_max_m']<=.25 and row['H_RMS_m']<=.05 and row['u_relative_L2']<=.01 and row['volume_relative']<=.001 and row['export_relative']<=.01 and row['max_held_BP']<=1e-3)
        else:
            row['error']=phases[-1].get('error') if phases else 'missing result'
            row['passed']=False
        out.append(row)
    result=dict(status='completed',cases=out,all_passed=all(r['passed'] for r in out),default_changed=False,baseline_replayed=False,timing_note='Historical saved timings, includes all new advance/checkpoint/rebuild times and both phases of intentional restart; setup separately reported.')
    write_json(E/'completion.json',result)
    transition('completed',E/'completion.json',f'Four scoped adaptive BP comparisons accounted for; all engineering gates passed={result["all_passed"]}. Original CPU final residual and immutable checkpoints checked together. No production selection or scientific acceptance; see measured errors/timings, retained baseline not replayed.')
    REPORT.mkdir()
    cols=['case','passed','reference_seconds','advance_seconds','speedup','Newton','reference_Newton','rebuilds','skipped','H_max_m','H_RMS_m','u_relative_L2','CPU_BP_residual']
    import html
    table='<tr>'+''.join('<th>'+k+'</th>' for k in cols)+'</tr>'
    for row in out:
        table+='<tr>'+''.join('<td>'+html.escape(f'{row[k]:.4g}' if isinstance(row.get(k),float) else str(row.get(k,'')))+'</td>' for k in cols)+'</tr>'
    (REPORT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>BP 求解过程优化</title><style>body{font:16px system-ui;margin:40px;color:#173141}table{border-collapse:collapse;font-size:13px}td,th{padding:9px;border:1px solid #ccd9df}th{background:#eaf2f6}p{max-width:1000px;line-height:1.7}</style><h1>50 m BP 求解过程优化对比</h1><p>两个一年窗口，读取原始历史轨迹作为参照。exact 保持每步 BP 1e-6；lagged 允许不超过 60 天且残差不超过 1e-3 的速度复用，年度终点仍完整求解。当前主力版本未替换。</p><table>'+table+'</table><p>speedup = 原记录推进耗时 / 本次推进耗时。新结果包含检查点及预条件器重建；初始化单独记录。历史计时不是同时硬件受控测试，不能直接外推千年加速。</p><p>完整数据：completion.json</p>',encoding='utf-8')
    write_json(REPORT/'completion.json',result)
    print(json.dumps(result,indent=2),flush=True)


def main():
    if sys.argv[-1]=='prepare':prepare();return
    if sys.argv[-1]=='finish':finish();return
    from .runner import single_writer
    with single_writer(RUN):
        for folder in sorted(RUN.iterdir()):
            if not folder.is_dir() or not (folder/'config.json').exists():continue
            results=sorted(folder.glob('result-*.json'))
            if results and read(results[-1])['status']!='intentional-checkpoint-stop':continue
            if folder.name=='1480-exact' and not results:
                result=launch(folder,2)
                if result['status']!='intentional-checkpoint-stop':continue
            launch(folder)
        finish()


if __name__=='__main__':main()
