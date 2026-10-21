from .gpu_long_trial_v1 import *
def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['status'] in ['completed','failed']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-two-grid-pilot','planned-process-restart','two-grids-to-year100','conditional-working-solver-promotion'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,working_BP_solver_selected=(stage=='working-selected'),working_settings=ref(E/'working-settings.json') if stage=='working-selected' else None,next_legal_action='Run50/100m GPU handover toyear100' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 GPU long handover\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v31-{stage}'
    route['version'] += suffix; route[NAME] = b
    if 'scientific_contract' in route: route['scientific_contract'] = {**s['scientific_contract'],'sha256':fh}
    rp = E/f'route-{stage}.json'; write_json(rp,route)
    n = copy.deepcopy(s); n['research_branches'][NAME] = b
    n['authoritative_route'] = {**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']}
    n['state_id'] += suffix; n['updated_at'] = datetime.now(timezone.utc).isoformat()
    n['scientific_contract']['sha256'] = fh
    if sp.read_bytes() != raw or fp.read_bytes() != fb: raise RuntimeError('Concurrent state update')
    temp = sp.with_name(f'.LIA_CURRENT_STATE.acceleration.{os.getpid()}.tmp'); write_json(temp,n)
    if facts:
        with fp.open('ab') as f: f.write(addition.encode('utf-8')); f.flush(); os.fsync(f.fileno())
    os.replace(temp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser agreed to50/100m longhandover withaccuracy andrestart checks,then selectas workingBPsolver onlyifbothpass. NoOGGM191mainline/countchange. Same30day,float64,finalBP1e-6;annualAMGrefresh fromcurrentstate. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())

def main():
    validate();out=read(E/'completion.json');passed=out['status']=='completed' and all(c['passed'] for c in out['cases'].values())
    REPORT.mkdir(exist_ok=False);summaries={}
    for dx,c in out['cases'].items():
        if not c['passed']:
            summaries[dx]=dict(passed=False,error=c.get('error','Annualaccuracygate failed'));continue
        checks=c['checks'];phases=c['phases'];rows=[r for p in phases for r in p['rows']]
        summaries[dx]=dict(passed=True,year=100,accepted_steps=phases[-1]['accepted_steps'],wall_seconds=c['wall_seconds'],checkpoint_seconds=c['checkpoint_seconds'],hierarchy_seconds=c['hierarchy_seconds'],CPU_historical_seconds=c['historical_CPU_seconds'],
            H_max_m=max(r['H_max_m'] for r in checks),velocity_relative_L2=max(r['velocity_relative_L2'] for r in checks),BP_residual=max(r['BP_residual'] for r in checks),mass_relative=max(r['mass_relative'] for r in rows),peak_MiB=max(p['peak_memory_bytes'] for p in phases)/2**20,
            Newton=sum(p['Newton'] for p in phases),CG=sum(p['CG'] for p in phases),verified_checkpoints=c['verified_checkpoints'],annual_checks=len(checks),planned_restart_year=2,status_lock_survived=True)
    summary=dict(status='passed' if passed else 'not-selected',grids=summaries,source=ref(E/'completion.json'),working_selected=passed)
    write_json(REPORT/'summary.json',summary)
    if passed:
        pointer=R/'current-working-settings.json';previous_raw=pointer.read_bytes();previous=json.loads(previous_raw)
        write_json(E/'previous-working-pointer.json',previous);settings=read(PROJECT/previous['settings']['path'])
        settings.update(schema='polim-working-settings-gpu-v31',momentum_solver='CUDA exact cached CSR BP + Graph PCG',
            solver_module='polim_cpu.gpu_cached_v1',hierarchy_module='polim_cpu.gpu_long_hierarchy_v1',storage_module='polim_cpu.gpu_long_storage_v1',
            solver_implementation=hashes(),solver_precision='float64',momentum_tolerance=1e-6,inner_linear_target=.03,CG_graph_batch=4,
            preconditioner_update='Annually from current computed tangent; CPU AMG construction, GPU execution. No future reference geometry.',
            CPU_reference_solver='polim_cpu.bp_accelerated_v1',automatic_CPU_fallback=False,
            solver_validation=ref(REPORT/'summary.json'),long_term_scope='Working BP solver selected after50/100m1-100 tests andprocessrestart;fixedbed/material,zeroSMB validation. Climate/thermalcoupling remains separatework.',
            supersedes=previous['settings'],launch_authorized=False,user_decision='User agrees to selectthisversionaftercombinedtwo-gridlonghandoverpasses.')
        write_json(E/'working-settings.json',settings)
        new_pointer={**previous,'schema':'polim-working-settings-pointer-v2','momentum_solver':'gpu-cached-graph-v31','settings':ref(E/'working-settings.json')}
    rows=''
    for dx,s in summaries.items():
        if s['passed']:
            rows+=f"<tr><td>{dx}米</td><td>{s['wall_seconds']/60:.2f}分钟</td><td>{s['CPU_historical_seconds']/60:.2f}分钟</td><td>{s['checkpoint_seconds']/60:.2f}分钟</td><td>{s['hierarchy_seconds']:.2f}秒</td><td>{s['accepted_steps']}</td></tr>"
        else:rows+=f"<tr><td>{dx}米</td><td colspan='5'>未通过，查看完整证据</td></tr>"
    accuracy=''
    for dx,s in summaries.items():
        if s['passed']:accuracy+=f"<tr><td>{dx}米</td><td>{s['H_max_m']:.3e}米</td><td>{s['velocity_relative_L2']:.3e}</td><td>{s['BP_residual']:.3e}</td><td>{s['mass_relative']:.3e}</td></tr>"
    title='GPU版本通过长期交接，已设为后续工作求解器' if passed else '长期交接未全部通过，保留原工作求解器'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPU长期交接</title>
<style>body{{margin:0;background:#edf3f5;color:#243d4a;font:17px "Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:30px}}p,li{{line-height:1.85}}.lead{{background:#e1f2ee;border-left:5px solid #228f85;padding:20px}}table{{width:100%;background:white;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #dbe5e9;text-align:left}}.note{{font-size:14px;color:#526976}}a{{color:#147f90}}</style>
<main><h1>{title}</h1><p class="lead">50米与100米分别从已有第1年状态继续推进到第100年，每条轨迹新推进99年。合并完成精度对比、断点恢复和状态文件写入失败测试。{'默认保留50米网格、30天上限，动力求解器改为GPU缓存矩阵＋CUDA Graph。' if passed else '本次未变更默认设置。'}</p>
<table><tr><th>网格</th><th>GPU续算总耗时</th><th>原CPU历史耗时</th><th>GPU保存耗时</th><th>辅助矩阵准备</th><th>接受步数</th></tr>{rows}</table>
<p>耗时包含两段推进的载入、逐月保存、GPU求解、输运、年度辅助矩阵更新和状态发布；不包括外部解释器启动、初始CPU导出和完成后的独立CPU核对。历史CPU采用原始求解器与更密集的牛顿保存，不能把差异全归因于GPU。早期几何变化和年度更新均纳入此次长期实测。</p>
<h2>每个网格分别与对应CPU参考对比</h2><table><tr><th>网格</th><th>最大冰厚差</th><th>速度相对L2差</th><th>原CPU BP最大残差</th><th>质量相对误差</th></tr>{accuracy}</table>
<p>预先规定冰厚差≤0.001米、速度相对差≤0.001、BP残差≤10⁻⁶、质量相对误差≤10⁻¹⁰。各网格核对100个年度状态；两种网格之间原有的分辨率差异不作为求解器误差。</p>
<h2>恢复与长期执行</h2><p>先推进第1—2年，保存并正常退出GPU进程，再由新进程读取第2年状态继续到第100年。保存状态与GPU重载逐元素核对。首个时间步人为制造状态文件替换失败，模型写入备用快照并继续，后续时间步正常发布状态。</p>
<p>每个输运待求解状态、每个接受状态以及每5次牛顿迭代都原子保存；已完成结果保持不可覆盖。每年根据当前计算状态重新建立一次辅助矩阵层级，避免长期几何变化使初始预条件器失效。精确当前刚度仍在每次牛顿更新时重算。辅助构建耗时已计入总时间。</p>
<h2>后续使用范围</h2><p>{'本项目BP模型后续默认使用这版GPU求解器，CPU AMG保留作对照，未实现自动CPU回退。' if passed else '失败网格的已完成状态保留，详细原因见数据文件。'} 默认网格和时间步上限保持50米与30天，CFL仍可缩短步长。当前长期验证使用固定冰床、固定材料参数和零表面质量平衡；后续气候驱动与热力耦合在这套求解器上继续开发。独立OGGM批量主线与已接受数量保持不变。</p>
<p><a href="summary.json">完整摘要</a> · <a href="../gpu-cached-v1/index.html">一年优化试验</a> · <a href="../../index.html">项目入口</a></p></main></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8')
    evidence=dict(status=summary['status'],summary=ref(REPORT/'summary.json'),report=ref(REPORT/'index.html'),working_selected=passed)
    if passed:evidence['working_settings']=ref(E/'working-settings.json')
    write_json(E/'review.json',evidence)
    facts=f"- Combinedtwo-gridGPUhandover {'passed; selectasworkingBPsolver' if passed else 'notpassed; existingworkingpointerretained'}. Pergridstatistics {json.dumps(summaries,separators=(',',':'))}. AnnualoriginalCPU checks;year2newprocessresume;statuslocksurvived. Same50mdefault/30daycap/BP1e-6/float64/fixedbed/A/zeroSMB. Annualcurrent-tangentAMGrefresh CPU,CUDAcachedCSR/Graph execution. CPUAMG retainedasreference,noautomaticCPUfallback. No newclimate/thermalrun orOGGM191countchange. Evidence `{ref(E/'review.json')['path']}`."
    transition('working-selected' if passed else 'reviewed',E/'review.json',facts)
    if passed:
        assert pointer.read_bytes()==previous_raw
        temp=pointer.with_name(f'.working-{os.getpid()}.json');write_json(temp,new_pointer);os.replace(temp,pointer)
        write_json(E/'working-pointer-delivery.json',dict(pointer=ref(pointer),previous=ref(E/'previous-working-pointer.json')))
    p=R/'index.html';text=p.read_text(encoding='utf-8');tag='<h1>2387 冰川模型</h1>';text=text.replace(tag,tag+f'<a class="current" href="reports/gpu-long-handover-v1/index.html"><b>{title}</b><p>50米／100米长期推进、精度对比与进程恢复。</p></a>',1);p.write_text(text,encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

