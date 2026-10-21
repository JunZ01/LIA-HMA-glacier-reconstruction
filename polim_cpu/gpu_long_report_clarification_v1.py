from .gpu_long_recovery_trial_v1 import *
def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['status'] in ['reviewed']
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
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v31r1-{stage}'
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
    old=REPORT/'index.html';text=old.read_text(encoding='utf-8')
    text=text.replace('各网格核对100个年度状态；两种网格之间原有的分辨率差异不作为求解器误差。','100米已核对100个年度状态；50米尚未完成整段年度核对。两种网格之间原有的分辨率差异不作为求解器误差。')
    text=text.replace('先推进第1—2年，保存并正常退出GPU进程，再由新进程读取第2年状态继续到第100年。','两组均先推进第1—2年，保存并正常退出GPU进程，再由新进程读取第2年状态接续。100米已到第100年；50米的完成进度为第39.08年。')
    dest=REPORT/'index-v2.html';dest.write_text(text,encoding='utf-8')
    evidence=dict(status='reviewed',report=ref(dest),supersedes_report=ref(old),summary=ref(REPORT/'summary.json'),prepared_fix=ref(R/'gpu_reusable_geometry_v1.py'),proposal=ref(E/'next-continuation.proposed.json'),default_unchanged=True,launch_authorized=False)
    write_json(E/'report-clarification.json',evidence)
    transition('report-clarified',E/'report-clarification.json','- Report wording clarifies100m completed100annualchecks and50m accepted495steps to39.082years;50m annualhandover incomplete. Defaultnotchanged. Stablegeometrybufferfix prepared/unrun;additionalcontinuation proposalnotauthorized afterone technicalretry exhausted. Evidence `'+ref(E/'report-clarification.json')['path']+'`.')
    p=R/'index.html';s=p.read_text(encoding='utf-8').replace('reports/gpu-long-handover-v1/index.html','reports/gpu-long-handover-v1/index-v2.html');p.write_text(s,encoding='utf-8')
if __name__=='__main__':main()
