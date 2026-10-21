"""Finite registered physics-trained graph-network + BP pilot,100m99-100."""
from .solver_acceleration_v1 import *
from . import physics_neural_v1 as neural
import gc,psutil

NAME='polim_v27_physics_neural_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v27-physics-neural-v1'
REPORT=R/'reports/physics-neural-v1'
BASELINE=R/'runs/v23-solver-acceleration-v1/grid-100m-year-099'
TRAIN_YEARS=[1,5,10,20,30,40,50,60,70,79]
VALID_YEARS=[85,90]


def hashes():
    files=['physics_neural_v1.py','physics_neural_trial_v1.py','solver_acceleration_v1.py',
           'bp_accelerated_v1.py','v13r1/bp.py','v12/run.py','v10/model.py','v4/transport.py',
           'grid_triplet_v1.py','v9/storage.py','v3/runner.py']
    return {(R/f).relative_to(PROJECT).as_posix():digest(R/f) for f in files}


def transition(stage, evidence, facts=''):
    validate(); sp = PROJECT/'LIA_CURRENT_STATE.json'; raw = sp.read_bytes(); s = json.loads(raw)
    assert s['current_focus']['state'] != 'decision-required'
    authorized = stage == 'authorized'
    if authorized: assert NAME not in s['research_branches']
    else: assert s['research_branches'][NAME]['launch_authorized']
    b = dict(status=stage,launch_authorized=authorized,real_momentum_authorized=authorized,
             real_glacier_evolution_authorized=authorized,full_stokes_authorized=False,
             allowed_stages=['combined-preflight','neural-training','one-100m-year99-100-continuation'] if authorized else [],
             plan=ref(E/'plan.frozen.json'),evidence=ref(evidence),accepted_count_effect=0,
             production_replacement_authorized=False,next_legal_action='Run bounded100m physics-trained spatial-network pilot' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 physics-trained spatial neural BP test\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v27-{stage}'
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
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized neural optimization test. Train100m BP energy on earlier states,online4updates without reference labels,holdout99-100;30daycap,unchangedBP correction,reference results read only. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())


def register():
    validate();E.mkdir(exist_ok=False);inputs={}
    paths={GRIDS/'grid-100m/mesh.npz'}
    for year in TRAIN_YEARS+VALID_YEARS+[99]:
        for i in [(year-1)*13,(year-1)*13+1]:
            for f in ['metadata.json','state.npz','seal.json']:paths.add(SOURCE/f'grid-100m/frame-{i:03d}'/f)
    paths.update(p for p in BASELINE.rglob('*') if p.is_file())
    for p in paths:inputs[p.relative_to(PROJECT).as_posix()]=digest(p)
    plan=dict(schema=NAME,user_authorization='测试小型空间网络+BP能量训练+100m连续一年对照',
        mechanism='New physics-trained graph network,not prior supervisedPOD. Two graph-convolution layers16channels,tanh,4free-velocity outputs/node. Row-normalized mesh adjacency including self; analytic transpose backprop. Hard basal zero and dry masks.',
        physical_input='Existing-velocity BP residual divided by positive exact tangent diagonal,computed locally without assembling/factoring fullK. Geometry,FE surface slopes,A,previousvelocity,H change and normalized diagonal correction,16features.',
        output='previousvelocity + diagonalcorrection + localcorrectionscale*networkoutput; network learns spatial correction. BP energy and exact r=dE/du train weights, same quadrature/A/REG, no surrogate physical operator.',
        training=dict(years=TRAIN_YEARS,states_per_year=1,targets='No new referencevelocity labels; previous state velocity is input only. First transported geometry per listed year.',
            seed=2387,offline_LBFGS_iterations=40,offline_max_evaluations=500,validation_years=VALID_YEARS,validation='Diagnostic only,no fitting or hyperparameter selection.'),
        holdout=dict(years=[99,100],online_LBFGS_iterations_per_step=4,max_evaluations_per_step=60,
            rule='Warm-start network weights acrosssteps. Online training uses current geometry and BP energy,not reference trajectories. Evaluate offline/prior-weights and updated-network candidates,select lowest trueBP residual onlyif finite and energy no larger than previousvelocity. Otherwise previousvelocity fallback. OriginalAMG corrects tosame1e-6.'),
        physics='Same100m mesh/fixedbed/A/zeroSMB/noslip/fixedfootprint,30daycap/CFL0.35,originalBP tolerance1e-6.',
        tolerances=dict(momentum=1e-6,mass_relative=1e-10,H_difference_m=.001,velocity_relative_L2=.001),
        preflight='One combined originaloperator/storage/restart/status-failure fixture; exact diagonalvsfullK; neural energy directional finite-difference gradient(relative<=.002); exact networkreload. No broad hyperparameter search.',
        recovery='Immutable neural weights each accepted optimizer iteration plus seals; finaloffline checkpoint; everypending/Newton/accepted dynamics state. Online weights andoptimizer diagnostics sealed perstep. Explicit samecontract recovery maycontinue fromaccepted state; completed steps neverreplayed.',
        resources=dict(workers=1,CPU_threads=8,BLAS_threads=1,GPU=False,memory_limit_GiB=3,training_seconds=600,case_seconds=900,estimated_minutes=[1,5]),
        success='Accuracy gates AND <=80percent historicalAMG continuation wall AND fewer Newtonupdates. Historical timing alone not causalproof. Report first-use cost,recurringcost,trainingamortization; no automatic century/default promotion.',
        input_hashes=inputs,implementation=hashes(),accepted_count_effect=0)
    write_json(E/'plan.frozen.json',plan);transition('authorized',E/'plan.frozen.json')


def pair(year):
    old,u,_=source(100,year);p=SOURCE/f'grid-100m/frame-{(year-1)*13+1:03d}'
    st,_,_=old_storage.restore(p,read(p/'metadata.json')['contract'])
    return old,u,st


def weight_checkpoint(root,net,tag,theta,info):
    folder=root/tag;folder.mkdir();previous=net.theta;net.theta=theta
    neural.save_network(folder/'network.npz',net);net.theta=previous
    write_json(folder/'metadata.json',dict(info=info,network_sha256=digest(folder/'network.npz')))
    write_json(folder/'seal.json',dict(metadata_sha256=digest(folder/'metadata.json')))


def training(mesh):
    root=RUN/'training';root.mkdir();clock=time.perf_counter();cache=fast.HorizontalCache(mesh)
    contexts=[];preparation=0.
    for year in TRAIN_YEARS:
        a,u,b=pair(year);c=neural.context(mesh,b,a['h'],u,b['time']-a['time'],cache);contexts.append(c);preparation+=c['prepare_seconds']
        assert psutil.Process().memory_info().rss<3*2**30
    raw=np.vstack([c['x'] for c in contexts]);mean=raw.mean(0);std=np.maximum(raw.std(0),1e-3);del raw
    net=neural.Network(neural.graph(mesh),mean,std)
    a,u,_=source(100,99);fixture(root,seal(dict(plan=digest(E/'plan.frozen.json'),stage='preflight')),mesh,a,u,cache)
    c=contexts[0];_,_,K,_=fast.evaluate(c['a'],c['old'],True)
    diagonal_error=float(np.max(abs(K.diagonal()-c['diagonal']))/max(np.max(abs(K.diagonal())),1.));assert diagonal_error<=1e-12
    value,g=net.objective(c,net.theta);direction=g/max(np.linalg.norm(g),1e-30);epsilon=.001
    plus,_=net.objective(c,net.theta+epsilon*direction);minus,_=net.objective(c,net.theta-epsilon*direction)
    numerical=(plus-minus)/(2*epsilon);analytical=float(g@direction);gradient_error=abs(numerical-analytical)/max(abs(numerical),abs(analytical),1e-12)
    assert gradient_error<=.002,(gradient_error,numerical,analytical)
    neural.save_network(root/'initial-network.npz',net);restored=neural.load_network(root/'initial-network.npz',net.p)
    assert np.array_equal(restored.forward(c,restored.theta)[0],net.forward(c,net.theta)[0])
    write_json(root/'gradient-preflight.json',dict(status='pass',diagonal_relative_error=diagonal_error,energy_gradient_relative_error=gradient_error,hard_basal_zero=True,exact_reload=True))
    preflight_and_preparation=time.perf_counter()-clock
    def checkpoint(it,theta,info):
        if time.perf_counter()-clock>600:raise RuntimeError('Training runtime budget exceeded')
        weight_checkpoint(root,net,f'iteration-{it:03d}',theta,info)
        if it%10==0:print(json.dumps(dict(stage='energy-training',iteration=it,seconds=time.perf_counter()-clock)),flush=True)
    trained=net.optimize(contexts,40,500,checkpoint);neural.save_network(root/'network.npz',net)
    write_json(root/'model-seal.json',dict(network_sha256=digest(root/'network.npz'),plan_sha256=digest(E/'plan.frozen.json')))
    del contexts;gc.collect();validation=[]
    for year in VALID_YEARS:
        old,u,st=pair(year);c=neural.context(mesh,st,old['h'],u,st['time']-old['time'],cache)
        pred,_=net.forward(c,net.theta);rr,ee,_,_=fast.evaluate(c['a'],pred,False)
        validation.append(dict(year=year,old_residual=c['old_residual'],neural_residual=float(np.linalg.norm(rr)/c['force_norm']),normalized_energy_change=(ee-c['energy'])/c['loss_scale']))
    result=dict(status='completed',parameters=len(net.theta),preflight_and_preparation_seconds=preflight_and_preparation,
        feature_preparation_seconds_including_first_JIT=preparation,optimizer=trained,validation=validation,total_seconds=time.perf_counter()-clock,
        model_sha256=digest(root/'network.npz'),physics_loss='Original discrete BP energy; no targetvelocity fitting')
    write_json(root/'result.json',result);return result


def candidate(net,c):
    vel,_=net.forward(c,net.theta);r,e,_,_=fast.evaluate(c['a'],vel,False)
    return vel,float(np.linalg.norm(r)/c['force_norm']),float(e)


def case(mesh):
    root=RUN/'continuation';root.mkdir();initial,u,_=source(100,99);state=initial
    p=neural.graph(mesh);networkpath=RUN/'training/network.npz'
    assert digest(networkpath)==read(RUN/'training/model-seal.json')['network_sha256']
    net=neural.load_network(networkpath,p);cache=fast.HorizontalCache(mesh);geo=topology(mesh);rf=reference({**mesh,'thickness':initial['h']},geo,u)
    contract=seal(dict(plan=digest(E/'plan.frozen.json'),model=digest(networkpath)))
    store=Store(root,contract,initial['A']);store.save('frame-000',state,u,dict(rows=[],exported_m3=0.))
    neural.save_network(root/'network-000.npz',net);rows=[];exported=0.;step=0;V0=float(initial['h']@geo['dual_area'])
    begin=time.perf_counter()
    while state['time']<100-1e-12:
        assert step<30 and time.perf_counter()-begin<900
        nxt=step+1;oldh=state['h'].copy();now=state['time'];limit=cfl(mesh,geo,state,u)
        end=min(100,now+30/365,now+limit/365 if limit else 100)
        t=time.perf_counter();state,ledger=window(mesh,geo,state,u,end,rf);transport=time.perf_counter()-t
        store.save(f'pending-{nxt:04d}',state,u,dict(ledger=ledger,previous_network_sha256=digest(root/f'network-{step:03d}.npz')))
        t=time.perf_counter();c=neural.context(mesh,state,oldh,u,state['time']-now,cache)
        prior_theta=net.theta.copy();v0,r0,e0=candidate(net,c);initial_inference=time.perf_counter()-t
        trainroot=root/f'online-{nxt:03d}';trainroot.mkdir()
        online=net.optimize([c],4,60,lambda it,theta,info:weight_checkpoint(trainroot,net,f'iteration-{it:03d}',theta,info))
        t=time.perf_counter();v1,r1,e1=candidate(net,c);guess=c['old'];chosen='previous-velocity';residual=c['old_residual']
        for label,vel,r,e,theta in [('prior-network',v0,r0,e0,prior_theta),('updated-network',v1,r1,e1,net.theta.copy())]:
            if np.isfinite(r) and r<residual and e<=c['energy']:
                guess=vel;residual=r;chosen=label
        screening=time.perf_counter()-t
        # Persist the optimizer's weights,even when its velocity needs BP correction.
        neural.save_network(root/f'network-{nxt:03d}.npz',net)
        write_json(trainroot/'result.json',dict(optimizer=online,chosen=chosen,old_residual=c['old_residual'],prior_residual=r0,updated_residual=r1,network_sha256=digest(root/f'network-{nxt:03d}.npz')))
        if residual<=1e-6:
            u=np.zeros_like(u);u[:,1:]=guess.reshape(-1,2,2)
            info=dict(iterations=0,relative_residual=residual,wall_seconds=0.,amg_builds=0,tangent_assemblies=0)
        else:u,info=fast.solve({**mesh,'thickness':state['h']},guess.reshape(-1,2,2),state['A'],cache=cache,checkpoint=store.callback(nxt))
        exported+=sum(v['transport']['exported_volume_m3'] for v in ledger);volume=float(state['h']@geo['dual_area']);mass=abs(volume-V0+exported)/V0
        assert mass<=1e-10 and state['h'].min()>=0 and np.isfinite(u).all()
        step=nxt;rows.append(dict(step=step,year=state['time'],mass_relative=mass,chosen=chosen,old_initial_residual=c['old_residual'],
            prior_network_residual=r0,updated_network_residual=r1,selected_residual=residual,transport_seconds=transport,
            feature_and_initial_screen_seconds=initial_inference,online=online,final_screen_seconds=screening,momentum=info))
        store.save(f'frame-{step:03d}',state,u,dict(rows=rows,exported_m3=exported,network_sha256=digest(root/f'network-{step:03d}.npz')))
        old_storage.status(root,dict(status='running',year=state['time'],steps=step,chosen=chosen,Newtons=info['iterations']))
        print(json.dumps(dict(stage='continuation',step=step,year=state['time'],chosen=chosen,old_residual=c['old_residual'],neural_residual=r1,Newtons=info['iterations'],seconds=time.perf_counter()-begin)),flush=True)
        del c
    wall=time.perf_counter()-begin;base=read(BASELINE/'result.json');bs=Store(BASELINE,base['contract'],initial['A']);compare=[]
    verify_start=time.perf_counter()
    for i in range(14):
        st,v,m=store.restore(f'frame-{i:03d}');old,ov,_=bs.restore(f'frame-{i:03d}');assert abs(st['time']-old['time'])<1e-10
        if i:
            assert digest(root/f'network-{i:03d}.npz')==m['info']['network_sha256']
            for folder in (root/f'online-{i:03d}').glob('iteration-*'):
                meta=read(folder/'metadata.json');assert digest(folder/'metadata.json')==read(folder/'seal.json')['metadata_sha256']
                assert digest(folder/'network.npz')==meta['network_sha256']
        a=original.prepare({**mesh,'thickness':st['h']});rr,_,_,f=original.evaluate(a,v[:,1:].ravel(),False,A=st['A'])
        residual=float(np.linalg.norm(rr)/max(np.linalg.norm(f),1.));w=geo['dual_area'][:,None];du=v[:,-1]-ov[:,-1]
        compare.append(dict(step=i,H_max_m=float(abs(st['h']-old['h']).max()),velocity_relative_L2=float(np.sqrt(np.sum(w*du*du)/np.sum(w*ov[:,-1]**2))),original_BP_residual=residual))
    passed=all(c['H_max_m']<=.001 and c['velocity_relative_L2']<=.001 and c['original_BP_residual']<=1e-6 for c in compare)
    result=dict(status='completed',contract=contract,wall_seconds=wall,verification_seconds=time.perf_counter()-verify_start,steps=step,
        rows=rows,comparison=compare,engineering_equivalence=passed,historical_baseline_seconds=base['wall_seconds'],
        baseline_Newton=sum(r['momentum']['iterations'] for r in base['rows']),new_Newton=sum(r['momentum']['iterations'] for r in rows),
        neural_selected=sum(r['chosen']!='previous-velocity' for r in rows),BP_skipped=sum(r['momentum']['amg_builds']==0 for r in rows),
        baseline_BP_seconds=sum(r['momentum']['wall_seconds'] for r in base['rows']),new_BP_seconds=sum(r['momentum']['wall_seconds'] for r in rows),
        verified_states=14)
    result['acceleration_gate']=bool(passed and wall<=.8*base['wall_seconds'] and result['new_Newton']<result['baseline_Newton'])
    write_json(root/'result.json',result);return result


def run():
    validate();p=read(E/'plan.frozen.json');assert read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    assert p['implementation']==hashes()
    for path,h in p['input_hashes'].items():assert digest(PROJECT/path)==h,path
    RUN.mkdir(exist_ok=False);write_json(E/f'fingerprint-{seal(p)}.json',dict(output=str(RUN)))
    write_json(RUN/'launch.json',dict(pid=os.getpid(),command='python -m polim_cpu.physics_neural_trial_v1 run',start=datetime.now(timezone.utc).isoformat(),resources=p['resources'],checkpoint=p['recovery'],stop='600s training,900s continuation; finite13stepwindow'))
    begin=time.perf_counter();mesh=arrays(GRIDS/'grid-100m/mesh.npz')
    try:
        t=training(mesh);gc.collect();c=case(mesh)
        result=dict(status='completed',training=t,continuation=c,total_seconds=time.perf_counter()-begin,
            first_use_seconds=t['total_seconds']+c['wall_seconds'],timing_note='Includes neural preparation/training/onlineupdates/screens/BP/checkpoints. First-use includes combinedfixture; postrun comparison separate. Historicalv23 baseline not contemporaneous.')
        write_json(RUN/'result.json',result)
    except Exception as exc:
        write_json(RUN/'failure.json',dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-begin));raise


if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='run':run()
