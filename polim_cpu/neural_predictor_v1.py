"""Registered 100m neural warm-start test with unchanged BP correction."""
from .solver_acceleration_v1 import *
import subprocess

NAME='polim_v26_neural_predictor_v1'
E=PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN=R/'runs/v26-neural-predictor-v1'
REPORT=R/'reports/neural-predictor-v1'
TRAIN=list(range(1,1028,5))
VALID=list(range(1028,1158,5))
BASELINE=R/'runs/v23-solver-acceleration-v1/grid-100m-year-099'


def hashes():
    files=['neural_predictor_v1.py','neural_train_v1.py','solver_acceleration_v1.py',
           'bp_accelerated_v1.py','v13r1/bp.py','v12/run.py','v10/model.py','v4/transport.py',
           'grid_triplet_v1.py','v9/storage.py','v3/runner.py']
    return {(R/f).relative_to(PROJECT).as_posix():digest(R/f) for f in files}


def load_frame(i):
    p=SOURCE/f'grid-100m/frame-{i:03d}'
    return old_storage.restore(p,read(p/'metadata.json')['contract'])


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
             production_replacement_authorized=False,next_legal_action='Run bounded100m neural warmstart pilot' if authorized else 'Use measured comparison; no automatic extension')
    fp = PROJECT/'LIA_PROJECT_FACTS.md'; fb = fp.read_bytes(); fh = digest(fp).upper()
    assert fh == s['scientific_contract']['sha256'].upper()
    if facts:
        number = max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
        addition = f'\n\n## {number}. System2387 Neural BP warm-start test\n\n{facts}\n'
        fh = hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper()
    route = read(PROJECT/s['authoritative_route']['path']); suffix = f'+polim-v26-{stage}'
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
        f.write(f'\n\n## {n["updated_at"]} - BP acceleration {stage}\n\nUser authorized neural optimization test. Train100m saved earlier states,holdout99-100;30daycap,unchangedBP correction,reference results read only. `{ref(evidence)["path"]}`.\n')
    write_json(E/f'{stage}-validation.json',validate())


def register():
    validate();E.mkdir(exist_ok=False)
    paths=set()
    for i in TRAIN+VALID+[1274,1287]:
        for j in [max(i-1,0),i]:
            for name in ['state.npz','metadata.json','seal.json']:
                paths.add(SOURCE/f'grid-100m/frame-{j:03d}'/name)
    paths.add(GRIDS/'grid-100m/mesh.npz')
    paths.update(p for p in BASELINE.rglob('*') if p.is_file())
    p=dict(schema=NAME,user_authorization='测试深度学习优化：100m预测、BP校正和完整耗时对照',
        train_frames=TRAIN,validation_frames=VALID,test_years=[99,100],
        method='Supervised global reduced-order MLP predicts velocity increment/year. Rank12 POD of H, previous velocity and velocity-rate targets; 37 inputs,64 tanh,64 tanh,12 outputs. All bases/scales fit train only. Adam1200 epochs lr0.002,seed2387; choose lowest chronological validation MSE every25epochs. No online tuning on test.',
        distinction='Inspired by hybrid neural/physical ice solvers, not a reproduction of IGM energy-trained CNN. Fixed single-glacier domain and material only.',
        correction='Compute original-equivalent BP residual for previous velocity and neural candidate; use smaller. If <=1e-6 accept, else exact retained AMG-CG correction. No changed physics/tolerance/transport.',
        physics='100m/fixedbed/fixedA/zeroSMB/noslip/3sigma nodes/30daycap/CFL0.35; year99-100,13expectedsteps.',
        tolerances=dict(momentum=1e-6,mass_relative=1e-10,H_difference_m=.001,velocity_relative_L2=.001),
        resources=dict(workers=1,BP_CPU_threads=8,BLAS_threads=1,torch_CPU_threads=1,GPU=False,estimated_minutes=[2,8],timeout_seconds=900),
        timing='Report dataset+POD+training subprocess including imports/export+inference+residual screening+BP+all checkpoints; compilation/preflight/postrun verification separately. Retained v23 historical timing, no baseline replay.',
        recovery='Immutable trained model and each pending/accepted/Newton state; same fingerprint continuation resumes from last accepted step with unchanged weights; completed steps never replayed.',
        preflight='One combined inherited operator/storage/restart/status-failure fixture and exported-network agreement; finite pilot not production.',
        input_hashes={p.relative_to(PROJECT).as_posix():digest(p) for p in sorted(paths)},implementation=hashes(),accepted_count_effect=0)
    write_json(E/'plan.frozen.json',p);transition('authorized',E/'plan.frozen.json')


def pod(x):
    mean=x.mean(axis=0);_,_,vh=np.linalg.svd(x-mean,full_matrices=False)
    return mean,vh[:12].T.copy()


def build_data():
    start=time.perf_counter();hs=[];dh=[];us=[];rates=[];dts=[]
    for i in TRAIN+VALID:
        a,u,_=load_frame(i-1);b,v,_=load_frame(i);dt=b['time']-a['time']
        hs.append(b['h']);dh.append(b['h']-a['h']);us.append(u[:,1:].ravel())
        rates.append((v[:,1:]-u[:,1:]).ravel()/dt);dts.append(dt)
    h=np.array(hs);d=np.array(dh);u=np.array(us);y=np.array(rates);n=len(TRAIN)
    hm,hb=pod(h[:n]);um,ub=pod(u[:n]);ym,yb=pod(y[:n])
    x=np.column_stack(((h-hm)@hb,d@hb,(u-um)@ub,np.array(dts)))
    target=(y-ym)@yb;xm=x[:n].mean(0);xs=np.maximum(x[:n].std(0),1e-12)
    tm=target[:n].mean(0);ts=np.maximum(target[:n].std(0),1e-12)
    with (RUN/'basis.npz').open('xb') as f:np.savez(f,hm=hm,hb=hb,um=um,ub=ub,ym=ym,yb=yb,xm=xm,xs=xs,tm=tm,ts=ts)
    with (RUN/'training.npz').open('xb') as f:np.savez(f,x=(x-xm)/xs,y=(target-tm)/ts,ntrain=n)
    out=dict(seconds=time.perf_counter()-start,train_samples=n,validation_samples=len(VALID),
             train_max_year=1+TRAIN[-1]/13,validation_max_year=1+VALID[-1]/13,
             validation_rate_projection_relative=float(np.linalg.norm(y[n:]-ym-target[n:]@yb.T)/np.linalg.norm(y[n:])))
    write_json(RUN/'dataset-result.json',out);return out


def train():
    env=os.environ.copy();assert not env.get('KMP_DUPLICATE_LIB_OK')
    site=PROJECT/'.runtime/torch-cu128-v1/site-packages';win=Path(os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=str(site)
    t=time.perf_counter()
    result=subprocess.run(['D:/Anaconda/envs/SS4/python.exe',str(R/'neural_train_v1.py'),str(RUN)],env=env,capture_output=True,text=True,timeout=300)
    (RUN/'training-stdout.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr)
    write_json(RUN/'training-process.json',dict(seconds=time.perf_counter()-t,returncode=result.returncode))
    write_json(RUN/'model-seal.json',dict(files={p:digest(RUN/p) for p in ['basis.npz','network.npz','training.npz','training-result.json']}))


class Predictor:
    def __init__(self):
        for p,h in read(RUN/'model-seal.json')['files'].items():assert digest(RUN/p)==h
        self.b=arrays(RUN/'basis.npz');self.w=arrays(RUN/'network.npz')
    def __call__(self,h,oldh,u,dt):
        b=self.b;x=np.concatenate(((h-b['hm'])@b['hb'],(h-oldh)@b['hb'],(u[:,1:].ravel()-b['um'])@b['ub'],[dt]))
        x=(x-b['xm'])/b['xs']
        for i in range(3):
            x=x@self.w[f'w{i}'].T+self.w[f'b{i}']
            if i<2:x=np.tanh(x)
        rate=b['ym']+(x*b['ts']+b['tm'])@b['yb'].T
        return u[:,1:]+dt*rate.reshape(-1,2,2)


def case(resume=False):
    root=RUN/'continuation';initial,u0,_=source(100,99);mesh=arrays(GRIDS/'grid-100m/mesh.npz')
    geo=topology(mesh);cache=fast.HorizontalCache(mesh);rf=reference({**mesh,'thickness':initial['h']},geo,u0)
    contract=seal(dict(plan=digest(E/'plan.frozen.json'),model=digest(RUN/'model-seal.json')))
    if not resume:root.mkdir();fixture(root,contract,mesh,initial,u0,cache)
    store=Store(root,contract,initial['A']);predict=Predictor();state=initial;u=u0;rows=[];exported=0.;step=0
    if resume:
        last=sorted(root.glob('frame-*'))[-1];state,u,m=store.restore(last.name)
        rows=m['info']['rows'];exported=m['info']['exported_m3'];step=len(rows)
    else:store.save('frame-000',state,u,dict(rows=[],exported_m3=0.))
    t0=time.perf_counter();V0=float(initial['h']@geo['dual_area'])
    while state['time']<100-1e-12:
        assert step<30 and time.perf_counter()-t0<900
        oldh=state['h'].copy();now=state['time'];limit=cfl(mesh,geo,state,u)
        end=min(100,now+30/365,now+limit/365 if limit else 100);nxt=step+1
        pending=f'pending-{nxt:04d}'
        if (root/pending).exists():state,u,m=store.restore(pending);ledger=m['info']['ledger']
        else:
            state,ledger=window(mesh,geo,state,u,end,rf);store.save(pending,state,u,dict(ledger=ledger))
        t=time.perf_counter();guess=predict(state['h'],oldh,u,state['time']-now);inference=time.perf_counter()-t
        t=time.perf_counter();a=fast.prepare({**mesh,'thickness':state['h']},state['A'],cache)
        g=guess.ravel().copy();prev=u[:,1:].ravel().copy();g[~a['wet']]=0.;prev[~a['wet']]=0.
        rr,_,_,f=fast.evaluate(a,prev,False);rp=float(np.linalg.norm(rr)/max(np.linalg.norm(f),1.))
        rr,_,_,f=fast.evaluate(a,g,False);rn=float(np.linalg.norm(rr)/max(np.linalg.norm(f),1.))
        selected=bool(np.isfinite(rn) and rn<rp);guess=(g if selected else prev).reshape(-1,2,2)
        residual=min(rp,rn) if selected else rp;screen=time.perf_counter()-t
        offset=0;saved=sorted(root.glob(f'newton-{nxt:04d}-*'))
        if saved:
            p=saved[-1];m=read(p/'metadata.json');assert digest(p/'metadata.json')==read(p/'seal.json')['metadata_sha256']
            assert digest(p/'state.npz')==m['state_sha256'] and m['contract']==contract
            guess=arrays(p/'state.npz')['u'];offset=m['iteration'];residual=float('inf')
        if residual<=1e-6:
            u=np.zeros_like(u);u[:,1:]=guess
            info=dict(iterations=0,relative_residual=residual,wall_seconds=0.,amg_builds=0,tangent_assemblies=0)
        else:u,info=fast.solve({**mesh,'thickness':state['h']},guess,state['A'],cache=cache,checkpoint=store.callback(nxt),iteration_offset=offset)
        exported+=sum(v['transport']['exported_volume_m3'] for v in ledger);volume=float(state['h']@geo['dual_area']);mass=abs(volume-V0+exported)/V0
        assert mass<=1e-10 and state['h'].min()>=0 and np.isfinite(u).all()
        step=nxt;rows.append(dict(step=step,year=state['time'],inference_seconds=inference,screen_seconds=screen,
            old_initial_residual=rp,neural_initial_residual=rn,selected=selected,momentum=info,mass_relative=mass))
        store.save(f'frame-{step:03d}',state,u,dict(rows=rows,exported_m3=exported))
        old_storage.status(root,dict(year=state['time'],steps=step,status='running'))
        print(json.dumps(rows[-1],default=str),flush=True)
    wall=time.perf_counter()-t0;base=read(BASELINE/'result.json');bs=Store(BASELINE,base['contract'],initial['A']);comparison=[]
    for i in range(1,step+1):
        st,v,_=store.restore(f'frame-{i:03d}');old,ov,_=bs.restore(f'frame-{i:03d}');assert abs(st['time']-old['time'])<1e-10
        a=original.prepare({**mesh,'thickness':st['h']});rr,_,_,f=original.evaluate(a,v[:,1:].ravel(),False,A=st['A'])
        residual=float(np.linalg.norm(rr)/max(np.linalg.norm(f),1.));w=geo['dual_area'][:,None];du=v[:,-1]-ov[:,-1]
        comparison.append(dict(step=i,H_max_m=float(abs(st['h']-old['h']).max()),velocity_relative_L2=float(np.sqrt(np.sum(w*du*du)/np.sum(w*ov[:,-1]**2))),original_BP_residual=residual))
    passed=all(c['H_max_m']<=.001 and c['velocity_relative_L2']<=.001 and c['original_BP_residual']<=1e-6 for c in comparison)
    out=dict(status='completed',contract=contract,wall_seconds=wall,steps=step,rows=rows,comparison=comparison,
        engineering_equivalence=passed,historical_baseline_seconds=base['wall_seconds'],
        baseline_Newton=sum(r['momentum']['iterations'] for r in base['rows']),new_Newton=sum(r['momentum']['iterations'] for r in rows),
        neural_selected=sum(r['selected'] for r in rows),BP_skipped=sum(r['momentum']['amg_builds']==0 for r in rows))
    write_json(root/'result.json',out);return out


def run():
    validate();p=read(E/'plan.frozen.json');s=read(PROJECT/'LIA_CURRENT_STATE.json')
    assert s['research_branches'][NAME]['launch_authorized'] and p['implementation']==hashes()
    for path,h in p['input_hashes'].items():assert digest(PROJECT/path)==h,path
    RUN.mkdir(exist_ok=False);write_json(E/f'fingerprint-{seal(p)}.json',dict(output=str(RUN)))
    write_json(RUN/'launch.json',dict(pid=os.getpid(),start=datetime.now(timezone.utc).isoformat(),command='python -m polim_cpu.neural_predictor_v1 run',resources=p['resources'],stop='900s bounded test; no automatic retry'))
    start=time.perf_counter();build_data();train();result=case()
    out=dict(status='completed',continuation=result,dataset=read(RUN/'dataset-result.json'),training=read(RUN/'training-result.json'),
        training_process=read(RUN/'training-process.json'),total_elapsed_seconds=time.perf_counter()-start)
    out['amortized_test_seconds']=out['dataset']['seconds']+out['training_process']['seconds']+result['wall_seconds']
    write_json(RUN/'result.json',out)


if __name__=='__main__':
    if sys.argv[1]=='register':register()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='resume':case(True)
