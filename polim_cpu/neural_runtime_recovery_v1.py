"""Disclosed second environment repair under user's continue-on-error instruction."""
import shutil
from . import neural_predictor_v1 as n
OLD_E=n.E.with_name('polim_v26r1_neural_predictor_v1')
OLD_RUN=n.RUN.with_name('v26r1-neural-predictor-v1')
NAME='polim_v26r2_neural_predictor_v1'
E=OLD_E.with_name(NAME);RUN=OLD_RUN.with_name('v26r2-neural-predictor-v1')
def configure():n.NAME=NAME;n.E=E;n.RUN=RUN
def register():
    n.NAME='polim_v26r1_neural_predictor_v1';n.E=OLD_E;n.RUN=OLD_RUN;n.validate()
    f=dict(status='technical-failure-before-training',cause='SS4 Triton2 incompatible with torch2.11 optional compiler import; no optimizer step,network output or BP evolution.',
      user_override='User explicitly requested 即使有错误，也可以继续. This second environment repair exceeds the generic one-technical-retry default and is disclosed before execution.',
      change='CPU eager training blocks optional triton import via process-local module entry; original neural training algorithm and all physical code unchanged. No package installation or global environment modification.')
    n.write_json(OLD_E/'technical-failure.json',f);n.transition('technical-failure',OLD_E/'technical-failure.json')
    configure();E.mkdir(exist_ok=False);p=n.read(OLD_E/'plan.frozen.json');p.update(schema=NAME,second_execution_repair=f)
    for fn in ['neural_runtime_recovery_v1.py','neural_cpu_entry_v1.py']:
        q=n.R/fn;p['implementation'][q.relative_to(n.PROJECT).as_posix()]=n.digest(q)
    n.write_json(E/'plan.frozen.json',p);n.transition('authorized',E/'plan.frozen.json')
def run():
    configure();n.validate();p=n.read(E/'plan.frozen.json')
    assert n.read(n.PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for f,h in {**p['input_hashes'],**p['implementation']}.items():assert n.digest(n.PROJECT/f)==h
    RUN.mkdir(exist_ok=False);n.write_json(E/f'fingerprint-{n.seal(p)}.json',dict(output=str(RUN)))
    n.write_json(RUN/'launch.json',dict(pid=n.os.getpid(),start=n.datetime.now(n.timezone.utc).isoformat(),command='python -m polim_cpu.neural_runtime_recovery_v1 run',resources=p['resources'],checkpoint='every pending/accepted/Newton state',stop='900s finite test'))
    for f in ['basis.npz','training.npz','dataset-result.json']:shutil.copyfile(OLD_RUN/f,RUN/f)
    env=n.os.environ.copy();env.pop('KMP_DUPLICATE_LIB_OK',None)
    site=n.PROJECT/'.runtime/torch-cu128-v1/site-packages';win=n.Path(n.os.environ.get('SystemRoot','C:/Windows'))
    env['PATH']=n.os.pathsep.join(map(str,[site/'torch/lib',win/'System32',win]));env['PYTHONPATH']=str(site)
    start=n.time.perf_counter()
    result=n.subprocess.run(['D:/Anaconda/envs/SS4/python.exe',str(n.R/'neural_cpu_entry_v1.py'),str(n.R/'neural_train_v1.py'),str(RUN)],env=env,capture_output=True,text=True,timeout=300)
    (RUN/'training-stdout.txt').write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError(result.stderr)
    n.write_json(RUN/'training-process.json',dict(seconds=n.time.perf_counter()-start,returncode=result.returncode,optional_triton_disabled=True))
    n.write_json(RUN/'model-seal.json',dict(files={p:n.digest(RUN/p) for p in ['basis.npz','network.npz','training.npz','training-result.json']}))
    result=n.case()
    out=dict(status='completed',continuation=result,dataset=n.read(RUN/'dataset-result.json'),training=n.read(RUN/'training-result.json'),training_process=n.read(RUN/'training-process.json'),total_elapsed_seconds=n.time.perf_counter()-start)
    out['amortized_test_seconds']=out['dataset']['seconds']+out['training_process']['seconds']+result['wall_seconds']
    n.write_json(RUN/'result.json',out)
if __name__=='__main__':
    if n.sys.argv[1]=='register':register()
    elif n.sys.argv[1]=='run':run()
