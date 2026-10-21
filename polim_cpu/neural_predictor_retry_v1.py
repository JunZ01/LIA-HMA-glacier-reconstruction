"""One execution-only retry: remove threadpoolctl's override before torch child."""
import shutil
from . import neural_predictor_v1 as n

OLD_E=n.E;OLD_RUN=n.RUN
NEW_NAME='polim_v26r1_neural_predictor_v1'
NEW_E=OLD_E.with_name(NEW_NAME)
NEW_RUN=OLD_RUN.with_name('v26r1-neural-predictor-v1')
def configure():
    n.NAME=NEW_NAME;n.E=NEW_E;n.RUN=NEW_RUN

def register():
    n.validate()
    failure=dict(status='technical-failure-before-training',cause='Anaconda threadpoolctl.py line37 sets KMP_DUPLICATE_LIB_OK=True at import. Strict parent guard stopped before torch child; no training/BP/dynamics.',
                 preserved={f:n.ref(OLD_RUN/f) for f in ['basis.npz','training.npz','dataset-result.json']},
                 correction='After scientific imports, remove override from this controller process before original train() builds child env. Child still asserts override absent. No numerical/data/code change or repetition.',retry_count=1)
    n.write_json(OLD_E/'technical-failure.json',failure);n.transition('technical-failure',OLD_E/'technical-failure.json')
    configure();NEW_E.mkdir(exist_ok=False);plan=n.read(OLD_E/'plan.frozen.json')
    plan.update(schema=NEW_NAME,technical_retry=failure)
    plan['implementation'][n.Path(__file__).relative_to(n.PROJECT).as_posix()]=n.digest(__file__)
    for f in failure['preserved'].values():plan['input_hashes'][f['path']]=f['sha256'].lower()
    n.write_json(NEW_E/'plan.frozen.json',plan);n.transition('authorized',NEW_E/'plan.frozen.json')

def run():
    configure();n.validate();p=n.read(NEW_E/'plan.frozen.json')
    assert n.read(n.PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NEW_NAME]['launch_authorized']
    for f,h in {**p['input_hashes'],**p['implementation']}.items():assert n.digest(n.PROJECT/f)==h
    NEW_RUN.mkdir(exist_ok=False)
    n.write_json(NEW_E/f'fingerprint-{n.seal(p)}.json',dict(output=str(NEW_RUN)))
    n.write_json(NEW_RUN/'launch.json',dict(pid=n.os.getpid(),start=n.datetime.now(n.timezone.utc).isoformat(),command='python -m polim_cpu.neural_predictor_retry_v1 run',resources=p['resources'],checkpoint='every pending/accepted/Newton state',stop='900s finite test; no further automatic retry'))
    for f in ['basis.npz','training.npz','dataset-result.json']:shutil.copyfile(OLD_RUN/f,NEW_RUN/f)
    n.os.environ.pop('KMP_DUPLICATE_LIB_OK',None)
    start=n.time.perf_counter();n.train();result=n.case()
    out=dict(status='completed',continuation=result,dataset=n.read(NEW_RUN/'dataset-result.json'),training=n.read(NEW_RUN/'training-result.json'),training_process=n.read(NEW_RUN/'training-process.json'),total_elapsed_seconds=n.time.perf_counter()-start)
    out['amortized_test_seconds']=out['dataset']['seconds']+out['training_process']['seconds']+result['wall_seconds']
    n.write_json(NEW_RUN/'result.json',out)

if __name__=='__main__':
    if n.sys.argv[1]=='register':register()
    elif n.sys.argv[1]=='run':run()
