"""One original N-quarter command, observed under an external Nsight wrapper."""
import json,sys,time,subprocess,traceback
from pathlib import Path
from . import observe
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'polim_cpu/runs/v59r5-path-capture-v1'
observe.init('SUP',RUN/'trace')
with observe.span('import-controller'):
    from . import control
def main():
    runtime=control.read(RUN/'wrapper-runtime.json');deadline=runtime['deadline'];root=RUN/'baseline';(root/'supervisor').mkdir(parents=True);(root/'native').mkdir()
    start=time.perf_counter();result={}
    try:
        with observe.span('authorization-before-CPU'):plan=control.gate('profile-science-launch',False)
        forcing=plan['profile']['forcing'];cfg=dict(case='baseline',segment='quarter',anchor_enabled=False,root=str(root),forcing=forcing,deadline=deadline)
        path=root/'supervisor/config.json';control.write(path,cfg)
        cmd=[sys.executable,'-m','polim_cpu.path_profile_result_v1.runner_entry',str(path)];control.write(root/'supervisor/launch.json',dict(command=cmd,deadline=deadline))
        with (root/'supervisor/worker.log').open('x',encoding='utf8') as log:
            with observe.span('wait-scientific-CPU-process'):
                cpu_started=time.perf_counter();cpu_started_epoch=time.time()
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
                try:proc.wait(max(.001,deadline-time.time()))
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True);proc.wait();raise
        result=dict(exit_code=proc.returncode,cpu_pid=proc.pid,seconds=time.perf_counter()-start,cpu_command_seconds=time.perf_counter()-cpu_started,cpu_started_epoch=cpu_started_epoch,cpu_ended_epoch=time.time(),result_path=str(root/'supervisor/result.json'))
        with observe.span('authorization-after-CPU'):control.gate('profile-science-collected',False)
    except BaseException as ex:result.update(status='failed',error=repr(ex),traceback=traceback.format_exc(),seconds=time.perf_counter()-start)
    finally:control.write(RUN/'science-supervisor-result.json',result)
with observe.span('scientific-supervisor-command'):main()
