"""Bounded synthetic tests and immutable-matrix benchmark, no production launch."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TESTS = HERE.parent/'tests'
FIXTURE = ROOT/'validation_data/results/paleo_surface_validation_v0/lia_gpu_operator_offline_20260905T075022Z/fixed_equations.npz'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)


def child(args):
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch,_,_=import_torch_with_stubs_filtered()
    import numpy as np
    import unittest
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ['LIA_TEST_DEVICE']=args.device
    sys.path.insert(0,str(TESTS))
    suite=unittest.defaultTestLoader.discover(str(TESTS),pattern='test_lia_active_prefix_v1.py')
    tests=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(device=args.device, tests=tests.testsRun, failures=len(tests.failures),errors=len(tests.errors),
        status='tests-pass' if tests.wasSuccessful() else 'tests-fail',production_modified=False,
        real_glacier_trajectory_started=False)
    if not tests.wasSuccessful() or args.device=='cpu':
        write(args.output/'summary.json',report)
        return 0 if tests.wasSuccessful() else 1
    from torch_flowline_1d_active_prefix_v1 import cached_pcr
    from torch_flowline_1d_batched_semiimplicit_v1 import pcr_tridiagonal_solve
    from test_lia_active_prefix_v1 import synthetic
    torch.cuda.set_per_process_memory_fraction(.06)
    started=time.perf_counter()
    def timed(fn):
        torch.cuda.synchronize()
        start=time.perf_counter()
        result=fn()
        torch.cuda.synchronize()
        return result,(time.perf_counter()-start)*1000
    with np.load(FIXTURE,allow_pickle=False) as blob:
        n=max(len(blob['case%02d_rhs'%i]) for i in range(24))
        matrices=[np.zeros((24,n)) for _ in range(4)]
        matrices[1].fill(1)
        mask=np.zeros((24,n),bool)
        cut=max(int(blob['case%02d_cut'%i]) for i in range(24))
        for i in range(24):
            size=len(blob['case%02d_rhs'%i]); mask[i,:size]=True
            for a,name in zip(matrices,('lower','diagonal','upper','rhs')):
                a[i,:size]=blob['case%02d_%s'%(i,name)]
    rows=[]
    for lanes in (24,128,512):
        index=np.arange(lanes)%24
        coeff=[torch.as_tensor(x[index],device='cuda') for x in matrices]
        valid=torch.as_tensor(mask[index],device='cuda')
        def baseline():
            return pcr_tridiagonal_solve(*coeff,valid)
        def optimized():
            result=torch.zeros_like(coeff[3])
            result[:,:cut]=torch.clamp(cached_pcr(*(x[:,:cut] for x in coeff),valid[:,:cut]),min=0)
            return result
        a,_=timed(baseline); b,_=timed(optimized)
        error=float((torch.clamp(a,min=0)-b).abs().max().item())
        old,new=[],[]
        for _ in range(5):
            _,t=timed(baseline); old.append(t)
            _,t=timed(optimized); new.append(t)
        rows.append(dict(kind='fixed-equation-solve',lanes=lanes,full_width=n,solve_width=cut,
            thickness_max_difference_m=error,baseline_ms=old,optimized_ms=new,
            median_speed_ratio=float(np.median(old)/np.median(new))))
        del coeff,valid,a,b
        torch.cuda.empty_cache()
        a=synthetic(lanes=lanes,width=4255,occupied=500,device='cuda')
        b=synthetic(lanes=lanes,width=4255,occupied=500,device='cuda',optimized=True)
        initial=a.thick.clone()
        for state in (a,b):
            state._annual_mb_cache=torch.full_like(initial,-1e-8)
            state._annual_mb_cache_year=0
        # Each call starts from the SAME synthetic state, never accumulating
        # a real glacier trajectory. Timing includes prefix detection and budgets.
        def full_step(state):
            state.thick=initial.clone()
            state.time_seconds=0.
            state.cumulative_climatic_mass_balance_m3.zero_()
            state.cumulative_mass_budget_residual_m3.zero_()
            state.step(108000.)
        _,_=timed(lambda: full_step(a)); _,_=timed(lambda: full_step(b))
        error=float((a.thick-b.thick).abs().max().item())
        old,new=[],[]
        for _ in range(5):
            _,t=timed(lambda: full_step(a)); old.append(t)
            _,t=timed(lambda: full_step(b)); new.append(t)
        rows.append(dict(kind='synthetic-full-step-cached-annual-MB',lanes=lanes,
            full_width=4255,solve_width=b.last_solve_width,thickness_max_difference_m=error,
            baseline_ms=old,optimized_ms=new,median_speed_ratio=float(np.median(old)/np.median(new))))
        del a,b,initial
        torch.cuda.empty_cache()
        print(json.dumps(rows[-2:]),flush=True)
        if time.perf_counter()-started>90:
            raise RuntimeError('bounded benchmark time budget exhausted')
    report.update(status='fixture-and-synthetic-pass' if max(r['thickness_max_difference_m'] for r in rows)<=1e-6 else 'fixture-difference-fail',
        gpu=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda=torch.version.cuda,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),measurements=rows,
        production_may_be_concurrent=True,timing_kind='interleaved synchronized wall times; not exclusive GPU capacity',
        long_run_equivalence_qualified=False,production_replacement_authorized=False,
        benchmark_wall_seconds=time.perf_counter()-started)
    write(args.output/'summary.json',report)
    return 0 if report['status']=='fixture-and-synthetic-pass' else 1


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=('cpu','cuda'),required=True)
    parser.add_argument('--child',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.child:
        return child(args)
    subprocess.run([sys.executable,'-B',str(HERE/'validate_lia_execution_state_v1.py')],check=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=ROOT/'validation_data/results/paleo_surface_validation_v0'/('lia_gpu_active_prefix_'+args.device+'_'+stamp)
    out.mkdir()
    sources=[Path(__file__),HERE/'torch_flowline_1d_active_prefix_v1.py',TESTS/'test_lia_active_prefix_v1.py',FIXTURE,
        HERE/'torch_flowline_1d_batched_semiimplicit_v1.py',HERE/'torch_flowline_1d_matrix_v2.py']
    before={str(p.relative_to(ROOT)):sha(p) for p in sources}
    shared=sha(ROOT/'LIA_CURRENT_STATE.json')
    write(out/'scope.json',dict(user_request='Complete optimization',device=args.device,
        purpose='synthetic implementation tests and fixed-equation timing only',real_glacier_runs=0,
        production_control_allowed=False,shared_state_write_allowed=False,source_hashes=before,
        cuda_memory_fraction=.06,child_timeout_seconds=180,scientific_parameters_changed=False))
    from launch_lia_191_gpu_spinup_v10_239 import child_environment,SS4_PYTHON
    command=[str(SS4_PYTHON),'-B',str(Path(__file__)),'--child','--device',args.device,'--output',str(out)]
    with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
        result=subprocess.run(command,env=child_environment(),cwd=ROOT,stdout=stdout,stderr=stderr,timeout=180)
    write(out/'terminal.json',dict(exit_code=result.returncode,source_hashes_unchanged=all(sha(ROOT/p)==h for p,h in before.items()),
        shared_state_unchanged=sha(ROOT/'LIA_CURRENT_STATE.json')==shared,
        stdout_sha256=sha(out/'stdout.log'),stderr_sha256=sha(out/'stderr.log')))
    print(str(out),flush=True)
    print((out/'stderr.log').read_text(),flush=True)
    if (out/'summary.json').exists():
        print((out/'summary.json').read_text(),flush=True)
    return result.returncode


if __name__=='__main__':
    raise SystemExit(main())
