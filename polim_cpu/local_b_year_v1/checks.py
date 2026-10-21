"""Synthetic scheduling/ledger/subprocess checks only; no real geometry/model."""
import os,sys,json,time,tempfile,inspect
from pathlib import Path
import numpy as np
from .schedule import times,can_enter_annual,can_enter_C
from polim_cpu.c_control_repair_v1.process import launch,native_result
from polim_cpu.geometry_probe_v2.common import ROOT,write,read
from polim_cpu.geometry_window_v1.ledger import checkpoint,restore
from polim_cpu.cn_validation_v1.ledger import commit_one,KEYS,zero_ledger
from polim_cpu.cn_flux_time_v1.mass import accepted

def main(e):
    started=time.perf_counter();checks={};records=[]
    with tempfile.TemporaryDirectory(prefix='v61-no-model-') as temp:
        temp=Path(temp)
        for mode in ('outer','outer-fail'):
            root=temp/mode;(root/'native').mkdir(parents=True);(root/'supervisor').mkdir()
            r=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(root),mode],root/'supervisor',root,time.time()+20)
            records.append(dict(mode=mode,process=r,files=sorted(str(x.relative_to(root)) for x in root.rglob('*') if x.is_file())))
            checks[mode]=(r['exit_code']==0 if mode=='outer' else r['exit_code']!=0)
            checks[mode+'-layout']=all((root/x/'worker.log').is_file() and (root/x/'launch.json').is_file() for x in ('supervisor','native'))
        collision=temp/'collision';collision.mkdir();(collision/'worker.log').write_text('reserved')
        r=launch([sys.executable,'-c','raise RuntimeError("must not launch")'],collision,collision,time.time()+10)
        checks['exclusive_conflict_blocks_spawn']=r['exit_code']==-2 and 'pid' not in r
        area=np.array([2.,3.]);h=np.array([10.,10.]);u=np.zeros(8);ledger=zero_ledger(float(area@h));parts={k:np.zeros(2) for k in KEYS}
        inherit=zero_ledger(49.);inherit.update(volume=50.,source=1.)
        checkpoints=[]
        for year in times('N'):
            state=dict(H=h+.1,actual_nodes=area*.1,potential_nodes=area*.1,unmet=np.zeros(2),unmet_nodes=np.zeros(2),internal_volume=np.zeros(2),boundary_volume=np.zeros(2))
            nxt,tot=commit_one(ledger,state,parts,area);inh,_=commit_one(inherit,state,parts,area)
            cp=temp/f'accepted-{year+.25:.4f}.npz';meta=dict(year=year+.25,ledger=nxt,inherited_window_ledger=inh,forcing=None,forcing_used=None)
            checkpoint(cp,state['H'],u,tot,meta);saved,sm=restore(cp)
            assert np.array_equal(saved['H'],state['H']) and sm['ledger']==nxt and sm['inherited_window_ledger']==inh
            h=state['H'];ledger=nxt;parts=tot;inherit=inh;checkpoints.append(str(cp.name))
        checks['four_step_schedule']=times('N')==[1553.,1553.25,1553.5,1553.75] and len(set(checkpoints))==4 and sm['year']==1554.
        checks['independent_increment_and_inherited']=abs(ledger['source']-2.)<1e-12 and abs(inherit['source']-3.)<1e-12
        checks['D_schedule']=times('D_old')==times('D_new')==[1482.5]
        checks['conditional_science_only']=can_enter_annual(True,'completed','completed',True) and not can_enter_annual(True,'stopped','completed',True) and can_enter_C('completed') and not can_enter_C('stopped')
        source=inspect.getsource(accepted);checks['active_equal_not_hard_gate']='active_equal' not in source
        runner=(ROOT/'polim_cpu/local_b_year_v1/runner.py').read_text();checks['unchanged_coupling_import']='from polim_cpu.anchor_efficiency_v1.coupling import attempt' in runner and 'backend.anchor_enabled=False' in runner and 'inherited=meta[' in runner
        write(e/'preparation.json',dict(passed=all(checks.values()),checks=checks,mock_processes=records,seconds=time.perf_counter()-started,real_geometry_loaded=False,real_BP=0,real_residual=0,real_assembly=0,real_linear=0,time_steps=0,checkpoint_test='synthetic serialization only, not actual interrupted model recovery'))
    assert all(checks.values()),checks
if __name__=='__main__':main(Path(sys.argv[1]))
