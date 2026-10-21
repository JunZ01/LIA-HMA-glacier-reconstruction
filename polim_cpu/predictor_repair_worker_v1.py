"""One 1552 quarter-year attempt, only predictor replaced; no retry/halving."""
import os,sys,time,json,traceback,subprocess,queue,threading
from pathlib import Path
from types import SimpleNamespace
from .gpu_energy_worker_v1 import torch,np
from .large_step_phase2_worker_v1 import BP,attempt,Rejected,npz,status
from .large_step_transport_client_v1 import IsolatedTransportService
from . import packed_storage_v1 as st


class PredictorTransport(IsolatedTransportService):
    def __init__(self,z,root,case_root,contract):
        self.root=root;self.case_root=case_root;self.contract=contract;self.outer=0;root.mkdir()
        self.static_path=root/'transport-static.npz';npz(self.static_path,**{k:v for k,v in z.items() if k.startswith('geo_')})
        project=Path(__file__).resolve().parents[1];env=os.environ.copy();env.pop('POLIM_TORCH_SITE',None);env.pop('KMP_DUPLICATE_LIB_OK',None)
        home=Path('D:/ProgramData/Anaconda3');win=Path(os.environ.get('SystemRoot','C:/Windows'))
        env['PATH']=os.pathsep.join(map(str,[home,home/'Library/bin',home/'DLLs',win/'System32',win]));env['PYTHONPATH']=str(project)
        self.stderr=(root/'stderr.log').open('x')
        self.process=subprocess.Popen([str(home/'python.exe'),'-m','polim_cpu.diffusion_predictor_service_v2',str(self.static_path)],
            cwd=project,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,creationflags=0x08000000)
        self.messages=queue.Queue()
        def lines():
            for line in self.process.stdout:self.messages.put(line)
            self.messages.put('')
        threading.Thread(target=lines,daemon=True).start();assert json.loads(self.messages.get(timeout=60))['ready']
        self.reader=self.process.stdout;self.process.stdout=SimpleNamespace(readline=lambda:self.messages.get(timeout=120))

    def predict(self,h,u,bed,smb,dt):
        data=self._call('predict',thickness_old=h,velocity_old=u,bed=bed,smb_requested=smb,dt_years=np.array(dt))
        records=list(self.root.glob('*.diagnostics.json'));assert len(records)==1
        self.diagnostics=st.read(records[0]);st.write(self.case_root/'predictor-diagnostics.json',self.diagnostics)
        npz(self.case_root/'predictor.npz',**data)
        return data['thickness']

    def correct(self,h0,h,u,smb,dt):
        c=super().correct(h0,h,u,smb,dt);self.outer+=1
        npz(self.case_root/f'outer-ledger-{self.outer:02d}.npz',**c)
        st.save(self.case_root,f'outer-{self.outer:02d}',h,u,
            dict(year=1552.25,not_accepted=True,outer=self.outer,
                 mass_relative=float(c['mass_relative']),local_relative=float(c['thickness_equation_relative']),
                 fixed_point_relative=float(c['fixed_point_relative']),actual_smb_sha256=st.array_hash(c['actual_smb'])),self.contract)
        return c


def run(root):
    start=time.perf_counter();cfg=st.read(root/'config.json');project=Path(__file__).resolve().parents[1]
    assert st.read(project/'LIA_CURRENT_STATE.json')['research_branches'][cfg['branch_name']]['launch_authorized']
    contract=st.read(root/'contract.json')['contract'];source=Path(cfg['source_root']);prior=Path(cfg['prior_case'])
    z=st.arrays(source/'input.npz');saved,meta=st.restore(prior/'frame-000000',st.read(prior/'contract.json')['contract'])
    assert meta['year']==1552. and st.array_hash(saved['h'])==cfg['h_sha256'] and st.array_hash(saved['u'])==cfg['u_sha256']
    assert st.digest(prior/'forcing-1552.npz')==cfg['forcing_sha256'];smb=st.arrays(prior/'forcing-1552.npz')['smb']
    torch.cuda.set_per_process_memory_fraction(.10);bp=None;tr=None;record={}
    try:
        st.save(root,'initial',saved['h'],saved['u'],dict(year=1552.,forcing_sha256=cfg['forcing_sha256']),contract)
        tr=PredictorTransport(z,root/'transport',root,contract);bp=BP(z,root/'amg')
        initialization=time.perf_counter()-start
        out=attempt(saved['h'],saved['u'],z,smb,.25,tr,bp)
        st.save(root,'accepted',out['h'],out['u'],dict(year=1552.25,history=out['history'],predictor_only_replacement=True),contract)
        record=dict(status='accepted',accepted_without_halving=True,
            bp_calls=bp.total_calls,newton=sum(x['newton'] for x in out['history']),cg=bp.total_cg,
            final=out['history'][-1],history=out['history'],timing=attempt.last['timing'])
    except Exception as exc:
        record=dict(status='rejected' if isinstance(exc,Rejected) else 'technical-failure',error=repr(exc),traceback=traceback.format_exc(),
            accepted_without_halving=False,bp_calls=bp.total_calls if bp else 0,cg=bp.total_cg if bp else 0)
        if hasattr(attempt,'last'):
            record.update(attempt.last);record['newton']=sum(x.get('newton',0) for x in record['history'])
            record['final']=record['history'][-1] if record['history'] else None
    finally:
        for item in (bp,tr):
            if item:
                try:item.close()
                except Exception as exc:record['close_error']=repr(exc);record['status']='technical-failure'
    record.update(wall_seconds=time.perf_counter()-start,initialization_seconds=locals().get('initialization'),
        peak_memory_bytes=torch.cuda.max_memory_allocated(),halving_performed=False,attempts=1)
    st.write(root/'completion.json',record);status(root,{k:v for k,v in record.items() if k not in ('history','traceback')})
    print(json.dumps({k:v for k,v in record.items() if k not in ('history','traceback')}),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]))
