"""GPU long handover with annual AMG setup and recoverable monthly states."""
from .gpu_energy_worker_v1 import torch,tensor,Transport,Path,np,os,sys,time,json,traceback
from .gpu_allocator_safe_v1 import BP,PCG,Multigrid,solve
from . import gpu_long_storage_v1 as st
import subprocess,gc

class Hierarchy:
    def __init__(self,root,z):
        self.root=root;self.z=z;self.seconds=0.;self.process=None;self.calls=0
    def build(self,bp,year):
        start=time.perf_counter();folder=self.root/f'hierarchy-{year:03d}'
        if not folder.exists():
            folder.mkdir()
            with (folder/'matrix.npz').open('xb') as f:np.savez(f,data=bp.values.cpu().numpy(),indices=self.z['csr_indices'],indptr=self.z['csr_indptr'],shape=np.array([bp.ndof,bp.ndof]),wet=bp.wet.cpu().numpy())
            if self.process is None:
                project=Path(__file__).resolve().parents[1];environment=os.environ.copy();environment.pop('POLIM_TORCH_SITE',None);environment.pop('KMP_DUPLICATE_LIB_OK',None)
                home=Path('d:/ProgramData/Anaconda3');win=Path(os.environ.get('SystemRoot','C:/Windows'))
                environment['PATH']=os.pathsep.join(map(str,[home,home/'Library/bin',home/'DLLs',win/'System32',win]));environment['PYTHONPATH']=str(project)
                self.log=(self.root/'hierarchy-stderr.log').open('a',encoding='utf-8')
                self.process=subprocess.Popen([str(home/'python.exe'),'-m','polim_cpu.gpu_long_hierarchy_v1'],cwd=project,env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True)
            self.process.stdin.write(json.dumps(dict(folder=str(folder)))+'\n');self.process.stdin.flush()
            answer=json.loads(self.process.stdout.readline());assert answer['ok'],answer
        meta=st.read(folder/'result.json');assert st.digest(folder/'hierarchy.npz')==meta['sha256'];assert st.digest(folder/'matrix.npz')==meta['matrix_sha256']
        mg=Multigrid(st.arrays(folder/'hierarchy.npz'));self.seconds+=time.perf_counter()-start;self.calls+=1;return mg
    def close(self):
        if self.process:self.process.stdin.close();self.process.wait(timeout=30);self.log.close()

def run(root,phase):
    start=time.perf_counter();z=st.arrays(root/'input.npz');contract=st.read(root/'contract.json')['contract'];torch.cuda.set_per_process_memory_fraction(.10)
    bp=BP(z);tr=Transport(z);hierarchy=Hierarchy(root,z);io=0.;rows=[];newton=0;cg=0;captures=0.;lastyear=None
    st.write(root/f'launch-{phase}.json',dict(pid=os.getpid(),phase=phase,command=sys.argv,start=time.time(),grid_m=int(z['dx']),checkpoint='everyaccepted/pending and5Newton',estimate_minutes=[3,20],max_seconds=3600,device=torch.cuda.get_device_name(0)))
    def save(name,h,u,meta):
        nonlocal io
        t=time.perf_counter();torch.cuda.synchronize();st.save(root,name,h.cpu().numpy(),u.cpu().numpy(),meta,contract);io+=time.perf_counter()-t
    try:
        source=st.read(root/'recovery-source.json');oldroot=Path(source['root']);oldcontract=source['contract']
        # Combined technical fixture: both sizes plus the saved failed pending geometry.
        fixture_results={}
        for name in ['100','50']:
            zz=st.arrays(root/f'fixture-{name}.npz');bb=BP(zz);bb.prepare(tensor(zz['fixture_h']));ptr=bb.B.data_ptr();bb.prepare(tensor(zz['fixture_h']));assert bb.B.data_ptr()==ptr;rr,ee=bb.evaluate(tensor(zz['fixture_u']),True)
            err=float((bb.hvp(tensor(zz['direction']))-tensor(zz['fixture_hvp'])).norm()/tensor(zz['fixture_hvp']).norm().clamp_min(1.))
            forceerr=float((bb.f-tensor(zz['fixture_f'])).norm()/bb.force_norm);reserr=float((rr-tensor(zz['fixture_r'])).norm()/bb.force_norm)
            assert max(err,forceerr,reserr)<=1e-9
            fixture_results[name]=dict(hvp=err,force=forceerr,residual=reserr,geometry_buffer_address_reused=True)
            del bb,rr,ee,zz;gc.collect();torch.cuda.empty_cache()
        st.write(root/'technical-fixture.json',dict(status='pass',checks=fixture_results))
        rz,rm=st.restore(oldroot/'frame-1110',oldcontract)
        h=tensor(rz['h']);u=tensor(rz['u']);now=rm['year'];step=rm['step'];exported=rm['exported'];V0=rm['V0'];target=100.;resume_start=now
        save('frame-1110',h,u,dict(year=now,step=step,exported=exported,V0=V0,inherited=True))
        restored,_=st.restore(root/'frame-1110',contract);assert np.array_equal(restored['h'],rz['h']) and np.array_equal(restored['u'],rz['u'])
        st.write(root/'restart.json',dict(status='pass',source=str(oldroot/'frame-1110'),year=now,step=step,retained_prefix=True,transport_pending_reused='pending-1111'))
        st.status(root,dict(stage='running',phase=phase,year=now,accepted_steps=step,rejected_steps=0))
        while now<target-1e-12:
            assert step<3000 and time.perf_counter()-start<3600
            previous=now;year=int(np.floor(previous+1e-10));t=time.perf_counter()
            if step==1110:
                pending,meta=st.restore(oldroot/'pending-1111',oldcontract);h=tensor(pending['h']);u=tensor(pending['u']);now=meta['year'];exported=meta['exported']
                st.write(root/'pending-adoption.json',dict(source=str(oldroot/'pending-1111'),source_state_sha256=st.digest(oldroot/'pending-1111/state.npz'),transport_recomputed=False))
                meta=dict(year=now,step=step+1,exported=exported,V0=V0)
            else:
                end=min(target,year+1.,previous+30/365);q=tr.flux(h,u);vol=h*tr.dual;donor=torch.where(q>=0,tr.links[:,0],tr.links[:,1])
                out=torch.zeros_like(h);out.index_add_(0,donor,q.abs());wet=(vol>0)&(out>0)
                if bool(wet.any()):end=min(end,previous+float((.35*vol[wet]/out[wet]).min()))
                if end-previous<1e-13:raise RuntimeError('CFL collapse')
                h,now,ex,subs=tr.window(h,u,previous,end);exported+=ex;meta=dict(year=now,step=step+1,exported=exported,V0=V0)
            save(f'pending-{step+1:04d}',h,u,meta)
            address=bp.B.data_ptr() if hasattr(bp,'B') else None
            bp.prepare(h)
            if address is not None:assert bp.B.data_ptr()==address
            if year!=lastyear:
                if hasattr(bp,'pcg'):
                    torch.cuda.synchronize();bp.pcg.graph.reset();del bp.pcg;del mg;gc.collect();torch.cuda.empty_cache()
                bp.evaluate(u,True);mg=hierarchy.build(bp,year);bp.pcg=PCG(bp,mg,True);lastyear=year
            cap0=bp.pcg.capture_seconds
            def callback(it,v,info):save(f'newton-{step+1:04d}-{it:03d}',h,v,{**meta,'iteration':it,'momentum':info})
            u,info=solve(bp,u,mg,callback);captures+=bp.pcg.capture_seconds-cap0
            mass=abs(float(h@tr.dual)-V0+exported)/V0
            assert mass<=1e-10 and bool(torch.isfinite(u).all()) and bool((h>=0).all())
            step+=1;newton+=info['iterations'];ncg=sum(r.get('cg_iterations',0) for r in info['history']);cg+=ncg
            row=dict(step=step,year=now,mass_relative=mass,Newton=info['iterations'],CG=ncg,BP_residual=info['relative_residual'],seconds=time.perf_counter()-t)
            save(f'frame-{step:04d}',h,u,{**meta,'row':row});rows.append(row)
            # Test an unavailable status target and continue to the next accepted step.
            ok=st.status(root,dict(stage='running',phase=phase,year=now,accepted_steps=step,rejected_steps=0,last_checkpoint=f'frame-{step:04d}',new_error=None),inject=phase=='pilot' and step==1)
            if phase=='pilot' and step==1:assert not ok
            if phase=='pilot' and step==2:
                assert ok and len(list(root.glob('status-fallback-*')))==1
                st.write(root/'status-survival.json',dict(status='pass',continued_to_step=step))
            if abs(now-round(now))<1e-9:
                annual=dict(year=int(round(now)),frame=step,volume_m3=float(h@tr.dual),max_speed=float(u.reshape(-1,2,2)[:,-1].norm(dim=1).max()),phase_seconds=time.perf_counter()-start)
                st.write(root/f'year-{int(round(now)):03d}.json',annual);print(json.dumps(dict(grid_m=int(z['dx']),**annual)),flush=True)
            if phase=='pilot':assert torch.cuda.max_memory_allocated()<.08*torch.cuda.get_device_properties(0).total_memory
        result=dict(status='completed',phase=phase,start_year=resume_start,end_year=now,accepted_steps=step,rows=rows,wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,hierarchy_seconds=hierarchy.seconds,graph_capture_seconds=captures,Newton=newton,CG=cg,peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:result=dict(status='failed',phase=phase,error=repr(exc),traceback=traceback.format_exc(),year=locals().get('now'),accepted_steps=locals().get('step'),wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,rows=rows)
    finally:hierarchy.close()
    st.write(root/f'result-{phase}.json',result);st.status(root,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2])
