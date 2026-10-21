"""Checkpointed historical forcing driver for the selected GPU BP solver."""
from .gpu_allocator_recovery_worker_v1 import Hierarchy
from .gpu_energy_worker_v1 import torch,tensor,Transport,Path,np,os,sys,time,json,traceback
from .gpu_allocator_safe_v1 import BP,PCG,solve
from .historical_climate_v1 import annual
from . import gpu_long_storage_v1 as st
from types import SimpleNamespace
import gc

def run(root, target, phase):
    start=time.perf_counter();cfg=st.read(root/'config.json');z=st.arrays(root/'input.npz')
    climate=st.arrays(root/'climate.npz');contract=st.read(root/'contract.json')['contract']
    torch.cuda.set_per_process_memory_fraction(.10)
    backend=SimpleNamespace(xp=torch,array=tensor)
    bp=BP(z);tr=Transport(z);hierarchy=Hierarchy(root,z);rows=[];io=0.;last_year=None;last_wet=None;mg=None
    def save(name,h,u,meta):
        nonlocal io
        t=time.perf_counter();st.save(root,name,h.cpu().numpy(),u.cpu().numpy(),meta,contract);io+=time.perf_counter()-t
    try:
        frames=sorted(p for p in root.glob('frame-*') if p.is_dir())
        if frames:
            state,meta=st.restore(frames[-1],contract);h=tensor(state['h']);u=tensor(state['u'])
            step=meta['step'];now=meta['year'];exported=meta['exported'];smb_volume=meta['smb_volume'];V0=meta['V0']
            st.write(root/f'restart-{phase}.json',dict(status='pass',source=frames[-1].name,step=step,year=now))
        else:
            h=tensor(z['h']);u=tensor(z['u']);step=0;now=float(cfg['start_year']);exported=0.;smb_volume=0.;V0=float(h@tr.dual)
            bp.prepare(h);r,_=bp.evaluate(u,True)
            op=float((bp.hvp(tensor(z['direction']))-tensor(z['fixture_hvp'])).norm()/tensor(z['fixture_hvp']).norm().clamp_min(1.))
            assert op<=1e-9
            mb_err=0.
            for j,year in enumerate(climate['years']):
                smb=annual(bp.bed+h,climate,cfg['parameters'],int(year),backend)
                mb_err=max(mb_err,float((smb-tensor(z['fixture_smb'][j])).abs().max()))
            assert mb_err<=1e-10
            # A manufactured ice-limited source checks the exact volume ledger.
            test_h=tensor([0.,1.,2.]);test_mb=tensor([2.,-5.,1.]);test_new=(test_h+test_mb*.5).clamp_min(0)
            assert float((test_new-tensor([1.,0.,2.5])).abs().max())==0
            st.write(root/'technical-fixture.json',dict(status='pass',HVP_relative=op,annual_MB_max_m_year=mb_err,years_checked=len(climate['years']),source_limiter=True))
            save('frame-000000',h,u,dict(year=now,step=0,exported=0.,smb_volume=0.,V0=V0))
            state,_=st.restore(root/'frame-000000',contract);assert np.array_equal(state['h'],z['h'])
        while now<target-1e-10:
            assert step<20000 and time.perf_counter()-start<7200
            previous=now;year=int(np.floor(now+1e-9));t=time.perf_counter()
            cache=root/f'forcing-{year}.npz'
            if year!=last_year:
                if cache.exists():
                    seal=st.read(root/f'forcing-{year}.json');assert st.digest(cache)==seal['sha256'];smb=tensor(st.arrays(cache)['smb'])
                else:
                    smb=annual(bp.bed+h,climate,cfg['parameters'],year,backend)
                    with cache.open('xb') as f:np.savez(f,smb=smb.cpu().numpy(),surface=(bp.bed+h).cpu().numpy())
                    st.write(root/f'forcing-{year}.json',dict(sha256=st.digest(cache),year=year))
            pending=root/f'pending-{step+1:06d}'
            if pending.exists():
                state,meta=st.restore(pending,contract);h=tensor(state['h']);u=tensor(state['u']);now=meta['year'];smb_volume=meta['smb_volume'];exported=meta['exported']
            else:
                end=min(target,year+1.,now+cfg['time_step_max_days']/365)
                q=tr.flux(h,u);vol=h*tr.dual;donor=torch.where(q>=0,tr.links[:,0],tr.links[:,1]);out=torch.zeros_like(h);out.index_add_(0,donor,q.abs());wet=(vol>0)&(out>0)
                if bool(wet.any()):end=min(end,now+float((.35*vol[wet]/out[wet]).min()))
                if end-now<1e-12:raise RuntimeError('CFL collapse')
                forced=(h+smb*(end-now)).clamp_min(0);smb_volume+=float((forced-h)@tr.dual)
                h,now,ex,substeps=tr.window(forced,u,now,end);exported+=ex
                meta=dict(year=now,step=step+1,exported=exported,smb_volume=smb_volume,V0=V0)
                save(pending.name,h,u,meta)
            # Recover any saved nonlinear iterate without replaying transport.
            iterations=sorted(root.glob(f'newton-{step+1:06d}-*'))
            if iterations:
                state,_=st.restore(iterations[-1],contract);u=tensor(state['u'])
            bp.prepare(h)
            changed=last_wet is None or not torch.equal(bp.wet,last_wet)
            if year!=last_year or changed:
                if hasattr(bp,'pcg'):
                    torch.cuda.synchronize()
                    if bp.pcg.graph is not None:bp.pcg.graph.reset()
                    del bp.pcg;del mg;gc.collect();torch.cuda.empty_cache()
                bp.evaluate(u,True);mg=hierarchy.build(bp,step+1);bp.pcg=PCG(bp,mg,True);last_wet=bp.wet.clone()
            last_year=year
            def callback(it,v,info):
                name=f'newton-{step+1:06d}-{phase}-{it:03d}'
                if not (root/name).exists():save(name,h,v,{**meta,'iteration':it,'momentum':info})
            u,info=solve(bp,u,mg,callback)
            volume=float(h@tr.dual);mass=abs(volume-V0-smb_volume+exported)/max(V0,abs(smb_volume),1.)
            assert mass<=1e-10 and bool(torch.isfinite(h).all()) and bool(torch.isfinite(u).all()) and bool((h>=0).all())
            step+=1
            row=dict(step=step,year=now,volume_m3=volume,smb_volume_m3=smb_volume,exported_m3=exported,mass_relative=mass,Newton=info['iterations'],CG=sum(x.get('cg_iterations',0) for x in info['history']),BP_residual=info['relative_residual'],max_speed_m_year=float(u.reshape(-1,2,2)[:,-1].norm(dim=1).max()),max_H_m=float(h.max()),ice_area_m2=float(tr.dual[h>0].sum()),seconds=time.perf_counter()-t)
            save(f'frame-{step:06d}',h,u,{**meta,'row':row});rows.append(row)
            ok=st.status(root,dict(stage='running',year=now,step=step,last_checkpoint=f'frame-{step:06d}',pid=os.getpid()),inject=step==1)
            if step==1:assert not ok
            if step==2:st.write(root/'status-survival.json',dict(status='pass',continued_step=2))
            if abs(now-round(now))<1e-8:
                st.write(root/f'year-{round(now)}.json',row);print(json.dumps(row),flush=True)
        result=dict(status='completed',phase=phase,year=now,accepted_steps=step,rows=rows,wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,hierarchy_seconds=hierarchy.seconds,peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        result=dict(status='failed',phase=phase,year=locals().get('now'),accepted_steps=locals().get('step'),error=repr(exc),traceback=traceback.format_exc(),rows=rows,wall_seconds=time.perf_counter()-start)
    finally:hierarchy.close()
    st.write(root/f'result-{phase}.json',result);st.status(root,{k:v for k,v in result.items() if k!='rows'});print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),float(sys.argv[2]),sys.argv[3])
