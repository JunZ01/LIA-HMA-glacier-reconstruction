"""Bounded saved-state historical comparisons; all outputs remain experimental."""
from polim_cpu.gpu_allocator_recovery_worker_v1 import Hierarchy
from polim_cpu.gpu_energy_worker_v1 import torch, tensor, Transport, Path, np, os, sys, time, json, traceback
from polim_cpu.gpu_allocator_safe_v1 import BP, PCG
from polim_cpu.gpu_bp_v1 import scatter,T
from polim_cpu.adaptive_bp_solve_v1 import solve
from polim_cpu.historical_climate_v1 import annual
from polim_cpu import packed_storage_v1 as st
from polim_cpu.projected_amg_v1 import ProjectedMultigrid
from types import SimpleNamespace
import gc
from collections import defaultdict

class Meter:
    def __init__(self):self.counts=defaultdict(int);self.times=defaultdict(float);self.start=time.perf_counter()
    def tick(self):
        if time.time()>=float(os.environ['CN_C_DEADLINE']):raise RuntimeError('C600s/shared wall budget')
    def record(self):return dict(counts=dict(self.counts),seconds=dict(self.times),wall=time.perf_counter()-self.start,
        note='Nested component timers overlap; subprocess total is authoritative. GPU graph warm/capture CG steps counted separately.')
meter=Meter()
native_solve=solve
def solve(bp,u,refresh,checkpoint=None):
    meter.tick()
    if meter.counts['full_BP']>=128:raise RuntimeError('C128completeBP budget')
    meter.counts['full_BP']+=1;t=time.perf_counter();info=None
    try:
        v,info=native_solve(bp,u,refresh,checkpoint);return v,info
    except BaseException:
        tb=sys.exc_info()[2]
        while tb:
            if tb.tb_frame.f_code is native_solve.__code__:info=dict(iterations=sum('step_length' in x for x in tb.tb_frame.f_locals.get('history',[])))
            tb=tb.tb_next
        raise
    finally:
        if info:meter.counts['Newton']+=info['iterations']
        meter.times['full_BP']+=time.perf_counter()-t

def instrument():
    def wrap(cls,name,key,after=None):
        original=getattr(cls,name)
        def action(self,*a,**kw):
            meter.tick();meter.counts[key]+=1;t=time.perf_counter();result=None
            try:result=original(self,*a,**kw);return result
            finally:
                torch.cuda.synchronize();meter.times[key]+=time.perf_counter()-t
                if after:after(self,a,kw,result,original)
        setattr(cls,name,action)
    def evaluation(self,a,kw,result,original):
        if kw.get('linearize',a[1] if len(a)>1 else False):meter.counts['matrix_build']+=1
    def krylov(self,a,kw,result,original):
        if result is not None:meter.counts['CG']+=result[1]
        else:
            tb=sys.exc_info()[2]
            while tb:
                if tb.tb_frame.f_code is original.__code__:meter.counts['CG']+=tb.tb_frame.f_locals.get('k',0)+tb.tb_frame.f_locals.get('batch',1)
                tb=tb.tb_next
    def capture(self,a,kw,result,original):meter.counts['CG_warm_and_capture']+=8
    wrap(BP,'prepare','geometry_build');wrap(BP,'evaluate','residual_matrix',evaluation)
    wrap(PCG,'solve','Krylov_solve',krylov);wrap(PCG,'capture','graph_capture',capture)
    wrap(Hierarchy,'build','AMG_build');wrap(Transport,'flux','flux_evaluation')


def fast_status(root, obj, inject=False):
    # Monitoring is replaceable; scientific annual checkpoints remain fsynced.
    import uuid
    tmp=root/f'.status-{uuid.uuid4().hex}.json'
    tmp.write_text(json.dumps(obj,allow_nan=False),encoding='utf-8')
    try:
        if inject:raise PermissionError('Injected status lock')
        os.replace(tmp,root/'status.json')
        return True
    except OSError:
        tmp.rename(root/f'status-fallback-{uuid.uuid4().hex}.json')
        return False


def run(root):
    cfg = st.read(root/'config.json')
    contract = st.read(root/'contract.json')['contract']
    source = Path(cfg['source_root'])
    z = st.arrays(source/'input.npz')
    climate = st.arrays(source/'climate.npz')
    old_contract = st.read(source/'contract.json')['contract']
    torch.cuda.set_per_process_memory_fraction(.10)
    start = time.perf_counter()
    meter.start=start
    bp = BP(z)
    positions = []
    for i in range(bp.ndof):
        a, b = z['csr_indptr'][i:i+2]
        pos = np.flatnonzero(z['csr_indices'][a:b] == i)
        assert len(pos) == 1
        positions.append(int(a + pos[0]))
    bp.diagonal_positions = tensor(positions, True)
    tr = RecordedTransport(z)
    backend = SimpleNamespace(xp=torch, array=tensor)
    cache = root/'amg-cache'
    cache.mkdir(exist_ok=True)
    hierarchy = Hierarchy(cache, z)
    mg = None
    io = 0.
    rows = []
    rebuilds = 0
    pending = None
    def save(name, h, u, meta):
        nonlocal io
        # Persist complete years and the explicit restart fixture only.
        if not name.startswith('frame-'):
            return
        if meta['step'] != 0 and meta['year'] < cfg['end_year']-1e-9 and not (len(sys.argv)>2 and meta['step']==int(sys.argv[2])):
            return
        t = time.perf_counter()
        st.save(root, name, h.cpu().numpy(), u.cpu().numpy(), meta, contract)
        io += time.perf_counter() - t
    try:
        frames = sorted(p for p in root.glob('frame-*') if p.is_dir())
        if frames:
            data, meta = st.restore(frames[-1], contract)
            if meta['year'] >= cfg['end_year'] - 1e-9:
                raise RuntimeError('Completed window cannot be replayed')
        else:
            data, old_meta = st.restore(source/cfg['start_frame'], old_contract)
            meta = dict(year=float(cfg['start_year']), step=0, exported=0., smb_volume=0., V0=float(tensor(data['h'])@tr.dual), last_solve=float(cfg['start_year']))
            save('frame-000000', tensor(data['h']), tensor(data['u']), meta)
        h, u = tensor(data['h']), tensor(data['u'])
        now, step = meta['year'], meta['step']
        rows = list(meta.get('window_rows',[]))
        V0, exported, smb_volume = meta['V0'], meta['exported'], meta['smb_volume']
        last_solve = meta['last_solve']
        prev_u = None
        prev_dt = None
        if step == 0:
            # Warm the predictor using a preceding saved state, never future data.
            prior, prior_meta = st.restore(source/cfg['previous_frame'], old_contract)
            prev_u = tensor(prior['u'])
            prev_dt = now-prior_meta['year']
        # Retained v35 operator/transport evidence is reused; no new real fixture.
        bp.prepare(h)
        smb=tensor(st.arrays(Path(cfg['shared_forcing']))['smb'])
        actual_nodes=torch.zeros_like(h);potential_nodes=torch.zeros_like(h)
        setup = time.perf_counter()-start
        advance_start = time.perf_counter()
        build_index = len(list(cache.glob('hierarchy-*')))
        def refresh(it):
            nonlocal mg, rebuilds, build_index
            if hasattr(bp, 'pcg'):
                torch.cuda.synchronize()
                if bp.pcg.graph is not None:
                    bp.pcg.graph.reset()
                del bp.pcg
                mg = None
                gc.collect()
                torch.cuda.empty_cache()
            build_index += 1
            mg = hierarchy.build(bp, build_index)
            bp.pcg = PCG(bp, ProjectedMultigrid(mg,bp), True)
            rebuilds += 1
            work = cache/f'hierarchy-{build_index:03d}'
            hashes = {n:st.digest(work/n) for n in ['matrix.npz','hierarchy.npz']}
            for n in hashes:
                p=(work/n).resolve()
                assert p.parent.parent == cache.resolve()
                p.unlink()
            st.write(work/'cache-released.json',dict(source_hashes=hashes,derived_cache=True))
        while now < cfg['end_year']-1e-10:
            assert time.perf_counter()-start < 600 and step < 100
            meter.tick()
            t = time.perf_counter()
            old_h, old_u, previous = h.clone(), u.clone(), now
            pending = root/f'pending-{step+1:06d}'
            if pending.exists():
                data, pm = st.restore(pending, contract)
                h,u=tensor(data['h']),tensor(data['u'])
                now, exported, smb_volume = pm['year'],pm['exported'],pm['smb_volume']
            else:
                end=min(cfg['end_year'],now+30/365)
                q=tr.flux(h,u)
                vol=h*tr.dual
                donor=torch.where(q>=0,tr.links[:,0],tr.links[:,1])
                out=torch.zeros_like(h)
                out.index_add_(0,donor,q.abs())
                wet=(vol>0)&(out>0)
                if bool(wet.any()):
                    end=min(end,now+float((.35*vol[wet]/out[wet]).min()))
                if end-now < 1e-12:
                    raise RuntimeError('CFL collapse')
                forced=(h+smb*(end-now)).clamp_min(0)
                actual_nodes+=(forced-h)*tr.dual
                potential_nodes+=smb*(end-now)*tr.dual
                smb_volume+=float((forced-h)@tr.dual)
                h,now,ex,substeps=tr.window(forced,u,now,end)
                exported+=ex
                pm=dict(year=now,step=step+1,exported=exported,smb_volume=smb_volume,V0=V0,last_solve=last_solve)
                save(pending.name,h,u,pm)
            bp.prepare(h)
            u=u.clone()
            u[~bp.wet]=0.
            base_r,base_e=bp.evaluate(u)
            res=float(base_r.norm()/bp.force_norm)
            # This variant explicitly permits bounded diagnostic imbalance only
            # between complete BP solves. Annual endpoints always use 1e-6.
            skip=(cfg['mode']=='lagged' and now<cfg['end_year']-1e-9 and now-last_solve<60/365-1e-10 and res<=1e-3)
            predictor='held'
            if skip:
                info=dict(iterations=0,relative_residual=res,history=[])
            else:
                base_norm=base_r.norm()
                ratio=torch.where(old_h>0,h/old_h.clamp_min(1e-300),torch.ones_like(h)).clamp(.1,10.)
                candidates=[('thickness3',u*ratio.pow(3).repeat_interleave(4)),('thickness4',u*ratio.pow(4).repeat_interleave(4))]
                if prev_u is not None and prev_dt and prev_dt>0:
                    extrap=u+(u-prev_u)*min((now-previous)/prev_dt,2.)
                    candidates.insert(0,('time-secant',extrap))
                for label,candidate in candidates:
                    candidate[~bp.wet]=0.
                    rr,ee=bp.evaluate(candidate)
                    if bool(ee<base_e) and bool(rr.norm()<base_norm):
                        u=candidate
                        base_e,base_norm=ee,rr.norm()
                        predictor=label
                iterations=sorted(root.glob(f'newton-{step+1:06d}-*'))
                if iterations:
                    saved,_=st.restore(iterations[-1],contract)
                    u=tensor(saved['u'])
                if mg is None:
                    bp.evaluate(u,True)
                    refresh(0)
                def callback(it,v,history):
                    name=f'newton-{step+1:06d}-{time.time_ns()}-{it:03d}'
                    save(name,h,v,{**pm,'iteration':it,'momentum':history})
                u,info=solve(bp,u,refresh,callback)
                last_solve=now
            mass=abs(float(h@tr.dual)-V0-smb_volume+exported)/max(V0,abs(smb_volume),1.)
            assert mass<=1e-10 and bool(torch.isfinite(u).all()) and bool((h>=0).all())
            step+=1
            row=dict(step=step,year=now,skipped=skip,predictor=predictor,Newton=info['iterations'],CG=sum(x.get('cg_iterations',0) for x in info['history']),BP_residual=info['relative_residual'],mass_relative=mass,seconds=time.perf_counter()-t,rebuilds=rebuilds)
            meta=dict(year=now,step=step,exported=exported,smb_volume=smb_volume,V0=V0,last_solve=last_solve,row=row,window_rows=rows+[row])
            save(f'frame-{step:06d}',h,u,meta)
            rows.append(row)
            ok=fast_status(root,dict(status='running',pid=os.getpid(),**row),inject=False)
            prev_u,prev_dt=old_u,now-previous
            print(json.dumps(row),flush=True)
            if len(sys.argv)>2 and step>=int(sys.argv[2]):
                break
        torch.cuda.synchronize()
        bp.prepare(h);final_r,_=bp.evaluate(u)
        final_bp=float(final_r.norm()/bp.force_norm)
        assert final_bp<=1e-6
        with (root/'annual-ledger.npz').open('xb') as f:
            np.savez(f,H=h.cpu().numpy(),u=u.cpu().numpy(),actual_nodes=actual_nodes.cpu().numpy(),
                potential_nodes=potential_nodes.cpu().numpy(),unmet_nodes=(actual_nodes-potential_nodes).cpu().numpy(),
                internal_volume=tr.integrals[:len(tr.edges)].cpu().numpy(),boundary_volume=tr.integrals[len(tr.edges):].cpu().numpy())
            f.flush();os.fsync(f.fileno())
        st.write(root/'endpoint-audit.json',dict(year=now,BP=final_bp,additional_full_BP=0,metadata=meta))
        result=dict(status='completed' if now>=cfg['end_year']-1e-9 else 'intentional-checkpoint-stop',year=now,steps=step,rows=rows,setup_seconds=setup,advance_seconds=time.perf_counter()-advance_start,wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,hierarchy_seconds=hierarchy.seconds,rebuilds=rebuilds,peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        if 'h' in locals():
            with (root/'failure-state.npz').open('xb') as f:np.savez(f,H=h.cpu().numpy(),u=u.cpu().numpy(),year=np.array(now),accepted=np.array(False))
        result=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),rows=rows,wall_seconds=time.perf_counter()-start)
    finally:
        hierarchy.close()
        st.write(root/'native-costs.json',meter.record())
    st.write(root/f'result-{time.time_ns()}.json',result)
    fast_status(root,{k:v for k,v in result.items() if k!='rows'})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)




class RecordedTransport(Transport):
    def __init__(self,z):
        super().__init__(z);self.integrals=torch.zeros(len(self.links),device=self.dual.device,dtype=self.dual.dtype)
    def window(self,h,u,now,end):
        exported=0.;steps=0
        while now<end-1e-16:
            q=self.flux(h,u);volume=h*self.dual;donor=torch.where(q>=0,self.links[:,0],self.links[:,1]);receiver=torch.where(q>=0,self.links[:,1],self.links[:,0])
            out=scatter(q.abs(),donor,self.n);wet=(volume>0)&(out>0)
            dt=min(end-now,float((.35*volume[wet]/out[wet]).min()) if bool(wet.any()) else end-now)
            if dt<1e-13:raise RuntimeError('GPU CFL collapse')
            requested=q.abs()*dt;outgoing=scatter(requested,donor,self.n);limited=outgoing>volume
            factor=torch.where(limited,volume/outgoing.clamp_min(1e-300)*(1-64*torch.finfo(T).eps),torch.ones_like(volume))
            actual=requested*factor[donor];retained=volume-scatter(actual,donor,self.n)
            if bool((retained<0).any()):raise RuntimeError('GPU donor limiting failed')
            internal=receiver>=0;arriving=scatter(actual[internal],receiver[internal],self.n)
            self.integrals+=torch.sign(q)*actual
            meter.counts["transport_substeps"]+=1
            h=(retained+arriving)/self.dual;exported+=float(actual[~internal].sum());now+=dt;steps+=1
            if steps>100:raise RuntimeError('GPU transport budget')
        return h,now,exported,steps


if __name__=='__main__':
    instrument()
    run(Path(sys.argv[1]))
