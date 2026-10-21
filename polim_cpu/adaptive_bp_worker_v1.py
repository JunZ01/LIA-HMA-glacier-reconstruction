"""Bounded saved-state historical comparisons; all outputs remain experimental."""
from .gpu_allocator_recovery_worker_v1 import Hierarchy
from .gpu_energy_worker_v1 import torch, tensor, Transport, Path, np, os, sys, time, json, traceback
from .gpu_allocator_safe_v1 import BP, PCG
from .adaptive_bp_solve_v1 import solve
from .historical_climate_v1 import annual
from . import gpu_long_storage_v1 as st
from types import SimpleNamespace
import gc


def run(root):
    cfg = st.read(root/'config.json')
    contract = st.read(root/'contract.json')['contract']
    source = Path(cfg['source_root'])
    z = st.arrays(source/'input.npz')
    climate = st.arrays(source/'climate.npz')
    old_contract = st.read(source/'contract.json')['contract']
    torch.cuda.set_per_process_memory_fraction(.10)
    start = time.perf_counter()
    bp = BP(z)
    positions = []
    for i in range(bp.ndof):
        a, b = z['csr_indptr'][i:i+2]
        pos = np.flatnonzero(z['csr_indices'][a:b] == i)
        assert len(pos) == 1
        positions.append(int(a + pos[0]))
    bp.diagonal_positions = tensor(positions, True)
    tr = Transport(z)
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
            meta = dict(year=float(cfg['start_year']), step=0, exported=0., smb_volume=0., V0=float(data['h']@z['geo_dual_area']), last_solve=float(cfg['start_year']))
            save('frame-000000', tensor(data['h']), tensor(data['u']), meta)
        h, u = tensor(data['h']), tensor(data['u'])
        now, step = meta['year'], meta['step']
        V0, exported, smb_volume = meta['V0'], meta['exported'], meta['smb_volume']
        last_solve = meta['last_solve']
        prev_u = None
        prev_dt = None
        if step == 0:
            # Warm the predictor using a preceding saved state, never future data.
            prior, prior_meta = st.restore(source/cfg['previous_frame'], old_contract)
            prev_u = tensor(prior['u'])
            prev_dt = now-prior_meta['year']
        fixture_state, _ = st.restore(source/cfg['start_frame'], old_contract)
        bp.prepare(tensor(fixture_state['h']))
        r, _ = bp.evaluate(tensor(fixture_state['u']), True)
        fixture = st.arrays(root/'fixture.npz')
        op = float((bp.hvp(tensor(z['direction']))-tensor(fixture['hvp'])).norm()/tensor(fixture['hvp']).norm().clamp_min(1.))
        assert op <= 1e-9
        bp.prepare(h)
        smb = annual(bp.bed+h, climate, cfg['parameters'], cfg['start_year'], backend)
        if step == 0:
            assert float((smb-tensor(fixture['smb'])).abs().max()) <= 1e-10
            st.write(root/'fixture.json', dict(operator_relative=op, status='pass'))
            with (root/'forcing.npz').open('xb') as f:
                np.savez(f, smb=smb.cpu().numpy())
            st.write(root/'forcing.json', dict(sha256=st.digest(root/'forcing.npz')))
            loaded, _ = st.restore(root/'frame-000000', contract)
            assert np.array_equal(loaded['h'],data['h']) and np.array_equal(loaded['u'],data['u'])
        else:
            assert st.digest(root/'forcing.npz') == st.read(root/'forcing.json')['sha256']
            smb = tensor(st.arrays(root/'forcing.npz')['smb'])
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
            bp.pcg = PCG(bp, mg, True)
            rebuilds += 1
            work = cache/f'hierarchy-{build_index:03d}'
            hashes = {n:st.digest(work/n) for n in ['matrix.npz','hierarchy.npz']}
            for n in hashes:
                p=(work/n).resolve()
                assert p.parent.parent == cache.resolve()
                p.unlink()
            st.write(work/'cache-released.json',dict(source_hashes=hashes,derived_cache=True))
        while now < cfg['end_year']-1e-10:
            assert time.perf_counter()-start < 900 and step < 100
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
            meta=dict(year=now,step=step,exported=exported,smb_volume=smb_volume,V0=V0,last_solve=last_solve,row=row)
            save(f'frame-{step:06d}',h,u,meta)
            rows.append(row)
            ok=st.status(root,dict(status='running',pid=os.getpid(),**row),inject=step==1)
            if step==1:
                assert not ok
            if step==2:
                assert ok
            prev_u,prev_dt=old_u,now-previous
            print(json.dumps(row),flush=True)
            if len(sys.argv)>2 and step>=int(sys.argv[2]):
                break
        torch.cuda.synchronize()
        result=dict(status='completed' if now>=cfg['end_year']-1e-9 else 'intentional-checkpoint-stop',year=now,steps=step,rows=rows,setup_seconds=setup,advance_seconds=time.perf_counter()-advance_start,wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,hierarchy_seconds=hierarchy.seconds,rebuilds=rebuilds,peak_memory_bytes=torch.cuda.max_memory_allocated())
    except Exception as exc:
        result=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),rows=rows,wall_seconds=time.perf_counter()-start)
    finally:
        hierarchy.close()
    st.write(root/f'result-{time.time_ns()}.json',result)
    st.status(root,{k:v for k,v in result.items() if k!='rows'})
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    run(Path(sys.argv[1]))
