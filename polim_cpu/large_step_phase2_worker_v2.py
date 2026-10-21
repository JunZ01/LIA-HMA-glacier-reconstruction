"""One disclosed phase2 repair: cumulative mass acceptance before committing state."""
import sys, os, json, time, traceback, subprocess, queue, threading
from pathlib import Path
from types import SimpleNamespace
from .gpu_energy_worker_v1 import np, torch, tensor
from .historical_climate_v1 import annual
from .large_step_gpu_bridge_v1 import ProjectedAdaptiveBPSolver
from .large_step_transport_client_v1 import IsolatedTransportService
from . import packed_storage_v1 as st
from .large_step_phase2_common_v1 import SOURCE, P1, P1E, E, NAME, PROJECT, GATES, BUDGET
from .large_step_phase2_solve_v1 import solve as traced_solve


def npz(path, **arrays):
    with path.open('xb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())


def status(root, record, inject=False):
    try:
        return st.status(root, record, inject=inject)
    except OSError as exc:
        print(json.dumps(dict(status_write_error=repr(exc), **record)), flush=True)
        return False


class Transport(IsolatedTransportService):
    def __init__(self, z, root):
        self.root=root;root.mkdir()
        self.static_path=root/'transport-static.npz';npz(self.static_path,**{k:v for k,v in z.items() if k.startswith('geo_')})
        env=os.environ.copy();env.pop('POLIM_TORCH_SITE',None);env.pop('KMP_DUPLICATE_LIB_OK',None)
        home=Path('D:/ProgramData/Anaconda3');win=Path(os.environ.get('SystemRoot','C:/Windows'))
        env['PATH']=os.pathsep.join(map(str,[home,home/'Library/bin',home/'DLLs',win/'System32',win]));env['PYTHONPATH']=str(PROJECT)
        self.stderr=(root/'stderr.log').open('x')
        self.process=subprocess.Popen([str(home/'python.exe'),'-m','polim_cpu.large_step_phase2_service_v1',str(self.static_path)],
            cwd=PROJECT,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,text=True,creationflags=0x08000000)
        self.messages=queue.Queue()
        def readlines():
            for line in self.process.stdout:self.messages.put(line)
            self.messages.put('')
        threading.Thread(target=readlines,daemon=True).start()
        assert json.loads(self.messages.get(timeout=60))['ready']
        # Base RPC retains immutable request/response handling; bound its reads.
        self.reader=self.process.stdout
        self.process.stdout=SimpleNamespace(readline=lambda:self.messages.get(timeout=120))


class BP(ProjectedAdaptiveBPSolver):
    def __init__(self,z,root):
        super().__init__(z,root);self.build_seconds=0.;self.call_seconds=0.;self.trace=[];self.total_calls=0
        self.total_cg=0;self.graph_setup_cg=0
        owner=self
        class TracedPCG(self._PCG):
            def solve(self,r):
                if self.use_graph and self.graph is None:
                    self.capture(r);owner.graph_setup_cg+=8
                self.reset(r);rn=r.norm();batch=4 if self.use_graph else 1
                for k in range(0,200,batch):
                    if self.graph:self.graph.replay()
                    else:self.step()
                    owner.total_cg+=batch
                    ratio=float((self.rr/self.mg.d).norm()/rn)
                    if ratio<=.03:break
                else:raise RuntimeError('200CG budget')
                if bool(self.bad>0):raise RuntimeError('Nonpositive CG curvature')
                delta=self.mg.d*self.y;true=float((self.bp.hvp(delta)+r).norm()/rn)
                if not true<=.031:raise RuntimeError('True linear residual failed')
                return delta,k+batch,true
        self._PCG=TracedPCG
        self._solve=lambda bp,u,refresh:traced_solve(bp,u,refresh,trace=self.trace)

    def _refresh(self,iteration):
        t=time.perf_counter()
        try:super()._refresh(iteration)
        finally:self.build_seconds+=time.perf_counter()-t

    def __call__(self,h,u):
        self.trace.clear();self.total_calls+=1;t=time.perf_counter()
        try:return super().__call__(h,u)
        finally:self.call_seconds+=time.perf_counter()-t


class Rejected(RuntimeError):
    pass


def attempt(h0,u0,z,smb,dt,tr,bp,ledger):
    """Never update H after a successful BP check without another counted BP call."""
    t=time.perf_counter();timing=dict(predict=0.,transport=0.,bp=0.,build=0.)
    history=[];b0=bp.build_seconds;c0=bp.call_seconds
    try:
        tp=time.perf_counter();h=tr.predict(h0,u0,z['bed'],smb,dt);timing['predict']+=time.perf_counter()-tp
        u_guess=u0
        for outer in range(1,BUDGET['bp_per_attempt']+1):
            if not np.isfinite(h).all() or np.any(h<0):raise Rejected('invalid predicted/corrected thickness')
            row=dict(outer=outer)
            history.append(row)
            cg0=bp.total_cg;graph0=bp.graph_setup_cg
            try:u,info=bp(h,u_guess)
            finally:
                row['newton']=sum(x.get('newton_attempted',False) for x in bp.trace)
                row['cg']=bp.total_cg-cg0
                row['graph_setup_cg']=bp.graph_setup_cg-graph0
                row['nonlinear_history']=list(bp.trace)
            row['bp']=float(info['relative_residual'])
            tt=time.perf_counter();c=tr.correct(h0,h,u,smb,dt);timing['transport']+=time.perf_counter()-tt
            for key in ('mass_relative','thickness_equation_relative','fixed_point_relative'):
                row[key]=float(c[key])
            row['active_set_equal']=bool(c['active_set_equal'])
            row['active_iterations']=int(c['active_iterations'])
            source_total=ledger['source']+float(c['actual_source_volume'])
            export_total=ledger['exported']+float(c['boundary_export_volume'])
            row['cumulative_mass_relative']=abs(float(np.sum(z['geo_dual_area']*h))-ledger['V0']-source_total+export_total)/max(ledger['V0'],abs(source_total),abs(export_total),1.)
            finite=all(np.isfinite(a).all() for a in (h,u,smb,*c.values()))
            if row['cumulative_mass_relative']<=GATES['mass'] and finite and row['active_set_equal'] and row['bp']<=GATES['bp'] and row['mass_relative']<=GATES['mass'] and row['thickness_equation_relative']<=GATES['local'] and row['fixed_point_relative']<=GATES['local']:
                # h is exactly the BP geometry and the flux donor geometry.
                return dict(h=h,u=u,correction=c,history=history)
            h=c['thickness'];u_guess=u  # Frozen full Picard update, omega=1.
        raise Rejected('eight complete BP corrections exhausted')
    finally:
        timing['build']=bp.build_seconds-b0
        timing['bp']=bp.call_seconds-c0-timing['build']
        timing['total']=time.perf_counter()-t
        attempt.last=dict(history=history,timing=timing)


def run(root,target,segment):
    global NAME,E
    started=time.perf_counter();cfg=st.read(root/'config.json');contract=st.read(root/'contract.json')['contract']
    NAME=cfg.get('branch_name',NAME);E=Path(cfg.get('audit_evidence_root',str(E)))
    assert st.read(PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    z=st.arrays(SOURCE/'input.npz');climate=st.arrays(SOURCE/'climate.npz')
    torch.cuda.set_per_process_memory_fraction(.10)
    backend=SimpleNamespace(xp=torch,array=tensor);area=z['geo_dual_area']
    io=0.;restore_seconds=0.;attempts=[];bp=None;tr=None;report={};initialization=0.
    try:
        t=time.perf_counter();frames=sorted(root.glob('frame-*'))
        if frames:
            state,meta=st.restore(frames[-1],contract)
            # Hash-protected forcing cached by the prior process, never refreshed at an intermediate state.
            fmeta=meta['forcing'];cache=root/fmeta['path'];assert st.digest(cache)==fmeta['sha256']
            loaded=st.arrays(cache);assert st.array_hash(loaded['smb'])==fmeta['smb_sha256']
            restore_evidence=dict(source=frames[-1].name,year=meta['year'],forcing_sha256=fmeta['sha256'],
                h_sha256=st.array_hash(state['h']),u_sha256=st.array_hash(state['u']),pid=os.getpid())
            st.write(root/f'restore-{segment}.json',restore_evidence)
        else:
            state,sm=st.restore(SOURCE/cfg['source_frame'],st.read(SOURCE/'contract.json')['contract'])
            meta=dict(year=float(cfg['start']),dt_cap=cfg['max_dt'],accepted=0,rejected=0,attempt_count=0,bp_calls=0,
                source=0.,potential=0.,unmet=0.,exported=0.,V0=float(np.sum(area*state['h'])),rows=[],prior_seconds=0.)
        restore_seconds=time.perf_counter()-t;h,u=state['h'],state['u'];now=meta['year']
        def forcing(year,at_h):
            nonlocal io
            t=time.perf_counter();path=root/f'forcing-{year}.npz'
            if path.exists():
                record=st.read(root/f'forcing-{year}.json');assert st.digest(path)==record['sha256']
                f=st.arrays(path);assert st.array_hash(f['smb'])==record['smb_sha256']
                if now==float(year):assert np.array_equal(f['surface'],z['bed']+at_h)
            else:
                surface=z['bed']+at_h
                smb=annual(tensor(surface),climate,cfg['parameters'],year,backend).cpu().numpy()
                npz(path,smb=smb,surface=surface)
                record=dict(path=path.name,sha256=st.digest(path),smb_sha256=st.array_hash(smb),year=year,
                            climate_sha256=cfg['climate_sha256'],start_h_sha256=st.array_hash(at_h))
                st.write(root/f'forcing-{year}.json',record);f=dict(smb=smb,surface=surface)
            io+=time.perf_counter()-t
            return f['smb'],record

        smb,fr=forcing(int(now),h)
        if not frames and cfg['adopt_phase1']:
            # Same independent 1482 source; retain its already completed identical first quarter.
            p,pm=st.restore(P1/'frame-000001',st.read(P1/'contract.json')['contract'])
            audit=st.read(E/'phase1-audit.json')
            assert np.array_equal(smb,st.arrays(SOURCE/'forcing-1482.npz')['smb'])
            old=h;h,u=p['h'],p['u'];now=1482.25
            row=dict(start=1482.,end=now,dt=.25,adopted=True,bp_calls=6,history=pm['history'],
                source=pm['actual_source_volume'],potential=audit['potential_source_volume'],
                unmet=audit['unmet_ablation_volume'],exported=pm['boundary_export_volume'],
                mass_relative=pm['mass_relative'],local_relative=pm['thickness_equation_relative'],bp=pm['bp_relative'],
                cost=st.read(P1/'completion.json')['wall_seconds'])
            meta.update(year=now,accepted=1,bp_calls=6,source=row['source'],potential=row['potential'],
                        unmet=row['unmet'],exported=row['exported'],rows=[row],prior_seconds=row['cost'])
        def checkpoint(label):
            nonlocal io
            t=time.perf_counter();meta.update(year=now,forcing=fr)
            st.save(root,label,h,u,meta,contract);restored,rm=st.restore(root/label,contract)
            assert np.array_equal(restored['h'],h) and np.array_equal(restored['u'],u)
            io+=time.perf_counter()-t
        if not frames:checkpoint('frame-000000')
        survival=status(root,dict(status='running',year=now,pid=os.getpid()),inject=cfg['adopt_phase1'] and segment=='first')
        tr=Transport(z,root/f'transport-{segment}');bp=BP(z,root/f'amg-{segment}')
        initialization=time.perf_counter()-started
        prior_at_entry=meta['prior_seconds']
        while now<target:
            if time.perf_counter()-started+prior_at_entry>=BUDGET['case_seconds']:raise Rejected('case wall-time budget exhausted')
            if meta['attempt_count']>=BUDGET['attempts'] or meta['accepted']>=BUDGET['accepted_steps'] or meta['bp_calls']+8>BUDGET['bp_calls']:raise Rejected('case total work budget exhausted')
            year=int(now);smb,fr=forcing(year,h)
            dt=min(meta['dt_cap'],year+1.-now,target-now)
            attempt_id=meta['attempt_count']+1;t=time.perf_counter();calls0=bp.total_calls
            failure=None;result=None
            try:result=attempt(h,u,z,smb,dt,tr,bp,meta)
            except (Rejected,RuntimeError) as exc:
                # Only declared nonlinear failures permit halving. IO/helper errors are technical failures.
                if isinstance(exc,Rejected) or str(exc) in ('Adaptive Newton budget','Line search failed','200CG budget','True linear residual failed','Nonpositive CG curvature','Non-descent Newton direction'):
                    failure=repr(exc)
                else:raise
            cost=attempt.last;record=dict(id=attempt_id,start=now,dt=dt,bp_calls=bp.total_calls-calls0,
                status='rejected' if failure else 'accepted',error=failure,**cost)
            meta['attempt_count']+=1;meta['bp_calls']+=record['bp_calls']
            if failure:
                meta['rejected']+=1;record['ledger_unchanged']=True
                st.write(root/f'attempt-{attempt_id:04d}.json',record);attempts.append(record)
                status(root,dict(status='running',year=now,accepted=meta['accepted'],rejected=meta['rejected'],last_error=failure))
                if meta['rejected']>=BUDGET['rejections'] or dt/2<BUDGET['min_dt']:
                    raise Rejected('bounded fallback exhausted: '+failure)
                meta['dt_cap']=dt/2 # No automatic growth after a fallback.
                continue
            candidate_h,candidate_u=result['h'],result['u'];c=result['correction']
            source_total=meta['source']+float(c['actual_source_volume']);export_total=meta['exported']+float(c['boundary_export_volume'])
            global_mass=abs(float(np.sum(area*candidate_h))-meta['V0']-source_total+export_total)/max(meta['V0'],abs(source_total),abs(export_total),1.)
            if global_mass>GATES['mass']:raise RuntimeError('Accepted candidate failed cumulative mass verification before commit')
            h,u=candidate_h,candidate_u;oldnow=now;now+=dt
            source=float(c['actual_source_volume']);potential=float(c['potential_source_volume'])
            unmet=float(c['unmet_ablation_volume'].sum());exported=float(c['boundary_export_volume'])
            meta['accepted']+=1
            for key,value in [('source',source),('potential',potential),('unmet',unmet),('exported',exported)]:meta[key]+=value
            global_mass=abs(float(np.sum(area*h))-meta['V0']-meta['source']+meta['exported'])/max(meta['V0'],abs(meta['source']),abs(meta['exported']),1.)
            assert global_mass<=GATES['mass']
            ti=time.perf_counter();ledger=root/f'ledger-{attempt_id:04d}.npz'
            npz(ledger,potential_smb=smb,actual_smb=c['actual_smb'],unmet_ablation_volume=c['unmet_ablation_volume'],
                internal_flux=c['internal_flux'],boundary_flux=c['boundary_flux'],local_residual=c['local_residual'],
                transport_active=h==0,BP_active=bp.bp.wet.cpu().numpy())
            record.update(end=now,source=source,potential=potential,unmet=unmet,exported=exported,
                mass_relative=float(c['mass_relative']),local_relative=float(c['thickness_equation_relative']),
                global_mass_relative=global_mass,bp=result['history'][-1]['bp'],h_sha256=st.array_hash(h),u_sha256=st.array_hash(u),
                ledger=ledger.name,ledger_sha256=st.digest(ledger),forcing=fr)
            st.write(root/f'attempt-{attempt_id:04d}.json',record);io+=time.perf_counter()-ti
            attempts.append(record);meta['rows'].append(record)
            if now==float(int(now)):
                # Cache the next climate year at its exact initial surface before the process boundary.
                if now<cfg['end']:smb,fr=forcing(int(now),h)
                meta['prior_seconds']=prior_at_entry+time.perf_counter()-started
                checkpoint(f'frame-{int(now):06d}')
            status(root,dict(status='running',year=now,accepted=meta['accepted'],rejected=meta['rejected'],pid=os.getpid()))
            print(json.dumps(dict(year=now,dt=dt,bp_calls=record['bp_calls'],accepted=meta['accepted'],rejected=meta['rejected'])),flush=True)
        report=dict(status='completed',year=now,accepted=meta['accepted'],rejected=meta['rejected'],
            source=meta['source'],potential=meta['potential'],unmet=meta['unmet'],exported=meta['exported'],
            progress_unavailable_survived=cfg['adopt_phase1'] and segment=='first' and not survival,
            resumed_and_advanced=bool(frames and now>restore_evidence['year']))
    except Exception as exc:
        if bp and hasattr(attempt,'last'):
            st.write(root/f'interrupted-{segment}.json',dict(total_bp_calls=bp.total_calls,total_cg=bp.total_cg,attempt=getattr(attempt,'last',{})))
        report=dict(status='numerical-failure' if isinstance(exc,Rejected) else 'technical-failure',error=repr(exc),
            traceback=traceback.format_exc(),year=locals().get('now'),accepted=locals().get('meta',{}).get('accepted',0),
            rejected=locals().get('meta',{}).get('rejected',0))
    finally:
        for resource in (bp,tr):
            if resource:
                try:resource.close()
                except Exception as exc:report.update(status='technical-failure',close_error=repr(exc))
    report.update(segment=segment,pid=os.getpid(),wall_seconds=time.perf_counter()-started,
        initialization_seconds=initialization,restore_seconds=restore_seconds,io_seconds=io,attempts=attempts,
        total_bp_calls=bp.total_calls if bp else 0,total_cg=bp.total_cg if bp else 0,
        build_seconds=bp.build_seconds if bp else 0.,bp_including_build_seconds=bp.call_seconds if bp else 0.,
        peak_memory_bytes=torch.cuda.max_memory_allocated())
    st.write(root/f'result-{segment}.json',report);status(root,{k:v for k,v in report.items() if k not in ('attempts','traceback')})
    print(json.dumps({k:v for k,v in report.items() if k not in ('attempts','traceback')}),flush=True)


if __name__=='__main__':run(Path(sys.argv[1]),float(sys.argv[2]),sys.argv[3])
