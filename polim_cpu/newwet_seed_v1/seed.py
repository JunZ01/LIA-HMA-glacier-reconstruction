"""Bounded, transactional new-wet initial guess after ordinary adoption."""
import copy,hashlib
import numpy as np
from .harmonic import predict

EQ_KEYS=('h0','smb','dt','q0','o0','alpha','beta','revision','area','V0')
def snapshot(p,base):
    return dict(eq={k:copy.deepcopy(getattr(p.mass.eq,k)) for k in EQ_KEYS},
        last={k:copy.deepcopy(getattr(p.mass,k)) for k in ('last','last_h','last_u')},
        ledger=copy.deepcopy(p.mass.ledger),inherited=copy.deepcopy(p.mass.inherited),
        selection=p.mass.selection,current=p.b.current,base=copy.deepcopy(base))
def restore(p,back):
    for k,v in back['eq'].items():setattr(p.mass.eq,k,copy.deepcopy(v))
    for k,v in back['last'].items():setattr(p.mass,k,copy.deepcopy(v))
    p.mass.ledger=copy.deepcopy(back['ledger']);p.mass.inherited=copy.deepcopy(back['inherited']);p.mass.selection=back['selection']
    # Pure GPU residuals do not change the bind or AMG. An identity failure is
    # a technical stop, never a hidden rebind/assembly to repair a rollback.
    if p.b.current!=back['current']:raise RuntimeError('Seed mutated bound operator identity')
    return copy.deepcopy(back['base'])
def scientific_hash(p,base):
    d=hashlib.sha256()
    for k in EQ_KEYS:
        a=np.asarray(getattr(p.mass.eq,k));d.update(k.encode());d.update(a.tobytes())
    for k in ('H','u','F','force','wet'):d.update(np.ascontiguousarray(base[k]).tobytes())
    return d.hexdigest()

def after_adopt(p,parent,base,ordinary_used):
    from .coupling import scales,merit
    from polim_cpu.cn_flux_time_v1.mass import accepted
    new=base['wet'][::4]&~parent['wet'][::4]
    if p.seed_seen or not new.any():return base
    p.seed_seen=True;nodes=np.flatnonzero(new);w=p.w
    w.event('seed-trigger',enabled=p.seed_enabled,parent=parent['name'],base=base['name'],new_nodes=nodes.tolist(),ordinary_used=ordinary_used,remaining_candidates=8-ordinary_used)
    if not p.seed_enabled:return base
    if ordinary_used>=8 or w.counts['pure_joint_residual']>=128:
        w.event('seed-skipped',reason='shared candidate/evaluation budget');return base
    with w.span('seed_initialization'):
        back=snapshot(p,base);signature=scientific_hash(p,base);scale=scales(base,p.mass);bm=merit(base,scale)
        w.save('seed-rollback-base',H=base['H'],u=base['u'],F=base['F'],R=base['e']['R'],wet=base['wet'],alpha=p.mass.eq.alpha,beta=p.mass.eq.beta,
               actual_nodes=base['e']['source'],unmet=base['e']['U'],internal_volume=base['e']['qi'],boundary_volume=base['e']['oi'])
        try:
            w.count('seed_graph_construction',1)
            with w.span('seed_graph'):
                guess,info,data=predict(p.z['xy'],p.z['tri'],base['H'],base['u'],parent['wet'][::4],base['wet'][::4],work=w)
            delta=guess-base['u'];mask=np.repeat(new,4);assert np.all(delta[~mask]==0)
            w.save('seed-graph',H=base['H'],u_base=base['u'],u_seed=guess,delta_u=delta,new_nodes=nodes,**data)
            w.event('seed-graph',**info,delta_norm=float(np.linalg.norm(delta)),u_seed_norm=float(np.linalg.norm(guess[mask])))
            if not info.get('effective',False):
                w.event('seed-unavailable',reason=info['reason']);return restore(p,back)
            for index,lam in enumerate((1.,.5,.25,.125)):
                if ordinary_used+index>=8 or w.counts['pure_joint_residual']>=128:
                    w.event('seed-budget-stop',ordinary_used=ordinary_used,seed_used=index);break
                w.count('seed_trials',4);w.count('shared_direction_candidates')
                u=base['u']+lam*delta;u[~base['wet']]=0.
                assert np.array_equal(u[~mask],base['u'][~mask])
                t,_=p.state(base['H'],u,'seed',locked=True,material_nodes=nodes)
                consistent=all(np.array_equal(getattr(p.mass.eq,k),v) for k,v in back['eq'].items())
                if not consistent:raise RuntimeError('Locked seed trial changed discrete equation')
                physical=t['d']['physical_path'] and t['d']['all_finite'] and t['d']['H_min']>=0
                fixed_ok=t['locked']['ok'];tm=merit(t,scale) if fixed_ok else None
                good=bool(fixed_ok and physical and (tm<=(1-1e-4*lam)*bm or accepted(t['d'],t['bp'])))
                why='accepted' if good else ('fixed-branch transport/path requires change or failed' if not fixed_ok else 'physical path' if not physical else 'joint descent')
                w.save(f'seed-trial-{index+1}',H=t['H'],u=t['u'],delta_u=u-base['u'],F=t['F'],R=t['e']['R'],alpha=p.mass.eq.alpha,beta=p.mass.eq.beta)
                w.event('seed-trial',index=index+1,lam=lam,base=base['name'],trial=t['name'],base_merit=bm,trial_merit=tm,
                    bp=t['bp'],local=t['d']['local'],mass=t['d']['mass'],fixed_point=t['d']['fixed_point'],cumulative_mass=t['d']['cumulative_mass_relative'],
                    physical=physical,locked=t['locked'],adopted=good,reason=why,shared_candidates_used=ordinary_used+index+1,
                    delta_norm=float(np.linalg.norm(u-base['u'])),H_unchanged=bool(np.array_equal(t['H'],base['H'])),bound_point=p.b.current)
                if good:
                    w.count('seed_adopted');w.event('seed-next-J-required',state=t['name'],point=t['point'],old_bound=p.b.current)
                    return t
                # Recover scientific state, not counters, diagnostic serials or
                # immutable evidence. No GPU work is needed for this restoration.
                with w.span('seed_rollback'):restore(p,back)
            result=restore(p,back)
            assert scientific_hash(p,result)==signature
            w.event('seed-rollback-complete',base=base['name'],scientific_hash=signature,accepted=False)
            return result
        except BaseException as exc:
            result=restore(p,back)
            w.event('seed-exception',error=repr(exc),restored=scientific_hash(p,result)==signature,base=base['name'])
            raise
