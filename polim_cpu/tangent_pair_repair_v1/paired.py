"""One actual Fu, one shared hierarchy, isolated workspaces; no BP/time step."""
import sys,time,json,hashlib,traceback,copy
from pathlib import Path
import numpy as np
from polim_cpu.cn_cuda_v1.device import Engine,CG,Meter,fh,tensor,torch,T
from polim_cpu.tangent_reuse_v1.space import Space,solve_initial,VERSION

def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def save(p,**a):
    with Path(p).open('xb') as f:np.savez(f,**a)

class PrivatePreconditioner:
    def __init__(self,base,d,active):self.base=base;self.d=d.clone();self.active=active.clone()
    def cycle(self,b):return self.base.cycle(b*self.active)*self.active

def hierarchy_coefficients(base):
    out=dict(coarse_inverse=base.coarse_inverse.cpu().numpy(),base_scaling=base.d.cpu().numpy())
    for kind in ['a','p','r']:
        for i,m in enumerate(getattr(base,kind)):
            for key,val in [('values',m.values()),('indptr',m.crow_indices()),('indices',m.col_indices())]:out[f'{kind}{i}_{key}']=val.cpu().numpy()
    for i,val in enumerate(base.inv):out[f'inverse_diagonal{i}']=val.cpu().numpy()
    out['omega']=np.asarray(base.omega)
    return out

def run(config):
    cfg=read(config);root=Path(cfg['root']);plan=read(cfg['plan']);states=plan['replay_states'];start=time.perf_counter();e=None;space=Space();rows=[];groups=[];candidate_ok=True;common_ok=True;reason=None;requests=0
    try:
        with np.load(cfg['input']) as z:inputs={k:z[k] for k in z.files}
        e=Engine(inputs,root,False);e.begin(cfg['deadline']);work={}
        write(root/'environment.json',dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name(),python=sys.version))
        for group,s in enumerate(states):
            e.m.tick();gp=root/f'group-{group:02d}';gp.mkdir();gstart=time.perf_counter()
            with e.m.span('common_input_assembly'):
                a=dict(np.load(s['candidate']));n=dict(np.load(s['newton']));F,_=e.bind(a['H'],a['u'],True)
                dh=np.where(e.bp.wet[::4].cpu().numpy(),n['deltaH'],0.)
                rhs0=F.cpu().numpy().copy();rhs1=(-fh(e.bp,e.bound_u,tensor(dh))).cpu().numpy()
            with e.m.span('common_input_save_hash'):
                saved=dict(H=a['H'],u=a['u'],indptr=inputs['csr_indptr'],indices=inputs['csr_indices'],values=e.bp.values.cpu().numpy().copy(),wet=e.bp.wet.cpu().numpy(),active=e.bp.active.cpu().numpy().copy(),rhs0=rhs0,rhs1=rhs1)
                hashes={k:ah(v) for k,v in saved.items()};fid=hashlib.sha256(json.dumps({k:hashes[k] for k in ['indptr','indices','values','active','wet','H','u']},sort_keys=True).encode()+e.bp.configuration.encode()).hexdigest()
                save(gp/'common-Fu-rhs.npz',**saved);write(gp/'common-input.json',dict(Fu_id=fid,hashes=hashes,file_sha256=digest(gp/'common-Fu-rhs.npz'),operator='diag(active)*CSR(values)*diag(active)+diag(1-active)',configuration=e.bp.configuration,version=VERSION))
            # Original AMG hierarchy constructed exactly once for this Fu.
            if e.cg and e.cg.graph:e.cg.graph.reset()
            e.cg=None;e.mg=None;e.rebuild_pending=False;e.build_wet=None
            bt=time.perf_counter();e.ensure_preconditioner();build_seconds=time.perf_counter()-bt
            base=e.mg.base;d=e.mg.d;active=e.bp.active
            pre=[PrivatePreconditioner(base,d,active) for _ in range(2)];cg=[CG(e.bp,pre[k],True) for k in range(2)]
            assert pre[0].d.data_ptr()!=pre[1].d.data_ptr() and pre[0].active.data_ptr()!=pre[1].active.data_ptr()
            for field in ['y','rr','p','rz','bad']:assert getattr(cg[0],field).data_ptr()!=getattr(cg[1],field).data_ptr()
            common_levels=root/'amg'/f'hierarchy-{e.builds:03d}'
            with e.m.span('common_AMG_save_hash'):
                actual_mg=hierarchy_coefficients(base);mg_hashes={k:ah(v) for k,v in actual_mg.items()}
                save(gp/'actual-preconditioner.npz',d=d.cpu().numpy(),active=active.cpu().numpy(),**actual_mg)
            prehash=digest(gp/'actual-preconditioner.npz');levelhash=digest(common_levels/'hierarchy.npz')
            space.bind(fid);order=[0,1] if group%2==0 else [1,0];solutions={};pair=[];group_rows=[]
            for strategy in order:
                if strategy==1 and not candidate_ok:continue
                m=Meter();m.deadline=cfg['deadline'];key='reuse' if strategy else 'zero';st=time.perf_counter()
                # Common input errors stop the batch, including on the reuse side.
                shared=dict(np.load(gp/'common-Fu-rhs.npz'))
                assert all(ah(shared[k])==v for k,v in hashes.items())
                try:
                    for j in [0,1]:
                        if requests>=128:raise RuntimeError('128 replay requests')
                        requests+=1;request_start=time.perf_counter();m.counts['tangent']+=1;before=len(space.Z) if strategy else 0
                        with m.span('rhs_transfer'):b=tensor(shared['rhs'+str(j)])
                        with m.span('initial_guess'):
                            x0,info=space.guess(b,e.bp.wet,e.bp.hvp) if strategy else (torch.zeros_like(b),dict(reason='original-zero',rank=0,initial_relative=1.))
                        with m.span('initial_true_residual'):bn=float(b.norm());initial=float((e.bp.hvp(x0)-b).norm());xn=float(x0.norm())
                        with m.span('CG_solve'):
                            if strategy:x,k,true=solve_initial(cg[strategy],b,x0,1e-10,1e-8,200,m,'tangent')
                            elif bn==0:x,k,true=torch.zeros_like(b),0,0.
                            else:x,k,true=cg[strategy].solve_checked(b,1e-10,1e-8,200,m,'tangent')
                        with m.span('final_true_residual'):ax=e.bp.hvp(x);true=float((ax-b).norm())/(bn or 1.)
                        qualified=true<=1e-8 and bool(torch.isfinite(x).all()) and bool((x[~e.bp.wet]==0).all()) and k<=200
                        update='disabled';after=0
                        if strategy and qualified:
                            with m.span('space_maintenance'):update=space.add(x,ax,e.bp.wet)
                            after=len(space.Z)
                        torch.cuda.synchronize();solve_seconds=time.perf_counter()-request_start
                        with m.span('response_save'):
                            xcpu=x.cpu().numpy();save(gp/f'{key}-{j}.npz',rhs=shared['rhs'+str(j)],x0=x0.cpu().numpy(),x=xcpu,Ax=ax.cpu().numpy())
                        row=dict(group=group,case=s['case'],strategy=key,rhs=j,Fu_id=fid,rhs_hash=hashes['rhs'+str(j)],space_before=before,space_after=after,
                            hit=bool(strategy and info['reason']=='SVD'),update=update,evicted=bool(strategy and before==8 and update=='added'),clear_reason='new Fu group' if j==0 and strategy else None,
                            b_norm=bn,x0_norm=xn,initial_residual=initial,initial_relative=initial/(bn or 1.),initial_improvement=1-initial/bn if bn else None,initial=info,CG_iterations=k,true_relative=true,
                            passed=qualified,solve_seconds=solve_seconds,graph_preparation_in_iteration_count=8 if j==0 and k else 0)
                        write(gp/f'{key}-{j}.json',row);rows.append(row);group_rows.append(row);solutions[(strategy,j)]=xcpu
                        if not qualified:raise RuntimeError(key+' true residual/projection failed')
                except BaseException as exc:
                    failure=dict(group=group,strategy=key,error=repr(exc),traceback=traceback.format_exc(),work=m.record());write(gp/(key+'-failure.json'),failure)
                    if not strategy:raise
                    candidate_ok=False
                finally:
                    torch.cuda.synchronize();work[(group,strategy)]=dict(seconds=time.perf_counter()-st,meter=m.record())
                    write(gp/(key+'-work.json'),work[(group,strategy)])
                    if cg[strategy].graph:cg[strategy].graph.reset()
            # Save all difference quantities before classifying the candidate.
            for j in [0,1]:
                if (0,j) in solutions and (1,j) in solutions:
                    xx=solutions[(0,j)];yy=solutions[(1,j)];norm=float(np.linalg.norm(xx));err=float(np.linalg.norm(yy-xx));ok=err/norm<=1e-5 if norm>1e-10 else err<=1e-10
                    pair.append(dict(rhs=j,absolute=err,denominator=norm,relative=err/norm if norm else None,threshold=1e-5 if norm>1e-10 else 1e-10,passed=bool(ok)))
            write(gp/'comparison.json',pair)
            if len(pair)!=2 or not all(x['passed'] for x in pair):candidate_ok=False
            with e.m.span('common_final_identity'):
                actual=e.bp.values.cpu().numpy();identity=ah(actual)==hashes['values'] and ah(e.bp.active.cpu().numpy())==hashes['active'] and digest(common_levels/'hierarchy.npz')==levelhash
                identity=identity and torch.equal(pre[0].d,pre[1].d) and torch.equal(pre[0].active,pre[1].active)
                after_mg=hierarchy_coefficients(base);identity=identity and all(ah(after_mg[k])==v for k,v in mg_hashes.items())
            gr=dict(group=group,case=s['case'],start=s['start'],outer=s['outer'],order=order,Fu_id=fid,common_identity=bool(identity),common_AMG_build_seconds=build_seconds,
                hierarchy=dict(path=str(common_levels/'hierarchy.npz'),sha256=levelhash),preconditioner=dict(path=str(gp/'actual-preconditioner.npz'),sha256=prehash),
                workspace_isolation=True,common_input_sha256=digest(gp/'common-Fu-rhs.npz'),rows=group_rows,paired=pair,seconds=time.perf_counter()-gstart)
            write(gp/'group.json',gr);groups.append(gr)
            if not identity:raise RuntimeError('Common operator/preconditioner changed')
        zero={case:sum(r['solve_seconds'] for r in rows if r['strategy']=='zero' and r['case']==case) for case in ['A','D']}
        reuse={case:sum(r['solve_seconds'] for r in rows if r['strategy']=='reuse' and r['case']==case) for case in ['A','D']}
        choose=candidate_ok and sum(reuse.values())<=.95*sum(zero.values()) and all(reuse[k]<=1.1*zero[k] for k in zero)
        selection=dict(candidate_correct=candidate_ok,baseline_correct=True,common_correct=True,selected_reuse=bool(choose),zero_seconds=zero,reuse_seconds=reuse,
            saving=1-sum(reuse.values())/sum(zero.values()),rule='>=5% aggregate saving; A,D each <=110%; otherwise original zero initial tangent.',common_build_seconds=sum(g['common_AMG_build_seconds'] for g in groups))
    except BaseException as exc:common_ok=False;reason=dict(error=repr(exc),traceback=traceback.format_exc());selection=None
    finally:
        if e:e.close()
    out=dict(status='completed' if common_ok else 'stopped',reason=reason,requests=requests,full_BP=0,time_steps=0,groups=groups,rows=rows,selection=selection,wall_seconds=time.perf_counter()-start,common_work=e.m.record() if e else {})
    write(root/'result.json',out)
    if not common_ok:raise RuntimeError(reason)

if __name__=='__main__':run(sys.argv[1])
