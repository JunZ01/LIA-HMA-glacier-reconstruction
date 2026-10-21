"""One isolated GPU process per CN year; all IPC and initialization are charged."""
import sys,os,json,time,traceback
from pathlib import Path
from multiprocessing.connection import Client
from .device import Engine,fh,tensor,torch,np,T,CG,ProjectedMultigrid
from .policy import VERSION

def synthetic(e):
    from types import SimpleNamespace
    n=24;i=torch.arange(n,device='cuda',dtype=T);M=torch.eye(n,device='cuda',dtype=T)*4
    M+=.05*torch.cos(i[:,None]+2*i[None,:]);M=M.T@M
    active=torch.ones(n,device='cuda',dtype=T);active[0]=0
    class B:
        def hvp(self,v):return (M@(v*active))*active+v*(1-active)
    b=B();base=SimpleNamespace(d=torch.ones(n,device='cuda',dtype=T)/4,cycle=lambda v:v)
    pre=ProjectedMultigrid(base,SimpleNamespace(active=active));cg=CG(b,pre,True)
    rhs=torch.sin(i)*active
    v,k,r=cg.solve_checked(rhs,1e-10,1e-8,200,e.m,'synthetic')
    M.add_(torch.eye(n,device='cuda',dtype=T)*.2);active[1]=0;rhs*=active
    v2,k2,r2=cg.solve_checked(rhs,1e-10,1e-8,200,e.m,'synthetic')
    cg.graph.reset()
    return dict(first_true=r,reused_true=r2,iterations=[k,k2],projected=bool((v2[active==0]==0).all()),pass_=max(r,r2)<=1e-8)

def run(config):
    cfg=json.loads(Path(config).read_text(encoding='utf-8'));root=Path(cfg['root']);conn=Client(tuple(cfg['address']),authkey=bytes.fromhex(cfg['auth']))
    e=None
    try:
        with np.load(cfg['input'],allow_pickle=False) as f:z={k:f[k] for k in f.files}
        e=Engine(z,root,cfg['prepare_only']);conn.send(dict(ok=True,version=VERSION,device=str(torch.cuda.get_device_name()),torch=str(torch.__version__),cuda=str(torch.version.cuda),meter=e.m.record()))
        while True:
            req=conn.recv();kind=req.pop('kind')
            if kind=='close':break
            try:
                if kind=='begin':e.begin(req['deadline']);out={}
                elif kind=='bind':
                    r,en=e.bind(req['H'],req['u'],req.get('linear',False))
                    with e.m.span('D2H'):out=dict(F=r.cpu().numpy(),force=e.bp.f.cpu().numpy(),wet=e.bp.wet.cpu().numpy(),point=e.state_id,energy=float(en))
                elif kind=='full':out=e.full(req['H'],req['u']);out['point']=e.state_id
                elif kind=='synthetic':out=synthetic(e)
                else:
                    assert req['point']==e.state_id,'Stale linearization identifier'
                    if kind=='operator':
                        with e.m.span('Fu_action'):fu=e.bp.hvp(tensor(req['du']))
                        with e.m.span('FH_action'):hv=fh(e.bp,e.bound_u,tensor(req['dH']))
                        with e.m.span('flux'):q=e.flux.flux(e.bound_h,e.bound_u)
                        with e.m.span('Ru_action'):ru,ve,vo=e.flux.feedback(e.bound_h,e.bound_u,tensor(req['du']),tuple(tensor(x) for x in req['weights']))
                        with e.m.span('D2H'):out=dict(Fu=fu.cpu().numpy(),FH=hv.cpu().numpy(),flux=q.cpu().numpy(),Ru=ru.cpu().numpy(),velocity_internal=ve.cpu().numpy(),velocity_boundary=vo.cpu().numpy())
                    elif kind=='fh':
                        with e.m.span('FH_action'):v=fh(e.bp,e.bound_u,tensor(req['dH']))
                        with e.m.span('D2H'):out=dict(value=v.cpu().numpy())
                    elif kind=='solve':
                        with e.m.span('H2D'):rhs=tensor(req['rhs'])
                        v=e.tangent(rhs)
                        with e.m.span('D2H'):out=dict(value=v.cpu().numpy())
                    elif kind=='feedback':
                        with e.m.span('H2D'):direction=tensor(req['dH']);weights=tuple(tensor(x) for x in req['weights'])
                        with e.m.span('FH_action'):rhs=-fh(e.bp,e.bound_u,direction)
                        v=e.tangent(rhs)
                        with e.m.span('Ru_action'):ru,_,_=e.flux.feedback(e.bound_h,e.bound_u,v,weights)
                        with e.m.span('D2H'):out=dict(du=v.cpu().numpy(),Ru=ru.cpu().numpy())
                    elif kind=='ru':
                        with e.m.span('H2D'):du=tensor(req['du']);weights=tuple(tensor(x) for x in req['weights'])
                        with e.m.span('Ru_action'):ru,_,_=e.flux.feedback(e.bound_h,e.bound_u,du,weights)
                        with e.m.span('D2H'):out=dict(value=ru.cpu().numpy())
                    else:raise ValueError(kind)
                torch.cuda.synchronize();conn.send(dict(ok=True,result=out,meter=e.m.record(),peak_memory=torch.cuda.max_memory_allocated()))
            except BaseException as exc:
                if e.bound_h is not None:
                    with (root/f'failure-{time.time_ns()}.npz').open('xb') as f:np.savez(f,H=e.bound_h.cpu().numpy(),u=e.bound_u.cpu().numpy())
                conn.send(dict(ok=False,error=repr(exc),traceback=traceback.format_exc(),meter=e.m.record()))
    except BaseException as exc:
        try:conn.send(dict(ok=False,error=repr(exc),traceback=traceback.format_exc()))
        except Exception:pass
    finally:
        if e:e.close()
        conn.close()

if __name__=='__main__':run(sys.argv[1])
