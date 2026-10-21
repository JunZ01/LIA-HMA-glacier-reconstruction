import sys,json,time,traceback
from pathlib import Path
from multiprocessing.connection import Client
from .device import Engine,torch,tensor,np,fh

def run(path):
    cfg=json.loads(Path(path).read_text());root=Path(cfg['root']);conn=Client(tuple(cfg['address']),authkey=bytes.fromhex(cfg['auth']));e=None
    try:
        with np.load(cfg['input'],allow_pickle=False) as z:data={k:z[k] for k in z.files}
        e=Engine(data,root,cfg['prepare_only'])
        conn.send(dict(ok=True,meter=e.m.record(),device=torch.cuda.get_device_name(),torch=str(torch.__version__),cuda=str(torch.version.cuda)))
        while True:
            req=conn.recv();kind=req.pop('kind')
            if kind=='close':break
            try:
                if kind=='begin':e.begin(req['deadline']);out={}
                elif kind=='pure':out=e.pure(req['H'],req['u'])
                elif kind=='blocks':out=e.blocks(req['H'],req['u'],req.get('prepare_amg',True))
                elif kind=='synthetic-error':raise ValueError('intentional no-model serialization/exception check')
                else:
                    if req['point']!=e.state_id:raise RuntimeError('Stale v52 linearization')
                    if kind=='V':out=e.V(req['rhs'])
                    elif kind=='operator':
                        with e.m.span('original_Fu_action'):fu=e.bp.hvp(tensor(req['du']))
                        with e.m.span('original_FH_action'):bh=fh(e.bp,e.bound_u,tensor(req['dH']))
                        with e.m.span('original_Ru_action'):cu,qi,oi=e.flux.feedback(e.bound_h,e.bound_u,tensor(req['du']),tuple(tensor(x) for x in req['weights']))
                        with e.m.span('original_flux'):q=e.flux.flux(e.bound_h,e.bound_u)
                        with e.m.span('D2H'):out=dict(Fu=fu.cpu().numpy(),FH=bh.cpu().numpy(),Ru=cu.cpu().numpy(),flux=q.cpu().numpy())
                    elif kind=='identity':
                        out=dict(point=e.state_id,H=e.bound_h.cpu().numpy(),u=e.bound_u.cpu().numpy(),wet=e.bp.wet.cpu().numpy(),A_values=e.bp.values.cpu().numpy().copy())
                    else:raise ValueError('Unsupported v52 operation '+kind)
                torch.cuda.synchronize();conn.send(dict(ok=True,result=out,meter=e.m.record()))
            except BaseException as exc:
                err=dict(error=repr(exc),traceback=traceback.format_exc(),kind=kind,meter=e.m.record())
                stamp=str(time.time_ns())
                with (root/('failure-'+stamp+'.json')).open('x') as f:json.dump(err,f)
                if e.bound_h is not None:
                    with (root/('failure-'+stamp+'.npz')).open('xb') as f:np.savez(f,H=e.bound_h.cpu().numpy(),u=e.bound_u.cpu().numpy())
                conn.send(dict(ok=False,**err))
    except BaseException as exc:
        err=dict(error=repr(exc),traceback=traceback.format_exc())
        with (root/'fatal.json').open('x') as f:json.dump(err,f)
        try:conn.send(dict(ok=False,**err))
        except Exception:pass
    finally:
        if e:e.close()
        conn.close()

if __name__=='__main__':run(sys.argv[1])
