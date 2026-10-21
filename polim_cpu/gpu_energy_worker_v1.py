"""Isolated eager-CUDA worker. No CPU scientific solver imports."""
import sys,os,json,time,hashlib,traceback,uuid
from pathlib import Path
sys.modules['triton']=None
from .v3.gpu import initialize
torch=initialize()
import numpy as np
from .gpu_bp_v1 import BP,Multigrid,Transport,tensor,solve
from .gpu_neural_v1 import Network,Energy

def write(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2,allow_nan=False)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def checkpoint(root,name,h,u,year,info,net=None):
    start=time.perf_counter();torch.cuda.synchronize();target=root/name
    if target.exists():raise RuntimeError('Immutable checkpoint exists')
    p=root/f'.pending-{uuid.uuid4().hex}';p.mkdir()
    data=dict(h=h.cpu().numpy(),u=u.cpu().numpy())
    if net is not None:data['theta']=net.theta.detach().cpu().numpy()
    with (p/'state.npz').open('xb') as f:np.savez(f,**data);f.flush();os.fsync(f.fileno())
    write(p/'metadata.json',dict(year=year,info=info,sha256=digest(p/'state.npz'),contract=digest(root.parent/'input.npz')))
    write(p/'seal.json',dict(sha256=digest(p/'metadata.json')))
    p.rename(target)
    return time.perf_counter()-start

def preflight(root,z,bp,tr,net):
    start=time.perf_counter();h=tensor(z['pending_h']);u=tensor(z['u']);bp.prepare(h);r,e=bp.evaluate(u,True)
    def rel(a,b):return float((a-tensor(b)).norm()/tensor(b).norm().clamp_min(1.))
    out=dict(force=rel(bp.f,z['fixture_f']),residual=rel(r,z['fixture_r']),energy=abs(float(e)-float(z['fixture_e']))/max(abs(float(z['fixture_e'])),1.),diagonal=rel(bp.diagonal(),z['fixture_diag']),hvp=rel(bp.hvp(tensor(z['direction'])),z['fixture_hvp']))
    assert max(out.values())<1e-9,out
    ht,now,export,steps=tr.window(tensor(z['h']),u,99.,99+30/365)
    out.update(transport_H_max=float((ht-h).abs().max()),transport_steps=steps)
    assert out['transport_H_max']<1e-10
    c=net.context(bp,tensor(z['h']),u,30/365);v=net.forward(c,bp)
    out['network_velocity']=rel(v,z['fixture_neural_u'])
    loss=Energy.apply(v,bp,c['energy'],c['loss_scale']);loss.backward()
    out['network_gradient']=rel(net.theta.grad,z['fixture_neural_gradient']);net.theta.grad=None
    assert out['network_velocity']<1e-9 and out['network_gradient']<1e-7,out
    tmp=root/'preflight';tmp.mkdir();checkpoint(tmp,'restore',h,u,99+30/365,{},net)
    restored=load(tmp/'restore/state.npz')
    assert torch.equal(tensor(restored['h']),h) and torch.equal(tensor(restored['u']),u) and torch.equal(tensor(restored['theta']),net.theta)
    out.update(status='pass',checkpoint_restore_exact=True,seconds=time.perf_counter()-start,peak_memory_bytes=torch.cuda.max_memory_allocated())
    assert out['peak_memory_bytes']<.08*torch.cuda.get_device_properties(0).total_memory
    write(root/'preflight.json',out);print(json.dumps(out),flush=True)

def case(root,z,bp,tr,mg,kind):
    folder=root/kind;folder.mkdir();net=Network(z) if kind=='neural' else None
    h=tensor(z['h']);u=tensor(z['u']);now=99.;exported=0.;volume0=float(h@tr.dual);rows=[];io=0.
    start=time.perf_counter();io+=checkpoint(folder,'frame-000',h,u,now,{},net)
    try:
        for i in range(1,31):
            if now>=100-1e-12:break
            h0=h.clone();oldtime=now;end=min(100.,now+30/365)
            t=time.perf_counter();h,now,out,nsteps=tr.window(h,u,now,end);torch.cuda.synchronize();transport=time.perf_counter()-t
            exported+=out;io+=checkpoint(folder,f'pending-{i:04d}',h,u,now,dict(exported=exported),net)
            t=time.perf_counter();bp.prepare(h);prediction={};guess=u
            if net:guess,prediction=net.predict(bp,h0,u,now-oldtime)
            torch.cuda.synchronize();prepare=time.perf_counter()-t
            def cb(it,v,info):
                nonlocal io
                io+=checkpoint(folder,f'newton-{i:03d}-{it:03d}',h,v,now,info,net)
            u,info=solve(bp,guess,mg,cb)
            mass=abs(float(h@tr.dual)-volume0+exported)/volume0
            assert mass<=1e-10 and bool(torch.isfinite(u).all())
            row=dict(step=i,year=now,transport_seconds=transport,substeps=nsteps,prepare_neural_seconds=prepare,prediction=prediction,momentum=info,mass_relative=mass)
            rows.append(row);io+=checkpoint(folder,f'frame-{i:03d}',h,u,now,dict(rows=rows,exported=exported),net)
            print(json.dumps(dict(case=kind,step=i,year=now,Newton=info['iterations'],seconds=time.perf_counter()-start)),flush=True)
            if time.perf_counter()-start>900:raise RuntimeError('Case900second budget')
        assert now>=100-1e-12
        result=dict(status='completed',wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,rows=rows,year=now)
    except Exception as exc:result=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.perf_counter()-start,checkpoint_seconds=io,rows=rows,year=now)
    result['compute_seconds']=result['wall_seconds']-io
    write(folder/'result.json',result);return result

def main():
    root=Path(sys.argv[1]);torch.cuda.set_per_process_memory_fraction(.10)
    write(root/'gpu-launch.json',dict(pid=os.getpid(),device=torch.cuda.get_device_name(0),torch=torch.__version__,start=time.time(),command=sys.argv,checkpoint='pending and accepted steps; every5Newton',estimate_minutes=[1,10],stop='1e-6 BP,20Newton,200CG,120seconds/BP,900seconds/case'))
    try:
        t=time.perf_counter();z=load(root/'input.npz');bp=BP(z);tr=Transport(z);mg=Multigrid(z);net=Network(z);torch.cuda.synchronize()
        setup=time.perf_counter()-t;preflight(root,z,bp,tr,net);del net
        results={k:case(root,z,bp,tr,mg,k) for k in ['direct','neural']}
        write(root/'gpu-result.json',dict(status='completed' if all(v['status']=='completed' for v in results.values()) else 'failed',setup_seconds=setup,cases=results,peak_memory_bytes=torch.cuda.max_memory_allocated()))
    except Exception as exc:
        write(root/'gpu-result.json',dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
