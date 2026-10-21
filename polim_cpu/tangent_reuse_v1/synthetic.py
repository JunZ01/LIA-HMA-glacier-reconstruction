"""Synthetic-only CUDA fixture; no glacier state, BP equilibrium or tangent."""
import sys,json,time
from pathlib import Path
import torch
from types import SimpleNamespace
from .space import Space,solve_initial
from polim_cpu.cn_cuda_v1.device import CG,Meter,ProjectedMultigrid,T

def run(root):
    root=Path(root);start=time.perf_counter();m=Meter();n=24;i=torch.arange(n,device='cuda',dtype=T)
    q=torch.cos(i[:,None]*.37+i[None,:]*.13);A=q.T@q+torch.eye(n,device='cuda',dtype=T)*4
    wet=torch.ones(n,device='cuda',dtype=torch.bool);wet[0]=False
    class Op:
        def hvp(self,v):return (A@(v*wet))*wet+v*(~wet)
    op=Op();base=SimpleNamespace(d=torch.ones(n,device='cuda',dtype=T)/4,cycle=lambda x:x)
    cg=CG(op,ProjectedMultigrid(base,SimpleNamespace(active=wet.to(T))),True)
    space=Space();space.bind(('A','wet0'));rows=[];raw={}
    for j,b in enumerate([torch.sin(i)*wet,2*torch.sin(i)*wet,torch.cos(i)*wet,torch.zeros_like(i),1e-14*torch.cos(i)*wet]):
        x0,info=space.guess(b,wet,op.hvp);x,k,r=solve_initial(cg,b,x0,1e-10,1e-8,200,m,'synthetic')
        ax=op.hvp(x);den=float(b.norm());true=float((ax-b).norm())/(den or 1.)
        added=space.add(x,ax,wet);rows.append(dict(test='rhs-'+str(j),initial=info,iterations=k,true_relative=true,denominator=den,passed=true<=1e-8 and bool((x[~wet]==0).all()),added=added))
        raw['b'+str(j)]=b.cpu().numpy();raw['x'+str(j)]=x.cpu().numpy()
    rows.append(dict(test='span-reuse-zero-CG',passed=rows[1]['iterations']==0))
    space.Z.append(space.Z[0].clone());space.B.append(space.B[0].clone())
    x,info=space.guess(torch.sin(i)*wet,wet,op.hvp);rows.append(dict(test='rank-deficient-space',passed=info['reason']=='SVD' and info['rank']<len(space.B),info=info))
    space.Z[0].fill_(float('nan'));x,info=space.guess(torch.sin(i)*wet,wet,op.hvp)
    rows.append(dict(test='nonfinite-fallback',passed=bool((x==0).all()) and info['reason'].startswith('invalid'),info=info))
    changed=space.bind(('new-A','wet0'));rows.append(dict(test='Fu-change-clear',passed=changed and len(space.Z)==0))
    space.add(torch.ones_like(i)*wet,op.hvp(torch.ones_like(i)*wet),wet);changed=space.bind(('new-A','wet1'))
    rows.append(dict(test='active-change-clear',passed=changed and len(space.Z)==0))
    b=torch.sin(i)*wet;exact=torch.linalg.solve(A[wet][:,wet],b[wet]);x0=torch.zeros_like(i);x0[wet]=exact*(1-1e-6)
    x,k,r=solve_initial(cg,b,x0,1e-10,1e-8,200,m,'synthetic');true=float((op.hvp(x)-b).norm()/b.norm())
    rows.append(dict(test='nonzero-initial-original-b-scale',passed=true<=1e-8,true_relative=true,iterations=k))
    cg.graph.reset();torch.cuda.synchronize()
    import numpy as np
    with (root/'synthetic-arrays.npz').open('xb') as f:np.savez(f,**raw)
    result=dict(status='pass' if all(x['passed'] for x in rows) else 'fail',rows=rows,seconds=time.perf_counter()-start,work=m.record(),real_BP=0,real_glacier_tangent=0,time_steps=0,
        environment=dict(torch=torch.__version__,CUDA=torch.version.cuda,device=torch.cuda.get_device_name()))
    with (root/'synthetic.json').open('x') as f:json.dump(result,f,indent=2)
    assert result['status']=='pass'

if __name__=='__main__':run(sys.argv[1])
