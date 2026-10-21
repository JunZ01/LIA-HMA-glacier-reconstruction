"""Exact CUDA tangent caching and inexact Newton; optional PCG CUDA Graph."""
import time
import torch
from .gpu_bp_v1 import BP as Base,Multigrid as BaseMG,tensor,scatter,mv,T

class BP(Base):
    def __init__(self,z):
        super().__init__(z)
        self.mapping=tensor(z['csr_mapping'],True);self.values=torch.zeros(len(z['csr_indices']),device='cuda',dtype=T)
        self.matrix=torch.sparse_csr_tensor(tensor(z['csr_indptr'],True),tensor(z['csr_indices'],True),self.values,size=(self.ndof,self.ndof))
        self.active=torch.ones(self.ndof,device='cuda',dtype=T)
    def evaluate(self,u,linearize=False):
        r,e=super().evaluate(u,linearize)
        if linearize:
            B=self.B;CB=torch.stack([4*B[:,:,0]+2*B[:,:,1],2*B[:,:,0]+4*B[:,:,1],B[:,:,2],B[:,:,3],B[:,:,4]],2)
            factor=self.weight*self.lin_eta
            left=B.reshape(-1,60,12).transpose(1,2)
            right=(factor[:,:,None,None]*CB).reshape(-1,60,12)
            local=torch.bmm(left,right)-torch.bmm(self.lin_bu.transpose(1,2),(factor/(6*self.lin_e2))[:,:,None]*self.lin_bu)
            self.values.zero_();self.values.index_add_(0,self.mapping,local.ravel());self.active.copy_(self.wet)
        return r,e
    def hvp(self,v):return mv(self.matrix,v*self.active)*self.active+v*(1-self.active)

class Multigrid(BaseMG):
    def __init__(self,z):
        super().__init__(z);self.coarse_inverse=torch.cholesky_inverse(self.chol)
    def cycle(self,b,i=0):
        if i==len(self.a)-1:return self.coarse_inverse@b
        x=torch.zeros_like(b)
        for _ in range(2):x=x+self.omega[i]*self.inv[i]*(b-mv(self.a[i],x))
        x=x+mv(self.p[i],self.cycle(mv(self.r[i],b-mv(self.a[i],x)),i+1))
        for _ in range(2):x=x+self.omega[i]*self.inv[i]*(b-mv(self.a[i],x))
        return x

class PCG:
    def __init__(self,bp,mg,use_graph=False):
        self.bp=bp;self.mg=mg;self.graph=None;self.use_graph=use_graph
        self.y=torch.zeros_like(mg.d);self.rr=self.y.clone();self.p=self.y.clone();self.rz=torch.ones((),device='cuda',dtype=T);self.bad=torch.zeros((),device='cuda',dtype=T)
        self.capture_seconds=0.
    def reset(self,r):
        self.y.zero_();self.rr.copy_(-self.mg.d*r);z=self.mg.cycle(self.rr);self.p.copy_(z);self.rz.copy_((self.rr*z).sum());self.bad.zero_()
    def step(self):
        ap=self.mg.d*self.bp.hvp(self.mg.d*self.p);den=(self.p*ap).sum()
        self.bad.copy_(torch.maximum(self.bad,(den<=0).to(T)))
        alpha=self.rz/den.clamp_min(1e-300);self.y.add_(alpha*self.p);self.rr.sub_(alpha*ap)
        z=self.mg.cycle(self.rr);rz=(self.rr*z).sum();self.p.copy_(z+rz/self.rz.clamp_min(1e-300)*self.p);self.rz.copy_(rz)
    def capture(self,r):
        start=time.perf_counter();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.reset(r)
            for _ in range(4):self.step()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();self.reset(r)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for _ in range(4):self.step()
        torch.cuda.synchronize();self.capture_seconds+=time.perf_counter()-start
    def solve(self,r):
        if self.use_graph and self.graph is None:self.capture(r)
        self.reset(r);rn=r.norm();batch=4 if self.use_graph else 1
        for k in range(0,200,batch):
            if self.graph:self.graph.replay()
            else:self.step()
            ratio=float((self.rr/self.mg.d).norm()/rn)
            if ratio<=.03:break
        else:raise RuntimeError('200CG budget')
        if bool(self.bad>0):raise RuntimeError('Nonpositive CG curvature')
        delta=self.mg.d*self.y;true=float((self.bp.hvp(delta)+r).norm()/rn)
        if not true<=.031:raise RuntimeError('True linear residual failed')
        return delta,k+batch,true

def solve(bp,u,mg,checkpoint=None):
    start=time.perf_counter();u=u.clone();u[~bp.wet]=0.;history=[]
    for it in range(21):
        r,e=bp.evaluate(u,True);res=float(r.norm()/bp.force_norm);history.append(dict(iteration=it,residual=res))
        if res<=1e-6:return u,dict(converged=True,iterations=it,relative_residual=res,history=history,seconds=time.perf_counter()-start)
        if it==20 or time.perf_counter()-start>120:raise RuntimeError('Newton budget')
        delta,k,true=bp.pcg.solve(r);rn=r.norm();descent=float(r@delta);alpha=1.;energy=float(e)
        if descent>=0:raise RuntimeError('Non-descent Newton direction')
        for _ in range(35):
            trial=u+alpha*delta;rt,et=bp.evaluate(trial)
            resolved=abs(1e-4*alpha*descent)>64*torch.finfo(T).eps*max(abs(energy),1.)
            decreased=float(rt.norm())<(1-1e-4*alpha)*float(rn)
            if (resolved and float(et)<=energy+1e-4*alpha*descent) or (not resolved and decreased) or float(rt.norm())<.5*float(rn):u=trial;break
            alpha*=.5
        else:raise RuntimeError('Line search failed')
        history[-1].update(cg_iterations=k,linear_residual=true,step_length=alpha)
        if checkpoint and (it+1)%5==0:checkpoint(it+1,u,history[-1])
