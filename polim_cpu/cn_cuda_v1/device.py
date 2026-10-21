"""CUDA true BP derivatives; no approximation to Fu or FH."""
from polim_cpu.gpu_energy_worker_v1 import torch,tensor,np,time,Path
from polim_cpu.gpu_allocator_safe_v1 import BP as NativeBP
from polim_cpu.gpu_bp_v1 import scatter,T,Transport
from polim_cpu.gpu_cached_v1 import PCG as NativePCG
from polim_cpu.gpu_allocator_recovery_worker_v1 import Hierarchy
from polim_cpu.projected_amg_v1 import ProjectedMultigrid
from .policy import SOLVER,VERSION
from contextlib import contextmanager
from collections import defaultdict
import hashlib,os

class Meter:
    def __init__(self):self.counts=defaultdict(int);self.times=defaultdict(float);self.deadline=None;self.events=[]
    def tick(self):
        if self.deadline and time.time()>=self.deadline:raise RuntimeError('Registered GPU step/shared wall budget')
    @contextmanager
    def span(self,key):
        self.tick();self.counts[key]+=1;torch.cuda.synchronize();t=time.perf_counter()
        try:yield
        finally:torch.cuda.synchronize();self.times[key]+=time.perf_counter()-t
    def record(self):return dict(counts=dict(self.counts),seconds=dict(self.times),events=self.events)

class CachedBP(NativeBP):
    def __init__(self,z,m):
        super().__init__(z);self.m=m;self.h_copy=None;self.u_copy=None;self.result=None;self.matrix_u=None
        config=hashlib.sha256(VERSION.encode())
        for k in sorted(z):config.update(k.encode());config.update(np.ascontiguousarray(z[k]).tobytes())
        self.configuration=config.hexdigest();self.fh_stress=None
        positions=[]
        for i in range(self.ndof):
            a,b=z['csr_indptr'][i:i+2];positions.append(int(a+np.flatnonzero(z['csr_indices'][a:b]==i)[0]))
        self.diagonal_positions=tensor(positions,True)
    def prepare(self,h):
        if self.h_copy is not None and torch.equal(h.view(torch.int64),self.h_copy.view(torch.int64)):self.m.counts['geometry_cache_hit']+=1;return
        with self.m.span('geometry_build'):super().prepare(h)
        self.h_copy=h.clone();self.u_copy=None;self.result=None;self.matrix_u=None;self.fh_stress=None
    def evaluate(self,u,linearize=False):
        same=self.u_copy is not None and torch.equal(u.view(torch.int64),self.u_copy.view(torch.int64))
        matrix_same=self.matrix_u is not None and torch.equal(u.view(torch.int64),self.matrix_u.view(torch.int64))
        if same and (not linearize or matrix_same):self.m.counts['residual_cache_hit']+=1;return self.result
        with self.m.span('residual_and_matrix'):
            if linearize:self.m.counts['matrix_build']+=1
            result=super().evaluate(u,linearize)
        self.u_copy=u.clone();self.result=result
        if linearize:self.matrix_u=u.clone();self.fh_stress=None
        return result

def fh(bp,u,dh):
    """Differentiate strain mapping, integration weights, viscosity and drive."""
    if bool((dh[~bp.wet[::4]]!=0).any()):raise ValueError('FH direction crosses current dry momentum branch')
    physical=torch.where(bp.wet[::4],bp.h,torch.zeros_like(bp.h));H=physical[bp.tri];dH=dh[bp.tri]
    local=u[bp.ids];out=torch.zeros((len(H),12),device='cuda',dtype=T)
    if bp.fh_stress is None:
        s=torch.einsum('eqsi,ei->eqs',bp.B,local)
        bp.fh_stress=torch.stack([4*s[:,:,0]+2*s[:,:,1],2*s[:,:,0]+4*s[:,:,1],s[:,:,2],s[:,:,3],s[:,:,4]],2)
    else:bp.m.counts['FH_intermediate_cache_hit']+=1
    for start in range(0,len(H),2048):
        stop=min(start+2048,len(H));sl=slice(start,stop);hh=H[sl];dd=dH[sl];grad=bp.grad[sl]
        height=hh@bp.lam.T;dheight=dd@bp.lam.T;ratio=torch.where(height>0,dheight/height.clamp_min(1e-300),torch.zeros_like(height))
        dz=.5*height;safe=torch.where(dz>0,dz,torch.ones_like(dz));gz=bp.sign[None,:,:]/safe[:,:,None];dgz=-gz*ratio[:,:,None]
        zx=torch.einsum('eqj,eja->eqa',bp.bed[bp.tri[sl]][:,None,:]+bp.sigma[None,:,None]*hh[:,None,:],grad)
        dzx=torch.einsum('eqj,eja->eqa',bp.sigma[None,:,None]*dd[:,None,:],grad)
        dx=-dgz*zx[:,:,0,None]-gz*dzx[:,:,0,None];dy=-dgz*zx[:,:,1,None]-gz*dzx[:,:,1,None]
        B=bp.B[sl];dB=torch.zeros_like(B);dB[:,:,0,0::2]=dx;dB[:,:,1,1::2]=dy;dB[:,:,2,0::2]=dy;dB[:,:,2,1::2]=dx;dB[:,:,3,0::2]=dgz;dB[:,:,4,1::2]=dgz
        ds=torch.einsum('eqsi,ei->eqs',dB,local[sl])
        def stress(x):return torch.stack([4*x[:,:,0]+2*x[:,:,1],2*x[:,:,0]+4*x[:,:,1],x[:,:,2],x[:,:,3],x[:,:,4]],2)
        ss=bp.fh_stress[sl];dss=stress(ds);bu=bp.lin_bu[sl]
        dbu=torch.einsum('eqsi,eqs->eqi',dB,ss)+torch.einsum('eqsi,eqs->eqi',B,dss)
        e2=bp.lin_e2[sl];de2=.5*(ss*ds).sum(2);eta=bp.lin_eta[sl];deta=-eta*de2/(3*e2)
        internal=(bp.weight[sl,:,None]*((ratio*eta+deta)[:,:,None]*bu+eta[:,:,None]*dbu)).sum(1)
        dgs=torch.einsum('ej,eja->ea',dd,grad)
        drive=900*9.80665*(dgs[:,None,None,:]+bp.gs[sl,None,None,:]*ratio[:,:,None,None])*bp.phi[None,:,:,None]*bp.weight[sl,:,None,None]
        out[sl]=internal+drive.reshape(stop-start,12,12).sum(1)
    result=scatter(out,bp.ids,bp.ndof);result[~bp.wet]=0.;return result

class Flux(Transport):
    def rates(self,u):
        v=u.reshape(-1,2,2);mean=.5*v[:,0,:]+.25*v[:,1,:]
        vv=5/12*(mean[self.a]+mean[self.b])+1/6*mean[self.third]
        c=scatter((vv*self.normal).sum(1),self.inverse,len(self.edges))
        bo=torch.cat([((.75*mean[self.outer_a]+.25*mean[self.outer_b])*self.outer_normal).sum(1),((.75*mean[self.outer_b]+.25*mean[self.outer_a])*self.outer_normal).sum(1)])
        return c,bo
    def feedback(self,h,u,du,weights):
        c,bo=self.rates(u);dc,dbo=self.rates(du);donor=torch.where(c>=0,self.edges[:,0],self.edges[:,1]);bd=self.links[len(self.edges):,0]
        q=h[donor]*dc;o=h[bd]*dbo*(bo>0);wa,wb,wo=weights
        r=scatter(wa*q,self.edges[:,0],self.n)-scatter(wb*q,self.edges[:,1],self.n)+scatter(wo*o,bd,self.n)
        return r,q,o

class CG(NativePCG):
    def solve_checked(self,rhs,rtol,true_tol,limit,meter,label):
        # Native recurrence solves -r; r=-rhs yields the requested positive rhs.
        r=-rhs;used=0
        if self.use_graph and self.graph is None:
            with meter.span('graph_capture'):self.capture(r)
            used+=8;meter.counts[label+'_Krylov_setup']+=8
        self.reset(r);den=(self.mg.d*rhs).norm();norm=rhs.norm()
        if float(norm)==0:return torch.zeros_like(rhs),used,0.
        while used+4<=limit:
            meter.tick();self.graph.replay();used+=4;meter.counts[label+'_Krylov']+=4
            rel=float(self.rr.norm()/den)
            if bool(self.bad>0):raise RuntimeError('Nonpositive CG curvature')
            if rel<=rtol:
                du=self.mg.d*self.y;true=float((self.bp.hvp(du)-rhs).norm()/norm)
                if true<=true_tol:return du,used,true
        du=self.mg.d*self.y;true=float((self.bp.hvp(du)-rhs).norm()/norm)
        raise RuntimeError(f'{label} CG budget exhausted {used}/{limit}, true={true}')

class Engine:
    def __init__(self,z,root,prepare_only=False):
        self.m=Meter();self.root=root;self.prepare_only=prepare_only;self.z=z
        torch.cuda.set_per_process_memory_fraction(.10)
        with self.m.span('GPU_initialization'):self.bp=CachedBP(z,self.m);self.flux=Flux(z)
        self.hierarchy=Hierarchy(root/'amg',z);(root/'amg').mkdir();self.mg=None;self.cg=None;self.build_wet=None;self.rebuild_pending=False;self.builds=0
        self.state_id=None;self.bound_h=None;self.bound_u=None;self.step_calls=defaultdict(int)
    def begin(self,deadline):self.m.deadline=deadline;self.step_calls.clear()
    def bind(self,h,u,linear=False):
        with self.m.span('H2D'):hh=tensor(h);uu=tensor(u)
        self.bp.prepare(hh)
        if bool((uu[~self.bp.wet]!=0).any()):raise ValueError('Nonzero dry velocity')
        r,e=self.bp.evaluate(uu,linear);self.bound_h=hh;self.bound_u=uu
        self.state_id=hashlib.sha256(self.bp.configuration.encode()+np.ascontiguousarray(h).tobytes()+np.ascontiguousarray(u).tobytes()+self.bp.wet.cpu().numpy().tobytes()).hexdigest()
        return r,e
    def ensure_preconditioner(self):
        wet=self.bp.wet;changed=0 if self.build_wet is None else int((wet!=self.build_wet).sum())
        need=self.mg is None or self.rebuild_pending or changed>SOLVER['rebuild_wet_fraction']*len(wet)
        if need:
            if self.cg and self.cg.graph:self.cg.graph.reset()
            with self.m.span('AMG_build'):
                self.builds+=1;self.mg=ProjectedMultigrid(self.hierarchy.build(self.bp,self.builds),self.bp);self.cg=CG(self.bp,self.mg,True)
            self.build_wet=wet.clone();self.rebuild_pending=False
            self.m.events.append(dict(event='AMG-rebuild',build=self.builds,wet_changes=changed))
        else:self.m.counts['AMG_reuse']+=1
        diag=torch.where(wet,self.bp.values[self.bp.diagonal_positions],torch.ones_like(self.bp.f))
        if bool((diag<=0).any()):raise RuntimeError('Nonpositive Fu diagonal')
        self.mg.d.copy_(diag.rsqrt())
    def linear_solve(self,rhs,label,rtol,limit,true_tol):
        self.ensure_preconditioner()
        with self.m.span(label+'_Krylov_solve'):v,it,true=self.cg.solve_checked(rhs,rtol,true_tol,limit,self.m,label)
        if it>SOLVER['rebuild_iteration_trigger']:self.rebuild_pending=True
        v[~self.bp.wet]=0.;self.m.events.append(dict(event='linear',kind=label,iterations=it,true_relative=true));return v,it,true
    def tangent(self,rhs):
        if self.prepare_only:raise RuntimeError('Real tangent solve disabled during preparation')
        if self.step_calls['tangent']>=64:raise RuntimeError('64tangent budget')
        self.step_calls['tangent']+=1
        with self.m.span('tangent'):
            if not bool(rhs.any()):return torch.zeros_like(rhs)
            return self.linear_solve(rhs,'tangent',1e-10,200,1e-8)[0]
    def full(self,h,u):
        if self.prepare_only:raise RuntimeError('Real BP disabled during preparation')
        if self.step_calls['full_BP']>=8:raise RuntimeError('8completeBP budget')
        self.step_calls['full_BP']+=1
        with self.m.span('full_BP'):
            hh=tensor(h);v=tensor(u).clone();self.bp.prepare(hh);v[~self.bp.wet]=0.;history=[];self.bound_h=hh;self.bound_u=v
            for it in range(81):
                self.m.tick();self.bound_u=v;r,energy=self.bp.evaluate(v,True);res=float(r.norm()/self.bp.force_norm)
                info=dict(iteration=it,relative_residual=res,energy=float(energy));history.append(info)
                self.m.events.append(dict(event='BP-iteration',solve=self.step_calls['full_BP'],**info))
                if res<=1e-6:break
                if it==80:raise RuntimeError('80BP Newton budget')
                tol=min(.1,max(1e-4,.05*np.sqrt(res)))
                delta,k,true=self.linear_solve(-r,'BP',tol,500,.1);descent=float(r@delta)
                if not bool(torch.isfinite(delta).all()) or descent>=0:raise RuntimeError('Invalid BP direction')
                for ls in range(35):
                    alpha=2.**(-ls);trial=v+alpha*delta;rt,et=self.bp.evaluate(trial)
                    resolved=abs(1e-4*alpha*descent)>64*torch.finfo(T).eps*max(abs(float(energy)),1.)
                    if (resolved and float(et)<=float(energy)+1e-4*alpha*descent) or (not resolved and float(rt.norm())<(1-1e-4*alpha)*float(r.norm())) or float(rt.norm())<.5*float(r.norm()):v=trial;break
                else:raise RuntimeError('35 BP line candidates exhausted')
                self.m.counts['BP_Newton_updates']+=1;info.update(cg_iterations=k,true_linear=true,alpha=alpha)
            self.m.events.append(dict(event='BP-history',history=history));self.bound_h=hh;self.bound_u=v
            with self.m.span('D2H'):result=v.cpu().numpy()
            self.bind(h,result,True) # exact cache hit, establishes immutable point identifier.
            return dict(u=result,info=dict(converged=True,iterations=it,relative_residual=res,history=history))
    def close(self):
        if self.cg and self.cg.graph:self.cg.graph.reset()
        self.hierarchy.close()
