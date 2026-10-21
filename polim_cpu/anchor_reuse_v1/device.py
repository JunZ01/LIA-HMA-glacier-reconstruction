"""Isolated CUDA operators. No complete BP or exact Fu solve is exposed."""
from polim_cpu.cn_cuda_v1.device import Engine as BaseEngine,fh,torch,tensor,np,T,scatter

def stress(s):
    return torch.stack([4*s[:,:,0]+2*s[:,:,1],2*s[:,:,0]+4*s[:,:,1],s[:,:,2],s[:,:,3],s[:,:,4]],2)

def strain(dx,dy,dz):
    b=torch.zeros((len(dx),12,5,12),device='cuda',dtype=T)
    b[:,:,0,0::2]=dx;b[:,:,1,1::2]=dy;b[:,:,2,0::2]=dy;b[:,:,2,1::2]=dx;b[:,:,3,0::2]=dz;b[:,:,4,1::2]=dz
    return b

def pure_residual(bp,h,u):
    """Read immutable mesh/quadrature only; never touch BP caches or projection."""
    threshold=64*torch.finfo(T).eps*torch.maximum(torch.ones_like(h),torch.maximum(bp.bed.abs(),(bp.bed+h).abs()))
    wet=h>threshold;physical=torch.where(wet,h,torch.zeros_like(h))
    r=torch.zeros(bp.ndof,device='cuda',dtype=T);force=torch.zeros_like(r)
    for start in range(0,len(bp.tri),1024):
        sl=slice(start,start+1024);tri=bp.tri[sl];ids=bp.ids[sl];grad=bp.grad[sl];hh=physical[tri];bed=bp.bed[tri]
        height=hh@bp.lam.T;dz=.5*height;safe=torch.where(dz>0,dz,torch.ones_like(dz));weight=bp.area[sl,None]*dz/6
        zx=torch.einsum('eqj,eja->eqa',bed[:,None,:]+bp.sigma[None,:,None]*hh[:,None,:],grad)
        gz=bp.sign[None,:,:]/safe[:,:,None]
        gx=grad[:,:,0].repeat_interleave(2,dim=1)[:,None,:]*bp.vphi[None,:,:]-gz*zx[:,:,0,None]
        gy=grad[:,:,1].repeat_interleave(2,dim=1)[:,None,:]*bp.vphi[None,:,:]-gz*zx[:,:,1,None]
        b=strain(gx,gy,gz);gs=torch.einsum('ej,eja->ea',bed+hh,grad)
        fl=(-900*9.80665*gs[:,None,None,:]*bp.phi[None,:,:,None]*weight[:,:,None,None]).reshape(len(tri),12,12).sum(1)
        aa=bp.A[tri];aq=aa[:,:,bp.layer]*(1-bp.fraction)[None,None,:]+aa[:,:,bp.layer+1]*bp.fraction[None,None,:]
        coeff=torch.einsum('ejq,qj->eq',aq,bp.lam).pow(-1/3)
        s=torch.einsum('eqsi,ei->eqs',b,u[ids]);e2=s[:,:,0]**2+s[:,:,1]**2+s[:,:,0]*s[:,:,1]+.25*(s[:,:,2:]**2).sum(2)+1e-20
        bu=torch.einsum('eqsi,eqs->eqi',b,stress(s));eta=.5*coeff*e2.pow(-1/3)
        local=(weight[:,:,None]*eta[:,:,None]*bu).sum(1)-fl
        r.index_add_(0,ids.ravel(),local.ravel());force.index_add_(0,ids.ravel(),fl.ravel())
    fullwet=wet.repeat_interleave(4)
    return torch.where(fullwet,r,u),torch.where(fullwet,force,torch.zeros_like(force)),fullwet

def local_geometry_blocks(bp,u):
    """Three local triangle basis templates, NOT global-column FH probing."""
    wet=bp.wet[::4];physical=torch.where(wet,bp.h,torch.zeros_like(bp.h));local=u[bp.ids]
    out=torch.zeros((len(bp.tri),12,3),device='cuda',dtype=T);material=[]
    for start in range(0,len(bp.tri),1024):
        sl=slice(start,start+1024);tri=bp.tri[sl];hh=physical[tri];grad=bp.grad[sl];height=hh@bp.lam.T
        dz=.5*height;safe=torch.where(dz>0,dz,torch.ones_like(dz));gz=bp.sign[None,:,:]/safe[:,:,None]
        zx=torch.einsum('eqj,eja->eqa',bp.bed[tri][:,None,:]+bp.sigma[None,:,None]*hh[:,None,:],grad)
        b=bp.B[sl];raw_strain=torch.einsum('eqsi,ei->eqs',b,local[sl]);ss=stress(raw_strain);bu=bp.lin_bu[sl];eta=bp.lin_eta[sl];e2=bp.lin_e2[sl]
        # Copy existing intermediates only. No extra strain/operator evaluation.
        with bp.m.span('diagnostic_material_extract'):
            selected=((tri[:,:,None]==tensor([540,9954,9955,9956,9999],True)[None,None,:]).any(2)).any(1)
            if bool(selected.any()):
                material.append(dict(elements=(torch.arange(start,start+len(tri),device='cuda')[selected]).cpu().numpy(),
                    triangles=tri[selected].cpu().numpy(),strain=raw_strain[selected].cpu().numpy(),
                    e2=e2[selected].cpu().numpy(),eta=eta[selected].cpu().numpy(),weight=bp.weight[sl][selected].cpu().numpy()))
        for j in range(3):
            dd=torch.zeros_like(hh);dd[:,j]=wet[tri[:,j]]
            ratio=torch.where(height>0,(dd@bp.lam.T)/height.clamp_min(1e-300),torch.zeros_like(height));dgz=-gz*ratio[:,:,None]
            dzx=torch.einsum('eqj,eja->eqa',bp.sigma[None,:,None]*dd[:,None,:],grad)
            db=strain(-dgz*zx[:,:,0,None]-gz*dzx[:,:,0,None],-dgz*zx[:,:,1,None]-gz*dzx[:,:,1,None],dgz)
            ds=torch.einsum('eqsi,ei->eqs',db,local[sl]);dbu=torch.einsum('eqsi,eqs->eqi',db,ss)+torch.einsum('eqsi,eqs->eqi',b,stress(ds))
            deta=-eta*(.5*(ss*ds).sum(2))/(3*e2)
            internal=(bp.weight[sl,:,None]*((ratio*eta+deta)[:,:,None]*bu+eta[:,:,None]*dbu)).sum(1)
            dgs=torch.einsum('ej,eja->ea',dd,grad)
            drive=900*9.80665*(dgs[:,None,None,:]+bp.gs[sl,None,None,:]*ratio[:,:,None,None])*bp.phi[None,:,:,None]*bp.weight[sl,:,None,None]
            out[sl,:,j]=(internal+drive.reshape(len(tri),12,12).sum(1))*bp.wet[bp.ids[sl]]
    bp.diagnostic_material={k:np.concatenate([part[k] for part in material]) for k in material[0]}
    return out

class Engine(BaseEngine):
    def full(self,*a,**kw):raise RuntimeError('v52 complete BP forbidden')
    def tangent(self,*a,**kw):raise RuntimeError('v52 exact Fu inverse forbidden')
    def pure(self,h,u):
        with self.m.span('pure_F'):
            f,force,wet=pure_residual(self.bp,tensor(h),tensor(u))
        with self.m.span('D2H'):return dict(F=f.cpu().numpy(),force=force.cpu().numpy(),wet=wet.cpu().numpy(),bound_point=self.state_id)
    def blocks(self,h,u,prepare_amg=True):
        f,en=self.bind(h,u,True)
        with self.m.span('local_B_assembly'):b=local_geometry_blocks(self.bp,self.bound_u)
        if prepare_amg:self.ensure_preconditioner()
        with self.m.span('D2H'):
            return dict(F=f.cpu().numpy(),force=self.bp.f.cpu().numpy(),wet=self.bp.wet.cpu().numpy(),point=self.state_id,
                A_values=self.bp.values.cpu().numpy().copy(),B_local=b.cpu().numpy(),peak_memory=torch.cuda.max_memory_allocated(),
                memory_quota=int(torch.cuda.get_device_properties(0).total_memory*.10),material=self.bp.diagnostic_material)
    def V(self,rhs):
        if self.step_calls['V']>=256:raise RuntimeError('256 V cycles exhausted')
        self.step_calls['V']+=1
        with self.m.span('V_cycle'):
            self.m.counts['AMG_coarse_solve']+=1;r=tensor(rhs);d=self.mg.d
            v=d*self.mg.cycle(d*r);v[~self.bp.wet]=0.
        with self.m.span('D2H'):return dict(value=v.cpu().numpy())
