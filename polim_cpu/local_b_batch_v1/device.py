"""Batch three LOCAL bases; all current coefficients remain freshly evaluated."""
from polim_cpu.anchor_efficiency_v1.device import *
from polim_cpu.anchor_efficiency_v1.device import Engine as OriginalEngine,local_geometry_blocks as original_B
import os

def basis_batch(bp,sl,hh,grad,height,gz,zx,b,ss,bu,eta,e2,local):
    n=len(hh);wet=bp.wet[::4]
    dd=torch.eye(3,device=hh.device,dtype=T)[None,:,:]*wet[bp.tri[sl]][:,None,:]
    ratio=torch.where(height[:,None,:]>0,(dd@bp.lam.T)/height[:,None,:].clamp_min(1e-300),torch.zeros((n,3,12),device=hh.device,dtype=T))
    dgz=-gz[:,None,:,:]*ratio[:,:,:,None]
    dzx=torch.einsum('ejqk,eka->ejqa',bp.sigma[None,None,:,None]*dd[:,:,None,:],grad)
    dx=-dgz*zx[:,None,:,0,None]-gz[:,None,:,:]*dzx[:,:,:,0,None]
    dy=-dgz*zx[:,None,:,1,None]-gz[:,None,:,:]*dzx[:,:,:,1,None]
    db=strain(dx.reshape(n*3,12,6),dy.reshape(n*3,12,6),dgz.reshape(n*3,12,6)).reshape(n,3,12,5,12)
    ds=torch.einsum('ejqsi,ei->ejqs',db,local)
    dss=stress(ds.reshape(n*3,12,5)).reshape(n,3,12,5)
    dbu=torch.einsum('ejqsi,eqs->ejqi',db,ss)+torch.einsum('eqsi,ejqs->ejqi',b,dss)
    deta=-eta[:,None,:]*(.5*(ss[:,None,:,:]*ds).sum(3))/(3*e2[:,None,:])
    internal=(bp.weight[sl,None,:,None]*((ratio*eta[:,None,:]+deta)[:,:,:,None]*bu[:,None,:,:]+eta[:,None,:,None]*dbu)).sum(2)
    dgs=torch.einsum('ejk,eka->eja',dd,grad)
    drive=900*9.80665*(dgs[:,:,None,None,:]+bp.gs[sl,None,None,None,:]*ratio[:,:,:,None,None])*bp.phi[None,None,:,:,None]*bp.weight[sl,None,:,None,None]
    return ((internal+drive.reshape(n,3,12,12).sum(2))*bp.wet[bp.ids[sl]][:,None,:]).transpose(1,2)

def local_geometry_blocks(bp,u):
    wet=bp.wet[::4];physical=torch.where(wet,bp.h,torch.zeros_like(bp.h));local=u[bp.ids]
    out=torch.zeros((len(bp.tri),12,3),device='cuda',dtype=T);material=[]
    for start in range(0,len(bp.tri),1024):
        sl=slice(start,start+1024);tri=bp.tri[sl];hh=physical[tri];grad=bp.grad[sl];height=hh@bp.lam.T
        dz=.5*height;safe=torch.where(dz>0,dz,torch.ones_like(dz));gz=bp.sign[None,:,:]/safe[:,:,None]
        zx=torch.einsum('eqj,eja->eqa',bp.bed[tri][:,None,:]+bp.sigma[None,:,None]*hh[:,None,:],grad)
        b=bp.B[sl];raw_strain=torch.einsum('eqsi,ei->eqs',b,local[sl]);ss=stress(raw_strain);bu=bp.lin_bu[sl];eta=bp.lin_eta[sl];e2=bp.lin_e2[sl]
        # Original scientific material extraction and synchronization unchanged.
        with bp.m.span('diagnostic_material_extract'):
            selected=((tri[:,:,None]==tensor([540,9954,9955,9956,9999],True)[None,None,:]).any(2)).any(1)
            if bool(selected.any()):
                material.append(dict(elements=(torch.arange(start,start+len(tri),device='cuda')[selected]).cpu().numpy(),triangles=tri[selected].cpu().numpy(),strain=raw_strain[selected].cpu().numpy(),e2=e2[selected].cpu().numpy(),eta=eta[selected].cpu().numpy(),weight=bp.weight[sl][selected].cpu().numpy()))
        out[sl]=basis_batch(bp,sl,hh,grad,height,gz,zx,b,ss,bu,eta,e2,local[sl])
    bp.diagnostic_material={k:np.concatenate([part[k] for part in material]) for k in material[0]}
    return out

class Engine(OriginalEngine):
    def blocks(self,h,u,prepare_amg=True):
        f,en=self.bind(h,u,True)
        fn=local_geometry_blocks if os.environ.get('POLIM_BATCH_LOCAL_B')=='1' else original_B
        with self.m.span('local_B_assembly'):b=fn(self.bp,self.bound_u)
        if prepare_amg:self.ensure_preconditioner()
        with self.m.span('D2H'):
            return dict(F=f.cpu().numpy(),force=self.bp.f.cpu().numpy(),wet=self.bp.wet.cpu().numpy(),point=self.state_id,A_values=self.bp.values.cpu().numpy().copy(),B_local=b.cpu().numpy(),peak_memory=torch.cuda.max_memory_allocated(),memory_quota=int(torch.cuda.get_device_properties(0).total_memory*.10),material=self.bp.diagnostic_material)
