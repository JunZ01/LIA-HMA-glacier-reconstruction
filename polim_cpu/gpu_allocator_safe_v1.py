"""Exact BP with reusable geometry, chunked diagonals and inactive-cache release."""
import torch
from .gpu_bp_v1 import T,DEVICE,scatter
from .gpu_chunked_v1 import BP as Chunked,PCG,Multigrid,solve

class BP(Chunked):
    def prepare(self,h):
        # Release only inactive allocator blocks, outside CUDA Graph capture.
        torch.cuda.empty_cache()
        self.h=h
        threshold=64*torch.finfo(T).eps*torch.maximum(torch.ones_like(h),torch.maximum(self.bed.abs(),(self.bed+h).abs()))
        physical=torch.where(h>threshold,h,torch.zeros_like(h));self.wet=(physical>0).repeat_interleave(4)
        H=physical[self.tri];bed=self.bed[self.tri];height=H@self.lam.T;dz=height*.5
        safe=torch.where(dz>0,dz,torch.ones_like(dz));self.weight=self.area[:,None]*dz/6
        zx=torch.einsum('eqj,eja->eqa',bed[:,None,:]+self.sigma[None,:,None]*H[:,None,:],self.grad)
        gz=self.sign[None,:,:]/safe[:,:,None]
        gx=self.grad[:,:,0].repeat_interleave(2,dim=1)[:,None,:]*self.vphi[None,:,:]-gz*zx[:,:,0,None]
        gy=self.grad[:,:,1].repeat_interleave(2,dim=1)[:,None,:]*self.vphi[None,:,:]-gz*zx[:,:,1,None]
        if not hasattr(self,'B'):
            self.B=torch.empty((len(H),12,5,12),device=DEVICE,dtype=T)
        B=self.B
        B.zero_()
        B[:,:,0,0::2]=gx;B[:,:,1,1::2]=gy;B[:,:,2,0::2]=gy;B[:,:,2,1::2]=gx;B[:,:,3,0::2]=gz;B[:,:,4,1::2]=gz
        self.B=B
        gs=torch.einsum('ej,eja->ea',bed+H,self.grad);self.gs=gs
        force=-900*9.80665*gs[:,None,None,:]*self.phi[None,:,:,None]*self.weight[:,:,None,None]
        self.flocal=force.reshape(len(H),12,12).sum(1);self.f=scatter(self.flocal,self.ids,self.ndof);self.f[~self.wet]=0
        aa=self.A[self.tri];aq=aa[:,:,self.layer]*(1-self.fraction)[None,None,:]+aa[:,:,self.layer+1]*self.fraction[None,None,:]
        self.coeff=torch.einsum('ejq,qj->eq',aq,self.lam).pow(-1/3)
        if not hasattr(self,'bdiag'):
            self.bdiag=torch.empty((len(H),12,12),device=DEVICE,dtype=T)
        for start in range(0,len(H),2048):
            end=min(start+2048,len(H));b=B[start:end]
            self.bdiag[start:end]=4*b[:,:,0,:]**2+4*b[:,:,1,:]**2+4*b[:,:,0,:]*b[:,:,1,:]+(b[:,:,2:,:]**2).sum(2)
        self.force_norm=self.f.norm().clamp_min(1.)

    def evaluate(self,u,linearize=False):
        if linearize:
            torch.cuda.empty_cache()
        return super().evaluate(u,linearize)
