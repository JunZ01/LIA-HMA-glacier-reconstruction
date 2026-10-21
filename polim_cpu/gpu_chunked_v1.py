"""Memory-bounded exact tangent assembly; same v30 BP and Graph solver."""
import torch
from .gpu_cached_v1 import BP as Cached,PCG,Multigrid,solve
from .gpu_bp_v1 import BP as Original

class BP(Cached):
    def evaluate(self,u,linearize=False):
        r,e=Original.evaluate(self,u,linearize)
        if linearize:
            self.values.zero_()
            # Keep temporary CB/weighted-CB arrays bounded on the50m mesh.
            for start in range(0,len(self.B),2048):
                end=min(start+2048,len(self.B));B=self.B[start:end]
                CB=torch.stack([4*B[:,:,0]+2*B[:,:,1],2*B[:,:,0]+4*B[:,:,1],B[:,:,2],B[:,:,3],B[:,:,4]],2)
                factor=self.weight[start:end]*self.lin_eta[start:end];bu=self.lin_bu[start:end]
                local=torch.bmm(B.reshape(-1,60,12).transpose(1,2),(factor[:,:,None,None]*CB).reshape(-1,60,12))
                local-=torch.bmm(bu.transpose(1,2),(factor/(6*self.lin_e2[start:end]))[:,:,None]*bu)
                self.values.index_add_(0,self.mapping[start*144:end*144],local.ravel())
            self.active.copy_(self.wet)
        return r,e
