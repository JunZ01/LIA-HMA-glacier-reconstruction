"""CUDA graph predictor with exact discrete BP energy custom backward."""
import torch
from .gpu_bp_v1 import tensor,scatter,sparse,T

class Energy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,bp,base,scale):
        r,e=bp.evaluate(u);ctx.save_for_backward(r/scale)
        return (e-base)/scale
    @staticmethod
    def backward(ctx,g):return g*ctx.saved_tensors[0],None,None,None

class Network:
    def __init__(self,z):
        self.theta=torch.nn.Parameter(tensor(z['theta']).clone());self.mean=tensor(z['mean']);self.std=tensor(z['std']);self.p=sparse(z,'graph')
        self.shapes=[(16,16),(16,16),(16,),(16,16),(16,16),(16,),(16,4),(4,)]
    def context(self,bp,h0,u,dt):
        old=torch.where(bp.wet,u,torch.zeros_like(u));r,e=bp.evaluate(old,True);cheap=(-r/bp.diagonal()).reshape(-1,4)
        scale=cheap.square().mean(1).sqrt()[:,None];legacy=torch.maximum(scale,.1*cheap.square().mean().sqrt().clamp_min(1e-12))
        active=(bp.h[bp.tri].sum(1)>0);area=bp.area*active
        weight=scatter(area.repeat_interleave(3),bp.tri,bp.n).clamp_min(1.)
        slope=torch.stack([scatter((bp.gs[:,k]*area).repeat_interleave(3),bp.tri,bp.n)/weight for k in range(2)],1)
        x=torch.cat([(bp.xy-bp.xy.mean(0))/10000,bp.bed[:,None]/5000,bp.h[:,None]/100,((bp.h-h0)/dt)[:,None]/10,slope,bp.A.mean(1).log10()[:,None]/20,old.reshape(-1,4)/10,cheap/legacy],1)
        return dict(old=old,cheap=cheap,scale=scale,x=(x-self.mean)/self.std,energy=e,loss_scale=(-r@cheap.ravel()).clamp_min(1.),residual=float(r.norm()/bp.force_norm))
    def forward(self,c,bp):
        values=[];i=0
        for shape in self.shapes:
            n=1
            for v in shape:n*=v
            values.append(self.theta[i:i+n].reshape(shape));i+=n
        w0,v0,b0,w1,v1,b1,w2,b2=values;x=c['x']
        a=torch.tanh(x@w0+torch.sparse.mm(self.p,x)@v0+b0)
        b=torch.tanh(a@w1+torch.sparse.mm(self.p,a)@v1+b1)
        u=c['old']+(c['cheap']+c['scale']*(b@w2+b2)).ravel()
        return torch.where(bp.wet,u,torch.zeros_like(u))
    def predict(self,bp,h0,u,dt):
        c=self.context(bp,h0,u,dt)
        with torch.no_grad():prior=self.forward(c,bp).clone()
        optimizer=torch.optim.LBFGS([self.theta],max_iter=4,max_eval=60,history_size=8,line_search_fn='strong_wolfe',tolerance_grad=1e-8,tolerance_change=1e-12)
        calls=0
        def closure():
            nonlocal calls
            calls+=1
            if calls>60:raise RuntimeError('Neural objective budget exceeded')
            optimizer.zero_grad();value=Energy.apply(self.forward(c,bp),bp,c['energy'],c['loss_scale']);value.backward();return value
        optimizer.step(closure)
        with torch.no_grad():
            updated=self.forward(c,bp);chosen=c['old'];best=c['residual'];label='previous';residuals=[]
            for name,v in [('prior',prior),('updated',updated)]:
                r,e=bp.evaluate(v);rr=float(r.norm()/bp.force_norm);residuals.append(rr)
                if rr<best and bool(e<=c['energy']):chosen=v.clone();best=rr;label=name
        return chosen,dict(chosen=label,old_residual=c['residual'],neural_residuals=residuals,objective_calls=calls)
