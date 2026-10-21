"""Small graph network trained by the unchanged discrete BP energy gradient.

All neural arithmetic is float64 NumPy. Graph aggregation has an explicit
transpose in backpropagation. Basal velocity is excluded from network outputs.
"""
import time
import numpy as np
from numba import njit,prange
from scipy.sparse import coo_matrix,diags
from scipy.optimize import minimize
from . import bp_accelerated_v1 as bp


@njit(cache=True,parallel=True)
def local_diagonal(local,bases,forces,coefficients,weights,regularization):
    result=np.zeros((len(local),12))
    for e in prange(len(local)):
        for p in range(12):
            b=bases[e,p];s=b@local[e]
            stress=np.array([4*s[0]+2*s[1],2*s[0]+4*s[1],s[2],s[3],s[4]])
            bu=b.T@stress
            e2=s[0]**2+s[1]**2+s[0]*s[1]+.25*(s[2]**2+s[3]**2+s[4]**2)+regularization
            eta=.5*coefficients[e,p]*e2**(-1./3.)
            for j in range(12):
                form=4*b[0,j]**2+4*b[1,j]**2+4*b[0,j]*b[1,j]+b[2,j]**2+b[3,j]**2+b[4,j]**2
                result[e,j]+=weights[e,p]*eta*(form-bu[j]**2/(6*e2))
    return result


def graph(mesh):
    t=mesh['triangles'];edges=np.unique(np.sort(np.vstack([t[:,[0,1]],t[:,[1,2]],t[:,[2,0]]]),axis=1),axis=0)
    nn=len(mesh['xy']);i=np.r_[edges[:,0],edges[:,1],np.arange(nn)];j=np.r_[edges[:,1],edges[:,0],np.arange(nn)]
    p=coo_matrix((np.ones(len(i)),(i,j)),shape=(nn,nn)).tocsr()
    p=diags(1/np.asarray(p.sum(axis=1)).ravel())@p
    return p.tocsr()


def context(mesh,state,previous_h,u,dt,cache):
    start=time.perf_counter();a=bp.prepare({**mesh,'thickness':state['h']},state['A'],cache)
    old=u[:,1:].ravel().copy();old[~a['wet']]=0
    r,e,_,f=bp.evaluate(a,old,False)
    dlocal=local_diagonal(old[a['ids']],*a['quadrature'],bp.original.REG)
    d=np.bincount(a['ids'].ravel(),weights=dlocal.ravel(),minlength=a['ndof']);d[~a['wet']]=1.
    assert np.all(d>0) and np.isfinite(d).all()
    cheap=(-r/d).reshape(-1,4);rms=max(float(np.sqrt(np.mean(cheap**2))),1e-12)
    scale=np.maximum(np.sqrt(np.mean(cheap**2,axis=1)),.1*rms)[:,None]
    # FE surface gradients are area-averaged onto the conforming mesh nodes.
    tri=a['tri'];gs=np.einsum('ti,tia->ta',a['bed']+a['H'],a['grad'])
    weight=np.bincount(tri.ravel(),weights=np.repeat(a['area'],3),minlength=len(u));weight=np.maximum(weight,1.)
    slope=np.column_stack([np.bincount(tri.ravel(),weights=np.repeat(gs[:,k]*a['area'],3),minlength=len(u))/weight for k in range(2)])
    xy=mesh['xy'];h=state['h'];rate=(h-previous_h)/max(dt,1e-12)
    x=np.column_stack(((xy-xy.mean(0))/10000,mesh['bed']/5000,h/100,rate/10,slope,
          np.log10(np.maximum(state['A'].mean(axis=1),1e-40))/20,old.reshape(-1,4)/10,cheap/scale))
    return dict(a=a,old=old,cheap=cheap,scale=scale,x=x,energy=e,
        loss_scale=max(float(-r@cheap.ravel()),1.),force_norm=max(float(np.linalg.norm(f)),1.),
        old_residual=float(np.linalg.norm(r)/max(np.linalg.norm(f),1.)),diagonal=d,prepare_seconds=time.perf_counter()-start)


class Network:
    def __init__(self,p,mean,std,theta=None):
        self.p=p;self.pt=p.T.tocsr();self.mean=mean;self.std=std
        nf=len(mean);self.shapes=[(nf,16),(nf,16),(16,),(16,16),(16,16),(16,),(16,4),(4,)]
        self.sizes=[int(np.prod(s)) for s in self.shapes];self.offset=np.r_[0,np.cumsum(self.sizes)]
        rng=np.random.default_rng(2387)
        values=[rng.normal(0,.15,size=s) if len(s)==2 else np.zeros(s) for s in self.shapes]
        values[-2]*=0.;values[-1]*=0.
        self.theta=np.concatenate([v.ravel() for v in values]) if theta is None else theta.copy()
    def unpack(self,theta):return [theta[self.offset[i]:self.offset[i+1]].reshape(s) for i,s in enumerate(self.shapes)]
    def forward(self,c,theta):
        w0,v0,b0,w1,v1,b1,w2,b2=self.unpack(theta);x=(c['x']-self.mean)/self.std;px=self.p@x
        a=np.tanh(x@w0+px@v0+b0);pa=self.p@a
        b=np.tanh(a@w1+pa@v1+b1);out=b@w2+b2
        velocity=c['old']+(c['cheap']+c['scale']*out).ravel()
        velocity[~c['a']['wet']]=0.
        return velocity,(x,px,a,pa,b)
    def objective(self,c,theta):
        u,saved=self.forward(c,theta);r,e,_,f=bp.evaluate(c['a'],u,False)
        w0,v0,b0,w1,v1,b1,w2,b2=self.unpack(theta);x,px,a,pa,b=saved
        r[~c['a']['wet']]=0.;g=r.reshape(-1,4)*c['scale']/c['loss_scale']
        gw2=b.T@g;gb2=g.sum(0);gb=(g@w2.T)*(1-b*b)
        gw1=a.T@gb;gv1=pa.T@gb;gb1=gb.sum(0)
        ga=(gb@w1.T+self.pt@(gb@v1.T))*(1-a*a)
        grads=[x.T@ga,px.T@ga,ga.sum(0),gw1,gv1,gb1,gw2,gb2]
        return (e-c['energy'])/c['loss_scale'],np.concatenate([v.ravel() for v in grads])
    def optimize(self,contexts,iterations,maxcalls,callback=None):
        start=time.perf_counter();calls=0;history=[]
        def objective(theta):
            nonlocal calls
            calls+=1
            if calls>maxcalls:raise RuntimeError('Registered neural objective-call budget exceeded')
            value=0.;g=np.zeros_like(theta)
            for c in contexts:
                v,grad=self.objective(c,theta);value+=v;g+=grad
            return value/len(contexts),g/len(contexts)
        def accepted(theta):
            history.append(dict(iteration=len(history)+1,objective_calls=calls,seconds=time.perf_counter()-start))
            if callback:callback(len(history),theta,history[-1])
        result=minimize(objective,self.theta.copy(),method='L-BFGS-B',jac=True,callback=accepted,
            options=dict(maxiter=iterations,maxfun=maxcalls,maxls=8,ftol=1e-12,gtol=1e-8,maxcor=8))
        self.theta=result.x.copy()
        return dict(seconds=time.perf_counter()-start,iterations=int(result.nit),objective_calls=calls,
            success=bool(result.success),message=str(result.message),objective=float(result.fun),history=history)


def save_network(path,net):
    with path.open('xb') as f:np.savez(f,theta=net.theta,mean=net.mean,std=net.std)


def load_network(path,p):
    with np.load(path,allow_pickle=False) as z:return Network(p,z['mean'],z['std'],z['theta'])
