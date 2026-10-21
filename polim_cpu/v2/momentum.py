"""Q1 terrain-following finite elements; exact Newton tangent and Armijo search.

This is a new discretization, NOT numerical equivalence to the MATLAB stencil.
SI lengths, years, Pa; arrays use (horizontal node, vertical node).
"""
import numpy as np


class Backend:
    def __init__(self, device='cpu'):
        self.device = device
        self.gpu = device == 'cuda'
        if self.gpu:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
            torch.set_deterministic_debug_mode("error")
            self.xp = torch
        elif device == 'cpu':
            self.xp = np
        else:
            raise ValueError('device must be cpu or cuda')

    def array(self, value, integer=False):
        if self.gpu:
            return self.xp.as_tensor(value, device='cuda', dtype=self.xp.int64 if integer else self.xp.float64)
        return np.asarray(value, dtype=np.int64 if integer else np.float64)

    def host(self, value):
        return value.detach().cpu().numpy() if self.gpu else np.asarray(value)

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.xp.float64, **({'device':'cuda'} if self.gpu else {}))

    def clone(self, value):
        return value.clone() if self.gpu else value.copy()

    def add(self, vector, ids, values):
        if self.gpu:
            vector.index_add_(0, ids.reshape(-1), values.reshape(-1))
        else:
            np.add.at(vector, ids.reshape(-1), values.reshape(-1))

    def linear(self, ids, matrices, rhs, free):
        ndof = rhs.size if not self.gpu else rhs.numel()
        rows = ids[:, :, None] + self.array(np.zeros((1,1,4)), True)
        cols = ids[:, None, :] + self.array(np.zeros((1,4,1)), True)
        if self.gpu:
            # Dense CUDA direct solve is intentional for a flowline prototype.
            # Refuse an oversized job before allocating; do not cap the grid silently.
            required = ndof*ndof*8*5
            available, _ = self.xp.cuda.mem_get_info()
            if required > available*.7:
                raise MemoryError(f'CUDA direct solver needs about {required} bytes; use a smaller explicit grid or CPU sparse backend')
            matrix = self.zeros((ndof,ndof))
            matrix.index_put_((rows.reshape(-1), cols.reshape(-1)), matrices.reshape(-1), accumulate=True)
            result = self.zeros(ndof)
            result[free] = self.xp.linalg.solve(matrix[free][:,free], rhs[free])
            return result
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import spsolve
        matrix = coo_matrix((matrices.reshape(-1),(rows.reshape(-1),cols.reshape(-1))),shape=(ndof,ndof)).tocsr()
        result = np.zeros(ndof)
        result[free] = spsolve(matrix[free][:,free].tocsc(), rhs[free])
        return result


def validate_geometry(x, bed, h, width, layers):
    if len(x)<3 or layers<3 or any(np.shape(a)!=np.shape(x) for a in (bed,h,width)):
        raise ValueError('At least three horizontal/vertical nodes and matching geometry arrays required')
    if any(not np.isfinite(a).all() for a in (x,bed,h,width)) or np.any(np.diff(x)<=0):
        raise ValueError('Geometry must be finite and x strictly increasing')
    if np.any(h<=0) or np.any(width<=0):
        raise ValueError('Active-domain thickness and width must be positive; no automatic thickness floor')


def elements(x, bed, h, width, layers, flowband=False):
    validate_geometry(x,bed,h,width,layers)
    ids=[]; forms=[]; weights=[]; shapes=[]; loads=[]
    gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for i in range(len(x)-1):
        dx=x[i+1]-x[i]
        slope=(bed[i+1]+h[i+1]-bed[i]-h[i])/dx
        for j in range(layers-1):
            nodes=[i*layers+j,i*layers+j+1,(i+1)*layers+j,(i+1)*layers+j+1]
            for xi in gauss:
                for eta in gauss:
                    a=(xi+1)/2; zeta=(j+(eta+1)/2)/(layers-1)
                    shape=np.array([(1-xi)*(1-eta),(1-xi)*(1+eta),(1+xi)*(1-eta),(1+xi)*(1+eta)])/4
                    dxi=np.array([-(1-eta),-(1+eta),1-eta,1+eta])/4
                    deta=np.array([-(1-xi),1-xi,-(1+xi),1+xi])/4
                    ht=(1-a)*h[i]+a*h[i+1]; w=(1-a)*width[i]+a*width[i+1]
                    zx=(bed[i+1]-bed[i]+zeta*(h[i+1]-h[i]))/2
                    ze=ht/(2*(layers-1))
                    gx=(dxi-deta*zx/ze)/(dx/2); gz=deta/ze
                    B=4*np.outer(gx,gx)+np.outer(gz,gz)
                    if flowband:
                        gy=shape*(width[i+1]-width[i])/dx/w
                        B+=4*np.outer(gy,gy)+2*np.outer(gx,gy)+2*np.outer(gy,gx)+np.outer(shape,shape)/w**2
                    weight=w*dx/2*ze
                    ids.append(nodes); forms.append(B); weights.append(weight); shapes.append(shape)
                    loads.append(-910*9.81*slope*shape*weight)
    return np.array(ids),np.array(forms),np.array(weights),np.array(shapes),np.array(loads)


def solve(x,bed,h,width,layers,A,backend,parameters=None,initial=None):
    p=parameters or {}; xp=backend.xp; b=backend
    n=float(p.get('glen_n',3)); reg=float(p.get('strain_regularization',1e-30))
    tolerance=float(p.get('relative_tolerance',1e-6)); limit=int(p.get('max_iterations',80))
    if n<1 or reg<=0 or not np.isfinite([n,reg,tolerance]).all() or tolerance<=0 or limit<1:
        raise ValueError('Invalid momentum parameters')
    raw=elements(x,bed,h,width,layers,p.get('flowband',False))
    ids,B,weight,shape,local_f=[b.array(v,integer=k==0) for k,v in enumerate(raw)]
    ndof=len(x)*layers
    A=b.array(A) if np.isscalar(A) else A
    if A.ndim==0: aq=A+weight*0
    else: aq=xp.sum(A.reshape(-1)[ids]*shape,axis=1)
    if bool(xp.any(aq<=0)): raise ValueError('Glen A must be positive')
    coeff=aq**(-1/n); power=(1-n)/(2*n)
    force=b.zeros(ndof); b.add(force,ids,local_f)
    fixed=np.zeros(ndof,bool)
    basal=p.get('basal','zero')
    beta=float(p.get('beta2',1000))
    if basal=='zero': fixed[np.arange(len(x))*layers]=True
    elif basal=='linear':
        if beta<=0: raise ValueError('Linear basal beta2 must be positive')
    else: raise ValueError('basal must be zero or linear')
    if p.get('headwall',True): fixed[:layers]=True
    free=b.array(np.flatnonzero(~fixed),True)
    basal_diag=b.zeros(ndof)
    if basal=='linear':
        cv=np.r_[np.diff(x)[0]/2,(x[2:]-x[:-2])/2,np.diff(x)[-1]/2]
        basal_diag[b.array(np.arange(len(x))*layers,True)]=b.array(beta*width*cv)
    # Fold a lumped basal drag into the element matrices with partition weights.
    occurrences=np.bincount(raw[0].reshape(-1),minlength=ndof)
    drag=basal_diag[ids]/b.array(occurrences)[ids]
    eye=b.array(np.eye(4))
    def evaluate(u,tangent=False):
        local=u[ids]; Bu=xp.einsum('qij,qj->qi',B,local)
        e=xp.sum(local*Bu,axis=1)/4+reg
        viscosity=.5*coeff*e**power
        r=b.zeros(ndof); b.add(r,ids,weight[:,None]*viscosity[:,None]*Bu)
        r+=basal_diag*u-force
        energy=xp.sum(weight*(2*n/(n+1))*coeff*(e**((n+1)/(2*n))-reg**((n+1)/(2*n))))-xp.sum(force*u)+.5*xp.sum(basal_diag*u*u)
        if not tangent: return r,energy
        K=viscosity[:,None,None]*(B+(power/(2*e))[:,None,None]*Bu[:,:,None]*Bu[:,None,:])*weight[:,None,None]
        K+=eye[None,:,:]*drag[:,:,None]
        return r,energy,K
    if initial is None:
        K=1e6*weight[:,None,None]*B+eye[None,:,:]*drag[:,:,None]
        u=b.linear(ids,K,force,free)
    else: u=b.clone(initial).reshape(-1); u[b.array(np.flatnonzero(fixed),True)]=0
    norm=float(xp.sqrt(xp.sum(force[free]**2))); history=[]
    for iteration in range(limit+1):
        r,energy,K=evaluate(u,True)
        residual=float(xp.sqrt(xp.sum(r[free]**2)))/max(norm,1.)
        history.append(residual)
        if residual<=tolerance: break
        if iteration==limit: raise RuntimeError(f'Newton did not converge: residual={residual:.6g}, iterations={limit}')
        delta=b.linear(ids,K,-r,free)
        descent=float(xp.sum(r*delta)); alpha=1.
        for attempt in range(35):
            trial=u+alpha*delta; trial_r,trial_energy=evaluate(trial)
            trial_res=float(xp.sqrt(xp.sum(trial_r[free]**2)))/max(norm,1.)
            # Residual fallback handles energy cancellation at the final Newton step.
            if float(trial_energy)<=float(energy)+1e-4*alpha*descent or trial_res<residual*.5:
                u=trial; break
            alpha*=.5
        else: raise RuntimeError(f'Newton line search failed at residual {residual}')
    return u.reshape(len(x),layers),{'converged':True,'iterations':iteration,'relative_residual':residual,'residual_history':history}
