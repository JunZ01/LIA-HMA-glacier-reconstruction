"""CUDA-resident BP energy, analytic gradient/Hessian products and transport.

Imported only by the isolated CUDA worker after safe DLL initialization.
Same P1 triangles, three vertical nodes, quadrature and dry test as v13r1.
"""
import torch
T=torch.float64
DEVICE='cuda'
REG=1e-20


def tensor(x,integer=False):return torch.as_tensor(x,dtype=torch.int64 if integer else T,device=DEVICE)
def scatter(values,ids,n):
    out=torch.zeros(n,dtype=T,device=DEVICE);out.index_add_(0,ids.reshape(-1),values.reshape(-1));return out


class BP:
    def __init__(self,data):
        self.n=len(data['bed']);self.tri=tensor(data['tri'],True);self.ids=tensor(data['ids'],True)
        self.grad=tensor(data['grad']);self.area=tensor(data['area']);self.bed=tensor(data['bed']);self.A=tensor(data['A'])
        self.xy=tensor(data['xy']);self.ndof=self.n*4
        ls=[];phis=[];signs=[];sigmas=[];layers=[];fractions=[]
        for k in range(2):
            for b in range(3):
                lam=[1/6]*3;lam[b]=2/3
                for q in range(2):
                    tt=(1+(2*q-1)/3**.5)*.5;phi=[];sign=[]
                    for j in range(3):
                        for level in [1,2]:
                            if level==k:phi.append((1-tt)*lam[j]);sign.append(-lam[j])
                            elif level==k+1:phi.append(tt*lam[j]);sign.append(lam[j])
                            else:phi.append(0.);sign.append(0.)
                    ls.append(lam);phis.append(phi);signs.append(sign);sigmas.append((k+tt)*.5);layers.append(k);fractions.append(tt)
        self.lam=tensor(ls);self.phi=tensor(phis);self.sign=tensor(signs);self.sigma=tensor(sigmas);self.layer=tensor(layers,True);self.fraction=tensor(fractions)
        # phi already includes barycentric lambda; horizontal gradient needs phi/lambda.
        self.vphi=self.phi/self.lam.repeat_interleave(2,dim=1)
    def prepare(self,h):
        self.h=h
        threshold=64*torch.finfo(T).eps*torch.maximum(torch.ones_like(h),torch.maximum(self.bed.abs(),(self.bed+h).abs()))
        physical=torch.where(h>threshold,h,torch.zeros_like(h));self.wet=(physical>0).repeat_interleave(4)
        H=physical[self.tri];bed=self.bed[self.tri];height=H@self.lam.T;dz=height*.5
        safe=torch.where(dz>0,dz,torch.ones_like(dz));self.weight=self.area[:,None]*dz/6
        zx=torch.einsum('eqj,eja->eqa',bed[:,None,:]+self.sigma[None,:,None]*H[:,None,:],self.grad)
        gz=self.sign[None,:,:]/safe[:,:,None]
        gx=self.grad[:,:,0].repeat_interleave(2,dim=1)[:,None,:]*self.vphi[None,:,:]-gz*zx[:,:,0,None]
        gy=self.grad[:,:,1].repeat_interleave(2,dim=1)[:,None,:]*self.vphi[None,:,:]-gz*zx[:,:,1,None]
        B=torch.zeros((len(H),12,5,12),device=DEVICE,dtype=T)
        B[:,:,0,0::2]=gx;B[:,:,1,1::2]=gy;B[:,:,2,0::2]=gy;B[:,:,2,1::2]=gx;B[:,:,3,0::2]=gz;B[:,:,4,1::2]=gz
        self.B=B
        gs=torch.einsum('ej,eja->ea',bed+H,self.grad);self.gs=gs
        force=-900*9.80665*gs[:,None,None,:]*self.phi[None,:,:,None]*self.weight[:,:,None,None]
        self.flocal=force.reshape(len(H),12,12).sum(1);self.f=scatter(self.flocal,self.ids,self.ndof);self.f[~self.wet]=0
        aa=self.A[self.tri];aq=aa[:,:,self.layer]*(1-self.fraction)[None,None,:]+aa[:,:,self.layer+1]*self.fraction[None,None,:]
        self.coeff=torch.einsum('ejq,qj->eq',aq,self.lam).pow(-1/3)
        self.bdiag=4*B[:,:,0,:]**2+4*B[:,:,1,:]**2+4*B[:,:,0,:]*B[:,:,1,:]+(B[:,:,2:,:]**2).sum(2)
        self.force_norm=self.f.norm().clamp_min(1.)
    def evaluate(self,u,linearize=False):
        local=u[self.ids];s=torch.einsum('eqsi,ei->eqs',self.B,local)
        e2=s[:,:,0]**2+s[:,:,1]**2+s[:,:,0]*s[:,:,1]+.25*(s[:,:,2:]**2).sum(2)+REG
        stress=torch.stack([4*s[:,:,0]+2*s[:,:,1],2*s[:,:,0]+4*s[:,:,1],s[:,:,2],s[:,:,3],s[:,:,4]],dim=2)
        bu=torch.einsum('eqsi,eqs->eqi',self.B,stress);eta=.5*self.coeff*e2.pow(-1/3)
        localr=(self.weight[:,:,None]*eta[:,:,None]*bu).sum(1)-self.flocal
        r=scatter(localr,self.ids,self.ndof);r=torch.where(self.wet,r,u)
        energy=(self.weight*1.5*self.coeff*(e2.pow(2/3)-REG**(2/3))).sum()-(self.f*u).sum()+.5*(u[~self.wet]**2).sum()
        if linearize:
            self.lin_eta=eta;self.lin_bu=bu;self.lin_e2=e2
        return r,energy
    def diagonal(self):
        local=(self.weight[:,:,None]*self.lin_eta[:,:,None]*(self.bdiag-self.lin_bu**2/(6*self.lin_e2[:,:,None]))).sum(1)
        return torch.where(self.wet,scatter(local,self.ids,self.ndof),torch.ones(self.ndof,device=DEVICE,dtype=T))
    def hvp(self,v):
        vv=torch.where(self.wet,v,torch.zeros_like(v));local=vv[self.ids];s=torch.einsum('eqsi,ei->eqs',self.B,local)
        ss=torch.stack([4*s[:,:,0]+2*s[:,:,1],2*s[:,:,0]+4*s[:,:,1],s[:,:,2],s[:,:,3],s[:,:,4]],dim=2)
        x=torch.einsum('eqsi,eqs->eqi',self.B,ss)-self.lin_bu*(self.lin_bu*local[:,None,:]).sum(2)[:,:,None]/(6*self.lin_e2[:,:,None])
        localx=(self.weight[:,:,None]*self.lin_eta[:,:,None]*x).sum(1)
        return torch.where(self.wet,scatter(localx,self.ids,self.ndof),v)


def sparse(z,prefix):
    indices=tensor(z[prefix+'_indices'],True);indptr=tensor(z[prefix+'_indptr'],True);values=tensor(z[prefix+'_data'])
    shape=tuple(int(x) for x in z[prefix+'_shape'])
    return torch.sparse_csr_tensor(indptr,indices,values,size=shape,device=DEVICE,dtype=T)
def mv(a,x):return torch.sparse.mm(a,x[:,None]).ravel()


class Multigrid:
    def __init__(self,z):
        self.d=tensor(z['scaling']);self.a=[];self.p=[];self.r=[];self.inv=[];self.omega=[]
        levels=int(z['levels'])
        for i in range(levels):
            self.a.append(sparse(z,f'A{i}'));self.inv.append(tensor(z[f'invdiag{i}']));self.omega.append(float(z[f'omega{i}']))
            if i<levels-1:self.p.append(sparse(z,f'P{i}'));self.r.append(sparse(z,f'R{i}'))
        self.chol=torch.linalg.cholesky(self.a[-1].to_dense())
    def cycle(self,b,i=0):
        if i==len(self.a)-1:return torch.cholesky_solve(b[:,None],self.chol).ravel()
        x=torch.zeros_like(b)
        for _ in range(2):x=x+self.omega[i]*self.inv[i]*(b-mv(self.a[i],x))
        x=x+mv(self.p[i],self.cycle(mv(self.r[i],b-mv(self.a[i],x)),i+1))
        for _ in range(2):x=x+self.omega[i]*self.inv[i]*(b-mv(self.a[i],x))
        return x


def solve(bp,u,mg,checkpoint=None):
    import time
    start=time.perf_counter();u=u.clone();u[~bp.wet]=0.;history=[]
    for it in range(21):
        r,e=bp.evaluate(u,True);res=float(r.norm()/bp.force_norm)
        history.append(dict(iteration=it,residual=res))
        if res<=1e-6:return u,dict(converged=True,iterations=it,relative_residual=res,history=history,seconds=time.perf_counter()-start)
        if it==20 or time.perf_counter()-start>120:raise RuntimeError('GPU Newton budget exceeded')
        d=mg.d;b=-d*r;y=torch.zeros_like(b);rr=b.clone();z=mg.cycle(rr);p=z.clone();rz=(rr*z).sum();rn=r.norm()
        for k in range(200):
            ap=d*bp.hvp(d*p);den=(p*ap).sum()
            if not bool(den>0):raise RuntimeError('Nonpositive GPU PCG curvature')
            alpha=rz/den;y=y+alpha*p;rr=rr-alpha*ap
            # Match physical rather than scaled residual, as in the CPU solver.
            ratio=float((rr/d).norm()/rn)
            if ratio<=min(.1,max(1e-4,.05*res**.5)):break
            z=mg.cycle(rr);newrz=(rr*z).sum();p=z+(newrz/rz)*p;rz=newrz
        else:raise RuntimeError('GPU PCG budget exceeded')
        delta=d*y;true_ratio=float((bp.hvp(delta)+r).norm()/rn)
        if true_ratio>.1:raise RuntimeError('GPU true linear residual failed')
        descent=float(r@delta);alpha=1.;energy=float(e)
        if descent>=0:raise RuntimeError('Non-descent GPU Newton direction')
        for _ in range(35):
            trial=u+alpha*delta;rt,et=bp.evaluate(trial)
            resolved=abs(1e-4*alpha*descent)>64*torch.finfo(T).eps*max(abs(energy),1.)
            decreased=float(rt.norm())<(1-1e-4*alpha)*float(rn)
            if (resolved and float(et)<=energy+1e-4*alpha*descent) or (not resolved and decreased) or float(rt.norm())<.5*float(rn):
                u=trial;break
            alpha*=.5
        else:raise RuntimeError('GPU line search failed')
        history[-1].update(cg_iterations=k+1,linear_residual=true_ratio,step_length=alpha)
        if checkpoint and (it+1)%5==0:checkpoint(it+1,u,history[-1])


class Transport:
    def __init__(self,z):
        self.n=len(z['h']);self.dual=tensor(z['geo_dual_area'])
        for k in ['a','b','third','inverse','edges','outer_a','outer_b']:setattr(self,k,tensor(z['geo_'+k],True))
        self.normal=tensor(z['geo_normal']);self.outer_normal=tensor(z['geo_outer_normal'])
        self.links=torch.cat([self.edges,torch.stack([torch.cat([self.outer_a,self.outer_b]),torch.full((2*len(self.outer_a),),-1,device=DEVICE,dtype=torch.int64)],dim=1)])
    def flux(self,h,u):
        vel=u.reshape(-1,2,2);mean=.5*vel[:,0,:]+.25*vel[:,1,:]
        v=5/12*(mean[self.a]+mean[self.b])+1/6*mean[self.third]
        integrated=scatter((v*self.normal).sum(1),self.inverse,len(self.edges))
        donor=torch.where(integrated>=0,self.edges[:,0],self.edges[:,1]);q=integrated*h[donor]
        ext=torch.cat([(((.75*mean[self.outer_a]+.25*mean[self.outer_b])*self.outer_normal).sum(1)).clamp_min(0)*h[self.outer_a],(((.75*mean[self.outer_b]+.25*mean[self.outer_a])*self.outer_normal).sum(1)).clamp_min(0)*h[self.outer_b]])
        return torch.cat([q,ext])
    def window(self,h,u,now,end):
        exported=0.;steps=0
        while now<end-1e-16:
            q=self.flux(h,u);volume=h*self.dual;donor=torch.where(q>=0,self.links[:,0],self.links[:,1]);receiver=torch.where(q>=0,self.links[:,1],self.links[:,0])
            out=scatter(q.abs(),donor,self.n);wet=(volume>0)&(out>0)
            dt=min(end-now,float((.35*volume[wet]/out[wet]).min()) if bool(wet.any()) else end-now)
            if dt<1e-13:raise RuntimeError('GPU CFL collapse')
            requested=q.abs()*dt;outgoing=scatter(requested,donor,self.n);limited=outgoing>volume
            factor=torch.where(limited,volume/outgoing.clamp_min(1e-300)*(1-64*torch.finfo(T).eps),torch.ones_like(volume))
            actual=requested*factor[donor];retained=volume-scatter(actual,donor,self.n)
            if bool((retained<0).any()):raise RuntimeError('GPU donor limiting failed')
            internal=receiver>=0;arriving=scatter(actual[internal],receiver[internal],self.n)
            h=(retained+arriving)/self.dual;exported+=float(actual[~internal].sum());now+=dt;steps+=1
            if steps>100:raise RuntimeError('GPU transport budget')
        return h,now,exported,steps
