"""Projected damped reduced Newton; original BE/upwind equations decide acceptance."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres,LinearOperator,splu
from polim_cpu.large_step_transport_v1 import implicit_conservative_step
from polim_cpu.diffusion_predictor_v2 import predict
from polim_cpu.geometry_probe_v2.response import decompose,wet,freeze_masks,branch_events,flux
from .backend import LinearFailure,BudgetStop

def physical_source(h,raw,smb,area,dt):
    """Only dry cells may leave requested negative SMB unrealized.

    Projected Newton displacement is never an SMB contribution. Outside the
    admissible range, retain an equation residual instead of making mass up.
    """
    allowed=dt*area*np.maximum(-smb,0.)
    unmet=np.where(h==0,np.minimum(np.maximum(raw,0.),allowed),0.)
    return smb+unmet/(dt*area),unmet

def feasible(h,delta,alpha):
    trial=h+alpha*delta;hit=trial<0
    # This is a candidate on the obstacle face, NOT an accepted mass update.
    out=np.where(hit,0.,trial)
    return out,hit

class Mass:
    def __init__(self,adapter,h_old,smb,work,dt=.25):
        self.adapter=adapter;self.old=h_old.copy();self.smb=smb.copy();self.w=work;self.dt=dt
        self.old.flags.writeable=False;self.smb.flags.writeable=False
        self.area=adapter.topology.dual_area;self.V0=float(self.area@self.old);self.rhs=self.area*(self.old+dt*self.smb)
    def evaluate(self,h,u):
        with self.w.span('formal_transport'):
            rates=self.adapter.rates(u);T=sparse.diags(self.area)+self.dt*rates.outward_operator();raw=T@h-self.rhs
            actual,unmet=physical_source(h,raw,self.smb,self.area,self.dt)
            residual=raw-unmet;source=self.dt*float(self.area@actual);export=self.dt*rates.boundary_export_rate(h)
            scale=max(self.V0+abs(source)+self.dt*rates.absolute_flux_rate(h),1.)
            fixed=implicit_conservative_step(self.old,self.smb,self.dt,rates)
            # SAME original service definitions retained alongside local complementarity residual.
            official=T@h-self.rhs-fixed.unmet_ablation_volume
            official_source=self.dt*float(self.area@fixed.actual_smb)
            oscale=max(self.V0+abs(official_source)+self.dt*rates.absolute_flux_rate(h),1.)
            local=float(abs(official).sum()/oscale)
            mass=abs(float(self.area@h)-self.V0-official_source+export)/max(self.V0,abs(official_source),abs(export),1.)
            natural=h-np.maximum(0.,h-raw/self.area)
            nr=float(self.area@abs(natural)/max(self.V0,1.))
            active_equal=np.array_equal(h==0,fixed.thickness==0)
            fixed_point=float(self.area@abs(h-fixed.thickness)/max(self.V0,1.))
            physical_relative=float(abs(residual).sum()/scale)
            physical_mass=abs(float(self.area@h)-self.V0-source+export)/max(self.V0,abs(source),abs(export),1.)
            d=dict(local=local,mass=mass,physical_local=physical_relative,physical_mass=physical_mass,
                natural=nr,fixed_point=fixed_point,active_equal=bool(active_equal),
                source=source,export=export,unmet=float(unmet.sum()),potential=self.dt*float(self.area@self.smb),
                H_min=float(h.min()),active_count=int((h==0).sum()),
                merit=max(nr,local,mass),all_finite=bool(all(np.isfinite(x).all() for x in [h,u,raw,actual])))
            return d,T.tocsr(),raw,actual,unmet,fixed
    def ru(self,h,u,du):
        with self.w.span('Ru_action'):
            # Only Hd*dC; C*dHd already belongs to T, including outward diagonal.
            return decompose(self.adapter,h,u,np.zeros_like(h),du)['K_direction']*(self.dt/.25)

def accepted(d,bp):
    return (bp<=1e-6 and d['local']<=1e-8 and d['mass']<=1e-10 and d['physical_local']<=1e-8
            and d['physical_mass']<=1e-10 and d['fixed_point']<=1e-8 and d['active_equal']
            and d['all_finite'] and d['H_min']>=0)

def attempt(backend,adapter,h_old,u_old,smb):
    w=backend.w;mass=Mass(adapter,h_old,smb,w);bed=backend.mesh['bed'];history=[]
    with w.span('predictor'):
        h,diag,parts=predict(h_old,bed,smb,.25,adapter.rates(u_old))
    w.save('predictor',H=h,**parts);w.event('predictor',diagnostics=diag)
    u=u_old.copy();u[~np.repeat(wet(h,bed),4)]=0.
    u,info=backend.full(h,u)
    # Every line-search candidate is solved and re-evaluated at its OWN geometry.
    for outer in range(1,9):
        w.location='quarter residual '+str(outer)
        F,_,force,a=backend.evaluate(h,u);bp=float(np.linalg.norm(F)/max(np.linalg.norm(force),1.))
        d,T,raw,actual,unmet,fixed=mass.evaluate(h,u)
        row=dict(outer=outer,quarter_BP=w.counts['quarter_BP'],bp=bp,**d);history.append(row)
        w.event('quarter-residual',**row)
        qi,qo=flux(adapter,h,u)
        w.save(f'candidate-{outer:02d}',H=h,u=u,actual_smb=actual,unmet=unmet,residual=raw-unmet,F=F,
            internal_flux=qi,outlet_flux=qo,active=h==0,momentum_wet=a['wet'])
        if accepted(d,bp):
            w.save('accepted-1552.25',H=h,u=u,actual_smb=actual,unmet=unmet,internal_flux=qi,outlet_flux=qo)
            return dict(status='accepted',accepted=True,history=history)
        if bp>1e-6:raise LinearFailure('original BP residual fails after complete solve')
        if w.counts['quarter_BP']>=8:raise BudgetStop('8 quarter complete BP calls exhausted')
        linear=backend.linearize(h,u)
        # Primal-dual active branch; dry cells with net supply become free.
        active=h-raw/T.diagonal()<=0;free=~active
        w.event('linearization-branch',outer=outer,obstacle_active=int(active.sum()),
            newly_released_dry=np.flatnonzero((h==0)&free).tolist())
        physical_wet=linear.a['wet'][::4]
        correction=linear.solve(linear.F,'nonzero-F right-hand-side')
        rhs=-raw+mass.ru(h,u,correction)
        rhs=np.where(active,-h,rhs/mass.area)
        P=(sparse.diags(1./mass.area)@T).tolil();P[active,:]=0;P[active,active]=1.;P=P.tocsc()
        with w.span('T_preconditioner_build'):lu=splu(P)
        def pre(v):
            with w.span('T_preconditioner_action'):return lu.solve(v)
        def matvec(v):
            with w.span('J_action'):
                # A released dry coordinate is a nonsmooth predictor direction.
                # Its T part supplies ice; FH is restricted to the current wet branch.
                dh=np.where(physical_wet,v,0.)
                if np.any(dh):du=linear.solve(-linear.fh(dh),'K action');kv=mass.ru(h,u,du)
                else:kv=np.zeros_like(v)
                return np.where(active,v,(T@v+kv)/mass.area)
        op=LinearOperator(T.shape,matvec=matvec,dtype=float);preop=LinearOperator(T.shape,matvec=pre,dtype=float)
        def cb(r):w.count('outer_GMRES_iterations');w.event('outer-GMRES',outer=outer,preconditioned_relative=float(r))
        with w.span('outer_GMRES'):
            delta,code=gmres(op,rhs,M=preop,restart=8,maxiter=1,rtol=.02,atol=0,callback=cb,callback_type='pr_norm')
            rr=float(np.linalg.norm(matvec(delta)-rhs)/max(np.linalg.norm(rhs),1e-300))
        w.save(f'newton-{outer:02d}',deltaH=delta,rhs=rhs,obstacle_active=active)
        w.event('outer-linear-result',outer=outer,status=int(code),true_relative=rr)
        if not np.isfinite(delta).all() or rr>.2:raise LinearFailure('reduced GMRES true residual >0.2')
        masks=freeze_masks(adapter,h,u,bed);taken=False
        for backtrack in range(8):
            alpha=2.**(-backtrack);ht,hit=feasible(h,delta,alpha)
            guess=u.copy();guess[~np.repeat(wet(ht,bed),4)]=0.
            # Save feasible geometry BEFORE a potentially failing or budget-stopped solve.
            serial=w.counts['quarter_BP']+1
            w.save(f'trial-{serial:02d}',H=ht,u_guess=guess,projected_nodes=hit)
            w.event('trial',outer=outer,alpha=alpha,projected_nodes=np.flatnonzero(hit).tolist(),
                candidate_projection_volume=float(mass.area@(ht-(h+alpha*delta))),booked_as_SMB=False)
            ut,_=backend.full(ht,guess)
            dt,_,_,_,_,_=mass.evaluate(ht,ut);events=branch_events(adapter,ht,ut,bed,masks)
            w.event('branch-change',outer=outer,alpha=alpha,events=events,merit=dt['merit'])
            if dt['merit']<=(1-1e-4*alpha)*d['merit'] or dt['merit']<1e-10:
                h,u=ht,ut;taken=True;break
        if not taken:raise LinearFailure('projected Newton backtracking exhausted')
    raise BudgetStop('outer iteration limit')
