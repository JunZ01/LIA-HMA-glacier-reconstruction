"""Projected damped reduced Newton; original BE/upwind equations decide acceptance."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres,LinearOperator,splu
from polim_cpu.large_step_transport_v1 import implicit_conservative_step
from polim_cpu.diffusion_predictor_v2 import predict
from polim_cpu.geometry_probe_v2.response import decompose,wet,freeze_masks,branch_events,flux
from polim_cpu.geometry_coupled_v1.backend import LinearFailure,BudgetStop
from polim_cpu.geometry_coupled_v1.coupling import Mass,accepted,feasible
from .velocity_candidate import VelocityCandidate

def attempt(backend,adapter,h_old,u_old,smb):
    w=backend.w;mass=Mass(adapter,h_old,smb,w);bed=backend.mesh['bed'];history=[]
    with w.span('predictor'):
        h,diag,parts=predict(h_old,bed,smb,.25,adapter.rates(u_old))
    w.save('predictor',H=h,**parts);w.event('predictor',diagnostics=diag)
    u=u_old.copy();u[~np.repeat(wet(h,bed),4)]=0.
    initial_guess=u.copy()
    w.save('initial-guess',H=h,u_guess=u,deltaH=h-h_old,delta_u_predicted=u-u_old)
    u,info=backend.full(h,u);cached_evaluation=None
    # Every line-search candidate is solved and re-evaluated at its OWN geometry.
    for outer in range(1,9):
        w.location='quarter residual '+str(outer)
        F,_,force,a=backend.evaluate(h,u) if cached_evaluation is None else cached_evaluation
        cached_evaluation=None;bp=float(np.linalg.norm(F)/max(np.linalg.norm(force),1.))
        d,T,raw,actual,unmet,fixed=mass.evaluate(h,u)
        row=dict(outer=outer,quarter_BP=w.counts['quarter_BP'],bp=bp,**d);history.append(row)
        w.event('quarter-residual',**row)
        if outer==1:
            w.event('candidate-result',quarter_BP=1,alpha=None,origin='diffusion-predictor',
                guess_changed_from_old=bool(not np.array_equal(initial_guess,u_old)),
                returned_changed_from_guess=bool(not np.array_equal(u,initial_guess)),
                BP_Newton=info['iterations'],BP_original=bp,acceptance=bool(accepted(d,bp)),**d)
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
        velocity_candidate=VelocityCandidate(linear,correction,active,bed)
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
                else:du=np.zeros_like(u);kv=np.zeros_like(v)
                velocity_candidate.capture(v,du)
                return np.where(active,v,(T@v+kv)/mass.area)
        op=LinearOperator(T.shape,matvec=matvec,dtype=float);preop=LinearOperator(T.shape,matvec=pre,dtype=float)
        def cb(r):w.count('outer_GMRES_iterations');w.event('outer-GMRES',outer=outer,preconditioned_relative=float(r))
        with w.span('outer_GMRES'):
            delta,code=gmres(op,rhs,M=preop,restart=8,maxiter=1,rtol=.02,atol=0,callback=cb,callback_type='pr_norm')
            rr=float(np.linalg.norm(matvec(delta)-rhs)/max(np.linalg.norm(rhs),1e-300))
        # The final explicit matvec(delta) above is bound to THIS point/direction.
        w.save(f'newton-{outer:02d}',deltaH=delta,rhs=rhs,obstacle_active=active,
            c=correction,z=velocity_candidate.z,delta_u_full=velocity_candidate.z-correction)
        w.event('outer-linear-result',outer=outer,status=int(code),true_relative=rr)
        if not np.isfinite(delta).all() or rr>.2:raise LinearFailure('reduced GMRES true residual >0.2')
        masks=freeze_masks(adapter,h,u,bed);taken=False
        for backtrack in range(8):
            alpha=2.**(-backtrack);ht,hit=feasible(h,delta,alpha)
            with w.span('velocity_candidate'):
                guess,actual_displacement,predicted_velocity,cache_info=velocity_candidate.propose(ht,delta,alpha)
                w.count('velocity_cache_hit' if cache_info['cache_hit'] else 'velocity_extra_tangent')
            # Save feasible geometry BEFORE a potentially failing or budget-stopped solve.
            serial=w.counts['quarter_BP']+1
            w.save(f'trial-{serial:02d}',H=ht,u_guess=guess,projected_nodes=hit,
                deltaH_actual=actual_displacement,delta_u_predicted=predicted_velocity,
                delta_u_applied=guess-u,alpha=np.array(alpha),H_linearization=h,u_linearization=u)
            w.event('trial',outer=outer,alpha=alpha,projected_nodes=np.flatnonzero(hit).tolist(),
                candidate_projection_volume=float(mass.area@(ht-(h+alpha*delta))),booked_as_SMB=False,velocity=cache_info)
            ut,trial_info=backend.full(ht,guess)
            evaluation=backend.evaluate(ht,ut)
            trial_bp=float(np.linalg.norm(evaluation[0])/max(np.linalg.norm(evaluation[2]),1.))
            dt,_,_,_,_,_=mass.evaluate(ht,ut);events=branch_events(adapter,ht,ut,bed,masks)
            w.event('branch-change',outer=outer,alpha=alpha,events=events,merit=dt['merit'])
            w.save(f'trial-result-{serial:02d}',H=ht,u=ut,F=evaluation[0],force=evaluation[2])
            w.event('candidate-result',quarter_BP=serial,outer=outer,alpha=alpha,
                guess_changed_from_old=cache_info['guess_changed_from_old'],
                returned_changed_from_guess=bool(not np.array_equal(ut,guess)),
                returned_minus_guess_l2=float(np.linalg.norm(ut-guess)),
                returned_changed_from_old=bool(not np.array_equal(ut,u)),
                BP_Newton=trial_info['iterations'],BP_original=trial_bp,
                acceptance=bool(accepted(dt,trial_bp)),**dt)
            if dt['merit']<=(1-1e-4*alpha)*d['merit'] or dt['merit']<1e-10:
                h,u=ht,ut;cached_evaluation=evaluation;taken=True;break
        if not taken:raise LinearFailure('projected Newton backtracking exhausted')
    raise BudgetStop('outer iteration limit')
