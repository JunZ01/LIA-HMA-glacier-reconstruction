"""Joint original BP / reflected CN Newton. No nested complete BP solve."""
import numpy as np
from scipy.sparse.linalg import gmres,LinearOperator,splu
from polim_cpu.cn_cuda_v1.client import Backend as OriginalBackend
from polim_cpu.cn_sparse_v1.mass import Mass as OriginalMass,accepted
from polim_cpu.cn_sparse_v1.assembly import build_P
from polim_cpu.diffusion_predictor_v2 import predict
from polim_cpu.geometry_coupled_v1.coupling import feasible
from polim_cpu.geometry_coupled_v1.backend import BudgetStop,LinearFailure
from polim_cpu.geometry_coupled_v2.velocity_candidate import VelocityCandidate
from polim_cpu.geometry_probe_v2.response import wet,freeze_masks,branch_events
from polim_cpu.geometry_window_v1.ledger import preview
from polim_cpu.cn_flux_time_v1.transport import TransportFailure

class Backend(OriginalBackend):
    def full(self,*a,**kw):raise RuntimeError('v51 independent complete BP/rescue prohibited')

class Mass(OriginalMass):
    def __init__(self,*a,inherited=None,**kw):super().__init__(*a,**kw);self.inherited=inherited
    def evaluate(self,h,u):
        r=super().evaluate(h,u);d=r[0]
        if self.inherited is not None:
            other=preview(self.inherited,float(self.area@h),d['source'],d['export'])
            d['annual_cumulative_mass_relative']=d['cumulative_mass_relative']
            d['inherited_cumulative_mass_relative']=other['cumulative_mass_relative']
            d['cumulative_mass_relative']=max(d['cumulative_mass_relative'],other['cumulative_mass_relative'])
        return r

def merit(d,bp):
    return max(bp/1e-6,d['local']/1e-8,d['mass']/1e-10,d['cumulative_mass_relative']/1e-10,d['fixed_point']/1e-8)

def event_class(e):return np.where(e['path']['ell']<=0,0,np.where(e['path']['theta_star']==1,1,2))

def attempt(backend,adapter,h_old,u_old,smb,*,end_year,ledger,step_dt,inherited=None):
    w=backend.w;bed=backend.mesh['bed'];mass=Mass(adapter,h_old,u_old,smb,w,ledger=ledger,dt=step_dt,backend=backend,inherited=inherited)
    history=[];serial=0
    with w.span('predictor'):h,diag,parts=predict(h_old,bed,smb,step_dt,adapter.rates(u_old))
    u=u_old.copy();u[~np.repeat(wet(h,bed),4)]=0.
    w.save('predictor',H=h,u=u,**parts);w.event('predictor',diagnostics=diag)
    # Eight updates permit nine residual evaluations, including the final one.
    for update in range(9):
        w.location='joint residual '+str(update)
        F,_,force,projection=backend.evaluate(h,u)
        bp=float(np.linalg.norm(F)/max(np.linalg.norm(force),1.))
        d,T,R,actual,unmet,fixed=mass.evaluate(h,u)
        row=dict(joint_updates=update,bp=bp,joint_merit=merit(d,bp),**d);history.append(row)
        mass.save(f'candidate-{update:02d}',h,u);w.save(f'momentum-{update:02d}',H=h,u=u,F=F,force=force,wet=projection['wet'])
        w.event('joint-residual',**row)
        if accepted(d,bp):
            revision=mass.eq.revision;final,*_=mass.evaluate(h,u)
            if revision==mass.eq.revision and accepted(final,bp):
                mass.save('accepted-state',h,u)
                w.save('accepted-momentum',H=h,u=u,F=F,force=force,wet=projection['wet'])
                return dict(status='accepted',accepted=True,history=history,final=final)
            d,T,R,actual,unmet,fixed=mass.evaluate(h,u)
        if update==8:raise BudgetStop('8 joint Newton updates exhausted')
        w.count('joint_Newton_updates',8)
        linear=backend.linearize(h,u)
        active=h-R/T.diagonal()<=0;physical_wet=linear.a['wet'][::4]
        w.event('linearization-branch',update=update+1,point=linear.point,obstacle_active=int(active.sum()),newly_released_dry=np.flatnonzero((h==0)&~active).tolist())
        # c=Fu^-1 F; all momentum correction and feedback solves use the same
        # original 1e-8 true residual / zero initial tangent path and 64 cap.
        c=linear.solve(linear.F,'joint nonzero-F correction')
        vc=VelocityCandidate(linear,c,active,bed)
        with w.span('rhs_Ru'):rhs=-R+mass.ru(h,u,c)
        rhs=np.where(active,-h,rhs/mass.area)
        P=build_P(T,mass.area,active,w,'new')
        with w.span('P_LU_build'):lu=splu(P)
        def pre(v):
            with w.span('P_LU_apply'):return lu.solve(v)
        weights=mass.eq.weights(mass.last);first=[True]
        def action(v):
            with w.span('J_action'):
                dh=np.where(physical_wet,v,0.)
                if np.any(dh):
                    if first[0]:
                        with w.span('first_K_action'):du,kv=linear.feedback(dh,weights)
                        first[0]=False
                    else:du,kv=linear.feedback(dh,weights)
                else:du=np.zeros_like(u);kv=np.zeros_like(v)
                vc.capture(v,du)
                return np.where(active,v,(T@v+kv)/mass.area)
        op=LinearOperator(T.shape,matvec=action,dtype=float);preop=LinearOperator(T.shape,matvec=pre,dtype=float)
        def cb(r):w.count('outer_GMRES_iterations');w.event('joint-GMRES',update=update+1,preconditioned_relative=float(r))
        with w.span('outer_GMRES'):
            delta,code=gmres(op,rhs,M=preop,restart=8,maxiter=1,rtol=.02,atol=0,callback=cb,callback_type='pr_norm')
            rr=float(np.linalg.norm(action(delta)-rhs)/max(np.linalg.norm(rhs),1e-300))
        w.save(f'newton-{update+1:02d}',deltaH=delta,rhs=rhs,c=c,z=vc.z,delta_u=vc.z-c,obstacle_active=active)
        w.event('joint-linear-result',update=update+1,status=int(code),true_relative=rr)
        if not np.isfinite(delta).all() or rr>.2:raise LinearFailure('Reduced GMRES true residual >0.2')
        masks=freeze_masks(adapter,h,u,bed);taken=False
        for backtrack in range(8):
            alpha=2.**(-backtrack);ht,hit=feasible(h,delta,alpha);serial+=1;w.count('joint_line_candidates')
            with w.span('velocity_candidate'):ut,actual_shift,predicted,cache=vc.propose(ht,delta,alpha)
            w.count('velocity_cache_hit' if cache['cache_hit'] else 'velocity_extra_tangent')
            w.save(f'trial-{serial:02d}',H=ht,u=ut,deltaH=actual_shift,delta_u=predicted,alpha=np.array(alpha),projected_nodes=hit)
            # No equilibrium BP call here: the paired trial is evaluated as-is.
            Ft,_,ft,at=backend.evaluate(ht,ut);bpt=float(np.linalg.norm(Ft)/max(np.linalg.norm(ft),1.))
            w.save(f'trial-momentum-{serial:02d}',H=ht,u=ut,F=Ft,force=ft,wet=at['wet'])
            revision=mass.eq.revision;old_events=event_class(mass.last)
            try:dt,*_=mass.evaluate(ht,ut)
            except TransportFailure as exc:
                w.event('joint-trial-transport-rejected',serial=serial,alpha=alpha,error=str(exc))
                continue
            new_events=event_class(mass.last)
            candidate_merit=merit(dt,bpt);candidate_revision=mass.eq.revision
            mass.save(f'trial-mass-{serial:02d}',ht,ut)
            # If the monotone limiter changed, compare both physical states
            # under the NEW shared coefficients; never compare two equations.
            base_d=d
            same_equation=True
            if candidate_revision!=revision:
                base_d,*_=mass.evaluate(h,u)
                if mass.eq.revision!=candidate_revision:
                    same_equation=False
                w.event('limiter-base-reevaluation',old_revision=revision,new_revision=mass.eq.revision)
                d=base_d
            branch=branch_events(adapter,ht,ut,bed,masks)
            good=same_equation and dt['physical_path'] and (candidate_merit<=(1-1e-4*alpha)*merit(base_d,bp) or accepted(dt,bpt))
            w.event('joint-trial-result',serial=serial,update=update+1,alpha=alpha,bp=bpt,joint_merit=candidate_merit,base_merit=merit(base_d,bp),
                accepted_line=bool(good),branch=branch,event_class_changed=bool(not np.array_equal(old_events,new_events)),same_equation=same_equation,
                projection_booked_as_SMB=False,velocity=cache,**dt)
            if good:h,u=ht,ut;taken=True;break
        if not taken:raise LinearFailure('8 paired joint backtracking candidates exhausted')
    raise AssertionError('Unreachable joint update boundary')
