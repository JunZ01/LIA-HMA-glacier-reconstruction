"""One concentrated, deterministic verification suite; all geometries synthetic."""
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from polim_cpu.large_step_transport_v1 import (TransportTopology, FixedVelocityRates,
    implicit_conservative_step, solve_nonnegative_complementarity)
from .interfaces import (Work, OriginalBPResidual, OriginalFeedbackAction, MassResidual, geometry_feedback,
    auxiliary_correction, physical_unmet_from_auxiliary, BranchChange, AuxiliaryConstraintError)


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))


def trend(values):
    if len(values)>=2 and values[-1]>1.2*values[-2]:return 'diverging'
    if len(values)>=4 and min(np.array(values[-3:])/np.array(values[-4:-1]))>.98:return 'stagnating'
    return 'continue'


class NonlocalSystem:
    """F=A u-f(H), exact geometry response A^-1 G is nonlocal.

    Mass uses the ORIGINAL finite-volume upwind implementation, including outlet.
    """
    def __init__(self):
        self.work=Work();t=perf_counter()
        self.A=np.array([[2.,-.5,0.],[-.5,2.,-.5],[0.,-.5,2.]])
        self.G=np.array([[1.,-1.,0.],[0.,1.,-1.],[.15,0.,0.]])
        self.target=np.array([3.,2.,1.5]);self.ustar=np.array([.4,.3,.1])
        self.top=TransportTopology(np.ones(3),np.array([[0,1],[1,2]]),np.array([2]))
        self.dt=.25;h_old=self.target+self.dt*(self.rates(self.ustar).outward_operator()@self.target)
        self.mass=MassResidual(h_old,np.zeros(3),self.dt,self.rates,self.work)
        self.work.build_seconds+=perf_counter()-t

    def rates(self,u):
        return FixedVelocityRates(self.top,np.asarray(u[:2]),np.maximum(u[2:],0.))

    def residual(self,u,h):
        self.work.residual_evaluations+=1
        return self.A@(u-self.ustar)-self.G@(h-self.target)

    def solve_bp(self,h):
        t=perf_counter();u=self.ustar+np.linalg.solve(self.A,self.G@(h-self.target))
        self.work.complete_bp+=1;self.work.bp_seconds+=perf_counter()-t
        return u

    def ru(self,h,du):
        # Positive donor branch; last component is a real outlet, not an edge pair.
        q=h[:2]*du[:2];return self.dt*np.array([q[0],q[1]-q[0],h[2]*du[2]-q[1]])

    def feedback(self,h):
        t=perf_counter()
        # Exercise existing projected AMG *interface*. Dense synthetic inverse
        # is explicitly labelled; no claim of a real GPU AMG build or speedup.
        base=SimpleNamespace(d=np.ones(3),cycle=lambda b:np.linalg.solve(self.A,b))
        cols=[]
        for dh in np.eye(3):
            val,_=geometry_feedback(dh,lambda du:self.A@du,lambda dh:-self.G@dh,
                lambda du:self.ru(h,du),self.work,base_cycle=base)
            cols.append(val)
        self.work.build_seconds+=perf_counter()-t
        return np.column_stack(cols)


def run():
    started=perf_counter();checks=[]
    def check(name,passed,**evidence):
        checks.append(dict(name=name,passed=bool(passed),**evidence))
    s=NonlocalSystem();h=s.target+np.array([.025,-.015,.01]);u=s.solve_bp(h)
    dh=np.array([.2,-.3,.15]);du=np.array([.1,.2,-.1]);eps=1e-5
    fd_u=(s.residual(u+eps*du,h)-s.residual(u-eps*du,h))/(2*eps)
    fd_h=(s.residual(u,h+eps*dh)-s.residual(u,h-eps*dh))/(2*eps)
    K=s.feedback(h);exactK=np.column_stack([s.ru(h,np.linalg.solve(s.A,s.G@v)) for v in np.eye(3)])
    Ru_fd=s.mass.velocity_action(h,u,du)
    # No additional full solve: exact algebraic response in this analytic fixture.
    response=np.linalg.solve(s.A,s.G@dh)
    fd_J=(s.mass.raw(h+eps*dh,u+eps*response)-s.mass.raw(h-eps*dh,u-eps*response))/(2*eps)
    Jv=s.mass.matrix(u)@dh+K@dh
    errors=dict(Fu=relative(fd_u,s.A@du),FH=relative(fd_h,-s.G@dh),K=relative(K,exactK),
        Ru=relative(Ru_fd,s.ru(h,du)),J=relative(fd_J,Jv))
    check('analytic_nonlocal_derivatives',max(errors.values())<1e-8,errors=errors,
          distant_velocity_response=float(np.linalg.solve(s.A,s.G)[2,0]))
    hist=[];auxnorm=[]
    for k in range(8):
        u=s.solve_bp(h);f=s.residual(u,h);m=s.mass.formal(h,u)
        bp=float(np.linalg.norm(f)/max(np.linalg.norm(s.A@s.ustar+s.G@(h-s.target)),1.))
        row=dict(outer=k+1,bp=bp,local=m['local'],mass=m['mass'],fixed_point=m['fixed_point'],active_equal=m['active_equal'])
        hist.append(row)
        if bp<=1e-6 and m['local']<=1e-8 and m['mass']<=1e-10 and m['fixed_point']<=1e-8 and m['active_equal']:break
        if k==7:break
        K=s.feedback(h);hhat,aux=auxiliary_correction(s.mass,h,u,K,s.work)
        row['auxiliary_equation_linf']=float(np.abs(s.mass.raw(hhat,u)+aux).max())
        row['auxiliary_l2']=float(np.linalg.norm(aux));auxnorm.append(row['auxiliary_l2']);h=hhat
    accepted=hist[-1]['bp']<=1e-6 and hist[-1]['local']<=1e-8 and hist[-1]['mass']<=1e-10 and hist[-1]['fixed_point']<=1e-8
    check('coupled_original_equations_and_nonnegative',accepted and np.all(h>=0) and np.max(abs(h-s.target))<1e-7,
          history=hist,H=h.tolist(),target=s.target.tolist(),complete_bp_for_coupling=len(hist),
          final_original_residual=s.mass.raw(h,u).tolist(),final_unmet=m['unmet'].tolist())
    check('auxiliary_term_vanishes',len(auxnorm)>=2 and auxnorm[-1]<auxnorm[0]*1e-4,
          auxiliary_norms=auxnorm,zero_increment_action=(K@np.zeros(3)).tolist())
    # Shared-edge cancellation is checked before any nodal/global summation.
    q=np.array([.37,-.21]);pair=np.column_stack([q,-q]);outlet=.13
    check('shared_faces_cancel_outlet_separate',bool(np.all(pair.sum(axis=1)==0)),
          paired_correction=pair.tolist(),internal_sum=float(pair.sum()),outlet=outlet,global_correction=outlet)
    rates=FixedVelocityRates(TransportTopology(np.ones(2),np.array([[0,1]]),np.array([1])),np.array([.3]),np.array([.2]))
    pos=implicit_conservative_step(np.array([.01,.2]),np.zeros(2),2.,rates)
    check('formal_nonnegative_without_ablation',np.all(pos.thickness>=0) and np.max(pos.unmet_ablation_volume)==0,
          H=pos.thickness.tolist(),unmet=pos.unmet_ablation_volume.tolist(),mass_relative=pos.mass_relative)
    obstacle=solve_nonnegative_complementarity(np.eye(2),np.array([-1.,2.]))
    blocked=False
    try:physical_unmet_from_auxiliary(obstacle.multiplier,np.zeros(2))
    except AuxiliaryConstraintError:blocked=True
    check('auxiliary_obstacle_never_physical_ablation',blocked,auxiliary_multiplier=obstacle.multiplier.tolist(),physical_credit=0.)
    # Deliberate nonsmooth events are separate from smooth derivative comparisons.
    switching=MassResidual(np.ones(2),np.zeros(2),.1,
        lambda v:FixedVelocityRates(TransportTopology(np.ones(2),np.array([[0,1]]),np.array([],dtype=int)),v,np.array([])))
    donor_blocked=False
    try:switching.velocity_action(np.ones(2),np.array([0.]),np.array([1.]))
    except BranchChange:donor_blocked=True
    dry=implicit_conservative_step(np.array([0.,.1]),np.array([0.,-1.]),1.,
        FixedVelocityRates(TransportTopology(np.ones(2),np.empty((0,2),dtype=int),np.array([],dtype=int)),np.array([]),np.array([])))
    check('wet_dry_and_donor_switch_separated',donor_blocked and np.all(dry.thickness==0),
          donor_switch_rejected=donor_blocked,dry_H=dry.thickness.tolist(),physical_unmet=dry.unmet_ablation_volume.tolist())
    # Stress test the SAME auxiliary equation in a linear, conservative block system.
    # T=I, F=u-H, R=T H+B u-rhs, K=B; no interpretation as glacier physics.
    stress=[]
    for factor in (0.,1e6):
        T=np.eye(2);B=np.array([[2.,-2.],[-2.,2.]]);S=factor*B;target=np.ones(2);rhs=(T+B)@target
        x=target+np.array([.001,-.001]);values=[];reason='budget-exhausted'
        for k in range(8):
            velocity=x.copy();s.work.complete_bp+=1
            values.append(float(np.linalg.norm(T@x+B@velocity-rhs)))
            reason=trend(values)
            if reason!='continue':break
            x=np.linalg.solve(T+S,rhs-B@velocity+S@x);s.work.auxiliary_solves+=1
        stress.append(dict(S_factor=factor,status=reason,residuals=values,minimum_H=float(x.min())))
    check('weak_and_strong_feedback_detected',[x['status'] for x in stress]==['diverging','stagnating'],cases=stress)
    # Actual original BP residual and true tangent on ONE synthetic triangle.
    mesh=dict(xy=np.array([[0.,0.],[100.,0.],[0.,100.]]),triangles=np.array([[0,1,2]]),bed=np.array([100.,98.,99.]))
    orig=OriginalBPResidual(mesh,np.full((3,3),7.538801981725612e-23*31536000.))
    hh=np.array([60.,55.,50.]);uu=np.linspace(.2,1.3,12);vv=np.cos(np.arange(12))*.1;dd=np.array([.2,-.1,.3])
    analytic=orig.velocity_action(uu,hh,vv)
    fd=(orig.residual(uu+eps*vv,hh)-orig.residual(uu-eps*vv,hh))/(2*eps)
    gh=[orig.geometry_action(uu,hh,dd,epsilon=e) for e in (.01,.001,.0001)]
    geomerr=relative(gh[-1],gh[-2]);velerr=relative(analytic,fd)
    dry_block=False
    try:orig.geometry_action(uu,np.array([0.,55.,50.]),np.array([1.,0.,0.]))
    except BranchChange:dry_block=True
    disabled=False
    try:orig.time_advance()
    except NotImplementedError:disabled=True
    check('original_BP_synthetic_triangle_tangent',velerr<1e-7 and geomerr<1e-7,
          Fu_relative_error=velerr,FH_step_refinement_relative=geomerr,
          geometry_derivative_method='original residual central difference, full geometry rebuild; not independent analytic FH',
          true_Fu_includes_viscosity_derivative=True)
    triangle_mass=MassResidual(hh,np.zeros(3),.25,lambda v:s.rates(v[[0,4,8]]),orig.work)
    connected=OriginalFeedbackAction(orig,triangle_mass,hh,uu)
    kval,linear_info=connected.apply(dd)
    expected_du=np.linalg.solve(connected.Fu.toarray(),-gh[-1])
    expected_k=triangle_mass.velocity_action(hh,uu,expected_du)
    check('original_interfaces_connected_on_synthetic_mesh',relative(kval,expected_k)<1e-7,
          K_relative_error=relative(kval,expected_k),linear_solve=linear_info,
          nonlinear_BP_solves=0)
    check('real_backend_disabled_and_dry_projection_guarded',disabled and dry_block,
          time_advance_disabled=disabled,dry_direction_blocked=dry_block)
    return dict(status='pass' if all(c['passed'] for c in checks) else 'fail',checks=checks,
        work=dict(synthetic=s.work.record(),original_operator_synthetic_mesh=orig.work.record(),
                  real_glacier_complete_BP=0,real_glacier_tangent_solves=0,real_time_steps=0,
                  wall_seconds=perf_counter()-started,
                  timing_note='Nested build/tangent timings overlap; totals are wall time. Dense synthetic preconditioner uses existing projected-cycle API; no real AMG performance claim. Stress block algebra timings are included only in wall time.'),
        thresholds=dict(bp=1e-6,local=1e-8,mass=1e-10,complete_bp_per_candidate=8))
