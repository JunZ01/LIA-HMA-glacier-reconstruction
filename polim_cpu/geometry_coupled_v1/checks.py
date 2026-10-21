"""Only NEW implementation checks; no glacier BP or tangent solves."""
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import gmres,LinearOperator
from polim_cpu.v13r1 import bp
from polim_cpu.large_step_transport_v1 import TransportTopology,FixedVelocityRates,implicit_conservative_step
from .geometry import action
from .coupling import physical_source,feasible

def run():
    start=time.perf_counter();checks={};metrics={}
    xy=np.array([[0.,0.],[50.,0.],[50.,50.],[0.,50.]])
    tri=np.array([[0,1,2],[0,2,3]]);h=np.array([100.,110.,95.,80.]);bed=np.array([10.,8.,5.,7.])
    dh=np.array([.7,-.3,.4,.2]);u=np.arange(16,dtype=float)*.03+.1;du=np.sin(np.arange(16))*.02
    mesh=dict(xy=xy,triangles=tri,bed=bed);A=bp.A_SI*bp.YEAR*np.ones((4,3));eps=1e-3
    a=bp.prepare({**mesh,'thickness':h});F,_,Fu,_=bp.evaluate(a,u,True,A=A)
    def residual(hx,ux):return bp.evaluate(bp.prepare({**mesh,'thickness':hx}),ux,A=A)[0]
    fh=action(a,u,dh,A);fd=(residual(h+eps*dh,u)-residual(h-eps*dh,u))/(2*eps)
    metrics['FH_relative']=float(np.linalg.norm(fh-fd)/np.linalg.norm(fd));checks['analytic_FH_mapping_weights_drive_viscosity']=metrics['FH_relative']<2e-7
    fdu=(residual(h,u+eps*du)-residual(h,u-eps*du))/(2*eps)
    metrics['Fu_relative']=float(np.linalg.norm(Fu@du-fdu)/np.linalg.norm(fdu));checks['true_Fu']=metrics['Fu_relative']<2e-7
    # Nonlocal response and nonzero momentum residual: verify reduced RHS sign.
    B=np.array([[.2,.1,0.],[-.1,.3,.2],[.05,-.1,.1]])
    G=np.array([[1.,-.2,.1],[-1.,.2,.4],[0.,0.,.3]])
    T=np.array([[3.,-.2,0.],[-.3,2.,-.1],[0.,-.1,2.]])
    U=np.array([[2.,.1,0.],[.1,2.,.1],[0.,.1,1.]])
    H=np.array([1.,2.,3.]);v=np.array([.4,.8,.9]);c=np.array([.1,.2,.3]);b=np.array([3.,4.,5.])
    F=U@v-B@H-c;R=T@H+G@v-b;K=G@np.linalg.solve(U,B);rhs=-R+G@np.linalg.solve(U,F)
    op=LinearOperator(T.shape,matvec=lambda x:T@x+G@np.linalg.solve(U,B@x),dtype=float)
    x,info=gmres(op,rhs,rtol=1e-12,atol=0)
    dv=np.linalg.solve(U,-F+B@x)
    checks['nonzero_F_rhs_and_nonlocal_matrix_free_GMRES']=bool(info==0 and np.linalg.norm(T@(H+x)+G@(v+dv)-b)<1e-12 and np.linalg.norm(U@(v+dv)-B@(H+x)-c)<1e-12)
    wrong=np.linalg.solve(T+K,-R);checks['omitted_nonzero_F_detectable']=bool(np.linalg.norm(wrong-x)>1e-3)
    top=TransportTopology(np.ones(2),np.array([[0,1]]),np.array([1]));rates=FixedVelocityRates(top,np.array([2.]),np.array([.3]))
    step=implicit_conservative_step(np.array([1.,0.]),np.zeros(2),.25,rates)
    checks['dry_receives_ice']=bool(step.thickness[1]>0)
    checks['shared_flux_cancels_outlet_separate']=bool(abs(step.thickness.sum()-1+step.boundary_export_volume)<1e-14)
    actual,unmet=physical_source(np.zeros(2),np.array([1.,-1.]),np.zeros(2),np.ones(2),.25)
    checks['no_SMB_no_artificial_ablation']=bool(not unmet.any() and not actual.any())
    actual,unmet=physical_source(np.zeros(2),np.array([1.,-1.]),np.array([-2.,-2.]),np.ones(2),.25)
    checks['ablation_capped_by_negative_SMB']=bool(np.array_equal(unmet,[.5,0.]) and np.array_equal(actual,[0.,-2.]))
    candidate,hit=feasible(np.array([0.,1.]),np.array([.3,-2.]),1.)
    checks['feasible_transition_not_mass_source']=bool(np.array_equal(candidate,[.3,0.]) and np.array_equal(hit,[False,True]))
    hd=2.;C=3.;dhd=.4;dC=-.2;dt=C*dhd;dv=hd*dC
    ref=((C+eps*dC)*(hd+eps*dhd)-(C-eps*dC)*(hd-eps*dhd))/(2*eps)
    vel=(hd*(C+eps*dC)-hd*(C-eps*dC))/(2*eps)
    checks['two_nonzero_flux_terms']=bool(abs(ref-dt-dv)<1e-10 and abs(vel-dv)<1e-10 and abs(ref-dt)>.1 and abs(ref-dv)>.1)
    # Sparse active row assembly and release are the SAME code idiom as the real driver.
    P=sparse.csr_matrix(T).tolil();active=np.array([True,False,False]);P[active,:]=0;P[active,active]=1.
    checks['active_identity_rows']=bool(np.array_equal(P.toarray()[0],[1.,0.,0.]))
    return dict(status='pass' if all(checks.values()) else 'fail',checks=checks,metrics=metrics,
        seconds=time.perf_counter()-start,real_full_BP=0,real_tangent=0,real_time_steps=0,
        reused_v41_v42_tests_not_rerun=True)
