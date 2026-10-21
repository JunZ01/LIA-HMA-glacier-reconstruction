"""NEW tests only: protection gate, split flux derivatives, screening/branches."""
import numpy as np
from .response import (overall_success,ExistingMeshFluxAdapter,decompose,reference_parts,
    flux,mean_velocity,raw_boundary,freeze_masks,branch_events,comparison,residual_screen)
from .common import authorization_allowed


def run():
    checks=[]
    def add(name,ok,**info):checks.append(dict(name=name,passed=bool(ok),**info))
    protected={'main':True,'physics':True,'v41':True}
    injected={k:overall_success({'synthetic':True},{**protected,k:False},True,True) for k in protected}
    add('protection_false_forces_failure',overall_success({'synthetic':True},protected,True,True)
        and not any(injected.values()),injected_statuses=injected,real_files_modified=False)
    # Three-node quadrature, including reverse donor and both outlet integration pieces.
    z=dict(geo_dual_area=np.array([2.,3.,4.]),geo_edges=np.array([[0,1],[1,2]]),
        geo_outer_a=np.array([0]),geo_outer_b=np.array([2]),geo_a=np.array([0,1]),geo_b=np.array([1,2]),
        geo_third=np.array([2,0]),geo_inverse=np.array([0,1]),geo_normal=np.array([[2.,0.],[-3.,0.]]),
        geo_outer_normal=np.array([[1.5,0.]]))
    adapter=ExistingMeshFluxAdapter(z);h=np.array([10.,20.,30.]);dh=np.array([.3,.2,.4])
    u=np.array([[[2.,.2],[4.,.4]],[[3.,.1],[5.,.2]],[[4.,.3],[6.,.6]]]).reshape(-1)
    du=np.array([[[.4,.03],[.6,.08]],[[.5,.04],[.8,.1]],[[.7,.06],[1.,.12]]]).reshape(-1)
    epsilon=.01;parts=decompose(adapter,h,u,dh,du)
    refs=reference_parts(adapter,h,u,dh,u+epsilon*du,u-epsilon*du,epsilon)
    errors={k:float(np.max(abs(parts[k]-refs[k]))) for k in parts if k in refs}
    nonzero=all(np.all(abs(parts[k])>1e-6) for k in ['thickness_internal','velocity_internal','thickness_boundary','velocity_boundary'])
    omitted={side:dict(omit_thickness=float(np.linalg.norm(parts['velocity_'+side]-refs['total_'+side])),
        omit_velocity=float(np.linalg.norm(parts['thickness_'+side]-refs['total_'+side]))) for side in ['internal','boundary']}
    add('both_flux_terms_and_sum',max(errors.values())<1e-9 and nonzero and all(min(x.values())>.01 for x in omitted.values()),
        errors=errors,omission_errors=omitted,thickness_internal=parts['thickness_internal'].tolist(),
        velocity_internal=parts['velocity_internal'].tolist(),thickness_boundary=parts['thickness_boundary'].tolist(),
        velocity_boundary=parts['velocity_boundary'].tolist())
    add('velocity_reference_uses_base_H',np.linalg.norm(refs['total_internal']-refs['velocity_internal'])>.01,
        full_minus_velocity= (refs['total_internal']-refs['velocity_internal']).tolist())
    mean=mean_velocity(u);wrong=u.reshape(-1,2,2)[:,1]
    add('original_mean_and_outlet_quadrature',np.allclose(adapter.rates(u).boundary_outward,np.maximum(raw_boundary(adapter,u),0.))
        and np.linalg.norm(mean-wrong)>1.,mean_velocity=mean.tolist())
    internal=parts['paired_internal'];outlet=.25*parts['velocity_boundary'].sum()
    add('feedback_conservation_and_outlet',np.all(internal.sum(1)==0) and abs(parts['K_direction'].sum()-outlet)<1e-12,
        K_sum=float(parts['K_direction'].sum()),outlet_feedback=float(outlet))
    masks=freeze_masks(adapter,h,u,np.zeros(3));events=branch_events(adapter,h,-u,np.zeros(3),masks)
    add('fixed_scope_switch_detected',events['affects_evaluation'],events=events)
    small=comparison(np.ones(3)*1e-30,np.zeros(3),np.ones(3),np.ones(3),np.ones(3,dtype=bool),epsilon)
    low=residual_screen(np.ones(3),np.ones(3)*.01,np.ones(3)*.01,np.ones(3)*.01,[1.,1.,1.],epsilon)
    high=residual_screen(np.ones(3),np.ones(3)*1e-8,np.ones(3)*1e-8,np.ones(3)*1e-8,[1.,1.,1.],epsilon)
    add('small_signal_inconclusive_not_pass_or_fail',small['status'].startswith('inconclusive') and low['status'].startswith('inconclusive')
        and high['status']=='screen-pass',nearzero=small,noise=low,velocity_error_bound=False)
    add('authorization_closed_rejected',not authorization_allowed({}, {}, 'x','y'),backend_imported=False)
    return dict(status='pass' if all(x['passed'] for x in checks) else 'fail',checks=checks,
                full_BP=0,real_tangent=0,time_steps=0,original_v41_tests_reexecuted=False)
