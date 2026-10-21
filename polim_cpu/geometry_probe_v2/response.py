"""Original shared-face fluxes, separate thickness/velocity derivatives and gates."""
import numpy as np
from polim_cpu.large_step_transport_v1 import ExistingMeshFluxAdapter


def overall_success(checks, protected, controller_ok, authorization_consistent):
    # An empty protection dictionary is NOT evidence of protection.
    return bool(checks and all(checks.values()) and protected and all(protected.values())
                and controller_ok and authorization_consistent)


def mean_velocity(u):
    v=np.asarray(u).reshape(-1,2,2)
    return .5*v[:,0,:]+.25*v[:,1,:]  # Original trapezoidal vertical integral, basal u=0.


def raw_boundary(adapter,u):
    """Signed pre-positive-part outlet rates, using the original edge quadrature."""
    v=mean_velocity(u);a=adapter.outer_a;b=adapter.outer_b;n=adapter.outer_normal
    return np.r_[np.einsum('ij,ij->i',.75*v[a]+.25*v[b],n),
                 np.einsum('ij,ij->i',.75*v[b]+.25*v[a],n)]


def flux(adapter,h,u):
    r=adapter.rates(u);d=r.donor_receiver_rate()[0]
    return r.internal_signed*h[d],r.boundary_outward*h[r.topology.boundary_donors]


def wet(h,bed):
    return h>64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(bed),abs(bed+h)))


def freeze_masks(adapter,h,u,bed):
    r=adapter.rates(u);c=r.internal_signed;bo=raw_boundary(adapter,u);w=wet(h,bed)
    floor=128*np.finfo(float).eps*max(float(abs(c).max(initial=0)),float(abs(bo).max(initial=0)),1.)
    scope=w[adapter.topology.internal_edges].any(axis=1);bscope=w[adapter.topology.boundary_donors]
    return dict(wet=w,internal_scope=scope,boundary_scope=bscope,
        internal=scope&(abs(c)>floor),boundary=bscope&(bo>floor),
        donor_sign=np.sign(c).astype(np.int8),outlet_sign=np.sign(bo).astype(np.int8),
        rate_floor=np.array(floor))


def branch_events(adapter,h,u,bed,masks):
    r=adapter.rates(u)
    a=wet(h,bed)!=masks['wet'];b=np.sign(r.internal_signed)!=masks['donor_sign']
    c=np.sign(raw_boundary(adapter,u))!=masks['outlet_sign']
    # The larger fixed scope also guards originally excluded zero-rate faces.
    return dict(wet_changed=np.flatnonzero(a).tolist(),donor_changed=np.flatnonzero(b).tolist(),
        outlet_changed=np.flatnonzero(c).tolist(),
        affects_evaluation=bool(a.any() or (b&masks['internal_scope']).any() or (c&masks['boundary_scope']).any()))


def divergence(adapter,q,bo):
    t=adapter.topology;r=np.zeros(t.size)
    np.add.at(r,t.internal_edges[:,0],q);np.add.at(r,t.internal_edges[:,1],-q)
    np.add.at(r,t.boundary_donors,bo);return r


def decompose(adapter,h,u,dh,du):
    r=adapter.rates(u);d=r.donor_receiver_rate()[0];bd=adapter.topology.boundary_donors
    dc=adapter.rates(du).internal_signed
    db=raw_boundary(adapter,du)*(raw_boundary(adapter,u)>0)
    th=r.internal_signed*dh[d];ve=h[d]*dc
    bt=r.boundary_outward*dh[bd];bv=h[bd]*db
    return dict(thickness_internal=th,velocity_internal=ve,total_internal=th+ve,
                thickness_boundary=bt,velocity_boundary=bv,total_boundary=bt+bv,
                K_direction=.25*divergence(adapter,ve,bv),
                paired_internal=np.column_stack((ve,-ve)))


def reference_parts(adapter,h,u,dh,up,um,epsilon):
    def cd(hp,upp,hm,umm):
        a,b=flux(adapter,hp,upp);c,d=flux(adapter,hm,umm)
        return (a-c)/(2*epsilon),(b-d)/(2*epsilon)
    full,fb=cd(h+epsilon*dh,up,h-epsilon*dh,um)
    vel,vb=cd(h,up,h,um)  # CRITICAL: same BASE H for both velocity-only references.
    thick,tb=cd(h+epsilon*dh,u,h-epsilon*dh,u)
    return dict(velocity=(up-um)/(2*epsilon),total_internal=full,total_boundary=fb,
        velocity_internal=vel,velocity_boundary=vb,thickness_internal=thick,thickness_boundary=tb,
        K_direction=.25*divergence(adapter,vel,vb))


def norm(x,weights,mask):
    x=np.asarray(x);w=np.asarray(weights)[mask];y=x[mask]
    if not len(w) or float(w.sum())==0:return 0.
    sq=y*y
    if sq.ndim>1:sq=sq.reshape(len(y),-1).sum(axis=1)
    return float(np.sqrt(np.sum(w*sq)/w.sum()))


def comparison(pred,ref,base,weights,mask,epsilon):
    denom=norm(ref,weights,mask)
    # Finite precision scale only; explicitly NOT nonlinear-solver uncertainty.
    floor=128*np.finfo(float).eps*max(norm(base,weights,mask),1.)/epsilon
    if not np.any(mask):return dict(status='inconclusive-empty-mask',relative=None,reference_norm=denom,roundoff_floor=floor)
    if denom<100*floor:return dict(status='inconclusive-small-response',relative=None,reference_norm=denom,roundoff_floor=floor)
    rel=norm(pred-ref,weights,mask)/denom
    return dict(status='pass' if rel<=.05 else 'response-disagreement',relative=rel,reference_norm=denom,roundoff_floor=floor)


def residual_screen(FH,F0,Fp,Fm,force_norms,epsilon):
    signal=epsilon*float(np.linalg.norm(FH))
    floor=128*np.finfo(float).eps*max(*force_norms,1.)
    noise=max(float(np.linalg.norm(F0)),float(np.linalg.norm(Fp)),float(np.linalg.norm(Fm)),floor)
    ratio=signal/noise
    return dict(status='screen-pass' if ratio>=100 else 'inconclusive-residual-noise',
                signal=signal,residual_noise_proxy=noise,ratio=ratio,threshold=100,
                velocity_error_bound=False)


def assess(adapter,h,u,dh,du,up,um,bed,masks,epsilon,FH,F0,Fp,Fm,force_norms):
    p=decompose(adapter,h,u,dh,du);r=reference_parts(adapter,h,u,dh,up,um,epsilon)
    events={name:branch_events(adapter,hx,ux,bed,masks) for name,hx,ux in
            [('plus',h+epsilon*dh,up),('minus',h-epsilon*dh,um)]}
    area=adapter.topology.dual_area;mi=masks['internal'];mb=masks['boundary'];mw=masks['wet']
    q,bq=flux(adapter,h,u);metrics={}
    # Four free velocity components, equal layer weights: RMS difference factor cancels in ratio.
    metrics['velocity_free']=comparison(du.reshape(-1,4),r['velocity'].reshape(-1,4),u.reshape(-1,4),area,mw,epsilon)
    metrics['velocity_depth_average']=comparison(mean_velocity(du),mean_velocity(r['velocity']),mean_velocity(u),area,mw,epsilon)
    metrics['velocity_surface']=comparison(du.reshape(-1,2,2)[:,1],r['velocity'].reshape(-1,2,2)[:,1],u.reshape(-1,2,2)[:,1],area,mw,epsilon)
    for part in ('thickness','velocity','total'):
        for side,mask,base in [('internal',mi,q),('boundary',mb,bq)]:
            key=part+'_'+side;metrics[key]=comparison(p[key],r[key],base,np.ones(len(base)),mask,epsilon)
    # K has nodal integrated-volume units; report area-normalized thickness response.
    metrics['K_direction']=comparison(p['K_direction']/area,r['K_direction']/area,h,area,mw,epsilon)
    screen=residual_screen(FH,F0,Fp,Fm,force_norms,epsilon)
    if any(x['affects_evaluation'] for x in events.values()):status='nonsmooth-fixed-scope'
    elif screen['status']!='screen-pass':status='inconclusive-residual-noise'
    elif any(x['status'].startswith('inconclusive') for x in metrics.values()):status='inconclusive-small-response'
    elif any(x['status']!='pass' for x in metrics.values()):status='response-disagreement'
    else:status='pass-at-this-base-and-direction-only'
    return dict(status=status,metrics=metrics,events=events,screen=screen),p,r
