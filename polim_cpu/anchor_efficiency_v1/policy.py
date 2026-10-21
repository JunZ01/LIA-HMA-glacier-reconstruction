"""Pure, frozen anchor lifetime decisions; no model evaluation or solves."""
import numpy as np
from polim_cpu.geometry_probe_v2.response import raw_boundary

def branches(adapter,s,alpha,beta):
    rates=adapter.rates(s['u']);path=s['e']['path']
    # Original CN reflection-envelope derivative classes: inactive, endpoint,
    # interior. Continuous theta/dry interval endpoints are not identity keys.
    category=np.where(path['ell']<=0,0,np.where(path['theta_star']==1,1,2)).astype(np.int8)
    return dict(wet=s['wet'].copy(),alpha=alpha.copy(),beta=beta.copy(),revision=s['revision'],
                donor=np.sign(rates.internal_signed).astype(np.int8),outlet=np.sign(raw_boundary(adapter,s['u'])).astype(np.int8),reflection=category)

def refresh_reasons(old,new,model=None):
    if old is None:return ['no-anchor']
    why=[]
    for k in ('wet','donor','outlet','reflection'):
        if not np.array_equal(old[k],new[k]):why.append(k+'-branch')
    if old['revision']!=new['revision'] or not np.array_equal(old['alpha'],new['alpha']) or not np.array_equal(old['beta'],new['beta']):why.append('formal-discrete-revision')
    if model is not None:
        if not model['valid']:why.append('model-invalid')
        elif model['rho']<.25:why.append('rho-below-0.25')
    return why

def model_quality(G,trial,Js,sr,same_equation):
    pred=G+Js;defect=trial-pred
    phi0=.5*float(np.linalg.norm(sr*G)**2);phip=.5*float(np.linalg.norm(sr*pred)**2);phit=.5*float(np.linalg.norm(sr*trial)**2)
    predicted=phi0-phip;actual=phi0-phit;floor=100*np.finfo(np.float64).eps*max(phi0,1.)
    valid=bool(same_equation and np.isfinite([phi0,phip,phit,predicted,actual]).all() and predicted>floor)
    rho=actual/predicted if valid else None
    if rho is not None and not np.isfinite(rho):valid=False;rho=None
    def safe(x):return float(x) if np.isfinite(x) else None
    return dict(valid=valid,rho=rho,phi_current=safe(phi0),phi_predicted=safe(phip),phi_trial=safe(phit),predicted_reduction=safe(predicted),actual_reduction=safe(actual),positive_floor=safe(floor),same_equation=bool(same_equation)),pred,defect

def may_retry(anchor_point,parent_point,already_refreshed):
    return anchor_point!=parent_point and not already_refreshed
