"""Flux-anchored local diffusion predictor; never an accepted physical update.

For each oriented face Q0=C*H_donor, g=abs(Q0)/max(abs(ds0),delta_s).
Bare diffusion g*ds0 is not Q0 on uphill or denominator-limited faces.
The frozen signed remainder r0=Q0-g*ds0 preserves that distinction:
    Q_predict = g*(s_predict_i-s_predict_j) + r0.
This equals Q0 only at the OLD geometry. It is not an exact new BP flux.
Outward boundary rates use the original C_out*H_predict donor diagonal.
The same nonnegative obstacle solver is used for an initial guess only.
"""
import numpy as np
from scipy import sparse
from .large_step_transport_v1 import _float_vector, solve_nonnegative_complementarity


def diffusion_conductance_from_flux(thickness, bed, rates, *, surface_difference_floor_m=1.):
    n=rates.topology.size
    h=_float_vector(thickness,'thickness',n);b=_float_vector(bed,'bed',n)
    floor=float(surface_difference_floor_m)
    if np.any(h<0) or not np.isfinite(floor) or floor<=0:raise ValueError('Invalid diffusion input')
    donor,_,_=rates.donor_receiver_rate();edges=rates.topology.internal_edges
    drop=(b+h)[edges[:,0]]-(b+h)[edges[:,1]]
    return np.abs(rates.internal_signed)*h[donor]/np.maximum(np.abs(drop),floor)


def predict(thickness,bed,smb,dt_years,rates,*,surface_difference_floor_m=1.):
    n=rates.topology.size;area=rates.topology.dual_area;edges=rates.topology.internal_edges
    h=_float_vector(thickness,'thickness',n);b=_float_vector(bed,'bed',n);m=_float_vector(smb,'smb',n)
    dt=float(dt_years)
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid prediction dt')
    g=diffusion_conductance_from_flux(h,b,rates,surface_difference_floor_m=surface_difference_floor_m)
    donor,_,_=rates.donor_receiver_rate();q0=rates.internal_signed*h[donor]
    ds=(b+h)[edges[:,0]]-(b+h)[edges[:,1]];remainder=q0-g*ds
    rows=np.r_[edges[:,0],edges[:,1],edges[:,0],edges[:,1]]
    cols=np.r_[edges[:,0],edges[:,1],edges[:,1],edges[:,0]]
    lap=sparse.coo_matrix((np.r_[g,g,-g,-g],(rows,cols)),shape=(n,n)).tocsr()
    div_r=np.zeros(n);np.add.at(div_r,edges[:,0],remainder);np.add.at(div_r,edges[:,1],-remainder)
    out_rate=np.bincount(rates.topology.boundary_donors,weights=rates.boundary_outward,minlength=n)
    matrix=sparse.diags(area+dt*out_rate)+dt*lap
    rhs=area*(h+dt*m)-dt*(lap@b+div_r)
    solution=solve_nonnegative_complementarity(matrix,rhs)
    predicted=solution.value
    if not np.isfinite(predicted).all() or np.any(predicted<0):raise RuntimeError('Invalid predictor output')
    new_ds=(b+predicted)[edges[:,0]]-(b+predicted)[edges[:,1]]
    q_pred=g*new_ds+remainder
    boundary=rates.boundary_outward*predicted[rates.topology.boundary_donors]
    q_anchor=g*ds+remainder
    active=np.abs(q0)>0
    diagnostics=dict(edges=len(g),old_nonzero_flux_edges=int(active.sum()),
        bare_diffusion_opposes_old_flux=int(np.sum((q0*ds<0)&active)),
        denominator_limited_nonzero_faces=int(np.sum((np.abs(ds)<surface_difference_floor_m)&active)),
        flat_nonzero_faces=int(np.sum((ds==0)&active)),
        anchored_old_flux_max_error=float(np.abs(q_anchor-q0).max(initial=0.)),
        predicted_flux_direction_reversals=int(np.sum(q_pred*q0<0)),
        predicted_boundary_export=float(dt*boundary.sum()),
        predictor_obstacle_multiplier_sum=float(solution.multiplier.sum()),
        predicted_H_min=float(predicted.min()),predicted_H_max=float(predicted.max()),
        notes='g alone is limited/downhill diffusion. Signed remainder anchors only OLD flux. New predicted flux may change sign as geometry changes; report reversals. Predictor obstacle is not physical unmet-ablation bookkeeping; final unchanged transport complementarity defines the physical source.')
    return predicted,diagnostics,dict(conductance=g,signed_remainder=remainder,old_flux=q0,
        bare_old_diffusion_flux=g*ds,anchored_old_flux=q_anchor,predicted_flux=q_pred,predicted_boundary_flux=boundary)
