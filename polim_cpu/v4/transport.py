"""Simultaneous conservative volume and specific-enthalpy transport on a graph.

Flux is supplied in m3/year, volume in m3, enthalpy in J/kg. No velocity,
temperature, geometry or climatic forcing is inferred by this operator.
All layers share the volume flux and preserve their declared mass fractions.
"""
import numpy as np

RHO=900.


def advance(volume,enthalpy,edges,flux,dt,*,layer_weights=None,
            addition=None,addition_enthalpy=None,removal=None):
    v=np.asarray(volume,dtype=np.float64); e=np.asarray(enthalpy,dtype=np.float64)
    links=np.asarray(edges); q=np.asarray(flux,dtype=np.float64)
    if v.ndim!=1 or e.ndim!=2 or e.shape[0]!=len(v): raise ValueError('State shapes differ')
    n,l=e.shape
    if links.ndim!=2 or links.shape[1]!=2 or links.dtype.kind not in 'iu' or q.shape!=(len(links),): raise ValueError('Invalid graph/flux')
    if not np.isfinite(dt) or dt<=0 or not all(np.isfinite(x).all() for x in (v,e,q)) or np.any(v<0): raise ValueError('Invalid finite state/time')
    a,b=links.T
    if np.any((a<0)|(a>=n)|(b< -1)|(b>=n)|(a==b)): raise ValueError('Invalid edge endpoints')
    if np.any((b==-1)&(q<0)): raise ValueError('Outside inflow requires explicit source volume and enthalpy')
    weights=np.asarray(layer_weights if layer_weights is not None else np.ones(l)/l,dtype=float)
    if weights.shape!=(l,) or not np.isfinite(weights).all() or np.any(weights<=0) or not np.isclose(weights.sum(),1.,rtol=0,atol=1e-14): raise ValueError('Invalid layer mass weights')
    add=np.zeros(n) if addition is None else np.asarray(addition,dtype=float)
    loss=np.zeros(n) if removal is None else np.asarray(removal,dtype=float)
    ae=np.zeros_like(e) if addition_enthalpy is None else np.asarray(addition_enthalpy,dtype=float)
    if add.shape!=(n,) or loss.shape!=(n,) or ae.shape!=e.shape or np.any(add<0) or np.any(loss<0) or not all(np.isfinite(x).all() for x in (add,loss,ae)): raise ValueError('Invalid source arrays')
    if np.any(add>0) and addition_enthalpy is None: raise ValueError('Positive ice source needs source enthalpy')
    donor=np.where(q>=0,a,b); receiver=np.where(q>=0,b,a); requested=abs(q)*dt
    outgoing=np.zeros(n); np.add.at(outgoing,donor,requested)
    factor=np.ones(n); limited=outgoing>v
    factor[limited]=(v[limited]/outgoing[limited])*(1-64*np.finfo(float).eps)
    actual=requested*factor[donor]
    sent=np.zeros(n); np.add.at(sent,donor,actual); retained=v-sent
    if np.any(retained<0): raise RuntimeError('Donor limiting failed; no volume clipping performed')
    arriving=np.zeros(n); incoming_energy=np.zeros_like(e); internal=receiver>=0
    np.add.at(arriving,receiver[internal],actual[internal])
    np.add.at(incoming_energy,receiver[internal],actual[internal,None]*e[donor[internal]])
    available=retained+arriving+add
    extensive=retained[:,None]*e+incoming_energy+add[:,None]*ae
    mixed=np.divide(extensive,available[:,None],out=np.zeros_like(e),where=available[:,None]>0)
    removed=np.minimum(loss,available); final_v=available-removed
    final_e=np.where(final_v[:,None]>0,mixed,0.)
    boundary=~internal; exported=float(actual[boundary].sum())
    exported_e=np.sum(actual[boundary,None]*e[donor[boundary]],axis=0)
    external_layer=RHO*(np.sum(add[:,None]*ae-removed[:,None]*mixed,axis=0)-exported_e)
    old_layer=RHO*np.sum(v[:,None]*e,axis=0); new_layer=RHO*np.sum(final_v[:,None]*final_e,axis=0)
    source_volume=float(add.sum()-removed.sum()-exported)
    info={'mass_error_m3':float(final_v.sum()-v.sum()-source_volume),
          'enthalpy_transport_error_j':float(np.dot(new_layer-old_layer-external_layer,weights)),
          'layer_extensive_error_j':(new_layer-old_layer-external_layer).tolist(),
          'source_volume_m3':source_volume,'exported_volume_m3':exported,
          'exported_enthalpy_j':float(RHO*np.dot(exported_e,weights)),
          'added_volume_m3':float(add.sum()),'removed_volume_m3':float(removed.sum()),
          'unrealized_removal_m3':float((loss-removed).sum()),'limited_donors':int(limited.sum()),
          'newly_wet_nodes':int(((v==0)&(final_v>0)).sum()),'newly_dry_nodes':int(((v>0)&(final_v==0)).sum()),
          'edge_transfer_m3':np.sign(q)*actual,'edge_enthalpy_transfer_j':np.sign(q)*actual*RHO*np.sum(e[donor]*weights,axis=1)}
    return final_v,final_e,info
