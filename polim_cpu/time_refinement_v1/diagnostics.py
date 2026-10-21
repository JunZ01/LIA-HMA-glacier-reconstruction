"""Read-only accepted-step ledgers; no momentum evaluation or solve."""
import numpy as np
from polim_cpu.geometry_probe_v2.response import wet,raw_boundary

PARTS=['actual_nodes','potential_nodes','unmet_nodes','internal_volume','boundary_volume']

def zero_parts(adapter):
    n=adapter.topology.size
    return dict(actual_nodes=np.zeros(n),potential_nodes=np.zeros(n),unmet_nodes=np.zeros(n),
        internal_volume=np.zeros(len(adapter.topology.internal_edges)),
        boundary_volume=np.zeros(len(adapter.topology.boundary_donors)))

def node_flows(adapter,q,bo):
    t=adapter.topology;a,b=t.internal_edges.T
    incoming=np.zeros(t.size);outgoing=np.zeros(t.size);boundary=np.zeros(t.size)
    np.add.at(outgoing,a,np.maximum(q,0));np.add.at(incoming,b,np.maximum(q,0))
    np.add.at(outgoing,b,np.maximum(-q,0));np.add.at(incoming,a,np.maximum(-q,0))
    np.add.at(boundary,t.boundary_donors,bo)
    return incoming,outgoing,boundary

def branches(adapter,H,u,bed):
    return dict(wet=wet(H,bed),ice=H>0,donor=np.sign(adapter.rates(u).internal_signed).astype(np.int8),
        outlet=np.sign(raw_boundary(adapter,u)).astype(np.int8))

def patch(adapter):
    edges=adapter.topology.internal_edges
    return np.unique(np.r_[10103,10190,edges[np.isin(edges,[10103,10190]).any(axis=1)].ravel()])

def branch_record(adapter,old,new,bed):
    before=branches(adapter,old['H'],old['u'],bed);after=branches(adapter,new['H'],new['u'],bed)
    return {k:np.flatnonzero(before[k]!=after[k]).tolist() for k in before}
