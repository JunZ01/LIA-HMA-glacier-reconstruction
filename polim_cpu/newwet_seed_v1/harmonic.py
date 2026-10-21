"""One CPU sparse graph factorization for four u/H right-hand sides.

No BP operator, model residual, geometry regularization or physical source.
The four free DOFs are the two horizontal components at two non-basal levels.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

def predict(xy,tri,h,u,oldwet,newwet,free=None,work=None):
    h=np.asarray(h);v=np.asarray(u).reshape(-1,4);N=np.flatnonzero(newwet&~oldwet);out=v.copy();reports=[]
    if free is None:free=np.ones_like(v,dtype=bool)
    edges=np.unique(np.sort(np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1),axis=0)
    if not len(N):return out.ravel(),dict(nodes=[],components=[],factorizations=0,reason='no-newwet'),dict()
    lookup=np.full(len(h),-1,dtype=int);lookup[N]=np.arange(len(N));ij=edges[np.all(lookup[edges]>=0,axis=1)]
    graph=sparse.coo_matrix((np.ones(2*len(ij)),(np.r_[lookup[ij[:,0]],lookup[ij[:,1]]],np.r_[lookup[ij[:,1]],lookup[ij[:,0]]])),shape=(len(N),len(N))).tocsr()
    nc,labels=connected_components(graph,directed=False);anchored=[];groups=[]
    for k in range(nc):
        nodes=N[labels==k];touch=edges[np.isin(edges,nodes).any(axis=1)];boundary=np.unique(touch[oldwet[touch]&newwet[touch]])
        r=dict(nodes=nodes.tolist(),boundary=boundary.tolist(),reason='anchored' if len(boundary) else 'no-wet-boundary')
        reports.append(r)
        if len(boundary):anchored.extend(nodes);groups.append((nodes,boundary,touch,r))
    if not anchored:return out.ravel(),dict(nodes=N.tolist(),components=reports,factorizations=0,reason='no-anchored-component'),dict(edges=edges)
    active=np.sort(np.asarray(anchored));idx=np.full(len(h),-1,dtype=int);idx[active]=np.arange(len(active));rows=[];cols=[];vals=[];rhs=np.zeros((len(active),4))
    valid=True
    for nodes,boundary,touch,r in groups:
        relevant=touch[np.isin(touch,np.r_[nodes,boundary]).all(axis=1)];dist2=np.sum((xy[relevant[:,0]]-xy[relevant[:,1]])**2,axis=1)
        if not np.isfinite(dist2).all() or np.any(dist2<=0):r['reason']='invalid-edge-length';valid=False;continue
        if not np.all(np.isfinite(v[boundary]/h[boundary,None])):r['reason']='nonfinite-boundary-u-over-H';valid=False;continue
        r['boundary_all_zero']=bool(np.all(v[boundary]==0))
        for (a,b),w in zip(relevant,1/dist2):
            for i,j in ((a,b),(b,a)):
                if idx[i]<0:continue
                ii=idx[i];rows.append(ii);cols.append(ii);vals.append(w)
                if idx[j]>=0:rows.append(ii);cols.append(idx[j]);vals.append(-w)
                else:rhs[ii]+=w*(v[j]/h[j])
    if not valid:return out.ravel(),dict(nodes=N.tolist(),components=reports,factorizations=0,reason='invalid-input'),dict(edges=edges,rhs=rhs)
    mat=sparse.coo_matrix((vals,(rows,cols)),shape=(len(active),len(active))).tocsc()
    if work is None:lu=splu(mat);q=lu.solve(rhs)
    else:
        work.count('seed_graph_factorization',1)
        with work.span('seed_graph_LU'):lu=splu(mat)
        work.counts['seed_graph_rhs']+=4
        with work.span('seed_graph_four_rhs'):q=lu.solve(rhs)
    vel=h[active,None]*q
    finite=np.isfinite(q).all() and np.isfinite(vel).all()
    if finite:out[active]=np.where(free[active],vel,v[active])
    for nodes,boundary,touch,r in groups:
        r['prediction_norm']=float(np.linalg.norm(out[nodes]));r['increment_norm']=float(np.linalg.norm(out[nodes]-v[nodes]))
        if not finite:r['reason']='nonfinite-prediction'
        elif r['boundary_all_zero']:r['reason']='all-zero-boundary'
        elif np.all(out[nodes]==0):r['reason']='vector-cancellation'
    return (out.ravel(),dict(nodes=N.tolist(),components=reports,factorizations=1,linear_RHS=4,reason='finite' if finite else 'nonfinite-prediction',
        effective=bool(finite and np.any(out[N]!=v[N])),graph_relative_residual=float(np.linalg.norm(mat@q-rhs)/max(np.linalg.norm(rhs),1.))),
        dict(edges=edges,active=active,rhs=rhs,q=q,graph_data=mat.data,graph_indices=mat.indices,graph_indptr=mat.indptr))
