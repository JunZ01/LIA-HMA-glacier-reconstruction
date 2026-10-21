"""Explicit boundary semantics. Strong constraints use an affine nullspace.

Velocity constraints and impermeable normals are exact; no penalty coefficient.
Periodic pairs identify equal velocities at equal sigma, allowing elevation shifts.
Pressure-front loads include ice overburden minus prescribed water pressure.
"""
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix


def edge_geometry(mesh):
    tri=mesh['triangles'];xy=mesh['xy']
    oriented=np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]])
    keys=np.sort(oriented,axis=1);_,first,count=np.unique(keys,axis=0,return_index=True,return_counts=True)
    if np.any(count>2):raise ValueError('Nonmanifold')
    edges=oriented[first[count==1]];tangent=xy[edges[:,1]]-xy[edges[:,0]];length=np.linalg.norm(tangent,axis=1)
    normals=np.column_stack([tangent[:,1],-tangent[:,0]])/length[:,None]
    return edges,normals,length


def validated(mesh,bc):
    edges,normals,length=edge_geometry(mesh);tags=bc.get('edges',[])
    if len(tags)!=len(edges):raise ValueError('Every perimeter edge requires explicit semantics')
    allowed={'natural','wall','velocity','traction','pressure_front','periodic'}
    if any(tag.get('kind') not in allowed for tag in tags):raise ValueError('Unknown boundary kind')
    return edges,normals,length,tags


def constraints(mesh,layers,bc):
    edges,normals,_,tags=validated(mesh,bc);nn=len(mesh['xy']);parent=np.arange(nn)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in bc.get('periodic_pairs',[]):
        if not (0<=a<nn and 0<=b<nn):raise ValueError('Periodic node out of range')
        ra,rb=find(int(a)),find(int(b));parent[max(ra,rb)]=min(ra,rb)
    roots=np.array([find(i) for i in range(nn)]);_,groups=np.unique(roots,return_inverse=True)
    ng=int(groups.max()+1);conditions={}
    def add(node,level,direction,value):
        if not np.isfinite(value):raise ValueError('Nonfinite constraint')
        key=int(groups[node])*layers+level;conditions.setdefault(key,[]).append((np.asarray(direction,dtype=float),float(value)))
    if bc.get('basal','no-slip')!='no-slip':raise ValueError('This version retains no-slip basal physics')
    for node in range(nn):
        levels=range(layers) if mesh['thickness'][node]==0 else [0]
        for level in levels:
            add(node,level,[1,0],0);add(node,level,[0,1],0)
    sigma=np.linspace(0,1,layers)
    periodic_nodes=set(np.asarray(bc.get('periodic_pairs',[]),dtype=int).ravel().tolist())
    for e,(edge,n,tag) in enumerate(zip(edges,normals,tags)):
        kind=tag['kind']
        if kind=='periodic' and not all(int(i) in periodic_nodes for i in edge):raise ValueError('Periodic edge missing node pairs')
        if kind=='wall':
            for node in edge:
                for level in range(layers):add(node,level,n,0)
        elif kind=='velocity':
            for node in edge:
                value=tag['value'](mesh['xy'][node],sigma,mesh['bed'][node]+sigma*mesh['thickness'][node]) if callable(tag['value']) else tag['value']
                values=np.broadcast_to(np.asarray(value,dtype=float),(layers,2))
                for level in range(layers):
                    for axis in range(2):
                        if not np.isnan(values[level,axis]):add(node,level,np.eye(2)[axis],values[level,axis])
    bases=[];particular=[];offsets=[];nfree=0
    for key in range(ng*layers):
        c=conditions.get(key,[])
        if not c:basis=np.eye(2);v=np.zeros(2)
        else:
            matrix=np.array([p[0] for p in c]);values=np.array([p[1] for p in c]);_,sing,V=np.linalg.svd(matrix,full_matrices=True)
            rank=int(np.sum(sing>1e-10));v=np.linalg.lstsq(matrix,values,rcond=1e-10)[0]
            if np.max(abs(matrix@v-values))>1e-9*max(1.,np.max(abs(values))):raise ValueError('Conflicting boundary constraints')
            basis=V[rank:].T
        offsets.append(nfree);nfree+=basis.shape[1];bases.append(basis);particular.append(v)
    rr=[];cc=[];vv=[];lift=np.zeros(nn*layers*2)
    for node in range(nn):
        for level in range(layers):
            key=int(groups[node])*layers+level;rows=(node*layers+level)*2+np.arange(2)
            lift[rows]=particular[key];basis=bases[key]
            for axis in range(2):
                for col in range(basis.shape[1]):
                    if basis[axis,col]!=0:rr.append(rows[axis]);cc.append(offsets[key]+col);vv.append(basis[axis,col])
    T=coo_matrix((vv,(rr,cc)),shape=(len(lift),nfree)).tocsr()
    return T,lift,dict(periodic_node_groups=ng,original_planar_nodes=nn,free_unknowns=nfree,
        constraint_method='exact affine nullspace, no penalty',zero_thickness_nodes=int(np.count_nonzero(mesh['thickness']==0)))


def loads(mesh,layers,bc,rho=910.,g=9.81):
    edges,normals,length,tags=validated(mesh,bc);ndof=len(mesh['xy'])*layers*2
    force=np.zeros(ndof);rows=[];cols=[];data=[];total=np.zeros(2)
    gauss,weights=np.polynomial.legendre.leggauss(3);gauss=(gauss+1)/2;weights=weights/2
    for edge,n,L,tag in zip(edges,normals,length,tags):
        if tag['kind'] not in ('traction','pressure_front','wall'):continue
        beta=float(tag.get('friction',0.)) if tag['kind']=='wall' else 0.
        if beta<0:raise ValueError('Negative wall friction')
        if tag['kind']=='wall' and beta==0:continue
        for s,ws in zip(gauss,weights):
            lam=np.array([1-s,s]);H=float(mesh['thickness'][edge]@lam);bed=float(mesh['bed'][edge]@lam);surface=bed+H
            if H<=0:continue
            xy=lam@mesh['xy'][edge]
            for k in range(layers-1):
                lo=k/(layers-1);hi=(k+1)/(layers-1);breaks=[lo,hi]
                # Integrate each side of the waterline separately, avoiding a quadrature kink.
                if tag['kind']=='pressure_front' and tag.get('water_level') is not None:
                    water=(float(tag['water_level'])-bed)/H
                    if lo<water<hi:breaks=[lo,water,hi]
                dofs=np.array([(int(node)*layers+lev)*2+axis for lev in (k,k+1) for node in edge for axis in (0,1)])
                for lower,upper in zip(breaks[:-1],breaks[1:]):
                    for t,wt in zip(gauss,weights):
                        sigma=lower+t*(upper-lower);z=bed+sigma*H;local=(sigma-lo)/(hi-lo)
                        shape=np.r_[(1-local)*lam,local*lam];weight=ws*wt*(upper-lower)*L*H
                        if tag['kind']=='pressure_front':
                            water_pressure=0. if tag.get('water_level') is None else float(tag.get('water_density',1000.))*g*max(float(tag['water_level'])-z,0.)
                            traction=(rho*g*(surface-z)-water_pressure)*n
                        elif tag['kind']=='traction':
                            value=tag['value'];traction=np.asarray(value(xy,z,surface,n) if callable(value) else value,dtype=float)
                        else:traction=np.zeros(2)
                        if traction.shape!=(2,) or not np.isfinite(traction).all():raise ValueError('Invalid effective traction')
                        localforce=(shape[:,None]*traction[None,:]*weight).ravel();np.add.at(force,dofs,localforce);total+=weight*traction
                        if beta:
                            tangent=np.eye(2)-np.outer(n,n);matrix=beta*weight*np.kron(np.outer(shape,shape),tangent)
                            rows.extend(np.repeat(dofs,8));cols.extend(np.tile(dofs,8));data.extend(matrix.ravel())
    drag=coo_matrix((data,(rows,cols)),shape=(ndof,ndof)).tocsr()
    return force,drag,dict(integrated_effective_load_N=total.tolist(),wall_drag_nonzeros=drag.nnz,
        pressure_front='ice overburden minus explicit water pressure; exact waterline split',all_edges_explicit=True)
