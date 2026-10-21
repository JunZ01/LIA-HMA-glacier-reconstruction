"""Fixed COO insertion/sort/reduction maps, with freshly owned float64 values.

The marker CSR uses the same SciPy COO row insertion and CSR sort order as
the reference. add.at then sums duplicates sequentially in that order.
No material/state coefficient is cached. CPU process only.
"""
import hashlib,time
import numpy as np
from scipy import sparse

VERSION='v55-structure-1'

def key_for(z,ad):
    d=hashlib.sha256((VERSION+'float64').encode())
    for a in (z['xy'],z['tri'],z['ids'],ad.topology.internal_edges,
              ad.topology.boundary_donors,ad.a,ad.b,ad.third,ad.inverse,
              ad.outer_a,ad.outer_b,ad.normal,ad.outer_normal):
        a=np.ascontiguousarray(a);d.update(str((a.shape,a.dtype.str)).encode());d.update(a.tobytes())
    return d.hexdigest()

def b_indices(z,shape):
    return (np.broadcast_to(z['ids'][:,:,None],shape).ravel(),
            np.broadcast_to(z['tri'][:,None,:],shape).ravel())

def c_indices(ad):
    t=ad.topology;ea,eb=t.internal_edges.T;rows=[];cols=[]
    def put(row,node):
        for level in range(2):
            for component in range(2):rows.append(row);cols.append(4*node+2*level+component)
    for node in (ad.a,ad.b,ad.third):
        put(ea[ad.inverse],node);put(eb[ad.inverse],node)
    for j,(own,other) in enumerate(((ad.outer_a,ad.outer_b),(ad.outer_b,ad.outer_a))):
        bd=t.boundary_donors[j*len(own):(j+1)*len(own)]
        put(bd,own);put(bd,other)
    return np.concatenate(rows),np.concatenate(cols)

def c_values(ad,h,u,weights,wet):
    # Same multiplication and append order as v54 flux_matrix, no new model.
    wa,wb,wo=weights;rates=ad.rates(u);donor=rates.donor_receiver_rate()[0];values=[]
    def put(node,normal,factor):
        for level,depth in enumerate((.5,.25)):
            for component in range(2):values.append(factor*normal[:,component]*depth*wet[node])
    for node,qw in ((ad.a,5/12),(ad.b,5/12),(ad.third,1/6)):
        edge=ad.inverse;factor=qw*h[donor[edge]]
        put(node,ad.normal,factor*wa[edge]);put(node,ad.normal,-factor*wb[edge])
    bd=ad.topology.boundary_donors
    for j,(own,other) in enumerate(((ad.outer_a,ad.outer_b),(ad.outer_b,ad.outer_a))):
        sl=slice(j*len(own),(j+1)*len(own));factor=wo[sl]*h[bd[sl]]*(rates.boundary_outward[sl]>0)
        put(own,ad.outer_normal,.75*factor);put(other,ad.outer_normal,.25*factor)
    return np.concatenate(values),dict(donor=donor,boundary_outward=rates.boundary_outward)

class Pattern:
    def __init__(self,rows,cols,shape):
        # COO->CSR insertion is row-stable. scipy's identical CSR sort then
        # establishes the exact order used by the legacy duplicate summation.
        order=np.argsort(rows,kind='stable');ptr=np.r_[0,np.cumsum(np.bincount(rows,minlength=shape[0]))]
        marker=sparse.csr_matrix((order.astype(np.int64),cols[order].copy(),ptr),shape=shape)
        marker.sort_indices();self.order=marker.data.copy()
        rr=np.repeat(np.arange(shape[0]),np.diff(marker.indptr));cc=marker.indices
        first=np.r_[True,(rr[1:]!=rr[:-1])|(cc[1:]!=cc[:-1])]
        self.group=np.cumsum(first)-1;self.indices=cc[first].copy()
        self.indptr=np.r_[0,np.cumsum(np.bincount(rr[first],minlength=shape[0]))].astype(marker.indptr.dtype)
        self.shape=shape
        for a in (self.order,self.group,self.indices,self.indptr):a.flags.writeable=False
    def update(self,values):
        if values.dtype!=np.float64 or len(values)!=len(self.order):raise ValueError('float64 contribution shape')
        data=np.zeros(len(self.indices),dtype=np.float64)
        np.add.at(data,self.group,values[self.order])
        out=sparse.csr_matrix((data,self.indices,self.indptr),shape=self.shape,copy=False)
        out.has_sorted_indices=True;out.has_canonical_format=True
        return out

class Cache:
    def __init__(self):self.key=None;self.patterns={};self.hits=0;self.builds=0;self.invalidations=0
    def update(self,z,ad,bv,cv):
        start=time.perf_counter();key=key_for(z,ad);kt=time.perf_counter()-start;cold=key!=self.key
        bt=time.perf_counter()
        if cold:
            if self.key is not None:self.invalidations+=1
            n=len(z['bed']);br,bc=b_indices(z,bv.shape);cr,cc=c_indices(ad)
            self.patterns={'B':Pattern(br,bc,(4*n,n)),'C':Pattern(cr,cc,(n,4*n))};self.key=key;self.builds+=1
        else:self.hits+=1
        build=time.perf_counter()-bt;t=time.perf_counter();B=self.patterns['B'].update(bv.ravel());bt=time.perf_counter()-t
        t=time.perf_counter();C=self.patterns['C'].update(cv);ct=time.perf_counter()-t
        return B,C,dict(key_seconds=kt,cold_build_seconds=build,B_update_seconds=bt,C_update_seconds=ct,cold=cold,
                        hits=self.hits,builds=self.builds,invalidations=self.invalidations,key=key)

def legacy(z,ad,bv,cv):
    n=len(z['bed']);t=time.perf_counter();br,bc=b_indices(z,bv.shape);cr,cc=c_indices(ad);index=time.perf_counter()-t
    t=time.perf_counter();B=sparse.coo_matrix((bv.ravel(),(br,bc)),shape=(4*n,n)).tocsr();bt=time.perf_counter()-t
    t=time.perf_counter();C=sparse.coo_matrix((cv,(cr,cc)),shape=(n,4*n)).tocsr();ct=time.perf_counter()-t
    return B,C,dict(index_template_seconds=index,B_convert_seconds=bt,C_convert_seconds=ct)
