"""Right flexible GMRES; analytic flux templates and whole-column blocks.

CPU process only. The scientific Jacobian is never replaced by its preconditioner.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from polim_cpu.geometry_coupled_v1.backend import LinearFailure

def flux_matrix(adapter,h,u,weights,wet):
    """Exact local Ru assembly: endpoint-specific reflected durations, not dt/2."""
    t=adapter.topology;n=t.size;ea,eb=t.internal_edges.T;wa,wb,wo=weights
    rates=adapter.rates(u);donor=rates.donor_receiver_rate()[0]
    rows=[];cols=[];values=[]
    def put(row,node,normal,factor):
        for level,depth in enumerate((.5,.25)):
            for component in range(2):
                rows.append(row);cols.append(4*node+2*level+component)
                values.append(factor*normal[:,component]*depth*wet[node])
    for node,qw in ((adapter.a,5/12),(adapter.b,5/12),(adapter.third,1/6)):
        edge=adapter.inverse;factor=qw*h[donor[edge]]
        put(ea[edge],node,adapter.normal,factor*wa[edge])
        put(eb[edge],node,adapter.normal,-factor*wb[edge])
    bd=t.boundary_donors
    for j,(own,other) in enumerate(((adapter.outer_a,adapter.outer_b),(adapter.outer_b,adapter.outer_a))):
        sl=slice(j*len(own),(j+1)*len(own));factor=wo[sl]*h[bd[sl]]*(rates.boundary_outward[sl]>0)
        put(bd[sl],own,adapter.outer_normal,.75*factor)
        put(bd[sl],other,adapter.outer_normal,.25*factor)
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(n,4*n)).tocsr()

def column_WB(A,B):
    """Only 4x4 factors; apply to BSR column blocks, never form an inverse."""
    n=A.shape[0]//4;blocks=np.empty((n,4,4))
    for i in range(4):
        for j in range(4):blocks[:,i,j]=np.asarray(A[4*np.arange(n)+i,4*np.arange(n)+j]).ravel()
    factors=np.linalg.cholesky(blocks)
    bs=B.tobsr(blocksize=(4,1));nodes=np.repeat(np.arange(n),np.diff(bs.indptr))
    mid=np.linalg.solve(factors[nodes],bs.data)
    data=np.linalg.solve(factors[nodes].transpose(0,2,1),mid)
    wb=sparse.bsr_matrix((data,bs.indices,bs.indptr),shape=B.shape).tocsr()
    return wb,factors

def auxiliary(A,B,C,D,work):
    with work.span('column_factor_and_WB'):WB,factors=column_WB(A,B)
    with work.span('P_symbolic'):
        # A full column factor couples its four rows. Explicit zeros remain in
        # BSR and conservatively count as possible structural entries.
        cb=C.copy();wb=WB.copy();dd=D.copy()
        for x in (cb,wb,dd):x.data=np.ones_like(x.data)
        pattern=(dd+cb@wb).tocsr();pattern.data[:]=1
        symbolic_nnz=pattern.nnz
    with work.span('P_numeric'):P=(D-C@WB).tocsc()
    info=dict(symbolic_nnz=int(symbolic_nnz),nnz=int(P.nnz),storage_bytes=int(P.data.nbytes+P.indices.nbytes+P.indptr.nbytes),
        column_factor_bytes=int(factors.nbytes),symbolic_LU_upper_entries=int(P.shape[0]**2),symbolic_LU_note='Dense worst-case bound; actual SuperLU fill is measured before real launch on saved states; no SPD assumption.')
    work.event('P-before-factorization',**info)
    with work.span('P_LU_build'):lu=splu(P)
    info.update(L_nnz=int(lu.L.nnz),U_nnz=int(lu.U.nnz),LU_bytes=int(sum(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes for x in (lu.L,lu.U))))
    work.event('P-factorization',**info)
    return P,lu,info

def fgmres(action,pre,rhs,work,save_name):
    """Fixed target; variable right preconditioner. 8x2, at most16 applications."""
    norm=float(np.linalg.norm(rhs));x=np.zeros_like(rhs);records=[];used=0
    if norm==0:return x,dict(true_relative=0.,iterations=0)
    for cycle in range(2):
        r=rhs-action(x);beta=float(np.linalg.norm(r));vs=[r/beta];zs=[]
        H=np.zeros((9,8));g=np.zeros(9);g[0]=beta
        for j in range(8):
            work.count('block_precondition',None);z=pre(vs[j]);zs.append(z);v=action(z)
            for _ in range(2):
                for i,q in enumerate(vs):
                    h=float(q@v);H[i,j]+=h;v-=h*q
            H[j+1,j]=np.linalg.norm(v);used+=1;work.count('joint_Arnoldi')
            y=np.linalg.lstsq(H[:j+2,:j+1],g[:j+2],rcond=None)[0]
            est=float(np.linalg.norm(g[:j+2]-H[:j+2,:j+1]@y)/norm)
            records.append(dict(cycle=cycle,arnoldi=j+1,estimate=est))
            if est<=.1 or H[j+1,j]<=1e-30 or j==7:
                trial=x+np.column_stack(zs)@y;true=float(np.linalg.norm(action(trial)-rhs)/norm)
                work.event('FGMRES-explicit',cycle=cycle,iterations=used,true_relative=true,estimate=est)
                if true<=.1:
                    work.save(save_name,solution=trial,rhs=rhs,true_residual=action(trial)-rhs)
                    return trial,dict(true_relative=true,iterations=used,history=records)
                if j==7 or H[j+1,j]<=1e-30:x=trial;break
            vs.append(v/H[j+1,j])
    residual=action(x)-rhs;true=float(np.linalg.norm(residual)/norm)
    work.save(save_name,solution=x,rhs=rhs,true_residual=residual)
    work.event('joint-linear-failure',true_relative=true,iterations=used,history=records)
    raise LinearFailure('Joint FGMRES 16 Arnoldi steps: explicit scaled residual '+str(true)+' >0.1')
