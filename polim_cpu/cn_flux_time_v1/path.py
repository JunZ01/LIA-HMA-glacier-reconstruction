"""Analytic quadratic reflection. Float64, no time sampling or thickness floor."""
import math
import numpy as np
from polim_cpu.geometry_probe_v2.response import divergence

EPS=np.finfo(float).eps
ROUND=128*EPS

def roots(a,b,c):
    """Scaled, cancellation-resistant real quadratic roots; linear exact branch."""
    scale=max(abs(a),abs(b),abs(c))
    if scale==0:return []
    a,b,c=a/scale,b/scale,c/scale
    if c==0:return [] if b==0 else [-a/b]
    disc=math.fsum([b*b,-4*a*c])
    if disc<0:return []
    sd=math.sqrt(disc);q=-.5*(b+math.copysign(sd,b))
    if q==0:return [-b/(2*c)]
    return sorted(set([q/c,a/q]))

def absolute_integral(qstart,qend,dt):
    a=np.abs(qstart);b=np.abs(qend);den=a+b
    same=qstart*qend>=0
    return dt*.5*np.where(same,den,np.divide(a*a+b*b,den,out=np.zeros_like(den),where=den>0))

def reconstruct(adapter,h0,smb,dt,q0,o0,q1,o1,alpha,beta):
    """Each edge has ONE alpha; boundary quadrature donors handled separately."""
    A=adapter.topology.dual_area
    qa=q1+alpha*(q0-q1);qb=-alpha*(q0-q1)
    oa=o1+beta*(o0-o1);ob=-beta*(o0-o1)
    b=dt*smb-dt*divergence(adapter,qa,oa)/A
    c=-.5*dt*divergence(adapter,qb,ob)/A
    return polynomial(h0,b,c,smb,dt,adapter,qa,qb,oa,ob)

def polynomial(a,b,c,smb,dt,adapter=None,qa=None,qb=None,oa=None,ob=None):
    a,b,c,smb=map(lambda x:np.asarray(x,dtype=float),(a,b,c,smb))
    if dt<=0 or (a<0).any() or not all(np.isfinite(x).all() for x in [a,b,c,smb]):raise ValueError('Invalid path inputs')
    # Earliest minimizer is the frozen generalized derivative at exact ties.
    end=a+b+c;theta=np.zeros(len(a));minimum=a.copy()
    mask=end<minimum;minimum[mask]=end[mask];theta[mask]=1.
    vertex=np.divide(-b,2*c,out=np.zeros_like(b),where=c!=0)
    inside=(c>0)&(vertex>0)&(vertex<1)
    value=a+vertex*(b+c*vertex);better=inside&(value<minimum)
    theta[better]=vertex[better];minimum[better]=value[better]
    ell=np.maximum(-minimum,0.);h1=end+ell
    lo=np.zeros(len(a));hi=np.zeros(len(a));dry=np.zeros(len(a),bool);ambiguous=[]
    # One decreasing record-minimum interval is possible for a quadratic.
    for i in np.flatnonzero((minimum<=0)|((a==0)&(b==0)&(c==0))):
        aa,bb,cc=a[i],b[i],c[i]
        if bb==0 and cc==0:
            if aa==0:dry[i]=True;hi[i]=1.
            continue
        left=0.;right=1.
        if cc>0:right=min(1.,-bb/(2*cc))
        elif cc<0:left=max(0.,-bb/(2*cc))
        elif bb>=0:continue
        if right<=left or right<=0 or left>=1:continue
        left=max(left,0.);right=min(right,1.)
        def f(t):return math.fsum([aa,bb*t,cc*t*t])
        if f(right)>=0:continue # isolated zero at minimum/end is not a dry interval
        if f(left)>0:
            rr=[r for r in roots(aa,bb,cc) if left<=r<=right]
            if not rr:
                ambiguous.append(int(i));continue
            left=min(rr)
        if right>left:dry[i]=True;lo[i]=left;hi[i]=right
    rate0=-(b+2*c*lo)/dt;rate1=-(b+2*c*hi)/dt
    rate_max=np.maximum(rate0,rate1);rate_min=np.minimum(rate0,rate1)
    rt=ROUND*np.maximum.reduce([np.ones_like(b),abs(smb),abs(b/dt),abs(2*c/dt)])
    ht=ROUND*np.maximum.reduce([np.ones_like(a),abs(a),abs(b),abs(c)])
    invalid_rate=dry&((rate_min < -rt)|(rate_max>np.maximum(-smb,0)+rt))
    invalid_source=(smb>=0)&(ell>ht)
    invalid=np.flatnonzero(invalid_rate|invalid_source).tolist();bad_edges=[];bad_out=[]
    if adapter is not None:
        edges=adapter.topology.internal_edges
        for side,sgn in [(0,1.),(1,-1.)]:
            nodes=edges[:,side]
            maxima=np.maximum(sgn*(qa+qb*lo[nodes]),sgn*(qa+qb*hi[nodes]))
            tol=ROUND*np.maximum.reduce([np.ones_like(qa),abs(qa),abs(qa+qb)])
            ids=np.flatnonzero(dry[nodes]&(maxima>tol))
            bad_edges.extend(ids.tolist());invalid.extend(nodes[ids].tolist())
        nodes=adapter.topology.boundary_donors
        maxima=np.maximum(oa+ob*lo[nodes],oa+ob*hi[nodes])
        tol=ROUND*np.maximum.reduce([np.ones_like(oa),abs(oa),abs(oa+ob)])
        ids=np.flatnonzero(dry[nodes]&(maxima>tol));bad_out=ids.tolist();invalid.extend(nodes[ids].tolist())
    invalid=sorted(set(invalid+ambiguous))
    return dict(a=a,b=b,c=c,minimum=minimum,theta_star=theta,ell=ell,H_path=h1,dry=dry,dry_lo=lo,dry_hi=hi,
        lambda_min=np.where(dry,rate_min,0.),lambda_max=np.where(dry,rate_max,0.),
        invalid_nodes=invalid,rate_nodes=np.flatnonzero(invalid_rate|invalid_source).tolist(),
        dry_out_edges=sorted(set(bad_edges)),dry_out_boundary=bad_out,ambiguous_nodes=ambiguous,
        physical=not invalid,roundoff_thickness=ht,roundoff_rate=rt,
        exhaustion=np.where(dry,lo,-1.),rewetting=np.where(dry&(hi<1),hi,-1.))
