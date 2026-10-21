"""Fixed-u reflected transport; monotone binary limiter; explicit full-BE fallback."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from polim_cpu.large_step_transport_v1 import implicit_conservative_step
from polim_cpu.geometry_probe_v2.response import flux,divergence,decompose
from .path import reconstruct,absolute_integral

class TransportFailure(RuntimeError):pass

class Equation:
    def __init__(self,adapter,h0,u0,smb,dt,work):
        self.adapter=adapter;self.h0=h0.copy();self.smb=smb.copy();self.dt=dt;self.w=work
        self.area=adapter.topology.dual_area;self.V0=float(self.area@h0)
        self.q0,self.o0=flux(adapter,h0,u0)
        self.alpha=np.ones(len(self.q0));self.beta=np.ones(len(self.o0));self.revision=0
    def evaluate(self,h,u):
        with self.w.span('events_and_reflection'):
            q1,o1=flux(self.adapter,h,u)
            path=reconstruct(self.adapter,self.h0,self.smb,self.dt,self.q0,self.o0,q1,o1,self.alpha,self.beta)
            qi=self.dt*((1-self.alpha/2)*q1+self.alpha*self.q0/2)
            oi=self.dt*((1-self.beta/2)*o1+self.beta*self.o0/2)
            source=self.area*(self.dt*self.smb+path['ell'])
            R=self.area*(h-self.h0)+divergence(self.adapter,qi,oi)-source
            absolute_q=absolute_integral(q1+self.alpha*(self.q0-q1),q1,self.dt)
            absolute_o=absolute_integral(o1+self.beta*(self.o0-o1),o1,self.dt)
            scale=max(self.V0+abs(float(source.sum()))+float(absolute_q.sum()+absolute_o.sum()),1.)
            mass=abs(float(self.area@(h-self.h0)-source.sum()+oi.sum()))/max(self.V0,abs(float(source.sum())),abs(float(oi.sum())),1.)
            return dict(R=R,path=path,q1=q1,o1=o1,qi=qi,oi=oi,source=source,U=self.area*path['ell'],
                scale=scale,local=float(abs(R).sum()/scale),mass=mass,absolute_q=absolute_q,absolute_o=absolute_o)
    def duration(self,e):return np.where(e['path']['ell']>0,self.dt*(1-e['path']['theta_star']),self.dt)
    def weights(self,e):
        t=self.adapter.topology;d=self.duration(e);a,b=t.internal_edges.T
        def w(nodes,alpha):return d[nodes]-alpha*d[nodes]**2/(2*self.dt)
        return w(a,self.alpha),w(b,self.alpha),w(t.boundary_donors,self.beta)
    def matrix(self,u,e):
        with self.w.span('transport_J_build'):
            t=self.adapter.topology;r=self.adapter.rates(u);donor=r.donor_receiver_rate()[0];a,b=t.internal_edges.T
            wa,wb,wo=self.weights(e)
            L=sparse.coo_matrix((np.r_[wa*r.internal_signed,-wb*r.internal_signed,wo*r.boundary_outward],
                (np.r_[a,b,t.boundary_donors],np.r_[donor,donor,t.boundary_donors])),shape=(t.size,t.size)).tocsr()
            return sparse.diags(self.area)+L
    def ru(self,h,u,du,e):
        t=self.adapter.topology;parts=decompose(self.adapter,h,u,np.zeros_like(h),du)
        return self.weighted(parts['velocity_internal'],parts['velocity_boundary'],e)
    def weighted(self,q,o,e):
        t=self.adapter.topology;wa,wb,wo=self.weights(e)
        out=np.zeros(t.size);np.add.at(out,t.internal_edges[:,0],wa*q);np.add.at(out,t.internal_edges[:,1],-wb*q)
        np.add.at(out,t.boundary_donors,wo*o);return out
    def lower(self,nodes,reason,full=False):
        t=self.adapter.topology
        edges=np.arange(len(self.alpha)) if full else np.flatnonzero(np.isin(t.internal_edges,nodes).any(axis=1))
        outs=np.arange(len(self.beta)) if full else np.flatnonzero(np.isin(t.boundary_donors,nodes))
        edges=edges[self.alpha[edges]!=0];outs=outs[self.beta[outs]!=0]
        if len(edges)+len(outs):self.alpha[edges]=0.;self.beta[outs]=0.;self.revision+=1
        self.w.event('limiter',nodes=list(map(int,nodes)),internal_edges=edges.tolist(),boundary=outs.tolist(),reason=reason,
            full_BE=full,revision=self.revision)
        return len(edges)+len(outs)
    def fixed_newton(self,u,initial):
        h=initial.copy()
        for iteration in range(21):
            e=self.evaluate(h,u)
            if e['local']<=1e-12 and e['mass']<=1e-12:return h,e
            if iteration==20:raise TransportFailure('20 fixed-u Newton updates exhausted')
            with self.w.span('transport_Newton'):
                T=self.matrix(u,e)
                with self.w.span('transport_LU'):delta=spsolve(T.tocsc(),-e['R'])
                if not np.isfinite(delta).all():raise TransportFailure('nonfinite transport Newton direction')
                merit=float(abs(e['R']).sum()/max(self.V0,1.));taken=False
                for ls in range(8):
                    self.w.count('transport_line_candidate');alpha=2.**(-ls)
                    ht=np.maximum(h+alpha*delta,0.) # candidate only, never a booked source
                    trial=self.evaluate(ht,u);new=float(abs(trial['R']).sum()/max(self.V0,1.))
                    if new<=(1-1e-4*alpha)*merit or (trial['local']<=1e-12 and trial['mass']<=1e-12):
                        h=ht;taken=True;break
                if not taken:raise TransportFailure('8 fixed-u line candidates exhausted')
    def limited(self,u,initial):
        h=initial.copy();failure=None
        for modification in range(7):
            try:h,e=self.fixed_newton(u,h)
            except TransportFailure as exc:failure=str(exc);break
            if e['path']['physical']:return h,e
            if modification==6:failure='6 local limiter modifications exhausted';break
            bad=e['path']['invalid_nodes']
            changed=self.lower(bad,dict(rate_nodes=e['path']['rate_nodes'],dry_out_edges=e['path']['dry_out_edges'],
                dry_out_boundary=e['path']['dry_out_boundary'],ambiguous=e['path']['ambiguous_nodes']))
            if not changed:failure='already-BE incident edges still violate path';break
        self.lower([],failure,full=True)
        with self.w.span('all_BE_fallback'):
            r=implicit_conservative_step(self.h0,self.smb,self.dt,self.adapter.rates(u));h=r.thickness;e=self.evaluate(h,u)
        if not e['path']['physical'] or e['local']>1e-8 or e['mass']>1e-10:
            raise TransportFailure('Full BE fallback fails reflected residual/physical path')
        self.w.event('all-BE-result',local=e['local'],mass=e['mass'],physical=e['path']['physical'])
        return h,e
