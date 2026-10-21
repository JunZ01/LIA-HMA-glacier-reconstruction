"""Residual-consistent feedback actions, CPU only; no complete glacier BP solve.

Mass residuals have volume units. All evaluations retain one immutable H_old.
Derivatives are valid only on a fixed wet/dry and donor/outlet branch.
"""
from dataclasses import dataclass, asdict
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg
from polim_cpu.large_step_transport_v1 import implicit_conservative_step
from polim_cpu.projected_amg_v1 import ProjectedMultigrid


class BranchChange(RuntimeError):
    pass


class AuxiliaryConstraintError(RuntimeError):
    pass


@dataclass
class Work:
    complete_bp: int = 0
    glacier_complete_bp: int = 0
    residual_evaluations: int = 0
    tangent_solves: int = 0
    krylov_iterations: int = 0
    tangent_seconds: float = 0.
    bp_seconds: float = 0.
    build_seconds: float = 0.
    transport_seconds: float = 0.
    auxiliary_seconds: float = 0.
    auxiliary_solves: int = 0

    def record(self):
        return asdict(self)


def momentum_wet(h, bed):
    return h > 64*np.finfo(float).eps*np.maximum(1., np.maximum(abs(bed), abs(bed+h)))


class OriginalBPResidual:
    """Adapter to original CPU BP prepare/evaluate, never bp.solve.

    H derivative rebuilds mapping, basis, quadrature weights and driving force.
    The vertical fluidity array A and bed are frozen as in phase 2.
    Geometry finite differences cost TWO residual/build evaluations per action.
    """
    def __init__(self, mesh, fluidity, work=None):
        from polim_cpu.v13r1 import bp
        self.kernel = bp
        self.mesh = {k: np.array(v, copy=True) for k,v in mesh.items()}
        self.fluidity = np.array(fluidity, copy=True)
        self.work = work or Work()

    def _evaluate(self, u, h, tangent=False):
        start=perf_counter()
        a=self.kernel.prepare({**self.mesh, 'thickness': np.asarray(h)})
        self.work.build_seconds += perf_counter()-start
        u=np.array(u, dtype=float, copy=True).ravel(); u[~a['wet']]=0.
        self.work.residual_evaluations += 1
        r,_,matrix,force=self.kernel.evaluate(a,u,tangent,A=self.fluidity)
        return r,matrix,force,a['wet']

    def residual(self, u, h):
        return self._evaluate(u,h)[0]

    def velocity_action(self, u, h, du):
        _,matrix,_,wet=self._evaluate(u,h,True)
        v=np.array(du,copy=True);v[~wet]=0.
        return matrix@v

    def geometry_action(self, u, h, dh, epsilon=1e-4):
        h=np.asarray(h);dh=np.asarray(dh);bed=self.mesh['bed']
        if np.any(dh[~momentum_wet(h,bed)] != 0):
            raise BranchChange('Dry geometry directions are excluded from the smooth tangent')
        for trial in (h+epsilon*dh,h-epsilon*dh):
            if np.any(trial<0) or not np.array_equal(momentum_wet(trial,bed),momentum_wet(h,bed)):
                raise BranchChange('Geometry difference crosses momentum wet/dry branch')
        return (self.residual(u,h+epsilon*dh)-self.residual(u,h-epsilon*dh))/(2*epsilon)

    def time_advance(self, *args, **kwargs):
        raise NotImplementedError('Real glacier backend is disabled; separate registration and authorization required')


def tangent_solve(fu, rhs, work, *, base_cycle=None, wet=None, tolerance=1e-10, maximum=200):
    """Positive projected Fu solve; reuse projected AMG cycle interface if supplied.

    Fu must be the true fixed-geometry tangent, not an old hierarchy matrix.
    Failure is explicit. No nonlinear BP solve or automatic retry is hidden here.
    """
    start=perf_counter();rhs=np.asarray(rhs);n=len(rhs)
    wet=np.ones(n,dtype=bool) if wet is None else np.asarray(wet,dtype=bool)
    b=rhs*wet;op=LinearOperator((n,n),matvec=lambda v: wet*fu(wet*v)+(~wet)*v)
    pre=None
    if base_cycle is not None:
        projected=ProjectedMultigrid(base_cycle,SimpleNamespace(active=wet))
        pre=LinearOperator((n,n),matvec=projected.cycle)
    counter=[0]
    def count(_): counter[0]+=1
    x,info=cg(op,b,M=pre,rtol=tolerance,atol=0.,maxiter=maximum,callback=count)
    relative=float(np.linalg.norm(op@x-b)/max(np.linalg.norm(b),1e-300))
    work.tangent_solves+=1;work.krylov_iterations+=counter[0];work.tangent_seconds+=perf_counter()-start
    if info or not np.isfinite(x).all() or relative>max(tolerance*5,1e-12):
        raise RuntimeError(f'Tangent solve failed: status={info}, true relative residual={relative}')
    return x,dict(iterations=counter[0],relative=relative)


def geometry_feedback(dh, fu, fh, ru, work, **linear_options):
    """K dh = R_u du, Fu du = -FH dh. No BP equilibrium re-solve."""
    du,info=tangent_solve(fu,-fh(dh),work,**linear_options)
    return ru(du),info


class OriginalFeedbackAction:
    """Frozen-point actual BP/mass wiring; no nonlinear solve, no time step.

    Construction assembles Fu once. FH rebuilds geometry twice per action.
    Active thickness DOFs are excluded; the caller must handle branch changes
    outside this smooth interface. No usable glacier S is claimed until tested.
    """
    def __init__(self,bp,mass,h,u,base_cycle=None):
        self.bp=bp;self.mass=mass;self.h=np.array(h,copy=True);self.u=np.array(u,copy=True)
        self.base_cycle=base_cycle
        _,self.Fu,_,self.wet=bp._evaluate(self.u,self.h,True)

    def apply(self,dh,epsilon=1e-4):
        if np.any(np.asarray(dh)[self.h==0]!=0):
            raise BranchChange('Fixed thickness obstacle set excludes dry directions')
        return geometry_feedback(dh,lambda du:self.Fu@du,
            lambda v:self.bp.geometry_action(self.u,self.h,v,epsilon),
            lambda du:self.mass.velocity_action(self.h,self.u,du),self.bp.work,
            wet=self.wet,base_cycle=self.base_cycle)


class MassResidual:
    def __init__(self, h_old, smb, dt, rates, work=None):
        self.h_old=np.array(h_old,dtype=float,copy=True);self.h_old.flags.writeable=False
        self.smb=np.array(smb,dtype=float,copy=True);self.smb.flags.writeable=False
        self.dt=float(dt);self.rates=rates;self.work=work or Work()

    def matrix(self,u):
        rates=self.rates(u)
        return sparse.diags(rates.topology.dual_area)+self.dt*rates.outward_operator()

    def rhs(self,u):
        return self.rates(u).topology.dual_area*(self.h_old+self.dt*self.smb)

    def raw(self,h,u):
        """Potential-SMB equation before the physical obstacle multiplier."""
        return self.matrix(u)@h-self.rhs(u)

    def velocity_action(self,h,u,du,epsilon=1e-6):
        """Difference ONLY the face-rate map, not a transport solve."""
        rates=self.rates(u);rp=self.rates(u+epsilon*du);rm=self.rates(u-epsilon*du)
        for r in (rp,rm):
            if not np.array_equal(np.sign(r.internal_signed),np.sign(rates.internal_signed)) or not np.array_equal(r.boundary_outward>0,rates.boundary_outward>0):
                raise BranchChange('Face donor or outward-only boundary switch')
        donor=rates.donor_receiver_rate()[0];top=rates.topology
        q=(rp.internal_signed-rm.internal_signed)/(2*epsilon)*h[donor]
        bo=(rp.boundary_outward-rm.boundary_outward)/(2*epsilon)*h[top.boundary_donors]
        div=np.zeros(len(h));np.add.at(div,top.internal_edges[:,0],q);np.add.at(div,top.internal_edges[:,1],-q)
        np.add.at(div,top.boundary_donors,bo)
        return self.dt*div

    def formal(self,h,u):
        start=perf_counter();rates=self.rates(u);area=rates.topology.dual_area
        fixed=implicit_conservative_step(self.h_old,self.smb,self.dt,rates)
        residual=self.raw(h,u)-fixed.unmet_ablation_volume
        source=self.dt*float(area@fixed.actual_smb);export=self.dt*rates.boundary_export_rate(h)
        v0=float(area@self.h_old)
        scale=max(v0+abs(source)+self.dt*rates.absolute_flux_rate(h),1.)
        mass=abs(float(area@h)-v0-source+export)/max(v0,abs(source),abs(export),1.)
        self.work.transport_seconds+=perf_counter()-start
        return dict(local=float(np.abs(residual).sum()/scale),mass=mass,
                    fixed_point=float(area@np.abs(fixed.thickness-h)/max(v0,1.)),
                    active_equal=bool(np.array_equal(h==0,fixed.thickness==0)),
                    actual_smb=fixed.actual_smb,unmet=fixed.unmet_ablation_volume,
                    source=source,export=export,residual=residual)


def auxiliary_correction(mass,h,u,S,work):
    """Fixed positive-thickness branch prototype, solve R(Hhat,u)+S(Hhat-H)=0.

    General S need not be an M-matrix. Do not silently apply the transport LCP
    theorem to it. This first prototype stops on branch changes; obstacle
    generalization is deliberately NOT implemented for real trajectories.
    """
    if np.any(h<=0):raise BranchChange('Auxiliary prototype requires fixed positive thickness')
    start=perf_counter();matrix=mass.matrix(u).toarray()+S
    hhat=np.linalg.solve(matrix,mass.rhs(u)+S@h)
    work.auxiliary_solves+=1;work.auxiliary_seconds+=perf_counter()-start
    if not np.isfinite(hhat).all() or np.any(hhat<=0):
        raise AuxiliaryConstraintError('Auxiliary correction leaves smooth positive branch; no physical ablation credited')
    aux=S@(hhat-h)
    return hhat,aux


def physical_unmet_from_auxiliary(*args, **kwargs):
    raise AuxiliaryConstraintError('Auxiliary multipliers cannot be recorded as physical unmet ablation')
