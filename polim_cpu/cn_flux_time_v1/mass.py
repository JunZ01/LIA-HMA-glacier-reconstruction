"""New time-discrete mass equation, not the original backward Euler residual."""
import numpy as np
from types import SimpleNamespace
from polim_cpu.geometry_window_v1.ledger import preview
from .transport import Equation

def accepted(d,bp):
    return (bp<=1e-6 and d['local']<=1e-8 and d['mass']<=1e-10 and d['cumulative_mass_relative']<=1e-10
        and d['physical_path'] and d['fixed_point']<=1e-8 and d['all_finite'] and d['H_min']>=0)

class Mass:
    def __init__(self,adapter,h_old,u_old,smb,work,dt,ledger):
        self.eq=Equation(adapter,h_old,u_old,smb,dt,work);self.area=self.eq.area;self.w=work;self.ledger=ledger
        self.last=None;self.last_h=None;self.last_u=None
    def evaluate(self,h,u):
        with self.w.span('formal_transport'):
            fixed,aux=self.eq.limited(u,h)
            e=self.eq.evaluate(h,u);self.last=e;self.last_h=h.copy();self.last_u=u.copy()
            R=e['R'];source=float(e['source'].sum());export=float(e['oi'].sum())
            natural=float(abs(R).sum()/max(self.eq.V0,1.));fp=float(self.area@abs(h-fixed)/max(self.eq.V0,1.))
            d=dict(local=e['local'],mass=e['mass'],physical_local=e['local'],physical_mass=e['mass'],
                natural=natural,fixed_point=fp,active_equal=bool(np.array_equal(h==0,fixed==0)),physical_path=e['path']['physical'],
                source=source,export=export,unmet=float(e['U'].sum()),potential=self.eq.dt*float(self.area@self.eq.smb),
                H_min=float(h.min()),active_count=int((h==0).sum()),merit=max(natural,e['mass']),
                all_finite=bool(all(np.isfinite(x).all() for x in [h,u,R,e['source'],e['U']])),
                residual_L1_m3=float(abs(R).sum()),residual_sum_m3=float(R.sum()),scale=e['scale'],
                limiter_revision=self.eq.revision,BE_edges=int((self.eq.alpha==0).sum()),BE_outlets=int((self.eq.beta==0).sum()))
            d.update(preview(self.ledger,float(self.area@h),source,export))
            return d,self.eq.matrix(u,e).tocsr(),R,e['source']/(self.eq.dt*self.area),e['U'],SimpleNamespace(thickness=fixed)
    def ru(self,h,u,du):
        with self.w.span('Ru_action'):
            e=self.last if np.array_equal(h,self.last_h) and np.array_equal(u,self.last_u) else self.eq.evaluate(h,u)
            return self.eq.ru(h,u,du,e)
    def save(self,name,h,u):
        # Independent final reconstruction on the actual candidate, no auxiliary ledger.
        with self.w.span('final_path_reconstruction'):e=self.eq.evaluate(h,u)
        path=e['path'];self.w.save(name,H=h,u=u,actual_smb=e['source']/(self.eq.dt*self.area),unmet=e['U'],
            potential_nodes=self.eq.dt*self.area*self.eq.smb,actual_nodes=e['source'],residual=e['R'],
            Q0=self.eq.q0,Q1=e['q1'],O0=self.eq.o0,O1=e['o1'],alpha=self.eq.alpha,beta=self.eq.beta,
            internal_volume=e['qi'],boundary_volume=e['oi'],absolute_internal_volume=e['absolute_q'],absolute_boundary_volume=e['absolute_o'],
            **{k:path[k] for k in ['a','b','c','theta_star','ell','H_path','dry','dry_lo','dry_hi','lambda_min','lambda_max','exhaustion','rewetting']})
        self.w.event('path-audit',name=name,physical=path['physical'],invalid_nodes=path['invalid_nodes'],
            dry_nodes=int(path['dry'].sum()),positive_final_with_unmet=int(((h>0)&(e['U']>0)).sum()),
            BE_volume_fraction=float((e['absolute_q'][self.eq.alpha==0].sum()+e['absolute_o'][self.eq.beta==0].sum())/max(e['absolute_q'].sum()+e['absolute_o'].sum(),1e-300)))
        return e
