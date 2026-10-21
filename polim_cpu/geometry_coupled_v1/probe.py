"""Integrated real-grid diagnostic. Noise ambiguity alone does not stop quarter."""
import numpy as np
from polim_cpu.geometry_probe_v2.common import restore_base,arrays,ROOT
from polim_cpu.geometry_probe_v2.response import (freeze_masks,branch_events,decompose,reference_parts,
    flux,comparison,norm,mean_velocity,residual_screen)

class WiringError(RuntimeError):pass

def relative(a,b):return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

def run(backend,adapter):
    w=backend.w;w.location='integrated fixed-geometry verification'
    h,u,meta=restore_base();dh=np.zeros_like(h);dh[10190]=1.;eps=.05;bed=backend.mesh['bed']
    masks=freeze_masks(adapter,h,u,bed);w.save('probe-base',H=h,u=u,dH=dh,**masks);w.event('probe-identity',metadata=meta)
    lin=backend.linearize(h,u);baser=float(np.linalg.norm(lin.F)/max(np.linalg.norm(lin.f),1.))
    if baser>1e-6:raise WiringError('saved base original residual fails; no repair solve')
    fh=lin.fh(dh)
    fp=backend.evaluate(h+eps*dh,u)[0];fm=backend.evaluate(h-eps*dh,u)[0];fd=(fp-fm)/(2*eps)
    old=arrays(ROOT/'polim_cpu/runs/v42-geometric-probe-v2/geometry-derivative.npz')['FH']
    checks=dict(FH_vs_005=relative(fh,fd),FH_vs_existing_001=relative(fh,old),FD_005_vs_001=relative(fd,old))
    w.save('probe-geometry-derivative',FH=fh,FD_005=fd,FD_existing_001=old,F0=lin.F,force0=lin.f)
    w.event('geometry-derivative-check',threshold=5e-4,**checks)
    if max(checks.values())>5e-4:raise WiringError('FH derivative mismatch against fixed-u differences')
    du=lin.solve(-fh,'probe geometry response');w.save('probe-tangent',du=du,linear_residual=lin.Fu@du+fh)
    # Reuse the original residual, fixed geometry: verify the assembled Fu wiring.
    fu_fd=(backend.evaluate(h,u+eps*du)[0]-backend.evaluate(h,u-eps*du)[0])/(2*eps)
    fu_error=relative(lin.Fu@du,fu_fd);w.event('Fu-direction-check',relative=fu_error,threshold=5e-4)
    if fu_error>5e-4:raise WiringError('original Fu direction disagrees with original residual difference')
    sols={};res={};force={};events={}
    for name,sign in [('plus',1),('minus',-1)]:
        hx=h+sign*eps*dh;ux,info=backend.full(hx,u,'probe',1e-8)
        F,_,f,_=backend.evaluate(hx,ux)
        if np.linalg.norm(F)/max(np.linalg.norm(f),1.)>1e-8:raise WiringError(name+' diagnostic BP target missed')
        w.save('probe-'+name,H=hx,u=ux,F=F,force=f)
        sols[name]=ux;res[name]=F;force[name]=f;events[name]=branch_events(adapter,hx,ux,bed,masks)
    with w.span('probe_flux_comparison'):
        p=decompose(adapter,h,u,dh,du);r=reference_parts(adapter,h,u,dh,sols['plus'],sols['minus'],eps)
        area=adapter.topology.dual_area;mw=masks['wet'];qi,qo=flux(adapter,h,u);metrics={}
        metrics['velocity']=comparison(du.reshape(-1,4),r['velocity'].reshape(-1,4),u.reshape(-1,4),area,mw,eps)
        metrics['depth_average']=comparison(mean_velocity(du),mean_velocity(r['velocity']),mean_velocity(u),area,mw,eps)
        for part in ('thickness','velocity','total'):
            for side,mask,base in [('internal',masks['internal'],qi),('boundary',masks['boundary'],qo)]:
                key=part+'_'+side
                # Proven BEFORE inspecting reference: node10190 is not an outlet donor.
                zero=(part=='thickness' and side=='boundary' and not np.any(dh[adapter.topology.boundary_donors]))
                if zero:
                    tolerance=128*np.finfo(float).eps*max(norm(base,np.ones(len(base)),mask),1.)/eps
                    error=norm(p[key]-r[key],np.ones(len(base)),mask)
                    metrics[key]=dict(status='pass-structural-zero' if error<=tolerance else 'wiring-error',absolute=error,threshold=tolerance,relative=None)
                else:metrics[key]=comparison(p[key],r[key],base,np.ones(len(base)),mask,eps)
        metrics['K']=comparison(p['K_direction']/area,r['K_direction']/area,h,area,mw,eps)
        noise=residual_screen(fh,lin.F,res['plus'],res['minus'],[np.linalg.norm(lin.f),np.linalg.norm(force['plus']),np.linalg.norm(force['minus'])],eps)
        nonsmooth=any(e['affects_evaluation'] for e in events.values())
        ambiguous=noise['status']!='screen-pass' or any(v['status'].startswith('inconclusive') for v in metrics.values())
        disagreement=any(v['status'] in ['response-disagreement','wiring-error'] for v in metrics.values())
        status='nonsmooth' if nonsmooth else ('inconclusive' if ambiguous else ('failed-response' if disagreement else 'pass-this-direction'))
    w.save('probe-predicted',**p);w.save('probe-reference',**r)
    outcome=dict(status=status,metrics=metrics,noise=noise,events=events,base_BP=baser,geometry_checks=checks,Fu_check=fu_error,
        derivative_accuracy_confirmed=status=='pass-this-direction',quarter_may_proceed=status in ['pass-this-direction','inconclusive','nonsmooth'])
    w.event('probe-result',**outcome)
    if any(v['status']=='wiring-error' for v in metrics.values()):raise WiringError('structurally zero boundary thickness term is nonzero')
    if status=='failed-response':raise WiringError('resolvable smooth response mismatch; stop before quarter')
    return outcome
