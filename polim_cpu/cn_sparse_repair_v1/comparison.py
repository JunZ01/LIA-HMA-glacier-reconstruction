"""Frozen v48 scales; evidence persistence precedes any acceptance decision."""
from pathlib import Path
import numpy as np
from polim_cpu.geometry_probe_v2.common import write,digest,ahash
from polim_cpu.cn_cuda_v1.policy import CONSISTENCY

def metric(name,current,reference,force):
    absolute=float(np.linalg.norm(current-reference));norm=float(np.linalg.norm(reference));force_norm=float(np.linalg.norm(force))
    finite=bool(np.isfinite(current).all() and np.isfinite(reference).all() and np.isfinite(force).all())
    if not finite:
        return dict(reference_norm=norm if np.isfinite(norm) else None,current_norm=None,absolute_error=None,
            relative_to_reference=None,comparison_value=None,denominator=None,scale='nonfinite array; raw values retained in NPZ',
            threshold=CONSISTENCY['residual_force_scaled'] if name=='F' else CONSISTENCY.get(name+'_relative'),
            absolute_tolerance=None,near_zero=None,finite=False,passed=False,old_v49_value=None,old_v49_threshold=1e-9,old_v49_pass=False)
    if name=='F':
        den=max(force_norm,1.);tol=CONSISTENCY['residual_force_scaled'];scale='max(norm(reference_force),1); v48 exact force scale';near=False
    else:
        key={'force':'force_relative','Fu':'Fu_relative','FH':'FH_relative','flux':'flux_relative','Ru':'Ru_relative'}[name]
        near=norm<=1e-10;den=1. if near else norm;tol=CONSISTENCY['near_zero_absolute'] if near else CONSISTENCY[key]
        scale='absolute near-zero norm<=1e-10' if near else 'L2 norm of reference quantity'
    value=absolute/den;limit=tol*den
    return dict(reference_norm=norm,current_norm=float(np.linalg.norm(current)),absolute_error=absolute,
        relative_to_reference=absolute/norm if norm>0 else None,comparison_value=value,denominator=den,scale=scale,
        threshold=tol,absolute_tolerance=limit,near_zero=near,finite=finite,passed=finite and value<=tol,
        old_v49_value=absolute/max(norm,1.),old_v49_threshold=1e-9,old_v49_pass=finite and absolute/max(norm,1.)<=1e-9)

def record_then_require(root,label,current,reference,force,context):
    root=Path(root);path=root/(label+'-arrays.npz')
    data={**{'current_'+k:v for k,v in current.items()},**{'reference_'+k:v for k,v in reference.items()},'reference_force':force}
    with path.open('xb') as f:np.savez_compressed(f,**data)
    # No assertion or all-pass short circuit before the complete metric file is durable.
    result=dict(context=context,arrays=dict(path=str(path),sha256=digest(path),hashes={k:ahash(v) for k,v in data.items()}),items={})
    for key in reference:
        try:result['items'][key]=metric(key,current[key],reference[key],force)
        except BaseException as exc:result['items'][key]=dict(passed=False,error=repr(exc),arrays_retained=True)
    result['failed_components']=[k for k,v in result['items'].items() if not v['passed']]
    result['passed']=not result['failed_components'];write(root/(label+'-comparisons.json'),result)
    if not result['passed']:raise ValueError(label+' comparison failed: '+','.join(result['failed_components']))
    return result
