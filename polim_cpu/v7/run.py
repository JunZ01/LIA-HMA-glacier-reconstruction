"""One combined fixture followed by one frozen real fixed-geometry diagnostic."""
import json,os,sys,time,subprocess,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='8'
import numpy as np
from scipy.sparse import coo_matrix
from . import bp
from .registration import active,implementation,EVIDENCE
from ..v6r1.mesh import synthetic
from ..v6r1.storage import callback,restore
from ..v3.runner import PROJECT,digest,write_json,seal
from ..spatial_pair_guard import require_paired_source

MESH=PROJECT/'polim_cpu/runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
SOURCE=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'


def identity(kind):
    return seal(dict(implementation=implementation(),kind=kind,mesh_sha256=digest(MESH),
        A_SI=bp.A_SI,n=3,layers=3,basal='no-slip',lateral='natural-effective-BP-traction',
        strain_regularization_year_minus2=bp.REG,tolerance=1e-6,max_iterations=80,max_seconds=7200,
        python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,
        pyamg=bp.pyamg.__version__,numba=__import__('numba').__version__,threads=8,
        initial='SIA slope averaged at native nodes',time_advanced_years=0,climate_read=False))


def resume_fixture(root):
    active();root=Path(root);mesh,_=synthetic();contract=identity('fixture')
    u,meta=restore(root/'baseline/iteration-0001',contract)
    result,info=bp.solve(mesh,initial=u,iteration_offset=1)
    np.savez_compressed(root/'restart.npz',velocity=result)
    write_json(root/'restart.json',info)


def fixture(root):
    from ..v6r1.bp import assemble
    mesh,_=synthetic();a=bp.prepare(mesh);old=assemble(mesh,beta=1e4)
    rng=np.random.default_rng(2387);u=rng.normal(size=a['ndof'])*.1
    full=np.zeros((len(mesh['xy']),3,2));full[:,1:]=u.reshape(-1,2,2)
    free=(np.arange(len(mesh['xy']))[:,None]*6+np.arange(2,6)).ravel()
    r,e,K,f=bp.evaluate(a,u,True)
    local=full.ravel()[old['ids']];B=old['forms'];w=old['weights']
    Bu=np.einsum('qij,qj->qi',B,local);e2=np.sum(local*Bu,axis=1)/4+bp.REG
    eta=.5*(bp.A_SI*bp.YEAR)**(-1/3)*e2**(-1/3)
    rr=np.bincount(old['ids'].ravel(),weights=(w[:,None]*eta[:,None]*Bu).ravel(),minlength=old['ndof'])-old['force']
    kk=w[:,None,None]*eta[:,None,None]*(B-1/(6*e2[:,None,None])*np.einsum('qi,qj->qij',Bu,Bu))
    reference=coo_matrix((kk.ravel(),(old['rows'],old['cols'])),shape=(old['ndof'],old['ndof'])).tocsr()[free][:,free]
    residual_error=float(np.linalg.norm(r-rr[free])/np.linalg.norm(rr[free]))
    tangent_error=float(np.linalg.norm((K-reference).data)/np.linalg.norm(reference.data))
    force_error=float(np.linalg.norm(f-old['force'][free])/np.linalg.norm(f))
    assert max(residual_error,tangent_error,force_error)<1e-10
    checks=[dict(name='same BP residual, tangent and load as frozen v6 with basal elimination',status='pass',
                 residual_error=residual_error,tangent_error=tangent_error,force_error=force_error)]
    # Thin cells retain their actual H: check the scaled linear solve algebra.
    thin={**mesh,'thickness':np.geomspace(1.01e-6,100.,len(mesh['xy']))}
    at=bp.prepare(thin);_,_,kt,_=bp.evaluate(at,np.zeros(at['ndof']),True)
    d=1/np.sqrt(kt.diagonal());probe=rng.normal(size=len(d));scaled_probe=d*(kt@(d*probe))
    assert np.isfinite(scaled_probe).all() and np.all(d>0)
    checks.append(dict(name='unmodified micro-metre thin ice has finite positive diagonal scaling',status='pass',min_H=float(thin['thickness'].min())))
    baseline,info=bp.solve(mesh,checkpoint=callback(root/'baseline',identity('fixture')))
    write_json(root/'baseline.json',info)
    child=subprocess.run([sys.executable,'-m','polim_cpu.v7.run','resume-fixture',str(root)],cwd=PROJECT,capture_output=True,text=True,timeout=300)
    (root/'restart.stdout.log').write_text(child.stdout,encoding='utf-8')
    (root/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
    if child.returncode:raise RuntimeError(child.stderr)
    with np.load(root/'restart.npz',allow_pickle=False) as z:restarted=z['velocity']
    restart_error=float(np.linalg.norm(restarted-baseline)/np.linalg.norm(baseline))
    assert restart_error<1e-12 and info['relative_residual']<1e-6
    checks.append(dict(name='AMG nonlinear convergence and independent-process sealed checkpoint recovery',status='pass',
                      relative_residual=info['relative_residual'],restart_relative_error=restart_error,bitwise_equal=bool(np.array_equal(restarted,baseline))))
    doc=json.loads(SOURCE.read_text(encoding='utf-8'));require_paired_source(SOURCE,PROJECT/doc['network']['path'])
    with np.load(MESH,allow_pickle=False) as z:real={k:z[k] for k in z.files}
    plan=active()
    assert digest(MESH)==plan['mesh_sha256'] and digest(SOURCE)==plan['source_package_sha256']
    assert len(real['xy'])==178230 and len(real['triangles'])==353096
    assert np.all(real['thickness']>0) and np.isclose(real['mesh_pixel_volume_m3'].sum(),doc['volume_m3'],rtol=1e-12)
    checks.append(dict(name='paired source, native mesh identity and volume scientific preflight',status='pass',
        geometry='same reconstructed diagnostic geometry; scientific suitability and velocity validation not evaluated',
        volume_m3=float(real['mesh_pixel_volume_m3'].sum()),nodes=len(real['xy']),triangles=len(real['triangles'])))
    write_json(root/'verification.json',dict(status='passed',passed=len(checks),checks=checks))
    return real


def run(root):
    plan=active();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    contract=identity('real-fixed-geometry')
    with (EVIDENCE/f'launch-fingerprint-{contract}.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(contract=contract,output=str(root),pid=os.getpid()),stream)
    write_json(root/'launch.json',dict(pid=os.getpid(),command='python -m polim_cpu.v7.run run',
        start_time=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        output=str(root),device='CPU',threads=8,estimated_seconds=[300,7200],estimated_peak_memory_GiB=16,
        checkpoint_cadence='each accepted Newton iteration',stop_conditions=plan['stop_conditions'],
        time_advanced_years=0,climate_read=False,contract=contract,plan_sha256=seal(plan)))
    try:
        (root/'fixture').mkdir();mesh=fixture(root/'fixture')
        write_json(root/'real-stage-launch.json',dict(status='started-after-combined-preflight',pid=os.getpid(),contract=contract))
        velocity,info=bp.solve(mesh,checkpoint=callback(root/'real',contract))
        stresses=bp.stress_centroids(mesh,velocity)
        if not np.isfinite(velocity).all() or not np.isfinite(stresses).all():raise RuntimeError('Nonfinite final fields')
        np.savez_compressed(root/'fields.npz',velocity_m_year=velocity,triangle_centroid_stress=stresses)
        speed=np.linalg.norm(velocity[:,-1],axis=1)
        write_json(root/'result.json',dict(status='completed',system=2387,contract=contract,**info,
            A_SI=bp.A_SI,n=3,basal='no-slip fs=0',surface_and_lateral='natural effective BP traction',
            surface_speed_m_year_quantiles=dict(zip(['min','p50','p90','p95','p99','max'],map(float,np.quantile(speed,[0,.5,.9,.95,.99,1])))),
            stress_columns=['tau_xx_Pa','tau_yy_Pa','tau_xy_Pa','tau_xz_Pa','tau_yz_Pa','tau_zz_Pa','effective_strain_year_minus1','viscosity_Pa_year'],
            stress_sigma=[.25,.75],stress_continuity='shared velocity DOFs and weak internal traction balance; pointwise stress jumps retained',
            real_velocity_solved=True,real_time_advanced=False,time_advanced_years=0,climate_read=False,
            thermodynamic_feedback=False,scientific_acceptance=False,accepted_count_effect=0,
            mesh_sha256=digest(MESH),source_package_sha256=digest(SOURCE),implementation=implementation(),
            limitations=['constant project A transferred from 1D calibration, not recalibrated on this geometry',
                'native pixel staircase and corner-interpolated geometry','3 vertical nodes; no mesh convergence study',
                'natural lateral effective BP traction including nonzero thickness perimeter',
                'no observed velocity validation, no time evolution, no thermal feedback',
                'stress map is a diagnostic; no quantitative confluence traction-flux error assessment'],
            total_wall_seconds=time.perf_counter()-start))
    except Exception as exc:
        write_json(root/'failure.json',dict(status='technical-failure',error=str(exc),traceback=traceback.format_exc(),
                    contract=contract,wall_seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    if sys.argv[1]=='resume-fixture':resume_fixture(sys.argv[2])
    elif sys.argv[1]=='run':run(sys.argv[2])
    else:raise ValueError('Unknown command')
