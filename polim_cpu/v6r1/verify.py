"""One combined mechanics and static geometry suite; no real glacier solve."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from .mesh import synthetic,build
from .bp import solve,assemble
from .storage import callback,restore
from .registration import active,implementation
from ..v3.runner import PROJECT,write_json,seal,digest
from ..spatial_pair_guard import require_paired_source


def setup():
    mesh,metrics=synthetic();boundary=np.unique(mesh['boundary_edges'])
    outlet_y=mesh['xy'][:,1].max()
    fixed=boundary[mesh['xy'][boundary,1]<outlet_y-1e-6]
    return mesh,metrics,fixed


def identity():
    return seal(dict(implementation=implementation(),python=sys.version,numpy=np.__version__,
        scipy=__import__('scipy').__version__,fixture='Y-vector-BP-3layers-beta1e4-contrast4-below850',threads=1))


def resume(root):
    active();root=Path(root);mesh,_,fixed=setup();contract=identity()
    u,meta=restore(root/'baseline/iteration-0002',contract)
    solved,info=solve(mesh,fixed_nodes=fixed,initial=u,iteration_offset=2,
                      checkpoint=callback(root/'resumed',contract))
    with (root/'restart-final.npz').open('xb') as f:np.savez_compressed(f,velocity=solved)
    write_json(root/'restart-result.json',info)


def real_mesh(out):
    package=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'
    doc=json.loads(package.read_text(encoding='utf-8'));network=PROJECT/doc['network']['path']
    require_paired_source(package,network)
    path=PROJECT/doc['array_archive']['path']
    with np.load(path,allow_pickle=False) as f:
        mesh,metrics=build(f['valid_mask'],f['x'],f['y'],f['bed_elevation_m'],f['thickness_m'])
    expected=doc['volume_m3']
    if not np.isclose(metrics['mesh_volume_m3'],expected,rtol=1e-12):raise ValueError('Real source volume differs')
    out.mkdir();np.savez_compressed(out/'mesh.npz',**mesh)
    metrics.update(source_package=dict(path=str(package),sha256=digest(package)),
        source_archive_sha256=digest(path),status='static-native-pixel-mesh-built-not-dynamics-approved',
        real_velocity_solved=False,real_time_advanced=False,climate_read=False,
        projected_unknowns_3_vertical_nodes=metrics['nodes']*6,
        limitations=['pixel staircase boundary retained','corner interpolation alters within-pixel B/H while center values remain exact',
                     'thin-ice conditioning and source geometry uncertainty unresolved','no mesh-resolution or observed-velocity validation',
                     'no source connectivity repair; disconnected components remain disconnected'])
    write_json(out/'geometry.json',metrics)
    return metrics


def verify(output):
    active();out=Path(output);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();checks=[];contract=identity()
    write_json(out/'launch.json',dict(pid=os.getpid(),command='python -m polim_cpu.v6r1 verify',output=str(out.resolve()),
        started_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        estimated_seconds=[10,180],device='CPU',threads=1,checkpoint_cadence='every Newton iteration',
        stop_conditions=['nonfinite','nonconvergence','invalid mesh','source or fingerprint mismatch'],real_glacier_solve=False))
    try:
        mesh,metrics,fixed=setup();np.savez_compressed(out/'synthetic-mesh.npz',**mesh,fixed_nodes=fixed)
        # Independent affine velocity field in physical xyz coordinates.
        a=assemble(mesh);xy=mesh['xy'];z=mesh['bed'][:,None]+mesh['thickness'][:,None]*np.linspace(0,1,3)
        coefficients=np.array([.013,-.008,.006,.011,-.009,.003])
        aa,bb,cc,dd,ee,ff=coefficients
        u=aa*xy[:,0,None]+bb*xy[:,1,None]+cc*z
        v=dd*xy[:,0,None]+ee*xy[:,1,None]+ff*z
        field=np.stack([u,v],axis=2).ravel();local=field[a['ids']]
        e2=np.einsum('qi,qij,qj->q',local,a['forms'],local)/4
        exact=aa*aa+ee*ee+aa*ee+(bb+dd)**2/4+cc*cc/4+ff*ff/4
        err=float(np.max(abs(e2-exact))/exact)
        assert err<1e-8
        checks.append(dict(name='independent affine physical-coordinate BP strain invariant',status='pass',relative_error=err))
        del a,local
        baseline,info=solve(mesh,fixed_nodes=fixed,checkpoint=callback(out/'baseline',contract))
        write_json(out/'baseline-result.json',info)
        beta=np.full(len(xy),1e4);beta[xy[:,1]>850]*=4
        changed,contrast=solve(mesh,fixed_nodes=fixed,beta=beta,initial=baseline,
                               checkpoint=callback(out/'drag-contrast',seal(dict(parent=contract,beta=beta.tolist()))))
        write_json(out/'drag-contrast-result.json',contrast)
        upstream=(xy[:,1]<550)&(np.linalg.norm(baseline[:,-1,:],axis=1)>1e-5)
        diff=np.linalg.norm(changed[:,-1,:]-baseline[:,-1,:],axis=1)
        response=float(np.linalg.norm((changed-baseline)[upstream])/np.linalg.norm(baseline[upstream]))
        assert upstream.any() and response>1e-6
        assert info['relative_residual']<1e-6 and contrast['relative_residual']<1e-6
        checks.append(dict(name='downstream resistance affects upstream tributaries through shared global momentum',status='pass',
            upstream_velocity_relative_change=response,max_upstream_surface_vector_change_m_year=float(diff[upstream].max()),
            baseline_residual=info['relative_residual'],contrast_residual=contrast['relative_residual'],
            changed_geometry=False,changed_basal_drag_region='y > 850 m',assessed_tributaries='y < 550 m'))
        # Same domain physically rotated; solve again from zero in the new coordinates.
        angle=np.deg2rad(31.);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        rotated={**mesh,'xy':xy@R.T};ur,ri=solve(rotated,fixed_nodes=fixed)
        rotation_error=float(np.linalg.norm(ur-baseline@R.T)/np.linalg.norm(baseline))
        assert rotation_error<1e-5
        checks.append(dict(name='horizontal coordinate rotation covariance',status='pass',relative_velocity_error=rotation_error))
        child=subprocess.run([sys.executable,'-m','polim_cpu.v6r1','resume-fixture',str(out.resolve())],cwd=PROJECT,capture_output=True,text=True,timeout=180)
        (out/'restart.stdout.log').write_text(child.stdout,encoding='utf-8');(out/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
        if child.returncode:raise RuntimeError(child.stderr)
        with np.load(out/'restart-final.npz',allow_pickle=False) as f:restored=f['velocity']
        restart_info=json.loads((out/'restart-result.json').read_text(encoding='utf-8'))
        assert np.array_equal(baseline,restored) and info['history'][2:]==restart_info['history']
        checks.append(dict(name='independent-process Newton checkpoint continuation',status='pass',bitwise_equal=True))
        geometry=real_mesh(out/'system2387-native-mesh')
        # A diagonal-only synthetic pixel pair must not share a velocity node.
        dm,mm=build(np.eye(2,dtype=bool),np.arange(2.),np.arange(2.),np.zeros((2,2)),np.ones((2,2)))
        assert not set(dm['pixel_corner_nodes'][0]).intersection(dm['pixel_corner_nodes'][1])
        checks.append(dict(name='paired native-pixel mesh area/volume and corner-only disconnection',status='pass',
            source_area_m2=geometry['source_area_m2'],mesh_area_m2=geometry['mesh_area_m2'],
            source_volume_m3=geometry['source_volume_m3'],mesh_volume_m3=geometry['mesh_volume_m3'],
            native_source_centers_exact=geometry['source_centers_exact']))
        np.savez_compressed(out/'fields.npz',baseline=baseline,drag_contrast=changed,surface_vector_change=diff,upstream_mask=upstream)
        summary=dict(status='completed',passed=len(checks),failed=0,checks=checks,synthetic_mesh=metrics,
            global_vector_bp=True,junction_traction_balance='weak form',real_mesh=geometry,
            real_velocity_solved=False,real_time_advanced=False,thermodynamic_coupling=False,
            scientific_acceptance=False,implementation=implementation(),contract=contract,wall_seconds=time.perf_counter()-started)
        write_json(out/'verification.json',summary)
        print(json.dumps(dict(status='completed',passed=len(checks),upstream_response=response,real_nodes=geometry['nodes'],wall_seconds=summary['wall_seconds'])))
    except Exception as exc:
        write_json(out/'failure.json',dict(status='technical-failure',checks=checks,error=str(exc),traceback=traceback.format_exc(),contract=contract))
        raise
