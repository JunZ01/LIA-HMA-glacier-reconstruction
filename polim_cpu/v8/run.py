"""Single combined suite; immutable member evidence and isolated failures."""
import os,sys,json,time,subprocess,traceback
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['NUMBA_NUM_THREADS']='4'
import numpy as np
from . import mechanics as bp
from .boundary import constraints,loads,edge_geometry
from .cases import case,rectangle,periodic,surface_profile,slab_surface_exact,REFERENCE
from .registration import active,implementation,EVIDENCE,register
from ..v3.runner import PROJECT,digest,write_json,seal
from ..v6r1.storage import callback,restore


def identity(name,nz,plan):
    return seal(dict(plan=seal(plan),case=name,vertical_nodes=nz,implementation=implementation(),
        runtime=dict(python=sys.version,numpy=np.__version__,scipy=__import__('scipy').__version__,numba=__import__('numba').__version__,pyamg=bp.pyamg.__version__),threads=4))


def boundary_fixture():
    mesh=rectangle([0.,100.,200.],[0.,50.,100.],lambda p:0.,lambda p:100.)
    edges,normals,length=edge_geometry(mesh);tags=[dict(kind='natural') for _ in edges]
    right=np.flatnonzero(np.all(mesh['xy'][edges,0]==200.,axis=1))
    checks=[]
    for title,value,expected in [('dry-front',None,100*9.81*.5*910*100**2),('partly-submerged-front',37.,100*9.81*.5*(910*100**2-1000*37**2))]:
        selected=[dict(t) for t in tags]
        for i in right:selected[i]=dict(kind='pressure_front',water_level=value,water_density=1000.)
        f,_,meta=loads(mesh,9,dict(edges=selected))
        total=f.reshape(-1,2).sum(axis=0);error=float(np.linalg.norm(total-[expected,0])/abs(expected))
        checks.append(dict(name=title,relative_error=error,status='pass' if error<1e-10 else 'fail'))
    selected=[dict(t) for t in tags]
    for i in right:selected[i]=dict(kind='traction',value=[123.,-41.])
    f,_,_=loads(mesh,9,dict(edges=selected));expected=100*100*np.array([123.,-41.])
    error=float(np.linalg.norm(f.reshape(-1,2).sum(axis=0)-expected)/np.linalg.norm(expected))
    checks.append(dict(name='constant-effective-traction-integral',relative_error=error,status='pass' if error<1e-10 else 'fail'))
    # One inlet with a nonzero profile and two impermeable frictional side walls.
    selected=[dict(t) for t in tags]
    for i,e in enumerate(edges):
        if np.all(mesh['xy'][e,0]==0.):selected[i]=dict(kind='velocity',value=lambda xy,s,z:np.column_stack([2*s,np.zeros_like(s)]))
        elif np.ptp(mesh['xy'][e,1])==0:selected[i]=dict(kind='wall',friction=7.)
    bc=dict(edges=selected);T,lift,_=constraints(mesh,9,bc);rng=np.random.default_rng(8)
    u=(T@rng.normal(size=T.shape[1])+lift).reshape(-1,9,2)
    wall_error=max(float(abs(u[e]@n).max()) for e,n,t in zip(edges,normals,selected) if t['kind']=='wall')
    inlet=mesh['xy'][:,0]==0.;inlet_error=float(np.max(abs(u[inlet,:,0]-2*np.linspace(0,1,9))))
    basal_error=float(abs(u[:,0]).max())
    _,drag,_=loads(mesh,9,bc)
    probe=np.zeros_like(u);probe[:,:,0]=2*np.linspace(0,1,9)
    # Two 200m walls, H100m, beta7; integral (2 sigma)^2 d sigma =4/3.
    exact_work=2*200*100*7*4/3
    work_error=abs(float(probe.ravel()@(drag@probe.ravel()))-exact_work)/exact_work
    error=max(wall_error,inlet_error,basal_error,work_error)
    checks.append(dict(name='inlet-wall-basal-and-friction-work',maximum_error=error,status='pass' if error<1e-10 else 'fail'))
    bc=periodic(mesh);T,lift,_=constraints(mesh,9,bc);u=(T@rng.normal(size=T.shape[1])+lift).reshape(-1,9,2)
    error=max(float(abs(u[a]-u[b]).max()) for a,b in bc['periodic_pairs'])
    checks.append(dict(name='periodic-velocity-identity',maximum_error=error,status='pass' if error<1e-10 else 'fail'))
    return dict(status='pass' if all(c['status']=='pass' for c in checks) else 'fail',checks=checks)


def resume(root):
    plan=active();root=Path(root);name='slab';nz=9;contract=identity(name,nz,plan)
    mesh,bc=case(name);u,meta=restore(root/'checkpoints/iteration-0001',contract)
    velocity,info=bp.solve(mesh,bc,layers=nz,initial=u,iteration_offset=1)
    np.savez_compressed(root/'restart.npz',velocity=velocity);write_json(root/'restart.json',info)


def execute(root):
    plan=active();root=Path(root).resolve();root.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    fingerprint=seal(dict(plan=seal(plan),implementation=implementation()))
    write_json(EVIDENCE/f'launch-fingerprint-{fingerprint}.json',dict(output=str(root),pid=os.getpid(),fingerprint=fingerprint))
    write_json(root/'launch.json',dict(pid=os.getpid(),command='python -m polim_cpu.v8.run run',start_time=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        output=str(root),device='CPU',threads=4,estimated_seconds=[120,1800],estimated_peak_memory_GiB=4,
        checkpoint_cadence='each Newton iteration',stop_conditions=plan['stop_conditions'],fingerprint=fingerprint,plan_sha256=seal(plan)))
    results=[];fixture=boundary_fixture();write_json(root/'boundary-checks.json',fixture)
    if fixture['status']!='pass':
        write_json(root/'terminal-evidence.json',dict(status='technical-failure',reason='boundary fixture failed',fixture=fixture,results=[],accepted_count_effect=0));return
    for name in ('slab','hom-a020','hom-e000'):
        for nz in (3,9):
            active();member=root/f'{name}-nz{nz}';member.mkdir();contract=identity(name,nz,plan)
            mesh,bc=case(name)
            write_json(member/'identity.json',dict(case=name,vertical_nodes=nz,contract=contract))
            try:
                velocity,info=bp.solve(mesh,bc,layers=nz,checkpoint=callback(member/'checkpoints',contract))
                np.savez_compressed(member/'fields.npz',**mesh,velocity=velocity)
                profile=surface_profile(name,mesh,velocity);np.savetxt(member/'surface-profile.txt',profile,header='x_m surface_horizontal_speed_m_year')
                outcome=dict(case=name,vertical_nodes=nz,contract=contract,status='completed',solver=info)
                if name=='slab':
                    exact=slab_surface_exact();error=float(np.max(abs(velocity[:,-1,0]-exact))/abs(exact))
                    outcome.update(analytic_surface_m_year=exact,surface_relative_error=error,agreement='pass' if error<(.2 if nz==3 else .02) else 'fail')
                else:
                    refname='rhi2a020.txt' if name=='hom-a020' else 'fpa1e000.txt'
                    refdata=np.loadtxt(REFERENCE/refname)
                    # Reference files are frozen as x_m and horizontal surface speed_m_year.
                    predicted=np.interp(refdata[:,0],profile[:,0],profile[:,1]);den=float(np.linalg.norm(refdata[:,1]))
                    error=float(np.linalg.norm(predicted-refdata[:,1])/den)
                    peak_error=float(abs(profile[:,1].max()-refdata[:,1].max())/refdata[:,1].max())
                    outcome.update(reference=refname,profile_relative_L2_error=error,peak_relative_difference=peak_error,
                                   agreement=('pass' if error<.10 and peak_error<.15 else 'fail') if nz==9 else 'diagnostic-coarse-grid')
                write_json(member/'result.json',outcome);results.append(outcome)
            except Exception as exc:
                outcome=dict(case=name,vertical_nodes=nz,contract=contract,status='technical-failure',error=repr(exc),traceback=traceback.format_exc())
                write_json(member/'failure.json',outcome);results.append(outcome);print(json.dumps(outcome),flush=True)
    restart=dict(status='not-evaluated')
    slab=root/'slab-nz9'
    if (slab/'result.json').exists():
        try:
            child=subprocess.run([sys.executable,'-m','polim_cpu.v8.run','resume',str(slab)],cwd=PROJECT,capture_output=True,text=True,timeout=180)
            (slab/'restart.stdout.log').write_text(child.stdout,encoding='utf-8');(slab/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
            if child.returncode:raise RuntimeError(child.stderr)
            with np.load(slab/'fields.npz',allow_pickle=False) as z:a=z['velocity']
            with np.load(slab/'restart.npz',allow_pickle=False) as z:b=z['velocity']
            err=float(np.linalg.norm(a-b)/np.linalg.norm(a));restart=dict(status='pass' if err<1e-12 else 'fail',relative_error=err,bitwise_equal=bool(np.array_equal(a,b)))
        except Exception as exc:restart=dict(status='technical-failure',error=repr(exc))
    vertical=[]
    for name in ('slab','hom-a020','hom-e000'):
        p3=root/f'{name}-nz3/surface-profile.txt';p9=root/f'{name}-nz9/surface-profile.txt'
        if p3.exists() and p9.exists():
            a=np.loadtxt(p3)[:,1];b=np.loadtxt(p9)[:,1]
            vertical.append(dict(case=name,relative_L2_change=float(np.linalg.norm(a-b)/np.linalg.norm(b)),meaning='two-resolution sensitivity; not proof of asymptotic convergence'))
    technical=any(x['status']=='technical-failure' for x in results) or restart['status']!='pass'
    agreement_fail=any(x.get('agreement')=='fail' for x in results)
    hashes={p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.txt')}
    summary=dict(status='technical-failure' if technical else 'engineering-agreement-failure' if agreement_fail else 'completed-engineering-agreement-pass',
        results=results,boundary=fixture,restart=restart,vertical_sensitivity=vertical,wall_seconds=time.perf_counter()-start,
        fingerprint=fingerprint,implementation=implementation(),output_hashes=hashes,maximum_model='Blatter-Pattyn',
        full_stokes_run=False,real_glacier_solved=False,time_advanced_years=0,scientific_acceptance=False,accepted_count_effect=0)
    write_json(root/'terminal-evidence.json',summary)
    print(json.dumps(dict(status=summary['status'],wall_seconds=summary['wall_seconds'],restart=restart)),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='register':register('development')
    elif sys.argv[1]=='resume':resume(sys.argv[2])
    elif sys.argv[1]=='run':execute(sys.argv[2])
    elif sys.argv[1]=='finish':register('terminal',Path(sys.argv[2]))
