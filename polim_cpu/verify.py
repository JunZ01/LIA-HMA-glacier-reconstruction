"""One consolidated verification after implementation; no real glacier input."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
from dataclasses import replace
import json
import subprocess
import sys
import numpy as np
from scipy.io import savemat,loadmat
from scipy.sparse.linalg import spsolve
from .core import Grid,Parameters,assemble,rheology,vertical_velocity,to_main,solve_velocity
from .runner import demo_config,input_geometry,run,digest,write_json,batch,ROOT

def prepare_oracle(out):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    p=Parameters(layers=9)
    cfg=demo_config(); cfg['geometry']['points']=13
    grid=Grid(*input_geometry(cfg),p.layers)
    z=grid.zeta[:,None]; x=grid.xs[None,:]/grid.x[-1]
    viscosity=3e5*(1+.2*z+.15*np.cos(np.pi*x))
    utest=30*np.sin(np.pi*x)**2*(2*z-z*z)
    savemat(out/'input.mat',{'x':grid.x,'bed':grid.bed,'thickness':grid.thickness,'width':grid.width,
                            'layers':p.layers,'visc_input':viscosity,'u_test':utest,'glen_a':p.glen_a})
    script="""global xi M Ms N dx dzeta zeta hS hB H W hS_s hB_s H_s W_s
global dhSdx_s dhSdx dhBdx_s dhBdx dzetadx_s dzetadx
global is_active is_active_s i_term i_term_s
global iter_u u_s_lst enth_lst hWL Neff_lst beta2_s usurf_s_obs
load('input.mat');
xi=x; M=numel(x); Ms=M+1; N=layers; dx=x(2)-x(1);
dzeta=1/(N-1); zeta=linspace(0,1,N)'; H=thickness; hB=bed; hS=bed+H;
W=ones(N,1)*width; is_active=[]; Neff_lst=[]; usurf_s_obs=[];
iter_u=1; u_s_lst=[]; enth_lst=[]; hWL=0; beta2_s=1000*ones(1,Ms);
set_staggered_grid(); p=params(); p.is_SBC_width=0; p.type_LBC='zero';
AGlen_s=glen_a*ones(N,Ms); visc_s=visc_input; visc=staggerX2main(visc_s);
for mode=1:3
    p.is_flowband=(mode==3);
    if mode==2; p.type_BBC='LFlaw'; else; p.type_BBC='zero'; end
    [u_ref,linear_system]=solver_u_core(visc_s,visc,AGlen_s,p);
    A_ref=linear_system.LHS; b_ref=linear_system.RHS;
    [eta_ref,eta_main_ref,heat_ref]=get_ice_viscosity(u_test,staggerX2main(u_test),AGlen_s,p);
    [w_vs_ref,w_ref]=get_ice_w(u_ref,staggerX2main(u_ref));
    save(sprintf('oracle-%d.mat',mode),'A_ref','b_ref','u_ref','eta_ref','heat_ref','w_ref','dzetadx_s','dhSdx_s');
end
disp('PoLIM synthetic oracle complete');
"""
    (out/'oracle.m').write_text(script)
    print(str(out))

def verify(out,oracle):
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    checks={}
    def record(name,condition,**metrics):
        checks[name]={'pass':bool(condition),**metrics}
    p=Parameters(layers=9)
    data=loadmat(Path(oracle)/'input.mat',squeeze_me=True)
    grid=Grid(data['x'],data['bed'],data['thickness'],data['width'],9)
    for mode in (1,2,3):
        pm=replace(p,basal='linear' if mode==2 else 'zero',flowband=mode==3)
        ref_path=Path(oracle)/f'oracle-{mode}.mat'
        if not ref_path.exists():
            checks[f'matlab_mode_{mode}']={'pass':None,'status':'unavailable','reason':'MATLAB batch initialization did not return a reference output'}
            continue
        ref=loadmat(ref_path)
        A,b=assemble(grid,data['visc_input'],pm)
        delta=A-ref['A_ref']; max_a=float(np.max(np.abs(delta.data))) if delta.nnz else 0.
        u=spsolve(A,b).reshape(grid.metric.shape,order='F')
        eta,heat=rheology(grid,data['u_test'],pm)
        metrics={
            'matrix_max_abs':max_a,
            'rhs_max_abs':float(np.max(np.abs(b-ref['b_ref'].ravel()))),
            'velocity_max_abs_m_per_year':float(np.max(np.abs(u-ref['u_ref']))),
            'viscosity_max_relative':float(np.max(np.abs(eta-ref['eta_ref'])/ref['eta_ref'])),
            'heat_max_relative':float(np.max(np.abs(heat-ref['heat_ref'])/np.maximum(np.abs(ref['heat_ref']),1e-30))),
            'vertical_velocity_max_abs':float(np.max(np.abs(vertical_velocity(grid,u)-ref['w_ref']))),
            'metric_max_abs':float(np.max(np.abs(grid.metric-ref['dzetadx_s'])))}
        condition=(np.allclose(A.toarray(),ref['A_ref'].toarray(),rtol=1e-12,atol=1e-7)
                   and metrics['rhs_max_abs']<1e-10 and metrics['velocity_max_abs_m_per_year']<1e-8
                   and metrics['viscosity_max_relative']<1e-12 and metrics['heat_max_relative']<1e-12
                   and metrics['vertical_velocity_max_abs']<1e-8 and metrics['metric_max_abs']<1e-12)
        record(f'matlab_mode_{mode}',condition,**metrics)
    cfg=demo_config(); cfg['geometry']['points']=17; cfg['parameters']={'layers':9}
    cfg['steps']=4
    full=run(cfg,out/'continuous',make_report=False)
    # Resume in a genuinely different Python process with the same code and input.
    run(cfg,out/'resumed',steps=2,make_report=False)
    proc=subprocess.run([sys.executable,'-m','polim_cpu','resume',str(out/'resumed'),'--steps','4'],capture_output=True,text=True,cwd=ROOT.parent)
    (out/'resume-process.log').write_text(proc.stdout+proc.stderr)
    equal=False
    if proc.returncode==0:
        with np.load(out/'continuous/step-000004/state.npz') as a,np.load(out/'resumed/step-000004/state.npz') as b:
            equal=all(np.array_equal(a[k],b[k]) for k in a.files)
        ma=json.loads((out/'continuous/step-000004/checkpoint.json').read_text())
        mb=json.loads((out/'resumed/step-000004/checkpoint.json').read_text())
        equal=equal and ma['history']==mb['history'] and ma['clock_years']==mb['clock_years']
    record('separate_process_resume',equal,exit_code=proc.returncode)
    mass=max(abs(r['cumulative_mass_error_m3']) for r in full['history'])
    residual=max(r['momentum_relative_residual'] for r in full['history'])
    record('closed_domain_mass',mass<1e-9*full['history'][0]['volume_m3'],max_error_m3=mass)
    record('nonlinear_convergence',all(r['velocity_converged'] for r in full['history']) and residual<1e-4,max_momentum_relative_residual=residual)
    rest=Grid(grid.x,np.zeros_like(grid.x),np.full_like(grid.x,100.),grid.width,9)
    result=solve_velocity(rest,p)
    record('horizontal_slab_at_rest',np.max(np.abs(result['u']))==0 and result['converged'])
    # Interior of a very long parallel-sided slab approaches this exact
    # constant-viscosity solution of the transformed momentum equation.
    xs=np.linspace(0,20000,101); slope=.05; height=100.; eta0=3e5
    slab=Grid(xs,3000-slope*xs,np.full_like(xs,height),np.full_like(xs,800.),21)
    aa,bb=assemble(slab,np.full(slab.metric.shape,eta0),Parameters())
    uu=spsolve(aa,bb).reshape(slab.metric.shape,order='F')
    analytic=p.rho*p.gravity*slope*height**2/(2*eta0*(1+4*slope**2))*(2*slab.zeta-slab.zeta**2)
    slab_error=float(np.max(np.abs(uu[:,51]-analytic))/np.max(analytic))
    record('parallel_slab_analytic_interior',slab_error<1e-4,max_relative_error=slab_error)
    # A failed configuration must not prevent the next valid case from running.
    bad={**cfg,'parameters':{'basal':'unsupported','layers':9}}
    good={**cfg,'steps':0}
    write_json(out/'bad.json',bad); write_json(out/'good.json',good)
    results=batch([out/'bad.json',out/'good.json'],out/'batch')
    record('batch_failure_isolation',results[0]['status']=='failed' and results[1]['status'].startswith('completed'))
    # Report failure must not destroy committed numerical results.
    from . import report
    render=report.render
    def fail_report(*args): raise OSError('injected report failure')
    report.render=fail_report
    try:
        survived=run(good,out/'report-failure')
        record('report_failure_nonfatal',survived['step']==0 and (out/'report-failure/step-000000/checkpoint.json').exists())
    finally: report.render=render
    # Finite-but-corrupt state must be rejected before continuation.
    corrupt=out/'report-failure/step-000000/state.npz'
    with corrupt.open('ab') as f: f.write(b'corrupt')
    try:
        run(None,out/'report-failure',resume=True,steps=1,make_report=False)
        rejected=False
    except ValueError: rejected=True
    record('corrupt_checkpoint_rejected',rejected)
    actual=[v['pass'] for v in checks.values() if v['pass'] is not None]
    status='fail' if not all(actual) else ('partial' if any(v['pass'] is None for v in checks.values()) else 'pass')
    summary={'status':status,'checks':checks,
             'scope':'synthetic engineering checks; not physical validation or LIA acceptance'}
    write_json(out/'verification.json',summary)
    print(json.dumps(summary,indent=2))
    return summary['status']=='pass'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','check']); parser.add_argument('--out',required=True); parser.add_argument('--oracle')
    args=parser.parse_args()
    if args.action=='prepare': prepare_oracle(args.out)
    else: sys.exit(0 if verify(args.out,args.oracle) else 1)
