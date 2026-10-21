"""One combined network transport acceptance, no real ice time integration."""
import json
import subprocess
import sys
from pathlib import Path
import time
import numpy as np
from .transport import advance,RHO
from .checkpoint import save,restore
from .adapter import bind
from .registration import active,implementation,EVIDENCE
from ..v3.runner import PROJECT,seal,write_json,digest


def serial(info):return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in info.items()}


def fixture():
    edges=np.array([[0,2],[1,2],[2,3],[3,-1]],dtype=np.int64)
    volume=np.array([100.,100.,0.,0.]); enthalpy=np.array([[10000.,12000.],[30000.,32000.],[0.,0.],[0.,0.]])
    flux=np.array([3.,2.,2.,.5]);return edges,volume,enthalpy,flux


def fixture_contract():
    edges,v,e,q=fixture();return seal({'implementation':implementation(),'edges':edges.tolist(),'initial_volume':v.tolist(),
        'initial_enthalpy':e.tolist(),'flux':q.tolist(),'dt_years':.25,'steps':8,'python':sys.version,'numpy':np.__version__})


def resume_fixture(root):
    active();root=Path(root);edges,_,_,flux=fixture();v,e,meta=restore(root/'step-000004',fixture_contract());history=meta['history']
    for step in range(5,9):
        v,e,info=advance(v,e,edges,flux,.25);history.append(serial(info));save(root,step,v,e,fixture_contract(),history)


def verify(out):
    active();out=Path(out);out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();checks=[]
    write_json(out/'launch.json',{'pid':__import__('os').getpid(),'command':'python -m polim_cpu.v4 verify','output':str(out),
        'estimated_seconds':[1,30],'checkpoint_cadence':'each synthetic step','stop_conditions':['nonfinite','negative volume','budget mismatch'],
        'purpose':'synthetic-only transport suite and static input binding'})
    # Exact two-branch confluence, independent scalar energy/mass expectation.
    edges=np.array([[0,2],[1,2]],dtype=int);v=np.array([80.,20.,0.]);e=np.array([[10000.,12000.],[30000.,32000.],[0.,0.]])
    vf,ef,info=advance(v,e,edges,np.array([80.,20.]),1.)
    assert np.array_equal(vf,[0.,0.,100.]) and np.array_equal(ef[2],[14000.,16000.])
    assert info['mass_error_m3']==0 and info['enthalpy_transport_error_j']==0
    checks.append({'name':'two tributaries fill a dry parent with exact volume-weighted enthalpy','status':'pass','received_m3':100.,'parent_enthalpy_j_kg':ef[2].tolist()})
    # Reverse flow makes a junction donor to multiple branches: total outflow cap.
    v=np.array([0.,0.,10.]);e=np.array([[0.],[0.],[25000.]])
    vf,ef,info=advance(v,e,edges,np.array([-100.,-100.]),1.)
    assert np.all(vf>=0) and np.allclose(vf[:2],[5.,5.],rtol=1e-12)
    assert abs(vf.sum()-10)<1e-12 and abs(info['enthalpy_transport_error_j'])<1e-6
    # Reordering graph edges preserves transfer because all use the same snapshot.
    vg,eg,_=advance(v,e,edges[::-1],np.array([-100.,-100.]),1.)
    assert np.array_equal(vf,vg) and np.array_equal(ef,eg)
    checks.append({'name':'reverse multi-edge outflow cannot overdraw a donor and is edge-order independent','status':'pass','mass_error_m3':info['mass_error_m3']})
    # Independent source/boundary budget, with complete ablation at one node.
    edges=np.array([[0,1],[1,-1]],dtype=int);v=np.array([20.,30.]);e=np.array([[10000.],[20000.]])
    vf,ef,info=advance(v,e,edges,np.array([10.,5.]),1.,addition=np.array([2.,0.]),addition_enthalpy=np.array([[40000.],[0.]]),removal=np.array([100.,0.]))
    assert np.array_equal(vf,[0.,35.]) and np.isclose(ef[1,0],600000/35)
    assert info['unrealized_removal_m3']==88 and info['exported_volume_m3']==5
    assert abs(info['mass_error_m3'])<1e-12 and abs(info['enthalpy_transport_error_j'])<1e-6
    checks.append({'name':'surface additions, complete ablation and outlet export close independent mass/enthalpy budgets','status':'pass','final_volume_m3':35.,'exported_volume_m3':5.})
    # Continuous baseline and independent-process restart on a synthetic network.
    edges,v,e,flux=fixture();history=[];continuous=out/'continuous';resumed=out/'resumed';contract=fixture_contract()
    save(continuous,0,v,e,contract,history);save(resumed,0,v,e,contract,history)
    for step in range(1,9):
        v,e,info=advance(v,e,edges,flux,.25);history.append(serial(info));save(continuous,step,v,e,contract,history)
        if step<=4:save(resumed,step,v,e,contract,history)
    child=subprocess.run([sys.executable,'-m','polim_cpu.v4','resume-fixture',str(resumed)],cwd=PROJECT,capture_output=True,text=True)
    (out/'restart.stdout.log').write_text(child.stdout,encoding='utf-8');(out/'restart.stderr.log').write_text(child.stderr,encoding='utf-8')
    if child.returncode:raise RuntimeError(child.stderr)
    vr,er,m=restore(resumed/'step-000008',contract)
    assert np.array_equal(v,vr) and np.array_equal(e,er) and history==m['history']
    try:restore(resumed/'step-000008','different-contract')
    except ValueError: rejected=True
    else:rejected=False
    assert rejected
    checks.append({'name':'independent-process checkpoint continuation restores all state and history exactly; wrong contract rejected','status':'pass','bitwise_equal':True})
    # This binding does not call advance or choose/solve any real-glacier velocity.
    binding=bind(PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired/input-package.json',out/'system2387-static-binding')
    assert binding['junction_edges']==7 and binding['storage_nodes']==429 and binding['geometry_nodes']==446
    checks.append({'name':'paired real input binds seven junctions without mixing centreline thickness with equivalent width','status':'pass','binding':binding})
    summary={'status':'completed','passed':len(checks),'failed':0,'checks':checks,'implementation':implementation(),
             'new_real_simulations':0,'momentum_solver_completed':False,'thermal_column_solver_completed':False,
             'wall_seconds':time.perf_counter()-started,'synthetic_contract':contract,'scientific_acceptance':False}
    write_json(out/'verification.json',summary)
    print(json.dumps({'passed':len(checks),'failed':0,'new_real_simulations':0,'binding_nodes':429}))
    return summary
