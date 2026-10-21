"""Annual native state binding; no model evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np
from polim_cpu import packed_storage_v1 as st

def identity(mesh,climate,parameters):
    return dict(mesh_sha256=st.digest(mesh),climate_sha256=st.digest(climate),parameters=parameters,
                method='v35-exact/projected-AMG/float64/CFL.35/cap30days/no-slip',version='native-candidate-v1',worker_sha256=st.digest(Path(__file__).parent/'worker.py'),solver_sha256=st.digest(Path(__file__).parent.parent/'adaptive_bp_solve_v1.py'))

def load_bundle(path,expected):
    p=Path(path);b=st.read(p)
    if b['kind']!='accepted-native-annual' or b['identity']!=expected:raise ValueError('Checkpoint identity is not an accepted matching native annual state')
    if st.digest(Path(b['frame'])/'state.npz')!=b['frame_sha256']:raise ValueError('Native checkpoint file hash mismatch')
    a,m=st.restore(b['frame'],b['contract'])
    if m!=b['metadata'] or float(m['year'])!=b['year']:raise ValueError('Checkpoint time/control metadata mismatch')
    if b.get('annual_ledger_path') and st.digest(b['annual_ledger_path'])!=b['annual_ledger_sha256']:raise ValueError('Annual ledger hash mismatch')
    if b.get('forcing'):
        f=b['forcing']
        if st.digest(f['path'])!=f['sha256']:raise ValueError('Prior annual climate cache mismatch')
    if not np.isfinite(a['h']).all() or not np.isfinite(a['u']).all() or (a['h']<0).any():raise ValueError('Invalid accepted state arrays')
    if b['window_ledger']['mass_relative']>1e-10:raise ValueError('Window ledger not accepted')
    return a,m,b

def ledger(V0,volume,source,exported):
    error=volume-V0-source+exported
    return dict(V0=V0,volume=volume,source=source,exported=exported,mass_error=error,mass_relative=abs(error)/max(V0,abs(source),1.))

def seal(path,frame,contract,ident,window,origin,forcing=None,annual_ledger_path=None):
    a,m=st.restore(frame,contract)
    b=dict(kind='accepted-native-annual',year=m['year'],frame=str(Path(frame).resolve()),frame_sha256=st.digest(Path(frame)/'state.npz'),contract=contract,metadata=m,identity=ident,window_ledger=window,window_origin=origin,forcing=forcing,
           annual_ledger_path=str(annual_ledger_path) if annual_ledger_path else None,annual_ledger_sha256=st.digest(annual_ledger_path) if annual_ledger_path else None,
           predictor_restart='Original v35 annual restored frame has step>0: prev_u=None, prev_dt=None; current u and last_solve retained. No GPU/AMG retained.',step_controller=dict(cap_days=30,CFL=.35,last_solve=m['last_solve'],step=m['step']))
    st.write(path,b);load_bundle(path,ident);return b
