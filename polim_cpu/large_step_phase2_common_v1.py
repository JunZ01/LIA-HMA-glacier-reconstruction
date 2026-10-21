"""CPU-side constants, frozen definitions and read-only evidence audit."""
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['NUMBA_NUM_THREADS'] = '8'
import json, time, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import packed_storage_v1 as st

PROJECT = Path(__file__).resolve().parents[1]
R = PROJECT/'polim_cpu'
NAME = 'polim_v38_large_step_phase2_v1'
E = PROJECT/'validation_data/results/paleo_surface_validation_v0'/NAME
RUN = R/'runs/v38-large-step-phase2-v1'
SOURCE = R/'runs/v32r4-historical-1470-1570-v1'
P1 = R/'runs/v37r1-large-step-phase1-v1/system2387-1482-quarter'
P1E = PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_v37r1_large_step_phase1_v1'
MESH = R/'runs/v20-grid-triplet-v1/grid-50m/mesh.npz'
NETWORK = R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json'
GATES = dict(bp=1e-6,local=1e-8,mass=1e-10,volume=.005,h_rms=.2,h_max=2.,velocity=.02,
             export_fraction=.01,export_initial_fraction=1e-5,performance_factor=2.)
BUDGET = dict(bp_per_attempt=8,min_dt=1/64,rejections=16,attempts=144,
              bp_calls=1152,accepted_steps=128,case_seconds=900,batch_seconds=5400)


def reference(year):
    row = st.read(SOURCE/f'year-{year}.json')
    path = SOURCE/f"frame-{row['step']:06d}"
    state, meta = st.restore(path, st.read(SOURCE/'contract.json')['contract'])
    assert meta['year'] == float(year)
    return state, meta, path


def cpu_bp(h, u, z):
    from .v13r1 import bp
    a = bp.prepare({**st.arrays(MESH), 'thickness':h})
    residual, _, _, force = bp.evaluate(a, u, False, A=z['A'])
    return float(np.linalg.norm(residual)/max(np.linalg.norm(force),1.)), a['wet']


def audit_phase1():
    """Evaluate saved states algebraically; never advance or solve BP again."""
    from scipy import sparse
    from .large_step_transport_v1 import ExistingMeshFluxAdapter
    cfg = st.read(P1/'config.json')
    final, meta = st.restore(P1/'frame-000001', st.read(P1/'contract.json')['contract'])
    old, _, _ = reference(1482)
    z = st.arrays(SOURCE/'input.npz'); area=z['geo_dual_area']
    forcing = st.arrays(SOURCE/'forcing-1482.npz')
    assert st.digest(SOURCE/'forcing-1482.npz') == st.read(SOURCE/'forcing-1482.json')['sha256']
    assert np.array_equal(forcing['surface'], z['bed']+old['h'])
    h,u,smb,dt = final['h'],final['u'],forcing['smb'],.25
    rates = ExistingMeshFluxAdapter(z).rates(u)
    matrix = sparse.diags(area)+dt*rates.outward_operator()
    rhs=area*(old['h']+dt*smb)
    multiplier=np.maximum(matrix@h-rhs,0.)
    actual=smb+multiplier/(dt*area)
    exact_hash=st.array_hash(actual)==meta['actual_smb_sha256']
    bp,wet=cpu_bp(h,u,z)
    export=dt*rates.boundary_export_rate(h);source=float(dt*(area@actual));v0=float(area@old['h'])
    local=np.abs(matrix@h-rhs-multiplier).sum()/max(v0+abs(source)+dt*rates.absolute_flux_rate(h),1.)
    mass=abs(float(area@h)-v0-source+export)/max(v0,abs(source),abs(export),1.)
    allowed=dt*area*np.maximum(-smb,0.)
    rounding=4096*np.finfo(float).eps*max(float(np.abs(rhs).max()),1.)
    passed=bool(exact_hash and cfg['outer_damping']==1. and meta['history'][-1]['fixed_point_relative']==0.
                and bp<=GATES['bp'] and local<=GATES['local'] and mass<=GATES['mass']
                and np.all(multiplier<=allowed+rounding) and np.all(multiplier[h>0]<=rounding)
                and all(np.isfinite(a).all() for a in (h,u,actual,multiplier)) and np.all(h>=0))
    return dict(status='pass' if passed else 'fail', original_CPU_BP=bp, mass_relative=mass,
                local_relative=float(local), actual_smb_hash_matches=exact_hash, damping=cfg['outer_damping'],
                saved_fixed_point_relative=meta['history'][-1]['fixed_point_relative'],
                h_sha256=st.array_hash(h),u_sha256=st.array_hash(u),BP_active_sha256=st.array_hash(wet),
                transport_active_sha256=st.array_hash(h==0),flux_rate_sha256=st.array_hash(rates.internal_signed),
                actual_source_volume=source,boundary_export_volume=export,
                potential_source_volume=float(dt*(area@smb)),unmet_ablation_volume=float(multiplier.sum()),
                no_BP_solve_or_time_advancement=True,phase1_validation_not_replayed=True)


def masks(start,z):
    old,_,_=reference(start); wet=old['h']>0; edges=z['geo_edges']
    margin=np.zeros(len(wet),bool);cross=wet[edges[:,0]]!=wet[edges[:,1]]
    margin[np.unique(edges[cross])]=True
    for _ in range(2):
        adjacent=margin[edges].any(axis=1);margin[np.unique(edges[adjacent])]=True
    net=st.read(NETWORK)
    points=np.asarray([b['flows_to_point_projected_m'] for b in net['branches'] if b['parent_index'] is not None])
    junction=np.any(np.sum((z['xy'][:,None,:]-points[None,:,:])**2,axis=2)<=500.**2,axis=1)
    speed=np.sqrt(np.sum(old['u'].reshape(-1,2,2)[:,-1,:]**2,axis=1))
    threshold=float(np.quantile(speed[wet],.9)); high=wet&(speed>=threshold)
    return dict(evaluation=np.ones(len(wet),bool),ice_edge=margin,junction=junction,original_fast=high),threshold
