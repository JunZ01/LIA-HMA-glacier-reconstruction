"""CPU split coupling; branch ends have natural BP stress conditions.

Junctions exchange mass and enthalpy only. No shared junction momentum DOFs,
vector turning stress, or vertical remapping is implied by this model.
"""
import numpy as np
from ..v3.momentum import Backend
from ..v3.model import diagnose, columns, initial_E, face_flux, mean_vertical
from ..v3.physics import RHO, CP, TREF
from ..v4.transport import advance


def fixture():
    """Three explicitly synthetic flowbands; never load a glacier input here."""
    b = Backend('cpu')
    p = dict(glen_a_pa_3_s_1=2.4e-24, glen_n=3, basal='zero',
             flowband=False, thermal_feedback=True, geothermal_w_m2=.05,
             water_cap=.03, momentum_transition_m=1., relative_tolerance=1e-6,
             max_iterations=80, strain_regularization=1e-30)
    branches = []
    for k, (height, width, slope, surface) in enumerate(
            [(65.,180.,.08,-12.),(80.,220.,.065,-6.),(95.,400.,.045,-10.)]):
        x = np.arange(7,dtype=float)*100
        h = np.full(7,height)
        state = dict(x=x, bed=3500-slope*x, h=h, width=np.full(7,width),
                     cell_width=np.array([50.,100.,100.,100.,100.,100.,50.]),
                     melt=np.zeros(7), surface_c=np.full(7,surface), smb=np.zeros(7))
        state['E'] = initial_E(h,state['surface_c'],5,.05,b)
        branches.append(state)
    # Each tributary ends at the second node of the main branch.
    return branches, [(0,2,1),(1,2,1)], p


def diagnose_all(branches,p):
    b=Backend('cpu'); infos=[]
    for state in branches:
        infos.append(diagnose(state,b,p,state.get('u')))
    return infos


def graph_flux(branches,junctions):
    b=Backend('cpu'); offsets=np.r_[0,np.cumsum([len(s['h']) for s in branches])]
    targets={}; edges=[]; flux=[]; junction_edges=[]
    for child,parent,node in junctions:
        if child in targets or child==parent or not 0<=child<len(branches) or not 0<=parent<len(branches):
            raise ValueError('Invalid or duplicate junction')
        if not 0<=node<len(branches[parent]['h']): raise ValueError('Invalid receiving node')
        targets[child]=(parent,node)
    for k in targets:
        seen=set(); at=k
        while at in targets:
            if at in seen: raise ValueError('Cyclic branch graph')
            seen.add(at); at=targets[at][0]
    for k,state in enumerate(branches):
        f,_=face_flux(state,b); count=len(state['h']); start=int(offsets[k])
        for i in range(count-1):
            edges.append([start+i,start+i+1]);flux.append(float(f[i+1]))
        velocity=float(mean_vertical(state['u'],b)[-1])
        end=start+count-1
        if k in targets:
            parent,node=targets[k]; receiver=int(offsets[parent]+node)
            # Tip-normal scalar velocity, donor thickness, child mouth width.
            donor_h=state['h'][-1] if velocity>=0 else branches[parent]['h'][node]
            junction_edges.append(len(edges));edges.append([end,receiver])
            flux.append(float(velocity*donor_h*state['width'][-1]))
        else:
            if velocity < -1e-10:
                raise ValueError('Open outlet requests external inflow; explicit exterior state required')
            edges.append([end,-1]);flux.append(float(max(velocity,0)*state['h'][-1]*state['width'][-1]))
    return np.array(edges,dtype=np.int64),np.array(flux),offsets,np.array(junction_edges,dtype=int)


def advance_coupled(branches,junctions,p,requested_dt):
    """Requires diagnosed input; updates branches only after all steps succeed."""
    b=Backend('cpu'); states=[{k:v.copy() for k,v in s.items()} for s in branches]
    edges,flux,offsets,joins=graph_flux(states,junctions)
    area=np.concatenate([s['width']*s['cell_width'] for s in states])
    h=np.concatenate([s['h'] for s in states]);volume=h*area
    enthalpy=np.concatenate([s['E'] for s in states]);layers=enthalpy.shape[1]
    weights=np.ones(layers)/(layers-1);weights[[0,-1]]*=.5
    # Restrict the total outgoing fraction, including confluence and outlet edges.
    donor=np.where(flux>=0,edges[:,0],edges[:,1]);outgoing=np.zeros(len(h))
    np.add.at(outgoing,donor,abs(flux)); active=(outgoing>0)&(volume>0)
    dt=min(float(requested_dt),float(np.min(.4*volume[active]/outgoing[active])) if active.any() else float(requested_dt))
    if not np.isfinite(dt) or dt<=1e-12:raise ValueError('Coupled timestep collapsed')
    smb=np.concatenate([s['smb'] for s in states]);melt=np.concatenate([s['melt'] for s in states])
    surface=np.concatenate([s['surface_c'] for s in states])
    source_E=np.repeat((CP*(surface+273.15-TREF))[:,None],layers,axis=1)
    new_v,new_E,tr=advance(volume,enthalpy,edges,flux,dt,layer_weights=weights,
        addition=np.maximum(smb,0)*area*dt,addition_enthalpy=source_E,
        removal=(np.maximum(-smb,0)+melt)*area*dt)
    # A dry column retains its boundary-temperature placeholder, with zero energy mass.
    new_E[new_v==0]=source_E[new_v==0]
    thermal=[]
    for k,s in enumerate(states):
        lo,hi=offsets[k:k+2];old_h=s['h'].copy();s['h']=new_v[lo:hi]/area[lo:hi];s['E']=new_E[lo:hi].copy()
        energy_before=RHO*float(np.sum(new_v[lo:hi]*mean_vertical(s['E'],b)))
        th=columns(s,old_h,s['surface_c'],dt,b,p)
        energy_after=RHO*float(np.sum(new_v[lo:hi]*mean_vertical(s['E'],b)))
        th['column_energy_change_j']=energy_after-energy_before
        thermal.append(th)
    info=diagnose_all(states,p)
    for s in states:
        if any(not np.isfinite(v).all() for v in s.values()) or np.any(s['h']<0):
            raise RuntimeError('Invalid coupled state; previous checkpoint retained')
    return states,dict(dt_years=dt,transport={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in tr.items()},
        thermal=thermal,momentum=info,junction_transfer_m3=tr['edge_transfer_m3'][joins].tolist(),
        junction_flux_m3_year=flux[joins].tolist(),volume_m3=float(new_v.sum()),
        branch_volumes_m3=[float(np.sum(s['h']*s['width']*s['cell_width'])) for s in states],
        max_speed_m_year=max(float(np.max(abs(s['u']))) for s in states),
        min_temperature_c=min(float(np.min(s['temperature']-273.15)) for s in states),
        max_water_fraction=max(float(np.max(s['water_fraction'])) for s in states))
