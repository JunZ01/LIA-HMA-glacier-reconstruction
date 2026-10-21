"""Global BP + conservative median-dual transport + split thermal columns.

The initial footprint is an open computational domain. Ice can leave, not enter;
no external ice reservoir or advancing ice outside this first-version domain.
All tributaries share the original conforming mesh. Units m,Pa,year,J/kg.
"""
import numpy as np
from . import bp
from ..v4.transport import advance
from ..v3.momentum import Backend
from ..v3.model import initial_E,climate_at
from ..v3.physics import glen,thermal_state,thermal_step,CP,TREF,RHO,YEAR,LATENT

B=Backend('cpu');NZ=9;WEIGHTS=np.r_[.5,np.ones(NZ-2),.5]/(NZ-1)


def topology(mesh):
    tri=mesh['triangles'];xy=mesh['xy'];p=xy[tri];n=len(xy)
    area=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])*.5
    if np.any(area<=0):raise ValueError('Inverted mesh')
    dual=np.bincount(tri.ravel(),weights=np.repeat(area/3,3),minlength=n)
    a=np.concatenate([tri[:,0],tri[:,1],tri[:,2]]);b=np.concatenate([tri[:,1],tri[:,2],tri[:,0]])
    third=np.concatenate([tri[:,2],tri[:,0],tri[:,1]])
    centre=xy[tri].mean(axis=1);delta=np.tile(centre,(3,1))-(xy[a]+xy[b])*.5
    normal=np.column_stack([delta[:,1],-delta[:,0]])
    reverse=a>b;normal[reverse]*=-1
    edges,inverse=np.unique(np.sort(np.column_stack([a,b]),axis=1),axis=0,return_inverse=True)
    # Physical exterior pieces are half of each oriented outer edge.
    keys=np.sort(np.column_stack([a,b]),axis=1);_,first,count=np.unique(keys,axis=0,return_index=True,return_counts=True)
    idx=first[count==1];oa=a[idx];ob=b[idx];tangent=xy[ob]-xy[oa]
    outer_normal=np.column_stack([tangent[:,1],-tangent[:,0]])*.5
    return dict(area=area,dual_area=dual,edges=edges,inverse=inverse,a=a,b=b,third=third,normal=normal,
                outer_a=oa,outer_b=ob,outer_normal=outer_normal)


def fluxes(mesh,geo,u):
    mean=.5*u[:,1]+.25*u[:,2]  # basal no-slip and trapezoidal sigma integration
    a,b,c=geo['a'],geo['b'],geo['third']
    velocity=5/12*(mean[a]+mean[b])+1/6*mean[c]
    integrated=np.bincount(geo['inverse'],weights=np.sum(velocity*geo['normal'],axis=1),minlength=len(geo['edges']))
    edges=geo['edges'];donor=np.where(integrated>=0,edges[:,0],edges[:,1]);q=integrated*mesh['thickness'][donor]
    oa,ob=geo['outer_a'],geo['outer_b'];normal=geo['outer_normal']
    exterior=np.r_[np.maximum(np.sum((.75*mean[oa]+.25*mean[ob])*normal,axis=1),0)*mesh['thickness'][oa],
                   np.maximum(np.sum((.75*mean[ob]+.25*mean[oa])*normal,axis=1),0)*mesh['thickness'][ob]]
    links=np.vstack([edges,np.column_stack([np.r_[oa,ob],np.full(2*len(oa),-1)])])
    return links,np.r_[q,exterior]


def fluidity(E,h,parameters):
    reference=3.985e-13*np.exp(-60000/(8.314*263.15))*YEAR
    values=glen(E,h,B)*parameters['glen_a_pa_3_s_1']*YEAR/reference
    return values[:,[0,4,8]]


def strain_heat(mesh,u,A,geo):
    # Reuse BP's physical-gradient stress diagnostic, projecting density by area.
    active=np.sum(mesh['thickness'][mesh['triangles']],axis=1)>0
    stress=bp.stress_centroids(mesh,u,A=A)
    heat=4*stress[:,:,7]*np.maximum(stress[:,:,6]**2-bp.REG,0.)
    tri=mesh['triangles'][active];area=geo['area'][active];den=np.bincount(tri.ravel(),weights=np.repeat(area,3),minlength=len(u))
    nodal=np.zeros((len(u),2))
    for j in range(2):nodal[:,j]=np.divide(np.bincount(tri.ravel(),weights=np.repeat(area*heat[:,j],3),minlength=len(u)),den,out=np.zeros(len(u)),where=den>0)
    sigma=np.linspace(0,1,NZ)
    return np.column_stack([nodal[:,0] if s<=.25 else nodal[:,1] if s>=.75 else nodal[:,0]*(.75-s)*2+nodal[:,1]*(s-.25)*2 for s in sigma])


def columns(E,h,surface,heat,dt,parameters,dual_area):
    """Existing implicit enthalpy column solver; horizontal transport done once.

    No resolved vertical advection in this simplified sigma-layer model. The
    dummy index passed to the reused column API has no horizontal operator:
    u,w and thickness-rate arguments are zero, horizontal_advection=False.
    """
    result=E.copy();melt=np.zeros(len(h));drain=0.;phase=0
    wet=np.flatnonzero(h>=1.)
    for begin in range(0,len(wet),4096):
        ids=wet[begin:begin+4096];real_count=len(ids)
        if real_count<3:ids=np.resize(ids,3)
        zero=np.zeros((len(ids),NZ));settings={**parameters,'surface_temperature_c':surface[ids],'horizontal_advection':False,'basal':'zero'}
        after,local,info=thermal_step(np.arange(len(ids),dtype=float),np.zeros(len(ids)),h[ids],np.ones(len(ids)),
            E[ids],zero,zero,heat[ids],np.zeros(len(ids)),dt,B,settings)
        result[ids[:real_count]]=after[:real_count];melt[ids[:real_count]]=local[:real_count]
        drain+=RHO*float(np.sum(h[ids[:real_count]]*dual_area[ids[:real_count]]*(info['drained_specific_energy'][:real_count]@WEIGHTS)))
        phase=max(phase,info['phase_iterations'])
    thin=h<1.;target=CP*(surface+273.15-TREF)
    thin_exchange=RHO*float(np.sum(h[thin]*dual_area[thin]*(target[thin]-E[thin]@WEIGHTS)))
    result[thin]=target[thin,None]
    _,_,_,ep=thermal_state(result,h,B);excess=np.maximum(result-ep-LATENT*parameters['water_cap'],0)
    drain+=RHO*float(np.sum(h*dual_area*(excess@WEIGHTS)));result-=excess
    change=RHO*float(np.sum(h*dual_area*((result-E)@WEIGHTS)))
    return result,melt,dict(column_energy_change_J=change,drained_energy_J=drain,thin_exchange_J=thin_exchange,phase_iterations=phase)


def initialize(mesh,parameters,climate=None):
    h=mesh['thickness'].copy();surface=mesh['bed']+h
    if climate is None:
        top=np.full(len(h),-10.);smb=np.where(mesh['xy'][:,1]<650,.3,-.2)
    else:smb,top=climate_at(surface,climate,parameters,1441,B)
    E=initial_E(h,top,NZ,.05,B)
    return dict(h=h,E=E,melt=np.zeros(len(h)),surface_c=top,smb=smb,time=0.,step=0)


def advance_window(mesh,geo,state,u,heat,end_time,parameters):
    stats=[];state={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in state.items()}
    while state['time']<end_time-1e-16:
        mesh={**mesh,'thickness':state['h']};edges,q=fluxes(mesh,geo,u)
        volume=state['h']*geo['dual_area'];donor=np.where(q>=0,edges[:,0],edges[:,1]);outgoing=np.bincount(donor,weights=abs(q),minlength=len(volume))
        positive=(volume>0)&(outgoing>0);dt=min(end_time-state['time'],float(np.min(.35*volume[positive]/outgoing[positive])) if positive.any() else end_time-state['time'])
        if not np.isfinite(dt) or dt<1e-13:raise RuntimeError('CFL timestep collapsed')
        source_E=np.broadcast_to(CP*(state['surface_c'][:,None]+273.15-TREF),state['E'].shape)
        volume,E,tr=advance(volume,state['E'],edges,q,dt,layer_weights=WEIGHTS,
            addition=np.maximum(state['smb'],0)*geo['dual_area']*dt,addition_enthalpy=source_E,
            removal=(np.maximum(-state['smb'],0)+state['melt'])*geo['dual_area']*dt)
        h=volume/geo['dual_area'];E,melt,thermal=columns(E,h,state['surface_c'],heat,dt,parameters,geo['dual_area'])
        state.update(h=h,E=E,melt=melt,time=state['time']+dt,step=state['step']+1)
        if not all(np.isfinite(state[k]).all() for k in ('h','E','melt')) or np.any(h<0):raise RuntimeError('Invalid evolved state')
        if state['step']>2048:raise RuntimeError('Registered substep budget reached')
        # Keep full edge transfers out of status; the conservative ledger is saved.
        tr={k:v for k,v in tr.items() if not isinstance(v,np.ndarray)}
        stats.append(dict(time=state['time'],dt=dt,transport=tr,thermal=thermal))
    return state,stats
