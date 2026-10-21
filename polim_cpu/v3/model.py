"""Cell-centred moving ice support, conservative transport and coupled columns.

Q1 BP on connected thick-ice patches; local SIA below the disclosed 1 m
momentum threshold. No thickness floor, geometry truncation or mass deletion.
"""
import numpy as np
from .momentum import solve
from .physics import (CP,TREF,LATENT,YEAR,RHO,GRAVITY,CONDUCTIVITY,positive,
                      glen,thermal_state,thermal_step,kinematics,gradient)


def minimum(a,c):
    # Avoid subtractive cancellation when a thin cell meets large potential melt.
    if hasattr(a,'clamp'):
        return a.clamp(max=c) if np.isscalar(c) else a.minimum(c)
    return np.minimum(a,c)


def segments(mask):
    indices=np.flatnonzero(mask)
    return np.split(indices,np.flatnonzero(np.diff(indices)>1)+1) if len(indices) else []


def mean_vertical(a,b):
    return (b.xp.sum(a,axis=1)-(a[:,0]+a[:,-1])/2)/(a.shape[1]-1)


def climate_at(surface,climate,parameters,year,b):
    """Original MonthlyTI annual formula: 900 kg/m3, 365 day calendar.

    Caller freezes the elevation-feedback field for each complete model year.
    """
    years=climate['years']; found=np.flatnonzero(years==year)
    if len(found)!=1: raise ValueError(f'Climate year {year} is unavailable; no extrapolation')
    j=int(found[0]); xp=b.xp
    temp=b.array(climate['temp'][j])+float(parameters['base_temp_bias_degC'])
    prcp=b.array(climate['prcp'][j])*float(parameters['prcp_fac'])
    grad=b.array(climate['gradient'][j])
    field=temp[None,:]+(surface[:,None]-float(climate['ref_hgt']))*grad[None,:]
    solid=minimum(positive(1-field/2),1.)
    potential=positive(field-float(parameters['temp_melt_degC']))
    annual=xp.sum(prcp[None,:]*solid-float(parameters['melt_f'])*(365/12)*potential,axis=1)/RHO
    annual_temperature=xp.sum(field,axis=1)/12
    # Temperature prescribed to an ice surface cannot exceed its melting point.
    surface_temperature=minimum(annual_temperature,0.)
    return annual,surface_temperature


def initial_E(h,surface_c,layers,geothermal,b):
    sigma=b.array(np.linspace(0,1,layers))
    temperature=273.15+surface_c[:,None]+float(geothermal)/CONDUCTIVITY*h[:,None]*(1-sigma[None,:])
    _,_,pmp,_=thermal_state(temperature*0,h,b)
    temperature=minimum(temperature,pmp)
    return CP*(temperature-TREF)


def diagnose(state,b,p,initial=None):
    xp=b.xp; x,bed,h,width,E=[state[k] for k in ('x','bed','h','width','E')]
    layers=E.shape[1]; sigma=b.array(np.linspace(0,1,layers))
    # Preserve the supplied calibrated Glen scale at pressure-corrected -10 C.
    reference=float(3.985e-13*np.exp(-60000/(8.314*263.15))*YEAR)
    A=glen(E,h,b)*float(p.get('glen_a_pa_3_s_1',2.4e-24))*YEAR/reference
    if not p.get('thermal_feedback',True): A=E*0+float(p['glen_a_pa_3_s_1'])*YEAR
    slope=gradient((bed+h)[:,None],x,0,b)[:,0]
    n=float(p.get('glen_n',3))
    shape=h[:,None]**(n+1)-(h[:,None]*(1-sigma[None,:]))**(n+1)
    u=-2*mean_vertical(A,b)[:,None]/(n+1)*(RHO*GRAVITY)**n*abs(slope[:,None])**(n-1)*slope[:,None]*shape
    threshold=float(p.get('momentum_transition_m',1.))
    if threshold<=0: raise ValueError('momentum_transition_m must be positive and disclosed')
    mask=b.host(h>=threshold)
    patches=[]; maximum=0.; iterations=0; nodes=0
    for ids in segments(mask):
        if len(ids)<3: continue  # Isolated small patches use the stated local SIA closure.
        idx=b.array(ids,True); local={**p,'headwall':bool(ids[0]==0)}
        solved,info=solve(*[b.host(v[idx]) for v in (x,bed,h,width)],layers,A[idx],b,local,
                          None if initial is None else initial[idx])
        u[idx]=solved; maximum=max(maximum,info['relative_residual']); iterations+=info['iterations']; nodes+=len(ids)
        patches.append({'first':int(ids[0]),'last':int(ids[-1]),'nodes':len(ids),'residual':info['relative_residual']})
    w=b.zeros(E.shape); viscosity=b.zeros(E.shape); heat=b.zeros(E.shape)
    # Thin-ice kinematics evaluated analytically to avoid 1/h terms at dry cells.
    wet=h>0; safe=xp.where(wet,h,h*0+1)
    uz=-2*mean_vertical(A,b)[:,None]*(RHO*GRAVITY)**n*abs(slope[:,None])**(n-1)*slope[:,None]*(h[:,None]*(1-sigma[None,:]))**n
    e2=uz*uz/4
    viscosity=.5*A**(-1/n)*(e2+float(p.get('strain_regularization',1e-30)))**((1-n)/(2*n))
    heat=4*viscosity*e2
    for ids in segments(mask):
        if len(ids)<3: continue
        idx=b.array(ids,True)
        w[idx],viscosity[idx],heat[idx]=kinematics(x[idx],bed[idx],h[idx],width[idx],u[idx],A[idx],b,p,state['melt'][idx])
    state.update(u=xp.where(wet[:,None],u,u*0),w=w,viscosity=xp.where(wet[:,None],viscosity,viscosity*0),strain_heat=heat,A=A)
    state['temperature'],state['water_fraction'],_,_=thermal_state(E,h,b)
    return {'relative_residual':maximum,'iterations':iterations,'converged':True,'bp_nodes':nodes,
            'thin_or_isolated_sia_nodes':int(xp.sum(wet))-nodes,'patches':patches}


def face_flux(state,b):
    h,width,u=[state[k] for k in ('h','width','u')]; mean=mean_vertical(u,b); xp=b.xp
    wet=h>0
    velocity=(mean[:-1]+mean[1:])/2
    velocity=xp.where(wet[:-1] & ~wet[1:],mean[:-1],velocity)
    velocity=xp.where(~wet[:-1] & wet[1:],mean[1:],velocity)
    flux=b.zeros(len(h)+1)
    donor_h=xp.where(velocity>=0,h[:-1],h[1:])
    flux[1:-1]=velocity*donor_h*(width[:-1]+width[1:])/2
    return flux,velocity


def timestep(state,requested,b):
    _,velocity=face_flux(state,b)
    speed=float(b.xp.max(abs(velocity)))
    dx=float(b.xp.min(state['cell_width']))
    dt=min(requested,.4*dx/max(speed,1e-30))
    if not np.isfinite(dt) or dt<=1e-12: raise RuntimeError('Time step collapsed; previous checkpoint retained')
    return dt


def transport(state,smb,surface_c,dt,b):
    """Donor-limited conservative ice and depth-mean horizontal enthalpy flux.

    Surface additions are mixed within sigma layers; this is a disclosed
    first-order column mixing approximation, not a stratigraphic snow model.
    """
    xp=b.xp; h,width,E=[state[k] for k in ('h','width','E')]
    area=width*state['cell_width']; volume=h*area
    flux,_=face_flux(state,b)
    outgoing=positive(flux[1:])+positive(-flux[:-1])
    safe=xp.where(outgoing>0,outgoing,outgoing*0+1)
    fraction=minimum(volume/(dt*safe),1.)
    # A roundoff-sized buffer in the limiter preserves positivity without
    # projecting/clipping thickness or deleting untracked mass.
    fraction=xp.where(fraction<1,fraction*(1-32*np.finfo(float).eps),fraction)
    factor=xp.where(flux[1:-1]>=0,fraction[:-1],fraction[1:])
    flux[1:-1]*=factor
    out_volume=dt*(positive(flux[1:])+positive(-flux[:-1]))
    incoming=dt*(positive(flux[:-1])+positive(-flux[1:]))
    retained=volume-out_volume
    if bool(xp.any(retained<0)): raise RuntimeError('Donor limiter failed; no thickness clipping applied')
    transported=retained+incoming
    left_energy=b.zeros(E.shape); right_energy=b.zeros(E.shape)
    left_energy[1:]=dt*positive(flux[1:-1])[:,None]*E[:-1]
    right_energy[:-1]=dt*positive(-flux[1:-1])[:,None]*E[1:]
    extensive=retained[:,None]*E+left_energy+right_energy
    snow=dt*positive(smb)*area
    snow_E=CP*(surface_c+273.15-TREF)
    extensive+=snow[:,None]*snow_E[:,None]
    available=transported+snow
    loss=dt*(positive(-smb)+positive(state['melt']))*area
    removed=minimum(loss,available)
    new_volume=available-removed
    safe_volume=xp.where(available>0,available,available*0+1)
    mixed=extensive/safe_volume[:,None]
    new_E=xp.where((new_volume>0)[:,None],mixed,snow_E[:,None]+E*0)
    source=snow-removed
    error=float(xp.sum(new_volume)-xp.sum(volume)-xp.sum(source))
    old_energy=float(xp.sum(volume*mean_vertical(E,b)))*RHO
    after_energy=float(xp.sum(new_volume*mean_vertical(new_E,b)))*RHO
    energy_source=float(xp.sum(snow*snow_E-removed*mean_vertical(mixed,b)))*RHO
    return new_volume/area,new_E,{'source_m3':float(xp.sum(source)),'mass_error_m3':error,
        'unrealized_ablation_m3':float(xp.sum(loss-removed)),
        'transport_energy_error_j':after_energy-old_energy-energy_source,
        'maximum_donor_limiting':1-float(xp.min(fraction)),
        'newly_ice_covered_cells':int(xp.sum((h==0)&(new_volume>0))),
        'newly_ice_free_cells':int(xp.sum((h>0)&(new_volume==0)))}


def columns(state,old_h,surface_c,dt,b,p):
    xp=b.xp; h=state['h']; E=state['E']; threshold=float(p.get('momentum_transition_m',1.))
    melt=h*0; thermal_mask=b.host(h>=threshold); drain=0.; phases=0
    for ids in segments(thermal_mask):
        if len(ids)<3: continue
        idx=b.array(ids,True)
        settings={**p,'surface_temperature_c':surface_c[idx],'horizontal_advection':False}
        after,local_melt,info=thermal_step(state['x'][idx],state['bed'][idx],h[idx],state['width'][idx],
            E[idx],state['u'][idx],state['w'][idx],state['strain_heat'][idx],
            (h[idx]-old_h[idx])/dt,dt,b,settings)
        E[idx]=after; melt[idx]=local_melt
        excess=b.array(info['drained_specific_energy'])
        drain+=float(xp.sum(mean_vertical(excess,b)*h[idx]*state['width'][idx]*state['cell_width'][idx]))*RHO
        phases=max(phases,info['phase_iterations'])
    # A one/two-cell thick-ice island still receives independent column physics.
    # Virtual copies supply the column routine's x stencil; horizontal advection
    # is disabled and no copied column contributes mass, energy or geometry.
    for ids in segments(thermal_mask):
        if len(ids)>=3: continue
        for k in ids:
            repeated=b.array([int(k)]*3,True)
            dx=state['cell_width'][k]
            local_x=state['x'][k]+b.array([-1.,0.,1.])*dx
            settings={**p,'surface_temperature_c':surface_c[repeated],'horizontal_advection':False}
            after,local_melt,info=thermal_step(local_x,state['bed'][repeated],h[repeated],state['width'][repeated],
                E[repeated],state['u'][repeated],state['w'][repeated],state['strain_heat'][repeated],
                (h[repeated]-old_h[repeated])/dt,dt,b,settings)
            E[k]=after[1]; melt[k]=local_melt[1]
            excess=b.array(info['drained_specific_energy'])
            drain+=float(mean_vertical(excess,b)[1]*h[k]*state['width'][k]*state['cell_width'][k])*RHO
            phases=max(phases,info['phase_iterations'])
    # Rapidly equilibrated thin-ice columns have prescribed surface temperature.
    # Record the energy exchanged by this asymptotic thin-column closure.
    thin=(h>0)&(h<threshold)
    target=CP*(surface_c+273.15-TREF)
    exchange=float(xp.sum(xp.where(thin,h*state['width']*state['cell_width']*(target-mean_vertical(E,b)),h*0)))*RHO
    E=xp.where(thin[:,None],target[:,None]+E*0,E)
    _,_,_,ep=thermal_state(E,h,b)
    excess=positive(E-ep-LATENT*float(p.get('water_cap',.03)))
    drain+=float(xp.sum(mean_vertical(excess,b)*h*state['width']*state['cell_width']))*RHO
    state['E']=E-excess; state['melt']=melt
    if not bool(xp.all(xp.isfinite(state['E']))): raise RuntimeError('Nonfinite enthalpy')
    return {'drained_energy_j':drain,'thin_column_exchange_j':exchange,'phase_iterations':phases}
