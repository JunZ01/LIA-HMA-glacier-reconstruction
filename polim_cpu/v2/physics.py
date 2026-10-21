"""Enthalpy transport and conservative thickness evolution (experimental).

Vertical conduction only; horizontal upwind advection, implicit vertical
advection/diffusion, pressure melting and explicitly recorded water drainage.
No claim of implementing PoLIM's MEGM or a subglacial hydrology model.
"""
import numpy as np

RHO=910.; GRAVITY=9.81; CP=2009.; LATENT=334000.; TREF=223.15
YEAR=31556926.; CONDUCTIVITY=2.1; BETA_CC=7.9e-8


def positive(a): return (a+abs(a))/2


def gradient(a, coordinates, axis, b):
    out=b.zeros(a.shape)
    if axis==0:
        out[1:-1]=(a[2:]-a[:-2])/(coordinates[2:]-coordinates[:-2])[:,None]
        out[0]=(a[1]-a[0])/(coordinates[1]-coordinates[0])
        out[-1]=(a[-1]-a[-2])/(coordinates[-1]-coordinates[-2])
    else:
        out[:,1:-1]=(a[:,2:]-a[:,:-2])/(coordinates[2:]-coordinates[:-2])[None,:]
        out[:,0]=(a[:,1]-a[:,0])/(coordinates[1]-coordinates[0])
        out[:,-1]=(a[:,-1]-a[:,-2])/(coordinates[-1]-coordinates[-2])
    return out


def thermal_state(E,h,b):
    sigma=b.array(np.linspace(0,1,E.shape[1]))
    pressure=RHO*GRAVITY*h[:,None]*(1-sigma[None,:])
    pmp=273.15-BETA_CC*pressure; ep=CP*(pmp-TREF)
    excess=positive(E-ep)
    temperature=TREF+(E-excess)/CP
    water=excess/LATENT
    return temperature,water,pmp,ep


def glen(E,h,b):
    T,water,pmp,_=thermal_state(E,h,b); xp=b.xp
    corrected=T+(273.15-pmp)
    cold=corrected<=263.15
    a0=xp.where(cold,corrected*0+3.985e-13,corrected*0+1.916e3)
    q=xp.where(cold,corrected*0+60000.,corrected*0+139000.)
    capped=water-positive(water-.03)
    return a0*xp.exp(-q/(8.314*corrected))*YEAR*(1+181.25*capped)


def initial_enthalpy(h,layers,b,p):
    sigma=b.array(np.linspace(0,1,layers))
    top=float(p.get('surface_temperature_c',-10)); base=float(p.get('initial_basal_temperature_c',-2))
    if not (-100<top<=0 and -100<base<=0): raise ValueError('Initial temperatures must lie in (-100,0] Celsius')
    E=b.zeros((len(h),layers)); _,_,pmp,_=thermal_state(E,h,b)
    temperature=273.15+base+(top-base)*sigma[None,:]+h[:,None]*0
    temperature-=positive(temperature-pmp)
    E=CP*(temperature-TREF)
    omega=float(p.get('initial_water_fraction',0))
    if not 0<=omega<=.03: raise ValueError('Initial water fraction must be in [0,.03]')
    if omega: E=b.xp.where(temperature>=pmp-1e-9,E+LATENT*omega,E)
    return E


def kinematics(x,bed,h,width,u,A,b,p,melt=None):
    xp=b.xp; sigma=b.array(np.linspace(0,1,u.shape[1]))
    bx=gradient(bed[:,None],x,0,b); hx=gradient(h[:,None],x,0,b)
    uz=gradient(u,sigma,1,b)/h[:,None]
    ux=gradient(u,x,0,b)-(bx+sigma[None,:]*hx)*uz
    vy=u*gradient(width[:,None],x,0,b)/width[:,None] if p.get('flowband',False) else u*0
    e2=ux*ux+vy*vy+ux*vy+uz*uz/4
    if p.get('flowband',False): e2+=u*u/(4*width[:,None]**2)
    n=float(p.get('glen_n',3)); eta=.5*A**(-1/n)*(e2+float(p.get('strain_regularization',1e-30)))**((1-n)/(2*n))
    w=b.zeros(u.shape); w[:,0]=u[:,0]*bx[:,0]
    if melt is not None: w[:,0]-=melt
    dz=h/(u.shape[1]-1)
    for j in range(1,u.shape[1]): w[:,j]=w[:,j-1]-.5*(ux[:,j-1]+vy[:,j-1]+ux[:,j]+vy[:,j])*dz
    return w,eta,4*eta*e2


def thickness_rate(x,h,width,u,smb,melt,b):
    mean=(b.xp.sum(u,axis=1)-(u[:,0]+u[:,-1])/2)/(u.shape[1]-1)
    faces=(mean[:-1]+mean[1:])/2
    upwind=b.xp.where(faces>=0,h[:-1],h[1:])
    flux=b.zeros(len(x)+1)
    flux[1:-1]=faces*upwind*(width[:-1]+width[1:])/2
    cv=b.array(np.r_[np.diff(b.host(x))[0]/2,(b.host(x)[2:]-b.host(x)[:-2])/2,np.diff(b.host(x))[-1]/2])
    # Both end faces closed. Physical terminus migration/calving is not modeled.
    source=smb-melt
    rate=source-(flux[1:]-flux[:-1])/(width*cv)
    return rate,source,cv,faces


def step_size(x,h,width,u,rate,dt,b):
    xp=b.xp
    speeds=abs(u); maxspeed=float(xp.max(speeds))
    dx=float(xp.min(x[1:]-x[:-1]))
    adv=.4*dx/max(maxspeed,1e-30)
    loss=positive(-rate)
    positive_limit=float(xp.min(.4*h/(loss+1e-30)))
    selected=min(dt,adv,positive_limit)
    if not np.isfinite(selected) or selected<=1e-12: raise RuntimeError('Time step collapsed; checkpoint retained')
    return selected


def tridiagonal(lower,diagonal,upper,rhs,b):
    c=b.clone(upper); d=b.clone(rhs); diag=b.clone(diagonal)
    for j in range(1,rhs.shape[1]):
        factor=lower[:,j]/diag[:,j-1]
        diag[:,j]-=factor*c[:,j-1]; d[:,j]-=factor*d[:,j-1]
    answer=b.zeros(rhs.shape); answer[:,-1]=d[:,-1]/diag[:,-1]
    for j in range(rhs.shape[1]-2,-1,-1): answer[:,j]=(d[:,j]-c[:,j]*answer[:,j+1])/diag[:,j]
    return answer


def thermal_step(x,bed,h,width,E,u,w,heat,rate,dt,b,p):
    xp=b.xp; layers=E.shape[1]; dz=h/(layers-1); sigma=b.array(np.linspace(0,1,layers))
    ratio=float(p.get('temperate_diffusivity_ratio',.001))
    if ratio<=0 or ratio>1: raise ValueError('Temperate/cold diffusivity ratio must be in (0,1]')
    cap=float(p.get('water_cap',.03))
    if cap<=0 or cap>.1: raise ValueError('water_cap must be in (0,.1]')
    bx=gradient(bed[:,None],x,0,b); hx=gradient(h[:,None],x,0,b)
    velocity=(w-u*(bx+sigma[None,:]*hx)-sigma[None,:]*rate[:,None])/h[:,None]
    # Zero-gradient horizontal inflow enthalpy; configurable inflow is a future extension.
    left=b.zeros(E.shape); right=b.zeros(E.shape)
    left[1:]=(E[1:]-E[:-1])/(x[1:]-x[:-1])[:,None]
    right[:-1]=(E[1:]-E[:-1])/(x[1:]-x[:-1])[:,None]
    advection=positive(u)*left-positive(-u)*right
    base_rhs=E+dt*(heat/RHO-advection)
    geothermal=float(p.get('geothermal_w_m2',.05))
    if geothermal<0 or not np.isfinite(geothermal): raise ValueError('Geothermal flux must be finite and nonnegative')
    friction=float(p.get('beta2',1000))*u[:,0]**2 if p.get('basal','zero')=='linear' else u[:,0]*0
    basal_flux=geothermal*YEAR+friction
    _,_,pmp,ep=thermal_state(E,h,b)
    surface_c=float(p.get('surface_temperature_c',-10))
    if not -100<surface_c<=0: raise ValueError('surface_temperature_c must be in (-100,0]')
    top=(273.15+surface_c)-positive(273.15+surface_c-pmp[:,-1])
    guess=b.clone(E); phase_iterations=0
    for phase_iterations in range(30):
        # Lagged enthalpy diffusivity, harmonic at cold/temperate interfaces.
        diffusivity=xp.where(guess>=ep-1e-8,guess*0+CONDUCTIVITY/(RHO*CP)*YEAR*ratio,guess*0+CONDUCTIVITY/(RHO*CP)*YEAR)
        edge=2*diffusivity[:,:-1]*diffusivity[:,1:]/(diffusivity[:,:-1]+diffusivity[:,1:])
        lower=b.zeros(E.shape); upper=b.zeros(E.shape); diagonal=E*0+1
        lower[:,1:]=-dt*edge/dz[:,None]**2
        upper[:,:-1]=-dt*edge/dz[:,None]**2
        upper[:,0]*=2
        lower[:,1:]-=dt*positive(velocity[:,1:])*(layers-1)
        upper[:,:-1]-=dt*positive(-velocity[:,:-1])*(layers-1)
        diagonal-=lower+upper
        rhs=b.clone(base_rhs); rhs[:,0]+=dt*2*basal_flux/(RHO*dz)
        # Temperate ice still conducts the pressure-melting temperature gradient
        # with physical ice conductivity; only excess-enthalpy diffusion is small.
        correction=(CONDUCTIVITY/(RHO*CP)*YEAR-edge)*(ep[:,1:]-ep[:,:-1])/dz[:,None]
        rhs[:,1:-1]+=dt*(correction[:,1:]-correction[:,:-1])/dz[:,None]
        rhs[:,0]+=dt*2*correction[:,0]/dz
        lower[:,-1]=0; upper[:,-1]=0; diagonal[:,-1]=1; rhs[:,-1]=CP*(top-TREF)
        trial=tridiagonal(lower,diagonal,upper,rhs,b)
        temperate_base=trial[:,0]>ep[:,0]
        saved_diag=b.clone(diagonal[:,0]); saved_upper=b.clone(upper[:,0]); saved_rhs=b.clone(rhs[:,0])
        diagonal[:,0]=xp.where(temperate_base,diagonal[:,0]*0+1,diagonal[:,0])
        upper[:,0]=xp.where(temperate_base,upper[:,0]*0,upper[:,0])
        rhs[:,0]=xp.where(temperate_base,ep[:,0],rhs[:,0])
        trial=tridiagonal(lower,diagonal,upper,rhs,b)
        # A 1e-8 J/kg phase tolerance avoids round-off toggling at E == E_pmp.
        # It changes classification only (about 5e-12 K), never the saved enthalpy.
        if bool(xp.all((trial>=ep-1e-8)==(guess>=ep-1e-8))):
            guess=trial; break
        guess=trial
    else: raise RuntimeError('Cold/temperate diffusivity iteration failed; reduce requested time step')
    reaction=saved_diag*guess[:,0]+saved_upper*guess[:,1]-saved_rhs
    melt=xp.where(temperate_base,positive(-reaction)*dz/(2*LATENT*dt),reaction*0)
    excess=positive(guess-ep-LATENT*cap)
    guess-=excess
    if not bool(xp.all(xp.isfinite(guess))): raise RuntimeError('Nonfinite enthalpy')
    return guess,melt,{'phase_iterations':phase_iterations+1,'drained_specific_energy':b.host(excess),
                       'basal_heat_w_m2':b.host(basal_flux/YEAR)}
