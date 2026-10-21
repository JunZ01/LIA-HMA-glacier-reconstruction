"""One material-law regression, including the previously wrong adapter path."""
import numpy as np
from . import model
from ..v3.physics import CP,TREF,LATENT,YEAR


def verify_material_law():
    # Broad independent published-scale checks catch seconds/year and legacy-A reuse.
    t=np.array([253.15,263.15,268.15,273.15]);a=model.material_softness_si(t,np.zeros(4))
    if not np.all((a>np.array([1.3e-25,3.2e-25,1.2e-24,4.5e-24])) & (a<np.array([1.6e-25,5.0e-25,1.6e-24,5.3e-24]))):
        raise AssertionError(f'Material softness outside reference scales: {a}')
    h=np.array([0.,300.]);cold=np.full((2,9),CP*(253.15-TREF))
    p={'material_enhancement':1.,'glen_a_pa_3_s_1':7.538801981725612e-23}
    field=model.fluidity(cold,h,p)
    if not np.isclose(field[0,0]/YEAR,a[0],rtol=1e-14):raise AssertionError('Seconds/year mismatch')
    if not field[1,0]>field[1,-1]:raise AssertionError('Pressure correction reversed')
    other=model.fluidity(cold,h,{**p,'glen_a_pa_3_s_1':1e10})
    if not np.array_equal(field,other):raise AssertionError('Legacy inverse A leaked into material law')
    z=np.zeros(3);wet=np.tile(CP*(273.15-TREF)+LATENT*np.array([0.,.01,.03])[:,None],(1,9))
    values=model.fluidity(wet,z,p)[:,0]
    if not np.isclose(values[1]/values[0],2.8125,rtol=1e-12) or not np.isclose(values[2],values[1],rtol=1e-12,atol=0):raise AssertionError('Observed water limit not applied')
    return dict(status='pass',temperatures_K=t.tolist(),softness_SI=a.tolist(),seconds_per_year=YEAR,legacy_A_invariant=True,pressure_correction=True,rheology_water_limit=.01,water_softening_at_limit=float(values[1]/values[0]))
