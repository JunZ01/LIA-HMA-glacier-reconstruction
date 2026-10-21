"""Small algebraic predictor fixtures, not glacier time-integration trials."""
import numpy as np
from .large_step_transport_v1 import TransportTopology,FixedVelocityRates
from .diffusion_predictor_v2 import predict,diffusion_conductance_from_flux


def verify():
    cases=[]
    for label,bed,coefficient in [('downhill',[5.,0.],2.),('reverse-donor',[0.,5.],-2.),
                                 ('uphill',[0.,5.],2.),('flat',[0.,0.],2.),('floor-limited',[.25,0.],2.)]:
        topo=TransportTopology(np.ones(2),np.array([[0,1]]),np.array([],dtype=int))
        rates=FixedVelocityRates(topo,np.array([coefficient]),np.array([]))
        h=np.array([100.,100.]);b=np.array(bed)
        g=diffusion_conductance_from_flux(h,b,rates);old_g=abs(coefficient)/max(abs(b[0]-b[1]),1.)
        assert np.allclose(g/old_g,100.,rtol=0,atol=1e-12)
        hp,d,a=predict(h,b,np.zeros(2),.1,rates)
        assert np.allclose(a['anchored_old_flux'],a['old_flux'],rtol=0,atol=1e-12)
        assert np.min(hp)>=0 and abs(hp.sum()-h.sum())<1e-10
        cases.append(dict(case=label,coefficient_ratio=float(g[0]/old_g),**d))
    topo=TransportTopology(np.array([1.]),np.empty((0,2),dtype=int),np.array([0]))
    rates=FixedVelocityRates(topo,np.array([]),np.array([1.]))
    hp,d,_=predict(np.array([2.]),np.array([0.]),np.array([0.]),.5,rates)
    assert np.allclose(hp,[2./1.5],rtol=0,atol=1e-14)
    assert abs(hp[0]-2.+d['predicted_boundary_export'])<1e-14
    cases.append(dict(case='outward-boundary',expected_H=2./1.5,**d))
    return dict(status='pass',cases=cases,glacier_dynamics_advanced=False)
