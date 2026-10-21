"""Only replace the output scale floor; retain original inputs and BP energy."""
import numpy as np
from . import physics_neural_v1 as previous
from .physics_neural_v1 import graph,Network,local_diagonal,save_network,load_network


def context(mesh,state,previous_h,u,dt,cache):
    c=previous.context(mesh,state,previous_h,u,dt,cache)
    # Keep the feature normalization bitwise unchanged for a controlled test.
    # |r/diag(K)| already incorporates local rheology, geometry and stiffness.
    # No lower bound from other nodes is added to the neural output amplitude.
    c['legacy_output_scale']=c['scale'].copy()
    c['scale']=np.sqrt(np.mean(c['cheap']**2,axis=1))[:,None]
    c['scale'][~c['a']['wet'].reshape(-1,4).any(axis=1)]=0.
    assert np.isfinite(c['scale']).all() and np.all(c['scale']<=c['legacy_output_scale'])
    return c
