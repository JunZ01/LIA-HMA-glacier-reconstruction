import numpy as np
from polim_cpu.geometry_probe_v2.common import ahash
from polim_cpu.geometry_probe_v2.response import wet

class VelocityCandidate:
    """Cache z=-Fu^-1 FH dH ONLY for this linearization, branch and direction.

    No projection: du_guess=alpha*(z-c). With projection, apply FH to the
    actual displacement on current wet nodes and use -alpha*c. Never book
    candidate projection as an actual source or claim a smooth branch crossing.
    """
    def __init__(self,linear,c,obstacle,bed):
        self.linear=linear;self.c=c.copy();self.obstacle=obstacle.copy();self.bed=bed
        self.wet=linear.a['wet'][::4].copy();self.direction=None;self.z=None
        self.binding=self.point_binding()
    def point_binding(self):
        return dict(H=ahash(self.linear.h),u=ahash(self.linear.u),
                    momentum_wet=ahash(self.linear.a['wet']),obstacle=ahash(self.obstacle))
    def check(self):
        if self.point_binding()!=self.binding:raise RuntimeError('Stale velocity response cache: linearization or active branch changed')
    def capture(self,direction,z):
        self.check();self.direction=direction.copy();self.z=z.copy()
    def propose(self,ht,delta,alpha):
        self.check();actual=ht-self.linear.h
        unprojected=np.array_equal(ht,self.linear.h+alpha*delta)
        cached=unprojected and self.direction is not None and np.array_equal(delta,self.direction)
        if cached:
            z_actual=alpha*self.z
        else:
            restricted=np.where(self.wet,actual,0.)
            z_actual=self.linear.solve(-self.linear.fh(restricted),'actual projected displacement velocity candidate')
        predicted=z_actual-alpha*self.c
        guess=self.linear.u+predicted
        dry=~np.repeat(wet(ht,self.bed),4);guess[dry]=0.
        return guess,actual,predicted,dict(cache_hit=bool(cached),projected=not unprojected,
            binding=self.binding,direction_sha256=ahash(delta),actual_displacement_sha256=ahash(actual),
            predicted_velocity_sha256=ahash(predicted),dry_velocity_projected_nodes=np.flatnonzero(dry[::4]).tolist(),
            guess_changed_from_old=bool(not np.array_equal(guess,self.linear.u)),
            guess_minus_old_l2=float(np.linalg.norm(guess-self.linear.u)))
