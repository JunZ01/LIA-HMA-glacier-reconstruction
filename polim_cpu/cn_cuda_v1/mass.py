"""Identical CPU CN mass/path/limiter; only Ru evaluation is device resident."""
import numpy as np
from polim_cpu.cn_flux_time_v1.mass import Mass as OriginalMass,accepted

class Mass(OriginalMass):
    def __init__(self,*args,backend,**kw):super().__init__(*args,**kw);self.backend=backend
    def ru(self,h,u,du):
        e=self.last if np.array_equal(h,self.last_h) and np.array_equal(u,self.last_u) else self.eq.evaluate(h,u)
        return self.backend.ru(h,u,du,self.eq.weights(e))
    def save(self,name,h,u):
        # Calls the retained saved-state consistency reconstruction, not a new algorithm.
        return super().save(name,h,u)
