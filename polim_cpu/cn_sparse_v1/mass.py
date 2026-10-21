from polim_cpu.cn_cuda_v1.mass import Mass as BaseMass,accepted
class Mass(BaseMass):
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw)
        original=self.eq.matrix
        def measured(*a,**kw):
            with self.w.span('T_assembly'):return original(*a,**kw)
        self.eq.matrix=measured
