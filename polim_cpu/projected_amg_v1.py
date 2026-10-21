"""SPD old hierarchy projected onto the current free (ice-covered) DOFs."""


class ProjectedMultigrid:
    def __init__(self,base,bp):
        self.base=base
        self.d=base.d
        self.active=bp.active  # Stable address updated in-place before each solve.

    def cycle(self,b):
        # P M P is positive definite on the current active subspace. The BP
        # tangent also operates on this subspace; no velocity enters dry nodes.
        return self.base.cycle(b*self.active)*self.active
