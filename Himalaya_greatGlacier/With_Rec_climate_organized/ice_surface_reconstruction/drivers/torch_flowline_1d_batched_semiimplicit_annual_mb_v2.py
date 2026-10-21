"""V2 backend: OGGM-compatible annual elevation-feedback cache."""
from __future__ import annotations
from torch_flowline_1d_batched_semiimplicit_v1 import BatchedFlowlineState

class BatchedFlowlineStateAnnualMB(BatchedFlowlineState):
    """Keep annual MB fixed within a calendar year, like OGGM's default."""
    def __post_init__(self) -> None:
        super().__post_init__()
        self._annual_mb_cache_year = None
        self._annual_mb_cache = None

    def annual_mass_balance(self):
        year = int(self.year // 1)
        if self._annual_mb_cache_year != year:
            self._annual_mb_cache = super().annual_mass_balance()
            self._annual_mb_cache_year = year
        return self._annual_mb_cache
