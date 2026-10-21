"""Run the checkpoint contract audit against the frozen-constant v3 driver."""

from __future__ import annotations

from . import audit_system2005_masked_steady50_checkpoint_contract_v1 as audit
from . import run_system2005_masked_steady50_checkpointed_v3 as driver


audit.driver = driver
audit.OUT = audit.BASE / "lia_191_system2005_masked_steady50_checkpoint_contract_v2"


if __name__ == "__main__":
    audit.main()
