"""V2 wrapper: target +1.30% so the derived lift stays within the 3.5% cap."""

from . import design_system2005_policy_v6_uniform_thickness_v3 as base

base.TARGET_DV = 0.013
base.OUT = base.RESULTS / "lia_191_system2005_policy_v6_uniform_thickness_v3_design_v2"

if __name__ == "__main__":
    raise SystemExit(base.main())
