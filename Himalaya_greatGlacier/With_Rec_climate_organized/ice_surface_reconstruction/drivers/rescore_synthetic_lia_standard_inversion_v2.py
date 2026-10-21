"""OGGM-1.6.3 schema wrapper for the saved inversion rescore."""

from oggm.core.flowline import MixedBedFlowline

from . import rescore_synthetic_lia_standard_inversion_v1 as v1


if not hasattr(MixedBedFlowline, "_flows_to_indice"):
    MixedBedFlowline._flows_to_indice = property(
        lambda self: self.flows_to_indice
    )


if __name__ == "__main__":
    v1.main()
