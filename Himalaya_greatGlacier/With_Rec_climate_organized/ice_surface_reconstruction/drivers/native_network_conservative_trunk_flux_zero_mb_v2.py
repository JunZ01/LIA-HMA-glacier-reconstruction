"""Technical boolean-literal compatibility entrypoint for flux zero-MB v1."""

import builtins

builtins.false = False

from .native_network_conservative_trunk_flux_zero_mb_v1 import main


if __name__ == "__main__":
    main()
