"""Path-compatibility-only wrapper for the independent system-1642 v2 output."""

from pathlib import Path

from . import build_negative_flux_monotone_variable_thinfilm_1642_v1 as implementation


def _sha256_compat(path) -> str:
    return implementation.hashlib.sha256(Path(path).read_bytes()).hexdigest()


implementation._sha256 = _sha256_compat


if __name__ == "__main__":
    implementation.main()
