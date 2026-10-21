"""Compression-schema shim for the v1 residual-reservoir audit."""

from __future__ import annotations

import gzip
import pickle

from . import audit_negative_flux2342_hypsometric_residual_reservoir_v1 as impl


def load_pickle_magic(path):
    with path.open("rb") as stream:
        magic = stream.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as stream:
        return pickle.load(stream)


def main() -> None:
    impl.OUTPUT = impl.PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_negative_flux2342_hypsometric_residual_reservoir_v2"
    impl.load_pickle = load_pickle_magic
    impl.main()


if __name__ == "__main__":
    main()
