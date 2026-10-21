"""Executable wrapper for the v1 fixed-topology feasibility audit."""

from __future__ import annotations

from . import audit_conservative_hypsometry_raise_only_feasibility_v1 as implementation


# The v1 payload used a JSON-style false token in three reporting-only fields.
# Defining that immutable module global preserves the scientific implementation.
implementation.false = False


if __name__ == "__main__":
    implementation.main()
