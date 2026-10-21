"""Synthetic fixtures for connected-ice MB masking and one-node front transport."""

import json
from pathlib import Path

import numpy as np

from .network_capacity_constrained_flux_fixture_v1 import project_flux_dag


PROJECT = Path(__file__).resolve().parents[4]
OUTPUT = PROJECT / "validation_data/results/paleo_surface_validation_v0/lia_191_network_connected_front_fixture_v1"


def run_fixtures():
    volume = np.array([10.0, 0.0, 0.0])
    requested_mb = np.array([1.0, 1.0, 1.0])
    connected = volume > 0
    post_mb = volume + np.where(connected, requested_mb, 0.0)
    assert np.array_equal(post_mb, np.array([11.0, 0.0, 0.0]))
    q, final, limited = project_flux_dag(post_mb[:2], np.array([0]), np.array([1]), np.array([0.5]), 1.0)
    assert np.array_equal(q, np.array([0.5]))
    assert np.array_equal(final, np.array([10.5, 0.5]))
    assert not limited[0]
    full = np.array([final[0], final[1], post_mb[2]])
    assert full[2] == 0.0
    assert abs(float(np.sum(full)) - float(np.sum(post_mb))) <= 1e-14
    return {
        "status": "fixture-pass-real-system-not-authorized",
        "mechanism": "connected-ice-mb-mask-plus-one-dry-front-capacity-transport-v1",
        "connected_mb_only": True,
        "one_front_node_activated_by_flux": True,
        "remote_tail_remains_zero": True,
        "mass_conservative": True,
        "final_volume": full.tolist(),
        "real_system_authorized": False,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    summary = run_fixtures()
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
