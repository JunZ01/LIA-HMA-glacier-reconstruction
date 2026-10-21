from __future__ import annotations

import argparse
from pathlib import Path

import run_temperature_candidate_b2_stage_c_real_proxy_v1 as v1
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
ORIGINAL_READER = v1.read_allowed_proxy_observations
DEFAULT_OUT = v1.LOOP / "temperature_balanced_pooled_candidate_b2_stage_c_real_proxy_v2"


def hash_guarded_reader(network, package, years):
    source = v1.replay.prior_tools.ARCHIVE / f"python_data/proxy_data_{network}.mat"
    before = core.sha256(source)
    result = ORIGINAL_READER(network, package, years)
    after = core.sha256(source)
    if before != after:
        raise RuntimeError(f"proxy-source-changed-during-read:{network}")
    return result


def main(receipt_path: Path, output: Path) -> None:
    v1.DRIVER = DRIVER
    v1.read_allowed_proxy_observations = hash_guarded_reader
    v1.main(receipt_path, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.authorization_receipt, args.output)
