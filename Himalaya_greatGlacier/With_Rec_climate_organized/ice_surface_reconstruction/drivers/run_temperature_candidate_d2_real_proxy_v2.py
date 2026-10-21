from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_temperature_candidate_d2_real_proxy_v1 as implementation
import temperature_candidate_b2_stage_c_core_v1 as core


DRIVER = Path(__file__).resolve()
EXPECTED_LAUNCH_FINGERPRINT = "3ECDA177269059BEA2CC92D787D6D0D999596A70B843FF31FFEC6FF20A19CFC7"


def validate_wrapper_receipt(receipt_path: Path) -> None:
    if not receipt_path.exists():
        raise PermissionError("Candidate D2 real-proxy authorization receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    required = {
        "schema": implementation.AUTHORIZATION_SCHEMA,
        "authorized": True,
        "authorization_text": implementation.AUTHORIZATION_TEXT,
        "static_package_sha256": implementation.EXPECTED_STATIC_SHA256,
        "driver_sha256": core.sha256(implementation.DRIVER),
        "wrapper_sha256": core.sha256(DRIVER),
        "launch_fingerprint_sha256": EXPECTED_LAUNCH_FINGERPRINT,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise PermissionError(f"authorization-receipt-mismatch:{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorization-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=implementation.DEFAULT_OUT)
    arguments = parser.parse_args()
    validate_wrapper_receipt(arguments.authorization_receipt)
    implementation.main(arguments.authorization_receipt, arguments.output)
