from __future__ import annotations

import json
from pathlib import Path

import evaluate_temperature_candidate_b2_no_proxy_background_v1 as v1


DRIVER = Path(__file__).resolve()
V1_DRIVER = Path(v1.__file__).resolve()
v1.OUT = v1.LOOP / "temperature_b2_background_eval_v2"


def main() -> None:
    v1.main()
    summary_path = v1.OUT / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": v1.OUT.name,
            "supersedes_technical_failure": (
                "temperature_balanced_pooled_candidate_b2_no_proxy_background_evaluation_v1"
            ),
            "v2_change": "use shorter versioned output root to remain below legacy Windows path limits",
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest_path = v1.OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_v1"] = {
        "path": str(V1_DRIVER),
        "sha256": v1.core.sha256(V1_DRIVER),
    }
    manifest["active_wrapper_v2"] = {
        "path": str(DRIVER),
        "sha256": v1.core.sha256(DRIVER),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
