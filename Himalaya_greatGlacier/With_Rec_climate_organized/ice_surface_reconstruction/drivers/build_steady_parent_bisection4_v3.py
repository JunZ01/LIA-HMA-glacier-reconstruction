"""Create the authorized four steady-temperature configs (static only).

This is the executable builder.  It retains the evidence/schema helpers from
v1/v2, treats the explicitly requested 3654 delta=0 as an interior bracket
refinement (its two evidence endpoints are not symmetric), and never imports
or starts OGGM.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import (
    build_steady_parent_bisection4_v1 as v1,
)
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.build_steady_parent_bisection4_v2 import (
    _evidence,
)


def main() -> int:
    root = Path(__file__).resolve().parents[4]
    config_root = root / v1.CONFIG_DIR
    manifest_path = root / v1.MANIFEST_NAME
    v1._require(v1.PYTHON.is_file(), f"python-missing:{v1.PYTHON}")

    destinations = [manifest_path]
    output_roots: list[Path] = []
    for system_id, spec in v1.CASES.items():
        tag = str(spec["target"]).replace("-", "m").replace(".", "p")
        destinations.append(config_root / f"steady_parent_bisection4_{system_id}_dt{tag}_v1.json")
        output_roots.append(root / f"w719steady_parent_bisection4_{system_id}_dt{tag}_v1")
    collisions = [str(path) for path in destinations + output_roots if path.exists()]
    v1._require(not collisions, "refusing-to-overwrite:" + repr(collisions))

    prepared = []
    for system_id, spec in v1.CASES.items():
        warm_config, _, warm_evidence = _evidence(root, system_id, str(spec["warm"]))
        cold_config, _, cold_evidence = _evidence(root, system_id, str(spec["cold"]))
        v1._require(v1._strip_allowed(warm_config) == v1._strip_allowed(cold_config), f"frozen-endpoint-conflict:{system_id}")
        v1._require(warm_evidence["delta_t_c"] == float(spec["warm_delta"]), f"warm-delta-conflict:{system_id}")
        v1._require(cold_evidence["delta_t_c"] == float(spec["cold_delta"]), f"cold-delta-conflict:{system_id}")
        v1._require(abs(warm_evidence["volume_change_fraction_50y"] - float(spec["warm_dv"])) < 1e-15, f"warm-dv-conflict:{system_id}")
        v1._require(abs(cold_evidence["volume_change_fraction_50y"] - float(spec["cold_dv"])) < 1e-15, f"cold-dv-conflict:{system_id}")

        target = float(spec["target"])
        cold_delta = float(spec["cold_delta"])
        warm_delta = float(spec["warm_delta"])
        if system_id in (1645, 4464):
            v1._require(abs(target - 0.5 * (warm_delta + cold_delta)) < 1e-15, f"midpoint-conflict:{system_id}")
            decision_rule = "exact-bisection-midpoint"
        elif system_id == 3654:
            v1._require(target == 0.0, "3654-explicit-target-conflict")
            v1._require(cold_delta < target < warm_delta, "3654-target-outside-bracket")
            decision_rule = "explicit-interior-bracket-refinement"
        else:
            v1._require(system_id == 4607 and target == -0.125, "4607-target-conflict")
            v1._require(warm_delta == 0.5 and cold_delta == 0.0, "4607-evidence-delta-conflict")
            v1._require(abs(float(spec["cold_dv"]) + 0.028160151979013295) < 1e-15, "4607-zero-dv-conflict")
            decision_rule = "finite-coldward-refinement-after-parent-plus0p5-loss-and-zero-dv-minus2p816pct"
        if system_id != 4607:
            v1._require(float(spec["warm_dv"]) < 0.0 < float(spec["cold_dv"]), f"unbracketed-dv:{system_id}")

        tag = str(target).replace("-", "m").replace(".", "p")
        output_name = f"w719steady_parent_bisection4_{system_id}_dt{tag}_v1"
        config_path = config_root / f"steady_parent_bisection4_{system_id}_dt{tag}_v1.json"
        payload = copy.deepcopy(warm_config)
        payload["stage2"]["delta_t_c"] = target
        payload["output_root"] = {"anchor": "project", "sub": output_name}
        v1._require(v1._strip_allowed(payload) == v1._strip_allowed(warm_config), f"frozen-new-config-conflict:{system_id}")
        prepared.append((config_path, payload, {
            "system_id": system_id,
            "decision_rule": decision_rule,
            "target_delta_t_c": target,
            "config": str(config_path.relative_to(root)),
            "output_root": output_name,
            "warm_parent_evidence": warm_evidence,
            "cold_parent_evidence": cold_evidence,
            "run_authorized": False,
        }))

    config_root.mkdir(parents=True, exist_ok=True)
    for config_path, payload, _ in prepared:
        with config_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    for config_path, payload, record in prepared:
        v1._require(v1._read_json(config_path) == payload, f"config-roundtrip-conflict:{config_path}")
        record["new_config_sha256"] = v1._sha256(config_path)
    v1._require(all(not path.exists() for path in output_roots), "builder-created-run-output")

    manifest = {
        "status": "configs-built-static-verified-not-run",
        "stage": "steady-parent-bisection-four-v1",
        "builder": str(Path(__file__).relative_to(root)),
        "count": len(prepared),
        "allowed_config_changes": ["output_root", "stage2.delta_t_c"],
        "runner_module_informational_only": v1.RUNNER_MODULE,
        "python": str(v1.PYTHON),
        "run_authorized": False,
        "oggm_imported": False,
        "oggm_started_by_builder": False,
        "output_roots_created": False,
        "static_validation": "pass",
        "cases": [record for _, _, record in prepared],
    }
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    v1._require(v1._read_json(manifest_path) == manifest, "manifest-roundtrip-conflict")
    print(json.dumps({"status": manifest["status"], "manifest": str(manifest_path), "count": len(prepared)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
