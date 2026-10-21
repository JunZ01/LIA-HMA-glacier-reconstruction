"""Deterministic staged search; no model imports or launch side effects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DESIGN = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_191_gpu_four_group_design_v10_262_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest().upper()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest().upper()


@dataclass(frozen=True)
class Candidate:
    system: int
    delta: float = 0.0
    forward_end: float = 0.0
    melt: float = 1.0
    prcp: float = 1.0
    glen: float = 1.0

    @property
    def key(self):
        return digest(asdict(self))


class Search:
    def __init__(self, matrix, baselines, contract):
        self.matrix = matrix
        self.baselines = baselines
        self.contract = contract
        self.beam = matrix["group_2_delta_t_forward_end"]["maximum_parent_states_per_system"]

    def spin_key(self, candidate):
        parameters = asdict(candidate)
        parameters.pop("forward_end")
        return digest({"stage": "spinup", "candidate": parameters,
                       "baseline": self.baselines[candidate.system], "contract": self.contract})

    def forward_key(self, candidate, state_hash):
        return digest({"stage": "forward", "candidate": asdict(candidate),
                       "spin_key": self.spin_key(candidate), "state_hash": state_hash,
                       "contract": self.contract})

    def candidates(self, group, records):
        if group not in (1, 2, 3, 4):
            raise ValueError("group must be 1..4")
        output = []
        for system in sorted(self.baselines):
            rows = [r for r in records if r["candidate"]["system"] == system]
            if any(r.get("forward", {}).get("provisional_pass") for r in rows):
                continue
            if group == 1:
                output.extend(Candidate(system, delta=d) for d in
                              self.matrix["group_1_delta_t_eq"]["values_degC"])
                continue
            if not rows:
                continue
            latest = max(r["group"] for r in rows)
            if latest >= group:
                raise ValueError("plan next group only from completed earlier groups")
            parents = [r for r in rows if r["group"] == latest and not technical(r)]
            if group == 2:
                parents = [r for r in parents if r["spinup"]["status"] == "converged"]
            parents = sorted(parents, key=ranking)[:self.beam]
            for row in parents:
                parent = Candidate(**row["candidate"])
                if group == 2:
                    output.extend(replace(parent, forward_end=f) for f in
                                  self.matrix["group_2_delta_t_forward_end"]["values_degC"])
                elif group == 3:
                    output.extend(replace(parent, melt=p["melt_f_multiplier"],
                                          prcp=p["prcp_fac_multiplier"]) for p in
                                  self.matrix["group_3_mass_balance"]["grid"])
                else:
                    output.extend(replace(parent, glen=g) for g in
                                  self.matrix["group_4_glen_a"]["multipliers"])
        # Stage provenance is separate from physical identity. Baseline multipliers
        # and different forward parents may therefore share one cached spin-up.
        return sorted({c.key: c for c in output}.values(), key=lambda c: (c.system, c.key))

    def pack(self, candidates, nx, maximum=512):
        if not 1 <= maximum <= 512:
            raise ValueError("batch size must be 1..512")
        ordered = sorted(candidates, key=lambda c: (nx[c.system], c.system, c.key))
        return [ordered[i:i + maximum] for i in range(0, len(ordered), maximum)]

    def outcome(self, system, records, completed_groups):
        rows = [r for r in records if r["candidate"]["system"] == system]
        if any(r.get("forward", {}).get("provisional_pass") for r in rows):
            return "provisional-pass-reproduction-required"
        if technical_any(rows):
            return "technical-incomplete"
        if set(completed_groups) != {1, 2, 3, 4} or not any(r["group"] == 4 for r in rows):
            return "search-incomplete"
        return "four-group-search-exhausted"


def technical(record):
    return (record["spinup"]["status"] == "technical-failure" or
            record.get("forward", {}).get("status") == "technical-failure")


def technical_any(records):
    return any(technical(r) for r in records)


def finite_metric(value, default=math.inf):
    return float(value) if value is not None and math.isfinite(float(value)) else default


def ranking(record):
    spin = record["spinup"]
    forward = record.get("forward", {})
    identity = Candidate(**record["candidate"]).key
    if forward.get("status") == "completed":
        errors = [finite_metric(forward.get(k)) for k in ("lia_error", "rgi_error")]
        return (0, -sum(forward["gates"].values()), max(errors), sum(errors),
                finite_metric(forward.get("amplitude_shortfall")),
                -finite_metric(forward.get("correlation"), -math.inf),
                -finite_metric(forward.get("sustained_share"), -math.inf), identity)
    # No future endpoint or lag metric is consulted before forward exists.
    return (1 if spin["status"] == "converged" else 2,
            0 if spin.get("domain_valid") else 1,
            finite_metric(spin.get("lia_error")),
            finite_metric(spin.get("volume_drift")),
            finite_metric(spin.get("length_drift_m")), 0, 0, identity)


def load_search(contract, design=DESIGN):
    summary = json.loads((design / "summary.json").read_text(encoding="utf-8"))
    for name in ("parameter_matrix", "curve_and_retention_rules", "gpu_execution_rules"):
        artifact = summary[name]
        if file_hash(ROOT / artifact["path"]) != artifact["sha256"]:
            raise ValueError(f"frozen design hash mismatch: {name}")
    matrix = json.loads((design / "four_group_parameter_matrix.json").read_text())
    source = matrix["baseline_source"]
    if file_hash(ROOT / source["path"]) != source["sha256"]:
        raise ValueError("baseline matrix hash mismatch")
    with (ROOT / source["path"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    baselines = {}
    fields = source["per_system_fields"]
    for row in rows:
        system = int(row["lia_unique_id"])
        base = {k: row["temp_melt_degC" if k == "temp_melt_degC" else k] for k in fields}
        if system in baselines and base != baselines[system]:
            raise ValueError(f"nonunique baseline for {system}")
        baselines[system] = base
    if len(baselines) != 191:
        raise ValueError("population must be 191")
    return Search(matrix, baselines, contract)
