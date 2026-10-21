"""One guarded GPU worker for the fixed four-group search.

Static --check is safe by default. Model execution requires an independently
registered, hash-bound GPU authorization; CPU authority is never consumed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from lia_gpu_four_group_search_v2 import ROOT, DESIGN, Candidate, load_search, file_hash, digest
from lia_gpu_matrix_io_v2 import Store, WriterLock, atomic_json, status

HERE = Path(__file__).resolve().parent
BRANCH = "lia_191_gpu_four_group_matrix_v2"


def check_inputs(inputs):
    manifest = json.loads((inputs / "manifest.json").read_text())
    basis = {k: v for k, v in manifest.items() if k not in
             ("contract", "group1_matrix_sha256", "group1_unique_spinups")}
    if digest(basis) != manifest["contract"]:
        raise ValueError("compiled manifest identity mismatch")
    if file_hash(inputs / "inputs.npz") != manifest["inputs_sha256"]:
        raise ValueError("compiled inputs changed")
    if file_hash(inputs / "group1_matrix.json") != manifest["group1_matrix_sha256"]:
        raise ValueError("group-1 matrix changed")
    if file_hash(DESIGN / "summary.json") != manifest["design_summary_sha256"]:
        raise ValueError("frozen design changed")
    for name, expected in manifest["source_files"].items():
        if Path(name).name != name or file_hash(HERE / name) != expected:
            raise ValueError(f"implementation changed: {name}")
    return manifest


def authorize(inputs, path):
    manifest = check_inputs(inputs)
    auth = json.loads(path.read_text())
    state = json.loads((ROOT / "LIA_CURRENT_STATE.json").read_text(encoding="utf-8-sig"))
    branch = state.get("research_branches", {}).get(BRANCH, {})
    if not (auth.get("launch_authorized") is True and branch.get("launch_authorized") is True
            and branch.get("authorization_sha256") == file_hash(path)):
        raise PermissionError("this GPU branch has no matching launch authorization")
    if auth.get("contract") != manifest["contract"] or auth.get("mode") not in ("preflight", "production"):
        raise PermissionError("GPU execution contract mismatch")
    if auth["mode"] == "production":
        proof = auth["preflight"]
        report_path = ROOT / proof["path"]
        if file_hash(report_path) != proof["sha256"]:
            raise PermissionError("preflight evidence changed")
        report = json.loads(report_path.read_text())
        if report.get("contract") != manifest["contract"] or report.get("status") != "pass":
            raise PermissionError("production requires this implementation's technical preflight")
    validation = subprocess.run([sys.executable, "-B", str(HERE / "validate_lia_execution_state_v1.py")],
                                capture_output=True, text=True, check=False)
    if validation.returncode:
        raise PermissionError("controller validation failed: " + validation.stdout)
    root = (ROOT / auth["output_root"]).resolve()
    if not root.is_relative_to(ROOT.resolve()) or len(str(root)) > 90 or root == ROOT.resolve():
        raise ValueError("a short project-local output root is required")
    return manifest, auth, root


class Runtime:
    def __init__(self, search, manifest, bundle, root, device):
        self.search, self.manifest, self.bundle = search, manifest, bundle
        self.root, self.device = root, device
        self.store = Store(root)
        self.index = {r["system"]: i for i, r in enumerate(manifest["systems"])}
        self.systems = {r["system"]: r for r in manifest["systems"]}
        self.nx = {r["system"]: r["nx"] for r in manifest["systems"]}
        self.completed = {}
        self.started = time.perf_counter()

    def notify(self, **values):
        status(self.root / "status.json", {"pid": os.getpid(), "wall_seconds": time.perf_counter() - self.started,
                                          **values})

    def cache(self, kind, key):
        if (kind, key) in self.completed:
            return self.completed[kind, key]
        value = self.store.read(kind, key)
        failure = self.root / "quarantine" / f"{kind}-{key[:16]}.json"
        if value is None and failure.exists():
            value = json.loads(failure.read_text())
            if value["key"] != key:
                raise ValueError("quarantine key collision")
        if value is not None:
            self.completed[kind, key] = value
        return value

    def persist(self, kind, key, value, arrays=None):
        try:
            result = self.store.commit(kind, key, value, arrays)
        except OSError as error:
            result = {"key": key, "status": "technical-failure", "error": repr(error),
                      "provisional_pass": False}
            atomic_json(self.root / "quarantine" / f"{kind}-{key[:16]}.json", result)
        self.completed[kind, key] = result
        return result

    def key(self, kind, candidate):
        spin_key = self.search.spin_key(candidate)
        if kind == "spin":
            return spin_key
        spin = self.cache("spin", spin_key)
        if spin is None or spin["status"] != "converged":
            raise ValueError("forward requires a qualified committed spin-up")
        return self.search.forward_key(candidate, spin["artifacts"][0]["sha256"])

    def execute(self, candidates, kind):
        import numpy as np
        from torch_flowline_1d_matrix_v2 import make_state, compact, snapshot
        from lia_gpu_matrix_scoring_v2 import stability, score
        unique = {self.key(kind, c): c for c in candidates}
        pending = [(key, c) for key, c in unique.items() if self.cache(kind, key) is None]
        cohorts = {}
        for key, c in pending:
            checkpoint = self.store.latest(kind, key)
            year = checkpoint[0]["year"] if checkpoint else (0 if kind == "spin" else 1470)
            cohorts.setdefault(year, []).append(c)
        for initial_year in sorted(cohorts):
            for batch in self.search.pack(cohorts[initial_year], self.nx):
                keys = [self.key(kind, c) for c in batch]
                checkpoints = [self.store.latest(kind, k) for k in keys]
                maximum_nx = max(self.nx[c.system] for c in batch)
                h = np.zeros((len(batch), maximum_nx))
                totals = np.zeros((len(batch), 2))
                histories = []
                count = None
                for i, (c, cp) in enumerate(zip(batch, checkpoints)):
                    n = self.nx[c.system]
                    if cp:
                        meta, arrays = cp
                        h[i, :n] = arrays["thick"]
                        totals[i] = arrays["totals"]
                        histories.append({int(y): row.tolist() for y, row in zip(arrays["years"], arrays["history"])})
                        if count is not None and meta["counters"] != count:
                            raise ValueError("resume cohort step counters differ")
                        count = meta["counters"]
                    elif kind == "spin":
                        h[i, :n] = self.bundle["system_thick"][self.index[c.system], :n]
                        histories.append({})
                    else:
                        h[i, :n] = self.store.arrays("spin", self.search.spin_key(c))["thick"]
                        histories.append({})
                state = make_state(batch, self.search.baselines, self.bundle, self.index, self.device,
                                   kind, h, initial_year, totals, count)
                metrics, _, _, _, _ = snapshot(state)
                for i in range(len(batch)):
                    histories[i].setdefault(initial_year, [*metrics[i].tolist(), 0., 0.])
                end = 1500 if kind == "spin" else 2000
                increment = 30 if kind == "spin" else 1
                confirmations = {330: 300, 630: 600, 930: 900, 1230: 1200, 1500: 1470}
                for year in range(initial_year + increment, end + 1, increment):
                    climatic = np.zeros(len(batch))
                    if kind == "forward":
                        import torch
                        ice = (state.thick > 0) & state.valid_mask
                        weights = state.widths_m * state.dx_meter[:, None] * ice
                        climatic = (torch.sum(state.annual_mass_balance() * weights, dim=1) /
                                    torch.clamp(torch.sum(weights, dim=1), min=1e-30) * 900 * 31536000).detach().cpu().numpy()
                    state.run_until(float(year))
                    metrics, thick, totals, valid, margins = snapshot(state)
                    for i in range(len(batch)):
                        previous = histories[i][year - increment]
                        area_mean = .5 * (previous[1] + metrics[i, 1])
                        volume_mb = 900 * (metrics[i, 2] - previous[2]) / max(area_mean, 1e-30) / increment
                        histories[i][year] = [*metrics[i].tolist(), float(climatic[i]), float(volume_mb)]
                    keep = []
                    for i, c in enumerate(batch):
                        history = histories[i]
                        value = None
                        target = self.systems[c.system]
                        if not valid[i]:
                            value = {"status": "scientific-domain-failure", "domain_valid": False,
                                     "lia_error": abs(float(metrics[i, 0]) - target["target_lia"]) / target["target_lia"],
                                     "provisional_pass": False}
                        elif kind == "spin" and year in confirmations:
                            boundary = stability(history, confirmations[year])
                            confirmation = stability(history, year)
                            passed = boundary["passed"] and confirmation["passed"]
                            if passed or year == end:
                                value = {"status": "converged" if passed else "max-duration-nonconverged",
                                         "domain_valid": True, "boundary": boundary, "confirmation": confirmation,
                                         "volume_drift": confirmation["volume_drift"],
                                         "length_drift_m": confirmation["length_drift_m"],
                                         "lia_error": abs(float(metrics[i, 0]) - target["target_lia"]) / target["target_lia"]}
                        elif kind == "forward" and year == end:
                            value = score(history, target["target_lia"], target["target_rgi"])
                        counter = {"accepted_steps": state.accepted_steps, "rejected_steps": state.rejected_steps,
                                   "minimum_actual_timestep_seconds": state.minimum_actual_timestep_seconds}
                        payload = {"thick": thick[i, :self.nx[c.system]], "totals": totals[i],
                                   "years": np.asarray(sorted(history)),
                                   "history": np.asarray([history[y] for y in sorted(history)])}
                        if value is not None:
                            if not np.isfinite(payload["thick"]).all() or not np.isfinite(payload["history"]).all():
                                value = {"status": "technical-failure", "error": "nonfinite numerical state", "provisional_pass": False}
                                payload = None
                            self.persist(kind, keys[i], {**value, "year": year, "counters": counter,
                                "free_domain_margin_m": margins[i], "candidate": asdict(c)}, payload)
                        else:
                            save = year in confirmations if kind == "spin" else (year - 1470) % 30 == 0
                            if save:
                                try:
                                    self.store.checkpoint(kind, keys[i], year, {"counters": counter}, payload)
                                except OSError as error:
                                    self.persist(kind, keys[i], {"status": "technical-failure", "error": repr(error),
                                                                "provisional_pass": False})
                                    continue
                            keep.append(i)
                    self.notify(stage=kind, model_year=year, batch_active=len(keep),
                                accepted_steps=state.accepted_steps, rejected_steps=state.rejected_steps,
                                minimum_actual_timestep_seconds=state.minimum_actual_timestep_seconds)
                    terminals = [self.completed[kind, k] for k in unique if (kind, k) in self.completed]
                    if len(terminals) >= 120 and sum(r["status"] == "technical-failure" for r in terminals) / len(terminals) > .15:
                        raise RuntimeError("technical invalidity exceeds 15 percent")
                    if not keep:
                        break
                    if len(keep) != len(batch):
                        compact(state, keep)
                        batch, keys, histories = ([batch[i] for i in keep], [keys[i] for i in keep], [histories[i] for i in keep])
        return {c.key: self.cache(kind, self.key(kind, c)) for c in candidates}

    def run(self):
        records = []
        for group in (1, 2, 3, 4):
            path = self.root / f"group-{group}.json"
            candidates = self.search.candidates(group, records)
            plan = [asdict(c) for c in candidates]
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["plan"] != plan or saved["contract"] != self.search.contract:
                    raise ValueError("recovery stage differs from deterministic plan")
                for row in saved["records"]:
                    c = Candidate(**row["candidate"])
                    if self.cache("spin", self.key("spin", c)) != row["spinup"]:
                        raise ValueError("recovery spin result changed")
                    if row.get("forward") and self.cache("forward", self.key("forward", c)) != row["forward"]:
                        raise ValueError("recovery forward result changed")
                records.extend(saved["records"])
                continue
            spins = self.execute(candidates, "spin")
            stable = [c for c in candidates if spins[c.key]["status"] == "converged"]
            forwards = self.execute(stable, "forward") if group > 1 else {}
            stage_rows = [{"group": group, "candidate": asdict(c), "spinup": spins[c.key],
                           "forward": forwards.get(c.key, {})} for c in candidates]
            atomic_json(path, {"contract": self.search.contract, "plan": plan, "records": stage_rows})
            records.extend(stage_rows)
        summary = {"status": "completed", "contract": self.search.contract,
                   "outcomes": {str(s): self.search.outcome(s, records, (1, 2, 3, 4)) for s in self.search.baselines},
                   "provisional_passes": [r for r in records if r.get("forward", {}).get("provisional_pass")],
                   "accepted_ledger_effect": 0, "wall_seconds": time.perf_counter() - self.started}
        atomic_json(self.root / "summary.json", summary)
        self.notify(**summary)
        return summary


def preflight(search, manifest, bundle, root, device):
    import numpy as np
    from torch_flowline_1d_matrix_v2 import make_state, snapshot
    from unittest.mock import patch
    index = {r["system"]: i for i, r in enumerate(manifest["systems"])}
    nx = {r["system"]: r["nx"] for r in manifest["systems"]}
    all_cells = search.candidates(1, [])
    batch = [all_cells[int(i)] for i in np.linspace(0, len(all_cells) - 1, 24)]
    state = make_state(batch, search.baselines, bundle, index, device, "spin")
    state.run_until(15.)
    _, h, totals, _, _ = snapshot(state)
    store = Store(root)
    key = digest({"contract": search.contract, "purpose": "preflight-midpoint"})
    saved = store.checkpoint("spin", key, 15, {}, {"thick": h, "totals": totals})
    _, restored = store.latest("spin", key)
    resumed = make_state(batch, search.baselines, bundle, index, device, "spin", restored["thick"], 15,
                         restored["totals"])
    with patch("lia_gpu_matrix_io_v2.atomic_json", side_effect=PermissionError("preflight injected status lock")):
        unavailable = status(root / "status.json", {"purpose": "injected status failure"})
        resumed.run_until(30.)
    state.run_until(30.)
    a, h1, _, _, _ = snapshot(state)
    b, h2, _, _, _ = snapshot(resumed)
    relative = np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)
    checks = {"volume": bool(np.all(relative[:, 2] <= 1e-8)), "area": bool(np.all(relative[:, 1] <= 1e-10)),
              "thickness": bool(np.all(np.abs(h1 - h2) <= 1e-6)), "length": bool(np.array_equal(a[:, 0], b[:, 0])),
              "status_failure_survived": not unavailable and resumed.year == 30.}
    timings = []
    for size in (24, 128, 512):
        selected = search.pack(all_cells, nx, maximum=size)[0]
        trial = make_state(selected, search.baselines, bundle, index, device, "spin")
        import torch
        torch.cuda.synchronize()
        start = time.perf_counter()
        trial.run_until(1.)
        torch.cuda.synchronize()
        timings.append({"lanes": size, "years": 1, "wall_seconds": time.perf_counter() - start,
                        "peak_memory_bytes": torch.cuda.max_memory_allocated()})
    report = {"status": "pass" if all(checks.values()) else "fail", "contract": search.contract,
              "checks": checks, "timings": timings, "checkpoint_artifacts": saved["artifacts"],
              "scientific_success_required": False, "production_launched": False}
    atomic_json(root / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        manifest = check_inputs(args.inputs)
        print(json.dumps({"status": "static-check-pass", "contract": manifest["contract"], "launch": False}))
        return 0
    if args.authorization is None:
        parser.error("explicit GPU authorization is required; use --check for static validation")
    manifest, auth, root = authorize(args.inputs, args.authorization)
    from launch_lia_191_gpu_spinup_v10_239 import SS4_PYTHON, child_environment
    if os.environ.get("LIA_GPU_MATRIX_V2_CHILD") != "1":
        env = child_environment()
        env["LIA_GPU_MATRIX_V2_CHILD"] = "1"
        return subprocess.run([str(SS4_PYTHON), "-B", str(Path(__file__)), *sys.argv[1:]], env=env).returncode
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch, _, _ = import_torch_with_stubs_filtered()
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(args.inputs / "inputs.npz", allow_pickle=False) as data:
        bundle = {k: data[k] for k in data.files}
    search = load_search(manifest["contract"])
    with WriterLock(root):
        launch = root / "launch.json"
        if launch.exists():
            prior = json.loads(launch.read_text())
            if not auth.get("resume") or prior["contract"] != search.contract or prior["mode"] != auth["mode"]:
                raise PermissionError("existing output requires explicit same-contract recovery authority")
        else:
            atomic_json(launch, {"contract": search.contract, "authorization_sha256": file_hash(args.authorization),
                                "pid": os.getpid(), "mode": auth["mode"]})
        try:
            if auth["mode"] == "preflight":
                report = preflight(search, manifest, bundle, root, torch.device("cuda:0"))
            else:
                report = Runtime(search, manifest, bundle, root, torch.device("cuda:0")).run()
            return 0 if report["status"] in ("pass", "completed") else 1
        except BaseException as error:
            status(root / "termination.json", {"status": "technical-failure", "error": repr(error),
                                               "contract": search.contract, "completed_results_preserved": True})
            raise


if __name__ == "__main__":
    raise SystemExit(main())
