"""Reuse the tested prefix dynamics and original execution loop, without scoring.

The bounded AST adaptation only changes dispatch policy and defers CPU scoring.
There is no Torch import at module load or while checking the adaptation.
"""

from __future__ import annotations

import ast
from collections import Counter
import copy
from dataclasses import asdict
from pathlib import Path
import time

from .contract import DRIVERS, ROOT, candidate_key, digest, file_hash, read_json, verify_ref


def execution_ast(source=None):
    source = source or (DRIVERS / "run_lia_gpu_four_group_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute"))
    counts = Counter()

    class Adapt(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            if node.module == "lia_gpu_matrix_scoring_v2":
                counts["defer_scoring_import"] += 1
                return ast.parse("stability = self.stability").body[0]
            if node.module == "torch_flowline_1d_matrix_v2":
                if [n.name for n in node.names] != ["make_state", "compact", "snapshot"]:
                    raise ValueError("state-factory import shape changed")
                counts["reuse_active_prefix"] += 1
                return ast.parse("from torch_flowline_1d_matrix_v2 import compact, snapshot\n"
                                 "from torch_flowline_1d_active_prefix_v1 import make_optimized_state as make_state").body
            return node

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            names = [x.id for x in node.targets if isinstance(x, ast.Name)]
            if names == ["end"]:
                counts["explicit_spinup_end"] += 1
                node.value = ast.parse("self.policy['spinup_end_year'] if kind == 'spin' else 2000", mode="eval").body
            elif names == ["confirmations"]:
                counts["explicit_confirmations"] += 1
                node.value = ast.parse("{p['year']: p['boundary'] for p in self.policy['confirmation_pairs']}", mode="eval").body
            elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "score":
                counts["defer_terminal_scoring"] += 1
                node.value = ast.parse("{'status': 'forward-complete-unscored', 'provisional_pass': False}", mode="eval").body
            return node

    fn = Adapt().visit(fn)
    expected = dict(defer_scoring_import=1, reuse_active_prefix=1, explicit_spinup_end=1,
                    explicit_confirmations=1, defer_terminal_scoring=1)
    if dict(counts) != expected:
        raise ValueError("original execution-loop shape changed: " + str(dict(counts)))
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "score" for n in ast.walk(fn)):
        raise ValueError("CPU scorer remains in GPU execution")
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), expected


def extracted_stability():
    # Exact original function, including NumPy arithmetic order, but do not import
    # its module (which also imports the CPU-only lag/scoring dependency chain).
    import numpy as np
    tree = ast.parse((DRIVERS / "lia_gpu_matrix_scoring_v2.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stability")
    scope = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "<frozen stability only>", "exec"), scope)
    return scope["stability"]


class FixedSearch:
    """A fixed member list; no four-group search, beam, or fixed GPU lane limit."""

    def __init__(self, baselines, source_contract, policy, maximum):
        self.baselines = baselines
        self.contract = digest({"source_contract": source_contract, "scientific_policy": policy})
        self.maximum = maximum

    def spin_key(self, candidate):
        parameters = asdict(candidate)
        parameters.pop("forward_end")
        return digest({"stage": "spin", "candidate": parameters,
                       "baseline": self.baselines[candidate.system], "contract": self.contract})

    def forward_key(self, candidate, state_hash):
        return digest({"stage": "forward", "candidate": asdict(candidate),
                       "spin_key": self.spin_key(candidate), "state_hash": state_hash, "contract": self.contract})

    def pack(self, candidates, nx):
        ordered = sorted(candidates, key=lambda c: (nx[c.system], c.system, c.key))
        return [ordered[i:i + self.maximum] for i in range(0, len(ordered), self.maximum)]


def build_runtime(plan, output_root, device, on_terminal=None):
    """GPU-child-only construction, after interpreter/environment/launch guards."""
    import numpy as np
    from lia_gpu_four_group_search_v2 import Candidate, load_search
    from lia_gpu_matrix_io_v2 import Store
    import run_lia_gpu_four_group_v2 as original

    inputs = verify_ref(ROOT, plan["inputs"]["manifest"])
    manifest = original.check_inputs(inputs.parent)
    original_search = load_search(manifest["contract"])
    if manifest["contract"] != plan["source_contract"]:
        raise ValueError("input source contract changed")
    bundle_path = verify_ref(ROOT, plan["inputs"]["arrays"])
    with np.load(bundle_path, allow_pickle=False) as z:
        bundle = {k: z[k] for k in z.files}
    search = FixedSearch(original_search.baselines, manifest["contract"], plan["scientific_policy"],
                         plan["resources"]["max_batch_size"])
    candidates = [Candidate(**m["candidate"]) for m in plan["members"]]
    parents = {}
    for member, candidate in zip(plan["members"], candidates):
        if member["parent"]:
            parent = member["parent"]
            value = read_json(verify_ref(ROOT, parent["result"]))
            verify_ref(ROOT, parent["state"])
            if original_search.spin_key(candidate) != value["key"]:
                raise ValueError("original parent identity is not preserved")
            parents[search.spin_key(candidate)] = parent

    class BoundStore(Store):
        def read(self, kind, key):
            if kind == "spin" and key in parents:
                parent = parents[key]
                value = read_json(verify_ref(ROOT, parent["result"]))
                verify_ref(ROOT, parent["state"])
                return value
            value = super().read(kind, key)
            if value is not None and value.get("scientific_fingerprint") != search.contract:
                raise ValueError("stored result scientific fingerprint changed")
            return value

        def arrays(self, kind, key):
            if kind == "spin" and key in parents:
                self.read(kind, key)
                with np.load(verify_ref(ROOT, parents[key]["state"]), allow_pickle=False) as z:
                    return {name: z[name].copy() for name in z.files}
            return super().arrays(kind, key)

        def commit(self, kind, key, value, arrays=None):
            if kind == "spin" and key in parents:
                raise PermissionError("inherited parent is read-only")
            return super().commit(kind, key, {**value, "scientific_fingerprint": search.contract}, arrays)

        def checkpoint(self, kind, key, year, value, arrays):
            return super().checkpoint(kind, key, year, {**value, "scientific_fingerprint": search.contract}, arrays)

        def latest(self, kind, key):
            saved = super().latest(kind, key)
            if saved is not None and saved[0].get("scientific_fingerprint") != search.contract:
                raise ValueError("checkpoint scientific fingerprint changed")
            return saved

    class ProductionRuntime(original.Runtime):
        def persist(self, kind, key, value, arrays=None):
            result = super().persist(kind, key, value, arrays)
            if on_terminal is not None:
                on_terminal(self, kind, key, result)
            return result

        def notify(self, **values):
            import torch
            import os
            # Monitoring failure is nonfatal; a numerical/timeout error is not.
            from lia_gpu_matrix_io_v2 import status
            elapsed = time.perf_counter() - self.started
            status(self.root / "status.json", {"wall_seconds": elapsed,
                   "pid": os.getpid(), "cpu_process_seconds": time.process_time(),
                   "gpu_allocated_bytes": torch.cuda.memory_allocated(),
                   "gpu_peak_bytes": torch.cuda.max_memory_allocated(), **values})
            if elapsed > plan["resources"]["phase_timeout_seconds"]:
                raise TimeoutError("production phase budget reached; committed states preserved")

    tree, changes = execution_ast()
    namespace = vars(original).copy()
    exec(compile(tree, "<production dispatch; original arithmetic>", "exec"), namespace)
    ProductionRuntime.execute = namespace["execute"]
    runtime = ProductionRuntime(search, manifest, bundle, output_root, device)
    runtime.store = BoundStore(output_root)
    runtime.policy = plan["scientific_policy"]
    runtime.stability = extracted_stability()
    return runtime, candidates, changes


def six_gate_revision(original):
    """Execute only the already-approved pure revision, never its route writer."""
    if "gates" not in original:
        return original
    path = ROOT / "validation_data/results/paleo_surface_validation_v0/lia_gpu_offline_direction_policy_v2/reassess.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == "revise") or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "GATES" for t in n.targets))]
    if len(nodes) != 2:
        raise ValueError("frozen six-gate revision shape changed")
    namespace = {"copy": copy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<frozen six-gate revision>", "exec"), namespace)
    return namespace["revise"](original)


def static_adapter_report():
    tree, changes = execution_ast()
    compile(tree, "<static production integration check>", "exec")
    return {"status": "adapter-ast-pass", "changes": changes,
            "source_sha256": file_hash(DRIVERS / "run_lia_gpu_four_group_v2.py"),
            "solver_modified": False, "model_calls": 0}
