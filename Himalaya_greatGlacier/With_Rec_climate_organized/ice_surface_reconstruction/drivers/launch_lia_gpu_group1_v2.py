"""Authorized first-group-only production using the unchanged tested runtime."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from lia_gpu_four_group_search_v2 import ROOT, file_hash, load_search
from lia_gpu_matrix_io_v2 import WriterLock, atomic_json, status
from prepare_lia_gpu_four_group_v2 import OUTPUT
from register_lia_gpu_matrix_implementation_v2 import PATCHER, relative
from run_lia_gpu_four_group_v2 import BRANCH, Runtime, authorize, check_inputs

HERE = Path(__file__).resolve().parent
TX = ROOT / 'validation_data/results/paleo_surface_validation_v0/lia_gpu_group1_v2_production_v1'
RUN = ROOT / 'wg2g1'
AUTH = TX / 'authorization.json'
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def now():
    return datetime.now(timezone.utc).isoformat()


def validate():
    subprocess.run([sys.executable, '-B', str(HERE / 'validate_lia_execution_state_v1.py')],
                   check=True, creationflags=NO_WINDOW)


def transition(label, evidence, launch=False, **extra):
    validate()
    path = ROOT / 'LIA_CURRENT_STATE.json'
    before = file_hash(path)
    state = json.loads(path.read_text(encoding='utf-8-sig'))
    facts_path = ROOT / 'LIA_PROJECT_FACTS.md'
    facts = facts_path.read_text(encoding='utf-8-sig')
    facts_before = file_hash(facts_path)
    if facts_before != state['scientific_contract']['sha256']:
        raise RuntimeError('concurrent facts change')
    proof = dict(path=relative(evidence), sha256=file_hash(evidence))
    prior = state['authoritative_route']
    route = json.loads((ROOT / prior['path']).read_text(encoding='utf-8-sig'))
    route['independent_gpu_group1'] = dict(status=label, evidence=proof,
        launch_authorized=launch, authorized_groups=[1], forward_authorized=False)
    route['superseded_route'] = prior
    route_path = TX / ('route-' + label + '.json')
    if route_path.exists():
        raise FileExistsError('immutable transition already exists')
    atomic_json(route_path, route)
    section = max(map(int, re.findall(r'^## (\d+)\.', facts, re.MULTILINE))) + 1
    description = {
        'authorized': 'The user replies start to the explicit first-group-only proposal: 191 systems at the 21 frozen equilibrium temperature offsets, 4,011 unique spin-ups, cap 1,500 years, early convergence under the frozen 300-year/30-year confirmation rule. Only group one is authorized; no forward or groups 2-4. The unchanged numerical runtime and input contract passed the 24-lane CUDA 30-year recovery/status-failure test. A new scope-only wrapper invokes Runtime.execute(candidates, spin) once and never invokes the all-group run method. One CUDA worker packs at most 512 lanes with the existing deterministic grid-length ordering. Every committed result and checkpoint is preserved; recovery requires separate explicit authority. The shared CPU terminal decision remains untouched and does not authorize any CPU launch. A hidden Windows-owned supervisor provides process lifetime independent of the interactive tool, with logs and an OS single-writer lock; no recurring schedule or service is installed.',
        'active': 'The first-group-only GPU worker has created its unique launch record and is active. Duplicate launch authority is consumed. It may execute only the frozen 4,011 spin-ups and must exit after group one. CPU fields, CPU outputs and prior GPU results remain unchanged.',
        'terminal': 'The first-group-only GPU process has exited. Immutable terminal evidence recounts and validates member results independently of status, records exit classification, log hashes and the final summary when present. No automatic retry, forward run, later group or acceptance promotion is authorized.'
    }[label]
    addition = '\n\n## ' + str(section) + '. Independent GPU first-group production ' + label + ' (2026-09-05)\n\n- ' + description + '\n- Evidence: `' + relative(evidence) + '`. Accepted count effect is zero.\n'
    log_path = ROOT / 'LIA_DECISION_LOG.md'
    log = log_path.read_text(encoding='utf-8-sig')
    log_add = '\n\n## 2026-09-05 GPU first-group ' + label + '\n\n- ' + description + '\n- Evidence: `' + relative(evidence) + '`.\n'
    patch = '*** Begin Patch\n'
    for target, old, extra_text in ((facts_path, facts, addition), (log_path, log, log_add)):
        patch += '*** Update File: ' + str(target).replace('\\', '/') + '\n@@\n'
        patch += '\n'.join(' ' + line for line in old.rstrip().splitlines()[-4:]) + '\n'
        patch += '\n'.join('+' + line for line in extra_text.splitlines()) + '\n'
    patch += '*** End Patch'
    if file_hash(path) != before or file_hash(facts_path) != facts_before:
        raise RuntimeError('concurrent shared update; do not overwrite')
    subprocess.run([str(PATCHER), '--codex-run-as-apply-patch', patch], check=True, creationflags=NO_WINDOW)
    if file_hash(path) != before:
        raise RuntimeError('concurrent state update after append; reconcile without overwrite')
    updated = deepcopy(state)
    updated['scientific_contract']['sha256'] = file_hash(facts_path)
    updated['state_id'] = state['state_id'] + '+gpu-group1-' + label
    updated['updated_at'] = now()
    updated['authoritative_route'] = dict(version=prior['version'] + '+gpu-group1-' + label,
        path=relative(route_path), sha256=file_hash(route_path), supersedes=prior['version'])
    updated['research_branches'][BRANCH].update(status='group1-' + label, evidence=proof,
        launch_authorized=launch, retry_authorized=False, authorized_groups=[1],
        forward_authorized=False, full_191_gpu_matrix_authorized=False,
        group1_production_authorized=(label != 'terminal'), output_root=relative(RUN), **extra)
    exempt = {'scientific_contract', 'state_id', 'updated_at', 'authoritative_route', 'research_branches'}
    for key in state.keys() - exempt:
        assert updated[key] == state[key], key
    for key in state['research_branches'].keys() - {BRANCH}:
        assert updated['research_branches'][key] == state['research_branches'][key], key
    atomic_json(path, updated)
    validate()


def prepare():
    validate()
    manifest = check_inputs(OUTPUT)
    search = load_search(manifest['contract'])
    cells = search.candidates(1, [])
    assert len(cells) == 4011 and len({c.system for c in cells}) == 191
    assert len({search.spin_key(c) for c in cells}) == 4011
    proof = ROOT / 'wg2pf1/summary.json'
    preflight = json.loads(proof.read_text())
    assert preflight['status'] == 'pass' and preflight['contract'] == manifest['contract']
    if TX.exists() or RUN.exists():
        raise FileExistsError('first-group transaction or output already exists; no duplicate')
    TX.mkdir(parents=True)
    atomic_json(TX / 'plan.json', [asdict(c) for c in cells])
    auth = dict(mode='production', launch_authorized=True, resume=False,
        contract=manifest['contract'], output_root=relative(RUN), created_at_utc=now(),
        preflight=dict(path=relative(proof), sha256=file_hash(proof)),
        authorized_groups=[1], forward_authorized=False, cells=4011, systems=191,
        batch_lanes=512, maximum_years=1500, early_convergence=True,
        checkpoint_years=[330, 630, 930, 1230], terminal_years=[330, 630, 930, 1230, 1500],
        diagnostics_interval_years=30, precision='float64',
        numerical_core_changed=False, wrapper_sha256=file_hash(Path(__file__)),
        plan_sha256=file_hash(TX / 'plan.json'),
        cpu_control_unchanged=True, full_four_group_search_authorized=False,
        estimated_first_group_hours=[26, 45], estimate_kind='rough extrapolation, not a deadline',
        stop_conditions=['first group complete', 'technical invalidity above 15 percent after 120 terminal cells',
                         'unhandled numerical or runtime exception; preserve outputs'],
        user_authorization='Start the explicitly proposed group one only: 4,011 GPU spin-ups; no forward or groups 2-4.')
    atomic_json(AUTH, auth)
    transition('authorized', AUTH, True, authorization_sha256=file_hash(AUTH),
               next_legal_action='launch first-group-only worker; no forward or later groups')
    print(json.dumps(dict(status='group1-authorized', authorization=relative(AUTH))), flush=True)


def verify_scope():
    manifest, auth, root = authorize(OUTPUT, AUTH)
    if auth['authorized_groups'] != [1] or auth['forward_authorized'] or auth['wrapper_sha256'] != file_hash(Path(__file__)):
        raise PermissionError('first-group scope or wrapper identity mismatch')
    if file_hash(TX / 'plan.json') != auth['plan_sha256']:
        raise PermissionError('first-group plan changed')
    return manifest, auth, root


class MonitoredRuntime(Runtime):
    def notify(self, **values):
        counts = Counter(row['status'] for (kind, key), row in self.completed.items() if kind == 'spin')
        super().notify(**values, stage_counts=dict(counts), completed=sum(counts.values()),
                       total=4011, authorized_groups=[1], forward_started=False)


def group_one(runtime, search):
    candidates = search.candidates(1, [])
    assert len(candidates) == 4011
    spins = runtime.execute(candidates, 'spin')
    return [dict(group=1, candidate=asdict(c), spinup=spins[c.key], forward={}) for c in candidates]


def worker():
    manifest, auth, root = verify_scope()
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch, _, _ = import_torch_with_stubs_filtered()
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with np.load(OUTPUT / 'inputs.npz', allow_pickle=False) as data:
        bundle = {key: data[key] for key in data.files}
    search = load_search(manifest['contract'])
    with WriterLock(root):
        if (root / 'launch.json').exists():
            raise PermissionError('existing production output; recovery is not authorized')
        atomic_json(root / 'launch.json', dict(pid=os.getpid(), started_at_utc=now(),
            contract=manifest['contract'], authorization_sha256=file_hash(AUTH), authorized_groups=[1],
            maximum_years=1500, batch_lanes=512, output_root=relative(root)))
        runtime = MonitoredRuntime(search, manifest, bundle, root, torch.device('cuda:0'))
        try:
            rows = group_one(runtime, search)
            atomic_json(root / 'group-1.json', dict(contract=search.contract,
                plan=[r['candidate'] for r in rows], records=rows))
            counts = dict(Counter(r['spinup']['status'] for r in rows))
            summary = dict(status='completed-group1', contract=search.contract, counts=counts,
                completed=len(rows), systems=191, forward_started=False, accepted_ledger_effect=0,
                wall_seconds=time.perf_counter() - runtime.started)
            atomic_json(root / 'summary.json', summary)
            runtime.notify(stage='spin', model_year=None, batch_active=0, terminal=True)
        except BaseException as error:
            status(root / 'termination.json', dict(status='technical-failure', error=repr(error),
                   completed_results_preserved=True, contract=search.contract, finished_at_utc=now()))
            raise


def supervise():
    sys.stdout = (TX / 'supervisor.stdout.log').open('x', encoding='utf-8', buffering=1)
    sys.stderr = (TX / 'supervisor.stderr.log').open('x', encoding='utf-8', buffering=1)
    with WriterLock(TX / 'supervisor'):
        manifest, auth, root = verify_scope()
        if root.exists():
            raise FileExistsError('output already exists; no automatic recovery')
        from launch_lia_191_gpu_spinup_v10_239 import child_environment, SS4_PYTHON
        command = [str(SS4_PYTHON), '-B', str(Path(__file__)), '--worker']
        started = time.perf_counter()
        with (TX / 'worker.stdout.log').open('x', encoding='utf-8') as out, (TX / 'worker.stderr.log').open('x', encoding='utf-8') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=out, stderr=err,
                                       creationflags=NO_WINDOW)
            atomic_json(TX / 'supervisor-launch.json', dict(supervisor_pid=os.getpid(), worker_pid=process.pid,
                command=command, started_at_utc=now(), ownership='Windows WMI-created hidden supervisor'))
            recorded = False
            while process.poll() is None:
                if not recorded and (root / 'launch.json').exists():
                    info = json.loads((root / 'launch.json').read_text())
                    info.update(supervisor_pid=os.getpid(), command=command)
                    atomic_json(TX / 'active.json', info)
                    transition('active', TX / 'active.json', gpu_pid=process.pid,
                        supervisor_pid=os.getpid(), next_legal_action='monitor existing group-one worker only')
                    recorded = True
                time.sleep(2)
            exit_code = process.wait()
        result = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else None
        from lia_gpu_matrix_io_v2 import Store
        store = Store(root)
        counts = Counter()
        corrupt = []
        search = load_search(manifest['contract'])
        for candidate in search.candidates(1, []):
            try:
                key = search.spin_key(candidate)
                row = store.read('spin', key)
                quarantine = root / 'quarantine' / ('spin-' + key[:16] + '.json')
                if row is None and quarantine.exists():
                    row = json.loads(quarantine.read_text())
                    if row.get('key') != key or row.get('status') != 'technical-failure':
                        raise ValueError('invalid quarantine record')
                if row:
                    counts[row['status']] += 1
            except Exception as error:
                corrupt.append(dict(candidate=asdict(candidate), error=repr(error)))
        terminal = dict(status='completed-group1' if exit_code == 0 and result and sum(counts.values()) == 4011 and not corrupt else 'technical-failure',
            exit_code=exit_code, finished_at_utc=now(), wall_seconds=time.perf_counter() - started,
            contract=manifest['contract'], result=result, verified_member_counts=dict(counts), corrupt=corrupt,
            pending=4011-sum(counts.values()), quarantine_files=len(list((root / 'quarantine').glob('*.json'))),
            logs={name:file_hash(TX/name) for name in ('worker.stdout.log','worker.stderr.log')},
            forward_started=False, cpu_control_unchanged=True, accepted_ledger_effect=0)
        atomic_json(TX / 'terminal.json', terminal)
        transition('terminal', TX / 'terminal.json', terminal_classification=terminal['status'],
                   next_legal_action='report group-one results; no automatic forward, later group or retry')


if __name__ == '__main__':
    if sys.argv[1:] == ['--prepare']:
        prepare()
    elif sys.argv[1:] == ['--supervise']:
        supervise()
    elif sys.argv[1:] == ['--worker']:
        worker()
    else:
        raise SystemExit('Use --prepare, --supervise or --worker; no implicit launch')
