"""One registered checkpoint-boundary handoff to the verified GPU backend."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from lia_gpu_four_group_search_v2 import ROOT, digest, file_hash, load_search
from lia_gpu_matrix_io_v2 import WriterLock, atomic_json, status
from lia_gpu_prefix_recovery_store_v1 import RecoveryStore, inventory_source
from prepare_lia_gpu_four_group_v2 import OUTPUT
from register_lia_gpu_matrix_implementation_v2 import PATCHER, relative
from run_lia_gpu_four_group_v2 import BRANCH as SOURCE_BRANCH, check_inputs
from launch_lia_gpu_group1_v2 import MonitoredRuntime, group_one

HERE = Path(__file__).resolve().parent
TX = ROOT / 'validation_data/results/paleo_surface_validation_v0/lia_gpu_prefix_production_v1'
SOURCE = ROOT / 'wg2g1'
SOURCE_TX = ROOT / 'validation_data/results/paleo_surface_validation_v0/lia_gpu_group1_v2_production_v1'
RUN = ROOT / 'wg2gp1'
AUTH = TX / 'authorization.json'
BRANCH = 'lia_gpu_prefix_production_v1'
PROOF = ROOT / 'validation_data/results/paleo_surface_validation_v0/lia_gpu_active_prefix_long_verification_v2/terminal.json'
NO_WINDOW = 0x08000000 if os.name == 'nt' else 0


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate():
    subprocess.run([sys.executable, '-B', str(HERE / 'validate_lia_execution_state_v1.py')],
                   check=True, creationflags=NO_WINDOW)


def transition(label, evidence, **extra):
    validate()
    path = ROOT / 'LIA_CURRENT_STATE.json'
    before = file_hash(path)
    state = read(path)
    facts_path = ROOT / 'LIA_PROJECT_FACTS.md'
    facts = facts_path.read_text(encoding='utf-8-sig')
    facts_before = file_hash(facts_path)
    if facts_before != state['scientific_contract']['sha256']:
        raise RuntimeError('concurrent facts change')
    prior = state['authoritative_route']
    route = read(ROOT / prior['path'])
    proof = dict(path=relative(evidence), sha256=file_hash(evidence))
    launch = label == 'authorized'
    route['independent_gpu_prefix_production'] = dict(status=label, evidence=proof,
        launch_authorized=launch, authorized_groups=[1], forward_authorized=False)
    route['superseded_route'] = prior
    destination = TX / ('route-' + label + '.json')
    if destination.exists():
        raise FileExistsError('immutable transition exists')
    atomic_json(destination, route)
    description = {
        'authorized': 'The user explicitly requests production integration after the twelve-lane 330-year original-versus-active-prefix comparison and independent process disk recovery both pass. One checkpoint-boundary handoff of the existing GPU group-one worker is authorized. Only the registered original GPU worker may be terminated, after a newly completed checkpoint or batch boundary; its supervisor must finish before the replacement starts. All immutable terminal results, including technical failures, are reused without recomputation; unfinished candidates resume hash-validated original checkpoints or start only when no earlier checkpoint exists. The original scientific/numerical cache contract is retained, with a separate execution fingerprint binding the optimized implementation, verification and inherited inventory. The new root wg2gp1 never writes into wg2g1. One CUDA worker, at most 512 lanes, float64, the same climate, parameters, timestep, full physical domain and gates, and the 1500-year cap remain unchanged. Only group one is authorized, not forward or groups 2-4. The measured twelve-lane speedup is 1.15077, not a promise of the synthetic 4-6-fold rate. No CPU control, queue, output or accepted count changes are authorized.',
        'waiting': 'The unique GPU handoff supervisor is waiting for the original worker to publish its next complete checkpoint or terminal batch boundary. It has not launched a second production GPU worker. CPU controls and the original implementation remain unchanged.',
        'source-stopped': 'The original GPU worker has intentionally stopped at the recorded complete boundary for the user-authorized optimized production handoff. Its old generic supervisor may label the native exit technical-failure; the new evidence explicitly supersedes that cause with intentional handoff, without changing the old evidence. A full independent recount and hash-bound inheritance inventory now covers all 4011 combinations. Only unfinished members may compute in the new output root.',
        'active': 'The optimized first-group GPU worker is active under the unique new execution fingerprint. The original worker and supervisor are absent, committed source results remain read-only, and the active source inventory is preserved. No completed member is rerun, no forward or later parameter group starts, and CPU controls remain unchanged.',
        'terminal': 'The optimized GPU production supervisor has recorded immutable termination evidence with an independent recount of inherited and new results. Launch authorization is consumed. No automatic retry or subsequent parameter group is authorized.'
    }[label]
    section = max(map(int, re.findall(r'^## (\d+)\.', facts, re.MULTILINE))) + 1
    addition = f'\n\n## {section}. GPU optimized production {label} (2026-09-05)\n\n- {description}\n- Evidence: `{relative(evidence)}`. Accepted count effect is zero.\n'
    log_path = ROOT / 'LIA_DECISION_LOG.md'
    log = log_path.read_text(encoding='utf-8-sig')
    patch = '*** Begin Patch\n'
    for target, old, extra_text in ((facts_path, facts, addition),
        (log_path, log, f'\n\n## GPU optimized production {label}\n\n- {description}\n- Evidence: `{relative(evidence)}`.\n')):
        patch += '*** Update File: ' + str(target).replace('\\', '/') + '\n@@\n'
        patch += '\n'.join(' ' + line for line in old.rstrip().splitlines()[-4:]) + '\n'
        patch += '\n'.join('+' + line for line in extra_text.splitlines()) + '\n'
    patch += '*** End Patch'
    if file_hash(path) != before or file_hash(facts_path) != facts_before:
        raise RuntimeError('concurrent shared update; no overwrite')
    subprocess.run([str(PATCHER), '--codex-run-as-apply-patch', patch], check=True, creationflags=NO_WINDOW)
    if file_hash(path) != before:
        raise RuntimeError('concurrent state update; reconcile')
    updated = deepcopy(state)
    updated['scientific_contract']['sha256'] = file_hash(facts_path)
    updated['state_id'] += '+gpu-prefix-production-' + label
    updated['updated_at'] = now()
    updated['authoritative_route'] = dict(version=prior['version'] + '+gpu-prefix-production-' + label,
        path=relative(destination), sha256=file_hash(destination), supersedes=prior['version'])
    updated['research_branches'][BRANCH] = dict(status=label, evidence=proof,
        launch_authorized=launch, authorization_sha256=file_hash(AUTH), output_root=relative(RUN),
        source_output_root=relative(SOURCE), authorized_groups=[1], forward_authorized=False,
        accepted_count_effect=0, cpu_control_unchanged=True, **extra)
    if label in ('source-stopped', 'active', 'terminal'):
        updated['research_branches'][SOURCE_BRANCH].update(status='superseded-by-optimized-group1',
            launch_authorized=False, group1_production_authorized=False,
            successor_branch=BRANCH, next_legal_action='read-only inherited source; no original worker restart')
    exempt = {'scientific_contract', 'state_id', 'updated_at', 'authoritative_route', 'research_branches'}
    assert all(updated[k] == state[k] for k in state.keys() - exempt)
    assert all(updated['research_branches'][k] == v for k, v in state['research_branches'].items()
               if k not in (BRANCH, SOURCE_BRANCH))
    atomic_json(path, updated)
    validate()


def process_identity(pid):
    import psutil
    p = psutil.Process(pid)
    return dict(pid=pid, created=p.create_time(), exe=p.exe(), command=p.cmdline())


def same_process(identity):
    import psutil
    try:
        return process_identity(identity['pid']) == identity
    except psutil.NoSuchProcess:
        return False


def prepare():
    validate()
    manifest = check_inputs(OUTPUT)
    proof = read(PROOF)
    if proof['status'] != 'pass' or not proof['long_run_comparison_passed'] or not proof['cross_process_disk_resume_passed']:
        raise PermissionError('required optimization verification missing')
    tested = read(PROOF.parent / 'authorization.json')
    if tested['source_files']['torch_flowline_1d_active_prefix_v1.py'] != file_hash(HERE / 'torch_flowline_1d_active_prefix_v1.py'):
        raise ValueError('optimized backend differs from verified version')
    if TX.exists() or RUN.exists():
        raise FileExistsError('handoff already registered; no duplicate')
    old_launch = read(SOURCE / 'launch.json')
    supervisor = read(SOURCE_TX / 'supervisor-launch.json')
    worker = process_identity(old_launch['pid'])
    parent = process_identity(supervisor['supervisor_pid'])
    if worker['command'] != supervisor['command'] or worker['pid'] != supervisor['worker_pid']:
        raise ValueError('original GPU process identity mismatch')
    if old_launch['contract'] != manifest['contract']:
        raise ValueError('original production contract changed')
    snapshot = read(SOURCE / 'status.json')
    # Wait for a fresh complete boundary, never kill while a checkpoint writes.
    minimum_wall = snapshot['wall_seconds']
    TX.mkdir(parents=True)
    auth = dict(user_authorization='production integration', created_at_utc=now(),
        contract=manifest['contract'], source_root=relative(SOURCE), output_root=relative(RUN),
        source_worker=worker, source_supervisor=parent, minimum_source_wall_seconds=minimum_wall,
        source_launch_sha256=file_hash(SOURCE / 'launch.json'),
        source_supervisor_launch_sha256=file_hash(SOURCE_TX / 'supervisor-launch.json'),
        verification=dict(path=relative(PROOF), sha256=file_hash(PROOF)),
        source_files={name:file_hash(HERE / name) for name in
            (Path(__file__).name, 'lia_gpu_prefix_recovery_store_v1.py', 'torch_flowline_1d_active_prefix_v1.py')},
        launch_authorized=True, authorized_groups=[1], forward_authorized=False,
        maximum_years=1500, batch_lanes=512, precision='float64', gpu_memory_fraction=.8,
        checkpoint_years=[330,630,930,1230], completed_recomputation_allowed=False,
        automatic_retry_allowed=False, cpu_control_unchanged=True,
        expected_remaining_hours=[15,32], estimate='rough; actual batch throughput pending',
        handoff_wait_limit_seconds=14400, old_supervisor_exit_timeout_seconds=180)
    auth['execution_fingerprint'] = digest(auth)
    atomic_json(AUTH, auth)
    transition('authorized', AUTH)
    print(json.dumps(dict(status='authorized', execution_fingerprint=auth['execution_fingerprint'])))


def verify(allowed):
    manifest = check_inputs(OUTPUT)
    auth = read(AUTH)
    if digest({k:v for k,v in auth.items() if k != 'execution_fingerprint'}) != auth['execution_fingerprint']:
        raise ValueError('execution fingerprint mismatch')
    if auth['contract'] != manifest['contract'] or auth['authorized_groups'] != [1] or auth['forward_authorized']:
        raise PermissionError('scope mismatch')
    for name, expected in auth['source_files'].items():
        if Path(name).name != name or file_hash(HERE / name) != expected:
            raise ValueError('optimized production source changed')
    if file_hash(PROOF) != auth['verification']['sha256'] or file_hash(SOURCE / 'launch.json') != auth['source_launch_sha256']:
        raise ValueError('inherited contract evidence changed')
    branch = read(ROOT / 'LIA_CURRENT_STATE.json')['research_branches'].get(BRANCH, {})
    if branch.get('status') not in allowed or branch.get('authorization_sha256') != file_hash(AUTH):
        raise PermissionError('no matching GPU-only handoff authority')
    return manifest, auth


def worker():
    manifest, auth = verify(('source-stopped',))
    if same_process(auth['source_worker']) or same_process(auth['source_supervisor']):
        raise RuntimeError('original GPU process still exists')
    proof = read(TX / 'source-stopped.json')
    if file_hash(TX / 'inventory.json') != proof['inventory_sha256']:
        raise ValueError('recovery inventory changed')
    inventory = read(TX / 'inventory.json')
    from run_lia_191_gpu_spinup_v10_239 import verify_child_environment, import_torch_with_stubs_filtered
    verify_child_environment()
    torch, _, _ = import_torch_with_stubs_filtered()
    import numpy as np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(auth['gpu_memory_fraction'])
    # Replace the factory in this new process only. Original files stay frozen.
    import torch_flowline_1d_matrix_v2 as factory
    from torch_flowline_1d_active_prefix_v1 import make_optimized_state
    factory.make_state = make_optimized_state
    search = load_search(manifest['contract'])
    with np.load(OUTPUT / 'inputs.npz', allow_pickle=False) as data:
        bundle = {k:data[k] for k in data.files}
    with WriterLock(RUN):
        if (RUN / 'launch.json').exists():
            raise PermissionError('duplicate optimized production launch')
        runtime = MonitoredRuntime(search, manifest, bundle, RUN, torch.device('cuda:0'))
        runtime.store = RecoveryStore(RUN, SOURCE, inventory)
        runtime.completed = {('spin',k):v['result'] for k,v in inventory.items() if v['mode']=='terminal'}
        atomic_json(RUN / 'launch.json', dict(pid=os.getpid(), started_at_utc=now(),
            execution_fingerprint=auth['execution_fingerprint'], contract=search.contract,
            inventory_sha256=proof['inventory_sha256'], backend='active-prefix-v1',
            inherited_counts=proof['counts'], source_root=relative(SOURCE)))
        try:
            rows = group_one(runtime, search)
            atomic_json(RUN / 'group-1.json', dict(contract=search.contract, records=rows,
                plan=[r['candidate'] for r in rows], execution_fingerprint=auth['execution_fingerprint']))
            summary = dict(status='completed-group1', contract=search.contract,
                execution_fingerprint=auth['execution_fingerprint'], completed=len(rows),
                counts=dict(Counter(r['spinup']['status'] for r in rows)),
                inherited_counts=proof['counts'], completed_recomputed=0, forward_started=False,
                wall_seconds=time.perf_counter()-runtime.started)
            atomic_json(RUN / 'summary.json', summary)
            runtime.notify(stage='spin', model_year=None, batch_active=0, terminal=True)
        except BaseException as error:
            status(RUN / 'termination.json', dict(status='technical-failure', error=repr(error),
                completed_results_preserved=True, execution_fingerprint=auth['execution_fingerprint']))
            raise


def supervise():
    sys.stdout = (TX / 'supervisor.stdout.log').open('x', encoding='utf-8', buffering=1)
    sys.stderr = (TX / 'supervisor.stderr.log').open('x', encoding='utf-8', buffering=1)
    with WriterLock(TX / 'supervisor'):
        manifest, auth = verify(('authorized',))
        validate()
        atomic_json(TX / 'waiting.json', dict(pid=os.getpid(), started_at_utc=now(),
            execution_fingerprint=auth['execution_fingerprint'], source_worker=auth['source_worker']))
        transition('waiting', TX / 'waiting.json', supervisor_pid=os.getpid())
        child = None
        try:
            started = time.monotonic()
            boundary = None
            while same_process(auth['source_worker']):
                if time.monotonic()-started > auth['handoff_wait_limit_seconds']:
                    raise TimeoutError('handoff timed out; original worker left running')
                snapshot = read(SOURCE / 'status.json')
                if (snapshot.get('wall_seconds', 0) > auth['minimum_source_wall_seconds'] and
                    snapshot.get('model_year') in (330,630,930,1230,1500)):
                    boundary = snapshot
                    break
                time.sleep(.5)
            if boundary is not None:
                # Recheck the full identity immediately before terminating only this child.
                verify(('waiting',))
                import psutil
                if not same_process(auth['source_worker']):
                    raise RuntimeError('source identity changed before handoff')
                atomic_json(TX / 'stop-intent.json', dict(reason='user-authorized-checkpoint-handoff',
                    boundary=boundary, worker=auth['source_worker'], created_at_utc=now()))
                p = psutil.Process(auth['source_worker']['pid'])
                p.terminate()
                p.wait(timeout=30)
            deadline = time.monotonic() + auth['old_supervisor_exit_timeout_seconds']
            while same_process(auth['source_supervisor']):
                if time.monotonic() > deadline:
                    raise TimeoutError('old supervisor has not closed; replacement not launched')
                time.sleep(1)
            if not (SOURCE_TX / 'terminal.json').exists():
                raise RuntimeError('old supervisor terminal evidence missing')
            if boundary is None and read(SOURCE_TX / 'terminal.json')['status'] != 'completed-group1':
                raise RuntimeError('unexpected original failure; automatic retry not authorized')
            inventory = inventory_source(SOURCE, load_search(manifest['contract']), manifest)
            atomic_json(TX / 'inventory.json', inventory)
            modes = dict(Counter(v['mode'] for v in inventory.values()))
            counts = dict(Counter(v['result']['status'] for v in inventory.values() if v['mode']=='terminal'))
            proof = dict(status='intentional-checkpoint-handoff' if boundary else 'source-already-exited',
                boundary=boundary, modes=modes, counts=counts, completed_recomputed=0,
                source_terminal_sha256=file_hash(SOURCE_TX / 'terminal.json'),
                inventory_sha256=file_hash(TX / 'inventory.json'), contract=manifest['contract'],
                finished_at_utc=now(), original_gpu_processes_absent=True)
            atomic_json(TX / 'source-stopped.json', proof)
            transition('source-stopped', TX / 'source-stopped.json', supervisor_pid=os.getpid())
            from launch_lia_191_gpu_spinup_v10_239 import SS4_PYTHON, child_environment
            command = [str(SS4_PYTHON), '-B', str(Path(__file__)), '--worker']
            with (TX / 'worker.stdout.log').open('x', encoding='utf-8') as out, (TX / 'worker.stderr.log').open('x', encoding='utf-8') as err:
                child = subprocess.Popen(command, cwd=ROOT, env=child_environment(), stdout=out,
                    stderr=err, creationflags=NO_WINDOW)
                atomic_json(TX / 'child-launch.json', dict(pid=child.pid, command=command, started_at_utc=now()))
                recorded = False
                while child.poll() is None:
                    if not recorded and (RUN / 'launch.json').exists():
                        active = read(RUN / 'launch.json')
                        active['supervisor_pid'] = os.getpid()
                        atomic_json(TX / 'active.json', active)
                        transition('active', TX / 'active.json', gpu_pid=child.pid, supervisor_pid=os.getpid())
                        recorded = True
                    time.sleep(2)
                exit_code = child.wait()
            store = RecoveryStore(RUN, SOURCE, inventory)
            rows, errors = [], []
            for key in inventory:
                try:
                    row = store.read('spin',key)
                    q = RUN / 'quarantine' / ('spin-' + key[:16] + '.json')
                    if row is None and q.exists():
                        row = read(q)
                        if row.get('key') != key or row.get('status') != 'technical-failure':
                            raise ValueError('invalid quarantine')
                    if row is not None:
                        rows.append(row)
                except Exception as error:
                    errors.append(dict(key=key,error=repr(error)))
            terminal = dict(status='completed-group1' if exit_code==0 and len(rows)==4011 and not errors else 'technical-failure',
                exit_code=exit_code, verified_counts=dict(Counter(r['status'] for r in rows)),
                completed=len(rows), pending=4011-len(rows), errors=errors,
                logs={n:file_hash(TX/n) for n in ('worker.stdout.log','worker.stderr.log')},
                completed_recomputed=0, forward_started=False, cpu_control_unchanged=True)
        except BaseException as error:
            if child is not None and child.poll() is None:
                # A record-writing failure must not kill an otherwise valid production worker.
                status(TX / 'supervisor-attention.json', dict(error=repr(error), worker_pid=child.pid,
                    worker_left_running=True))
                raise
            terminal = dict(status='technical-handoff-failure', error=repr(error),
                source_worker_alive=same_process(auth['source_worker']), replacement_started=child is not None)
        terminal.update(finished_at_utc=now(), execution_fingerprint=auth['execution_fingerprint'])
        atomic_json(TX / 'terminal.json', terminal)
        transition('terminal', TX / 'terminal.json')


if __name__ == '__main__':
    actions = {'--prepare':prepare, '--supervise':supervise, '--worker':worker}
    if len(sys.argv)!=2 or sys.argv[1] not in actions:
        raise SystemExit('explicit --prepare, --supervise or --worker required')
    actions[sys.argv[1]]()
