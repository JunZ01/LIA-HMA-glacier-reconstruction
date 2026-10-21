"""One explicitly authorized preflight with GPU-only shared-state transitions."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from lia_gpu_four_group_search_v2 import ROOT, file_hash
from lia_gpu_matrix_io_v2 import atomic_json
from prepare_lia_gpu_four_group_v2 import OUTPUT
from register_lia_gpu_matrix_implementation_v2 import PATCHER, relative
from run_lia_gpu_four_group_v2 import BRANCH, check_inputs

HERE = Path(__file__).resolve().parent
TX = ROOT / 'validation_data/results/paleo_surface_validation_v0/lia_191_gpu_matrix_v2_preflight_v1'
RUN = ROOT / 'wg2pf1'


def now():
    return datetime.now(timezone.utc).isoformat()


def validate():
    subprocess.run([sys.executable, '-B', str(HERE / 'validate_lia_execution_state_v1.py')], check=True)


def transition(label, evidence, launch=False, **fields):
    validate()
    path = ROOT / 'LIA_CURRENT_STATE.json'
    before = file_hash(path)
    state = json.loads(path.read_text(encoding='utf-8-sig'))
    facts_path = ROOT / 'LIA_PROJECT_FACTS.md'
    facts = facts_path.read_text(encoding='utf-8-sig')
    facts_before = file_hash(facts_path)
    if facts_before != state['scientific_contract']['sha256']:
        raise RuntimeError('concurrent scientific contract change')
    proof = {'path': relative(evidence), 'sha256': file_hash(evidence)}
    prior = state['authoritative_route']
    route = json.loads((ROOT / prior['path']).read_text(encoding='utf-8-sig'))
    route['independent_gpu_matrix_v2_preflight'] = dict(status=label, evidence=proof,
        launch_authorized=launch, full_matrix_authorized=False, cpu_control_unchanged=True)
    route['superseded_route'] = prior
    route_path = TX / ('route-' + label + '.json')
    if route_path.exists():
        raise FileExistsError('immutable transition already exists')
    atomic_json(route_path, route)
    section = max(map(int, re.findall(r'^## (\d+)\.', facts, re.MULTILINE))) + 1
    detail = {
        'authorized': 'The user explicitly permits updating shared project records solely to register and launch this GPU short test, preserving all CPU control fields and prohibiting full simulation. This new independent GPU route supersedes GPU no-launch status only; the CPU terminal decision remains unchanged. One 24-lane 30-year spin-up test saves year 15, restores continuation to year 30, compares with uninterrupted continuation, injects an unavailable status write, and measures one-year batches of 24, 128 and 512 lanes. Float64, one CUDA worker, the frozen input/code contract, and all numerical tolerances remain unchanged. No production or forward chain is authorized. Existing numerical/static test evidence is retained.',
        'active': 'The authorized GPU short test has created its unique launch record under wg2pf1 in a persistent foreground session. Duplicate launch authority is consumed. The single CUDA worker runs independently; no CPU process, output, queue or next-action field is changed.',
        'terminal': 'The single GPU short-test process has exited. The referenced immutable evidence records its actual exit code, recovery checks, timing results or technical error, and log hashes. Authorization is consumed; no retry, production, full matrix or acceptance promotion is authorized.'
    }[label]
    addition = '\n\n## ' + str(section) + '. Independent GPU matrix V2 short test ' + label + ' (2026-09-05)\n\n- ' + detail + '\n- Evidence: `' + relative(evidence) + '`. Accepted count effect is zero.\n'
    log_path = ROOT / 'LIA_DECISION_LOG.md'
    journal = log_path.read_text(encoding='utf-8-sig')
    log_addition = '\n\n## 2026-09-05 GPU matrix V2 short test ' + label + '\n\n- ' + detail + '\n- Evidence: `' + relative(evidence) + '`.\n'
    patch = '*** Begin Patch\n'
    for target, old, extra in ((facts_path, facts, addition), (log_path, journal, log_addition)):
        patch += '*** Update File: ' + str(target).replace('\\', '/') + '\n@@\n'
        patch += '\n'.join(' ' + line for line in old.rstrip().splitlines()[-4:]) + '\n'
        patch += '\n'.join('+' + line for line in extra.splitlines()) + '\n'
    patch += '*** End Patch'
    if file_hash(path) != before or file_hash(facts_path) != facts_before:
        raise RuntimeError('concurrent update; no shared files replaced')
    subprocess.run([str(PATCHER), '--codex-run-as-apply-patch', patch], check=True)
    if file_hash(path) != before:
        raise RuntimeError('concurrent state update; reconcile without overwrite')
    changed = deepcopy(state)
    changed['scientific_contract']['sha256'] = file_hash(facts_path)
    changed['authoritative_route'] = dict(version=prior['version'] + '+gpu-pf-' + label,
        path=relative(route_path), sha256=file_hash(route_path), supersedes=prior['version'])
    changed['state_id'] = state['state_id'] + '+gpu-pf-' + label
    changed['updated_at'] = now()
    branch = changed['research_branches'][BRANCH]
    branch.update(status='preflight-' + label, evidence=proof, launch_authorized=launch,
        retry_authorized=False, full_191_gpu_matrix_authorized=False,
        output_root=relative(RUN), **fields)
    # Only the independent GPU branch and shared audit pointers may change.
    exempt = {'scientific_contract', 'authoritative_route', 'state_id', 'updated_at', 'research_branches'}
    for key in state.keys() - exempt:
        assert changed[key] == state[key], key
    for key in state['research_branches'].keys() - {BRANCH}:
        assert changed['research_branches'][key] == state['research_branches'][key], key
    atomic_json(path, changed)
    validate()


def main():
    validate()
    manifest = check_inputs(OUTPUT)
    if TX.exists() or RUN.exists():
        raise FileExistsError('preflight already registered or started; no duplicate allowed')
    TX.mkdir(parents=True)
    auth_path = TX / 'authorization.json'
    auth = dict(mode='preflight', launch_authorized=True, resume=False,
        contract=manifest['contract'], output_root=relative(RUN), created_at_utc=now(),
        scope='24 lanes x 30 years with midpoint restoration; 24/128/512 x one-year timing only',
        checkpoint_year=15, full_matrix_authorized=False, cpu_control_unchanged=True,
        expected_wall_minutes=[5, 30], estimate_measured=False,
        stop_conditions=['native process error', 'exception', 'failed recovery check at completion'],
        user_authorization='Explicit permission to update shared records, register and launch GPU short test only; preserve every CPU control field.',
        implementation_manifest_sha256=file_hash(OUTPUT / 'manifest.json'),
        launcher_sha256=file_hash(Path(__file__)))
    atomic_json(auth_path, auth)
    transition('authorized', auth_path, True, authorization_sha256=file_hash(auth_path),
               next_legal_action='execute this single GPU preflight; no production')
    cmd = [sys.executable, '-B', str(HERE / 'run_lia_gpu_four_group_v2.py'),
           '--inputs', str(OUTPUT), '--authorization', str(auth_path)]
    start = time.perf_counter()
    observation = None
    with (TX / 'stdout.log').open('x', encoding='utf-8') as stdout, (TX / 'stderr.log').open('x', encoding='utf-8') as stderr:
        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, cwd=ROOT)
        atomic_json(TX / 'outer-launch.json', dict(pid=process.pid, command=cmd, started_at_utc=now()))
        print(json.dumps({'event': 'gpu-preflight-launch', 'outer_pid': process.pid, 'output': relative(RUN)}), flush=True)
        last = 0
        while process.poll() is None:
            if observation is None and (RUN / 'launch.json').exists():
                observation = json.loads((RUN / 'launch.json').read_text())
                observation.update(started_at_utc=now(), outer_pid=process.pid, command=cmd,
                                   output_root=relative(RUN), production_authorized=False)
                atomic_json(TX / 'active.json', observation)
                transition('active', TX / 'active.json', gpu_pid=observation['pid'],
                           next_legal_action='monitor this short test only')
            elapsed = time.perf_counter() - start
            if elapsed - last >= 30:
                checkpoints = list(RUN.glob('resume/**/result.json'))
                print(json.dumps(dict(event='gpu-preflight-alive', elapsed_seconds=round(elapsed, 1),
                    gpu_pid=observation['pid'] if observation else None, committed_checkpoints=len(checkpoints))), flush=True)
                last = elapsed
            time.sleep(1)
        code = process.wait()
    result = None
    if (RUN / 'summary.json').exists():
        result = json.loads((RUN / 'summary.json').read_text())
    error = None
    if (RUN / 'termination.json').exists():
        error = json.loads((RUN / 'termination.json').read_text())
    kind = 'completed-technical-preflight-pass' if code == 0 and result and result['status'] == 'pass' else 'technical-preflight-failure'
    proof = dict(status=kind, exit_code=code, finished_at_utc=now(),
        wall_seconds=time.perf_counter() - start, contract=manifest['contract'], result=result, error=error,
        accepted_ledger_effect=0, production_launched=False, cpu_control_unchanged=True,
        logs={name: file_hash(TX / name) for name in ('stdout.log', 'stderr.log')},
        stderr_bytes=(TX / 'stderr.log').stat().st_size,
        artifacts=[dict(path=relative(p), sha256=file_hash(p)) for p in RUN.rglob('*') if p.is_file()])
    atomic_json(TX / 'terminal.json', proof)
    transition('terminal', TX / 'terminal.json', terminal_classification=kind,
               next_legal_action='report short-test results; production requires explicit authorization')
    print(json.dumps(proof), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
