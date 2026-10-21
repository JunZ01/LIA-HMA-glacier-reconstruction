"""Passive sampling and immutable input/checkpoint analysis; no Torch import."""

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import psutil

from lia_gpu_four_group_search_v2 import ROOT, load_search, file_hash
from lia_gpu_matrix_io_v2 import Store, atomic_json
from prepare_lia_gpu_four_group_v2 import OUTPUT


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()), mean=float(a.mean()))


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = ROOT / 'validation_data/results/paleo_surface_validation_v0' / ('lia_gpu_cost_readonly_' + stamp)
    out.mkdir()
    shared_before = file_hash(ROOT / 'LIA_CURRENT_STATE.json')
    manifest = json.loads((OUTPUT / 'manifest.json').read_text())
    before = json.loads((ROOT / 'wg2g1/status.json').read_text())
    process = psutil.Process(before['pid'])
    samples = []
    for i in range(8):
        cpu = process.cpu_times()
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        samples.append(dict(monotonic=time.perf_counter(), cpu_seconds=cpu.user+cpu.system,
            gpu=[float(x.strip()) for x in result.stdout.strip().split(',')]))
        if i != 7:
            time.sleep(2)
    with np.load(OUTPUT / 'inputs.npz', allow_pickle=False) as blob:
        data = {k:blob[k] for k in ('system_nx','system_dx_meter','system_thick','system_bed_h','system_valid_mask')}
    grids = []
    index = {r['system']:i for i,r in enumerate(manifest['systems'])}
    nx_map = {r['system']:r['nx'] for r in manifest['systems']}
    for system, i in index.items():
        n, dx = int(data['system_nx'][i]), float(data['system_dx_meter'][i])
        h = data['system_thick'][i,:n]
        ice = np.flatnonzero(h > 0)
        last = int(ice[-1]) if len(ice) else -1
        grids.append(dict(system=system,nx=n,dx_m=dx,occupied_nodes=len(ice),
            empty_fraction=float(np.mean(h==0)),last_ice_index=last,
            trailing_empty_nodes=n-1-last,free_tail_km=(n-1-last)*dx/1000,
            full_domain_km=n*dx/1000,valid_nodes=int(data['system_valid_mask'][i,:n].sum()),
            tail_bed_min=float(data['system_bed_h'][i,last+1:n].min()) if last+1<n else None))
    search = load_search(manifest['contract'])
    first = search.pack(search.candidates(1,[]),nx_map)[0]
    width = max(nx_map[c.system] for c in first)
    store = Store(ROOT / 'wg2g1')
    observed = []
    for candidate in first:
        key = search.spin_key(candidate)
        result = store.read('spin',key)
        if result is not None:
            arrays = store.arrays('spin',key)
            year, kind = result['year'], result['status']
        else:
            cp = store.latest('spin',key)
            if cp is None:
                continue
            meta, arrays = cp
            year, kind = meta['year'],'active-checkpoint'
        h = arrays['thick']
        ice = np.flatnonzero(h>0)
        observed.append(dict(system=candidate.system,delta=candidate.delta,year=year,kind=kind,
            nx=len(h),occupied_nodes=len(ice),empty_fraction=float(np.mean(h==0)),
            trailing_empty_nodes=len(h)-1-int(ice[-1]) if len(ice) else len(h)))
    after = json.loads((ROOT / 'wg2g1/status.json').read_text())
    seconds = samples[-1]['monotonic']-samples[0]['monotonic']
    cpu_core_fraction = (samples[-1]['cpu_seconds']-samples[0]['cpu_seconds'])/seconds
    gpu = np.asarray([s['gpu'] for s in samples])
    levels = math.ceil(math.log2(width))
    result = dict(mode='read-only-no-model-launch-no-profiler-attachment', created_at_utc=stamp,
        contract=manifest['contract'], source_manifest_sha256=file_hash(OUTPUT/'manifest.json'),
        observation_nonatomic=True, status_before=before,status_after=after,
        sample_seconds=seconds, gpu_worker_cpu_core_equivalents=cpu_core_fraction,
        gpu_sample_mean=dict(zip(('gpu_busy_pct','memory_busy_pct','memory_MiB','power_W','sm_MHz','memory_MHz'),gpu.mean(axis=0).tolist())),
        samples=samples, initial_grid_stats={key:stats([g[key] for g in grids]) for key in
            ('nx','dx_m','occupied_nodes','empty_fraction','free_tail_km','full_domain_km')},
        initial_grids=grids, first_batch_snapshot=observed,
        first_batch_stats={key:stats([g[key] for g in observed]) for key in ('occupied_nodes','empty_fraction','trailing_empty_nodes')},
        static_solver=dict(first_batch_width=width,pcr_rounds_per_step=levels,
            four_float_arrays_two_neighbor_gathers_per_round=8,
            float_neighbor_gather_expressions_per_step=8*levels,
            logical_float_gather_output_bytes_per_step_at_current_lanes=8*levels*width*after['batch_active']*8,
            interpretation='Logical dense tensor outputs only, NOT measured DRAM bytes, kernel count or wall-time share.'),
        step_counts=dict(per_year=after['accepted_steps']/after['model_year'],rejected=after['rejected_steps']),
        shared_state_unchanged=(file_hash(ROOT/'LIA_CURRENT_STATE.json')==shared_before),
        constraints=['no scientific parameters or domain modified','no GPU workload launched','no CPU mainline process controlled'],
        unresolved=['per-kernel wall-time percentages','FP64 versus bandwidth bottleneck','safe effective-domain reduction','equal-workload batch-size optimum'])
    atomic_json(out/'analysis.json',result)
    lines = ['# GPU Cost: Passive Analysis', '',
        'No simulation, profiler attachment, process control or shared-state change was performed.', '',
        '## Observations',
        '- Status and checkpoint observations are not one atomic snapshot.',
        '- Mean GPU busy percent: %.1f; GPU-worker CPU core equivalents: %.2f.' % (gpu[:,0].mean(),cpu_core_fraction),
        '- Initial grid median: %.0f nodes; median occupied: %.0f; median zero-thickness share: %.2f%%.' %
            (result['initial_grid_stats']['nx']['median'],result['initial_grid_stats']['occupied_nodes']['median'],100*result['initial_grid_stats']['empty_fraction']['median']),
        '- Median initial free tail: %.2f km.' % result['initial_grid_stats']['free_tail_km']['median'],
        '- First-batch committed-state median zero-thickness share: %.2f%%.' % (100*result['first_batch_stats']['empty_fraction']['median']),
        '- Dense PCR width %d requires %d rounds and %d floating neighbor-gather expressions each step.' % (width,levels,8*levels),
        '- These are structural counts, not measured CUDA kernel counts or wall-time percentages.', '',
        '## Priority',
        '1. Attribute dense full-domain solver work with a later isolated kernel benchmark on saved states; preserve current production.',
        '2. Compare a vendor batched tridiagonal solver with existing PCR on identical matrices, float64 and the same GPU stream.',
        '3. Precompute invariant indices, masks, widths and bed terms; fuse repeated pointwise operations.',
        '4. Investigate an exact inactive-domain optimization only with boundary and positive-MB safeguards. Zero thickness alone does not imply removable domain.',
        '5. Compare 128/256/512 on identical members before changing scheduling. No continuous refill is implemented today.', '',
        '## Not Established',
        'No wall-time breakdown, CPU speedup ratio, optimal batch size, or promised acceleration factor is established.',
        'Do not truncate the scientific domain, relax steps, reduce precision or alter the live run on the basis of this report.', '',
        '## References',
        '- NVIDIA utilization definition: https://docs.nvidia.com/deploy/nvidia-smi/index.html',
        '- NVIDIA batched tridiagonal API: https://docs.nvidia.com/cuda/cusparse/index.html', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('samples','initial_grids','first_batch_snapshot')},indent=2))
    print(str(out))


if __name__=='__main__':
    main()
