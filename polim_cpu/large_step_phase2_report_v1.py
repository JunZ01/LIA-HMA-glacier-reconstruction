"""Independent original-CPU endpoint checks and frozen full-domain comparisons."""
from .large_step_phase2_common_v1 import *


def case_report(root):
    cfg=st.read(root/'config.json');contract=st.read(root/'contract.json')['contract'];z=st.arrays(SOURCE/'input.npz')
    area=z['geo_dual_area'];initial,initial_meta,_=reference(cfg['start']);v0=float(area@initial['h'])
    segments=[st.read(p) for p in sorted(root.glob('result-*.json'))]
    attempts=[st.read(p) for p in sorted(root.glob('attempt-*.json'))]
    phase1=st.read(E/'phase1-audit.json') if cfg['adopt_phase1'] else None
    endpoints=[];masks=st.arrays(RUN/f"masks-{cfg['start']}.npz")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for year in range(cfg['start']+1,cfg['end']+1):
        checkpoint=root/f'frame-{year:06d}'
        if not checkpoint.exists():continue
        candidate,meta=st.restore(checkpoint,contract);ref,refmeta,_=reference(year)
        h,u=candidate['h'],candidate['u'];dh=h-ref['h']
        surf=u.reshape(-1,2,2)[:,-1,:];rsurf=ref['u'].reshape(-1,2,2)[:,-1,:];du=surf-rsurf
        bp,wet=cpu_bp(h,u,z)
        vref=float(area@ref['h']);exportref=refmeta['exported']-initial_meta['exported']
        limit=max(abs(exportref)*GATES['export_fraction'],v0*GATES['export_initial_fraction'])
        values=dict(year=year,volume_relative=abs(float(area@h)-vref)/vref,
            h_rms=float(np.sqrt(np.sum(area*dh**2)/area.sum())),h_max=float(np.abs(dh).max()),
            velocity_relative=float(np.sqrt(np.sum(area[:,None]*du**2)/max(np.sum(area[:,None]*rsurf**2),1e-300))),
            export_difference=abs(meta['exported']-exportref),export_limit=limit,export_reference=exportref,
            original_CPU_BP=bp,source=meta['source'],potential=meta['potential'],unmet=meta['unmet'],exported=meta['exported'])
        passed=dict(volume=values['volume_relative']<=GATES['volume'],h_rms=values['h_rms']<=GATES['h_rms'],
            h_max=values['h_max']<=GATES['h_max'],velocity=values['velocity_relative']<=GATES['velocity'],
            export=values['export_difference']<=limit,bp=bp<=GATES['bp'])
        values['gates']=passed;values['passed']=all(passed.values());endpoints.append(values)
        with (root/f'differences-{year}.npz').open('xb') as f:
            np.savez(f,xy=z['xy'],h_difference=dh,surface_velocity_difference=du,**masks)
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        xy=z['xy']/1000.;xy=xy-xy.min(axis=0)
        dmag=np.sqrt(np.sum(surf**2,axis=1))-np.sqrt(np.sum(rsurf**2,axis=1))
        for j,key in enumerate(('ice_edge','junction','original_fast')):
            mask=masks[key]
            for i,(field,label) in enumerate(((dh,'H difference (m)'),(dmag,'Surface speed difference (m/yr)'))):
                ax=axes[i,j];ax.scatter(xy[:,0],xy[:,1],s=1,c='#dddddd')
                bound=max(float(np.abs(field).max()),1e-12)
                sc=ax.scatter(xy[mask,0],xy[mask,1],c=field[mask],s=3,cmap='RdBu_r',vmin=-bound,vmax=bound)
                fig.colorbar(sc,ax=ax,label=label);ax.set_aspect('equal');ax.set_title(key);ax.set_xlabel('x (km)');ax.set_ylabel('y (km)')
        fig.suptitle(f"{cfg['case_id']} | endpoint {year} | candidate minus retained v32")
        fig.savefig(root/f'differences-{year}.png',dpi=150);plt.close(fig)
    oldrows=st.read(PROJECT/'validation_data/results/paleo_surface_validation_v0/polim_v32r4_historical_1470_1570_v1/completion.json')['rows']
    baseline=[x for x in oldrows if cfg['start']<x['year']<=cfg['end']]
    baseline_seconds=sum(x['seconds'] for x in baseline)
    prefix_seconds=st.read(P1/'completion.json')['wall_seconds'] if phase1 else 0.
    launches=[st.read(p) for p in root.glob('process-*.json')]
    total=prefix_seconds+sum(x['elapsed_seconds'] for x in launches)
    completed=len(endpoints)==2 and all(s['status']=='completed' for s in segments) and bool(segments)
    accuracy=completed and all(x['passed'] for x in endpoints)
    history=[r for a in attempts for r in a['history']]
    p1hist=st.restore(P1/'frame-000001',st.read(P1/'contract.json')['contract'])[1]['history'] if phase1 else []
    distribution={}
    for a in attempts:
        if a['status']=='accepted':distribution[str(a['dt'])]=distribution.get(str(a['dt']),0)+1
    if phase1:distribution['0.25']=distribution.get('0.25',0)+1
    counts=dict(bp_calls=sum(a['bp_calls'] for a in attempts)+(6 if phase1 else 0),
        outer_iterations=len(history)+len(p1hist),newton=sum(x.get('newton',0) for x in history)+sum(x['bp_newton'] for x in p1hist),
        cg=sum(x.get('cg',0) for x in history)+sum(x['bp_cg'] for x in p1hist),
        graph_setup_cg=sum(x.get('graph_setup_cg',0) for x in history),
        accepted_steps=sum(a['status']=='accepted' for a in attempts)+bool(phase1),rejected_steps=sum(a['status']=='rejected' for a in attempts))
    timing=dict(total_including_launch_restore_and_prefix=total,adopted_prefix_unsplit=prefix_seconds,
        initialization=sum(s['initialization_seconds'] for s in segments),restore=sum(s['restore_seconds'] for s in segments),
        io=sum(s['io_seconds'] for s in segments),prediction=sum(a['timing']['predict'] for a in attempts),
        transport=sum(a['timing']['transport'] for a in attempts),bp=sum(a['timing']['bp'] for a in attempts),
        build=sum(a['timing']['build'] for a in attempts),rejected_attempts=sum(a['timing']['total'] for a in attempts if a['status']=='rejected'),
        measured_worker_wall=sum(s['wall_seconds'] for s in segments),v32_step_seconds=baseline_seconds,
        historical_v32_ratio=baseline_seconds/total if total else None,
        note='Total includes process import/exit, all failures, helper startup, restore, IO and retained phase1 wall cost. Restore and initial IO are subsets of initialization; rejected attempts are subsets of component timings. Phase1 component split unavailable. v32 row seconds omit final accepted checkpoint IO; historical unpaired timing only. No same-window v35 control.')
    result=dict(case_id=cfg['case_id'],start=cfg['start'],end=cfg['end'],max_dt=cfg['max_dt'],
        status='completed' if completed else 'failed',accuracy_passed=bool(accuracy),endpoints=endpoints,
        failures=[dict(segment=s['segment'],error=s.get('error',s.get('close_error'))) for s in segments if s['status']!='completed'],
        failed_accuracy_gates={str(x['year']):[k for k,v in x['gates'].items() if not v] for x in endpoints if not x['passed']},
        steps=distribution,counts=counts,timing=timing,performance_2x_v32_proxy=bool(accuracy and baseline_seconds/total>=2.),
        performance_2x_v35='not-evaluated-no-matched-control',baseline_steps=len(baseline),
        baseline_BP_calls=len(baseline),stage3_authorized=False)
    st.write(root/'completion.json',result)
    return result


def summarize(results):
    candidates=[]
    for dt in (.25,.5,1.):
        cases=[r for r in results if r['max_dt']==dt]
        if len(cases)==2 and all(r['accuracy_passed'] for r in cases):
            candidates.append((sum(r['timing']['total_including_launch_restore_and_prefix'] for r in cases),dt))
    recommendation=min(candidates)[1] if candidates else None
    out=dict(status='completed',cases=results,recommended_max_dt= recommendation,
        selection='Minimum measured total across both windows among caps passing every frozen annual accuracy gate.',
        v35_performance_target='not-evaluated; no same-window v35 control requested',
        no_baseline_replay=True,phase3_authorized=False,launch_authorized=False,accepted_count_effect=0)
    st.write(E/'completion.json',out)
    lines=['# BP large-step phase 2: six frozen comparisons','',
        '| Window | Cap (yr) | Completion | Accuracy | Steps | Rejects | BP calls | Total s | v32 historical ratio |',
        '|---|---:|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        c=r['counts'];t=r['timing'];ratio=t['historical_v32_ratio']
        lines.append(f"| {r['start']}–{r['end']} | {r['max_dt']} | {r['status']} | {r['accuracy_passed']} | {c['accepted_steps']} | {c['rejected_steps']} | {c['bp_calls']} | {t['total_including_launch_restore_and_prefix']:.3f} | {ratio:.2f} |")
    lines+=['',f'Recommended cap: {recommendation if recommendation is not None else "none: no candidate passed both windows"}.','',
        'Timing includes rejected attempts, helpers, process initialization, checkpoint IO and resume. The adopted phase-1 quarter retains its recorded total cost without inventing component times. v32 uses saved per-step seconds from the same windows, is not a paired hardware benchmark, and excludes final checkpoint IO. No matched v35 timing exists; its 2× development target is not established.','',
        '| Case/year | Volume % | H RMS m | H max m | Surface velocity % | Export difference / limit m3 | CPU BP |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        for e in r['endpoints']:
            lines.append(f"| {r['case_id']}/{e['year']} | {100*e['volume_relative']:.5f} | {e['h_rms']:.5f} | {e['h_max']:.5f} | {100*e['velocity_relative']:.5f} | {e['export_difference']:.3f} / {e['export_limit']:.3f} | {e['original_CPU_BP']:.3g} |")
    for r in results:
        lines+=['',f"## {r['case_id']}",'',f"Step distribution: `{r['steps']}`. Counts: `{r['counts']}`.",
            f"Failures: `{r['failures']}`. Accuracy failures: `{r['failed_accuracy_gates']}`.",
            f"Components and ledgers: [completion](../../../../polim_cpu/runs/v38-large-step-phase2-v1/{r['case_id']}/completion.json)."]
    (E/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return out
