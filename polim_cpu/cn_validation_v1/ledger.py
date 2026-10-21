"""Dual window/year bookkeeping only; no modified CN equation."""
import copy
import numpy as np
from polim_cpu.geometry_window_v1.ledger import preview,checkpoint,restore
from polim_cpu.time_refinement_v1.diagnostics import zero_parts

KEYS=['actual_nodes','potential_nodes','unmet_nodes','internal_volume','boundary_volume']
def zero_ledger(volume):return dict(V0=volume,volume=volume,source=0.,potential=0.,unmet=0.,exported=0.,cumulative_mass_error=0.,cumulative_mass_relative=0.)
def dual_preview(ledger,volume,source,exported):
    main=preview(ledger,volume,source,exported)
    if '_increment' not in ledger:return main
    other=preview(ledger['_increment'],volume,source,exported)
    return dict(**{k:v for k,v in main.items() if k!='cumulative_mass_relative'},
        cumulative_mass_relative=max(main['cumulative_mass_relative'],other['cumulative_mass_relative']),
        window_mass_relative=main['cumulative_mass_relative'],increment_mass_relative=other['cumulative_mass_relative'],
        increment_mass_error=other['cumulative_mass_error'])
def commit_one(ledger,parts,old_parts,area):
    source=float(parts['actual_nodes'].sum());export=float(parts['boundary_volume'].sum());volume=float(area@parts['H'])
    d=preview(ledger,volume,source,export)
    if d['cumulative_mass_relative']>1e-10:raise RuntimeError('Cumulative ledger rejected before commit')
    n=copy.deepcopy(ledger);n.update(d);n['volume']=volume
    for k,v in [('source',source),('potential',float(parts['potential_nodes'].sum())),('unmet',float(parts['unmet'].sum())),('exported',export)]:n[k]+=v
    total={k:old_parts[k]+parts['unmet' if k=='unmet_nodes' else k] for k in KEYS}
    return n,total
