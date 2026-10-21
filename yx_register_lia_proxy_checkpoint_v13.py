from pathlib import Path
import json,hashlib
from yx_followup_transaction_v1 import register,sha
out=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit/lia_directed_proxy_preflight_v13')
p=json.loads((out/'proposal.json').read_text(encoding='utf-8'))
assert sha(out/'frozen_dependency_hashes.json').lower()==p['dependency_manifest_SHA256']
assert sha(out/'reconstruction_config.json').lower()==p['configuration_SHA256']
assert sha(p['worker_path']).lower()==p['worker_SHA256']
dependencies=json.loads((out/'frozen_dependency_hashes.json').read_text(encoding='utf-8'))
for path,h in dependencies.items():
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
 assert d.hexdigest().lower()==h.lower(),path
f=json.loads(Path(p['modern_checkpoint_reuse_fixture']).read_text(encoding='utf-8'))
assert f['status']=='passed' and f['new_scientific_model_solve_count']==0
assert len(f['real_modern_checkpoints_reused_without_any_solver'])==2
assert all(x['modern_solver_guard_blocks_entry'] and x['PP_2D_scientific_solver_calls']==0 and x['modern_checkpoint_H_copied_bit_exact'] and x['replay_no_immutable_copy_changes'] for x in f['real_modern_checkpoints_reused_without_any_solver'])
for record,g in p['geometry_pair'].items():
 assert sha(g['source_gate_evidence']).lower()==g['source_gate_evidence_SHA256']
 assert all(x['raised_head_count']==0 and x['head_outlet_role_conflicts']==0 and x['max_route_proxy_fill_m']<=188.8220631745703 for x in g['components'])
contract=dict(p)
contract.update(status='genuine-LIA-routing-pair-preflight-authorized-modern-checkpoint-only',implementation_hashes=dependencies,proposal_sha256=sha(out/'proposal.json'),final_full_field_H2000_m_gate_independently_enforced_before_qualification=True)
print(register('yx_lia_directed_proxy_checkpoint_v13',contract,
 'YX LIA candidate exterior routing alone: exact saved modern checkpoint pair preflight',
 'Following terminalv12 original-LIA-networkH2000 failure, register a genuine LIA-routing-topology change only. Reuse exactly verified modern calibration and fullfield checkpoints at04312tau71.97605172/08089tau75.82454603; zero modern PP/2D/recalibration entries are protected by fixture/guard. Preserve physical original DEM/hybrid nativebed-plus-DEM fallback, warp, F0.8, refinement16,40-260kPa bounds, exactPP, original anchors, empty-only compatibility and existing S obstacle. Frozen candidate graph04312 has82 original rasterheads/25 exterior roots/139 branches/57 exact junctions/fill39.584m;08089 has38/29/46/9/fill155.497m, both below unchanged188.822m empirical cap with no headraise. Original/independent DEM full LIA masks and outside-ring finite coverage pass; hybrid-reference finitebed is not native-product coverage. Candidate LIA routing uses modern terrain/ice DEM as proxy, not observed palaeo-surface/flow or validated palaeo-outlet. Source-raster definition resolution and independent120head reproduction retained; two partial cells and04312 unsupported1.521km2 modern member, supported-volume calibration and narrow-modern-component local fidelity worsening are disclosed.115 dependencies and real modern checkpoint-copy fixture frozen. Authorize only the two LIA endpoints at savedtau in two workers/16GiB,1-10min estimated/hard30min. Any original network, fullfieldH2000, anchor, boundary, finite/nonnegative, component or S-convergence failure closes composition; no retry or gate changes. Independent savedarray review is mandatory before any qualification/promotion. Original650 and independent8089 delivery remain unchanged; formal191 unaffected.',
 True,[1004312,1008089],'Run the frozen LIA-only pair using immutable modern checkpoints; independently verify full saved arrays before qualifying any addition.',phase='preflight'))
