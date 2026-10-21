from pathlib import Path
import json,hashlib
from yx_followup_transaction_v1 import register,sha
out=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit/bounded_modern_checkpoint_v15')
p=json.loads((out/'proposal.json').read_text(encoding='utf-8'))
for path,key in [(out/'frozen_dependency_hashes.json','dependency_manifest_SHA256'),(out/'reconstruction_config.json','configuration_SHA256'),(Path(p['worker_path']),'worker_SHA256')]:assert sha(path).lower()==p[key]
deps=json.loads((out/'frozen_dependency_hashes.json').read_text(encoding='utf-8'))
for path,h in deps.items():
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
 assert d.hexdigest().lower()==h.lower(),path
f=json.loads(Path(p['technical_fixture']).read_text(encoding='utf-8'));assert f['status']=='passed'
assert p['modern_scalar_scans_allowed']==p['modern_PP_or_2D_solver_entries_allowed']==0
assert p['relative_native_supported_target_residual']<p['original_relative_tolerance']==.005
assert all(x['outside_original_tolerance_rejected'] for x in p['technical_proofs']['actual_three_outside_tolerance_cached_fields_rejected'])
contract=dict(p);contract.update(status='single-technical-cached-modern-first-LIA-continuation-authorized',implementation_hashes=deps)
print(register('yx_11977_cached_boundary_calibration_fix_v15',contract,'YX11977 one-time technical continuation: cached modern40, first LIA40','Following independently evidenced and canonical resolved tolerance-versus-signcross difference, freeze one technical continuation for11977 only. This is disclosed post-result numerical implementation correction, not a preregistered exact-root pass or bracket/parameter rescue. Select the already computed tau40 modern field with0.321657785percent native-supported residual and all original fullfield fidelity/H/anchor/boundary/source gates;0.5percent/40-260/F0.8/refinement16/bed/DEM/warp/topology/fieldoperators remain unchanged.318 dependencies frozen, realcached fixture forbids every modern PP/2D entry, repeated copy preservesSHA/mtime, and the actual three over-tolerance cases are rejected without solving. Authorize one first LIA40 endpoint at frozen candidate graph/exactPP/S/compact; modern field exported from unchangedcache only. One worker8GiB/1-5min/hard30min; any new LIA scientific gate fails terminally with no retry. Independent saved-array reproduction required before addition. Originalfailed11977 record,three over-tolerance route closures,653 verified models and formal191 unchanged.',True,[1011977],'Run only first LIA40 with immutable modern cache; verify every saved-field gate independently before qualifying11977.',phase='technical-continuation'))
