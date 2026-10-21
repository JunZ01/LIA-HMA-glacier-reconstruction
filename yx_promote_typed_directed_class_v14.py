from pathlib import Path
import json,hashlib
from yx_followup_transaction_v1 import register,sha
out=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit/remaining9_directed_class_v14')
p=json.loads((out/'typed_class_proposal.json').read_text(encoding='utf-8'))
assert sha(out/'frozen_dependency_hashes.json').lower()==p['dependency_manifest_SHA256']
assert sha(out/'reconstruction_config.json').lower()==p['configuration_SHA256']
assert sha(p['worker_path']).lower()==p['worker_SHA256']
deps=json.loads((out/'frozen_dependency_hashes.json').read_text(encoding='utf-8'))
for path,h in deps.items():
 d=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
 assert d.hexdigest().lower()==h.lower(),path
state=json.loads(Path('LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
assert state['research_branches']['yx_lia_directed_proxy_checkpoint_v13']['status']=='pair-independent-confirmed-terminal-one-new-qualified'
assert p['scope_uid']==[1003621,1005176,1010304,1011977,1014077,1021852]
for member in p['members']:
 s=json.loads((out/member['record_id']/'source_only_member_summary.json').read_text(encoding='utf-8'))
 assert s['eligible_for_typed_class_proposal'] and s['scientific_solver_calls']==0
 assert all(x['status']=='source-geometry-pass' for x in s['source_gates_two_endpoints'])
contract=dict(p);contract.update(status='confirmed-both-endpoint-composition-typed-six-production-authorized',implementation_hashes=deps,full_saved_field_H2000_gate_before_any_qualification=True)
print(register('yx_both_endpoint_directed_class_v14',contract,
 'YX confirmed dual-endpoint candidate graph composition: input-gated six-member typed production',
 'V12 modern representative04312 and independent8089 plus V13 unchangedmodern/newLIA pair have passed all frozen gates and independent reproduction. Promote that disclosed dual-endpoint graph/exactPP/S/empty-only-anchor composition only to six independently diagnosed topology cases03621/05176/10304/11977/14077/21852. All12 endpoint input gates and frozen prebuilt graphs pass without PP/2D work. Own modern independentDEM source bands are45.0585/56.2858/43.1627/64.8538/38.4371/47.8832m; no pair-band substitution. Preserve original physicalDEM/nativebed+fallback/defaultwarp, source integer/alltouched roles/crops without movement/raise, common empirical routecap188.822m, fixed40-260kPa/F0.8/refinement16/H2000/anchor1e-8/realexterior0/S100iter1e-5. Each modern native-supported closure0.5percent and strict fullweighted S-versus-required-same-tau-H/own-sourceband must pass before LIA. Source finitehybridB is not nativecoverage; no palaeo DEM/flow validation claimed.283dependencies and7nativegdirs frozen; reuse prior numerical and technical checkpoint/status fixtures without repeated pair preflight. Two workers maximum/16GiB/hard30min per case; completed new combination caches retained, any scientific member failure terminally closes its tested route without automatic retry/gate rescue; shared input/code hash conflicts stop before any solve. Unclassified03055/05103/28236 excluded from scientificqueue pending independent actualsource diagnostics and new route. Original651/control8089/formal191 unchanged until independent per-member savedarray qualification.',
 True,p['scope_uid'],'Run at most two typed members concurrently; independently verify terminal saved fields before additions. Untyped three remain read-only source diagnosis.',phase='typed-production'))
