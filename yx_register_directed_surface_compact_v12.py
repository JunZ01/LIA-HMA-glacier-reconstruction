from pathlib import Path
import json,hashlib
from yx_followup_transaction_v1 import register,sha
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit')
out=root/'directed_surface_compact_preflight_v12'
proposal=json.loads((out/'proposal.json').read_text(encoding='utf-8'))
manifest=out/'frozen_dependency_hashes.json'
assert sha(manifest).lower()==proposal['dependency_manifest_sha256']
assert sha(out/'reconstruction_config.json').lower()==proposal['config_sha256']
assert sha(proposal['worker_path']).lower()==proposal['frozen_implementation_SHA256']
hashes=json.loads(manifest.read_text(encoding='utf-8'))
for p,h in hashes.items():
 d=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
 assert d.hexdigest().lower()==h.lower(),p
fixture=json.loads(Path(proposal['fixture']).read_text(encoding='utf-8'))
assert fixture['status']=='passed'
contract=dict(proposal)
contract.update(status='representative-and-independent-new-composition-preflight-authorized',authorized_uid=[1004312,1008089],implementation_hashes=hashes,independent_control8089_existing_delivery_not_replaced=True)
print(register('yx_directed_surface_compact_exact_v12',contract,
 'YX directed candidate outlet graph with exact PP, empty-anchor compatibility and S obstacle: frozen two-case composition preflight',
 'The preceding H-only graph composition is terminally closed:04312 secondary short graph loses all metric-filter anchors and08089 modern field fails its frozen independent saved-control fidelity gate. This new disclosed composition retains its candidate geometry, initialized head roles,188.822m empirical route-only fill cap, original DEM and native beds, exact continuous PP,40-260kPa and F0.8. Add only empty-set actual-raster interior PP-anchor compatibility plus the existing minimum-S-gradient bed obstacle. No head restoration or exact-warp numerical correction. Mandatory same-tau H initializer arrays and actual PP constraints are saved within each S evaluation; require full-component-pixel S RMSE strictly lower than this initializer and frozen94.411/400.230m source gates, with08089 also strictly below128.150349m. Native modern closure0.5%, all components retained, H<=2000m, real exteriorH0 and anchors1e-8 precede any LIA work. One/two-outlet, two-interior-anchor and immutable-cache replay fixtures passed; freeze70 dependencies including8.68GB DEM and5 native source gdirs. Authorize only representative04312 and independent08089 for30min per-case/two-worker16GiB preflight. Scientific failures close this composition without retry; both outcomes and independent saved-array checks are required for promotion. Original650 delivery and accepted191 remain unchanged.',
 True,[1004312,1008089],'Run only the frozen v12 pair; retain terminal checkpoints and independently audit any successes before class promotion.',phase='preflight'))
