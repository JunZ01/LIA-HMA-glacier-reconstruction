from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha

root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/surface_compact_forest_v09')
worker=root/'surface_compact_forest_worker.py'
config=root/'reconstruction_config.json'
assert sha(worker).lower()=='9cf74e8095c8d9cbcb5a8b145fcc960b14c622c4edcc2932498934768918943f'
assert sha(config).lower()=='2254b9d0f7c09a84bbaa1060228faf4e76926bb85be24640b338f54a74035144'
proposal=json.loads((root/'triple_operator_proposal.json').read_text(encoding='utf-8'))
contract=dict(proposal)
contract.update(status='representative-and-independent-preflight-authorized',output_root=str(root),
 implementation={'worker':str(worker),'sha256':sha(worker),'config':str(config),'config_sha256':sha(config)},
 fixture={'path':str(root/'surface_compact_forest_fixture_result.json'),'sha256':sha(root/'surface_compact_forest_fixture_result.json')},
 checkpoint='immutable per-tau JSON and NPZ; validated completed results reused; status writes nonfatal',
 runtime_estimate_minutes=[3,10],maximum_workers=2,
 stop_condition='Either representative or independent scientific failure closes this composition; no scalar expansion or post-result gate changes',
 successful_existing_model_outputs_replaced=False,authorized_uid=[1019319,1025361])
print(register('yx_surface_compact_forest_v09',contract,
 'YX surface compact forest: registered distinct three-operator preflight',
 'The user authorized full parallel completion. A passed two-outlet synthetic fixture supports a new disclosed composition: individual modern outlets, actual-raster compatibility only where the 100 m rule leaves no original PP anchors, then the existing minimum-surface-gradient bed-obstacle extension. This is distinct from the closed H-harmonic compact-forest family. Freeze YX-19319 representative and YX-25361 independent, original raw bed and fixed 40–260 kPa brackets, original anchors and true exterior zero. Both must close modern volume within 0.5%, pass all array/anchor gates, and strictly improve same-tau modern DEM fidelity against the required H initializer; 19319 must additionally improve its saved H-only tau260 control. No successful completed model or formal accepted ledger is replaced.',
 True,[1019319,1025361],'Observe only the two registered preflight members; independently verify both before any class promotion.'))
