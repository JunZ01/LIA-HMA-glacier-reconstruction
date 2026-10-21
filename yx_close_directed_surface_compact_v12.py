from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1')
out=root/'large_failure_audit/directed_surface_compact_preflight_v12'
f=out/'members/YX-04312/failure.json';failure=json.loads(f.read_text(encoding='utf-8'))
assert 'maximum-thickness-exceeded' in failure['error'] and 'LIA solve failed' in failure['error']
cal=json.loads((out/'members/YX-04312/calibration_farinotti2019.json').read_text(encoding='utf-8'))
fidelity=json.loads((out/'members/YX-04312/modern_full_field_fidelity_gate.json').read_text(encoding='utf-8'))
assert cal['status']=='passed' and fidelity['status']=='passed'
p=out/'members/YX-08089/result.json';control=json.loads(p.read_text(encoding='utf-8'));assert control['status']=='complete-primary'
a=root/'remaining_failure_audit/v12_independent_verification/YX-08089_independent_array_audit.json'
contract={'status':'scientific-preflight-LIA-route-failure-composition-closed','failure_path':str(f),'failure_sha256':sha(f),'04312_modern_calibration':cal,'04312_modern_fidelity':fidelity,'04312_LIA_old_network_failed':True,'independent8089_result':str(p),'independent8089_result_sha256':sha(p),'independent8089_array_audit':str(a),'independent8089_array_audit_sha256':sha(a),'qualified_new_systems':0,'old_successful8089_not_replaced':True,'class_promotion_authorized':False,'technical_retry_authorized':False,'all650_delivered_models_unchanged':True}
print(register('yx_directed_surface_compact_exact_v12',contract,
 'YX directed modern graph, exact PP and compact S: modern pair passes but original LIA network closes composition',
 'Both frozen modern endpoints pass full-field S-versus-mandatory-same-tau-H, source-band and native-volume gates.04312 calibrates71.97605172kPa, relative5.8423e-7 and S RMSE71.134054m versus H152.067511m within94.411032m source band. Independent08089 calibrates75.82454603kPa, relative2.8070e-7, S RMSE65.685425m versus H98.697648m and saved128.150349m; its original LIA route completes, savedNC/TIFF/anchors/exterior/sourcebed are independently verified with maxH1755.276m.04312 original LIA network244branches/81connectors fails fl73:seg3 above fixedH2000m before two-dimensional final extension. Close this composition with zero additions, no class promotion or rerun. Retain its successful modern fields and scalar calibration as immutable checkpoints, not formal LIA reconstruction. A genuine LIA-routing change can be separately frozen with its own input/geometry/independent-pair evidence and reuse modern checkpoints without recomputation.650 delivery and formal191 remain unchanged.',
 False,[],'Only read-only attribution or a genuinely new LIA-routing preflight may proceed; retain passed modern checkpoints without duplicate solve.',phase='scientific-closed'))
