from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha
root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit')
out=root/'directed_local_outlet_preflight_v10'
proposal=json.loads((out/'proposal.json').read_text(encoding='utf-8'))
manifest=out/'frozen_dependency_hashes.json'
assert sha(manifest).lower()==proposal['dependency_manifest_sha256']
assert sha(out/'reconstruction_config.json').lower()==proposal['config_sha256']
hashes=json.loads(manifest.read_text(encoding='utf-8'))
assert all(sha(p).lower()==h.lower() for p,h in hashes.items())
fixture=json.loads((root/'directed_graph_fixture_result.json').read_text(encoding='utf-8'))
assert fixture['status']=='passed' and fixture['single_head_single_outlet_branch_not_lost']
contract=dict(proposal)
contract.update(status='representative-and-independent-preflight-authorized',authorized_uid=[1004312,1008089],implementation_hashes=hashes,
 numerical_projection_assumption='Retain original separate endpoint projections of fixed native beds; no exact-reprojection correction or source replacement. Resolved v1 audit documents extent-dependent numerical differences; derived common-S products must mask discrepancies.',
 independent_control8089_existing_delivery_not_replaced=True)
print(register('yx_directed_local_outlets_exact_v10',contract,
 'YX true exterior directed outlet graph plus exact PP: two-case registered preflight',
 'Resolved native-bed projection audit releases the input-source hold: original native bed hashes remain fixed; the numerical endpoint projection qualification is disclosed without silent model repair. Freeze a genuinely new modern directed true local exterior outlet graph composed with the independently verified continuous exact PP kernel and original H-harmonic union interior. The source heads retain location/elevation and are separated from candidate outlets by the frozen100m/grid-aware role rule. Only exterior outlets at least100m from other unincluded modern glaciers with downhill support in both original and independentDEM are allowed; no internal-divide zero-H, raised head or unresolved branch collapse. Fixed common routing-only empirical cap188.822m derives from the lesser pair source-compatibility band and is not physical confidence or proof of flow. Geometry and analytic one/two-outlet fixtures pass. Authorize only representative YX-04312 and independent YX-08089, original40–260kPa/shape0.8/H2000 and0.5% native-modern closure;04312 full-field modernRMSE<=94.411m and08089 strictly improves saved128.150349m control. Pass modern gates before original LIA routes plus exactkernel. Both pair outcomes required for promotion. Source arrays, previous650 delivered models and accepted191 remain unchanged. Two workers/16GiB, hard30min per member with immutable evaluations/checkpoints and no automatic retry.',
 True,[1004312,1008089],'Observe only the registered two-case topology preflight; verify modern and LIA gates before any class promotion.',phase='preflight'))
