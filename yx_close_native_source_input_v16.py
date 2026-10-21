from pathlib import Path
import json
from yx_followup_transaction_v1 import register, sha

root = Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/large_failure_audit/fixed_native_COP_pair_v16')
evidence = {
    'proposal_and_terminal_input_decision.json': 'e669c8162127b4e956edec82ec86ac1eff0f8dc5cfa83454fa00179d92a3fa33',
    'frozen_input_dependencies.json': 'ec90f7af4de7074fcaa19c7e704d3794a3a59115fd30d88f57a186ee0f7ee2f5',
    'frozen_output_manifest.json': '91ed0f085dc60d5b894a4b9f725bea7c20068139b12be5fdc09481f4abd2789f',
}
for name, expected in evidence.items():
    assert sha(root / name).lower() == expected, name
decision = json.loads((root / 'proposal_and_terminal_input_decision.json').read_text(encoding='utf-8'))
assert decision['status'] == 'failed-input-gate-sourcehold'
assert not decision['model_worker_frozen_or_launched']
assert not decision['new_scientific_launch_authorized']
manifest = json.loads((root / 'frozen_output_manifest.json').read_text(encoding='utf-8'))
print('input-only sealed output manifest schema', list(manifest))
print(register(
    'yx_28236_dem_source_correction_v16',
    {'status': 'terminal-input-gate-failed-sourcehold-no-model-launched',
     'input_only_evidence_root': str(root), 'evidence_sha256': evidence,
     'decision': decision, 'verified_reconstruction_systems_preserved': 654,
     'control_original_model_preserved': True, 'all660_reconstructed': False},
    'YX28236 controlled native source correction: original-head input gate failed, no scientific model launched',
    'Controlled primary COP90 derivative and independent same-tile control28235 preserve original native F19 thickness and supported-volume targets through B_new=B_old+(COP-T_old); original native sources and qualified control remain unchanged. The bounded input-only pair fails the preregistered original-source-head no-raise role gate:28236 modern original flowline11 integer head[349,499] requires17.53838850m routing-proxy elevation, and independent control28235 LIA flowline1 head[465,65] requires15.13908788m. These failures occur before PP or2D evaluation despite finite source coverage and interior fill below frozen188.82206317m cap. No scientific worker was frozen or launched. No head shift/raise, new fidelity band, parameter variant or qualification follows. This source-correction mechanism is terminal input-failed/sourcehold; all654 qualified models retained. Any future source-head/shared-divide operator mechanism needs separately resolved evidence and a new disclosed contract, not continuation of this failed pair. Static research counts do not affect formal191 or unrelated climate controller.',
    False, [],
    'Deliver independently verified654 and explicit6 unestimated systems; retain28236 source/head-role hold. No further scientific launch for this failed input family.',
    phase='input-scientific-closed'))
