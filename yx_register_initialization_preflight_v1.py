from pathlib import Path
import json
from yx_followup_transaction_v1 import register,sha

root=Path('H:/ToYangxin/result_yx/completion_followup_20261009_v1/raw_bed_initialization_v11')
hashes=json.loads((root/'implementation_hashes.json').read_text(encoding='utf-8'))
assert all(sha(p).lower()==h.lower() for p,h in hashes.items())
contract=json.loads((root/'preflight_proposal.json').read_text(encoding='utf-8'))
contract.update(status='representative-and-independent-preflight-authorized',implementation_hashes=hashes,output_root=str(root),authorized_uid=[1003621,1011977])
print(register('yx_raw_bed_initialization_v11',contract,
 'YX raw-bed flowline initialization: typed head-crop adapter preflight',
 'Read-only saved geometry shows that observed-surface positive-slope head filtering was applied to a raw bed during modern flowline initialization: the parent head was displaced 5.317 km in YX-03621 and 4.424 km in independent YX-11977, creating disconnected tributary targets. A synthetic fixture reproduced this head crop and connector mismatch using an identical bed and geometric tree. Register a disclosed modern-only initialization adapter: compute_centerlines and its raw-bed routing/filtering stay fixed; initialize_flowlines alone locally disables the surface-head-slope crop and restores cfg immediately. This differs from the closed observed-DEM routing substitution family. Freeze these two preflight cases, original PP/H-harmonic, raw bed, union exterior, 100 m anchor rule and tau40–260 kPa/0.5% closure. Require input-tree/hash provenance, passed in-mask network connectors and valid anchored/boundary endpoint arrays. Either scientific failure stops class promotion. LIA routing and initialization remain frozen original copies, accepted191 unaffected.',
 True,[1003621,1011977],'Observe and independently audit only the two registered initialization-adapter preflight cases.'))
