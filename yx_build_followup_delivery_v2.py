"""Create a new immutable delivery; preserve the sealed 642-system selection."""
from pathlib import Path
import json,os,hashlib,shutil
import pandas as pd
import yx_build_combined_delivery_v1 as build

BASE=Path('H:/ToYangxin/result_yx')
FOLLOWUP=BASE/'completion_followup_20261009_v1'
SEALED=BASE/'completed_reconstruction_20261009_1791516946999696200'

def main():
 selection=json.loads((FOLLOWUP/'qualified_supplements.json').read_text(encoding='utf-8'))
 supplemental=selection['records']
 old=pd.read_csv(SEALED/'paired_volume_changes.csv').set_index('record_id')
 assert len(old)==642 and not set(old.index).intersection(supplemental)
 roots={int(v['selected_model_version']):Path(v['family_root']) for v in supplemental.values()}
 def source(version,rid):
  return (roots[version] if version in roots else BASE/f'surface_reconstruction_20261009_v{version}')/'members'/rid
 original_choose=build.choose
 def choose(row,affected):
  rid=row.record_id
  if rid in old.index:
   v=old.loc[rid]
   return int(v.selected_model_version),Path(v.selected_result_path).parent,[]
  if rid in supplemental:
   s=supplemental[rid]
   p=Path(s['family_root'])/'members'/rid/'result.json'
   assert hashlib.sha256(p.read_bytes()).hexdigest().lower()==s['result_sha256'].lower()
   return int(s['selected_model_version']),p.parent,[]
  return original_choose(row,affected)
 build.source=source;build.choose=choose
 build.FAMILIES+=list(roots)
 # Keep the public pointer at its sealed prior version until final checks finish.
 build.BASE=FOLLOWUP
 build.main()
 out=Path(os.environ['YX_DELIVERY_ROOT'])
 actual=pd.read_csv(out/'paired_volume_changes.csv').set_index('record_id')
 for col in ['lia_volume_km3','modern_volume_km3','tau_kpa']:
  assert (actual.loc[old.index,col]==old[col]).all(),f'Sealed successful selection changed: {col}'
 assert len(actual)==642+len(supplemental)
 for name in ['raw_F19_source_hash_manifest.csv','original_input_selection_summary.json','input_source_provenance.json']:
  if (SEALED/name).exists():shutil.copy2(SEALED/name,out/name)
 shutil.copy2(FOLLOWUP/'qualified_supplements.json',out/'new_model_supplements_provenance.json')
 (out/'prior_sealed_delivery_provenance.json').write_text(json.dumps({'path':str(SEALED),'summary_sha256':hashlib.sha256((SEALED/'summary.json').read_bytes()).hexdigest(),'old_successful_models':642,'old_successful_models_recomputed':False,'old_successful_rows_changed':False,'new_qualified_supplements':len(supplemental)},indent=2),encoding='utf-8')

if __name__=='__main__':main()
