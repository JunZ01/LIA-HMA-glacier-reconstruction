from pathlib import Path
import json,html
import pandas as pd
BASE=Path('H:/ToYangxin/result_yx');out=Path(json.loads((BASE/'latest_completed_reconstruction.json').read_text(encoding='utf-8'))['path'])
ledger=pd.read_csv(out/'all_660_system_status.csv');categories={}
for row in ledger.loc[ledger.status.ne('reconstructed-QC-passed')].itertuples():
 rid=row.record_id;folder=BASE/f'surface_reconstruction_20261009_v{10 if row.modern_merged_reference_corrected else 5}'/'members'/rid;reason=str(row.reason)
 if rid=='YX-11101':
  q=json.loads((folder/'result.json').read_text(encoding='utf-8'));reason='Modern calibration passed; LIA PP network failed with negative thickness in branch fl9:seg0. '+json.dumps(q['bed_failures'],ensure_ascii=False)
  ledger.loc[ledger.record_id.eq(rid),'reason']=reason;category='小冰期分支出现负冰厚'
 elif 'fidelity gate not passed' in reason:category='现代冰面偏差未达到预先设定的改善要求'
 elif rid=='YX-05661':category='组合方案单例通过，独立确认未通过，未推广'
 elif rid in ['YX-11977','YX-28236']:category='流向连接修正后，现代冰量仍未闭合'
 elif rid=='YX-19319':category='小组件约束兼容后，现代冰量仍未闭合'
 else:
  ev=folder/'evaluations/modern_farinotti2019_160.000000.json';j=json.loads(ev.read_text(encoding='utf-8')) if ev.exists() else {};components=j.get('components',[])
  if any('maximum-thickness' in x.get('status','') for x in components):category='超出原模型最大冰厚约束'
  elif any(x.get('status')=='head-exclusion-review' for x in components):category='小组件没有可用内部冰厚约束'
  elif any('connector' in x.get('failure_reason','') for x in components):category='现代支流连接无法在冰川范围内连续'
  else:category='现代冰量未能在既定参数范围内闭合'
 categories[rid]=category
ledger['review_category_cn']=ledger.record_id.map(categories).fillna('');ledger.to_csv(out/'all_660_system_status.csv',index=False,encoding='utf-8-sig');review=ledger.loc[ledger.status.ne('reconstructed-QC-passed'),['record_id','mountain','review_category_cn','reason']];review.to_csv(out/'scientific_review_required.csv',index=False,encoding='utf-8-sig')
mapping=pd.read_csv(out/'all_662_source_record_mapping.csv');mapping['review_category_cn']=mapping.system_record_id.map(categories).fillna('');mapping.loc[mapping.system_record_id.eq('YX-11101'),'reason']=ledger.loc[ledger.record_id.eq('YX-11101'),'reason'].iloc[0];mapping.to_csv(out/'all_662_source_record_mapping.csv',index=False,encoding='utf-8-sig')
# The published GIS copy receives the same scientific reason; geometry and reconstructed fields do not change.
from osgeo import ogr
ds=ogr.Open(str(out/'reconstruction_results.gpkg'),1);assert ds is not None
for name in ['lia_systems_and_volume_changes','modern_system_endpoints']:
 layer=ds.GetLayerByName(name)
 if layer.GetLayerDefn().GetFieldIndex('review_category_cn')<0:layer.CreateField(ogr.FieldDefn('review_category_cn',ogr.OFTString))
 for rid,cat in categories.items():
  layer.SetAttributeFilter(f"record_id='{rid}'");feature=layer.GetNextFeature();assert feature is not None
  feature.SetField('review_category_cn',cat);feature.SetField('reason',ledger.loc[ledger.record_id.eq(rid),'reason'].iloc[0]);assert layer.SetFeature(feature)==0
 layer.SetAttributeFilter(None)
ds=None
page=(out/'report.html').read_text(encoding='utf-8');new=review[['record_id','mountain','review_category_cn']].copy();new['位置图']=[f'<a href="review_maps/{rid}.png">查看</a>' for rid in new.record_id]
if '<h2>复核分类与位置图</h2>' in page:
 start=page.index('<h2>复核分类与位置图</h2>');end=page.index('<h2>待复核原始诊断</h2>',start);page=page[:start]+'<h2>复核分类与位置图</h2>'+new.to_html(index=False,escape=False)+page[end:]
else:page=page.replace('<h2>待复核记录</h2>', '<h2>复核分类与位置图</h2>'+new.to_html(index=False,escape=False)+'<h2>待复核原始诊断</h2>')
page=page.replace('冰面、冰厚、床面栅格索引','冰面、冰厚、床面及冰厚变化栅格索引');(out/'report.html').write_text(page,encoding='utf-8')
print(review[['record_id','review_category_cn']].to_string(index=False))
