from pathlib import Path
import json,html,argparse,shutil
import pandas as pd
from osgeo import ogr

def main(out):
 base=Path('H:/ToYangxin/result_yx');follow=base/'completion_followup_20261009_v1'
 ledger=pd.read_csv(out/'all_660_system_status.csv')
 categories={}
 for r in ledger.loc[ledger.status.ne('reconstructed-QC-passed')].itertuples():
  if r.record_id in ['YX-03055','YX-05103']:cat='现代冰面拟合未达到预设改善要求'
  elif r.record_id in ['YX-03621','YX-11977']:cat='支流连接已修正，现代冰量仍无法闭合'
  elif r.record_id=='YX-04312':cat='跨山脊路径与原最大冰厚约束不兼容'
  else:cat='现代冰量无法在既定应力范围内闭合'
  categories[r.record_id]=cat
  p=follow/'raw_bed_initialization_v11/members'/r.record_id/'failure.json'
  if p.exists():ledger.loc[ledger.record_id.eq(r.record_id),'reason']=json.loads(p.read_text(encoding='utf-8'))['error']
 ledger['review_category_cn']=ledger.record_id.map(categories).fillna('')
 ledger.to_csv(out/'all_660_system_status.csv',index=False,encoding='utf-8-sig')
 review=ledger.loc[ledger.status.ne('reconstructed-QC-passed'),['record_id','mountain','review_category_cn','reason']]
 review.to_csv(out/'scientific_review_required.csv',index=False,encoding='utf-8-sig')
 mapping=pd.read_csv(out/'all_662_source_record_mapping.csv')
 mapping['review_category_cn']=mapping.system_record_id.map(categories).fillna('')
 mapping['reason']=mapping.system_record_id.map(ledger.set_index('record_id').reason)
 mapping.to_csv(out/'all_662_source_record_mapping.csv',index=False,encoding='utf-8-sig')
 ds=ogr.Open(str(out/'reconstruction_results.gpkg'),1);assert ds is not None
 for layername in ['lia_systems_and_volume_changes','modern_system_endpoints']:
  layer=ds.GetLayerByName(layername)
  if layer.GetLayerDefn().GetFieldIndex('review_category_cn')<0:layer.CreateField(ogr.FieldDefn('review_category_cn',ogr.OFTString))
  for rid,cat in categories.items():
   layer.SetAttributeFilter(f"record_id='{rid}'");f=layer.GetNextFeature();assert f is not None
   f.SetField('review_category_cn',cat);f.SetField('reason',ledger.loc[ledger.record_id.eq(rid),'reason'].iloc[0]);assert layer.SetFeature(f)==0
  layer.SetAttributeFilter(None)
 ds=None
 page=(out/'report.html').read_text(encoding='utf-8')
 pos=page.index('<h2>待复核记录</h2>')
 visible=review[['record_id','mountain','review_category_cn']].copy()
 visible['位置图']=[f'<a href="review_maps/{rid}.png">查看边界</a>' for rid in visible.record_id]
 page=page[:pos]+'<h2>未估计系统与位置图</h2>'+visible.to_html(index=False,escape=False)+'<p><a href="scientific_review_required.csv">完整诊断表</a>。这些系统的体积尚未估计；区域冰量变化只使用通过检查的重建子样本。</p></html>'
 extras='<h2>全部编目基线与进一步分析</h2><p><a href="full_inventory_baseline/report.html">全部660系统：面积变化与现代冰厚产品体积</a> · <a href="quality_analysis/report.html">山脉完整性、数据差异与结果敏感性</a> · <a href="common_ice_surface_change_statistics.csv">共同有冰区域的冰面高程变化</a> · <a href="thickness_change_raster_QC.csv">完整冰厚变化图层核验</a></p><p>新增结果通过独立数组检查；原642条合格模型结果保持原值。冰面变化图层仅覆盖两期共同有冰区域，消失冰区的完整变化由冰厚变化图层描述。</p>'
 page=page.replace('<h2>方法与结果范围</h2>',extras+'<h2>方法与结果范围</h2>')
 page=page.replace('冰面、冰厚、床面栅格索引','冰面、冰厚、床面及变化栅格索引')
 (out/'report.html').write_text(page,encoding='utf-8')
 for folder in ['full_inventory_baseline']:
  assert not (out/folder).exists();shutil.copytree(follow/folder,out/folder)
 print(json.dumps({'reconstructed':int(ledger.status.eq('reconstructed-QC-passed').sum()),'unestimated':len(review),'human_readable_report_updated':True}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('output_directory');a=p.parse_args();main(Path(a.output_directory))
