from pathlib import Path
import sys,json,hashlib
import pandas as pd
from osgeo import ogr
out=Path(sys.argv[1]);assert not (out/'delivery_sha256_manifest.json').exists()
prior=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v3_651')
old=pd.read_csv(prior/'paired_volume_changes.csv').set_index('record_id');new=pd.read_csv(out/'paired_volume_changes.csv').set_index('record_id');assert len(new)==654
for c in old.columns:assert (old[c].fillna('')==new.loc[old.index,c].fillna('')).all(),c
ledger=pd.read_csv(out/'all_660_system_status.csv');reasons={
 'YX-03621':'最新双端定向图检验：最低应力40kPa现代体积残差7.3152%，超过原0.5%容差；固定区间未闭合，未开始LIA。',
 'YX-05176':'最新双端定向图检验：最低应力40kPa现代体积残差9.5147%，超过原0.5%容差；固定区间未闭合，未开始LIA。',
 'YX-14077':'最新双端定向图检验：最低应力40kPa现代体积残差3.5724%，超过原0.5%容差；固定区间未闭合，未开始LIA。',
 'YX-03055':'原S场未严格改善已保存现代H场误差40.4926m；独立原路线核对未发现可支持该拓扑类推广的长跨岭连接证据。',
 'YX-05103':'原S场未严格改善已保存现代H场误差37.3112m；独立原路线核对未发现可支持该拓扑类推广的长跨岭连接证据。',
 'YX-28236':'原GyZ局地来源异常：第三NASADEM支持COPDEM；原隐藏连接逆坡与源头裁剪问题已记录，源质量和新输入路由适用性保持待核，未采用宽误差门填入体积。'}
cats={rid:('现代体积未满足原容差' if rid in ['YX-03621','YX-05176','YX-14077'] else '地形来源与路由输入待核' if rid=='YX-28236' else '现代冰面改善未通过且拓扑类型未确认') for rid in reasons}
for rid,reason in reasons.items():ledger.loc[ledger.record_id.eq(rid),'reason']=reason;ledger.loc[ledger.record_id.eq(rid),'review_category_cn']=cats[rid]
ledger.to_csv(out/'all_660_system_status.csv',index=False,encoding='utf-8-sig')
review=ledger.loc[ledger.status.ne('reconstructed-QC-passed'),['record_id','mountain','review_category_cn','reason']];assert len(review)==6;review.to_csv(out/'scientific_review_required.csv',index=False,encoding='utf-8-sig')
mapping=pd.read_csv(out/'all_662_source_record_mapping.csv');mapping['review_category_cn']=mapping.system_record_id.map(ledger.set_index('record_id').review_category_cn);mapping['reason']=mapping.system_record_id.map(ledger.set_index('record_id').reason);mapping.to_csv(out/'all_662_source_record_mapping.csv',index=False,encoding='utf-8-sig')
ds=ogr.Open(str(out/'reconstruction_results.gpkg'),1)
for layername in ['lia_systems_and_volume_changes','modern_system_endpoints']:
 layer=ds.GetLayerByName(layername)
 for rid,reason in reasons.items():
  layer.SetAttributeFilter(f"record_id='{rid}'");feature=layer.GetNextFeature();assert feature;feature.SetField('reason',reason);feature.SetField('review_category_cn',cats[rid]);assert layer.SetFeature(feature)==0
 layer.SetAttributeFilter(None)
ds=None
page=(out/'report.html').read_text(encoding='utf-8');page=page.replace('原642条合格模型结果保持原值','原651条合格模型结果保持原值，新增3个系统经独立数组复核')
page=page.replace('冰面变化图层仅覆盖两期共同有冰区域，消失冰区的完整变化由冰厚变化图层描述。','冰面变化图层只使用两期共同有冰且床面投影数值差≤0.002m的像元，其他保持缺测；消失冰区的完整变化由冰厚变化图层描述。')
extra='<h2>新增结果与六个未估计系统</h2><p>新增10304、21852采用两期候选外出口图与原精确PP/S场算子；11977直接复用原现代40kPa缓存，其0.3217%残差符合原0.5%体积容差。原程序额外要求跨零区间，修正已单独登记，原失败记录保留；未改参数范围或重算现代场。11977体积差0.0011253km³小于现代标定目标残差0.0024397km³，保持条件模型估计解释。</p><p>三个系统的现代体积仍超原容差，两系统未通过原现代冰面改善门，另一个存在地形来源异常。所有未估计体积保持未知。候选小冰期流线使用当前DEM作为路由代理，来源差异、部分边界像元和资料缺口另列质量附录。</p><p><a href="common_grid_bed_alignment_diagnostics.csv">床面独立投影数值差异</a>已统计；固定原始源输入并不保证独立投影逐像元相同。</p>'
page=page.replace('<h2>方法与结果范围</h2>',extra+'<h2>方法与结果范围</h2>')
start=page.index('<h2>未估计系统与位置图</h2>');end=page.index('</html>',start)
visible=review[['record_id','mountain','review_category_cn']].copy();visible['位置图']=[f'<a href="review_maps/{rid}.png">边界图</a>' for rid in visible.record_id]
page=page[:start]+'<h2>未估计系统与位置图</h2>'+visible.to_html(index=False,escape=False)+'<p><a href="scientific_review_required.csv">最新六系统证据与原因</a>，未填零或外推HMA总冰量。</p>'+page[end:];(out/'report.html').write_text(page,encoding='utf-8')
with (out/'README.txt').open('a',encoding='utf-8') as f:f.write('\n原651全部配对字段与变化栅格保留，新10304/21852/11977经独立检查。11977现代0.3217%残差符合原0.5%界限，原跨零程序停止记录保留，现代缓存不重算；小变化小于标定残差，不能当独立精确观测损失。六系统LIA体积未知，最新科学/数据原因已同步CSV和GIS。\n')
(out/'prior651_preservation_provenance.json').write_text(json.dumps({'prior':str(prior),'all_paired_columns_unchanged':True,'prior651_model_recalculations':0,'prior651_change_rasters_byte_copied':True,'new_records':sorted(set(new.index)-set(old.index))},ensure_ascii=False,indent=2),encoding='utf-8')
files=[]
for p in sorted(out.rglob('*')):
 if not p.is_file():continue
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
 files.append({'relative_path':p.relative_to(out).as_posix(),'sha256':d.hexdigest(),'bytes':p.stat().st_size})
(out/'delivery_sha256_manifest.json').write_text(json.dumps({'schema':'YX-immutable-final-delivery-core-v4','core_reconstructed_count':654,'status':'core-sealed-quality-added-only-as-new-files','files':files},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'core_sealed':654,'files':len(files),'unestimated':6}))
