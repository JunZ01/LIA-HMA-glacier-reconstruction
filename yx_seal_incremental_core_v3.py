from pathlib import Path
import sys,json,hashlib
import pandas as pd
out=Path(sys.argv[1]);manifest=out/'delivery_sha256_manifest.json';assert not manifest.exists()
count=json.loads((out/'summary.json').read_text(encoding='utf-8'))['reconstructed_QC_pass'];assert count==651
old=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v2_650')
prior=pd.read_csv(old/'paired_volume_changes.csv').set_index('record_id');new=pd.read_csv(out/'paired_volume_changes.csv').set_index('record_id')
for c in ['lia_volume_km3','modern_volume_km3','tau_kpa']:assert (prior[c]==new.loc[prior.index,c]).all()
page=(out/'report.html').read_text(encoding='utf-8')
page=page.replace('原642条合格模型结果保持原值','原650条合格模型结果保持原值；新增系统YX-04312通过两项独立保存数组复核')
page=page.replace('冰面变化图层仅覆盖两期共同有冰区域，消失冰区的完整变化由冰厚变化图层描述。','冰面变化图层仅覆盖两期共同有冰且床面数值差≤0.002m的像元，其余保持缺测；消失冰区的完整变化由冰厚变化图层描述。')
extra='<h2>新增大系统的资料与模型适用范围</h2><p>YX-04312采用当前地形与冰面高程构建候选小冰期流线，候选出口与分水岭尚无古冰流观测验证。现代计算直接复用已通过的冰面与应力，原650条结果未重算。现代冰量标定使用三个有资料成员的原生冰厚积分；第四个1.521km²成员没有该产品支持，真实体积保持未知，计算采用原DEM参考床面假设。现代整体冰面检验通过，但小组件局部误差增大，详细质量附录分别披露。原取样规则包含与边界相交的像元，两个源像元仅部分落入轮廓。</p><p><a href="common_grid_bed_alignment_diagnostics.csv">固定源床面的两期投影数值差异</a>另表给出；固定原始输入并不意味着独立投影结果逐像元完全相同。</p>'
page=page.replace('<h2>方法与结果范围</h2>',extra+'<h2>方法与结果范围</h2>');(out/'report.html').write_text(page,encoding='utf-8')
with (out/'README.txt').open('a',encoding='utf-8') as f:f.write('\n原650模型和其变化栅格保留原值，仅新增YX-04312；新小冰期流线来自当前DEM路由代理，不视为古冰面或古出口独立观测。该系统现代标定146.200791135km3为有支持成员产品积分，1.521km2成员无产品支持，真实现代体积未知，原DEM参考床面假设保持不变。局部现代冰面误差增大、边界部分像元与来源限制在质量报告中披露。\n')
(out/'prior650_preservation_provenance.json').write_text(json.dumps({'prior_delivery':str(old),'original650_volume_and_tau_rows_unchanged':True,'original650_model_recalculations':0,'old650_derived_change_rasters_byte_copied':True,'new_system':'YX-04312','modern_field_for_new_case_checkpoint_reused':True},ensure_ascii=False,indent=2),encoding='utf-8')
files=[]
for p in sorted(out.rglob('*')):
 if not p.is_file():continue
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):d.update(b)
 files.append({'relative_path':p.relative_to(out).as_posix(),'sha256':d.hexdigest(),'bytes':p.stat().st_size})
manifest.write_text(json.dumps({'schema':'YX-immutable-incremental-delivery-manifest-v3','core_reconstructed_count':count,'status':'core-sealed-quality-to-be-added-only-as-new-files','files':files},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'core_sealed':count,'files':len(files),'manifest':str(manifest)}))
