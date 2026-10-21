from pathlib import Path
import json,hashlib,os,time,shutil
from yx_followup_transaction_v1 import register,sha
base=Path('H:/ToYangxin/result_yx')
out=base/'completed_reconstruction_20261009_v2_650'
summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
assert summary['reconstructed_QC_pass']==650 and summary['review_systems']==10
qc=json.loads((out/'quality_analysis/final_core_preservation_and_links_qc.json').read_text(encoding='utf-8'))
assert qc['status']=='pass' and qc['sealed_core_files_checked']==7851 and not qc['sealed_core_hash_mismatches']
assert qc['all_local_report_links_exist'] and qc['old642_all30_file_hashes_unchanged']
baseline=json.loads((out/'full_inventory_baseline/summary.json').read_text(encoding='utf-8'))
portal=out/'completion_overview.html'
assert not portal.exists()
portal.write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>HMA大冰川小冰期重建成果</title><style>body{font-family:Microsoft YaHei,Arial,sans-serif;max-width:1050px;margin:48px auto;line-height:1.8;color:#1f2937;background:#f8fafc}main{background:white;padding:38px;border-radius:12px}a{color:#0369a1}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}h1{font-size:26px}</style><main><h1>HMA大冰川小冰期重建与山脉统计</h1><p>筛选 needxd=1，剔除102条与现代边界完全相同的记录；662条变化记录合并为660个系统，使用本地10山脉分区。</p><table><tr><th>统计范围</th><th>已核验结果</th></tr><tr><td>全部660系统面积</td><td>小冰期27,331.54 km²；现代26,063.56 km²；减少1,267.97 km²（4.64%）</td></tr><tr><td>650系统冰面与冰量重建</td><td>小冰期3,196.18 km³；现代2,871.72 km³；估计减少324.46 km³（10.15%）</td></tr><tr><td>尚未估计</td><td>10系统；占入选小冰期面积3.20%；保留缺测，不填零</td></tr></table><p>冰面、冰厚和冰量为模型估计；现代冰量闭合属于标定。面积总体与冰量重建子集的统计范围不同，不能直接外推整个HMA或推算年变化率。</p><ul><li><a href="report.html">重建结果、山脉统计和地图</a></li><li><a href="quality_analysis/report.html">质量、不确定性证据和来源审查</a></li><li><a href="full_inventory_baseline/report.html">全部660系统面积与现代原生资料基线</a></li><li><a href="reconstruction_results.gpkg">GIS成果：小冰期、现代边界及模型结果</a></li><li><a href="paired_volume_changes.csv">逐系统冰量变化表</a></li><li><a href="local_10_mountain_summary.csv">本地10山脉冰量汇总</a></li><li><a href="scientific_review_required.csv">未估计系统与科学门说明</a></li><li><a href="common_ice_surface_change_statistics.csv">共同有冰区域冰面高程变化</a></li><li><a href="README.txt">栅格说明与适用范围</a></li></ul><p>原642条已通过模型保持不变，8条新增重建单独保存来源。完整结果和质量报告均经独立检查。</p></main></html>''',encoding='utf-8')
manifest_path=out/'full_delivery_sha256_manifest_v3.json';assert not manifest_path.exists()
core=json.loads((out/'delivery_sha256_manifest.json').read_text(encoding='utf-8'))
quality=json.loads((out/'quality_analysis/quality_analysis_sha256_manifest.json').read_text(encoding='utf-8'))
entries={x['relative_path']:dict(x) for x in core['files']}
for x in quality['files']:
 y=dict(x);y['relative_path']='quality_analysis/'+y['relative_path'];assert y['relative_path'] not in entries;entries[y['relative_path']]=y
for p in sorted(out.rglob('*')):
 if not p.is_file():continue
 rel=p.relative_to(out).as_posix()
 if rel in entries:
  assert p.stat().st_size==entries[rel]['bytes'],rel
 else:entries[rel]={'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p).lower()}
manifest={'status':'verified650-core-and-quality-sealed','core_files_independently_hash_verified':7851,'quality_files_independently_verified':len(quality['files']),'core_manifest_sha256':sha(out/'delivery_sha256_manifest.json'),'quality_manifest_sha256':sha(out/'quality_analysis/quality_analysis_sha256_manifest.json'),'independent_qc_path':'quality_analysis/final_core_preservation_and_links_qc.json','all_source_and_original642_models_preserved':True,'files':list(entries.values())}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
contract={'status':'verified650-static-delivery-with10-unestimated','delivery_path':str(out),'full_manifest_path':str(manifest_path),'full_manifest_sha256':sha(manifest_path),'summary':summary,'all660_baseline':baseline,'independent_quality_QC':qc,'original642_unchanged':True,'all660_reconstructed':False,'not_HMA_population_volume':True,'modern_closure_is_calibration':True,'ongoing_new_mechanisms_separately_registered':True}
print(register('yx_static_followup_delivery_v2',contract,'YX new inventory: verified650 reconstruction, complete660 area baseline and10 explicit unestimated cases','Seal650 qualified static endpoint reconstructions,650 native-grid thickness-change and common-surface-change rasters, all660 area/native-modern baseline, local10 mountain tables,662record mapping and complete QC/uncertainty evidence. Original642 selected rows and model files remain unchanged. Reported LIA3196.17647254,modern2871.72099246 and loss324.45548008km3/10.15136313percent describe the reconstructed subset only. Full660 area loss1267.97233186km2/4.63922813percent;10 LIA-volume-unestimated cases remain3.20438041percent of selected area and are never zero-filled. Modern closure is calibration; neither HMA-wide nor historical-climate acceptance is asserted. All7851 core hashes independently verified unchanged plus69 sealed quality outputs and checked local links. This derivative delivery can be published as latest verified650 while separately registered genuine mechanisms continue. Formal191 and unrelated climate state remain unchanged.',False,[],'Read the verified650 delivery; further additions require registered scientific mechanisms and independent QC.',phase='verified-delivery'))
pointer=base/'latest_completed_reconstruction.json';snapshot=base/f'latest_completed_reconstruction_prior_pointer_{time.time_ns()}.json'
if pointer.exists():shutil.copy2(pointer,snapshot)
value={'path':str(out),'report':str(portal),'results_report':str(out/'report.html'),'quality_report':str(out/'quality_analysis/report.html'),'full_manifest':str(manifest_path),'summary':summary,'full_population_reconstructed':False,'reconstructed':650,'unestimated':10,'previous_pointer_snapshot':str(snapshot)}
tmp=pointer.with_suffix('.650.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,pointer)
print('PUBLIC_VERIFIED_DELIVERY',str(portal),len(entries))
