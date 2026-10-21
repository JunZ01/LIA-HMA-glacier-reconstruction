from pathlib import Path
import json, shutil, os, time, hashlib
from html import escape
from yx_followup_transaction_v1 import register, sha

base = Path('H:/ToYangxin/result_yx')
out = base / 'completed_reconstruction_20261009_v4_654'
follow = base / 'completion_followup_20261009_v1'
s = json.loads((out/'summary.json').read_text(encoding='utf-8'))
assert s['reconstructed_QC_pass'] == 654 and s['review_systems'] == 6
q = json.loads((out/'quality_analysis/incremental_quality_final_verification.json').read_text(encoding='utf-8'))
assert q['status'] == 'pass' and not q['core_sha256_mismatches'] and not q['broken_local_links']
qa = json.loads((out/'quality_analysis/analysis_summary.json').read_text(encoding='utf-8'))
assert qa['reconstructed_systems'] == 654 and qa['missing_systems'] == 6
audit_path = Path(os.environ['YX_FINAL_CORE_AUDIT'])
audit_hash = os.environ['YX_FINAL_CORE_AUDIT_SHA'].lower()
assert sha(audit_path).lower() == audit_hash
independent = json.loads(audit_path.read_text(encoding='utf-8'))
assert independent['status'] == 'pass', independent.get('status')
coverage = 100-qa['missing_LIA_area_percent']
portal = out/'completion_overview.html'
assert not portal.exists()
portal.write_text(f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HMA小冰期冰面与冰量重建成果 · 654系统</title>
<style>body{{font-family:Microsoft YaHei,Arial,sans-serif;max-width:1060px;margin:40px auto;padding:0 18px;line-height:1.8;color:#1f2937;background:#f8fafc}}main{{background:white;padding:34px;border-radius:12px}}a{{color:#0369a1}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left;vertical-align:top}}h1{{font-size:26px}}li{{margin:5px 0}}.note{{background:#f1f5f9;padding:15px;border-radius:6px}}@media(max-width:600px){{main{{padding:18px}}h1{{font-size:22px}}td,th{{padding:8px}}}}</style>
<main><h1>HMA大冰川小冰期重建与山脉统计</h1>
<p>采用新编目 needxd=1，剔除102条与现代边界完全相同的记录。662条变化记录合并为660个冰川系统，按本地10山脉分区统计。</p>
<table><tr><th>统计范围</th><th>已核验结果</th></tr>
<tr><td>全部660系统面积</td><td>小冰期27,331.54 km²；现代26,063.56 km²；减少1,267.97 km²（4.64%）</td></tr>
<tr><td>654系统重建冰量</td><td>小冰期{s['lia_volume_deduplicated_km3']:,.2f} km³；现代{s['modern_volume_km3']:,.2f} km³；估计减少{s['volume_loss_km3']:,.2f} km³（{s['volume_loss_percent']:.2f}%）</td></tr>
<tr><td>重建覆盖</td><td>654/660系统；覆盖入选小冰期面积{coverage:.2f}%</td></tr>
<tr><td>6系统未估计</td><td>YX-03055、03621、05103、05176、14077、28236；共{qa['missing_LIA_area_km2']:.2f} km²（{qa['missing_LIA_area_percent']:.4f}%）。缺测保持未知，不填零。</td></tr></table>
<p class="note">冰面、冰厚和冰量是模型估计，现代冰量闭合属于标定。面积统计覆盖全部660系统，冰量统计覆盖通过核验的654系统。结果对应本次发生边界变化的大冰川样本；没有推算整个HMA总量或年变化率，未构建正式冰量置信区间。</p>
<ul><li><a href="report.html">重建报告、10山脉统计及地图</a></li>
<li><a href="quality_analysis/report.html">质量与不确定性证据报告</a></li>
<li><a href="full_inventory_baseline/report.html">全部660系统面积及原生现代资料基线</a></li>
<li><a href="reconstruction_results.gpkg">GIS图层</a></li>
<li><a href="paired_volume_changes.csv">逐系统冰量变化</a></li>
<li><a href="local_10_mountain_summary.csv">10山脉冰量汇总</a></li>
<li><a href="common_ice_surface_change_statistics.csv">共同有冰区域冰面变化统计</a></li>
<li><a href="gis_raster_index.csv">冰面、冰厚、床面及变化栅格索引</a></li>
<li><a href="scientific_review_required.csv">6个未估计系统及原因</a></li>
<li><a href="README.txt">资料与栅格说明</a></li>
<li><a href="full_delivery_sha256_manifest_v5.json">最终文件校验清单</a></li></ul>
<p>原651条已核验模型及变化栅格原值保留，新加入YX-10304、11977、21852。11977采用原定0.5%现代冰量容差内的已存检查点，只修正先查符号区间再查容差的程序条件；其0.00113 km³模型冰量差小于0.00244 km³现代闭合残差。28236的受控资料修正仅完成输入检查，代表和独立对照的原流线头部条件失败后停止，无新增模型。</p>
<p>10分区中7个含本次入选样本，西、中、东喜马拉雅3区无入选样本，统计为空。原生冰厚资料缺口、局部现代冰面误差、DEM路由代理及共同床面投影差异，在质量报告中单列说明。</p></main></html>''',encoding='utf-8')

core = json.loads((out/'delivery_sha256_manifest.json').read_text(encoding='utf-8'))
quality = json.loads((out/'quality_analysis/quality_analysis_sha256_manifest.json').read_text(encoding='utf-8'))
assert len(core['files']) == 7899
assert sha(out/'delivery_sha256_manifest.json').lower() == q['core_manifest_sha256'].lower()
core_paths = {r['relative_path'] for r in core['files']}
entries = {r['relative_path']:dict(r) for r in core['files']}
for item in quality['files']:
    item = dict(item); item['relative_path'] = 'quality_analysis/'+item['relative_path']
    assert item['relative_path'] not in entries
    entries[item['relative_path']] = item
for p in sorted(out.rglob('*')):
    if not p.is_file(): continue
    rel = p.relative_to(out).as_posix()
    if rel in entries: assert p.stat().st_size == entries[rel]['bytes'], rel
    else: entries[rel] = {'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p).lower()}
full = out/'full_delivery_sha256_manifest_v5.json'
assert not full.exists()
full.write_text(json.dumps({
    'status':'verified654-core-and-quality-sealed','reconstructed':654,'unestimated':6,
    'core_independently_verified':7899,'quality_files':len(quality['files']),
    'independent_core_QC_path':str(audit_path),'independent_core_QC_sha256':audit_hash,
    'original651_unchanged':True,'files':list(entries.values())
},ensure_ascii=False,indent=2),encoding='utf-8')

# Reuse the just-completed independent core audit; verify final additions once.
for r in entries.values():
    if r['relative_path'] in core_paths: continue
    p = out/r['relative_path']; h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    assert h.hexdigest() == r['sha256'].lower(), r['relative_path']

print(register(
    'yx_static_followup_delivery_v4',
    {'status':'verified654-static-delivery-six-unestimated','delivery_path':str(out),
     'full_manifest':str(full),'full_manifest_sha256':sha(full),'summary':s,'quality_summary':qa,
     'independent_core_QC':str(audit_path),'independent_core_QC_sha256':audit_hash,
     'full_delivery_files_verified':len(entries),'original651_unchanged':True,'all660_reconstructed':False},
    'YX inventory: final654 independently sealed ice-surface reconstruction and local10-mountain delivery',
    f'The immutable654 derivative retains all original651 paired-model values and prior endpoint/difference rasters; only independently verified10304/11977/21852 supplement the final population. All7899 core files, new difference-grid integrals and common-S masks, population/member and local10-mountain totals pass independent audit. Final LIA3351.2161671466/modern3022.6269297718/loss328.5892373748km3 and9.8050743666percent include the preserved approximate overlap correction. Full660 area baseline stays4.6392281313percent area loss. Six unestimated systems03055/03621/05103/05176/14077/28236 occupy122.2137356099km2 or0.4471528172percent selected LIA area; these volumes remain unknown. Quality report aligned654, formal confidence interval absent, modern-volume closure remains calibration; no annual-rate or HMA-wide inference.11977 original bounded0.5percent implementation condition was transparently corrected once using immutable modern tau40 cache; its modeled loss0.001125297171km3 is smaller than modern calibration residual0.002439664287km3.28236 COP-native source correction terminates at failed original-head input gates with no model launch. Final core/quality/portal combined {len(entries)} byte hashes are verified before public pointer publication. All tested reconstruction research branches are terminal and launchfalse; no formal191 or unrelated climate state change.',
    False, [],
    'Use final654 reconstruction, full660 area baseline and local10-mountain reports. Six scientific holds remain unknown; no duplicate or closed-mechanism launch.',
    phase='verified-final-delivery'))
pointer = base/'latest_completed_reconstruction.json'
previous = base/f'latest_completed_reconstruction_prior_pointer_{time.time_ns()}.json'
shutil.copy2(pointer, previous)
value = {'path':str(out),'report':str(portal),'results_report':str(out/'report.html'),
         'quality_report':str(out/'quality_analysis/report.html'),'full_manifest':str(full),
         'summary':s,'reconstructed':654,'unestimated':6,'full_population_reconstructed':False,
         'previous_pointer_snapshot':str(previous)}
tmp = pointer.with_suffix('.654.tmp')
tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8'); os.replace(tmp,pointer)
print(json.dumps({'public_verified':654,'files':len(entries),'coverage_percent':coverage,'portal':str(portal)},ensure_ascii=False))
