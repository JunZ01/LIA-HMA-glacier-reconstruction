from pathlib import Path
import json,hashlib
out=Path('H:/ToYangxin/result_yx/completed_reconstruction_20261009_v2_650')
p=out/'report.html';s=p.read_text(encoding='utf-8')
s=s.replace('冰面变化图层仅覆盖两期共同有冰区域，消失冰区的完整变化由冰厚变化图层描述。','冰面变化图层覆盖两期共同有冰且床面数值差≤0.002m的像元，其余像元保留缺测，覆盖率另表提供；消失冰区的完整变化由冰厚变化图层描述。')
s=s.replace('<h2>未估计系统与位置图</h2>','<p><a href="common_grid_bed_alignment_diagnostics.csv">两期床面投影差异诊断</a>。固定床面指原始输入固定；独立重投影的数值实现存在差异。局部独立DEM来源差异另列入质量报告。</p><h2>未估计系统与位置图</h2>')
p.write_text(s,encoding='utf-8')
manifest=out/'delivery_sha256_manifest.json';assert not manifest.exists()
files=[{'relative_path':f.relative_to(out).as_posix(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':f.stat().st_size} for f in out.rglob('*') if f.is_file() and f!=manifest]
manifest.write_text(json.dumps({'schema':'YX-immutable-delivery-manifest-v2','core_reconstructed_count':650,'status':'core-sealed-quality-report-to-be-added-as-new-files','files':files},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'core_files':len(files),'core_reconstruction':650,'manifest':str(manifest)}))
