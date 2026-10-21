"""Complete interpretation of saved results, including the broader fast tail."""
import json,shutil
import numpy as np
from .v13r1.registration import REPORT
from .v3.runner import write_json

def finalize():
    c=json.loads((REPORT/'comparison.json').read_text(encoding='utf-8'))
    with np.load(REPORT/'comparison-fields.npz',allow_pickle=False) as z:
        mask=z['common_mask'];n=int(mask.sum());tail={k:int(np.sum(z[k][mask]>1000)) for k in ['old_speed','new_speed','observed_speed']}
    summary=dict(common_cells=n,diagnostic_threshold_m_year=1000,above_threshold_cells=tail,interpretation='Numerical recovery succeeded. Surface-first reconstruction improves the junction and lowers the peak, but increases the upper velocity tail and reconstructed volume. Not promoted as a solution to the spatial velocity problem. The threshold is a descriptive count, not a universal glacier speed limit.')
    write_json(REPORT/'high-speed-summary.json',summary)
    path=REPORT/'index.html';archive=REPORT/'index-v1.html'
    if archive.exists():raise ValueError('Report already finalized')
    shutil.copyfile(path,archive);s=path.read_text(encoding='utf-8')
    s=s.replace('<div class="panel"><p id="outcome">','<div class="panel"><h2>结论：局部改善，整体高速问题尚未解决</h2><p id="outcome">',1)
    old,new,obs=c['common']['old_day1'],c['common']['new_day1'],c['common']['observed']
    paragraph=f'<p><b>不能只看最高速度：</b>共同范围速度第95百分位从 {old["p95"]:.1f} 升至 {new["p95"]:.1f} 米/年；超过1000米/年的像元从 {tail["old_speed"]} 增至 {tail["new_speed"]} 个（共 {n} 个）。因此保留为几何机制试算，不替换当前模型。1000米/年仅用于同尺度计数，不是所有山地冰川的速度上限。</p>'
    s=s.replace('<p id="spatialOutcome"></p>','<p id="spatialOutcome"></p>'+paragraph,1)
    s=s.replace('</table><img src="branches.png">', '</table><img src="branches.png">')
    row=f'<tr><td>共同范围第95百分位</td><td>{obs["p95"]:.3g}</td><td>{old["p95"]:.3g}</td><td>{new["p95"]:.3g}</td></tr>'
    s=s.replace('</table><img src="branches.png"',row+'</table><img src="branches.png"',1)
    s=s.replace('<a href="comparison.json">对照统计</a>','<a href="comparison.json">对照统计</a> · <a href="high-speed-summary.json">高速范围统计</a> · <a href="peak-diagnostics.json">高速点几何</a>')
    path.write_text(s,encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':finalize()
