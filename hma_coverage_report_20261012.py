import json,html,shutil,hashlib
from pathlib import Path
import pandas as pd
R=Path('E:/3spyderProject/1_LIA_himalaya_oggm');O=Path('H:/ToYangxin/result_yx/hma_mountain_coverage_20261012_v1')
e=html.escape
p=json.loads((O/'workbook_payload.json').read_text(encoding='utf-8'))
cl=json.loads((R/'hma_coverage_climate_audit_20261012.json').read_text(encoding='utf-8'))
for name in ['hma_coverage_climate_audit_20261012.json','hma_coverage_partition_audit_20261012.json']:
 shutil.copy2(R/name,O/name)
tab=pd.DataFrame(p['summary']);tab=tab[tab.mountain10!='Outside / unresolved']
cols=['mountain_zh','selected654_n','legacy245_n','legacy191_static_n','tanggula691_n','east_tianshan640_n']
labels=['本地10区','完成集合654','旧245轮廓','其中旧191重建','唐古拉来源','东天山来源']
table=tab[cols].rename(columns=dict(zip(cols,labels))).to_html(index=False,border=0,classes='data')
clrows=[]
for d in cl['datasets']:
 period=d.get('period',d.get('time',''))
 clrows.append({'资料':d['name'],'时段':period,'使用边界':d.get('scope_limit','')})
cltable=pd.DataFrame(clrows).to_html(index=False,border=0,classes='data')
src=''.join(f'<li><strong>{e(s["label"])} · {s["n_source_records"]}</strong><br><code>{e(s["source_path"])}</code></li>' for s in p['sources'])
rules=''.join(f'<li>{e(x)}</li>' for x in p['methods'])
doc=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HMA山脉资料覆盖与统一分析准备</title><style>
body{{font:16px/1.75 'Microsoft YaHei',sans-serif;color:#213841;background:#f4f7f6;margin:0}}main{{max-width:1180px;margin:auto;background:white;padding:42px}}h1{{font-size:30px;line-height:1.35}}h2{{font-size:23px;margin-top:38px;border-top:1px solid #d7e3e3;padding-top:24px}}h3{{font-size:18px}}a{{color:#007b90}}.kicker{{color:#51808b;letter-spacing:2px}}.note{{padding:17px 22px;background:#eef6f6;border-left:4px solid #147d92}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{border:1px solid #d7e3e3;padding:15px;border-radius:8px}}.big{{font-size:30px;font-weight:bold;color:#147d92}}.small{{font-size:13px;color:#60717b}}img{{max-width:100%}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:11px 13px;border-bottom:1px solid #dbe4e5;text-align:left;vertical-align:top}}th{{background:#eaf2f3}}code{{font:12px/1.5 Consolas,monospace;overflow-wrap:anywhere}}li{{margin:8px 0}}@media(max-width:700px){{main{{padding:20px}}.grid{{grid-template-columns:1fr 1fr}}}}@media print{{body{{background:white}}main{{padding:10px}}}}
</style><main><div class="kicker">LOCAL DATA AUDIT · 2026-10-12</div><h1>HMA山脉资料覆盖与统一分析准备</h1>
<p>已定位并核对四套小冰期轮廓来源，采用本地自定义10个山脉分区。喜马拉雅、唐古拉和东天山可以纳入统一研究设计；当前各批次的冰面重建和现代资料匹配处于不同阶段。</p>
<div class="grid"><div class="card"><div class="big">654</div>本次完成集合<br><span class="small">已存冰面及冰量结果</span></div><div class="card"><div class="big">245</div>旧喜马拉雅系统<br><span class="small">其中191套有试验重建</span></div><div class="card"><div class="big">691</div>唐古拉来源轮廓<br><span class="small">包含小冰川；待统一配对</span></div><div class="card"><div class="big">640</div>东天山来源轮廓<br><span class="small">包含小冰川；待统一配对</span></div></div>
<p class="note"><strong>这是资料覆盖审计，尚未完成合并后的科学分析。</strong>2230为四套文件的来源记录数。独立核查未发现四套之间的轮廓相交，但系统定义、完成标记、现代成员归属和年代仍需一致核验，不能据此宣布2230条大冰川全部可用。旧191属于旧245，不额外累加。</p>
<p><a href="HMA山脉资料覆盖.xlsx">下载覆盖工作簿</a> · <a href="HMA_mountain_inventory_coverage.png">地图PNG</a> · <a href="HMA_mountain_inventory_coverage.pdf">地图PDF</a> · <a href="coverage_geometry.gpkg">GIS图层</a> · <a href="source_system_registry.csv">逐来源记录表</a></p>
<h2>山脉与资料来源</h2><img src="HMA_mountain_inventory_coverage.png" alt="四套小冰期编目在本地10个山脉分区中的覆盖地图"><div class="scroll">{table}</div>
<p class="small">各列为来源记录数，不是总体覆盖率。分区缩写：HK兴都库什，WH/CH/EH西/中/东喜马拉雅，KK喀喇昆仑，WTS本地西天山边界，WKL西昆仑，NQ念青唐古拉，ITP高原内部山地，PM帕米尔。</p>
<h3>三处需要明确的地理口径</h3><ul><li><strong>旧245实际均在喜马拉雅：</strong>西段110、中段114、东段21；其中已有试验重建分别84、89、18。旧元数据里82条标为“Karakoram / Hindu-Kush”，它们的实际几何全在本地西喜马拉雅区。旧标签保留用于溯源，区域年代或气候先验需重新核对，不能跟随旧名称直接转用。</li><li><strong>唐古拉691条属于独立来源：</strong>均落在本地“高原内部山地”大区，不能把该区654集合中的83条全部称为唐古拉。念青唐古拉也应保持独立。</li><li><strong>本地“Western Tien Shan”边界覆盖东天山来源：</strong>保留原始分区名，同时保留东天山来源标签。若论文需要东西天山对比，应先确定细分边界；不能只按本地英文名字判断其地理范围。</li></ul>
<p>已检查本地15区文件：它属于GTN-G／RGI二级分区，不符合本次山脉分区要求。另有39多边形比较图层，包含未裁定边界，暂不作为正式替代。当前10区可完成主表与覆盖地图。</p>
<h2>哪些资料可以支持下一步</h2><div class="scroll">{cltable}</div>
<p class="note">已有全域气候或测地资料，不等于每个新系统已有有效观测。旧资料主要沿RGI6成员组织；654及新找到的唐古拉、东天山现代文件采用RGI7，必须先建立一对多的空间对应，再报告有效覆盖面积及缺测。气候格点也不是独立冰川观测样本。</p>
<h2>统一研究的主问题</h2><p><strong>小冰期以来，HMA大冰川的冰量损失在各山脉如何分配；相近面积退缩为何对应不同的冰量损失；这些差异在多大程度上与气候背景及冰川几何响应一致？</strong>这三个问题可放在同一研究中，先建立可比变化量，再检验解释。</p>
<ol><li><strong>损失量与损失比例的空间分布。</strong>同时报告绝对冰量变化、相对初始冰量变化、面积变化和已有轴长指标。分别显示山脉汇总与逐冰川分布，并量化结果是否由少数巨型冰川主导。把误差传播到山脉总量，避免把模型冰量当直接观测。</li><li><strong>面积退缩与三维减薄是否一致。</strong>在同床一致的范围内，分开统计消失区的冰量与保留冰区的厚度变化，检验“轮廓变化小但冰量变化大”是否具有系统性。使用已有长度指标时保留其参考轴定义，不能直接称为真实末端退缩距离。</li><li><strong>山脉差异能否跨区域解释。</strong>先以初始规模、高程分布、坡度、表碛、末端条件等建立比较框架，再加入有技能评估的气候量。采用留一山脉验证与空间／气候格点分组，检查关系能否推广。几何变量也是重建输入时，必须量化这种数学耦合，不能把高相关性直接视为物理机制证据。</li></ol>
<p>唐古拉与东天山的完整大小冰川来源还有一个价值：在完成一致的轮廓质量检查和现代配对后，可检验大冰川结果在各区域的代表性。当前仅作尺度诊断，唐古拉有40个、东天山有4个轮廓的LIA面积达到10平方公里；这个阈值不是正式样本筛选，654集合保持不变。</p>
<h2>建议的执行顺序</h2><ol><li><strong>冻结统一系统登记：</strong>以654和旧245作为优先核对的899条候选来源记录，建立现代成员、编目年代、完成状态、边界来源和大小定义；唐古拉、东天山按同一规则补充。当前无跨源几何相交，仍需检查相邻轮廓是否共享现代成员。</li><li><strong>统一重建比较标准：</strong>逐系统检查现代DEM／冰厚、床面、网格、参数来源及体积积分范围。旧191为几何闭合试验，先做可比性审计，不能直接与654加总；是否重建缺项依项目执行状态另行裁定。</li><li><strong>先完成区域冰量预算与三维变化分解：</strong>这些分析最直接利用当前资料的优势，构成论文结果主体。起始年代未落实时报告累计变化，不强设统一LIA年份计算年变化率。</li><li><strong>最后开展几何—气候解释：</strong>完成RGI跨版本、气候有效覆盖、观测独立性及分区敏感性检查后，再测试机制假设。保留不同LIA年代与重建方法的不确定性。</li></ol>
<h2>审计与溯源</h2><p>本次只读取源数据并生成派生表、覆盖图与审计记录，未启动冰川模拟。两套独立空间计算对2230条来源记录的区域归属完全一致，六种跨源组合均无相交。654原有山脉标签与重算结果一致。</p><ul>{src}</ul><details><summary>查看方法和解释规则</summary><ol>{rules}</ol></details><p><a href="coverage_QC.json">主核验记录</a> · <a href="independent_spatial_QC.json">独立空间核验</a> · <a href="hma_coverage_climate_audit_20261012.json">气候资料审计</a> · <a href="hma_coverage_partition_audit_20261012.json">分区来源审计</a> · <a href="supplementary_discoveries.json">未纳入主登记的模型／草稿资料</a></p>
</main></html>'''
(O/'report.html').write_text(doc,encoding='utf-8')
shutil.copy2(__file__,O/'report_reproduce.py')
print(str(O/'report.html'))
