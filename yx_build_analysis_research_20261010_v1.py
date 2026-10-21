"""Persist the literature scoping review and proposed next analyses; no new glacier results."""
from pathlib import Path
import json, hashlib, html, csv, subprocess, sys

REPO = Path(__file__).resolve().parent
BASE = Path('H:/ToYangxin/result_yx')
OUT = BASE/'analysis_research_20261010_v1'
SELECT = BASE/'selected_reconstruction_20261010_v1_654'
LENGTH = BASE/'paired_axis_length_selected654_20261010_v2'
QUALITY = BASE/'completed_reconstruction_20261009_v4_654/quality_analysis/analysis_summary.json'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,j):
    p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def ref(author,year,title,url,topic,access,evidence,use,limit,secondary=''):
    return dict(author=author,year=year,title=title,url=url,topic=topic,access=access,evidence=evidence,relevance=use,limits=limit,secondary_url=secondary)

refs=[
ref('Lee et al.',2021,'Accelerated mass loss of Himalayan glaciers since the Little Ice Age','https://www.nature.com/articles/s41598-021-03805-8','LIA区域重建','全文方法核查',
    '以LIA地貌轮廓和消融区表面插值估计喜马拉雅冰量差，并按表碛及末端类型比较。',
    '学习空间高程分析和表面类型分组；公开轮廓可作同区域方法对照。',
    '消融区表面差、地理范围和年代与本样本的全冰川配对厚度体积不同，不能直接比较总百分比。','https://doi.org/10.5518/939'),
ref('Li & Li',2014,'Topographic and geometric controls on glacier changes in the central Tien Shan, China, since the Little Ice Age','https://www.cambridge.org/core/journals/annals-of-glaciology/article/topographic-and-geometric-controls-on-glacier-changes-in-the-central-tien-shan-china-since-the-little-ice-age/A810AC5A1EDAD46F0F1E94C31068E654','规模和地形','全文方法核查',
    '分析487个冰川的规模、高程、坡向、形状与LIA以来面积变化，并讨论已消失冰川造成的样本偏差。',
    '山脉内部控制规模和地形后比较变化；审查大冰川选择与完成绘制规则。','其他山脉的系数和坡向关系不能直接迁移。'),
ref('Loibl et al.',2014,'Reconstructing glacier retreat since the Little Ice Age in SE Tibet by glacier mapping and equilibrium line altitude calculation','https://cris.fau.de/publications/106615344/','LIA区域重建','作者机构摘要与元数据；全文未成功获取',
    '利用东念青唐古拉冰川地貌和几何ELA开展区域比较。','区域、坡向和气候背景参考。','不将未访问的详细方法表述为已复现。'),
ref('Kayastha & Harrison',2008,'Changes of the equilibrium-line altitude since the Little Ice Age in the Nepalese Himalaya','https://www.cambridge.org/core/journals/annals-of-glaciology/article/changes-of-the-equilibriumline-altitude-since-the-little-ice-age-in-the-nepalese-himalaya/A057F604EF503444797F9CDB9CE12708','ELA和年代','全文方法核查',
    '用现代ELA校准个体THAR，分析LIA以来地区和坡向差异；厚表碛冰川未进入正式ELA分析。',
    '说明统一几何比例需要地区和冰川类型验证。','卷期年份是2008，网页后续数字化日期不是研究年份；不可统一THAR后作独立古气候证据。'),
ref('Van Tricht et al.',2021,'Reconstruction of the Historical (1750–2020) Mass Balance of Bordu, Kara-Batkak and Sary-Tor Glaciers in the Inner Tien Shan, Kyrgyzstan','https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2021.734802/full','历史重建','全文方法核查',
    '以冰碛高程约束三个冰川的LIA消融舌，独立测地数据检查现代模型，承认早期验证不足。',
    '地貌高程约束与不同空间区域的厚度变化分析。','其固定上部几何及逐年线性调整假设不自动适用于本项目。'),
ref('Huintjes et al.',2016,'A modelling approach to reconstruct Little Ice Age climate from remote-sensing glacier observations in southeastern Tibet','https://www.cambridge.org/core/journals/annals-of-glaciology/article/modelling-approach-to-reconstruct-little-ice-age-climate-from-remotesensing-glacier-observations-in-southeastern-tibet/CB9A08537354D3E9937D55E0155D2AA4','古气候反演','全文方法核查',
    '以能量/质量平衡联合改变温度和降水；不同组合可给出相近几何约束。',
    '未来设计温度—降水联合情景并引入独立代理。','仅凭ΔELA和固定递减率不能识别唯一温度变化。'),
ref('Reinthaler & Paul',2023,'Using a Web Map Service to map Little Ice Age glacier extents at regional scales','https://www.cambridge.org/core/journals/annals-of-glaciology/article/using-a-web-map-service-to-map-little-ice-age-glacier-extents-at-regional-scales/DA311C1803525639F4D16BAD7DBDE433','轮廓不确定性','全文方法核查',
    '区分重复描绘误差与地貌解释误差，采用重复和独立专家解释。','建立自己的轮廓可信度和重复描绘实验。','质量等级不等于概率误差；不能移植其误差百分比。'),
ref('Oien et al.',2022,'Testing the area–altitude balance ratio (AABR) and accumulation–area ratio (AAR) methods of calculating glacier equilibrium-line altitudes','https://doi.org/10.1017/jog.2021.100','ELA方法','全文方法核查',
    '64个现代冰川比较几何ELA与零净平衡ELA；全球推荐AABR1.56/AAR0.58，中亚偏差仍较大。',
    '两期几何ELA参数敏感性与现代独立雪线校核。','全球参数和误差不能直接作为654系统的真实参数或正式误差条。'),
ref('Henz et al.',2025,'Alps-wide high-resolution 3D modelling reconstruction of glacier geometry and climatic conditions for the Little Ice Age','https://tc.copernicus.org/articles/19/5913/2025/','三维重建和敏感性','全文方法核查',
    '50m三维模型分析LIA体积、ELA、区域标度和参数/分辨率/响应状态敏感性。','学习三维空间指标与结构敏感性设计。','其体积情景变化约14%属于自身模型，不能当作本项目置信区间或授权替换模型。'),
ref('Zhang et al.',2025,'Glacier mass loss on the southeastern Tibetan Plateau since the Little Ice Age','https://onlinelibrary.wiley.com/doi/10.1002/jqs.3715','LIA区域重建','作者上传原论文全文方法核查',
    '2454个LIA冰川，以RGI7/DEM、连接流线、现代厚度和PalaeoIce重建厚度/表面，并分析ELA、规模和湖末端。',
    '与当前全冰川静态重建非常接近；应把研究问题推进到空间损失预算和独立验证。',
    '现代F19对照和引用工具开发验证不等于本区独立实测验证；数据不能默认可直接下载。','https://www.researchgate.net/publication/390751603_Glacier_mass_loss_on_the_southeastern_Tibetan_Plateau_since_the_Little_Ice_Age'),
ref('Zhang et al.',2026,'Changes in glaciers and climate in the Karakoram since the Little Ice Age','https://www.sciencedirect.com/science/article/pii/S1674927826000766','最新区域重建','出版社摘要核查；全文未成功获取',
    '3883个LIA轮廓和流线厚度/表面，摘要报告LIA至2000面积减6.8%、体积减19.4%，并有分区和规模差异。',
    '喀喇昆仑LIA重建和区域差异已有系统工作，跨山脉预算与响应分离更值得研究。','未核验其完整参数、现代厚度来源及独立验证；不同总体不能直接与654比较。'),
ref('Brun et al.',2019,'Heterogeneous Influence of Glacier Morphology on the Mass Balance Variability in High Mountain Asia','https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JF004838','地形和类型','全文方法核查',
    '6470个冰川分12地区，冰舌坡度、海拔、表碛及雪崩指标解释8%–48%地区内部近期质量平衡差异。','预设少量变量、山脉内部比较、检验共线性和湖端类别。','近期相关不是LIA以来因果；表碛作用非普遍单向，不能仅做两类均值。'),
ref('Shean et al.',2020,'A Systematic, Regional Assessment of High Mountain Asia Glacier Mass Balance','https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2019.00363/full','现代观测和空间误差','全文方法核查',
    '多源DEM评估2000–2018年高程变化，处理空间相关误差。','现代背景观测与空间分块稳定性检验。','需要旧RGI到RGI7系统映射；近期速率不是LIA累计变化的直接验证。'),
ref('Miles et al.',2021,'Health and sustainability of glaciers in High Mountain Asia','https://www.nature.com/articles/s41467-021-23073-4','高程带和动态','论文及作者存档全文方法核查',
    '结合速度、厚度、高程变化估计通量散度和高程带表面质量平衡。','支持冰川内部高程分带分析和动态分类。','静态重建不提供真实表面质量平衡、积累量或未来健康度。','https://nora.nerc.ac.uk/id/eprint/530309/1/s41467-021-23073-4.pdf'),
ref('King et al.',2019,'Glacial lakes exacerbate Himalayan glacier mass loss','https://www.nature.com/articles/s41598-019-53733-x','湖末端和表碛','论文及作者存档全文方法核查',
    '近期喜马拉雅湖末端冰川有较大冰量损失，比较其与表碛及陆地末端类别。','检验湖冰实际接触与损失空间位置的联系。','缺乏接触或年代证据时不能以附近有湖代替湖末端；效应大小不直接移植。'),
ref('Sakai & Fujita',2017,'Contrasting glacier responses to recent climate change in high-mountain Asia','https://www.nature.com/articles/s41598-017-14256-5','气候敏感度','原论文核查',
    '相同气候强迫下冰川敏感度也存在地区差异。','区分气候强迫差异与几何/积累过程调节。','地理变化差异不能直接解释为升温差异。'),
ref('Maurer et al.',2019,'Acceleration of ice loss across the Himalayas over the past 40 years','https://doi.org/10.1126/sciadv.aav7266','时间序列','原论文核查',
    '在1975–2000和2000–2016实际观测时段比较损失，结合气候资料。','以后增加有明确年代的观测端点研究加速度。','两个静态LIA/现代端点不能给出全区域加速倍率或统一年率。'),
ref('Bahr, Meier & Peckham',1997,'The physical basis of glacier volume-area scaling','https://doi.org/10.1029/97JB01696','标度关系','原论文方法核查',
    '体积—面积标度由守恒与形状、坡度等尺度闭合假设推导。','两期标度残差作为几何一致性和异常诊断。','同一数据拟合的标度不能独立验证体积，也不能直接当单冰川时间演化律。'),
ref('Farinotti et al.',2017,'How accurate are estimates of glacier ice thickness? Results from ITMIX, the Ice Thickness Models Intercomparison eXperiment','https://tc.copernicus.org/articles/11/949/2017/','独立厚度验证','全文方法核查',
    '多模型的局部厚度存在差异，集合与观测对照改善总体表现。','区分局部厚度与全冰川体积的误差。','试验总体误差不能直接移植；本研究失败路线不能组成有效集合。'),
ref('Farinotti et al.',2021,'Results from the Ice Thickness Models Intercomparison eXperiment Phase 2 (ITMIX2)','https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.571923/full','独立厚度验证','全文方法核查',
    '按完整测线拆分校准/测试，检查稀疏及偏向低高程观测的影响。','以未参与上游产品与本模型校准的测线验证空间厚度。','密集相邻点随机拆分会泄漏；少量点不能保证空间厚度真实。'),
ref('Farinotti et al.',2019,'A consensus estimate for the ice thickness distribution of all glaciers on Earth','https://www.nature.com/articles/s41561-019-0300-3','现代厚度产品','论文及作者机构全文核查',
    '现代厚度是多模型集合估计，不是全部冰川的直接厚度观测。','说明现代F19目标的性质及产品误差来源。','闭合到F19体积属于校准，不能称独立观测验证。','https://folia.unifr.ch/documents/307744/files/hus_cht.pdf'),
ref('Welty et al.',2020,'Worldwide version-controlled database of glacier thickness observations','https://essd.copernicus.org/articles/12/3039/2020/','厚度观测数据','全文和数据描述核查',
    'GlaThiDa提供版本化厚度观测及元数据。','检查现代成员与测线交集、日期、误差及校准独立性。','观测覆盖不均；测线内空间相关不能按像元数简单降低误差。'),
ref('Sommer et al.',2023,'Constraining regional glacier reconstructions using past ice thickness of deglaciating areas – a case study in the European Alps','https://tc.copernicus.org/articles/17/2285/2023/','退冰区厚度约束','全文方法核查',
    '多期DEM约束过去厚度，但仅依赖低处退冰区观测会偏低，研究做地形相关修正。','用近现代历史DEM设计额外厚度约束和空间验证。','现代裸地DEM只能约束床；它本身不提供LIA冰面，文中约40%偏差不是我们的误差。'),
ref('Huss',2013,'Density assumptions for converting geodetic glacier volume change to mass change','https://tc.copernicus.org/articles/7/877/2013/','冰量换质量','全文方法核查',
    '有效换算密度随时间窗与粒雪变化而异。','未来将体积变质量时规定密度情景和适用条件。','常用850±60kg/m³不是所有静态LIA冰体或任意时间窗的精确密度。'),
ref('Loibl, Richter & Grünberg',2025,'Remote sensing-derived time series of transient glacier snowline altitudes for High Mountain Asia, 1985–2021','https://www.nature.com/articles/s41597-024-04309-6','独立雪线数据','全文与数据描述核查',
    'RGI6冰川全年瞬时雪线，1985–2021；约28336个冰川、966万条观测。','为现代几何ELA提供消融季末雪线校核，按实际源年代与成员映射。','全年记录不能直接平均为ELA；云/观测缺口常使峰值偏低，DEM基准固定且编号不同。','https://doi.pangaea.de/10.1594/PANGAEA.953619'),
ref('Larocca et al.',2026,'Glacier Equilibrium-Line Altitude Change Across Alaska and Adjacent Canada Indicates a Cold, Dry Little Ice Age and Weaker Aleutian Low','https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL119840','ELA和气候','主文方法核查；补充参数未完全取得',
    '215个LIA重建ELA对比2016–2024人工夏末雪线；地形控制后分析气候情景。','ELA值得开展，但应增加独立雪线与温度/降水条件。','区域均值SE不等于逐冰川模型误差；插值残差的相关点不独立，阿拉斯加参数不移植HMA。'),
ref('Zhou et al.',2026,'Mamba-trident: terrain-guided tri-modal Mamba network for mapping little ice age glaciers','https://www.sciencedirect.com/science/article/pii/S1569843226004401','最新轮廓方法','出版社摘要和元数据；全文未成功获取',
    '采用光学、雷达、地形多模态开展LIA轮廓识别，摘要报告跨区域验证。','未来扩展未完成编目或人工边界对照的候选方法。','当前边界质量仍须独立地貌解释；训练区泄漏和详细性能未在本轮核查。'),
ref('RGI Consortium',2023,'The Randolph Glacier Inventory version 7.0 User guide','https://zenodo.org/records/8362857','官方编目定义','官方指南核查',
    '区分单体、复合体和中心线产品，目标年代约2000而非统一拍摄年份。','核查1256现代成员、现代轮廓日期及旧编号空间映射。','不能把RGI约2000统一改成2026现代状态。'),
ref('Savoskul & Solomina',1996,'Late-Holocene glacier variations in the frontal and inner ranges of the Tian Shan, central Asia','https://journals.sagepub.com/doi/10.1177/095968369600600104','原始年代证据','论文摘要核查',
    '冰碛年代与ELA资料显示不同山地位置的晚全新世冰川过程需分别研究。','每系统LIA地貌年代和可信度登记。','本轮未复核全部测年细节；不将其年龄直接赋给当前轮廓。'),
ref('Rowan',2017,'The Little Ice Age in the Himalaya: A review of glacier advance driven by Northern Hemisphere temperature change','https://journals.sagepub.com/doi/10.1177/0959683616658530','年代背景综述','综述及年代汇编全文；背景材料',
    '汇编冰碛年龄并讨论地区与地貌期次。','用作找原始测年论文的线索，不作本样本统一年代赋值。','属于综述背景；不能替代七个山脉各自的原始年龄证据。','https://eprints.whiterose.ac.uk/id/eprint/103616/'),
]

plans=[
dict(priority='首选',question='冰量差由哪些空间区域贡献？',analysis='持续冰区厚度差＋LIA独有区原冰量－现代独有区冰量；同时按固定海拔和归一化高程分带。',
     ready='现有两期厚度/床面/轮廓及差分栅格，先核验守恒和共同床面。',outputs='654系统三项预算；山脉分解图；高程—冰量曲线；6–8个预设代表冰川地图/剖面。',
     guard='保留未知/无效像元、现代外溢与负值；复用原共床掩膜，厚度差与冰面差分开。重叠修正和守恒残差单列，不回写旧结果。'),
dict(priority='与首选并行',question='哪些结果可信，哪些结论受选择和重建误差影响？',analysis='完整654质量分层、山脉×规模内敏感性；轮廓/源日期/床支持/模型家族/长度旗标；独立厚度测线与LIA冰碛高程资料交集。',
     ready='已有QC可以直接整理；独立观测交集与新边界复核需新增资料。',outputs='证据/年代/支持矩阵；完整主样本与分层稳健性图；独立验证可行清单；误差来源预算方案。',
     guard='不增加剔除门；F19闭合为校准，数值审计不等于真实性验证。空间重采样不是正式模型置信区间。'),
dict(priority='第二模块',question='面积、主轴长度与冰量是否同向响应？',analysis='联合两期面积、最大保存主轴与模型体积，比较边界变化与内部变薄，控制基准规模和山脉。',
     ready='654长度已补齐；面积/冰量和已有地形字段可复用。',outputs='三指标响应图；29个现代轴更长、36个负体积损失等的逐例解释；七个有样本山脉的规模调整比较。',
     guard='主轴指标保留裁剪/主支差异；不改称实测末端退缩。零/负值不截断，避免只选显著正损失。'),
dict(priority='第二模块的解释',question='损失空间分配与哪些地形/类型因素相关？',analysis='少量预设变量：基准规模、冰舌坡度、海拔/高程分布、坡向；随后接表碛、湖冰接触、跃动/速度。',
     ready='基本地形可由已有资料整理；表碛、湖、速度与动态类别需年代匹配的外部资料。',outputs='山脉内效应与交互图；分块检验；损失分解与低高程冰量占比等关联。',
     guard='已有简单相关/回归不重复作为新主结果；坡度等也是厚度算法输入，关联可能有模型内生性，不能直接称因果。'),
dict(priority='第三模块',question='重建表面支持怎样的几何ELA变化？',analysis='两期AAR/AABR敏感性；现代消融季末雪线独立校核；单体与多出口系统的对应规则。',
     ready='表面高程场已有；可匹配1985–2021公开雪线记录与实际源日期。',outputs='654几何ELA主表与方法差异；有观测成员的验证子表；山脉ΔELA图和敏感性范围。',
     guard='现代观测缺测保持缺测；不用全年雪线均值，不把离散精度当总误差；雪线观察年份与DEM/轮廓分列。'),
dict(priority='后续条件工作',question='气候变化、年率、未来水资源能解释到何种程度？',analysis='先取得LIA年龄、多个可靠现代观测时段和独立温度/降水；之后设计联合气候情景及动态验证。',
     ready='当前静态两端点不足；需独立年代/气候/动态资料和新模型契约。',outputs='可识别的温度—降水响应范围或时段变化；届时再讨论年率、加速度和水资源。',
     guard='不统一设1850年；不用ΔELA×固定递减率直接宣布ΔT，不把累计冰量损失当某年融水供给。'),
]

def main():
    reads=[]
    for name in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
        p=REPO/name;s=p.read_text(encoding='utf-8-sig');reads.append({'path':str(p),'fully_read':True,'chars':len(s),'sha256':sha(p)})
    validator=REPO/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
    v=subprocess.run([sys.executable,str(validator)],cwd=REPO,capture_output=True,text=True,encoding='utf-8',check=True)
    assert json.loads(v.stdout)['status']=='pass'
    OUT.mkdir(exist_ok=False)
    save(OUT/'canonical_read_receipt.json',reads)
    save(OUT/'readonly_validator.json',json.loads(v.stdout))
    save(OUT/'literature_evidence_matrix.json',refs)
    with (OUT/'literature_evidence_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=refs[0].keys());w.writeheader();w.writerows(refs)
    existing=json.loads(QUALITY.read_text(encoding='utf-8-sig'))
    baseline=json.loads((SELECT/'summary.json').read_text(encoding='utf-8-sig'))
    axes=json.loads((LENGTH/'summary.json').read_text(encoding='utf-8-sig'))
    proposal={'status':'literature-scoping-review-and-proposed-analysis-plan-only','review_date':'2026-10-10',
        'source_count':len(refs),'not_a_claim_of_exhaustive_systematic_review':True,'research_question':'Selected HMA large glaciers: how is modeled LIA-to-modern ice-volume difference partitioned between changes in retained ice and changes in footprint, and how does that partition relate to mountain, elevation distribution and area/axis response?',
        'population':654,'source_records':655,'mountain_partition':'local10;7 represented and3 no-sample Himalaya partitions remain blank',
        'volume_metrics_are_model_estimates':True,'length_metric':axes['method'],
        'first_priority':'conservative three-part spatial volume budget plus fixed-elevation hypsometry',
        'parallel_priority':'quality, representativeness, chronology and independent validation feasibility',
        'proposed_modules':plans,'new_glacier_metrics_computed':False,'physical_model_launch_authority':False,
        'canonical_state_or_existing_deliveries_modified':False,'novelty_claim_not_proven':True,
        'existing_results_sources':{str(QUALITY):sha(QUALITY),str(SELECT/'summary.json'):sha(SELECT/'summary.json'),str(LENGTH/'summary.json'):sha(LENGTH/'summary.json')}}
    save(OUT/'proposed_analysis_plan.json',proposal)
    selected_files=['selected_system_status.csv','paired_volume_changes.csv','selected_inventory_area_and_modern_volume.csv','common_grid_bed_alignment_diagnostics.csv','common_ice_surface_change_statistics.csv','overlap_duplicate_volume_correction.json','thickness_change_raster_QC.csv']
    source_hashes={str(SELECT/f):sha(SELECT/f) for f in selected_files}
    source_hashes.update(proposal['existing_results_sources']);source_hashes[str(LENGTH/'paired_axis_lengths654.csv')]=sha(LENGTH/'paired_axis_lengths654.csv')
    save(OUT/'existing_readonly_input_hashes.json',source_hashes)

    def esc(s):return html.escape(str(s))
    rows=''.join(f'<tr data-topic="{esc(r["topic"])}"><td>{i}</td><td>{esc(r["author"])}<br>{r["year"]}</td><td><a href="{esc(r["url"])}">{esc(r["title"])}</a>'+(f'<br><a href="{esc(r["secondary_url"])}">作者原文/数据入口</a>' if r['secondary_url'] else '')+f'</td><td>{esc(r["topic"])}<br><small>{esc(r["access"])}</small></td><td>{esc(r["evidence"])}</td><td>{esc(r["relevance"])}</td><td>{esc(r["limits"])}</td></tr>' for i,r in enumerate(refs,1))
    prows=''.join(f'<tr><td>{esc(p["priority"])}</td><td><b>{esc(p["question"])}</b><br>{esc(p["analysis"])}</td><td>{esc(p["ready"])}</td><td>{esc(p["outputs"])}</td><td>{esc(p["guard"])}</td></tr>' for p in plans)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>654冰川后续分析：文献调研与确定方案</title>
    <style>body{{margin:30px auto;max-width:1250px;padding:0 20px;font:16px/1.75 Arial,"Microsoft YaHei",sans-serif;color:#25333f}}h1{{font-size:28px}}h2{{font-size:22px;margin-top:34px}}h3{{font-size:18px}}a{{color:#225e8a}}.call{{background:#edf4f7;border-left:5px solid #3f728f;padding:18px}}.note{{background:#f7f1e4;padding:15px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;min-width:1000px;width:100%;font-size:13px}}th,td{{border:1px solid #ced8dd;padding:8px;vertical-align:top}}th{{background:#eaf0f4;text-align:left}}small{{color:#51616b}}input{{width:min(90%,660px);font:inherit;padding:9px;margin:12px 0}}.equation{{background:#f3f6f8;padding:15px;font-size:17px}}footer{{font-size:13px;color:#5d6b74;margin-top:30px}}@media print{{input{{display:none}}body{{max-width:none}}table{{min-width:0;font-size:10px}}}}</style>
    <h1>654个大冰川：文献调研与下一轮分析方案</h1><p>调研日期：2026年10月10日。整理{len(refs)}项论文与官方资料，覆盖LIA区域编目/三维重建、几何与类型控制、ELA/雪线、厚度验证、误差与年代。文献全文、摘要及补充材料的访问程度逐项说明；这是一轮针对本项目的广泛范围调研，并非穷尽全部文献的系统综述。</p>
    <div class="call"><b>确定首选研究问题：</b>小冰期以来，高山亚洲选定大冰川的模型冰量差，有多少表现为覆盖范围变化，有多少表现为持续冰区厚度变化？这种空间分配在山脉、规模与高程分布间如何变化，并与面积和保存主轴长度响应联系？<br><b>先执行的分析方向：</b>守恒的三部分空间冰量预算＋固定高程分带；质量/代表性/独立验证与之并行。然后开展面积—主轴—体积联合响应，ELA作为有独立雪线支持的第三模块。</div>
    <h2>为什么这样确定</h2><p><a href="https://onlinelibrary.wiley.com/doi/10.1002/jqs.3715">Zhang等2025年藏东南研究</a>已经将轮廓、现代厚度、床面和流线结合，重建全冰川LIA厚度及表面，并分析ELA、规模和湖末端。<a href="https://www.sciencedirect.com/science/article/pii/S1674927826000766">2026年喀喇昆仑研究</a>也已开展区域LIA流线体积与分区对比。因此，仅增加总量、面积—体积散点或简单地形相关，难以充分发挥现有三维重建的价值。跨山脉统一样本的空间预算、三指标响应分离和自身独立验证是更具体的研究路径；本调研不宣布这些问题从未被任何人研究。</p>
    <p><a href="https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JF004838">Brun等2019年</a>支持控制规模/地区并检验地形和末端类型；<a href="https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.571923/full">ITMIX2</a>则说明厚度与体积闭合不能替代独立测线的空间验证。<a href="https://tc.copernicus.org/articles/19/5913/2025/">Henz等2025年</a>说明高阶三维研究仍需检验输入、物理参数和响应状态敏感性，其数值误差不能移植给本项目。</p>
    <h2>当前能力与已有结果</h2><p>所有主分析保持654个系统、655条源记录；本地10山脉分区中7个有样本，3个喜马拉雅分区留空。已有两期厚度、冰面、床面、轮廓、差分栅格、共同床面对齐诊断及完整两期保存主轴。当前面积减少4.612%、模型冰量减少9.805%、最大保存主轴长度加权减少10.345%分别有自身算子和权重，不能直接视为同一种退缩比例。</p>
    <p>旧质量附录已包含面积—体积相关、地形回归、留一山脉和质量子集对照。该附录的模型部分对应现有654，人口上下文仍是历史660，面积总量不能套用到当前654。现有36个负冰量损失系统、29个现代轴更长系统全部保留。现代源轮廓日期字段为1993–2008；DEM和雪线日期另列，不把端点称为2026状态。现代RMSE中位{existing['modern_surface_RMSE_median_m']:.2f}m是同源空间一致性诊断，不是独立观测误差或正式CI。</p>
    <p>当前330个系统的原生正冰厚床支持低于95%；14个系统有18个未支持现代成员。现代F19体积闭合为校准，文件/数组独立核查为实现验证。后续独立验证需测线、冰碛高程等未参与本模型或上游产品校准的证据。质量标记用于解释和分层，主654不再筛除。</p>
    <h2>优先分析清单</h2><div class="scroll"><table><thead><tr><th>顺序</th><th>问题与分析</th><th>数据是否可用</th><th>可交付结果</th><th>解释和实施条件</th></tr></thead><tbody>{prows}</tbody></table></div>
    <h2>首选分析的守恒定义</h2><div class="equation">ΔV = ∫<sub>L∩M</sub>(H<sub>L</sub>−H<sub>M</sub>)dA + ∫<sub>L∖M</sub>H<sub>L</sub>dA − ∫<sub>M∖L</sub>H<sub>M</sub>dA</div>
    <p>L、M为两期冰区，H为各期重建厚度。依次为持续冰区厚度差、LIA独有区冰量、现代独有区冰量。三项先报告绝对体积和相对LIA基准体积；不把所有系统强制转为各项占净损失的比例，避免净差接近零或负值时夸张。模型厚度差不直接称为实测历史表面质量平衡。</p>
    <p>先核查现有差分栅格的守恒生成规则、像元支持与两期掩膜；使用已验证共床范围解释冰面差，全域冰量分解用厚度积分。现有共同冰面差仅在既定床面一致性≤0.002m范围有效，其他共同像元保持nodata，不放松为新的阈值。保留20287/20368已冻结重叠修正的空间归属与扣减；无法定位的校正或支持差列为明确残差，不能用补零/重标定悄悄消去。</p>
    <p>高程分带使用一个事先声明、两期一致的参考高程及网格；绝对高程与归一化高程并列，给出分带宽度敏感性。独立用两期移动表面分箱会混合冰川变化与分箱移动，必须另作说明。分解之和回到封存的逐系统及总体体积差，沿用原积分QC精度，先做代表性只读计算检查后再完整654后处理。</p>
    <h2>ELA可以推进，但要加入独立现代雪线</h2><p><a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL119840">Larocca等2026年</a>比较LIA几何ELA与人工夏末雪线，温度/降水推断有明确条件。<a href="https://www.nature.com/articles/s41597-024-04309-6">Loibl等2025年HMA数据论文</a>及其<a href="https://doi.pangaea.de/10.1594/PANGAEA.953619">1985–2021雪线数据</a>提供现成独立资料入口。先匹配RGI6到当前RGI7成员和实际现代轮廓年代，提取质量合格消融季晚期峰值，并校核云/冰影/新雪；不是全年记录均值。多成员系统需明确单体对应，缺测成员不补成观测。</p>
    <p>几何ELA可保留654系统全表，观测校核子表注明有可靠数据的成员范围；AAR/AABR不同假设、表面误差与非平衡状态分开评估。区域平均的标准误、等高线离散精度、跨冰川空间标准差，都不是自动传播出的ELA模型误差。气候解释在年龄、季节积累、降水和独立代理约束之后，以温度—降水联合情景给出。</p>
    <h2>完整研究链与建议图件</h2><ol><li>样本/年代/输入支持图：654空间分布、母体覆盖与10山脉。</li><li>两期面积—主轴—模型冰量响应及其分离图。</li><li>三部分冰量预算：系统、规模组和7个有样本山脉。</li><li>高程带面积、厚度差和体积差曲线；低高程与持续冰区的贡献。</li><li>少量预设地形变量、末端类型与预算分配关系，地区/空间检验。</li><li>代表冰川地图与纵剖面，质量与独立验证图；ELA/雪线敏感性作为后续模块。</li></ol>
    <div class="note">主样本是needxd=1、完成绘制、排除102条完全相同轮廓并剔除6条未通过重建的选定大冰川系统。研究对象和完成/变化筛选需在方法中披露；不能自动推广整个HMA平均。研究范围、规模、现代日期与体积算子不同，当前百分比与喜马拉雅/藏东南/喀喇昆仑文献的差异不能直接判为科学冲突。建议题目范围：“高山亚洲选定大型冰川自小冰期以来的几何变化与冰量损失空间分配”。</div>
    <h2>文献证据矩阵：{len(refs)}项</h2><p>按问题、作者、年份或关键词筛选。摘要核查的论文仅使用摘要能支持的内容；综述作为查找原始测年证据的背景。</p><input id="filter" aria-label="筛选文献" placeholder="输入关键词，例如 ELA、2026、独立验证、湖末端"><span id="count"></span><div class="scroll"><table id="refs"><thead><tr><th>编号</th><th>作者/年份</th><th>论文与原始链接</th><th>主题/访问程度</th><th>核查证据</th><th>对本项目的价值</th><th>适用边界</th></tr></thead><tbody>{rows}</tbody></table></div>
    <h2>文件和现有成果</h2><ul><li><a href="literature_evidence_matrix.csv">30项文献证据表</a></li><li><a href="literature_evidence_matrix.json">证据表JSON</a></li><li><a href="proposed_analysis_plan.json">确定优先级的研究方案</a></li><li><a href="../selected_reconstruction_20261010_v1_654/report.html">现有654面积/冰量</a></li><li><a href="../paired_axis_length_selected654_20261010_v2/report.html">现有654两期参考轴</a></li></ul>
    <footer>本报告为调研和拟定方案，没有计算新的冰川指标、改动原成果、改变权威路线或授权启动模拟。若后续新增输入、床面、参数或模型机制，按项目事实与执行契约创建新版本；已有失败机制不因文献出现而自动重开。本报告不提供正式冰量置信区间，不承诺尚未验证的显著关系或全球首次创新。</footer>
    <script>const box=document.getElementById('filter'),rows=Array.from(document.querySelectorAll('#refs tbody tr'));function filter(){{const q=box.value.toLowerCase();let n=0;rows.forEach(r=>{{const show=r.textContent.toLowerCase().includes(q);r.style.display=show?'':'none';n+=show;}});document.getElementById('count').textContent=n+' / '+rows.length+' 项';}}box.addEventListener('input',filter);filter();</script></html>'''
    (OUT/'report.html').write_text(text,encoding='utf-8')
    save(OUT/'research_sources_and_provenance.json',{'report_creator':str(Path(__file__)),'report_creator_sha256':sha(Path(__file__)),
        'method':'parallel primary-literature scoping review; explicit access levels; local read-only evidence/capability assessment',
        'source_count':len(refs),'searches_include_2025_and_2026_publications':True,'glacier_output_mutations':0})
    # Link and source-preservation QA is appropriate for this saved research report.
    from html.parser import HTMLParser
    from urllib.parse import unquote
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':
                for k,v in attrs:
                    if k=='href' and not v.startswith(('https:','http:','#')):self.refs.append(v)
    links=Links();links.feed(text);missing=[r for r in links.refs if not (OUT/unquote(r)).resolve().exists()]
    assert not missing,missing
    assert len(refs)==30
    assert all(sha(p)==h for p,h in source_hashes.items()), 'Read-only original input changed'
    save(OUT/'report_QC.json',{'status':'pass','literature_entries':len(refs),'all_local_links_valid':len(links.refs),
         'input_hashes_unchanged':True,'no_new_glacier_statistics_or_model_runs':True,'browser_render_claim':False})
    files=[{'relative_path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.iterdir()) if p.is_file()]
    save(OUT/'research_report_sha256_manifest.json',{'status':'sealed-literature-review-and-proposed-plan','files':files})
    print(json.dumps({'report':str(OUT/'report.html'),'sources':len(refs),'status':'pass','first_priority':proposal['first_priority'],'glacier_results_modified':False},ensure_ascii=False))

if __name__=='__main__':main()
