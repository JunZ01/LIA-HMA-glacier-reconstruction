# LIA project instructions

Before analyzing, plotting, configuring, or running any LIA simulation, read
`LIA_PROJECT_FACTS.md` in full.

Treat that file as the canonical project contract. Do not replace its verified
facts with assumptions from old outputs or generic OGGM conventions. If a new
result conflicts with it, stop the affected batch, show the conflicting
evidence, and update the facts file only after the conflict is resolved.



## Fixed user method: batch-first parameter search (2026-10-14)

看曲线找偏差 → 固定其他参数 → 根据已有响应确定调整方向和范围 → 把有依据的邻域一起并行跑 → 看整轮曲线，再决定下一个参数。

spinup温差和正演温差分开，跑过的结果直接利用。

最重要的是批量：优先减少总墙钟时间、零碎补跑和交互轮次；控制变量不等于一次只跑一个点。

每轮先给完整参数表、组合数、复用项、并行安排及有依据的整批时间预算；中心、两侧和备用邻域在同轮有界覆盖，独立冰川共同安排。逐项控制变量，整轮看曲线后再选下一参数。本条是固定工作方法，不是新模拟或恢复旧取消队列的授权。具体权威记录见 LIA_PROJECT_FACTS.md 第1341条。


### Spinup/forward pairing clarification (2026-10-14; fact 1342)

人类进一步强调spinup与forward温差的参数组合。剩余四条下一轮先准备固定其它物理的delta×q配对设计；温差分开定义，同时成组比较配对，不能再用固定一个spinup、转去扫melt的建议代替。具体边界目前为助手建议，未授权启动；旧已算结果复用与数值失败限制保持。


## Human-locked continuous execution architecture (2026-10-16; fact 1353)

六条冰川的持续运行方法固定于 `LIA_EXECUTION_CONTRACT.md` 第10节。优先快速分组批量迭代和已有结果/父态/CP复用；小性能改动只做一次针对变化部分的短测试，已有验证直接引用，禁止嵌套审查、重复冷复现及未经人类规定的加速门槛。整轮完成后在既有授权内继续下一有限组，不重复索要同一授权。助手不得自行改变此架构，只有人类明确修改才版本化替代。当前在跑冻结批次及科学门保持；2342/3185未决数值路线与取消路线不因持续运行而解禁。第1342条记录的是当时设计状态，不得当作覆盖事实1348以后既有迭代授权的永久未授权状态。


## Human research-method memory (2026-10-16; fact 1367)

科研问题、前沿研究和原始文献证据先于技术选型。禁止用技术猜想、代码能跑、模型复杂程度、数据可得性或工程进展替代科学论证与方法适配性比较。
多区域LIA研究须正面比较Yingkui Li、Hongjie Zhang及相关论文；关键全文/补充无法获取时及时告知并请用户提供，用户已表示可以发送，不能仅凭摘要默默固定另一条路线。区分题录、摘要、主文、补充、复现与验证的阅读/执行状态。
Huintjes2016当前是候选方法，不能因已下载数据或技术样例跑通就默认为654大冰川的唯一主线；补齐关键原文比较后按既有权限和合同推进。不新增逐步审批、重复测试或对已授权工作的额外门槛。
详细记忆：`validation_data/results/paleo_surface_validation_v0/lia_research_method_memory_20261016_v1/科研方法选择记忆.txt`。权威事实见 `LIA_PROJECT_FACTS.md` 第1367条。本条不改变独立六冰川已冻结运行与持续执行安排，也不自动停止/重启数据下载。


## Temperature-first batch reminder (2026-10-18)

人类再次明确：优先调温度，搜索必须批量。delta与q分开定义并成组配对；先列完整中心/冷暖两侧、已算复用、总组合与并行预算。不要在已授权温度任务里默认退回三点/九点melt小提案、逐点补跑或重复确认批量方法。遵守原物理及稳定/评分门，有依据地冻结同包执行；已算历史不重复。


## Coupled endpoint selection (2026-10-18)

人类指出缺少“什么情况调什么参数”的可靠判据。仅温差/fs/melt允许范围保持；不得声称某参数只控制现代、另一参数独立控制峰或幅度。每轮用已有同基准响应同时说明两端移动量、目标区域和代价；考虑局部非线性，不用未验证线性补偿承诺解。两端通过后仍按原六门及完整曲线判断，不新增门。助手新增温度距平幅度k提案已撤回，不等待许可/实现/启动。批量与原固定物理不变。详见最新事实1444（如编号因并发变化，以事实标题及当前state为准）。


## Forward-temperature bilateral search reminder (2026-10-19)

人类再次明确“再多增加一点”“需要两侧考虑”。正演温差q的每轮搜索也必须明确中心、较冷和较暖两侧，不能只把双向要求用于spinup，或只扩冷侧。控制变量和物理依据保持，完整有限参数包先去重/复用再共同执行；冷暖两侧不预设哪一侧有利，不串行补点。当前修订方案以q=0为中心、冷暖各5档，详见next-temperature-search-proposal-v2.json；这条记忆不自动启动未登记模型或更改原门/预算。
