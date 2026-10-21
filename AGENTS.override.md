# LIA project instructions

Before analyzing, plotting, configuring, or running any LIA simulation, read
`LIA_PROJECT_FACTS.md` in full. Treat it as the canonical scientific contract.

Before taking any LIA task action, also read `LIA_EXECUTION_CONTRACT.md` and
`LIA_CURRENT_STATE.json`, then run the read-only controller validator:

`python Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py`

If the validator fails or the current state is `decision-required`, do not
launch a simulation. Reconcile the evidence and create a new versioned route
snapshot first. Conversation summaries and agent messages are context only;
they never override these files.

Do not replace verified facts with assumptions from old outputs or generic
OGGM conventions. If new evidence conflicts with the facts file, stop the
affected batch, show the conflict, and update the facts file only after it is
resolved.



## Human research-method memory (2026-10-16; fact 1367)

科研问题、前沿研究和原始文献证据先于技术选型。禁止用技术猜想、代码能跑、模型复杂程度、数据可得性或工程进展替代科学论证与方法适配性比较。
多区域LIA研究须正面比较Yingkui Li、Hongjie Zhang及相关论文；关键全文/补充无法获取时及时告知并请用户提供，用户已表示可以发送，不能仅凭摘要默默固定另一条路线。区分题录、摘要、主文、补充、复现与验证的阅读/执行状态。
Huintjes2016当前是候选方法，不能因已下载数据或技术样例跑通就默认为654大冰川的唯一主线；补齐关键原文比较后按既有权限和合同推进。不新增逐步审批、重复测试或对已授权工作的额外门槛。
详细记忆：`validation_data/results/paleo_surface_validation_v0/lia_research_method_memory_20261016_v1/科研方法选择记忆.txt`。权威事实见 `LIA_PROJECT_FACTS.md` 第1367条。本条不改变独立六冰川已冻结运行与持续执行安排，也不自动停止/重启数据下载。
