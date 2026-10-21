# LIA 191 execution contract

Version: `1.1`

This file governs **how work advances**. `LIA_PROJECT_FACTS.md` remains the
canonical scientific contract. The purpose of this execution contract is to
prevent chat memory, stale summaries, duplicate runs, and ad-hoc policy changes
from redirecting the project.

## 1. Authority order

When sources disagree, use this order and stop at the first conflict:

1. `LIA_PROJECT_FACTS.md` — scientific facts, frozen definitions, verified
   results, and explicit user policy.
2. `LIA_EXECUTION_CONTRACT.md` — controller and transition rules.
3. `LIA_CURRENT_STATE.json` — the single current pointer to the latest
   versioned route and evidence.
4. Versioned evidence under
   `validation_data/results/paleo_surface_validation_v0/`.
5. Conversation summaries and agent messages — useful context, never authority.

No simulation, plot, parameter search, promotion, or acceptance decision may
start when the first three sources have not been read or when
`validate_lia_execution_state_v1.py` fails.

## 2. One population ledger

- Population is always 191 unique `lia_unique_id` systems.
- The mutually exclusive controller states are `accepted`, `active`, `queued`,
  `decision-required`, `hold-new-mechanism`, and `closed-current-branch`.
- Every route snapshot must sum to 191 exactly and contain each system once.
- `accepted` can increase only after all applicable formal gates and an
  independent reproduction pass are recorded in versioned evidence.
- A promising pilot, a saved state, a steady pass, or a historical pass without
  the required reproduction is not accepted.

## 3. Stage state machine

The normal scientific sequence is:

`identity -> geometry -> inversion -> zero-MB -> steady -> historical -> reproduction -> accepted`

Rules:

- A downstream stage cannot repair or silently excuse an upstream failure.
- Every stage has an explicit `pass`, `fail`, `not-evaluated`, or
  `decision-required` result.
- A technical failure is not a scientific failure. It receives one new
  versioned technical retry only after the cause is identified and the
  scientific contract is unchanged.
- A scientific failure closes that mechanism for the tested class unless a
  genuinely new mechanism is registered.
- A new mechanism proceeds as representative -> independent confirmation ->
  class promotion. Failure at either of the first two levels stops promotion.

## 4. No-repeat and no-drift rules

- Every run has a unique fingerprint over system ID, input-state hashes,
  geometry, physics, forcing, solver, thresholds, time range, and driver
  version. An existing fingerprint blocks a duplicate launch.
- Output directories are versioned and must not be overwritten.
- A monitor may inspect an existing process but must never restart it.
- A stopped run resumes only from a validated atomic checkpoint with the same
  contract fingerprint.
- Threshold, target, operator, geometry, forcing, or solver changes require a
  new disclosed contract version. Post-result policy changes are never applied
  silently or described as if they were preregistered.
- Closed scalar brackets, smoothing families, or other exhausted mechanisms
  cannot be reopened under a new filename.

## 5. Long-run gate and progress contract

- Expensive runs require a passed numerical fixture or equivalent evidence, a
  short scientific preflight, a runtime estimate, and checkpoint/resume support.
- Preflight nesting is prohibited: at most one technical retry and one
  scientific preflight per registered mechanism before a formal run or closure.
- Independent systems or endpoints may run in parallel only when their output
  roots and memory/GPU budgets do not collide.
- On launch, record PID, command/module, start time, output root, checkpoint
  cadence, estimated runtime, and stop conditions.
- During a long run, report actual simulated year, accepted/rejected steps,
  CPU/GPU use, latest checkpoint, and new error. Do not report projected success
  as a result.
- On termination, immediately distinguish completed, intentional stop,
  technical failure, and scientific failure.

## 6. Evidence and transition transaction

Every state transition is one transaction:

1. Write immutable versioned evidence and a formal summary.
2. Check source and output hashes.
3. Create a new versioned current-route snapshot; never overwrite the prior one.
4. Update `LIA_CURRENT_STATE.json` to point to that snapshot.
5. Append the decision and reason to `LIA_DECISION_LOG.md`.
6. If the result changes a verified fact or policy, append a numbered section to
   `LIA_PROJECT_FACTS.md` only after conflict resolution.
7. Run the state validator. A failed validator leaves launch authorization false.

Counts never change from an agent message alone.

## 7. Mainline versus research branches

- The mainline is the route referenced by `LIA_CURRENT_STATE.json`.
- Alternative solvers, two-dimensional models, GPU ports, relaxed diagnostics,
  and feasibility audits are research branches until explicitly promoted.
- A research branch may prove numerical feasibility without changing accepted
  counts or replacing the mainline physical/scoring contract.
- Promotion requires a written equivalence or disclosed model-family change,
  representative success, independent confirmation, and a new route snapshot.

## 8. Current decision discipline

When `LIA_CURRENT_STATE.json` says `decision-required`, no model launch is
authorized. The next action is the stated decision or a read-only audit that
directly resolves it. Agents may not substitute a nearby parameter search,
another endpoint, or another solver simply to keep a process running.


## 9. Production survivability and lossless recovery

The following are hard launch requirements for every production batch:

- Each parameter combination writes its immutable result as soon as that
  combination finishes. A batch-level summary is never the only copy of a
  completed result.
- A recovery queue is built from validated result manifests and checkpoints.
  A completed combination is never recomputed unless the user explicitly
  authorizes that recomputation in a new versioned transaction.
- Progress and status files are monitoring aids, not scientific outputs. A
  temporary lock, rename failure, or other status-write error must be retried
  and logged, but must not terminate the controller or any model worker.
- Mutable status has exactly one writer. Readers consume completed snapshots
  and must not hold a lock that can block the writer. Temporary status files
  use unique names and a failed replacement falls back to a new immutable
  snapshot while computation continues.
- Checkpoint and member-result write failures are different from status-write
  failures. If a checkpoint or result cannot be safely validated, only that
  member is isolated as a technical failure; unaffected members continue
  under the registered batch stop rule.
- Controller failure must not delete or invalidate completed member results.
  A subsequent explicitly authorized recovery resumes only the unfinished
  queue from validated checkpoints under the same contract fingerprint.
- Before production launch, the technical preflight must demonstrate both
  restart from a saved checkpoint and continued computation while the status
  file is concurrently read or temporarily unavailable.
- Formal termination evidence must recount member result files independently
  of the last status snapshot, because a status snapshot may be stale when a
  controller fails.


## 10. 六条冰川持续分组率定架构（人类固定，2026-10-16）

本节只适用于1133/1743/2342/2692/3185/5465及其直接必需输入。用户要求“快速快速快速迭代”“分组考虑 参数分组 考虑选择区间 然后跑 然后迭代”，并明确反对反复核查消耗时间和额度。本节固定后续执行方法；已验证科学事实、其他任务和当前在跑批次的冻结计划不改。

### 固定目标和权限

- 目标是用尽可能短的总墙钟时间、较少交互和额度取得下一轮有用曲线。分析、调研、测试、登记只服务于选组、排除实际故障或保证结果可用，不作为独立进度。
- 助手不得自行改变本架构、增加核查层级或提高验收门槛。只有人类明确修改规则时才能以新版本替代；旧版本保留。助手可以并应当在已授权范围内选择有依据的参数区间、分组和并行安排。
- 人类已给出的分组执行和迭代授权持续有效；1133/5465整轮完成后，满足原门及本规则便设计、登记、执行下一有限组，不重复索要同一授权。该授权不解除2342/3185的数值路线限制、不重开取消路线、不自动正式接受候选。

### 唯一持续执行闭环

已有曲线和结果 → 每条冰川分别确定主要偏差及下一参数组 → 选有依据的中心、两侧与有限备用区间 → 去除已完成组合并复用父态/CP → 一次必要小测试或引用已有证据 → 冻结参数与预算并登记 → 同包批量运行 → 一次整轮结果和曲线评估 → 下一有限组。

| 阶段 | 必须完成的工作 | 直接进入下一阶段的条件 |
| --- | --- | --- |
| 选组 | 每条冰川独立选方向和区间；组内固定其他参数；列完整参数、基准、组合数、复用项、新spin/历史数、并行和耗时上限 | 有已有响应依据，且在持续授权范围内 |
| 小测试 | 只测试新增或变化部分；原版/改版使用同一代表输入，计完整耗时并比较原数值门；已有一致性、恢复和存活证据直接引用 | 一次必要短测试通过，或已有等价证据覆盖；不加第二层确认 |
| 运行 | 只启动唯一已登记controller；父态同包、合格forward同包；即时保存各成员和完整CP | 运行期间只观察，不因heartbeat重开、不热改冻结代码 |
| 收尾 | 根据成员终端、结果、原评分和全部曲线一次性评估两端、幅度及相位；当前冻结批次已约定的终端核对合并在此次完成 | 保存结论并闭合该组；完成工作不再复核、复算 |
| 迭代 | 针对整轮主要失败选择下一有限组；优先复用物理兼容父态、减少新的spin成本 | 按相同流程继续，不因等待重复确认而停在建议阶段 |

### 批量与科学边界

- 控制变量不等于逐点串行。独立冰川分别选组、共同调度，同数值合同的合格成员尽量同包；需要上一轮曲线才能确定的参数，仍在整轮后选择。不能用无依据的密集新spin或任意多参数笛卡尔网格凑批量。
- 只调温差、fs、melt_f；spin=base+delta，forward=base+恒定q，分开、不叠加、不分段。fs/melt变化不能同时温差补偿，各自必须有物理一致的稳定父态。
- 原数组、床、几何、气候、目标、gradient=-.0065、Tm=-1、Glen/prcp倍率1、970/1000稳定和域门、原六门、严格1500..2000 RAB及原权重保持。小测试通过不能代替父态稳定门、完整科学历史或正式接受决定。
- 完整结果、兼容父态、已保留前缀和CP直接复用；不重算已完成指纹、不冷复现。失败只阻断受影响成员或当前路线，不能据局部失败推断全部参数无解。

### 小改动与核查限额

- 同一物理、步长和精度下的局部性能改动，只需一次覆盖变化部分的短性能/数值对照；通过且有实际净收益后用于后续新版本批次。不得自动套用第3/7节的科学机制推广流程，再增加独立短测、整条冷复现或多层审查。
- 步长、精度、方程、强迫、原门的改变不属于小性能改动，不能藏在加速补丁内。当前在跑批次仍使用原冻结版本，不能用小测试理由中断重算或热替换。
- 测试前明确它要解决的问题、代表输入、原阈值、计时范围和短时硬上限。收益不足或失败就弃用；只有已找明具体技术原因且原授权允许时才有一次有限修正，不能无限改名重试、自动延长或反复调研。
- 提速按整步、整批可节省的墙钟与开发/测试成本判断，不编造2倍等未经人类规定的推广门槛，不用单个函数的提速冒充整批提速。
- 原四个权威文件完整读取要求保留，但同一轮已经读取且SHA未变的内容直接复用，不反复输出全文、重复解释或再做历史全量审计。入口validator一次，权威事务提交后一次；不在每个小工具调用前重复运行。例行heartbeat合并一次只读检查，没有新事件就结束。
- 若一个新增检查不能改变选组、是否运行或结果有效性的决定，就不增加该检查。助手不得把“保险起见”变成额外执行阶段，也不为规则文档再建立独立核查项目。

### 运行守护与持久化

- 既有自动任务64保持当前30分钟安排；只负责上述六条。运行时读取本批白名单状态/错误/终端及命名进程，命令和创建时间识别身份，active和旧progress不是启动许可。不查全局进程或其他任务。
- GPU/CPU终端及命名进程退出后才做整轮评估。短暂无状态更新不判挂起。技术异常保存已完成结果和有效CP，先查明具体原因；已授权范围内只接续未完成段，不重跑完成前缀。超出既有接续授权、数值路线限制或真实新科学决定时才请求人类决定。
- 参数表、硬上限、成员身份和输出根按批冻结。权威变化用既有WriterLock、新鲜SHA/CAS、不可变路线及提交后validator一次完成；counts/current_focus/next_action和无关分支逐项保持，不新增一套controller或重复调度器。
- 只通知新曲线、重要整轮结果、新原门候选、完成、重要新失败或确需新决定；例行健康进度保持安静。用户主动问进展则直接回答实际进度和耗时。


### 人类组批偏好更新（2026-10-17）

人类要求每轮充分考虑多种参数及组合，以便共同分析；后续不得默认只安排三点小组。保持控制变量和匹配基准，按科学问题把温差、融化、滑动及有依据的交互对照组成较充分的一包；优先复用稳定父态和已算历史，沿已实测并行资源冻结预算。不同规模旧批耗时接近是并行潜力证据，不保证任意条数同速，不为凑数密扫或运行任意大笛卡尔网格。当前在跑冻结计划不热改、不增加竞争调度；整轮收尾后直接依持续授权冻结执行下一包，无需重复询问同一许可。此偏好不放宽原科学/数值门，不恢复旧失败或取消路线。


### 人类spinup双向对照偏好（2026-10-17）

人类要求：spinup参数方向没有把握时，同时考虑相对中心的冷、暖两方向；优先复用已有合格父态和已有失败边界，只对有依据未算点作有限对照，不默认只往一侧选择。冷暖是输入方向，不保证初始长度和记忆响应；每个组合仍须自己的物理一致父态及原970/1000稳定域门。已有失败不重算，不延长spin或放门，不因一侧结果推断全空间；fs/melt变化不同时温差补偿。此偏好用于后续分组，不热改当前冻结生产，不改变原数值或评分门。
