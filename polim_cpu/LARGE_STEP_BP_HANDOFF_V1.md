# 新对话交接：BP 大步长积分开发

交接日期：2026-09-17。工作目录：`E:/3spyderProject/1_LIA_himalaya_oggm`。

用户要求将“隐式冰厚更新＋扩散预测＋BP校正”的开发工作放到新对话中开始。请先读本文件及规划，从阶段1实际实施，不要再次只提交一份规划。

## 权威与执行范围

本文件是交接背景，不覆盖项目契约。先完整读取 `LIA_PROJECT_FACTS.md`，再读 `LIA_EXECUTION_CONTRACT.md` 与 `LIA_CURRENT_STATE.json`，运行：

```powershell
& 'D:/ProgramData/Anaconda3/python.exe' Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py
```

读取 `polim_cpu/LARGE_STEP_BP_ROADMAP_V1.md`。规划已登记为 `polim_v36_large_step_plan_v1`，证据在 `validation_data/results/paleo_surface_validation_v0/polim_v36_large_step_plan_v1/`。该规划分支当前 `launch_authorized=false`，说明尚未登记具体数值试验，不是程序运行中。

新对话接到开始任务的用户指令后，可实施原型，并在数值试验前冻结实际代码、输入、门槛、范围、资源和新运行指纹，完成控制器登记。不得跳过登记直接把规划分支当作运行许可。若最新权威文件有变化，以其为准。

## 第一个交付

实现固定速度下的隐式守恒输运、冰厚非负和受可用冰量限制的消融；随后接上原BP的冰厚—速度外层校正，加入可关闭的扩散初值预测。明确局部质量方程残差，接受状态同时满足原BP和离散质量方程。扩散只用于求解加速，不作为永久人工平滑。

阶段1结束后，按规划组织一轮季度、半年、年度步长的集中对照。先50米系统2387，保持物理与气候不变。不直接运行千年，不改为全斯托克斯，不启动另一套神经网络开发。

## 可复用实现与证据

- 当前百年主入口：`polim_cpu/gpu_run.py` → `gpu_model_v5.py` / `historical_worker_v5.py`。1470—1570历史试算已完成，不重跑已完成基准。
- 加速候选：`adaptive_bp_worker_v4.py`、`adaptive_bp_solve_v1.py`、`projected_amg_v1.py`、`packed_storage_v1.py`；完整规划中有其阶段边界。
- 精确BP、输运及GPU构件：`gpu_allocator_safe_v1.py`、`gpu_chunked_v1.py`、`gpu_cached_v1.py`、`gpu_bp_v1.py`。
- 原始CPU BP：`polim_cpu/v13r1/bp.py`；气候：`historical_climate_v1.py`。
- 基准：`polim_cpu/runs/v32r4-historical-1470-1570-v1/`，含年度状态和气候资料。
- 最近测试：`polim_cpu/runs/v35-annual-checkpoint-v1/`；结果页：`polim_cpu/reports/annual-checkpoint-v1/index.html`。
- 首次对照拟用1482—1484、1552—1554，具体方案及拟定误差门槛见规划。尚未启动。

加速候选在1481—1482年由74.498秒降至41.122秒，在1551—1552年由39.142秒降至12.236秒。它仍每个小步完整求解BP，尚未接入百年主入口。速度收益主要来自线性迭代和落盘减少；不能声称已经实现年度积分。

## 已知实现注意点

- 旧AMG复用时必须投影到当前有冰节点；否则会向无冰节点引入速度修正，与BP切线的活动子空间不一致。
- GPU进程不要调用会另行加载CPU OpenMP的NumPy线性代数；此前初始体积CPU点积触发过库冲突。使用现有隔离启动环境，不设置 `KMP_DUPLICATE_LIB_OK`。
- 年度耐久检查点可接受中断后重算未保存的一年；积分步长与保存频率是两个独立设置。
- 隐式迎风输运矩阵通常不对称，不能直接套用BP的PCG。
- 固定速度隐式稳定不等于整个冰厚—BP耦合稳定；BP收敛也不等于时间离散误差合格。
- 启动依赖准备结果的进程前，必须等待准备进程真正完成；不要把返回的运行会话ID当作完成。
- 旧失败方案及其原因在版本证据中保留；不得盲目重复已关闭机制。

## 用户偏好

先做最小可运行产品，再快速迭代。验证尽量合并，避免反复预试和重复确认；报告实际结果与必要的限制，不堆砌假设问题。沿用授权范围自主实施，同时遵守项目的版本、守恒、恢复和运行登记约定。

本交接文件不表示已经创建了新对话、提交了启动消息或启动了模拟。以应用中新对话的实际消息和项目控制器状态为准。
