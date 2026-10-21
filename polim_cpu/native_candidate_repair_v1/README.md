# v62r1 原生 CUDA v35 限定运行候选（技术修复）

本版本修复 v62 跨年记录中的 Path JSON 序列化错误。v62 已完成且接受1552→1553，本次直接读取其完整原生检查点，执行 A续算1553→1554，再独立冷启动 R1553→1554。既有第一年不重算。原 v62 源码、故障和检查点保留。

## 实际入口与命令

在项目根目录 PowerShell 中：

```powershell
& D:/ProgramData/anaconda3/python.exe -m polim_cpu.native_candidate_repair_v1.entry check polim_cpu/native_candidate_repair_v1/configs/A.json
& D:/ProgramData/anaconda3/python.exe -m polim_cpu.native_candidate_repair_v1.entry check polim_cpu/native_candidate_repair_v1/configs/R.json
# 以下运行命令需要本版本控制器授权开启；本次完成后授权关闭，不能直接重跑。
& D:/ProgramData/anaconda3/python.exe -m polim_cpu.native_candidate_repair_v1.entry run polim_cpu/native_candidate_repair_v1/configs/A.json
& D:/ProgramData/anaconda3/python.exe -m polim_cpu.native_candidate_repair_v1.entry resume polim_cpu/native_candidate_repair_v1/configs/R.json
& D:/ProgramData/anaconda3/python.exe -m polim_cpu.native_candidate_repair_v1.entry status polim_cpu/runs/v62r1-native-candidate-v1/A
& D:/ProgramData/anaconda3/python.exe -m polim_cpu.native_candidate_repair_v1.entry status polim_cpu/runs/v62r1-native-candidate-v1/R
```

运行接口只允许冻结配置1553→1554，防覆盖；未来新窗口需要新登记与配置。`run`在本技术修订中明确为续接已接受1553检查点；`resume`为独立恢复对照。配置检查只读，不启动模型。`result.json`为科学状态；`failure.json`保留类别/位置/最后耐久年份；退出码不能替代科学验收。

## 方程、气候与生命周期

系统2387，50m，float64，固定床面、无滑动；原 BP Newton/PCG/projected-AMG。30天上限、CFL=.35，原生潜在SMB限量、固定速度守恒输运及向外输出。无CN、联合FGMRES或几何切线。材料和气候参数严格继承v35（见冻结配置）。每次年度入口独立GPU进程；原年度恢复将预测器prev_u/prev_dt置空，完整当前H/u及last_solve、原step、累计源/出口/体积和全部元数据保留。AMG冷启动。

A按本轨迹1553年初冰面，用原historical_climate_v1年度算法生成1553缓存；R核对H、年份、气候、物理与缓存字节后读取同一缓存。不是借用CN1553气候。完整1552→1554账本继承第一年，年度增量单列；原1551起点累计账本另检。

## 范围、证据及生产限制

入口差异、原依赖和旧版本选择证据引用v62的entry-differences.md、design.md及plan.frozen.json。本次worker.py和state.py与v62字节相同。准备只测此次改动，真实模型零工作。运行之后以本次report.md/audit.json为最终验收，不以准备通过冒充科学通过。

本候选不替换生产。原入口 `polim_cpu/gpu_run.py` 和 `gpu_model_v5`、默认设置和191系统账本保持。继续原工作方式即可使用原入口；未来切换应单独审查gpu_run.py后端导入、配置/气候/恢复适配和部署环境，不能仅把入口文件名替换。尚无v35百年、任意气候或任意冰川验证。v32百年证据不属于v35。

## 输出与失败

新输出：polim_cpu/runs/v62r1-native-candidate-v1/{A,R}/。监督日志在{A,R}-supervisor，数值日志在各year-1553/native，名称不冲突。每年封存checkpoints/accepted-1554.json，记录配套H/u、网格/物理/气候身份、账本及哈希。未接受候选不作为恢复入口。旧1553检查点索引在configs中，旧大数组不复制。

每后缀300秒/128完整BP、原80Newton/200PCG/35线搜索/120秒BP等更严限额保持；批次2400秒包含准备与审核。无救援重跑。修复前失败命令成本与本次续办成本全部保留，不能将两次管理命令描述成一次无中断轨迹计时。
