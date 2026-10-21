# 原生CUDA v35独立运行候选 v62

候选入口：`polim_cpu.native_candidate_v1.entry`。主入口`polim_cpu.gpu_run`及默认设置不变。此候选受控制器授权和逐文件SHA256门禁约束；授权关闭后run/resume拒绝模型启动，不可当作自由运行接口。

## 已实现命令（PowerShell）
在项目根目录执行，CPU管理解释器与原GPU解释器保持隔离：
```powershell
Set-Location E:/3spyderProject/1_LIA_himalaya_oggm
$cpu = 'D:/ProgramData/anaconda3/python.exe'
& $cpu -m polim_cpu.native_candidate_v1.entry check polim_cpu/native_candidate_v1/configs/A.json
& $cpu -m polim_cpu.native_candidate_v1.entry check polim_cpu/native_candidate_v1/configs/R.json
# 以下为本轮冻结授权开放期间的实际执行入口；验收完成关闭授权后不可重算。
& $cpu -m polim_cpu.native_candidate_v1.entry run polim_cpu/native_candidate_v1/configs/A.json
& $cpu -m polim_cpu.native_candidate_v1.entry resume polim_cpu/native_candidate_v1/configs/R.json
& $cpu -m polim_cpu.native_candidate_v1.entry status polim_cpu/runs/v62-native-candidate-v1/A
& $cpu -m polim_cpu.native_candidate_v1.entry status polim_cpu/runs/v62-native-candidate-v1/R
```
配置明确A1552→1554、R1553→1554；R必须指向A本轮accepted-1553.json。check为只读身份/年份/覆盖核验，R依赖尚未生成会标明；正式resume要求依赖实际存在。无隐式起止年份、无气候循环、无输出覆盖。
完整集中注册执行由`python -m polim_cpu.native_candidate_v1.batch`完成，仅一次。它不是跳过批准的快捷入口，调用现有控制器事务，重复版本会被拒绝。

## 方程与参数
系统2387，50m，float64，固定床面/材料A/无滑动。原全域BP残差与Newton/PCG/projected-AMG；每个30天或更短CFL(.35)步先把年度SMB按可用冰限制，再进行原固定速度公共边守恒输运、向外出口，再求完整BP。不是CN，不包含几何切线、FGMRES、旧J响应或新湿种子。
原MonthlyTI历史气候按本轨迹当前年初冰面计算一次并年内冻结：temp_bias=0，melt_f=10.207686460617214，prcp_fac=3.672361860575085，temp_melt=-1°C。原数据years1470…1569中仅用1552、1553，不把覆盖范围当已验证范围。R读取A自身1553年初生成的缓存并核对年/H/气候/参数/文件哈希，不借用CN1553缓存。
原BP<=1e-6、质量<=1e-10，原80Newton/120秒、35线搜索、200PCG及原预条件重建规则保持；GPU配额10%，一GPU进程。单年最多128完整BP、300秒；A600秒、R300秒；整批2400秒含准备和审计。

## 检查点、失败与恢复
原生packed年度frame包含h/u、准确year、step、last_solve、V0、smb_volume、exported及历史rows，按原CRC/独立数组SHA256/fsync/不可覆盖目录封存。候选accepted-YEAR.json只增加身份和账本引用，不把CN元数据冒充原生格式。它绑定frame的文件哈希、完整metadata、网格、气候、物理、代码身份、原生历史账本和1552起窗口账本。
每年使用新原生进程/冷AMG。原v35恢复step>0时prev_u/prev_dt为空，保留当前u与last_solve；A第二年与R完全采用这个相同原年度恢复规则，没有凭H重建u。初态来自原v35已验收1552终点，其原1551历史账本单列，不混入1552窗口出口。
失败保存native/failure-state.npz（明确未接受）、原result日志、外层failure.json及已封存年度。status中的last_accepted_year为最后耐久年度；故障记录另给最后内存已接受步。未封存半年的重算需要另外明确授权，本入口不会自行重试。
supervisor日志与各year/native日志分开。进程退出0不代表科学通过；以result.json、year/acceptance.json、annual工程结果、最终恢复报告共同判定。

## 已验证范围与使用限制
实际结果以独立证据目录report.md/verification.json为准。允许本轮验收的范围仅1552—1554及1553恢复后缀，不能宣称任意冰川/气候、长期生产或百年能力。
主入口继续为`python -m polim_cpu.gpu_run run <原入口配置>`及其resume/status/report，不应向它传候选配置。原主入口本身有其100年研究配置限制和授权检查；本轮不调用、不更改。
未来替换需要另行批准：配置兼容层、控制器登记与恢复迁移方案、长期耐久/并发状态文件与故障隔离验证、其他动力状态覆盖；最后才修改gpu_run.py导入及current-working-settings指针。现在两者均不改。
