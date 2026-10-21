# GPU BP 运行入口

在 `E:\3spyderProject\1_LIA_himalaya_oggm` 目录下使用 CPU Python 启动器。启动器会自动调用配置中指定的独立 CUDA 环境，不需手动更改全局 PATH。

```powershell
& 'D:\ProgramData\Anaconda3\python.exe' -m polim_cpu.gpu_run run polim_cpu/historical-1470-1570-v5.json
```

该命令创建新任务并运行。相同输出目录和已完成任务拒绝重复启动。本次配置用于系统2387、50米网格、1470—1570年；当前入口仅开放这一已登记科学范围，不支持随意改年份或换冰川直接启动。

查看进度：

```powershell
& 'D:\ProgramData\Anaconda3\python.exe' -m polim_cpu.gpu_run status polim_cpu/runs/v32r4-historical-1470-1570-v1
```

外部中断且原进程已退出后，在相同授权与代码、输入未变的条件下接续：

```powershell
& 'D:\ProgramData\Anaconda3\python.exe' -m polim_cpu.gpu_run resume polim_cpu/runs/v32r4-historical-1470-1570-v1
```

读取最新已接受状态及已保存的待求解状态，不重算输运。数值或技术失败已被正式关闭的任务，先定位原因并登记恢复版本；不会暗中反复重启。

重新绘制已保存结果：

```powershell
& 'D:\ProgramData\Anaconda3\python.exe' -m polim_cpu.gpu_run report polim_cpu/runs/v32r4-historical-1470-1570-v1
```

输入为已有数值调整状态、固定床面与材料参数、项目历史月温度和降水资料。每年计算随冰面高度变化的全年表面质量平衡，每个自适应时间步更新冰厚与BP流速。默认30天上限、CFL0.35、双精度、BP残差1e-6。降水异常关闭，无额外温度偏移，无热力反馈。初始状态不宣称为观测1470年几何或气候平衡态。

气候驱动首次带来节点退出冰区。当前驱动先清零无冰节点速度再构建辅助矩阵；牛顿迭代允许80次、每次BP求解最多120秒；线性迭代停滞时用当前精确切线刷新辅助矩阵一次。收敛精度保持不变。旧试验代码和失败记录保留为历史证据。

清理记录：`validation_data/results/paleo_surface_validation_v0/polim_cleanup_20260916_v1/result.json`。已删除275个缓存或无引用临时文件，11份重复输入通过NTFS硬链接去重。硬链接保持原路径、字节和哈希；这些科学输入仍按不可变文件管理。

网页是结果查看器；计算由上述入口执行。
