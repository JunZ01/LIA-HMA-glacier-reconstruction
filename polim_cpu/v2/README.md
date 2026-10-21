# 第二版：冰川流动、焓输运与 GPU

已实现并运行：粗网格非线性求解、温度/含水率对流动的反馈、CPU 与 CUDA、真实 Arolla 剖面输入、质量守恒、完整断点续算、失败成员隔离和离线报告。最终工程验收 **16 项通过，0 项失败**。

这是独立实验模型。动量方程使用新的 Q1 有限元离散，不能称为原 MATLAB 求解器的逐项等价移植；未完成 PoLIM 全功能复现、观测校准或 LIA 科学验收。

## 直接查看结果

- [真实 Arolla 剖面 · GPU](../runs/acceptance-v2-2/arolla-gpu/index.html)
- [粗网格耦合 · GPU](../runs/acceptance-v2-2/coupled-gpu/index.html)
- [温冰、含水率与底部融化 · GPU](../runs/acceptance-v2-2/warm-gpu/index.html)
- [最终验收证据](../runs/acceptance-v2-2/verification-final.json)
- [Arolla 输入与来源说明](examples/arolla/provenance.json)

## 运行和续算

从项目根目录 `E:\3spyderProject\1_LIA_himalaya_oggm` 运行。输出目录必须是尚不存在的新目录。

```powershell
python -m polim_cpu.v2 demo --out polim_cpu/runs/my-cpu
python -m polim_cpu.v2 demo --out polim_cpu/runs/my-gpu --device cuda
python -m polim_cpu.v2 run polim_cpu/v2/examples/arolla/case.json --out polim_cpu/runs/my-arolla --device cuda
python -m polim_cpu.v2 resume --out polim_cpu/runs/my-gpu --steps 20
python -m polim_cpu.v2 report polim_cpu/runs/my-gpu
```

`--steps` 是最终步号，而非额外步数。每次步进会按平流和冰厚正值条件缩短步长，实际模拟时间以报告中的 `clock_years` 为准。CPU/GPU 续算会自动核对原有运行环境和代码；不会重算已完成步。

批量运行允许单个失败后继续：

```powershell
python -m polim_cpu.v2 batch case-a.json case-b.json --out polim_cpu/runs/my-batch --device cuda
```

需要主动重新验收时才运行：

```powershell
python -m polim_cpu.v2 verify --out polim_cpu/runs/my-verification --gpu
```

## 输入

自有剖面 CSV 字段为 `x_m,bed_m,thickness_m,width_m`。允许非等距 x 和变化宽度；x 必须严格递增，活动域冰厚和宽度必须为正。输入不会被自动加厚、截断或重新采样。`geometry.unit_width=true` 时可以省略宽度列，输出仅代表 1 m 宽的流线。

复制 `examples/arolla/case.json` 后替换 `geometry.path` 即可。`parameters.layers` 设置垂向节点数；`basal` 可为 `zero` 或 `linear`，后者的 `beta2` 单位是 Pa·year/m。`flowband=true` 增加横向应变与阻力的参数化。`thermal_feedback=false` 可切换到固定 `glen_a`，用于对照。

表面温度、地热通量和初始底温在 `thermal` 中设置；SMB 单位为 m 冰当量/年。可选 `forcing_csv` 字段引用时间序列，表头为：

```text
time_years,smb_m_ice_per_year,surface_temperature_c
```

时间序列在节点间线性插值；步长会在强迫节点处分段，数据末尾停止，不进行外推。强迫数值和来源哈希保存在运行清单中，续算不重新读取外部强迫。

## 实现边界

- 动量：x–z BP 流线/矩形流带，Q1 单元、2×2 Gauss 积分、Glen 黏度、精确 Newton 切线和能量线搜索。默认上游固定、底部无滑动，其余边界为自然应力条件。
- 热力：`E = Cp(T−Tref)+Lω`；采用压力熔点、冷冰/温冰扩散系数、水平迎风平流、隐式垂向平流和传导。温冰中保留压力熔点温度梯度的物理传导。垂向坐标随冰厚变化，包含一阶 ALE 修正；不含水平热传导。
- 反馈：压力修正后的温度进入分段 Arrhenius 关系；含水率增强软化，用于该关系的含水率上限为 3%。形变热、线性滑动摩擦热和地热进入焓方程。底部超过熔点的热量计为融化；过量含水焓按阈值排出并记账。这是简化排水规则，没有实现 PoLIM 的 MEGM、SHMIP 或子冰川水文系统。
- 演化：迎风通量的冰厚控制体积更新；两端质量通量封闭，时间步显式受限。没有活动冰缘迁移、崩解或冻结增生。融化反馈滞后一时间步。冰厚不采用下限裁剪。
- 验收证明了离散质量守恒、冷冰传导和底部融化基准；没有据此宣称完整二维焓能量守恒或移动冷温界面精度已经验证。
- GPU：Float64 动量装配、Newton 线性求解、焓运算与厚度更新在 CUDA 上运行。网格几何预处理、文件读写和报告在 CPU 上完成。当前 GPU 使用稠密直接解法，分配前检查可用显存；大流线可显式选择 CPU 稀疏解法。小算例启动/装配成本较高，尚无生产规模加速比结论。

## Arolla 数据的处理

几何来自冻结的 PoLIM 上游示例 `geo_arolla.mat`，共 51 个 x 节点。文件中的 `H` 原样保留，包括上游已存在的 0.1 m 偏移；没有再加 MATLAB 驱动中的 `Hmin`。

文件只有 31 个宽度值，无法与 51 个 x 节点可靠对应，因此显式采用单位宽度并关闭流带参数化。温度、SMB 和地热通量是演示参数。报告中的速度不是对 Arolla 的校准预测，体积也不是整条冰川的真实体积。

## 可追溯性与运行环境

每个运行目录保存输入/代码/环境指纹、`code` 代码快照、逐步完整状态、校验和、报告及许可证。损坏的最新断点会被拒绝，不会悄悄退回较旧结果。修改求解代码后应开新输出目录；要精确继续旧结果，应使用该结果的 `code` 快照和原运行环境。

CPU 实测：Python 3.12.4、NumPy 1.26.4、SciPy 1.13.1。GPU 实测：Python 3.10.18、NumPy 2.2.6、Torch 2.11.0+cu128、RTX 5060 Ti。

本项目的 GPU 启动器使用 `D:\Anaconda\envs\SS4\python.exe` 和项目 `.runtime/torch-cu128-v1/site-packages`，在独立进程中隔离运行库，不修改安装文件。迁移环境时可通过 `POLIM_CUDA_PYTHON`、`POLIM_TORCH_SITE` 指定兼容环境。未设置静默 CPU 回退。

原有 LIA 控制文件和主线仿真结果均未改动。LIA 仿真的科学约束仍以项目事实、执行合同和当前状态为准。

## 验收记录与引用

粗网格固定 A 最大相对残差 `6.83×10⁻⁷`；耦合算例最大累计质量误差 `5.96×10⁻⁸ m³`。Arolla CPU/GPU 速度的归一化最大差异 `4.19×10⁻¹²`。CPU 和 GPU 均通过独立进程断点续算的全部状态数组及历史逐位一致检查。温冰 GPU 算例同时产生非零含水率和底部融化。

首次合并验收发现熔点处浮点舍入引起相态切换，修复后重跑相关验收。随后纠正了解析融化基准中热通量的符号，仅复核该项；原始失败记录保留，最终汇总见 `verification-final.json`。

上游：[PoLIM](https://github.com/WangYuzhe/PoLIM-Polythermal-Land-Ice-Model)，冻结提交 `cc5d08541ab67987f7652adcdc7a734530580e82`。本目录沿用上层 `LICENSE` 和 `NOTICE`，适用 PolyForm Noncommercial 1.0.0。

Required Notice: PoLIM (Polythermal Land Ice Model) was developed by Yuzhe Wang and Tong Zhang. Scientific users are requested to cite https://doi.org/10.1016/j.cageo.2020.104526.

焓模型基准背景：[Kleiner et al. (2015), Enthalpy benchmark experiments for numerical ice sheet models](https://tc.copernicus.org/articles/9/217/2015/)。本版的解析列测试不等于完成该论文的全部实验。

## 项目接入准备

已无损导出项目冻结的 191 条矩形拟流线，见 [输入接入报告](../lia-inputs-v1/index.html)。当前所有输入均有无冰区，不能直接进入第二版的处处正冰厚求解器。下一步是活动冰缘、原始单元中心的质量输运，以及项目空间质量平衡和明确的热力输入。此次没有执行 LIA 模拟。

转换器：`python -m polim_cpu.lia_inputs --out polim_cpu/lia-inputs-new`。输出仅含几何，所有气候数组不解码；已存在目录不会覆盖。
