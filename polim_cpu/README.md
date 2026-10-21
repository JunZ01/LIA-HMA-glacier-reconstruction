当前模型上限：**BP**；不开发、部署或运行 Full-Stokes。[范围约束](MODEL_SCOPE.md) · [修订后的调研与后续路线](reports/higher-order-model-review-bp-only-v1/index.html)。

最新只读诊断：[高阶模型调研与高速区定位](reports/higher-order-model-review-v1/index.html)。源字段与网格中心一致，陡坡已存在于源数据；下一步优先补边界分类与外部标准对照，尚未启动新模拟。

最新进展：[真实原生网格联合速度诊断](v7r1/README.md)已完成，20 轮收敛，残差 3.75e-10。[查看真实速度与汇流应力图](reports/native-global-bp-v1/index.html)。局部速度约 13.7 万 m/年，邻近重建冰面很陡；需先核查动力学几何与参数、边界影响，尚未用于真实演化或科学验收。

此前进展如下（历史记录）：

最新进展：[全局向量 BP 与真实像元网格](v6r1/README.md)已完成首版。合成下游阻力变化通过联合应力影响上游；真实 2387 已静态构网格，尚未计算真实速度。[查看关键结果](reports/global-bp-v1/index.html)。

此前进展如下（历史记录）：

最新进展：[第五版支流局部速度与热力耦合](v5/README.md)已完成合成演化 0.8 年，4 项合并检查通过。[查看演化图](reports/branch-coupling-v1/index.html)。真实网络的汇流应力、垂向重映射与动力学几何尚未完成。

此前进展如下（历史记录）：

当前进展：[多支流冰量与焓输运模块](v4/README.md)已完成，5 项合并检查通过；已静态接入修正后的配套冰厚与 7 个汇流连接。网络动量、垂向热力耦合和真实冰川演化尚未完成。[查看结果总览](index.html)。

空间输入以 `spatial-profiles-v3-paired` 为当前诊断版本；旧 `spatial-profiles-v1/v2` 因冰厚与网络版本混用而撤回作为模型输入。[查看逐支流修正](reports/tributary-thickness-correction-v1/index.html)。

此前[第三版单伪流线热力试算](v3/README.md)和[固定 A 对照](reports/thermal-attribution-v1/index.html)保留原合同，不代表多支流结果。下面是第一版历史说明。

# PoLIM Python CPU 原型 0.1

首版目标是接通 **输入 → 高阶速度 → 冰厚演化 → 输出 → 断点恢复**，然后集中检查。失败案例保留证据，其他案例继续；收敛警告不自动中止开发试算。

## 已实现

- NumPy/SciPy float64；沿流向 x 与垂向 z 的 Blatter–Pattyn 速度方程。
- 地形跟随网格、Glen 非线性流变、垂向速度、应变热诊断。
- 无滑动或线性底摩擦，表面无应力，两端速度为零。
- 恒定宽度；可选原 PoLIM 的恒宽流带侧阻力项。
- 新增守恒有限体积冰厚演化；输入质量平衡采用 m ice/year。
- 每步完整状态与同步历史共同提交，单进程写锁、SHA256、独立进程恢复。
- 批处理隔离失败；离线图表、报告失败不影响已保存的计算。
- `LinearBackend` 求解接口为后续 GPU 留出扩展位置；当前执行后端为 CPU。

## 使用

在项目根目录运行，输出目录须为新目录：

```powershell
python -m polim_cpu demo --out polim_cpu/runs/my-demo --steps 5
python -m polim_cpu resume polim_cpu/runs/my-demo --steps 10
python -m polim_cpu report polim_cpu/runs/my-demo
python -m polim_cpu run polim_cpu/demo.json --out polim_cpu/runs/my-case
python -m polim_cpu batch first.json second.json --out polim_cpu/runs/my-batch
```

`--steps` 是累计目标步数；恢复到已完成步数时不会重算。每步 `dt_years` 是上限，速度 CFL 或冰厚耗尽约束可缩短步长。报告显示实际经过时间。

JSON 示例见 `demo.json`。也支持 `"geometry": {"kind": "csv", "path": "geometry.csv"}`，CSV 列为 `x_m,bed_m,thickness_m,width_m`。路径相对 JSON；x 须等距递增、宽度恒定、厚度严格为正。原型不会用人工冰厚下限修补输入。

`parameters` 可设置 `layers`、`glen_a`、`glen_n`、`basal`（`zero` 或 `linear`）、`beta2`、`flowband`、`relaxation`、`max_iterations`、`relative_tolerance` 等。`glen_a` 单位是 Pa^-n/year；粘度是 Pa year；速度是 m/year。请勿将秒制 Glen 参数直接填入。

每个 `step-000000/` 等目录包含完整数组与 `checkpoint.json`。未完成的 `.pending-*` 不会作为有效检查点。修改代码、配置或数值依赖版本后，旧运行不能按“完全相同计算”续跑；已有结果仍可离线阅读。报告可随时重建。

## 当前边界

这是 PoLIM classic 的**受限移植与新控制器**。尚未实现焓/温度演化、冷温转换、水文、Coulomb 滑动、变宽流带、移动冰缘、崩解、GPU 或 OGGM 真实冰川适配。应变热仅作诊断，没有反馈到流变温度。

速度迭代采用本原型的有界求解策略，不宣称与上游完整时间驱动逐步等价；冰厚演化也不是原版扩散演化算法。原版 `set_staggered_grid` 全域循环未计算末列坐标导数的行为被显式保留，以便核对移植。上游固定几何速度矩阵和诊断可以直接对照，完整热力模型不能据此宣称已经验证。

非收敛但有限的解会明确标记，并允许继续原型迭代。科学使用时必须另外判断收敛与适用性。真实 LIA 模拟仍需遵守项目科学与执行合同；本次仅运行独立合成案例，没有修改生产控制器、已有计算或接受账本。

## 来源与修改

参考：[Wang 与 Zhang 的 PoLIM 仓库](https://github.com/WangYuzhe/PoLIM-Polythermal-Land-Ice-Model)，提交 `cc5d08541ab67987f7652adcdc7a734530580e82`，`PoLIM_classic`。原始压缩包和源码保存在 `reference/`。

当前 classic 使用 **PolyForm Noncommercial 1.0.0**，见 `LICENSE`；开发者声明见 `NOTICE`。早先设计稿中 MIT 与旧文件路径的描述不适用于本次冻结版本。

依据 `solver_u_core.m` 移植内部、表面及线性底部模板；`build_stencils.py` 可重新生成 `_stencils.py`。网格、流变和垂向速度据对应 MATLAB 文件改写。Python 对象接口、稀疏求解接口、迭代控制、有限体积演化、存储、恢复、批处理与报告为本原型新增工作。

研究使用请引用 Wang et al. (2020), Computers & Geosciences 141, 104526，DOI: [10.1016/j.cageo.2020.104526](https://doi.org/10.1016/j.cageo.2020.104526)。2021 年勘误正文尚未取得；当前实现以固定源码为依据，不声称完成论文及勘误的独立复现。

## 集中验证

`verify.py` 在开发完成后统一检查 MATLAB 固定粘度矩阵/速度/流变/垂向速度，静止平板、封闭域质量、非线性收敛、独立进程恢复、失败隔离和报告失败。它仅使用合成输入。实际结论见交付的 `runs/` 下验证报告，不以脚本存在代替检查结果。

```powershell
python -m polim_cpu.verify prepare --out polim_cpu/runs/oracle-new
# 在 MATLAB 中 addpath(reference/classic)，进入 oracle-new 并运行 oracle
python -m polim_cpu.verify check --oracle polim_cpu/runs/oracle-new --out polim_cpu/runs/check-new
```

后续迭代优先在这个可运行版本上补齐非线性稳定性与热力耦合；只重跑与修改有关的检查。


## BP boundary benchmark delivery v8

The BP-only boundary interface and combined standard suite are complete. See [the current report](reports/bp-boundary-benchmarks-v1/index.html) and [v8 interface notes](v8/README.md). Six fixed-geometry solves and independent exact restart completed; real2387 high velocity remains unresolved. Nine-node native-grid execution requires block assembly and a new scoped plan. Existing native-global-BP and literature-review reports remain archived.


## Runnable integrated model v9

The native2387 one-hour BP/thickness/thermal evolution is complete:7 saved coupled states,18 transport substeps,427.23seconds runtime. Open_Model.cmd is the entry; existing completed results open without repeating computation. See [the interactive timeline](reports/integrated-model-v1/index.html) and [model notes](v9/README.md). Volume balance and independent synthetic coupled restart passed. The open computational boundary and uncalibrated velocities are explicitly shown in the prototype.


## 当前入口：BP 材料流变修复版 v10

Open_Model.cmd 打开已完成的一小时修复结果，不重复计算。修正一维反演有效 A 作为 -10°C 材料参考值的额外放大，采用文档明确的 GPBLD 参数，增强系数为 1。相同初始冰厚/冰温最高速度从 509245 降为 2997 m/年；一小时最大局部冰厚改变量从 66.796 降为 0.772 m。无速度截断。原始几何仍需物理核实，本次修复不等于观测验证。结果：reports/material-rheology-v1/index.html。


## 几何对照 v11 已完成，默认计算版本仍为 v10

现代冰缘拼接处的局部冰床突跳由34.749m变为10.088m，原始地形为8.981m。63m源尺度转接锁住所有支流采样支撑，H、V、网格连接和初始E完全不变。一次一小时试算最高初始速度2997.451→3110.204m/年，增加3.76%，说明此转接没有解决剩余高速。保留试验图与结果，不把假设性修正当作已验证真实冰床。根页面提供当前模型与几何对照两个入口；Open_Model.cmd保持v10。
