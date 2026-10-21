# 第三版交付

已实现活动冰缘、完整原始网格的质量输运、空间质量平衡、温度与含水率反馈，以及 CPU/GPU 精确续算。系统 2387 已在 GPU 上完成 0.1 年、52 步试算，保存 53 个完整断点；4094 格原始物理域全部保留。

- [真实剖面报告](runs/lia-2387-v3-pilot-v1/index.html)
- [第三版使用说明](v3/README.md)
- [7 项合并检查结果](runs/v3-fixture-final/verification.json)
- [项目终态证据](../validation_data/results/paleo_surface_validation_v0/polim_lia_v3_system2387_pilot_v1/terminal.json)

最大累计质量误差为 2.02×10⁻⁷ m³。最终最大模拟流速约 13.4 km/年，仍需物理核查，不代表观测校准或 LIA 科学验收。原始结果如实保留，未为降低速度调整参数并覆盖重算。

采用了用户确认的气候冰温初始化与 0.05 W/m² 地热通量假设。研究分支已按执行合同登记和结束；主线运行字段及接受数 58/191 均保持不变。第二版记录保留为 `DELIVERY-v2.md`、`delivery-evidence-v2.json` 和 `index-v2.html`。
