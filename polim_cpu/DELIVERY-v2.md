# 第二版交付

已完成本轮的三项工作：新的粗网格非线性求解器、焓与温度反馈、GPU 和真实剖面输入。全部已有实际运行结果。

- [打开结果总览](index.html)
- [使用说明](v2/README.md)
- [Arolla 真实剖面 GPU 报告](runs/acceptance-v2-2/arolla-gpu/index.html)
- [粗网格热力耦合 GPU 报告](runs/acceptance-v2-2/coupled-gpu/index.html)
- [温冰、含水率与融化 GPU 报告](runs/acceptance-v2-2/warm-gpu/index.html)
- [最终验收：16 项通过](runs/acceptance-v2-2/verification-final.json)

粗网格收敛、解析热传导和融化、质量守恒、CPU/GPU 一致性、断点续算、错误隔离和报告重建均已覆盖。CPU/GPU 续算与连续运行的完整状态及历史逐位一致。验收中发现的问题及修复证据保留在结果目录中。

使用新的有限元离散，尚未完成 MATLAB 全模型等价验证；真实剖面示例采用单位宽度与演示热力参数，不能作为校准预测或 LIA 科学结论。原有 LIA 事实与当前状态文件哈希保持一致，主线仿真没有启动。

第一版交付记录保留在 `DELIVERY-v1.md` 和 `delivery-evidence-v1.json`。
