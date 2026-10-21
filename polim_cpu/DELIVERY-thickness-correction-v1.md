# 支流冰厚配对错误修正

用户指出不同支流冰厚有问题后，确认上一版将旧 v01/v2 二维冰厚与 anchor-reroute V3 网络混用，违反 `LIA_PROJECT_FACTS.md` 第 168 节的配套输入要求。此前的“源像元一致”和总体积闭合没有证明源与网络配套；将结果称为已完成的真实几何过于肯定。

旧 `spatial-profiles-v1/v2` 已撤回作为后续模型输入。修正版本 `spatial-profiles-v3-paired` 使用同一空间路径、配套 V3/F19 源和第 176 节规定的 `thickness > 1e-6 m` 掩膜。源数组及微小残余值原样保存。

同位置最大冰厚改变量：支流 0 为 173.7 m，支流 1 为 122.0 m，主流线为 32.1 m。例如支流 0 的 1.05 km 处由 187.3 m 改为 13.6 m；支流 1 的 1.8 km 处由 160.2 m 改为 38.3 m。这些不是观测校准结果，而是修正输入配对后从源像元重新读取的值。

修正后的 88,274 个有效冰体像元全部唯一归属，面积和体积与配套源闭合。2,596 个低于正式正冰厚阈值的微小残余像元仍保存在数组中，共约 0.061 m³，不计入正式冰体面积。中心线冰厚与分区等效冰厚继续分开保存，不能混用宽度与冰厚来计算体积。

配对错误已修正，动力学几何仍未验收：尚有 32 个沿程表面升高区间，连接段和分区属于重建假设，未运行动力学或重开已关闭的科学机制。原单伪流线热力试算保持其原合同与存档，不因本次更换空间源而追溯改写。

- [逐支流修正对照](reports/tributary-thickness-correction-v1/index.html)
- [配套空间采样输入](lia-inputs-v1/system-2387/spatial-profiles-v3-paired/input-package.json)
- [错误原因与逐点证据](../validation_data/results/paleo_surface_validation_v0/polim_lia_system2387_paired_source_correction_v1/correction-result.json)
