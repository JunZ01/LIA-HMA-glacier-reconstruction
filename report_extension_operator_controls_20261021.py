from pathlib import Path
import json
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('H:/ToYangxin/result_yx/extension_operator_controls_20261021_v1')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    controls = pd.read_csv(ROOT / 'operator_volume_controls.csv')
    bins = pd.read_csv(ROOT / 'operator_bias_by_constraint_distance.csv')
    transfer = pd.read_csv(ROOT / 'poisson_q_leave_one_out.csv')
    boundary = pd.read_csv(ROOT / 'boundary_sensitivity_v1/boundary_condition_sensitivity.csv')
    order = [3632, 3131, 2906]
    labels = {3632: '西喜马拉雅3632', 3131: '中喜马拉雅3131', 2906: '东喜马拉雅2906'}
    results = []
    for uid in order:
        sub = controls[(controls.lia_unique_id == uid) & (controls.boundary_exclusion_m == 100)]
        old = sub[sub.operator == 'legacy'].iloc[0]
        a = sub[sub.operator == 'A_surface'].iloc[0]
        b = boundary[(boundary.lia_unique_id == uid) & (boundary.boundary_exclusion_m == 100)].iloc[0]
        c = transfer[(transfer.lia_unique_id == uid) & (transfer.boundary_exclusion_m == 100)].iloc[0]
        seq = bins[(bins.lia_unique_id == uid) & (bins.boundary_exclusion_m == 100) & (bins.operator == 'legacy')]
        assert np.all(np.diff(seq.relative_volume_bias_percent) < 0)
        clip = 100 * a.clipped_volume_km3 / a.reference_volume_on_target_grid_km3
        results.append(dict(system_id=uid, region=labels[uid], legacy_bias_percent=float(old.volume_bias_percent),
            A_bias_percent=float(a.volume_bias_percent), A_raw_unclipped_bias_percent=float(a.volume_bias_percent - clip),
            A_raw_surface_rmse_m=float(a.surface_rmse_vs_dem_m), A_realized_surface_rmse_m=float(a.realized_surface_rmse_vs_dem_m),
            A_negative_raw_cells_percent=100 * float(a.negative_raw_thickness_fraction), A_clipped_volume_percent_of_reference=float(clip),
            boundary_direct_volume_percent=float(b.direct_boundary_share_percent), boundary_total_effect_percentage_points=float(b.full_boundary_effect_percentage_points),
            boundary_counterfactual_bias_percent=float(b.reference_boundary_bias_percent), legacy_deficit_removed_by_reference_boundary_percent=float(b.fraction_of_legacy_deficit_removed_by_boundary_change_percent),
            C_leave_one_out_bias_percent=float(c.volume_bias_percent), legacy_bias_monotonic_by_distance=True,
            native_1d_2d_volume_difference_percent=100 * (float(a.native_modern_volume_km3) / float(a.oggm_1d_inversion_volume_km3) - 1)))
    report = dict(status='completed', scope='Modern-reference operator diagnostic, not independent validation or LIA reconstruction', tests_passed=9,
        synthetic_cases_reproduced=7, real_systems=3, gaps_m=[0, 100], production_promoted=False, production_sources_unchanged=True,
        new_dynamic_runs=0, new_LIA_reconstructions=0, results_100m=results)
    (ROOT / 'reviewed_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pd.DataFrame(results).to_csv(ROOT / 'reviewed_results_100m.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    x = np.arange(3)
    for offset, key, label, color in [(-1, 'legacy_bias_percent', '旧调和冰厚', '#a33a36'), (0, 'boundary_counterfactual_bias_percent', '参考边缘值：归因对照', '#de9a36'), (1, 'A_bias_percent', 'A：调和冰面减冰床', '#217f94')]:
        values = [r[key] for r in results]
        bars = axes[0].bar(x + offset * .25, values, .25, label=label, color=color)
        axes[0].bar_label(bars, labels=[f'{v:.1f}' for v in values], padding=3, fontsize=9)
    axes[0].set_xticks(x, [labels[u] for u in order], fontsize=9)
    axes[0].axhline(0, color='gray', lw=.8)
    axes[0].set_ylim(-43, 7)
    axes[0].set_ylabel('相对现代参考体积偏差（%）')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[0].set_title('相同范围与流线约束；100 m排除方案')
    for uid, color in zip(order, ['#4166aa', '#9d467c', '#397b55']):
        seq = bins[(bins.lia_unique_id == uid) & (bins.boundary_exclusion_m == 100) & (bins.operator == 'legacy')]
        axes[1].plot(np.arange(len(seq)), seq.relative_volume_bias_percent, 'o-', label=labels[uid], color=color)
    axes[1].set_xticks(np.arange(6), ['0–100', '100–200', '200–400', '400–800', '800–1600', '≥1600'], rotation=30)
    axes[1].set_xlabel('距最近流线约束格点的距离（m）')
    axes[1].set_ylabel('分箱体积偏差（%）')
    axes[1].set_title('旧扩展：三个案例均随距离增大而偏薄')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=.2)
    fig.savefig(ROOT / 'operator_comparison.png', dpi=180)
    plt.close(fig)
    def table(headers, rows):
        return '<table><tr>' + ''.join('<th>' + html.escape(h) + '</th>' for h in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows) + '</table>'
    t1 = table(['案例', '旧体积偏差', 'A体积偏差', 'A原始冰面RMSE', 'A负冰厚格点', 'C留一转移偏差'], [[r['region'], f"{r['legacy_bias_percent']:.2f}%", f"{r['A_bias_percent']:.2f}%", f"{r['A_raw_surface_rmse_m']:.1f} m", f"{r['A_negative_raw_cells_percent']:.2f}%", f"{r['C_leave_one_out_bias_percent']:.2f}%"] for r in results])
    t2 = table(['案例', '边缘直接体积占比', '边缘改变总效应（百分点）', '原体积缺口消除比例', '改变边缘后偏差'], [[r['region'], f"{r['boundary_direct_volume_percent']:.2f}%", f"{r['boundary_total_effect_percentage_points']:.2f}", f"{r['legacy_deficit_removed_by_reference_boundary_percent']:.1f}%", f"{r['boundary_counterfactual_bias_percent']:.2f}%"] for r in results])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>扩展算子补丁：真实三冰川诊断</title><style>body{font:16px/1.8 'Microsoft YaHei',sans-serif;max-width:1150px;margin:35px auto;padding:0 24px;color:#23323e}h1{font-size:28px}h2{font-size:21px;margin-top:30px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #ccd5dd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f6}.note{background:#fff1d7;padding:15px;border-left:4px solid #c58b20}img{width:100%}</style><h1>扩展算子补丁：真实三冰川诊断</h1><p>2026-10-21。补丁已在本地独立研究分支应用；原生产代码保持。</p><p class="note"><b>结论：</b>A大幅改善现代参考总体积的一致性，但仍有23–33 m的原始冰面插值RMSE与3.6%–10.7%的负冰厚截断，不能仅凭总体积误差直接推广生产。C的跨冰川常数q转移失败。旧偏差同时包含边缘条件向内部传播和厚度空间形状未恢复，不能全部归为“横断面呈三角形”。</p><h2>执行与复现</h2><p>本机oggm_env2运行：原8项测试及1项负高程回归测试共9项通过；7组合成体积偏差与附件最大差异约8.3×10⁻¹³个百分点。真实诊断采用原先冻结的3131、2906、3632，原网格和流线采样位置，排除距离0/100 m两方案。主诊断约26.5秒，无新小冰期重建、气候或动力模拟。</p><h2>真实现代对照：100 m方案</h2>''' + t1 + '''<p>A的冰面RMSE以非边缘内部格点、相对现代DEM计算；负冰厚比例以全冰川掩膜格点为分母。负值被截为零会提高体积，原始未截断偏差另见reviewed_results_100m.csv。参考是模型估计，不是实测冰厚真值。</p><img src="operator_comparison.png"><h2>边缘条件的直接损失与内部传播</h2><p>边缘格点上的参考体积只占0.41%–1.16%，但这不是边缘置零的全部作用。补充6次对照只把栅格边缘厚度由零改为参考场在这些格点的值，其余拉普拉斯方程和流线约束保持。这是归因对照，并非连续物理边界的生产建议。</p>''' + t2 + '''<p>三个案例中，边缘条件及其内部传播解释了约35%–53%的旧体积缺口；仍残留约14%–24%的低估。该分解依赖固定网格、参考场和算子，不能推广为191条的统一比例。距离分箱偏差在三个案例均单调变负，支持内部填充不足；仅凭距离趋势不能区分边缘传播和横向形状。</p><h2>A的适用性边界</h2><p>A求解ΔS=0，再以H=max(S−B,0)计算厚度。未截断区域等价于−ΔH=ΔB：横向厚度结构由冰床提供。现代B=DEM−H_ref，A会通过冰床继承参考厚度形状，因此现代对照属于一致性检验，并非独立验证冰床或历史厚度。合成冰床由真值构造且冰面平滑，A表现优异符合这些前提。</p><p>约3.6%–10.7%的格点发生负冰厚截断，存在内部零厚度与总体积误差抵消。尚未检验：统一冰床后的完整现代PP/τ/网络输出、小冰期形态转移、退出区DEM时相及冰雪/湖泊覆盖、独立冰厚观测。未将A直接接入191条生产，未产生修复后的829条区域统计。</p><h2>参考场来源与形状因子更正</h2><p>本地OGGM代码确认：按相近高程选流线厚度并作反距离加权，再乘边缘距离的0.25次方和坡度修正，执行平滑，最后整体缩放到一维反演总体积。它并不逐高程带强制保体积；字段名F19不代表原始F19空间场。一维与分配二维场有结构性差异，不宜称为独立观测误差。</p><p>F=0.8不能单独确定抛物线横断面。项目流线方程中F以τ/F组合出现，代码未据此唯一指定二维横断面；固定F与特定截面形状不能直接等同。</p><h2>补丁审查</h2><p>修正重复约束最大值初始为0导致负高程错误冲突的问题，增加回归测试；修正固定代码根目录，禁止覆盖输出，记录原始及截断后冰面误差。原核心求解器、现代率定和小冰期生产路径未作修改。</p><p><a href="operator_volume_controls.csv">主对照</a> · <a href="operator_bias_by_constraint_distance.csv">距离分箱</a> · <a href="poisson_q_leave_one_out.csv">q留一法</a> · <a href="boundary_sensitivity_v1/boundary_condition_sensitivity.csv">边缘归因</a> · <a href="reviewed_results.json">审阅汇总</a></p></html>'''
    (ROOT / 'report.html').write_text(page, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True), flush=True)

if __name__ == '__main__':
    main()
