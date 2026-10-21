"""
Step 2 诊断脚本: Benn & Hulton (2010) 冰厚公式验证
==================================================

验证内容:
1. 使用固定参数 (τ=100kPa, F=0.8) 测试核心公式
2. 对比不同τ值的冰厚结果
3. 验证迭代方向（从末端向源头）
4. 检查数值稳定性（判别式<0的情况）
5. 绘制冰厚剖面图

Benn & Hulton (2010) 核心公式:
    d = sqrt((x_i - x_{i-1})² + (y_i - y_{i-1})²)  # 点间距
    b = -(z_{i-1} + z_i)                            # 床面高程项
    c = H_{i-1}*(z_i - (H_{i-1} - z_{i-1})) - 2*d*τ/(F*ρ*g)
    H_i = (-b + sqrt(b² - 4c)) / 2                  # 冰面高程

使用方法:
    python step2_benn_hulton.py
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point, LineString
from pyproj import CRS, Transformer
from shapely.ops import transform
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_id.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'output_dir': "/mnt/d/paleoice_output/step2_diagnostic/",

    # 测试冰川 (使用Step 1中验证过的)
    'test_glacier_id': "RGI60-15.06881",
    'test_flowline_idx': 4986,  # 主流线，最长的那条 (36km)

    # 采样参数
    'sampling_spacing_m': 30,

    # 物理常数
    'rho': 900,  # 冰密度 kg/m³
    'g': 9.81,  # 重力加速度 m/s²

    # 测试参数组
    'test_params': [
        {'tau_kpa': 50, 'F': 0.8, 'label': 'τ=50kPa, F=0.8'},
        {'tau_kpa': 100, 'F': 0.8, 'label': 'τ=100kPa, F=0.8 (默认)'},
        {'tau_kpa': 150, 'F': 0.8, 'label': 'τ=150kPa, F=0.8'},
        {'tau_kpa': 100, 'F': 0.6, 'label': 'τ=100kPa, F=0.6'},
        {'tau_kpa': 100, 'F': 1.0, 'label': 'τ=100kPa, F=1.0'},
    ]
}


# ============================================================================
# 坐标转换函数
# ============================================================================

def get_utm_crs(lon, lat):
    """根据经纬度获取UTM坐标系"""
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg), zone


def transform_geometry(geom, src_crs, dst_crs):
    """转换几何坐标系"""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


# ============================================================================
# DEM采样函数
# ============================================================================

def sample_dem_along_flowline(line_4326, dem_path, utm_crs, spacing_m=30):
    """
    沿流线采样DEM，返回UTM坐标下的结果
    """
    src_crs = CRS.from_epsg(4326)

    # 投影到UTM
    line_utm = transform_geometry(line_4326, src_crs, utm_crs)
    length_m = line_utm.length

    if length_m == 0:
        return None

    # 生成采样点
    num_points = max(int(length_m / spacing_m) + 1, 2)
    distances_m = np.linspace(0, length_m, num_points)

    points_utm = [line_utm.interpolate(d) for d in distances_m]
    xs_utm = np.array([p.x for p in points_utm])
    ys_utm = np.array([p.y for p in points_utm])

    # 转回4326采样DEM
    transformer_back = Transformer.from_crs(utm_crs, src_crs, always_xy=True)

    with rasterio.open(dem_path) as src:
        elevs = []
        for x_utm, y_utm in zip(xs_utm, ys_utm):
            lon, lat = transformer_back.transform(x_utm, y_utm)
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                    elevs.append(val if val > -9999 else np.nan)
                else:
                    elevs.append(np.nan)
            except:
                elevs.append(np.nan)

    return {
        'distances_m': distances_m,
        'xs_utm': xs_utm,
        'ys_utm': ys_utm,
        'bed_elevs': np.array(elevs),
        'length_m': length_m,
        'n_points': num_points
    }


# ============================================================================
# Benn & Hulton (2010) 冰厚计算
# ============================================================================

def ice_thickness_benn_hulton(bed_elevs, xs, ys, tau_pa, F, rho=900, g=9.81):
    """
    Benn & Hulton (2010) 冰厚计算

    从末端（索引0，低海拔）向源头（高海拔）迭代

    公式:
        d = 点间欧氏距离
        b = -(z_{i-1} + z_i)
        c = H_{i-1}*(z_i - (H_{i-1} - z_{i-1})) - 2*d*τ/(F*ρ*g)
        H_i = (-b + sqrt(b² - 4c)) / 2

    Args:
        bed_elevs: 床面高程数组（从末端到源头排列）
        xs, ys: UTM坐标
        tau_pa: 剪应力 (Pa)
        F: 形状因子
        rho: 冰密度 (kg/m³)
        g: 重力加速度 (m/s²)

    Returns:
        dict: 包含冰厚、冰面高程、诊断信息
    """
    n = len(bed_elevs)

    # 初始化
    ice_surface = np.zeros(n)
    thickness = np.zeros(n)
    diagnostics = {
        'discriminant': np.zeros(n),
        'discriminant_negative': np.zeros(n, dtype=bool),
        'd_values': np.zeros(n),
        'b_values': np.zeros(n),
        'c_values': np.zeros(n)
    }

    # 边界条件: 末端冰厚为0
    ice_surface[0] = bed_elevs[0]
    thickness[0] = 0

    # 从末端向源头迭代
    for i in range(1, n):
        # 点间距
        d = np.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2)
        diagnostics['d_values'][i] = d

        if d == 0:
            ice_surface[i] = ice_surface[i - 1]
            thickness[i] = thickness[i - 1]
            continue

        z_prev = bed_elevs[i - 1]
        z_curr = bed_elevs[i]
        H_prev = ice_surface[i - 1]  # 前一点的冰面高程

        # Benn & Hulton 公式
        b = -(z_prev + z_curr)
        c = H_prev * (z_curr - (H_prev - z_prev)) - (2 * d * tau_pa / F) / (rho * g)

        diagnostics['b_values'][i] = b
        diagnostics['c_values'][i] = c

        # 判别式
        discriminant = b ** 2 - 4 * c
        diagnostics['discriminant'][i] = discriminant

        if discriminant >= 0:
            H_curr = (-b + np.sqrt(discriminant)) / 2
        else:
            # 判别式为负：使用前一点的冰面高程
            diagnostics['discriminant_negative'][i] = True
            H_curr = H_prev

        # 计算冰厚
        thick = H_curr - z_curr

        # 处理异常值
        if thick < 0:
            thick = 0
            H_curr = z_curr

        if thick > 2000:  # 限制最大冰厚
            thick = 2000
            H_curr = z_curr + thick

        if np.isnan(thick) or np.isinf(thick):
            thick = thickness[i - 1]
            H_curr = z_curr + thick

        ice_surface[i] = H_curr
        thickness[i] = thick

    return {
        'thickness': thickness,
        'ice_surface': ice_surface,
        'diagnostics': diagnostics
    }


def reverse_flowline_data(sample_data):
    """
    反转流线数据（从源头→末端 变为 末端→源头）
    """
    return {
        'distances_m': sample_data['distances_m'][-1] - sample_data['distances_m'][::-1],
        'xs_utm': sample_data['xs_utm'][::-1],
        'ys_utm': sample_data['ys_utm'][::-1],
        'bed_elevs': sample_data['bed_elevs'][::-1],
        'length_m': sample_data['length_m'],
        'n_points': sample_data['n_points']
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_thickness_comparison(sample_data, results_list, output_path):
    """
    绘制不同参数下的冰厚对比图
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.5, 1, 1])

    distances_km = sample_data['distances_m'] / 1000
    bed_elevs = sample_data['bed_elevs']

    # ========== 1. 冰厚剖面对比 ==========
    ax1 = fig.add_subplot(gs[0, :])

    # 绘制床面
    ax1.fill_between(distances_km, 0, bed_elevs, alpha=0.3, color='brown', label='Bed')
    ax1.plot(distances_km, bed_elevs, 'k-', linewidth=1.5, label='Bed surface')

    # 绘制不同参数的冰面
    colors = plt.cm.Set1(np.linspace(0, 1, len(results_list)))

    for i, (params, result) in enumerate(results_list):
        ice_surface = result['ice_surface']
        ax1.plot(distances_km, ice_surface, '-', color=colors[i],
                 linewidth=2, label=params['label'])

    ax1.set_xlabel('Distance from terminus (km)', fontsize=12)
    ax1.set_ylabel('Elevation (m)', fontsize=12)
    ax1.set_title('Ice Surface Elevation - Parameter Comparison\n'
                  '(Benn & Hulton 2010 method)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, distances_km[-1])

    # ========== 2. 冰厚剖面 ==========
    ax2 = fig.add_subplot(gs[1, 0])

    for i, (params, result) in enumerate(results_list):
        thickness = result['thickness']
        ax2.plot(distances_km, thickness, '-', color=colors[i],
                 linewidth=2, label=params['label'])

    ax2.set_xlabel('Distance from terminus (km)', fontsize=11)
    ax2.set_ylabel('Ice thickness (m)', fontsize=11)
    ax2.set_title('Ice Thickness Profile', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, distances_km[-1])

    # ========== 3. τ敏感性分析 ==========
    ax3 = fig.add_subplot(gs[1, 1])

    # 筛选τ变化的结果 (F=0.8)
    tau_results = [(p, r) for p, r in results_list if p['F'] == 0.8]

    tau_values = [p['tau_kpa'] for p, r in tau_results]
    mean_thickness = [np.mean(r['thickness']) for p, r in tau_results]
    max_thickness = [np.max(r['thickness']) for p, r in tau_results]

    ax3.plot(tau_values, mean_thickness, 'bo-', linewidth=2, markersize=8, label='Mean thickness')
    ax3.plot(tau_values, max_thickness, 'rs-', linewidth=2, markersize=8, label='Max thickness')

    ax3.set_xlabel('Shear stress τ (kPa)', fontsize=11)
    ax3.set_ylabel('Thickness (m)', fontsize=11)
    ax3.set_title('Sensitivity to Shear Stress (F=0.8)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ========== 4. F因子敏感性分析 ==========
    ax4 = fig.add_subplot(gs[2, 0])

    # 筛选F变化的结果 (τ=100kPa)
    f_results = [(p, r) for p, r in results_list if p['tau_kpa'] == 100]

    f_values = [p['F'] for p, r in f_results]
    mean_thickness_f = [np.mean(r['thickness']) for p, r in f_results]
    max_thickness_f = [np.max(r['thickness']) for p, r in f_results]

    ax4.plot(f_values, mean_thickness_f, 'bo-', linewidth=2, markersize=8, label='Mean thickness')
    ax4.plot(f_values, max_thickness_f, 'rs-', linewidth=2, markersize=8, label='Max thickness')

    ax4.set_xlabel('Shape factor F', fontsize=11)
    ax4.set_ylabel('Thickness (m)', fontsize=11)
    ax4.set_title('Sensitivity to Shape Factor (τ=100kPa)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # ========== 5. 数值稳定性检查 ==========
    ax5 = fig.add_subplot(gs[2, 1])

    # 使用默认参数的结果
    default_result = [r for p, r in results_list if p['tau_kpa'] == 100 and p['F'] == 0.8][0]
    diagnostics = default_result['diagnostics']

    # 绘制判别式
    ax5.plot(distances_km, diagnostics['discriminant'], 'b-', linewidth=1, alpha=0.7)
    ax5.axhline(y=0, color='r', linestyle='--', linewidth=1, label='discriminant=0')

    # 标记判别式为负的点
    neg_mask = diagnostics['discriminant_negative']
    if np.any(neg_mask):
        ax5.scatter(distances_km[neg_mask], diagnostics['discriminant'][neg_mask],
                    c='red', s=20, zorder=5, label=f'Negative ({np.sum(neg_mask)} points)')

    ax5.set_xlabel('Distance from terminus (km)', fontsize=11)
    ax5.set_ylabel('Discriminant (b² - 4c)', fontsize=11)
    ax5.set_title('Numerical Stability Check (τ=100kPa, F=0.8)', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(0, distances_km[-1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_detailed_profile(sample_data, result, params, output_path):
    """
    绘制详细的单参数冰厚剖面
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    distances_km = sample_data['distances_m'] / 1000
    bed_elevs = sample_data['bed_elevs']
    ice_surface = result['ice_surface']
    thickness = result['thickness']

    # ========== 1. 完整剖面 ==========
    ax1 = axes[0, 0]

    ax1.fill_between(distances_km, bed_elevs, ice_surface, alpha=0.4, color='lightblue', label='Ice')
    ax1.fill_between(distances_km, 0, bed_elevs, alpha=0.3, color='brown', label='Bed')
    ax1.plot(distances_km, bed_elevs, 'k-', linewidth=1.5)
    ax1.plot(distances_km, ice_surface, 'b-', linewidth=2)

    ax1.set_xlabel('Distance from terminus (km)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_title(f'Cross Section Profile\n{params["label"]}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ========== 2. 冰厚分布 ==========
    ax2 = axes[0, 1]

    ax2.plot(distances_km, thickness, 'b-', linewidth=2)
    ax2.fill_between(distances_km, 0, thickness, alpha=0.3, color='lightblue')

    ax2.axhline(y=np.mean(thickness), color='r', linestyle='--',
                label=f'Mean: {np.mean(thickness):.1f}m')
    ax2.axhline(y=np.max(thickness), color='g', linestyle='--',
                label=f'Max: {np.max(thickness):.1f}m')

    ax2.set_xlabel('Distance from terminus (km)')
    ax2.set_ylabel('Ice thickness (m)')
    ax2.set_title('Ice Thickness Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ========== 3. 冰厚直方图 ==========
    ax3 = axes[1, 0]

    valid_thickness = thickness[thickness > 0]
    ax3.hist(valid_thickness, bins=50, color='steelblue', edgecolor='none', alpha=0.7)
    ax3.axvline(x=np.mean(valid_thickness), color='r', linestyle='--',
                label=f'Mean: {np.mean(valid_thickness):.1f}m')
    ax3.axvline(x=np.median(valid_thickness), color='g', linestyle='--',
                label=f'Median: {np.median(valid_thickness):.1f}m')

    ax3.set_xlabel('Ice thickness (m)')
    ax3.set_ylabel('Count')
    ax3.set_title('Thickness Histogram')
    ax3.legend()

    # ========== 4. 冰面坡度 vs 床面坡度 ==========
    ax4 = axes[1, 1]

    # 计算坡度 (m/km)
    bed_slope = np.gradient(bed_elevs, distances_km)
    ice_slope = np.gradient(ice_surface, distances_km)

    ax4.plot(distances_km, bed_slope, 'brown', linewidth=1.5, alpha=0.7, label='Bed slope')
    ax4.plot(distances_km, ice_slope, 'blue', linewidth=1.5, alpha=0.7, label='Ice surface slope')
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    ax4.set_xlabel('Distance from terminus (km)')
    ax4.set_ylabel('Slope (m/km)')
    ax4.set_title('Surface Slope Comparison\n(Ice surface should be smoother than bed)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# ============================================================================
# 主诊断函数
# ============================================================================

def run_diagnostic():
    """运行Step 2诊断"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 2 诊断: Benn & Hulton (2010) 冰厚公式验证")
    print("=" * 70)

    # ========== 1. 加载数据 ==========
    print("\n[1/5] 加载数据...")

    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])

    # 获取测试流线
    test_fl = flowlines_gdf[
        (flowlines_gdf['MainRGI'] == CONFIG['test_glacier_id']) &
        (flowlines_gdf.index == CONFIG['test_flowline_idx'])
        ]

    if len(test_fl) == 0:
        # 尝试获取该冰川的主流线（最长的）
        glacier_fl = flowlines_gdf[flowlines_gdf['MainRGI'] == CONFIG['test_glacier_id']]
        if len(glacier_fl) > 0:
            # 按长度排序
            glacier_fl = glacier_fl.copy()
            glacier_fl['length'] = glacier_fl.geometry.length
            test_fl = glacier_fl.nlargest(1, 'length')
            CONFIG['test_flowline_idx'] = test_fl.index[0]

    if len(test_fl) == 0:
        print(f"  ✗ 找不到测试流线: {CONFIG['test_glacier_id']}")
        return

    fl_geom = test_fl.geometry.values[0]
    if fl_geom.geom_type == 'MultiLineString':
        fl_geom = max(list(fl_geom.geoms), key=lambda x: x.length)

    print(f"  测试冰川: {CONFIG['test_glacier_id']}")
    print(f"  测试流线索引: {CONFIG['test_flowline_idx']}")

    # ========== 2. 获取UTM坐标系并采样 ==========
    print("\n[2/5] 采样DEM...")

    centroid = fl_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    print(f"  UTM Zone: {utm_zone}N (EPSG:{utm_crs.to_epsg()})")

    sample_data = sample_dem_along_flowline(
        fl_geom, CONFIG['dem_path'], utm_crs, CONFIG['sampling_spacing_m']
    )

    print(f"  流线长度: {sample_data['length_m']:.1f} m")
    print(f"  采样点数: {sample_data['n_points']}")
    print(f"  高程范围: {np.nanmin(sample_data['bed_elevs']):.1f} - "
          f"{np.nanmax(sample_data['bed_elevs']):.1f} m")

    # ========== 3. 反转流线方向 ==========
    print("\n[3/5] 反转流线方向 (源头→末端 变为 末端→源头)...")

    # 检查是否需要反转
    if sample_data['bed_elevs'][0] > sample_data['bed_elevs'][-1]:
        print("  流线方向: 源头→末端，需要反转")
        sample_data = reverse_flowline_data(sample_data)
        print("  ✓ 已反转")
    else:
        print("  流线方向: 末端→源头，无需反转")

    print(f"  反转后起点高程 (末端): {sample_data['bed_elevs'][0]:.1f} m")
    print(f"  反转后终点高程 (源头): {sample_data['bed_elevs'][-1]:.1f} m")

    # ========== 4. 测试不同参数 ==========
    print("\n[4/5] 测试不同参数组合...")

    results_list = []

    for params in CONFIG['test_params']:
        tau_pa = params['tau_kpa'] * 1000  # kPa → Pa
        F = params['F']

        result = ice_thickness_benn_hulton(
            sample_data['bed_elevs'],
            sample_data['xs_utm'],
            sample_data['ys_utm'],
            tau_pa, F,
            CONFIG['rho'], CONFIG['g']
        )

        results_list.append((params, result))

        # 统计
        thickness = result['thickness']
        n_negative_disc = np.sum(result['diagnostics']['discriminant_negative'])

        print(f"  {params['label']}:")
        print(f"    平均冰厚: {np.mean(thickness):.1f} m")
        print(f"    最大冰厚: {np.max(thickness):.1f} m")
        print(f"    判别式<0的点: {n_negative_disc} ({100 * n_negative_disc / len(thickness):.1f}%)")

    # ========== 5. 生成可视化 ==========
    print("\n[5/5] 生成可视化...")

    # 参数对比图
    comparison_path = CONFIG['output_dir'] + 'step2_parameter_comparison.png'
    plot_thickness_comparison(sample_data, results_list, comparison_path)
    print(f"  ✓ {comparison_path}")

    # 默认参数详细图
    default_params = {'tau_kpa': 100, 'F': 0.8, 'label': 'τ=100kPa, F=0.8'}
    default_result = [r for p, r in results_list if p['tau_kpa'] == 100 and p['F'] == 0.8][0]

    detail_path = CONFIG['output_dir'] + 'step2_detailed_profile.png'
    plot_detailed_profile(sample_data, default_result, default_params, detail_path)
    print(f"  ✓ {detail_path}")

    # ========== 生成报告 ==========
    report = []
    report.append("=" * 70)
    report.append("Step 2 诊断报告: Benn & Hulton (2010) 公式验证")
    report.append("=" * 70)

    report.append("\n## 测试配置")
    report.append(f"- 测试冰川: {CONFIG['test_glacier_id']}")
    report.append(f"- 测试流线: {CONFIG['test_flowline_idx']}")
    report.append(f"- 流线长度: {sample_data['length_m']:.1f} m")
    report.append(f"- 采样间距: {CONFIG['sampling_spacing_m']} m")
    report.append(f"- 采样点数: {sample_data['n_points']}")
    report.append(f"- UTM Zone: {utm_zone}N")

    report.append("\n## 参数测试结果")
    report.append("")
    report.append("| 参数 | 平均冰厚 | 最大冰厚 | 判别式<0 |")
    report.append("|------|----------|----------|----------|")

    for params, result in results_list:
        thickness = result['thickness']
        n_neg = np.sum(result['diagnostics']['discriminant_negative'])
        report.append(f"| {params['label']} | {np.mean(thickness):.1f} m | "
                      f"{np.max(thickness):.1f} m | {n_neg} ({100 * n_neg / len(thickness):.1f}%) |")

    report.append("\n## 数值稳定性分析")

    default_diag = default_result['diagnostics']
    n_neg = np.sum(default_diag['discriminant_negative'])

    if n_neg == 0:
        report.append("✓ 默认参数下所有点的判别式≥0，数值稳定")
    else:
        report.append(f"⚠ 默认参数下有 {n_neg} 个点判别式<0")
        report.append("  这些点使用了前一点的冰面高程作为替代")

        # 找出问题点的位置
        neg_indices = np.where(default_diag['discriminant_negative'])[0]
        neg_distances = sample_data['distances_m'][neg_indices] / 1000
        report.append(f"  问题点位置: {neg_distances[:10].tolist()} km ...")

    report.append("\n## 物理合理性检查")

    thickness = default_result['thickness']
    ice_surface = default_result['ice_surface']
    bed_elevs = sample_data['bed_elevs']

    # 冰面应该比床面更平滑
    bed_roughness = np.std(np.gradient(bed_elevs))
    ice_roughness = np.std(np.gradient(ice_surface))

    if ice_roughness < bed_roughness:
        report.append(f"✓ 冰面比床面更平滑 (冰面粗糙度={ice_roughness:.2f}, 床面={bed_roughness:.2f})")
    else:
        report.append(f"⚠ 冰面比床面更粗糙 (冰面粗糙度={ice_roughness:.2f}, 床面={bed_roughness:.2f})")

    # 冰厚范围检查
    if np.max(thickness) < 1500:
        report.append(f"✓ 最大冰厚 {np.max(thickness):.1f}m 在合理范围内 (<1500m)")
    else:
        report.append(f"⚠ 最大冰厚 {np.max(thickness):.1f}m 可能过大 (>1500m)")

    if np.mean(thickness) > 50:
        report.append(f"✓ 平均冰厚 {np.mean(thickness):.1f}m 在合理范围内 (>50m)")
    else:
        report.append(f"⚠ 平均冰厚 {np.mean(thickness):.1f}m 可能过小 (<50m)")

    report.append("\n## 结论")

    all_ok = (n_neg < len(thickness) * 0.05 and
              ice_roughness < bed_roughness and
              50 < np.mean(thickness) < 500)

    if all_ok:
        report.append("✓ Benn & Hulton 公式验证通过，可以进入 Step 3")
    else:
        report.append("⚠ 存在一些问题，请检查上述详情")

    report.append("=" * 70)

    # 保存报告
    report_text = "\n".join(report)
    report_path = CONFIG['output_dir'] + 'step2_diagnostic_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {report_path}")
    print(report_text)

    print("\n" + "=" * 70)
    print("诊断完成! 请将以下文件发给我:")
    print(f"  1. {report_path}")
    print(f"  2. {comparison_path}")
    print(f"  3. {detail_path}")
    print("=" * 70)

    return results_list


if __name__ == "__main__":
    results = run_diagnostic()