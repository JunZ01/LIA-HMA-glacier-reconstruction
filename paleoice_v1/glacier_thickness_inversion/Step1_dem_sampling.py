"""
Step 1 诊断脚本: DEM采样 + 流线方向验证
========================================
包含动态UTM投影功能

检查内容:
1. 动态UTM投影（根据冰川中心点自动选择UTM带）
2. 沿流线采样DEM高程
3. 验证流线方向（末端应在低处）
4. 绘制高程剖面图

使用方法:
    python step1_dem_sampling.py
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, box
from shapely.ops import transform
from pyproj import CRS, Transformer
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置路径
# ============================================================================
CONFIG = {
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_id.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'output_dir': "/mnt/d/paleoice_output/step1_diagnostic/",

    # 测试冰川数量
    'n_test_glaciers': 2,

    # 采样参数
    'sampling_spacing_m': 30,  # 采样间距（米）
}


# ============================================================================
# 动态UTM投影函数
# ============================================================================

def get_utm_crs(lon, lat):
    """
    根据经纬度获取对应的UTM坐标系

    Args:
        lon: 经度
        lat: 纬度

    Returns:
        pyproj.CRS: UTM坐标系
    """
    # 计算UTM带号
    zone = int((lon + 180) / 6) + 1

    # 北半球或南半球
    if lat >= 0:
        epsg = 32600 + zone  # 北半球: 326XX
    else:
        epsg = 32700 + zone  # 南半球: 327XX

    return CRS.from_epsg(epsg), zone


def get_glacier_utm_crs(glacier_geom):
    """
    根据冰川几何获取最佳UTM坐标系

    Args:
        glacier_geom: 冰川几何（EPSG:4326）

    Returns:
        pyproj.CRS, zone_number
    """
    # 获取冰川中心点
    centroid = glacier_geom.centroid
    return get_utm_crs(centroid.x, centroid.y)


def transform_geometry(geom, src_crs, dst_crs):
    """
    转换几何对象的坐标系

    Args:
        geom: Shapely几何对象
        src_crs: 源坐标系
        dst_crs: 目标坐标系

    Returns:
        转换后的几何对象
    """
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def sample_dem_with_projection(line_4326, dem_path, target_crs, spacing_m=30):
    """
    沿流线采样DEM高程（带坐标投影）

    流程:
    1. 将流线从4326投影到UTM
    2. 在UTM下按米计算采样点
    3. 将采样点转回4326
    4. 从DEM（4326）中采样高程

    Args:
        line_4326: EPSG:4326下的流线
        dem_path: DEM文件路径
        target_crs: 目标UTM坐标系
        spacing_m: 采样间距（米）

    Returns:
        dict: 包含采样结果的字典
    """
    src_crs = CRS.from_epsg(4326)

    # 1. 将流线投影到UTM
    line_utm = transform_geometry(line_4326, src_crs, target_crs)

    # 2. 计算流线长度（米）
    length_m = line_utm.length

    if length_m == 0:
        return None

    # 3. 在UTM下生成采样点
    num_points = max(int(length_m / spacing_m) + 1, 2)
    distances_m = np.linspace(0, length_m, num_points)

    points_utm = [line_utm.interpolate(d) for d in distances_m]

    # 4. 将采样点转回4326
    transformer_back = Transformer.from_crs(target_crs, src_crs, always_xy=True)
    points_4326 = []
    for p in points_utm:
        lon, lat = transformer_back.transform(p.x, p.y)
        points_4326.append((lon, lat))

    # 5. 从DEM采样高程
    with rasterio.open(dem_path) as src:
        elevs = []
        xs_utm = [p.x for p in points_utm]
        ys_utm = [p.y for p in points_utm]

        for lon, lat in points_4326:
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
        'elevs': np.array(elevs),
        'xs_utm': np.array(xs_utm),
        'ys_utm': np.array(ys_utm),
        'points_4326': points_4326,
        'length_m': length_m,
        'n_points': num_points
    }


# ============================================================================
# 流线方向分析
# ============================================================================

def analyze_flowline_direction(sample_result):
    """
    分析流线方向

    Benn & Hulton方法要求: 从末端（低海拔）向源头（高海拔）迭代
    因此流线的起点应该在末端（低处）

    Args:
        sample_result: sample_dem_with_projection的返回结果

    Returns:
        dict: 方向分析结果
    """
    elevs = sample_result['elevs']
    valid_mask = ~np.isnan(elevs)

    if np.sum(valid_mask) < 2:
        return {
            'valid': False,
            'reason': '有效高程点不足'
        }

    valid_elevs = elevs[valid_mask]
    valid_distances = sample_result['distances_m'][valid_mask]

    # 起点和终点高程
    start_elev = valid_elevs[0]
    end_elev = valid_elevs[-1]

    # 最低点和最高点
    min_elev = np.min(valid_elevs)
    max_elev = np.max(valid_elevs)
    min_idx = np.argmin(valid_elevs)
    max_idx = np.argmax(valid_elevs)

    # 判断方向
    # 正确方向: 起点在末端（低处），终点在源头（高处）
    # 即: start_elev < end_elev

    if start_elev < end_elev:
        direction_correct = True
        direction_desc = "正确 (起点=末端, 低→高)"
        needs_reverse = False
    else:
        direction_correct = False
        direction_desc = "需反转 (起点=源头, 高→低)"
        needs_reverse = True

    # 计算高程变化趋势（线性拟合斜率）
    slope, _ = np.polyfit(valid_distances, valid_elevs, 1)

    return {
        'valid': True,
        'start_elev': start_elev,
        'end_elev': end_elev,
        'min_elev': min_elev,
        'max_elev': max_elev,
        'elev_range': max_elev - min_elev,
        'min_idx': min_idx,
        'max_idx': max_idx,
        'direction_correct': direction_correct,
        'direction_desc': direction_desc,
        'needs_reverse': needs_reverse,
        'slope_m_per_m': slope,  # 高程变化率 (m/m)
        'n_valid_points': len(valid_elevs)
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_elevation_profile(sample_result, direction_result, glacier_id, flowline_idx, ax):
    """
    绘制高程剖面图
    """
    distances_km = sample_result['distances_m'] / 1000
    elevs = sample_result['elevs']

    # 绘制高程剖面
    ax.plot(distances_km, elevs, 'b-', linewidth=2, label='bed_elevation')

    # 标记起点和终点
    valid_mask = ~np.isnan(elevs)
    valid_distances_km = distances_km[valid_mask]
    valid_elevs = elevs[valid_mask]

    if len(valid_elevs) > 0:
        # 起点（绿色）
        ax.scatter([valid_distances_km[0]], [valid_elevs[0]],
                   c='green', s=100, zorder=5, label=f'Startpoint ({valid_elevs[0]:.0f}m)')
        # 终点（红色）
        ax.scatter([valid_distances_km[-1]], [valid_elevs[-1]],
                   c='red', s=100, zorder=5, label=f'Terminal ({valid_elevs[-1]:.0f}m)')

        # 最低点（如果不是起点）
        min_idx = direction_result['min_idx']
        if min_idx != 0 and min_idx != len(valid_elevs) - 1:
            ax.scatter([valid_distances_km[min_idx]], [valid_elevs[min_idx]],
                       c='blue', s=80, marker='v', zorder=5,
                       label=f'Most_Low_Point ({valid_elevs[min_idx]:.0f}m)')

    # 添加方向箭头
    if len(valid_elevs) > 5:
        mid_idx = len(valid_distances_km) // 2
        arrow_start = mid_idx - 2
        arrow_end = mid_idx + 2

        if direction_result['direction_correct']:
            color = 'green'
            text = 'Flowline_direction →'
        else:
            color = 'red'
            text = '← Flowline_direction (Invert required)'

        ax.annotate('',
                    xy=(valid_distances_km[arrow_end], valid_elevs[arrow_end]),
                    xytext=(valid_distances_km[arrow_start], valid_elevs[arrow_start]),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))

        ax.text(0.02, 0.98, text, transform=ax.transAxes,
                fontsize=10, va='top', color=color, fontweight='bold')

    # 设置标签和标题
    ax.set_xlabel('Distance along the line (km)')
    ax.set_ylabel('Elevation (m)')

    status = '✓' if direction_result['direction_correct'] else '✗ invert_required'
    ax.set_title(f"{glacier_id} | flowline {flowline_idx}\n"
                 f"Length: {sample_result['length_m']:.0f}m | "
                 f"elev_range: {direction_result['elev_range']:.0f}m | {status}")

    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_glacier_map(glacier_geom, flowlines, sample_points, glacier_id, utm_zone, ax):
    """
    绘制冰川平面图
    """
    # 绘制冰川边界
    if hasattr(glacier_geom, 'exterior'):
        x, y = glacier_geom.exterior.xy
        ax.fill(x, y, alpha=0.3, color='lightblue', label='LIA_Boundary')
        ax.plot(x, y, 'b-', linewidth=2)

    # 绘制流线
    colors = plt.cm.Set1(np.linspace(0, 1, len(flowlines)))
    for i, (fl_idx, fl_geom) in enumerate(flowlines):
        if fl_geom.geom_type == 'MultiLineString':
            for part in fl_geom.geoms:
                x, y = part.xy
                ax.plot(x, y, color=colors[i], linewidth=1.5)
        else:
            x, y = fl_geom.xy
            ax.plot(x, y, color=colors[i], linewidth=1.5, label=f'Flowline {fl_idx}')

    # 绘制采样点
    for points in sample_points:
        if points is not None:
            lons = [p[0] for p in points]
            lats = [p[1] for p in points]
            ax.scatter(lons, lats, s=5, alpha=0.5, c='red')

    ax.set_xlabel('longtitude (m)')
    ax.set_ylabel('latitude (m)')
    ax.set_title(f'{glacier_id}\nUTM Zone: {utm_zone}N')
    ax.legend(loc='best', fontsize=8)
    ax.set_aspect('equal')


# ============================================================================
# 主诊断函数
# ============================================================================

def run_diagnostic():
    """运行Step 1诊断"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 1 诊断: DEM采样 + 流线方向验证")
    print("=" * 70)

    # ========== 1. 加载数据 ==========
    print("\n[1/5] 加载数据...")

    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])

    print(f"  冰川数: {len(glacier_gdf)}")
    print(f"  流线数: {len(flowlines_gdf)}")

    # ========== 2. 选择测试冰川 ==========
    print("\n[2/5] 选择测试冰川...")

    # 找出有流线的冰川，按面积排序
    glacier_with_flowlines = []

    for idx, row in glacier_gdf.iterrows():
        gid = row['OrigRGIId']
        fl = flowlines_gdf[flowlines_gdf['MainRGI'] == gid]

        if len(fl) > 0:
            area_km2 = row.geometry.area * 111 ** 2  # 粗略转换为km²
            glacier_with_flowlines.append({
                'id': gid,
                'area_km2': area_km2,
                'n_flowlines': len(fl),
                'geometry': row.geometry
            })

    # 按面积排序
    glacier_with_flowlines.sort(key=lambda x: x['area_km2'], reverse=True)

    # 选择测试冰川: 1个大的 + 1个中等的
    n_test = CONFIG['n_test_glaciers']
    test_glaciers = []

    if len(glacier_with_flowlines) >= 1:
        test_glaciers.append(glacier_with_flowlines[0])  # 最大
    if len(glacier_with_flowlines) >= 2:
        mid_idx = len(glacier_with_flowlines) // 2
        test_glaciers.append(glacier_with_flowlines[mid_idx])  # 中等

    print(f"  选择了 {len(test_glaciers)} 个测试冰川:")
    for g in test_glaciers:
        print(f"    - {g['id']}: Area≈{g['area_km2']:.2f} km², {g['n_flowlines']} flowlines")

    # ========== 3. 对每个冰川进行分析 ==========
    print("\n[3/5] 分析流线...")

    all_results = []

    for glacier_info in test_glaciers:
        gid = glacier_info['id']
        glacier_geom = glacier_info['geometry']

        print(f"\n  --- {gid} ---")

        # 获取UTM坐标系
        utm_crs, utm_zone = get_glacier_utm_crs(glacier_geom)
        print(f"    UTM Zone: {utm_zone}N (EPSG:{utm_crs.to_epsg()})")

        # 获取该冰川的流线
        gl_flowlines = flowlines_gdf[flowlines_gdf['MainRGI'] == gid]

        glacier_results = {
            'glacier_id': gid,
            'utm_zone': utm_zone,
            'utm_epsg': utm_crs.to_epsg(),
            'glacier_geom': glacier_geom,
            'flowlines': [],
            'sample_results': [],
            'direction_results': []
        }

        for fl_idx, fl_row in gl_flowlines.iterrows():
            fl_geom = fl_row.geometry

            if fl_geom is None or fl_geom.is_empty:
                continue

            # 如果是MultiLineString，取最长的
            if fl_geom.geom_type == 'MultiLineString':
                parts = list(fl_geom.geoms)
                fl_geom = max(parts, key=lambda x: x.length)

            print(f"    Flowline {fl_idx}: ", end='')

            # 采样DEM
            sample_result = sample_dem_with_projection(
                fl_geom,
                CONFIG['dem_path'],
                utm_crs,
                CONFIG['sampling_spacing_m']
            )

            if sample_result is None:
                print("采样失败")
                continue

            # 分析方向
            direction_result = analyze_flowline_direction(sample_result)

            if not direction_result['valid']:
                print(f"无效 - {direction_result['reason']}")
                continue

            print(f"长度={sample_result['length_m']:.0f}m, "
                  f"高差={direction_result['elev_range']:.0f}m, "
                  f"方向={'✓' if direction_result['direction_correct'] else '✗需反转'}")

            glacier_results['flowlines'].append((fl_idx, fl_geom))
            glacier_results['sample_results'].append(sample_result)
            glacier_results['direction_results'].append(direction_result)

        all_results.append(glacier_results)

    # ========== 4. 生成可视化 ==========
    print("\n[4/5] 生成可视化...")

    # 为每个冰川生成一页图
    for glacier_results in all_results:
        gid = glacier_results['glacier_id']
        n_flowlines = len(glacier_results['flowlines'])

        if n_flowlines == 0:
            continue

        # 创建图形: 左边是平面图，右边是高程剖面
        n_cols = min(n_flowlines + 1, 4)
        fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

        if n_cols == 1:
            axes = axes.reshape(2, 1)

        # 平面图
        ax_map = axes[0, 0]
        plot_glacier_map(
            glacier_results['glacier_geom'],
            glacier_results['flowlines'],
            [sr['points_4326'] for sr in glacier_results['sample_results']],
            gid,
            glacier_results['utm_zone'],
            ax_map
        )

        # 高程剖面图
        for i, (fl_tuple, sample_result, direction_result) in enumerate(zip(
                glacier_results['flowlines'],
                glacier_results['sample_results'],
                glacier_results['direction_results']
        )):
            fl_idx, fl_geom = fl_tuple

            if i + 1 < n_cols:
                ax_profile = axes[0, i + 1]
            else:
                ax_profile = axes[1, i + 1 - n_cols] if i + 1 - n_cols < n_cols else None

            if ax_profile is not None:
                plot_elevation_profile(
                    sample_result, direction_result,
                    gid, fl_idx, ax_profile
                )

        # 隐藏多余的子图
        for i in range(n_flowlines + 1, 2 * n_cols):
            row = i // n_cols
            col = i % n_cols
            if row < 2 and col < n_cols:
                axes[row, col].set_visible(False)

        plt.tight_layout()
        fig_path = CONFIG['output_dir'] + f'step1_{gid.replace(".", "_")}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ {fig_path}")

    # ========== 5. 生成报告 ==========
    print("\n[5/5] 生成报告...")

    report = []
    report.append("=" * 70)
    report.append("Step 1 诊断报告: DEM采样 + 流线方向验证")
    report.append("=" * 70)

    report.append("\n## 配置参数")
    report.append(f"- 采样间距: {CONFIG['sampling_spacing_m']} m")
    report.append(f"- 测试冰川数: {len(all_results)}")

    # 统计
    total_flowlines = 0
    correct_direction = 0
    needs_reverse = 0

    for glacier_results in all_results:
        gid = glacier_results['glacier_id']
        report.append(f"\n## {gid}")
        report.append(f"- UTM Zone: {glacier_results['utm_zone']}N (EPSG:{glacier_results['utm_epsg']})")
        report.append(f"- 流线数: {len(glacier_results['flowlines'])}")

        for fl_tuple, sample_result, direction_result in zip(
                glacier_results['flowlines'],
                glacier_results['sample_results'],
                glacier_results['direction_results']
        ):
            fl_idx, _ = fl_tuple
            total_flowlines += 1

            if direction_result['direction_correct']:
                correct_direction += 1
                status = "✓ 正确"
            else:
                needs_reverse += 1
                status = "✗ 需反转"

            report.append(f"\n  ### 流线 {fl_idx}")
            report.append(f"  - 长度: {sample_result['length_m']:.1f} m")
            report.append(f"  - 采样点数: {sample_result['n_points']}")
            report.append(f"  - 有效点数: {direction_result['n_valid_points']}")
            report.append(f"  - 起点高程: {direction_result['start_elev']:.1f} m")
            report.append(f"  - 终点高程: {direction_result['end_elev']:.1f} m")
            report.append(f"  - 高程范围: {direction_result['min_elev']:.1f} - {direction_result['max_elev']:.1f} m")
            report.append(f"  - 高程变化率: {direction_result['slope_m_per_m'] * 1000:.2f} m/km")
            report.append(f"  - 方向: {status}")

    # 总结
    report.append("\n" + "=" * 70)
    report.append("## 总结")
    report.append(f"- 总流线数: {total_flowlines}")
    report.append(f"- 方向正确: {correct_direction} ({100 * correct_direction / max(1, total_flowlines):.1f}%)")
    report.append(f"- 需要反转: {needs_reverse} ({100 * needs_reverse / max(1, total_flowlines):.1f}%)")

    if needs_reverse > 0:
        report.append("\n⚠ 建议: 在冰厚计算前，根据高程自动判断并反转流线方向")
        report.append("   代码中已实现: 比较起点和终点高程，如果起点>终点则反转")
    else:
        report.append("\n✓ 所有流线方向正确，可以进入 Step 2")

    report.append("=" * 70)

    # 保存报告
    report_text = "\n".join(report)
    report_path = CONFIG['output_dir'] + 'step1_diagnostic_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {report_path}")
    print(report_text)

    # 保存详细数据
    detail_data = []
    for glacier_results in all_results:
        for fl_tuple, sample_result, direction_result in zip(
                glacier_results['flowlines'],
                glacier_results['sample_results'],
                glacier_results['direction_results']
        ):
            fl_idx, _ = fl_tuple
            detail_data.append({
                'glacier_id': glacier_results['glacier_id'],
                'flowline_idx': fl_idx,
                'utm_zone': glacier_results['utm_zone'],
                'length_m': sample_result['length_m'],
                'n_points': sample_result['n_points'],
                'n_valid_points': direction_result['n_valid_points'],
                'start_elev': direction_result['start_elev'],
                'end_elev': direction_result['end_elev'],
                'min_elev': direction_result['min_elev'],
                'max_elev': direction_result['max_elev'],
                'elev_range': direction_result['elev_range'],
                'slope_m_per_km': direction_result['slope_m_per_m'] * 1000,
                'direction_correct': direction_result['direction_correct'],
                'needs_reverse': direction_result['needs_reverse']
            })

    if detail_data:
        detail_df = pd.DataFrame(detail_data)
        csv_path = CONFIG['output_dir'] + 'step1_flowline_analysis.csv'
        detail_df.to_csv(csv_path, index=False)
        print(f"详细数据已保存: {csv_path}")

    print("\n" + "=" * 70)
    print("诊断完成! 请将以下文件发给我:")
    print(f"  1. {report_path}")
    print(f"  2. {CONFIG['output_dir']}step1_*.png (所有图片)")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_diagnostic()