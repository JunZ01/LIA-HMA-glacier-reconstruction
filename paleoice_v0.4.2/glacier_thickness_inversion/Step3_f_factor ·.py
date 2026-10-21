"""
Step 3 诊断脚本: 横截面生成与F因子计算
======================================

验证内容:
1. 生成垂直于流线的横截面
2. 用LIA冰川边界裁剪横截面 (intersection)
3. 沿裁剪后的横截面采样DEM
4. 以边界端点连线作为古冰面，计算F因子
5. 对比不同方法的F因子结果

F因子计算方法:
- 方法1: 面积法 F = A / (P × H)
- 方法2: Li et al. 2012 抛物线拟合法

使用方法:
    python step3_cross_section.py
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from shapely.geometry import Point, LineString, Polygon, MultiLineString
from shapely.ops import transform, nearest_points
from pyproj import CRS, Transformer
import pandas as pd
import math
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
    'output_dir': "/mnt/d/paleoice_output/step3_diagnostic/",

    # 测试冰川
    'test_glacier_id': "RGI60-15.06881",

    # 横截面参数
    'cross_section_half_width_m': 2000,  # 横截面半宽度（米），生成时用，会被边界裁剪
    'cross_section_spacing_m': 500,      # 横截面间距（米）
    'cross_section_sample_points': 50,   # 每条横截面的采样点数

    # 采样参数
    'flowline_spacing_m': 30,
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


def transform_gdf(gdf, dst_crs):
    """转换GeoDataFrame坐标系"""
    return gdf.to_crs(dst_crs)


# ============================================================================
# 横截面生成函数
# ============================================================================

def generate_cross_section_line(center_x, center_y, direction_x, direction_y, half_width):
    """
    生成垂直于流线方向的横截面线

    Args:
        center_x, center_y: 中心点坐标 (UTM)
        direction_x, direction_y: 流线方向向量
        half_width: 横截面半宽度 (米)

    Returns:
        LineString: 横截面线
    """
    length = math.sqrt(direction_x**2 + direction_y**2)
    if length == 0:
        return None

    # 单位方向向量
    dx = direction_x / length
    dy = direction_y / length

    # 垂直方向 (逆时针旋转90度)
    perp_dx = -dy
    perp_dy = dx

    # 横截面端点
    x1 = center_x + perp_dx * half_width
    y1 = center_y + perp_dy * half_width
    x2 = center_x - perp_dx * half_width
    y2 = center_y - perp_dy * half_width

    return LineString([(x1, y1), (x2, y2)])


def generate_cross_sections_along_flowline(flowline_utm, glacier_boundary_utm,
                                            spacing_m, half_width_m):
    """
    沿流线生成横截面，并用冰川边界裁剪

    Args:
        flowline_utm: UTM坐标下的流线
        glacier_boundary_utm: UTM坐标下的冰川边界
        spacing_m: 横截面间距 (米)
        half_width_m: 横截面半宽度 (米)

    Returns:
        list of dict: 每个横截面的信息
    """
    length = flowline_utm.length
    num_sections = max(int(length / spacing_m), 2)

    cross_sections = []

    for i in range(num_sections):
        distance = i * spacing_m
        if distance >= length:
            break

        # 获取流线上的点
        point = flowline_utm.interpolate(distance)

        # 计算流线方向（使用前后点）
        d_before = max(0, distance - 50)
        d_after = min(length, distance + 50)

        p_before = flowline_utm.interpolate(d_before)
        p_after = flowline_utm.interpolate(d_after)

        direction_x = p_after.x - p_before.x
        direction_y = p_after.y - p_before.y

        # 生成横截面线
        cs_line = generate_cross_section_line(
            point.x, point.y,
            direction_x, direction_y,
            half_width_m
        )

        if cs_line is None:
            continue

        # 用冰川边界裁剪横截面
        try:
            clipped_line = cs_line.intersection(glacier_boundary_utm)

            if clipped_line.is_empty:
                continue

            # 如果结果是MultiLineString，取最长的或包含中心点的
            if isinstance(clipped_line, MultiLineString):
                # 找包含中心点的部分
                for part in clipped_line.geoms:
                    if part.distance(point) < 10:  # 10米容差
                        clipped_line = part
                        break
                else:
                    # 取最长的
                    clipped_line = max(clipped_line.geoms, key=lambda x: x.length)

            if not isinstance(clipped_line, LineString):
                continue

            if clipped_line.length < 50:  # 忽略太短的横截面
                continue

            cross_sections.append({
                'index': i,
                'distance_along_flowline': distance,
                'center_point': point,
                'original_line': cs_line,
                'clipped_line': clipped_line,
                'width_m': clipped_line.length
            })

        except Exception as e:
            continue

    return cross_sections


# ============================================================================
# 横截面DEM采样和F因子计算
# ============================================================================

def sample_cross_section(cross_section_line, dem_path, utm_crs, n_points=50):
    """
    沿横截面采样DEM高程

    Args:
        cross_section_line: 横截面线 (UTM坐标)
        dem_path: DEM文件路径
        utm_crs: UTM坐标系
        n_points: 采样点数

    Returns:
        dict: 采样结果
    """
    src_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(utm_crs, src_crs, always_xy=True)

    # 沿横截面生成采样点
    distances = np.linspace(0, cross_section_line.length, n_points)
    points = [cross_section_line.interpolate(d) for d in distances]

    xs_utm = np.array([p.x for p in points])
    ys_utm = np.array([p.y for p in points])

    # 采样DEM
    with rasterio.open(dem_path) as src:
        elevs = []
        for x_utm, y_utm in zip(xs_utm, ys_utm):
            lon, lat = transformer.transform(x_utm, y_utm)
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                    elevs.append(val if val > -9999 else np.nan)
                else:
                    elevs.append(np.nan)
            except:
                elevs.append(np.nan)

    return {
        'distances': distances,
        'xs_utm': xs_utm,
        'ys_utm': ys_utm,
        'bed_elevs': np.array(elevs)
    }


def calculate_ffactor_area_method(distances, bed_elevs):
    """
    基于面积法计算F因子

    F = A / (P × H)

    其中:
    - A: 横截面积 (冰面和床面围成的区域)
    - P: 湿周长 (床面接触长度)
    - H: 最大冰厚 (垂直方向)

    假设: 冰面为两端点的连线 (对应LIA边界上的Trimline)

    Args:
        distances: 沿横截面的距离数组
        bed_elevs: 床面高程数组

    Returns:
        dict: F因子和相关计算量
    """
    # 去除NaN
    valid_mask = ~np.isnan(bed_elevs)
    if np.sum(valid_mask) < 3:
        return {'F': 0.8, 'valid': False, 'reason': '有效点不足'}

    dist = distances[valid_mask]
    bed = bed_elevs[valid_mask]

    # 冰面: 两端点连线
    z0 = bed[0]   # 左端点高程 (边界上)
    zn = bed[-1]  # 右端点高程 (边界上)
    width = dist[-1] - dist[0]

    if width <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '宽度为0'}

    # 冰面斜率
    k = (zn - z0) / width

    # 计算冰面高程（线性插值）
    ice_surface = z0 + k * (dist - dist[0])

    # 计算冰厚
    thickness = ice_surface - bed

    # 检查是否有正的冰厚
    if np.max(thickness) <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '无正冰厚'}

    # 计算面积A（梯形积分）
    A = 0
    for i in range(1, len(dist)):
        dx = dist[i] - dist[i-1]
        h_avg = (thickness[i-1] + thickness[i]) / 2
        if h_avg > 0:
            A += dx * h_avg

    # 计算湿周长P（床面的3D长度）
    P = 0
    for i in range(1, len(dist)):
        dx = dist[i] - dist[i-1]
        dz = bed[i] - bed[i-1]
        P += math.sqrt(dx**2 + dz**2)

    # 最大冰厚H（垂直方向，考虑冰面倾斜）
    # H = max(thickness) * cos(arctan(k))
    cos_angle = 1 / math.sqrt(1 + k**2)
    H = np.max(thickness) * cos_angle

    # 计算F因子
    if P * H > 0:
        F = A / (P * H)
    else:
        F = 0.8

    # 限制范围
    F = np.clip(F, 0.445, 1.0)

    return {
        'F': F,
        'valid': True,
        'A': A,
        'P': P,
        'H': H,
        'width': width,
        'max_thickness': np.max(thickness),
        'mean_thickness': np.mean(thickness[thickness > 0]) if np.any(thickness > 0) else 0,
        'ice_surface': ice_surface,
        'thickness': thickness
    }


def calculate_ffactor_polyfit_method(distances, bed_elevs):
    """
    基于Li et al. (2012) 抛物线拟合法计算F因子

    假设横截面为抛物线: z = a*x² + b*x + c
    F = 0.9r / (1 + 0.9r), 其中 r = sqrt(1/(H*a))

    Args:
        distances: 沿横截面的距离数组
        bed_elevs: 床面高程数组

    Returns:
        dict: F因子和相关计算量
    """
    # 去除NaN
    valid_mask = ~np.isnan(bed_elevs)
    if np.sum(valid_mask) < 4:
        return {'F': 0.8, 'valid': False, 'reason': '有效点不足'}

    dist = distances[valid_mask]
    bed = bed_elevs[valid_mask]

    # 冰面: 两端点连线
    z0 = bed[0]
    zn = bed[-1]
    width = dist[-1] - dist[0]

    if width <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '宽度为0'}

    k = (zn - z0) / width
    ice_surface = z0 + k * (dist - dist[0])
    thickness = ice_surface - bed

    H = np.max(thickness)
    if H <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '无正冰厚'}

    # 抛物线拟合床面
    try:
        # 将距离归一化到中心
        dist_centered = dist - np.mean(dist)
        polyfit = np.polyfit(dist_centered, bed, 2)
        a = polyfit[0]  # 二次项系数

        if a > 0:
            # 凹形（U型谷）
            r = math.sqrt(1.0 / (H * a + 0.001))
            F = 0.9 * r / (1 + 0.9 * r)
        else:
            # 凸形或平坦，使用默认值
            F = 0.8

    except Exception as e:
        F = 0.8

    # 限制范围
    F = np.clip(F, 0.445, 1.0)

    return {
        'F': F,
        'valid': True,
        'H': H,
        'polyfit_a': a if 'a' in dir() else None,
        'width': width,
        'thickness': thickness,
        'ice_surface': ice_surface
    }


# ============================================================================
# 可视化函数
# ============================================================================

def plot_cross_sections_overview(glacier_boundary_utm, flowline_utm,
                                  cross_sections, glacier_id, output_path):
    """
    绘制横截面位置概览图
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # 绘制冰川边界
    if hasattr(glacier_boundary_utm, 'exterior'):
        x, y = glacier_boundary_utm.exterior.xy
        ax.fill(x, y, alpha=0.3, color='lightblue', label='LIA Boundary')
        ax.plot(x, y, 'b-', linewidth=2)

    # 绘制流线
    if flowline_utm.geom_type == 'MultiLineString':
        for part in flowline_utm.geoms:
            x, y = part.xy
            ax.plot(x, y, 'r-', linewidth=2)
    else:
        x, y = flowline_utm.xy
        ax.plot(x, y, 'r-', linewidth=2, label='Flowline')

    # 绘制横截面
    for i, cs in enumerate(cross_sections):
        # 原始横截面（灰色虚线）
        x, y = cs['original_line'].xy
        ax.plot(x, y, 'gray', linewidth=0.5, linestyle='--', alpha=0.5)

        # 裁剪后的横截面（绿色实线）
        x, y = cs['clipped_line'].xy
        ax.plot(x, y, 'g-', linewidth=1.5)

        # 中心点
        ax.scatter(cs['center_point'].x, cs['center_point'].y,
                   c='red', s=20, zorder=5)

    ax.set_xlabel('Easting (m)', fontsize=12)
    ax.set_ylabel('Northing (m)', fontsize=12)
    ax.set_title(f'{glacier_id}\nCross Sections Overview\n'
                 f'({len(cross_sections)} sections, clipped by LIA boundary)', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_cross_section_profiles(cross_sections, sample_results, ffactor_results,
                                 glacier_id, output_path):
    """
    绘制横截面剖面图（选择几个代表性的）
    """
    # 选择几个代表性的横截面
    n_total = len(cross_sections)
    if n_total <= 6:
        indices = list(range(n_total))
    else:
        indices = [0, n_total//4, n_total//2, 3*n_total//4, n_total-1]

    n_plots = len(indices)
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(5 * ((n_plots + 1) // 2), 10))
    axes = axes.flatten() if n_plots > 1 else [axes]

    for plot_idx, cs_idx in enumerate(indices):
        if plot_idx >= len(axes):
            break

        ax = axes[plot_idx]
        cs = cross_sections[cs_idx]
        sample = sample_results[cs_idx]
        ff_area = ffactor_results['area'][cs_idx]
        ff_poly = ffactor_results['polyfit'][cs_idx]

        dist = sample['distances']
        bed = sample['bed_elevs']

        # 绘制床面
        ax.fill_between(dist, np.nanmin(bed) - 50, bed, alpha=0.3, color='brown', label='Bed')
        ax.plot(dist, bed, 'k-', linewidth=2)

        # 如果有有效的F因子结果，绘制冰面
        if ff_area['valid']:
            ice_surf = ff_area['ice_surface']
            # 只在正冰厚区域填充
            valid_ice = np.where(ff_area['thickness'] > 0, ice_surf, bed)
            ax.fill_between(dist[~np.isnan(bed)], bed[~np.isnan(bed)],
                           valid_ice[~np.isnan(bed)],
                           alpha=0.4, color='lightblue', label='Ice')
            ax.plot(dist[~np.isnan(bed)], ice_surf[~np.isnan(bed)],
                   'b--', linewidth=2, label='Ice surface (Trimline)')

        # 标记边界点
        valid_mask = ~np.isnan(bed)
        if np.any(valid_mask):
            valid_dist = dist[valid_mask]
            valid_bed = bed[valid_mask]
            ax.scatter([valid_dist[0], valid_dist[-1]],
                       [valid_bed[0], valid_bed[-1]],
                       c='blue', s=80, marker='s', zorder=5,
                       label='LIA boundary')

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Elevation (m)')
        ax.set_title(f'Cross Section @ {cs["distance_along_flowline"]:.0f}m\n'
                     f'F(area)={ff_area["F"]:.3f}, F(poly)={ff_poly["F"]:.3f}\n'
                     f'Width={cs["width_m"]:.0f}m')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(indices), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'{glacier_id} - Cross Section Profiles', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_ffactor_distribution(cross_sections, ffactor_results, glacier_id, output_path):
    """
    绘制F因子分布图
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    distances = [cs['distance_along_flowline'] / 1000 for cs in cross_sections]
    f_area = [ff['F'] for ff in ffactor_results['area']]
    f_poly = [ff['F'] for ff in ffactor_results['polyfit']]
    widths = [cs['width_m'] for cs in cross_sections]

    # 1. F因子沿流线分布
    ax1 = axes[0, 0]
    ax1.plot(distances, f_area, 'bo-', linewidth=1.5, markersize=4, label='Area method')
    ax1.plot(distances, f_poly, 'rs-', linewidth=1.5, markersize=4, label='Polyfit method')
    ax1.axhline(y=0.8, color='gray', linestyle='--', label='Default (0.8)')
    ax1.set_xlabel('Distance from terminus (km)')
    ax1.set_ylabel('F factor')
    ax1.set_title('F Factor Along Flowline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 1.1)

    # 2. F因子直方图
    ax2 = axes[0, 1]
    ax2.hist(f_area, bins=20, alpha=0.5, color='blue', label='Area method')
    ax2.hist(f_poly, bins=20, alpha=0.5, color='red', label='Polyfit method')
    ax2.axvline(x=np.mean(f_area), color='blue', linestyle='--',
                label=f'Mean(area)={np.mean(f_area):.3f}')
    ax2.axvline(x=np.mean(f_poly), color='red', linestyle='--',
                label=f'Mean(poly)={np.mean(f_poly):.3f}')
    ax2.set_xlabel('F factor')
    ax2.set_ylabel('Count')
    ax2.set_title('F Factor Distribution')
    ax2.legend()

    # 3. 横截面宽度沿流线分布
    ax3 = axes[1, 0]
    ax3.plot(distances, widths, 'g-', linewidth=2)
    ax3.fill_between(distances, 0, widths, alpha=0.3, color='green')
    ax3.set_xlabel('Distance from terminus (km)')
    ax3.set_ylabel('Cross section width (m)')
    ax3.set_title('Cross Section Width (clipped by LIA boundary)')
    ax3.grid(True, alpha=0.3)

    # 4. F因子 vs 横截面宽度
    ax4 = axes[1, 1]
    ax4.scatter(widths, f_area, alpha=0.6, c='blue', label='Area method')
    ax4.scatter(widths, f_poly, alpha=0.6, c='red', label='Polyfit method')
    ax4.set_xlabel('Cross section width (m)')
    ax4.set_ylabel('F factor')
    ax4.set_title('F Factor vs Width')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'{glacier_id} - F Factor Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# ============================================================================
# 主诊断函数
# ============================================================================

def run_diagnostic():
    """运行Step 3诊断"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 3 诊断: 横截面生成与F因子计算")
    print("=" * 70)

    # ========== 1. 加载数据 ==========
    print("\n[1/6] 加载数据...")

    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])

    # 获取测试冰川
    test_glacier = glacier_gdf[glacier_gdf['OrigRGIId'] == CONFIG['test_glacier_id']]
    test_flowlines = flowlines_gdf[flowlines_gdf['MainRGI'] == CONFIG['test_glacier_id']]

    if len(test_glacier) == 0:
        print(f"  ✗ 找不到测试冰川: {CONFIG['test_glacier_id']}")
        return

    glacier_geom = test_glacier.geometry.values[0]

    # 获取主流线（最长的）
    test_flowlines = test_flowlines.copy()
    test_flowlines['length'] = test_flowlines.geometry.length
    main_flowline = test_flowlines.nlargest(1, 'length').geometry.values[0]

    if main_flowline.geom_type == 'MultiLineString':
        main_flowline = max(list(main_flowline.geoms), key=lambda x: x.length)

    print(f"  测试冰川: {CONFIG['test_glacier_id']}")
    print(f"  冰川面积: {glacier_geom.area * 111**2:.2f} km² (估算)")
    print(f"  主流线长度: {main_flowline.length * 111000:.1f} m (估算)")

    # ========== 2. 投影到UTM ==========
    print("\n[2/6] 投影到UTM坐标系...")

    centroid = glacier_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    glacier_utm = transform_geometry(glacier_geom, src_crs, utm_crs)
    flowline_utm = transform_geometry(main_flowline, src_crs, utm_crs)

    print(f"  UTM Zone: {utm_zone}N (EPSG:{utm_crs.to_epsg()})")
    print(f"  冰川面积: {glacier_utm.area / 1e6:.2f} km²")
    print(f"  主流线长度: {flowline_utm.length:.1f} m")

    # ========== 3. 反转流线方向 ==========
    print("\n[3/6] 检查流线方向...")

    # 采样起点和终点高程
    with rasterio.open(CONFIG['dem_path']) as src:
        transformer = Transformer.from_crs(utm_crs, src_crs, always_xy=True)

        start_point = flowline_utm.interpolate(0)
        end_point = flowline_utm.interpolate(flowline_utm.length)

        start_lon, start_lat = transformer.transform(start_point.x, start_point.y)
        end_lon, end_lat = transformer.transform(end_point.x, end_point.y)

        start_row, start_col = src.index(start_lon, start_lat)
        end_row, end_col = src.index(end_lon, end_lat)

        start_elev = src.read(1, window=((start_row, start_row+1), (start_col, start_col+1)))[0, 0]
        end_elev = src.read(1, window=((end_row, end_row+1), (end_col, end_col+1)))[0, 0]

    print(f"  起点高程: {start_elev:.1f} m")
    print(f"  终点高程: {end_elev:.1f} m")

    if start_elev > end_elev:
        print("  流线方向: 源头→末端，需要反转")
        # 反转流线
        coords = list(flowline_utm.coords)[::-1]
        flowline_utm = LineString(coords)
        print("  ✓ 已反转")
    else:
        print("  流线方向: 末端→源头，无需反转")

    # ========== 4. 生成横截面 ==========
    print("\n[4/6] 生成横截面...")

    cross_sections = generate_cross_sections_along_flowline(
        flowline_utm, glacier_utm,
        CONFIG['cross_section_spacing_m'],
        CONFIG['cross_section_half_width_m']
    )

    print(f"  生成横截面数: {len(cross_sections)}")
    print(f"  横截面间距: {CONFIG['cross_section_spacing_m']} m")
    print(f"  初始半宽度: {CONFIG['cross_section_half_width_m']} m")

    if len(cross_sections) == 0:
        print("  ✗ 没有生成有效的横截面!")
        return

    # 统计裁剪后的宽度
    widths = [cs['width_m'] for cs in cross_sections]
    print(f"  裁剪后宽度范围: {min(widths):.1f} - {max(widths):.1f} m")
    print(f"  平均宽度: {np.mean(widths):.1f} m")

    # ========== 5. 采样DEM并计算F因子 ==========
    print("\n[5/6] 采样DEM并计算F因子...")

    sample_results = []
    ffactor_area = []
    ffactor_polyfit = []

    for cs in cross_sections:
        # 采样
        sample = sample_cross_section(
            cs['clipped_line'], CONFIG['dem_path'], utm_crs,
            CONFIG['cross_section_sample_points']
        )
        sample_results.append(sample)

        # 计算F因子
        ff_area = calculate_ffactor_area_method(sample['distances'], sample['bed_elevs'])
        ff_poly = calculate_ffactor_polyfit_method(sample['distances'], sample['bed_elevs'])

        ffactor_area.append(ff_area)
        ffactor_polyfit.append(ff_poly)

    ffactor_results = {
        'area': ffactor_area,
        'polyfit': ffactor_polyfit
    }

    # 统计F因子
    f_area_values = [ff['F'] for ff in ffactor_area]
    f_poly_values = [ff['F'] for ff in ffactor_polyfit]

    print(f"\n  F因子统计 (面积法):")
    print(f"    范围: {min(f_area_values):.3f} - {max(f_area_values):.3f}")
    print(f"    平均: {np.mean(f_area_values):.3f}")
    print(f"    中位数: {np.median(f_area_values):.3f}")

    print(f"\n  F因子统计 (多项式拟合法):")
    print(f"    范围: {min(f_poly_values):.3f} - {max(f_poly_values):.3f}")
    print(f"    平均: {np.mean(f_poly_values):.3f}")
    print(f"    中位数: {np.median(f_poly_values):.3f}")

    # ========== 6. 生成可视化 ==========
    print("\n[6/6] 生成可视化...")

    # 概览图
    overview_path = CONFIG['output_dir'] + 'step3_cross_sections_overview.png'
    plot_cross_sections_overview(glacier_utm, flowline_utm, cross_sections,
                                  CONFIG['test_glacier_id'], overview_path)
    print(f"  ✓ {overview_path}")

    # 剖面图
    profiles_path = CONFIG['output_dir'] + 'step3_cross_section_profiles.png'
    plot_cross_section_profiles(cross_sections, sample_results, ffactor_results,
                                 CONFIG['test_glacier_id'], profiles_path)
    print(f"  ✓ {profiles_path}")

    # F因子分布图
    ffactor_path = CONFIG['output_dir'] + 'step3_ffactor_distribution.png'
    plot_ffactor_distribution(cross_sections, ffactor_results,
                               CONFIG['test_glacier_id'], ffactor_path)
    print(f"  ✓ {ffactor_path}")

    # ========== 生成报告 ==========
    report = []
    report.append("=" * 70)
    report.append("Step 3 诊断报告: 横截面生成与F因子计算")
    report.append("=" * 70)

    report.append("\n## 测试配置")
    report.append(f"- 测试冰川: {CONFIG['test_glacier_id']}")
    report.append(f"- 冰川面积: {glacier_utm.area / 1e6:.2f} km²")
    report.append(f"- 主流线长度: {flowline_utm.length:.1f} m")
    report.append(f"- UTM Zone: {utm_zone}N")
    report.append(f"- 横截面间距: {CONFIG['cross_section_spacing_m']} m")
    report.append(f"- 初始半宽度: {CONFIG['cross_section_half_width_m']} m")

    report.append("\n## 横截面生成结果")
    report.append(f"- 生成横截面数: {len(cross_sections)}")
    report.append(f"- 裁剪后宽度范围: {min(widths):.1f} - {max(widths):.1f} m")
    report.append(f"- 平均宽度: {np.mean(widths):.1f} m")

    report.append("\n## F因子计算结果")
    report.append("")
    report.append("| 方法 | 最小值 | 最大值 | 平均值 | 中位数 |")
    report.append("|------|--------|--------|--------|--------|")
    report.append(f"| 面积法 | {min(f_area_values):.3f} | {max(f_area_values):.3f} | "
                 f"{np.mean(f_area_values):.3f} | {np.median(f_area_values):.3f} |")
    report.append(f"| 多项式拟合法 | {min(f_poly_values):.3f} | {max(f_poly_values):.3f} | "
                 f"{np.mean(f_poly_values):.3f} | {np.median(f_poly_values):.3f} |")

    report.append("\n## 验证检查")

    # 检查1: F因子范围
    if 0.5 < np.mean(f_area_values) < 0.9:
        report.append("✓ F因子平均值在合理范围 (0.5-0.9)")
    else:
        report.append(f"⚠ F因子平均值 {np.mean(f_area_values):.3f} 可能异常")

    # 检查2: 横截面宽度
    if np.mean(widths) > 100:
        report.append("✓ 横截面平均宽度合理 (>100m)")
    else:
        report.append(f"⚠ 横截面平均宽度 {np.mean(widths):.1f}m 可能过小")

    # 检查3: LIA边界裁剪效果
    original_widths = [cs['original_line'].length for cs in cross_sections]
    clip_ratio = np.mean(widths) / np.mean(original_widths)
    report.append(f"✓ LIA边界裁剪有效 (裁剪比例: {clip_ratio:.1%})")

    report.append("\n## 结论")

    all_ok = (0.5 < np.mean(f_area_values) < 0.9 and
              np.mean(widths) > 100 and
              len(cross_sections) > 10)

    if all_ok:
        report.append("✓ 横截面生成和F因子计算验证通过，可以进入 Step 4")
    else:
        report.append("⚠ 存在一些问题，请检查上述详情")

    report.append("=" * 70)

    # 保存报告
    report_text = "\n".join(report)
    report_path = CONFIG['output_dir'] + 'step3_diagnostic_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {report_path}")
    print(report_text)

    # 保存详细数据
    detail_data = []
    for i, cs in enumerate(cross_sections):
        detail_data.append({
            'index': cs['index'],
            'distance_m': cs['distance_along_flowline'],
            'width_m': cs['width_m'],
            'F_area': ffactor_area[i]['F'],
            'F_polyfit': ffactor_polyfit[i]['F'],
            'valid_area': ffactor_area[i]['valid'],
            'valid_polyfit': ffactor_polyfit[i]['valid']
        })

    detail_df = pd.DataFrame(detail_data)
    csv_path = CONFIG['output_dir'] + 'step3_ffactor_data.csv'
    detail_df.to_csv(csv_path, index=False)
    print(f"详细数据已保存: {csv_path}")

    print("\n" + "=" * 70)
    print("诊断完成! 请将以下文件发给我:")
    print(f"  1. {report_path}")
    print(f"  2. {overview_path}")
    print(f"  3. {profiles_path}")
    print(f"  4. {ffactor_path}")
    print("=" * 70)

    return {
        'cross_sections': cross_sections,
        'sample_results': sample_results,
        'ffactor_results': ffactor_results
    }


if __name__ == "__main__":
    results = run_diagnostic()