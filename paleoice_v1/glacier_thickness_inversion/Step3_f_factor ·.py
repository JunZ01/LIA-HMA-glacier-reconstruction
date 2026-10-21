"""
Step 3 修改版: 适配OGGM连接后的流线
=======================================

输入: lia_flowlines_connected.shp (或 lia_flowlines_smooth.shp)
输出: step3_ffactor_data.csv

修改要点:
1. 读取OGGM流线格式 (rgi_id, fl_id, is_main, order, flows_to)
2. 对所有流线(主流+支流)都计算F因子
3. 横截面生成逻辑保持不变
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import transform
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
    # 输入
    'flowlines_path': "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_smooth.shp",

    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'output_dir': "/mnt/d/paleoice_output/step3_oggm/",

    # 测试冰川 (可选,为空则处理所有冰川)
    'test_glacier_ids': [],  # 例: ['RGI60-15.06881', 'RGI60-15.03733']

    # 横截面参数
    'cross_section_half_width_m': 2000,
    'cross_section_spacing_m': 500,
    'cross_section_sample_points': 50,
}


# ============================================================================
# 工具函数
# ============================================================================

def get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg), zone


def transform_geometry(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


# ============================================================================
# 横截面生成
# ============================================================================

def generate_cross_section_line(center_x, center_y, direction_x, direction_y, half_width):
    """生成垂直于流线的横截面"""
    length = math.sqrt(direction_x**2 + direction_y**2)
    if length == 0:
        return None

    dx, dy = direction_x / length, direction_y / length
    perp_dx, perp_dy = -dy, dx

    x1 = center_x + perp_dx * half_width
    y1 = center_y + perp_dy * half_width
    x2 = center_x - perp_dx * half_width
    y2 = center_y - perp_dy * half_width

    return LineString([(x1, y1), (x2, y2)])


def generate_cross_sections_along_flowline(flowline_utm, glacier_boundary_utm,
                                            spacing_m, half_width_m):
    """沿流线生成横截面并用冰川边界裁剪"""
    length = flowline_utm.length
    num_sections = max(int(length / spacing_m), 2)

    cross_sections = []

    for i in range(num_sections):
        distance = i * spacing_m
        if distance >= length:
            break

        point = flowline_utm.interpolate(distance)

        # 计算流线方向
        d_before = max(0, distance - 50)
        d_after = min(length, distance + 50)
        p_before = flowline_utm.interpolate(d_before)
        p_after = flowline_utm.interpolate(d_after)

        direction_x = p_after.x - p_before.x
        direction_y = p_after.y - p_before.y

        cs_line = generate_cross_section_line(
            point.x, point.y, direction_x, direction_y, half_width_m
        )

        if cs_line is None:
            continue

        # 用冰川边界裁剪
        try:
            clipped_line = cs_line.intersection(glacier_boundary_utm)

            if clipped_line.is_empty:
                continue

            if isinstance(clipped_line, MultiLineString):
                for part in clipped_line.geoms:
                    if part.distance(point) < 10:
                        clipped_line = part
                        break
                else:
                    clipped_line = max(clipped_line.geoms, key=lambda x: x.length)

            if not isinstance(clipped_line, LineString):
                continue

            if clipped_line.length < 50:
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
# DEM采样
# ============================================================================

def sample_cross_section(cross_section_line, dem_path, utm_crs, n_points=50):
    """沿横截面采样DEM"""
    src_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(utm_crs, src_crs, always_xy=True)

    distances = np.linspace(0, cross_section_line.length, n_points)
    points = [cross_section_line.interpolate(d) for d in distances]

    xs_utm = np.array([p.x for p in points])
    ys_utm = np.array([p.y for p in points])

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


# ============================================================================
# F因子计算
# ============================================================================

def calculate_ffactor_area_method(distances, bed_elevs):
    """面积法计算F因子"""
    valid_mask = ~np.isnan(bed_elevs)
    if np.sum(valid_mask) < 3:
        return {'F': 0.8, 'valid': False, 'reason': '有效点不足'}

    dist = distances[valid_mask]
    bed = bed_elevs[valid_mask]

    z0, zn = bed[0], bed[-1]
    width = dist[-1] - dist[0]

    if width <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '宽度为0'}

    k = (zn - z0) / width
    ice_surface = z0 + k * (dist - dist[0])
    thickness = ice_surface - bed

    if np.max(thickness) <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '无正冰厚'}

    # 计算面积A
    A = 0
    for i in range(1, len(dist)):
        dx = dist[i] - dist[i-1]
        h_avg = (thickness[i-1] + thickness[i]) / 2
        if h_avg > 0:
            A += dx * h_avg

    # 计算湿周长P
    P = 0
    for i in range(1, len(dist)):
        dx = dist[i] - dist[i-1]
        dz = bed[i] - bed[i-1]
        P += math.sqrt(dx**2 + dz**2)

    # 最大冰厚H
    cos_angle = 1 / math.sqrt(1 + k**2)
    H = np.max(thickness) * cos_angle

    # F因子
    if P * H > 0:
        F = A / (P * H)
    else:
        F = 0.8

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
    """抛物线拟合法计算F因子"""
    valid_mask = ~np.isnan(bed_elevs)
    if np.sum(valid_mask) < 4:
        return {'F': 0.8, 'valid': False, 'reason': '有效点不足'}

    dist = distances[valid_mask]
    bed = bed_elevs[valid_mask]

    z0, zn = bed[0], bed[-1]
    width = dist[-1] - dist[0]

    if width <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '宽度为0'}

    k = (zn - z0) / width
    ice_surface = z0 + k * (dist - dist[0])
    thickness = ice_surface - bed

    H = np.max(thickness)
    if H <= 0:
        return {'F': 0.8, 'valid': False, 'reason': '无正冰厚'}

    try:
        dist_centered = dist - np.mean(dist)
        polyfit = np.polyfit(dist_centered, bed, 2)
        a = polyfit[0]

        if a > 0:
            r = math.sqrt(1.0 / (H * a + 0.001))
            F = 0.9 * r / (1 + 0.9 * r)
        else:
            F = 0.8
    except:
        F = 0.8

    F = np.clip(F, 0.445, 1.0)

    return {
        'F': F,
        'valid': True,
        'H': H,
        'a': a if 'a' in locals() else 0
    }


# ============================================================================
# 主处理函数
# ============================================================================

def process_glacier(unique_id, flowlines_gdf, glacier_gdf, dem_path, config):
    """处理单个冰川的所有流线"""

    print(f"\n处理冰川: {unique_id}")

    # 获取该冰川的边界和流线
    glacier = glacier_gdf[glacier_gdf['OrigRGIId'] == unique_id]
    if len(glacier) == 0:
        glacier = glacier_gdf[glacier_gdf['UniqueID'] == unique_id]

    if len(glacier) == 0:
        print(f"  ✗ 找不到冰川边界")
        return None

    glacier_geom = glacier.geometry.values[0]

    # 获取该冰川的所有流线
    glacier_flowlines = flowlines_gdf[flowlines_gdf['UniqueID'] == unique_id]

    if len(glacier_flowlines) == 0:
        print(f"  ✗ 找不到流线")
        return None

    print(f"  流线数: {len(glacier_flowlines)} (主流: {glacier_flowlines['is_main'].sum()}, "
          f"支流: {(~glacier_flowlines['is_main']).sum()})")

    # 获取UTM坐标系
    centroid = glacier_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    # 转换到UTM
    glacier_utm = transform_geometry(glacier_geom, src_crs, utm_crs)

    # 处理每条流线
    all_ffactor_data = []

    for idx, row in glacier_flowlines.iterrows():
        fl_id = row['fl_id']
        is_main = row['is_main']
        order = row['order']

        fl_geom = row.geometry

        # 处理MultiLineString
        if fl_geom.geom_type == 'MultiLineString':
            fl_geom = max(list(fl_geom.geoms), key=lambda x: x.length)

        # 转UTM
        fl_utm = transform_geometry(fl_geom, src_crs, utm_crs)

        print(f"  流线{fl_id} ({'主流' if is_main else f'{order}级支流'}): 长度{fl_utm.length:.0f}m")

        # 生成横截面
        cross_sections = generate_cross_sections_along_flowline(
            fl_utm, glacier_utm,
            config['cross_section_spacing_m'],
            config['cross_section_half_width_m']
        )

        if len(cross_sections) == 0:
            print(f"    ✗ 无有效横截面")
            continue

        print(f"    横截面数: {len(cross_sections)}")

        # 对每个横截面计算F因子
        for cs in cross_sections:
            # 采样DEM
            sample_result = sample_cross_section(
                cs['clipped_line'],
                dem_path,
                utm_crs,
                config['cross_section_sample_points']
            )

            # 计算F因子
            f_area = calculate_ffactor_area_method(
                sample_result['distances'],
                sample_result['bed_elevs']
            )

            f_poly = calculate_ffactor_polyfit_method(
                sample_result['distances'],
                sample_result['bed_elevs']
            )

            # 选择最终F值 (优先面积法)
            if f_area['valid']:
                F_final = f_area['F']
                method = 'area'
            elif f_poly['valid']:
                F_final = f_poly['F']
                method = 'polyfit'
            else:
                F_final = 0.8
                method = 'default'

            all_ffactor_data.append({
                'glacier_id': unique_id,
                'fl_id': fl_id,
                'is_main': is_main,
                'order': order,
                'cross_section_idx': cs['index'],
                'distance_along_flowline': cs['distance_along_flowline'],
                'F_area_method': f_area['F'] if f_area['valid'] else np.nan,
                'F_polyfit_method': f_poly['F'] if f_poly['valid'] else np.nan,
                'F_final': F_final,
                'F_method': method,
                'width_m': cs['width_m'],
                'max_thickness_m': f_area.get('max_thickness', np.nan),
                'mean_thickness_m': f_area.get('mean_thickness', np.nan),
                'x_utm': cs['center_point'].x,
                'y_utm': cs['center_point'].y,
                'utm_zone': utm_zone
            })

        print(f"    ✓ F因子: {len([d for d in all_ffactor_data if d['fl_id'] == fl_id])} 个横截面")

    return all_ffactor_data


def run_step3():
    """运行Step 3"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 3: F因子计算 (OGGM流线版)")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])

    print(f"  流线: {len(flowlines_gdf)} 条")
    print(f"  冰川: {len(glacier_gdf)} 个")

    # 检查字段
    required_fields = ['UniqueID', 'fl_id', 'is_main', 'order']
    missing = [f for f in required_fields if f not in flowlines_gdf.columns]
    if missing:
        raise ValueError(f"流线shapefile缺少字段: {missing}")

    # 确定处理的冰川
    if CONFIG['test_glacier_ids']:
        glacier_ids = CONFIG['test_glacier_ids']
    else:
        glacier_ids = flowlines_gdf['UniqueID'].unique()

    print(f"\n处理冰川数: {len(glacier_ids)}")

    # 处理每个冰川
    all_results = []

    for i, unique_id  in enumerate(glacier_ids, 1):
        print(f"\n[{i}/{len(glacier_ids)}]", end=' ')

        result = process_glacier(
            unique_id , flowlines_gdf, glacier_gdf,
            CONFIG['dem_path'], CONFIG
        )

        if result:
            all_results.extend(result)

    # 保存结果
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = os.path.join(CONFIG['output_dir'], 'step3_ffactor_data.csv')
        df.to_csv(output_csv, index=False)

        print(f"\n{'=' * 70}")
        print("统计:")
        print(f"  总横截面数: {len(df)}")
        print(f"  主流横截面: {df[df['is_main']].shape[0]}")
        print(f"  支流横截面: {df[~df['is_main']].shape[0]}")
        print(f"  F值范围: {df['F_final'].min():.3f} - {df['F_final'].max():.3f}")
        print(f"  F值均值: {df['F_final'].mean():.3f}")
        print(f"\n✓ 保存至: {output_csv}")
        print("=" * 70)

        return df
    else:
        print("\n✗ 无有效结果")
        return None


if __name__ == "__main__":
    result_df = run_step3()