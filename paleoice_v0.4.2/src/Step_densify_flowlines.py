"""
流线加密脚本: 基于DEM水文分析提取新流线
==========================================

目的:
- 解决支流谷流线覆盖不足的问题
- 使用DEM水文分析提取更密的流线网络
- 与OGGM现有流线合并，保留层级关系

流程:
[1] 加载现有OGGM流线（已有冰厚数据）
[2] 基于DEM提取水文流线网络
[3] 在LIA边界内裁剪
[4] 去重：移除与OGGM流线重复的部分
[5] 建立层级关系：新流线连接到最近的父流线
[6] 输出合并后的流线文件，准备运行Step 1-4

关键原则:
- 新流线必须能追溯到某条现有流线（建立父子关系）
- 保持从末端向源头的方向一致性
- 不破坏现有OGGM流线的冰厚数据

依赖:
- pysheds (DEM水文分析)
- geopandas, rasterio, shapely

使用方法:
    python Step_Densify_Flowlines.py

作者: Claude (Anthropic)
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point, LineString, MultiLineString, shape, mapping
from shapely.ops import transform, linemerge, nearest_points, unary_union
from pyproj import CRS, Transformer
from scipy.ndimage import label
from scipy.spatial import cKDTree
from collections import defaultdict
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    # 输入文件
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_hierarchy.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",

    # 输出目录
    'output_dir': "/mnt/d/paleoice_output/densified_flowlines/",

    # 测试冰川
    'test_glacier_id': "RGI60-15.06881",

    # 水文分析参数
    'flow_accumulation_threshold': 3000,  # 流量累积阈值（像元数）- 较小=更多流线
    'min_flowline_length_m': 500,  # 最小流线长度（米）

    # 去重参数
    'duplicate_buffer_m': 500,  # 与OGGM流线的重复判定缓冲区（米）
    'snap_distance_m': 500,  # 新流线末端吸附到父流线的最大距离

    # 流线采样参数
    'flowline_spacing_m': 50,  # 流线采样间距（米）
}


# ============================================================================
# 工具函数
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


def get_elevation_at_point(x, y, dem_data, transform):
    """从DEM数组获取高程"""
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)
    if 0 <= row < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
        val = dem_data[row, col]
        return val if val > -9999 else np.nan
    return np.nan


# ============================================================================
# DEM水文分析
# ============================================================================

def fill_depressions(dem, nodata=-9999):
    """
    简化的洼地填充算法
    使用迭代填充方法
    """
    filled = dem.copy()
    filled[filled <= nodata] = np.nan

    # 迭代填充
    max_iterations = 1000
    for _ in range(max_iterations):
        # 计算8邻域最小值
        padded = np.pad(filled, 1, mode='edge')
        neighbors = np.stack([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2], padded[1:-1, 2:],
            padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
        ])
        min_neighbor = np.nanmin(neighbors, axis=0)

        # 找到需要填充的洼地
        mask = filled < min_neighbor
        if not np.any(mask):
            break
        filled[mask] = min_neighbor[mask]

    return filled


def calculate_flow_direction_d8(dem):
    """
    计算D8流向
    返回流向编码 (1-8) 和 有效掩膜

    流向编码:
        1  2  3
        8  *  4
        7  6  5
    """
    rows, cols = dem.shape
    flow_dir = np.zeros((rows, cols), dtype=np.int8)

    # 方向偏移 (row_offset, col_offset)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]
    dir_codes = [1, 2, 3, 8, 4, 7, 6, 5]

    # 对角线距离系数
    dist = [1.414, 1.0, 1.414, 1.0, 1.0, 1.414, 1.0, 1.414]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.isnan(dem[i, j]):
                continue

            max_slope = 0
            max_dir = 0

            for k, (di, dj) in enumerate(directions):
                ni, nj = i + di, j + dj
                if np.isnan(dem[ni, nj]):
                    continue

                drop = dem[i, j] - dem[ni, nj]
                slope = drop / dist[k]

                if slope > max_slope:
                    max_slope = slope
                    max_dir = dir_codes[k]

            flow_dir[i, j] = max_dir

    return flow_dir


def calculate_flow_accumulation(flow_dir):
    """
    计算流量累积
    """
    rows, cols = flow_dir.shape
    acc = np.ones((rows, cols), dtype=np.float32)

    # 方向到偏移的映射
    dir_to_offset = {
        1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
        8: (0, -1), 4: (0, 1),
        7: (1, -1), 6: (1, 0), 5: (1, 1)
    }

    # 反向映射：哪些方向的邻居会流向当前格子
    reverse_dirs = {
        1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4
    }

    # 计算入度
    in_degree = np.zeros((rows, cols), dtype=np.int32)
    for i in range(rows):
        for j in range(cols):
            d = flow_dir[i, j]
            if d == 0:
                continue
            di, dj = dir_to_offset[d]
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                in_degree[ni, nj] += 1

    # 拓扑排序累积
    from collections import deque
    queue = deque()

    # 初始化：入度为0的格子
    for i in range(rows):
        for j in range(cols):
            if in_degree[i, j] == 0 and flow_dir[i, j] != 0:
                queue.append((i, j))

    while queue:
        i, j = queue.popleft()
        d = flow_dir[i, j]
        if d == 0:
            continue

        di, dj = dir_to_offset[d]
        ni, nj = i + di, j + dj

        if 0 <= ni < rows and 0 <= nj < cols:
            acc[ni, nj] += acc[i, j]
            in_degree[ni, nj] -= 1
            if in_degree[ni, nj] == 0:
                queue.append((ni, nj))

    return acc


def extract_stream_network(flow_acc, flow_dir, threshold, transform):
    """
    从流量累积和流向提取河流网络

    Args:
        flow_acc: 流量累积数组
        flow_dir: 流向数组
        threshold: 流量累积阈值
        transform: rasterio transform

    Returns:
        list of LineString geometries
    """
    # 标记河流像元
    stream_mask = flow_acc >= threshold

    # 连通区域标记
    labeled, num_features = label(stream_mask)

    # 方向到偏移的映射
    dir_to_offset = {
        1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
        8: (0, -1), 4: (0, 1),
        7: (1, -1), 6: (1, 0), 5: (1, 1)
    }

    # 对每个连通区域追踪流线
    streams = []
    rows, cols = flow_acc.shape

    # 找到每个河流区域的起点（流量最大但没有上游河流流入的点）
    for region_id in range(1, num_features + 1):
        region_mask = labeled == region_id
        region_indices = np.argwhere(region_mask)

        if len(region_indices) == 0:
            continue

        # 找起点：该区域内流量最大的点
        max_acc = 0
        start_point = None

        for i, j in region_indices:
            if flow_acc[i, j] > max_acc:
                max_acc = flow_acc[i, j]
                start_point = (i, j)

        if start_point is None:
            continue

        # 从起点向上游追踪
        # 这里我们从高流量点向低流量点追踪（实际是向下游）
        # 然后反转得到从源头到末端的流线

        # 追踪下游
        path = [start_point]
        visited = set([start_point])
        current = start_point

        while True:
            i, j = current
            d = flow_dir[i, j]
            if d == 0:
                break

            di, dj = dir_to_offset.get(d, (0, 0))
            ni, nj = i + di, j + dj

            if not (0 <= ni < rows and 0 <= nj < cols):
                break
            if (ni, nj) in visited:
                break
            if not stream_mask[ni, nj]:
                # 允许继续追踪到边界
                pass

            visited.add((ni, nj))
            path.append((ni, nj))
            current = (ni, nj)

            if len(path) > 10000:  # 防止无限循环
                break

        if len(path) < 3:
            continue

        # 转换为地理坐标
        coords = []
        for i, j in path:
            x = transform.c + j * transform.a + transform.a / 2
            y = transform.f + i * transform.e + transform.e / 2
            coords.append((x, y))

        if len(coords) >= 2:
            line = LineString(coords)
            streams.append(line)

    return streams


def trace_upstream_tributaries(flow_dir, flow_acc, stream_mask, transform,
                               min_length_pixels=10):
    """
    追踪上游支流

    从每个河流像元向上游追踪所有汇入的支流
    """
    rows, cols = flow_dir.shape

    # 反向流向映射
    reverse_dir_to_offset = {
        5: (-1, -1), 6: (-1, 0), 7: (-1, 1),
        4: (0, -1), 8: (0, 1),
        3: (1, -1), 2: (1, 0), 1: (1, 1)
    }

    dir_to_offset = {
        1: (-1, -1), 2: (-1, 0), 3: (-1, 1),
        8: (0, -1), 4: (0, 1),
        7: (1, -1), 6: (1, 0), 5: (1, 1)
    }

    # 找到所有河流像元
    stream_cells = np.argwhere(stream_mask)

    all_tributaries = []
    visited_global = set()

    for start_i, start_j in stream_cells:
        # 从每个河流像元向上游追踪
        # 检查8个方向，找到流向当前格子的邻居
        for d, (di, dj) in reverse_dir_to_offset.items():
            ni, nj = start_i + di, start_j + dj

            if not (0 <= ni < rows and 0 <= nj < cols):
                continue

            # 检查邻居是否流向当前格子
            neighbor_dir = flow_dir[ni, nj]
            if neighbor_dir == 0:
                continue

            ndi, ndj = dir_to_offset.get(neighbor_dir, (0, 0))
            target_i, target_j = ni + ndi, nj + ndj

            if (target_i, target_j) != (start_i, start_j):
                continue

            # 如果邻居不在主河流上且未访问过，追踪这条支流
            if stream_mask[ni, nj]:
                continue
            if (ni, nj) in visited_global:
                continue

            # 向上游追踪
            path = [(start_i, start_j), (ni, nj)]
            visited_global.add((ni, nj))
            current = (ni, nj)

            while True:
                ci, cj = current
                found_upstream = False

                # 找流向当前格子的上游
                for ud, (udi, udj) in reverse_dir_to_offset.items():
                    ui, uj = ci + udi, cj + udj

                    if not (0 <= ui < rows and 0 <= uj < cols):
                        continue
                    if (ui, uj) in visited_global:
                        continue

                    upstream_dir = flow_dir[ui, uj]
                    if upstream_dir == 0:
                        continue

                    udi2, udj2 = dir_to_offset.get(upstream_dir, (0, 0))
                    if (ui + udi2, uj + udj2) == (ci, cj):
                        # 找到上游
                        visited_global.add((ui, uj))
                        path.append((ui, uj))
                        current = (ui, uj)
                        found_upstream = True
                        break

                if not found_upstream:
                    break
                if len(path) > 5000:
                    break

            if len(path) >= min_length_pixels:
                # 转换为地理坐标（反转使其从源头到末端）
                coords = []
                for pi, pj in reversed(path):
                    x = transform.c + pj * transform.a + transform.a / 2
                    y = transform.f + pi * transform.e + transform.e / 2
                    coords.append((x, y))

                if len(coords) >= 2:
                    line = LineString(coords)
                    all_tributaries.append(line)

    return all_tributaries


# ============================================================================
# 流线处理
# ============================================================================

def clip_flowlines_to_glacier(flowlines, glacier_geom):
    """将流线裁剪到冰川边界内"""
    clipped = []
    for line in flowlines:
        try:
            intersection = line.intersection(glacier_geom)
            if intersection.is_empty:
                continue

            if isinstance(intersection, LineString):
                if intersection.length > 0:
                    clipped.append(intersection)
            elif isinstance(intersection, MultiLineString):
                for part in intersection.geoms:
                    if part.length > 0:
                        clipped.append(part)
        except:
            continue

    return clipped


def remove_duplicate_flowlines(new_flowlines, existing_flowlines, buffer_m=100):
    """
    移除与现有流线重复的新流线

    Args:
        new_flowlines: 新提取的流线列表
        existing_flowlines: 现有OGGM流线列表
        buffer_m: 重复判定缓冲区（米）

    Returns:
        去重后的新流线列表
    """
    if not existing_flowlines:
        return new_flowlines

    # 合并现有流线的缓冲区
    existing_buffer = unary_union([line.buffer(buffer_m) for line in existing_flowlines])

    unique_flowlines = []
    for line in new_flowlines:
        # 计算与现有流线重叠的比例
        intersection = line.intersection(existing_buffer)
        if intersection.is_empty:
            unique_flowlines.append(line)
        else:
            overlap_ratio = intersection.length / line.length if line.length > 0 else 1
            # 如果重叠少于50%，保留非重叠部分
            if overlap_ratio < 0.5:
                difference = line.difference(existing_buffer)
                if not difference.is_empty:
                    if isinstance(difference, LineString) and difference.length > CONFIG['min_flowline_length_m']:
                        unique_flowlines.append(difference)
                    elif isinstance(difference, MultiLineString):
                        for part in difference.geoms:
                            if part.length > CONFIG['min_flowline_length_m']:
                                unique_flowlines.append(part)

    return unique_flowlines


def orient_flowline_downslope(line, dem_data, transform):
    """
    确保流线方向从高到低（源头→末端）
    然后反转使其从末端→源头（用于Benn-Hulton迭代）

    实际上Benn-Hulton需要从末端向源头迭代，
    所以我们最终要让起点在低处（末端），终点在高处（源头）
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return line

    start_elev = get_elevation_at_point(coords[0][0], coords[0][1], dem_data, transform)
    end_elev = get_elevation_at_point(coords[-1][0], coords[-1][1], dem_data, transform)

    if np.isnan(start_elev) or np.isnan(end_elev):
        return line

    # 如果起点比终点高，说明当前是从源头到末端，需要反转
    if start_elev > end_elev:
        return LineString(coords[::-1])

    return line


def find_parent_flowline(new_line, existing_flowlines, existing_thickness_points,
                         max_snap_distance):
    """
    为新流线找到父流线

    策略：
    1. 新流线的起点（低端/末端）应该连接到某条现有流线
    2. 找到最近的现有流线作为父流线
    3. 返回父流线ID和连接点

    Args:
        new_line: 新流线（起点在末端/低处）
        existing_flowlines: 现有流线字典 {id: geometry}
        existing_thickness_points: 现有冰厚点 DataFrame
        max_snap_distance: 最大吸附距离

    Returns:
        dict: {
            'parent_id': 父流线ID,
            'snap_point': 吸附点坐标,
            'distance': 距离
        } 或 None
    """
    start_point = Point(new_line.coords[0])

    min_distance = float('inf')
    best_parent = None
    best_snap_point = None

    for fl_id, fl_geom in existing_flowlines.items():
        # 计算新流线起点到现有流线的最近距离
        nearest = nearest_points(start_point, fl_geom)
        distance = start_point.distance(nearest[1])

        if distance < min_distance and distance < max_snap_distance:
            min_distance = distance
            best_parent = fl_id
            best_snap_point = (nearest[1].x, nearest[1].y)

    if best_parent is not None:
        return {
            'parent_id': best_parent,
            'snap_point': best_snap_point,
            'distance': min_distance
        }

    return None


def connect_to_parent(new_line, snap_point):
    """
    将新流线连接到父流线上的吸附点

    Args:
        new_line: 新流线
        snap_point: 父流线上的吸附点坐标 (x, y)

    Returns:
        连接后的新流线（起点为吸附点）
    """
    coords = list(new_line.coords)
    # 将吸附点作为新的起点
    new_coords = [snap_point] + coords
    return LineString(new_coords)


# ============================================================================
# 主流程
# ============================================================================

def process_single_glacier(glacier_id, glacier_gdf, flowlines_gdf, dem_path,
                           output_dir, config):
    """
    处理单个冰川的流线加密

    Returns:
        dict: 处理结果
    """
    print(f"\n{'=' * 60}")
    print(f"处理冰川: {glacier_id}")
    print(f"{'=' * 60}")

    # 1. 获取冰川边界
    glacier_row = glacier_gdf[glacier_gdf['OrigRGIId'] == glacier_id]
    if len(glacier_row) == 0:
        print(f"  ✗ 找不到冰川边界")
        return None

    glacier_geom = glacier_row.geometry.values[0]

    # 2. 获取UTM坐标系
    centroid = glacier_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    print(f"  UTM Zone: {utm_zone}N")

    # 3. 投影到UTM
    glacier_utm = transform_geometry(glacier_geom, src_crs, utm_crs)
    print(f"  冰川面积: {glacier_utm.area / 1e6:.2f} km²")

    # 4. 加载现有OGGM流线
    glacier_flowlines = flowlines_gdf[flowlines_gdf['MainRGI'] == glacier_id].copy()
    print(f"  现有OGGM流线: {len(glacier_flowlines)} 条")

    existing_flowlines_utm = {}
    for idx, row in glacier_flowlines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'MultiLineString':
            geom = max(list(geom.geoms), key=lambda x: x.length)
        geom_utm = transform_geometry(geom, src_crs, utm_crs)
        existing_flowlines_utm[idx] = geom_utm

    # 5. 裁剪DEM到冰川范围
    print(f"\n[1] 裁剪DEM到冰川范围...")

    # 添加缓冲区以确保完整覆盖
    glacier_buffer = glacier_geom.buffer(0.01)  # ~1km缓冲区

    with rasterio.open(dem_path) as src:
        try:
            dem_clipped, dem_transform = rio_mask(src, [mapping(glacier_buffer)],
                                                  crop=True, nodata=-9999)
            dem_clipped = dem_clipped[0]
            dem_crs = src.crs
        except Exception as e:
            print(f"  ✗ DEM裁剪失败: {e}")
            return None

    print(f"  DEM大小: {dem_clipped.shape}")
    print(f"  DEM分辨率: {abs(dem_transform.a):.1f}m")

    # 6. 填充洼地
    print(f"\n[2] 填充DEM洼地...")
    dem_filled = fill_depressions(dem_clipped)
    print(f"  ✓ 洼地填充完成")

    # 7. 计算流向
    print(f"\n[3] 计算流向...")
    flow_dir = calculate_flow_direction_d8(dem_filled)
    print(f"  ✓ 流向计算完成")

    # 8. 计算流量累积
    print(f"\n[4] 计算流量累积...")
    flow_acc = calculate_flow_accumulation(flow_dir)
    print(f"  流量累积范围: {np.min(flow_acc[flow_acc > 0]):.0f} - {np.max(flow_acc):.0f}")

    # 9. 提取河流网络
    print(f"\n[5] 提取河流网络...")
    threshold = config['flow_accumulation_threshold']

    # 主河流网络
    stream_mask = flow_acc >= threshold
    print(f"  阈值: {threshold}, 河流像元数: {np.sum(stream_mask)}")

    # 提取主流线
    main_streams = extract_stream_network(flow_acc, flow_dir, threshold, dem_transform)
    print(f"  主流线数: {len(main_streams)}")

    # 提取支流
    tributaries = trace_upstream_tributaries(flow_dir, flow_acc, stream_mask,
                                             dem_transform, min_length_pixels=10)
    print(f"  支流数: {len(tributaries)}")

    all_new_streams = main_streams + tributaries
    print(f"  总新流线数: {len(all_new_streams)}")

    # 10. 裁剪到冰川边界
    print(f"\n[6] 裁剪到冰川边界...")

    # 注意：DEM裁剪后的坐标是WGS84
    clipped_streams = clip_flowlines_to_glacier(all_new_streams, glacier_geom)
    print(f"  裁剪后流线数: {len(clipped_streams)}")

    # 投影到UTM
    clipped_streams_utm = []
    for line in clipped_streams:
        line_utm = transform_geometry(line, src_crs, utm_crs)
        if line_utm.length >= config['min_flowline_length_m']:
            clipped_streams_utm.append(line_utm)

    print(f"  有效流线数 (>{config['min_flowline_length_m']}m): {len(clipped_streams_utm)}")

    # 11. 去重
    print(f"\n[7] 去除与OGGM流线重复的部分...")
    existing_list = list(existing_flowlines_utm.values())
    unique_streams = remove_duplicate_flowlines(clipped_streams_utm, existing_list,
                                                config['duplicate_buffer_m'])
    print(f"  去重后流线数: {len(unique_streams)}")

    # 12. 校正方向并建立层级关系
    print(f"\n[8] 校正方向并建立层级关系...")

    # 重新加载DEM用于高程查询（UTM坐标）
    with rasterio.open(dem_path) as src:
        dem_data_full = src.read(1)
        dem_transform_full = src.transform

    new_flowlines_data = []

    for i, line in enumerate(unique_streams):
        # 投影回WGS84查询高程，然后再投影回UTM
        line_wgs = transform_geometry(line, utm_crs, src_crs)
        line_oriented_wgs = orient_flowline_downslope(line_wgs, dem_data_full, dem_transform_full)
        line_oriented = transform_geometry(line_oriented_wgs, src_crs, utm_crs)

        # 找父流线
        parent_info = find_parent_flowline(line_oriented, existing_flowlines_utm,
                                           None, config['snap_distance_m'])

        if parent_info is not None:
            # 连接到父流线
            connected_line = connect_to_parent(line_oriented, parent_info['snap_point'])

            new_flowlines_data.append({
                'geometry': connected_line,
                'parent_id': parent_info['parent_id'],
                'snap_distance': parent_info['distance'],
                'length_m': connected_line.length,
                'is_new': True
            })
        else:
            # 没有找到父流线，可能是孤立的小流线
            # 可以选择保留或丢弃
            new_flowlines_data.append({
                'geometry': line_oriented,
                'parent_id': None,
                'snap_distance': None,
                'length_m': line_oriented.length,
                'is_new': True
            })

    connected_count = sum(1 for d in new_flowlines_data if d['parent_id'] is not None)
    print(f"  成功连接到父流线: {connected_count}/{len(new_flowlines_data)}")

    # 13. 合并输出
    print(f"\n[9] 生成输出...")

    # 创建GeoDataFrame
    # 首先添加现有OGGM流线
    all_flowlines = []

    # 现有流线保持原样
    for idx, row in glacier_flowlines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'MultiLineString':
            geom = max(list(geom.geoms), key=lambda x: x.length)

        geom_utm = transform_geometry(geom, src_crs, utm_crs)

        all_flowlines.append({
            'FlowlineID': f'OGGM_{idx}',
            'OriginalID': idx,
            'MainRGI': glacier_id,
            'IsNew': False,
            'ParentID': row.get('NEAR_FID', None),  # 保留原有父流线关系
            'Length_m': geom_utm.length,
            'geometry': geom_utm
        })

    # 添加新流线
    for i, data in enumerate(new_flowlines_data):
        if data['parent_id'] is not None:
            parent_str = f'OGGM_{data["parent_id"]}'
        else:
            parent_str = None

        all_flowlines.append({
            'FlowlineID': f'NEW_{i}',
            'OriginalID': None,
            'MainRGI': glacier_id,
            'IsNew': True,
            'ParentID': parent_str,
            'Length_m': data['length_m'],
            'geometry': data['geometry']
        })

    # 创建GeoDataFrame
    output_gdf = gpd.GeoDataFrame(all_flowlines, crs=utm_crs)

    # 转回WGS84保存
    output_gdf_wgs = output_gdf.to_crs(epsg=4326)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{glacier_id.replace(".", "_")}_densified_flowlines.shp')
    output_gdf_wgs.to_file(output_path)
    print(f"  ✓ 保存: {output_path}")

    # 统计
    n_oggm = sum(1 for d in all_flowlines if not d['IsNew'])
    n_new = sum(1 for d in all_flowlines if d['IsNew'])
    total_length_oggm = sum(d['Length_m'] for d in all_flowlines if not d['IsNew'])
    total_length_new = sum(d['Length_m'] for d in all_flowlines if d['IsNew'])

    print(f"\n  === 统计 ===")
    print(f"  OGGM流线: {n_oggm} 条, 总长度 {total_length_oggm / 1000:.1f} km")
    print(f"  新增流线: {n_new} 条, 总长度 {total_length_new / 1000:.1f} km")
    print(f"  流线密度提升: {(n_oggm + n_new) / n_oggm:.1f}x")

    return {
        'glacier_id': glacier_id,
        'output_path': output_path,
        'n_oggm': n_oggm,
        'n_new': n_new,
        'n_connected': connected_count,
        'total_length_oggm_km': total_length_oggm / 1000,
        'total_length_new_km': total_length_new / 1000
    }


def run():
    """主函数"""
    print("=" * 70)
    print("流线加密脚本: 基于DEM水文分析")
    print("=" * 70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 加载数据
    print("\n[加载数据]")
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])
    print(f"  冰川数: {len(glacier_gdf)}")
    print(f"  流线数: {len(flowlines_gdf)}")

    # 处理测试冰川
    result = process_single_glacier(
        CONFIG['test_glacier_id'],
        glacier_gdf,
        flowlines_gdf,
        CONFIG['dem_path'],
        CONFIG['output_dir'],
        CONFIG
    )

    if result:
        print(f"\n{'=' * 70}")
        print("✓ 流线加密完成!")
        print(f"{'=' * 70}")
        print(f"\n下一步:")
        print(f"  1. 检查输出文件: {result['output_path']}")
        print(f"  2. 对新流线运行 Step 1-4 计算冰厚")
        print(f"  3. 合并所有冰厚点，运行 Step 5 插值")

    return result


if __name__ == "__main__":
    run()