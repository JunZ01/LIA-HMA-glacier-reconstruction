"""
连接OGGM中心线 - 诊断版
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import CRS, Transformer
import os
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_hierarchy.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'output_dir': "/mnt/d/paleoice_output/connect_centerlines/",
    'test_glacier_id': "RGI60-15.06881",
    'search_distance': 2500,
}


def get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return CRS.from_epsg(32600 + zone if lat >= 0 else 32700 + zone)


def transform_geometry(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def get_elevation_at_point(x_utm, y_utm, dem_path, utm_crs):
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(x_utm, y_utm)
    with rasterio.open(dem_path) as src:
        try:
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                return src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
        except:
            pass
    return np.nan


def flip_line_to_low_start(line_utm, dem_path, utm_crs):
    coords = list(line_utm.coords)
    start_elev = get_elevation_at_point(coords[0][0], coords[0][1], dem_path, utm_crs)
    end_elev = get_elevation_at_point(coords[-1][0], coords[-1][1], dem_path, utm_crs)

    if not np.isnan(start_elev) and not np.isnan(end_elev) and start_elev > end_elev:
        return LineString(coords[::-1])
    return line_utm


def trace_downslope_to_line(start_x, start_y, target_line, dem_path, utm_crs,
                            step_size=30, max_steps=500):
    """
    从起点沿坡度下降追踪到目标流线
    改进：增加向目标方向的引导
    """
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

    # 8个方向
    directions = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]

    path = [(start_x, start_y)]
    x, y = start_x, start_y
    visited = set()

    with rasterio.open(dem_path) as src:
        for step in range(max_steps):
            visited.add((round(x, 0), round(y, 0)))

            # 检查是否到达目标线
            current_pt = Point(x, y)
            dist_to_target = target_line.distance(current_pt)

            if dist_to_target < step_size * 2:
                nearest_pt = target_line.interpolate(target_line.project(current_pt))
                path.append((nearest_pt.x, nearest_pt.y))
                return LineString(path), "success"

            # 获取当前高程
            lon, lat = transformer.transform(x, y)
            try:
                row, col = src.index(lon, lat)
                current_elev = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
            except:
                break

            if current_elev < -9999:
                break

            # 目标方向
            nearest_on_target = target_line.interpolate(target_line.project(current_pt))
            target_dx = nearest_on_target.x - x
            target_dy = nearest_on_target.y - y
            target_dist = np.sqrt(target_dx**2 + target_dy**2)
            if target_dist > 0:
                target_dx /= target_dist
                target_dy /= target_dist

            # 搜索8个方向
            best_score = float('inf')
            best_x, best_y = None, None

            for dx, dy in directions:
                nx = x + dx * step_size
                ny = y + dy * step_size

                # 避免回头
                if (round(nx, 0), round(ny, 0)) in visited:
                    continue

                lon_n, lat_n = transformer.transform(nx, ny)
                try:
                    row_n, col_n = src.index(lon_n, lat_n)
                    if 0 <= row_n < src.height and 0 <= col_n < src.width:
                        elev_n = src.read(1, window=((row_n, row_n+1), (col_n, col_n+1)))[0, 0]

                        if elev_n < -9999:
                            continue

                        # 方向与目标方向的一致性
                        norm = np.sqrt(dx**2 + dy**2)
                        dir_score = -(dx/norm * target_dx + dy/norm * target_dy)  # -1到1，越小越好

                        # 高程惩罚：上坡惩罚大
                        elev_diff = elev_n - current_elev
                        if elev_diff > 0:
                            elev_penalty = elev_diff * 5  # 上坡重罚
                        else:
                            elev_penalty = elev_diff * 0.5  # 下坡轻微奖励

                        # 距离目标
                        dist_n = target_line.distance(Point(nx, ny))

                        # 综合得分
                        score = elev_penalty + dist_n * 0.3 + dir_score * 50

                        if score < best_score:
                            best_score = score
                            best_x, best_y = nx, ny
                except:
                    continue

            # 如果找不到更好的点
            if best_x is None:
                # 直接连到目标
                nearest_pt = target_line.interpolate(target_line.project(current_pt))
                path.append((nearest_pt.x, nearest_pt.y))
                return LineString(path), f"stuck_at_step_{step}"

            x, y = best_x, best_y
            path.append((x, y))

    # 超过最大步数
    current_pt = Point(x, y)
    nearest_pt = target_line.interpolate(target_line.project(current_pt))
    path.append((nearest_pt.x, nearest_pt.y))
    return LineString(path), "max_steps"


def connect_flowlines(flowlines_utm, flowlines_info, dem_path, utm_crs, search_dis):
    """连接流线网络"""

    corrected = {}
    for idx, line in flowlines_utm.items():
        corrected[idx] = flip_line_to_low_start(line, dem_path, utm_crs)

    connection_lines = []
    junction_info = {}

    levels = sorted(set(info['level'] for info in flowlines_info.values()))

    print("\n=== 流线层级信息 ===")
    for fl_idx in sorted(flowlines_info.keys()):
        info = flowlines_info[fl_idx]
        print(f"  FL{fl_idx}: level={info['level']}, parent={info['parent']}")

    print("\n=== 开始连接 ===")

    for level in levels:
        if level == 0:
            continue

        fl_at_level = [idx for idx, info in flowlines_info.items() if info['level'] == level]

        for fl_idx in fl_at_level:
            parent_idx = flowlines_info[fl_idx].get('parent')

            if parent_idx is None:
                print(f"  FL{fl_idx}: parent=None，跳过")
                junction_info[fl_idx] = {'connected': False, 'reason': 'no_parent'}
                continue

            if parent_idx not in corrected:
                print(f"  FL{fl_idx}: parent={parent_idx} 不在有效流线中，跳过")
                junction_info[fl_idx] = {'connected': False, 'reason': 'parent_not_found'}
                continue

            child_line = corrected[fl_idx]
            parent_line = corrected[parent_idx]

            child_start = Point(child_line.coords[0])
            nearest_dist = parent_line.distance(child_start)

            if nearest_dist > search_dis:
                print(f"  FL{fl_idx}: 距父流线{nearest_dist:.0f}m > {search_dis}m，跳过")
                junction_info[fl_idx] = {'connected': False, 'reason': 'too_far', 'distance': nearest_dist}
                continue

            # 追踪
            conn_line, status = trace_downslope_to_line(
                child_start.x, child_start.y,
                parent_line, dem_path, utm_crs,
                step_size=30, max_steps=500
            )

            nearest_pt = parent_line.interpolate(parent_line.project(Point(conn_line.coords[-1])))
            final_dist = nearest_pt.distance(Point(conn_line.coords[-1]))

            connection_lines.append({
                'geometry': conn_line,
                'child': fl_idx,
                'parent': parent_idx,
                'distance': nearest_dist,
                'path_length': conn_line.length,
                'status': status
            })

            junction_info[fl_idx] = {
                'connected': True,
                'parent': parent_idx,
                'x': child_start.x,
                'y': child_start.y,
                'parent_x': nearest_pt.x,
                'parent_y': nearest_pt.y,
                'distance': nearest_dist,
                'status': status
            }

            print(f"  FL{fl_idx} → FL{parent_idx}: 直线{nearest_dist:.0f}m, 路径{conn_line.length:.0f}m, 状态={status}, 终点误差={final_dist:.1f}m")

    return corrected, connection_lines, junction_info


def plot_network(glacier_utm, flowlines_utm, flowlines_info, connection_lines,
                 junction_info, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    level_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd'}

    ax1 = axes[0]
    if glacier_utm.geom_type == 'Polygon':
        x, y = glacier_utm.exterior.xy
        ax1.fill(x, y, alpha=0.15, color='lightblue', edgecolor='blue', linewidth=1)

    for fl_idx, fl_utm in flowlines_utm.items():
        level = flowlines_info.get(fl_idx, {}).get('level', 0)
        color = level_colors.get(level, 'gray')
        x, y = fl_utm.xy
        ax1.plot(x, y, color=color, linewidth=2, alpha=0.8)
        ax1.scatter(x[0], y[0], c='green', s=50, marker='o', zorder=5)
        ax1.scatter(x[-1], y[-1], c='red', s=30, marker='^', zorder=5)
        mid = len(x) // 2
        ax1.annotate(f'{fl_idx}(L{level})', (x[mid], y[mid]), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax1.set_aspect('equal')
    ax1.set_title('Before Connection\n(green=start/low, red=end/high)', fontsize=11)
    ax1.set_xlabel('UTM X (m)')
    ax1.set_ylabel('UTM Y (m)')

    ax2 = axes[1]
    if glacier_utm.geom_type == 'Polygon':
        x, y = glacier_utm.exterior.xy
        ax2.fill(x, y, alpha=0.15, color='lightblue', edgecolor='blue', linewidth=1)

    for fl_idx, fl_utm in flowlines_utm.items():
        level = flowlines_info.get(fl_idx, {}).get('level', 0)
        color = level_colors.get(level, 'gray')
        x, y = fl_utm.xy
        ax2.plot(x, y, color=color, linewidth=2, alpha=0.8)
        mid = len(x) // 2
        ax2.annotate(f'{fl_idx}', (x[mid], y[mid]), fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    for conn in connection_lines:
        x, y = conn['geometry'].xy
        # 成功的用实线，失败的用虚线
        if conn['status'] == 'success':
            ax2.plot(x, y, 'r-', linewidth=2, alpha=0.8)
        else:
            ax2.plot(x, y, 'r--', linewidth=2, alpha=0.5)

    for fl_idx, jinfo in junction_info.items():
        if jinfo.get('connected'):
            ax2.scatter(jinfo['x'], jinfo['y'], c='red', s=100, marker='*',
                       edgecolor='black', zorder=10, linewidths=1)
            ax2.scatter(jinfo['parent_x'], jinfo['parent_y'], c='blue', s=80,
                       marker='o', edgecolor='black', zorder=10, linewidths=1)

    success_count = sum(1 for c in connection_lines if c['status'] == 'success')
    ax2.set_title(f'After Connection\n({success_count}/{len(connection_lines)} success)', fontsize=11)
    ax2.set_xlabel('UTM X (m)')
    ax2.set_ylabel('UTM Y (m)')

    from matplotlib.lines import Line2D
    max_level = max(info['level'] for info in flowlines_info.values())
    legend_elements = [Line2D([0], [0], color=level_colors[i], lw=2, label=f'Level {i}')
                      for i in range(max_level + 1)]
    legend_elements.append(Line2D([0], [0], color='red', lw=2, label='Connection (success)'))
    legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label='Connection (partial)'))
    ax2.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 保存: {output_path}")


def run():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])

    test_glacier = glacier_gdf[glacier_gdf['OrigRGIId'] == CONFIG['test_glacier_id']]
    test_flowlines = flowlines_gdf[flowlines_gdf['MainRGI'] == CONFIG['test_glacier_id']].copy()

    if len(test_glacier) == 0:
        print(f"✗ 找不到冰川: {CONFIG['test_glacier_id']}")
        return

    glacier_geom = test_glacier.geometry.values[0]
    print(f"冰川: {CONFIG['test_glacier_id']}, 流线: {len(test_flowlines)}")

    # 打印原始数据的parent字段
    print("\n=== 原始数据 ===")
    for _, row in test_flowlines.iterrows():
        fl_idx = row.get('fl_index', row.name)
        print(f"  fl_index={fl_idx}, level={row.get('level')}, parent={row.get('parent')}")

    centroid = glacier_geom.centroid
    utm_crs = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    glacier_utm = transform_geometry(glacier_geom, src_crs, utm_crs)

    flowlines_utm = {}
    flowlines_info = {}

    for _, row in test_flowlines.iterrows():
        fl_idx = row.get('fl_index', row.name)
        fl_geom = row.geometry

        if fl_geom.geom_type == 'MultiLineString':
            fl_geom = max(list(fl_geom.geoms), key=lambda x: x.length)

        fl_utm = transform_geometry(fl_geom, src_crs, utm_crs)
        if fl_utm.length < 300:
            print(f"  FL{fl_idx}: 长度{fl_utm.length:.0f}m < 300m，跳过")
            continue

        flowlines_utm[fl_idx] = fl_utm

        parent_val = row.get('parent')
        if pd.notna(parent_val) and parent_val != -1:
            parent = int(parent_val)
        else:
            parent = None

        flowlines_info[fl_idx] = {
            'level': int(row.get('level', 0)),
            'parent': parent
        }

    print(f"\n有效流线: {len(flowlines_utm)}")

    corrected_lines, connection_lines, junction_info = connect_flowlines(
        flowlines_utm, flowlines_info, CONFIG['dem_path'], utm_crs, CONFIG['search_distance'])

    print(f"\n=== 连接统计 ===")
    for conn in connection_lines:
        print(f"  FL{conn['child']} → FL{conn['parent']}: {conn['status']}")

    plot_network(glacier_utm, corrected_lines, flowlines_info, connection_lines,
                junction_info, CONFIG['output_dir'] + 'flowline_network_v2.png')

    # 保存
    all_lines = []
    for fl_idx, line in corrected_lines.items():
        all_lines.append({
            'geometry': line,
            'fl_index': fl_idx,
            'level': flowlines_info[fl_idx]['level'],
            'type': 'centerline'
        })
    for conn in connection_lines:
        all_lines.append({
            'geometry': conn['geometry'],
            'fl_index': conn['child'],
            'level': -1,
            'type': 'connection',
            'status': conn['status']
        })

    gdf_out = gpd.GeoDataFrame(all_lines, crs=utm_crs)
    gdf_out.to_file(CONFIG['output_dir'] + 'connected_flowlines_v2.shp')
    print(f"✓ 保存: {CONFIG['output_dir']}connected_flowlines_v2.shp")


if __name__ == "__main__":
    run()