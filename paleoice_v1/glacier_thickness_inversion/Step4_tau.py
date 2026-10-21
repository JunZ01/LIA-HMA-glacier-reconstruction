"""
Step 4 V8: 冰厚反演 (使用已连接流线)
================================
流程：
1. 加载已连接的流线和层级关系
2. 按层级计算冰厚：
   - 主流线：从最低点向上游迭代
   - 支流：从交汇点继承主流冰面，向上游迭代
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 物理常数
# ============================================================================
RHO_ICE = 900
G = 9.81
MIN_SLOPE = 1.0
MAX_DS = 30
MIN_F, MAX_F, DEFAULT_F = 0.445, 1.0, 0.8

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_connected.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'output_dir': "/mnt/d/paleoice_output/step4_v8/",
    'f_factor_csv': "/mnt/d/paleoice_output/step3_diagnostic/step3_ffactor_data.csv",
    'test_glacier_id': 2692,
    'flowline_spacing_m': 50,
    'catchment_grid_spacing': 100,
    'use_dynamic_f': True,
    'min_flowline_length': 500,
}


# ============================================================================
# 工具函数
# ============================================================================

def get_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return CRS.from_epsg(32600 + zone if lat >= 0 else 32700 + zone), zone


def transform_geometry(geom, src_crs, dst_crs):
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return transform(transformer.transform, geom)


def get_elevation_at_point(x_utm, y_utm, dem_path, utm_crs):
    """获取单点高程"""
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
    lon, lat = transformer.transform(x_utm, y_utm)
    with rasterio.open(dem_path) as src:
        try:
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                return val if val > -9999 else np.nan
        except:
            pass
    return np.nan


def sample_dem_along_line(line_utm, dem_path, utm_crs, spacing_m=50):
    """沿流线采样DEM"""
    transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

    length = line_utm.length
    num_points = max(int(length / spacing_m) + 1, 2)
    distances = np.linspace(0, length, num_points)

    points = [line_utm.interpolate(d) for d in distances]
    xs_utm = np.array([p.x for p in points])
    ys_utm = np.array([p.y for p in points])

    with rasterio.open(dem_path) as src:
        elevs = []
        for x, y in zip(xs_utm, ys_utm):
            lon, lat = transformer.transform(x, y)
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
        'elevs': np.array(elevs),
        'length': length
    }


def calculate_slope(elevs, distances):
    """计算坡度"""
    n = len(elevs)
    slopes = np.full(n, MIN_SLOPE)

    for i in range(n):
        if np.isnan(elevs[i]):
            continue
        if i == 0 and i + 1 < n and not np.isnan(elevs[i+1]):
            dz, dx = abs(elevs[i+1] - elevs[i]), distances[i+1] - distances[i]
        elif i == n - 1 and not np.isnan(elevs[i-1]):
            dz, dx = abs(elevs[i] - elevs[i-1]), distances[i] - distances[i-1]
        elif 0 < i < n-1 and not np.isnan(elevs[i-1]) and not np.isnan(elevs[i+1]):
            dz, dx = abs(elevs[i+1] - elevs[i-1]), distances[i+1] - distances[i-1]
        else:
            continue
        if dx > 0:
            slopes[i] = max(np.degrees(np.arctan(dz / dx)), MIN_SLOPE)

    return slopes


def estimate_tau(area_km2, mean_slope_deg):
    """估算剪应力τ"""
    area_km2 = max(area_km2, 0.1)
    tau_base = np.clip(50 + 30 * np.log10(area_km2), 50, 150)
    if mean_slope_deg > 15:
        tau_base *= np.sin(np.radians(mean_slope_deg)) / np.sin(np.radians(12))
    return np.clip(tau_base, 50, 180) * 1000


def calculate_catchment_areas(flowlines_utm, glacier_utm, spacing=100):
    """计算各流线汇水面积"""
    bounds = glacier_utm.bounds
    xs = np.arange(bounds[0], bounds[2], spacing)
    ys = np.arange(bounds[1], bounds[3], spacing)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    inside = np.array([glacier_utm.contains(Point(p)) for p in grid_points])
    inside_points = grid_points[inside]
    if len(inside_points) == 0:
        return {}

    fl_samples, fl_ids = [], []
    for fl_idx, fl in flowlines_utm.items():
        for d in np.linspace(0, fl.length, max(int(fl.length / 25), 3)):
            pt = fl.interpolate(d)
            fl_samples.append([pt.x, pt.y])
            fl_ids.append(fl_idx)

    tree = cKDTree(np.array(fl_samples))
    _, nearest = tree.query(inside_points)
    assigned_ids = np.array(fl_ids)[nearest]

    cell_area = (spacing ** 2) / 1e6
    return {fl_idx: np.sum(assigned_ids == fl_idx) * cell_area for fl_idx in flowlines_utm}


# ============================================================================
# 流线方向校正
# ============================================================================

def flip_line_to_low_start(line_utm, dem_path, utm_crs):
    """将流线翻转为低端在前（起点为低）"""
    coords = list(line_utm.coords)
    start_elev = get_elevation_at_point(coords[0][0], coords[0][1], dem_path, utm_crs)
    end_elev = get_elevation_at_point(coords[-1][0], coords[-1][1], dem_path, utm_crs)

    if not np.isnan(start_elev) and not np.isnan(end_elev) and start_elev > end_elev:
        return LineString(coords[::-1])
    return line_utm


# ============================================================================
# Benn-Hulton 冰厚迭代
# ============================================================================

def benn_hulton_iterate(elevs, distances, tau, F_arr, start_idx, direction,
                        thickness, ice_surface, slopes_rad):
    """从start_idx向direction方向迭代计算冰厚"""
    n = len(elevs)
    i = start_idx + direction

    while 0 <= i < n:
        prev_i = i - direction

        if np.isnan(elevs[i]) or np.isnan(elevs[prev_i]):
            thickness[i] = thickness[prev_i]
            ice_surface[i] = elevs[i] + thickness[i] if not np.isnan(elevs[i]) else np.nan
            i += direction
            continue

        dx = abs(distances[i] - distances[prev_i])
        if dx <= 0:
            thickness[i] = thickness[prev_i]
            ice_surface[i] = elevs[i] + thickness[i]
            i += direction
            continue

        sin_alpha = max(np.sin(slopes_rad[i]), np.sin(np.radians(MIN_SLOPE)))
        H_pp = min(tau / (RHO_ICE * G * sin_alpha * F_arr[i]), 1500)

        H_curr = H_pp
        for _ in range(10):
            H_avg = max((thickness[prev_i] + H_curr) / 2, 1)
            dS = min(tau * dx / (RHO_ICE * G * F_arr[i] * H_avg), MAX_DS)
            S_curr = ice_surface[prev_i] + dS
            H_new = S_curr - elevs[i]
            if abs(H_new - H_curr) < 1:
                break
            H_curr = H_new

        thickness[i] = np.clip(H_curr, 0, 1000)
        ice_surface[i] = elevs[i] + thickness[i]

        i += direction


def ice_thickness_main_flowline(elevs, distances, tau, F):
    """主流线冰厚计算：从最低点向上游迭代"""
    n = len(elevs)

    F_arr = np.full(n, F) if np.isscalar(F) else np.interp(
        np.linspace(0, 1, n), np.linspace(0, 1, len(F)), F)
    F_arr = np.clip(F_arr, MIN_F, MAX_F)

    if np.sum(~np.isnan(elevs)) < 3:
        return np.zeros(n), elevs.copy(), 0, 1

    start_idx = 0
    upstream_direction = +1

    thickness = np.zeros(n)
    ice_surface = elevs.copy()
    slopes_rad = np.radians(calculate_slope(elevs, distances))

    thickness[start_idx] = 0
    ice_surface[start_idx] = elevs[start_idx]

    benn_hulton_iterate(elevs, distances, tau, F_arr, start_idx, upstream_direction,
                        thickness, ice_surface, slopes_rad)

    return thickness, ice_surface, start_idx, upstream_direction


def ice_thickness_tributary(elevs, distances, tau, F, parent_ice_surface_at_junction):
    """支流冰厚计算：从交汇点继承父流线冰面，向上游迭代"""
    n = len(elevs)

    F_arr = np.full(n, F) if np.isscalar(F) else np.interp(
        np.linspace(0, 1, n), np.linspace(0, 1, len(F)), F)
    F_arr = np.clip(F_arr, MIN_F, MAX_F)

    if np.sum(~np.isnan(elevs)) < 3:
        return np.zeros(n), elevs.copy()

    junction_idx = 0
    upstream_direction = +1

    thickness = np.zeros(n)
    ice_surface = elevs.copy()
    slopes_rad = np.radians(calculate_slope(elevs, distances))

    ice_surface[junction_idx] = parent_ice_surface_at_junction
    thickness[junction_idx] = max(parent_ice_surface_at_junction - elevs[junction_idx], 0)
    thickness[junction_idx] = min(thickness[junction_idx], 1000)

    benn_hulton_iterate(elevs, distances, tau, F_arr, junction_idx, upstream_direction,
                        thickness, ice_surface, slopes_rad)

    return thickness, ice_surface


# ============================================================================
# 可视化
# ============================================================================

def plot_results(glacier_utm, flowlines_utm, all_results, flowlines_info,
                 junction_info, output_path):
    """诊断图"""
    n_flowlines = len(all_results)
    n_cols = min(4, n_flowlines)
    n_rows = (n_flowlines + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(20, 8 + 3 * n_rows))

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2 + n_rows, 4, figure=fig, height_ratios=[1, 1] + [0.8] * n_rows)

    level_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728', 4: '#9467bd'}

    # ===== 左上：流线网络 =====
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    if glacier_utm.geom_type == 'Polygon':
        x, y = glacier_utm.exterior.xy
        ax1.fill(x, y, alpha=0.15, color='lightblue', edgecolor='blue', linewidth=1)

    for fl_idx, fl_utm in flowlines_utm.items():
        level = flowlines_info.get(fl_idx, {}).get('level', 0)
        color = level_colors.get(level, 'gray')
        if fl_utm.geom_type == 'LineString':
            x, y = fl_utm.xy
            ax1.plot(x, y, color=color, linewidth=2, alpha=0.8)
            ax1.scatter(x[0], y[0], c='green', s=50, marker='o', zorder=5)
            mid = len(x) // 2
            ax1.annotate(f'{fl_idx}', (x[mid], y[mid]), fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    for fl_idx, jinfo in junction_info.items():
        if jinfo.get('connected'):
            ax1.scatter(jinfo['junction_x'], jinfo['junction_y'], c='red', s=150, marker='*',
                       edgecolor='black', zorder=10, linewidths=1)

    ax1.set_aspect('equal')
    ax1.set_title('Flowline Network\n(green=start, red★=tributary junction)', fontsize=11)
    ax1.set_xlabel('UTM X (m)')
    ax1.set_ylabel('UTM Y (m)')

    from matplotlib.lines import Line2D
    max_level = max(info.get('level', 0) for info in flowlines_info.values()) if flowlines_info else 0
    legend_elements = [
        Line2D([0], [0], color=level_colors.get(i, 'gray'), lw=2, label=f'Level {i}')
        for i in range(max_level + 1)
    ]
    legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                                   markersize=15, label='Junction'))
    ax1.legend(handles=legend_elements, loc='upper right')

    # ===== 右上：冰厚分布 =====
    ax2 = fig.add_subplot(gs[0:2, 2:4])

    if glacier_utm.geom_type == 'Polygon':
        x, y = glacier_utm.exterior.xy
        ax2.fill(x, y, alpha=0.1, color='gray', edgecolor='gray', linewidth=0.5)

    all_x, all_y, all_h = [], [], []
    for df in all_results.values():
        all_x.extend(df['x_utm'].values)
        all_y.extend(df['y_utm'].values)
        all_h.extend(df['thickness_m'].values)

    sc = ax2.scatter(all_x, all_y, c=all_h, cmap='viridis', s=15, vmin=0)
    plt.colorbar(sc, ax=ax2, label='Ice Thickness (m)')
    ax2.set_aspect('equal')
    ax2.set_title('Ice Thickness Distribution', fontsize=12)
    ax2.set_xlabel('UTM X (m)')
    ax2.set_ylabel('UTM Y (m)')

    # ===== 下方：每条流线独立剖面 =====
    sorted_fl_ids = sorted(all_results.keys(),
                           key=lambda x: flowlines_info.get(x, {}).get('level', 0))

    for i, fl_idx in enumerate(sorted_fl_ids):
        row = 2 + i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])

        df = all_results[fl_idx]
        d = df['distance_m'].values
        z_bed = df['z_bed'].values
        z_ice = df['z_ice_surface'].values
        h = df['thickness_m'].values

        level = flowlines_info.get(fl_idx, {}).get('level', 0)
        color = level_colors.get(level, 'gray')

        ice_mask = h > 0
        ax.fill_between(d, z_bed, z_ice, where=ice_mask, alpha=0.3, color=color)
        ax.plot(d, z_bed, 'k-', linewidth=1, label='Bed')
        ax.plot(d, z_ice, '-', color=color, linewidth=1.5, label='Ice surface')

        if fl_idx in junction_info and junction_info[fl_idx].get('connected'):
            ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Junction')
            parent_z = junction_info[fl_idx]['parent_ice_surface']
            ax.axhline(parent_z, color='blue', linestyle=':', alpha=0.5, label=f'Parent ice={parent_z:.0f}m')

        h_max = h.max()
        h_mean = h[h > 0].mean() if (h > 0).any() else 0

        ax.set_title(f'FL{fl_idx} (L{level}) H_max={h_max:.0f}m H_mean={h_mean:.0f}m', fontsize=10)
        ax.set_xlabel('Distance from junction (m)', fontsize=8)
        ax.set_ylabel('Elevation (m)', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 可视化: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def run():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 4 V8: 冰厚反演 (使用已连接流线)")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])

    test_glacier = glacier_gdf[glacier_gdf['UniqueID'] == CONFIG['test_glacier_id']]
    test_flowlines = flowlines_gdf[flowlines_gdf['UniqueID'] == CONFIG['test_glacier_id']].copy()

    if len(test_glacier) == 0:
        print(f"  ✗ 找不到冰川: {CONFIG['test_glacier_id']}")
        return

    glacier_geom = test_glacier.geometry.values[0]
    print(f"  冰川: {CONFIG['test_glacier_id']}, 流线: {len(test_flowlines)}")

    # 2. 坐标转换
    print("\n[2] 坐标转换...")
    centroid = glacier_geom.centroid
    utm_crs, _ = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    glacier_utm = transform_geometry(glacier_geom, src_crs, utm_crs)

    flowlines_utm = {}
    flowlines_info = {}

    for _, row in test_flowlines.iterrows():
        fl_idx = row['fl_id']
        fl_geom = row.geometry
        if fl_geom.geom_type == 'MultiLineString':
            fl_geom = max(list(fl_geom.geoms), key=lambda x: x.length)

        fl_utm = transform_geometry(fl_geom, src_crs, utm_crs)
        if fl_utm.length < CONFIG['min_flowline_length']:
            print(f"    FL{fl_idx}: 长度{fl_utm.length:.0f}m < {CONFIG['min_flowline_length']}m，跳过")
            continue

        # 翻转为低端在前
        fl_utm = flip_line_to_low_start(fl_utm, CONFIG['dem_path'], utm_crs)
        flowlines_utm[fl_idx] = fl_utm

        parent_val = row.get('flows_to')
        if pd.notna(parent_val) and parent_val != 0:
            parent = int(parent_val)
        else:
            parent = None

        flowlines_info[fl_idx] = {
            'level': int(row.get('order', 0)),
            'parent': parent,
            'is_main': bool(row.get('is_main', False))
        }

    print(f"  有效流线: {len(flowlines_utm)}")

    # 3. 准备连接信息
    print("\n[3] 准备连接信息...")
    connection_info = {}

    for fl_idx, info in flowlines_info.items():
        if info.get('is_main') or info.get('parent') is None:
            connection_info[fl_idx] = {'connected': False, 'reason': 'main_or_no_parent'}
        else:
            fl_utm = flowlines_utm[fl_idx]
            parent_idx = info['parent']
            if parent_idx in flowlines_utm:
                parent_line = flowlines_utm[parent_idx]
                junction_pt = Point(fl_utm.coords[0])
                parent_project_dist = parent_line.project(junction_pt)
                connection_info[fl_idx] = {
                    'connected': True,
                    'parent': parent_idx,
                    'junction_x': junction_pt.x,
                    'junction_y': junction_pt.y,
                    'parent_project_dist': parent_project_dist,
                }
            else:
                connection_info[fl_idx] = {'connected': False, 'reason': 'parent_not_found'}

    print(f"  主流线: {sum(1 for c in connection_info.values() if not c.get('connected'))}")
    print(f"  已连接支流: {sum(1 for c in connection_info.values() if c.get('connected'))}")

    # 4. 汇水面积
    print("\n[4] 计算汇水面积...")
    catchment_areas = calculate_catchment_areas(flowlines_utm, glacier_utm,
                                                 CONFIG['catchment_grid_spacing'])

    # 5. 加载F因子
    f_interp = None
    if CONFIG['use_dynamic_f'] and os.path.exists(CONFIG['f_factor_csv']):
        f_data = pd.read_csv(CONFIG['f_factor_csv'])
        if 'distance_m' in f_data.columns and 'F_area' in f_data.columns:
            valid = ~f_data['F_area'].isna()
            if valid.sum() >= 2:
                f_interp = interp1d(f_data['distance_m'][valid], f_data['F_area'][valid],
                                   bounds_error=False, fill_value=(f_data['F_area'][valid].iloc[0],
                                                                   f_data['F_area'][valid].iloc[-1]))
                print(f"  ✓ F因子: {valid.sum()}点")

    # 6. 按层级计算冰厚
    print("\n[5] 计算冰厚...")
    all_results = {}
    junction_info = {}

    levels = sorted(set(info['level'] for info in flowlines_info.values()), reverse=True)

    for level in levels:
        fl_at_level = [idx for idx, info in flowlines_info.items() if info['level'] == level]
        print(f"\n  Level {level}: {len(fl_at_level)} 条")

        for fl_idx in fl_at_level:
            fl_utm = flowlines_utm[fl_idx]

            # DEM采样
            sample = sample_dem_along_line(fl_utm, CONFIG['dem_path'], utm_crs,
                                           CONFIG['flowline_spacing_m'])
            distances = sample['distances']
            xs, ys = sample['xs_utm'], sample['ys_utm']
            elevs = sample['elevs']

            if np.sum(~np.isnan(elevs)) < 3:
                print(f"    FL{fl_idx}: 有效高程点不足，跳过")
                continue

            # τ和F
            slopes = calculate_slope(elevs, distances)
            tau = estimate_tau(catchment_areas.get(fl_idx, 1.0), np.nanmean(slopes))
            F = f_interp(distances) if f_interp else DEFAULT_F

            conn = connection_info.get(fl_idx, {})

            if not conn.get('connected'):
                # 主流线
                thickness, ice_surface, start_idx, _ = ice_thickness_main_flowline(
                    elevs, distances, tau, F)

                junction_info[fl_idx] = {'connected': False}
                print(f"    FL{fl_idx}: 主流线, 起点高程={elevs[0]:.0f}m")
            else:
                # 支流
                parent_idx = conn['parent']
                parent_df = all_results[parent_idx]

                junction_x, junction_y = conn['junction_x'], conn['junction_y']
                parent_coords = parent_df[['x_utm', 'y_utm']].values
                dists = np.sqrt((parent_coords[:, 0] - junction_x)**2 +
                               (parent_coords[:, 1] - junction_y)**2)
                nearest_idx = np.argmin(dists)
                parent_ice_surface = parent_df['z_ice_surface'].values[nearest_idx]
                parent_bed = parent_df['z_bed'].values[nearest_idx]

                thickness, ice_surface = ice_thickness_tributary(
                    elevs, distances, tau, F, parent_ice_surface)

                junction_info[fl_idx] = {
                    'connected': True,
                    'parent': parent_idx,
                    'junction_x': junction_x,
                    'junction_y': junction_y,
                    'parent_ice_surface': parent_ice_surface,
                    'parent_bed': parent_bed,
                    'my_bed_at_junction': elevs[0],
                    'my_ice_at_junction': ice_surface[0],
                    'inherited_thickness': thickness[0],
                    'bed_diff': elevs[0] - parent_bed,
                    'ice_surface_diff': ice_surface[0] - parent_ice_surface
                }

                print(f"    FL{fl_idx} → FL{parent_idx}: "
                      f"父冰面={parent_ice_surface:.0f}m, 交汇床面={elevs[0]:.0f}m, "
                      f"继承冰厚={thickness[0]:.0f}m")

            # 转回WGS84
            transformer = Transformer.from_crs(utm_crs, src_crs, always_xy=True)
            lons, lats = transformer.transform(xs, ys)

            all_results[fl_idx] = pd.DataFrame({
                'flowline_id': fl_idx,
                'level': flowlines_info[fl_idx]['level'],
                'point_id': np.arange(len(distances)),
                'distance_m': distances,
                'x_utm': xs,
                'y_utm': ys,
                'lon': lons,
                'lat': lats,
                'z_bed': elevs,
                'slope_deg': slopes,
                'tau_Pa': tau,
                'thickness_m': thickness,
                'z_ice_surface': ice_surface,
            })

    if not all_results:
        print("  ✗ 无有效结果")
        return

    # 7. 保存
    print("\n[6] 保存...")
    final_df = pd.concat(all_results.values(), ignore_index=True)
    final_df.to_csv(CONFIG['output_dir'] + 'step4_v8_flowlines.csv', index=False)
    print(f"  ✓ 流线数据: {CONFIG['output_dir']}step4_v8_flowlines.csv")

    junction_records = []
    for fl_idx, jinfo in junction_info.items():
        if jinfo.get('connected'):
            junction_records.append({
                'flowline_id': fl_idx,
                'parent_id': jinfo['parent'],
                'junction_x': jinfo['junction_x'],
                'junction_y': jinfo['junction_y'],
                'parent_ice_surface': jinfo['parent_ice_surface'],
                'parent_bed': jinfo['parent_bed'],
                'my_bed_at_junction': jinfo['my_bed_at_junction'],
                'my_ice_at_junction': jinfo['my_ice_at_junction'],
                'inherited_thickness': jinfo['inherited_thickness'],
                'bed_diff': jinfo['my_bed_at_junction'] - jinfo['parent_bed'],
                'ice_surface_diff': jinfo['my_ice_at_junction'] - jinfo['parent_ice_surface']
            })

    if junction_records:
        junction_df = pd.DataFrame(junction_records)
        junction_df.to_csv(CONFIG['output_dir'] + 'junction_info.csv', index=False)
        print(f"  ✓ 交汇点信息: {CONFIG['output_dir']}junction_info.csv")

    # 8. 可视化
    print("\n[7] 可视化...")
    plot_results(glacier_utm, flowlines_utm, all_results, flowlines_info,
                junction_info, CONFIG['output_dir'] + 'step4_v8_diagnostic.png')

    # 9. 验证
    print("\n[8] 交汇点连续性验证:")
    print("-" * 60)
    for fl_idx, jinfo in junction_info.items():
        if jinfo.get('connected'):
            ice_diff = abs(jinfo['ice_surface_diff'])
            bed_diff = jinfo['bed_diff']
            status = "✓" if ice_diff < 5 else "△" if ice_diff < 20 else "✗"
            print(f"  {status} FL{fl_idx}→FL{jinfo['parent']}: "
                  f"冰面Δ={ice_diff:.1f}m, 床面Δ={bed_diff:.1f}m, "
                  f"继承冰厚={jinfo['inherited_thickness']:.0f}m")

    # 统计
    print("\n[9] 总体统计:")
    all_h = final_df['thickness_m'].values
    valid_h = all_h[all_h > 0]
    if len(valid_h) > 0:
        print(f"  H_mean:   {np.mean(valid_h):.1f}m")
        print(f"  H_max:    {np.max(valid_h):.1f}m")
        print(f"  H_median: {np.median(valid_h):.1f}m")
        print(f"  有效点数: {len(valid_h)}/{len(all_h)}")

    print(f"\n{'='*70}")
    print("✓ 完成")
    print(f"{'='*70}")

    return all_results, junction_info


if __name__ == "__main__":
    run()