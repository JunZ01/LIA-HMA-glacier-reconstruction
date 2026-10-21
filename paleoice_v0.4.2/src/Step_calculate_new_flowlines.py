"""
新流线冰厚计算脚本
==================

对 Step_Densify_Flowlines.py 生成的新流线计算冰厚
然后与原有OGGM流线冰厚合并，准备进入 Step 5 插值

流程:
[1] 加载加密后的流线（OGGM + 新流线）
[2] 对新流线运行 Benn-Hulton 冰厚计算
[3] 关键：新流线从交汇点继承父流线的冰面高程
[4] 合并所有冰厚点
[5] 输出用于 Step 5 的 CSV

注意:
- 新流线的层级关系已在 Step_Densify_Flowlines.py 中建立
- ParentID 字段指向父流线
- 计算顺序：先计算无父流线的（主流线），再按层级计算支流

使用方法:
    python Step_Calculate_New_Flowlines.py
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
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 物理常数
# ============================================================================
RHO_ICE = 900
G = 9.81
MIN_SLOPE = 1.0
DEFAULT_TAU = 100000  # 100 kPa
DEFAULT_F = 0.8

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    # 输入文件
    'densified_flowlines_path': "/mnt/d/paleoice_output/densified_flowlines/RGI60-15_06881_densified_flowlines.shp",
    'original_thickness_csv': "/mnt/d/paleoice_output/step4_v8/step4_v8_flowlines.csv",  # 原有OGGM流线冰厚
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'f_factor_csv': "/mnt/d/paleoice_output/step3_diagnostic/step3_ffactor_data.csv",

    # 输出目录
    'output_dir': "/mnt/d/paleoice_output/densified_thickness/",

    # 测试冰川
    'test_glacier_id': "RGI60-15.06881",

    # 计算参数
    'flowline_spacing_m': 50,
    'use_dynamic_tau': True,
    'use_dynamic_f': True,
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
                    val = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
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
        if i == 0 and i + 1 < n and not np.isnan(elevs[i + 1]):
            dz, dx = abs(elevs[i + 1] - elevs[i]), distances[i + 1] - distances[i]
        elif i == n - 1 and not np.isnan(elevs[i - 1]):
            dz, dx = abs(elevs[i] - elevs[i - 1]), distances[i] - distances[i - 1]
        elif 0 < i < n - 1 and not np.isnan(elevs[i - 1]) and not np.isnan(elevs[i + 1]):
            dz, dx = abs(elevs[i + 1] - elevs[i - 1]), distances[i + 1] - distances[i - 1]
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


# ============================================================================
# Benn-Hulton 冰厚计算
# ============================================================================

def ice_thickness_from_terminus(bed_elevs, xs, ys, tau_pa, F,
                                rho=RHO_ICE, g=G):
    """
    从末端（索引0）向源头迭代计算冰厚
    边界条件：末端冰厚=0
    """
    n = len(bed_elevs)
    ice_surface = np.zeros(n)
    thickness = np.zeros(n)

    # 边界条件
    ice_surface[0] = bed_elevs[0]
    thickness[0] = 0

    # 处理F为数组或标量的情况
    if np.isscalar(F):
        F = np.full(n, F)

    for i in range(1, n):
        if np.isnan(bed_elevs[i]) or np.isnan(bed_elevs[i - 1]):
            ice_surface[i] = ice_surface[i - 1]
            thickness[i] = thickness[i - 1]
            continue

        d = np.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2)
        if d == 0:
            ice_surface[i] = ice_surface[i - 1]
            thickness[i] = thickness[i - 1]
            continue

        z_prev = bed_elevs[i - 1]
        z_curr = bed_elevs[i]
        H_prev = ice_surface[i - 1]
        F_curr = F[i] if isinstance(F, np.ndarray) else F

        # Benn-Hulton公式
        b = -(z_prev + z_curr)
        c = H_prev * (z_curr - (H_prev - z_prev)) - (2 * d * tau_pa / F_curr) / (rho * g)

        discriminant = b ** 2 - 4 * c

        if discriminant >= 0:
            H_curr = (-b + np.sqrt(discriminant)) / 2
        else:
            H_curr = H_prev

        thick = H_curr - z_curr

        # 限制
        thick = np.clip(thick, 0, 2000)
        if np.isnan(thick) or np.isinf(thick):
            thick = thickness[i - 1]
            H_curr = z_curr + thick

        ice_surface[i] = H_curr
        thickness[i] = thick

    return thickness, ice_surface


def ice_thickness_from_junction(bed_elevs, xs, ys, tau_pa, F,
                                parent_ice_surface, rho=RHO_ICE, g=G):
    """
    从交汇点继承父流线冰面高程，向上游迭代
    边界条件：交汇点（索引0）的冰面高程 = 父流线在该点的冰面高程
    """
    n = len(bed_elevs)
    ice_surface = np.zeros(n)
    thickness = np.zeros(n)

    # 边界条件：继承父流线冰面
    ice_surface[0] = parent_ice_surface
    thickness[0] = max(0, parent_ice_surface - bed_elevs[0])

    if np.isscalar(F):
        F = np.full(n, F)

    for i in range(1, n):
        if np.isnan(bed_elevs[i]) or np.isnan(bed_elevs[i - 1]):
            ice_surface[i] = ice_surface[i - 1]
            thickness[i] = thickness[i - 1]
            continue

        d = np.sqrt((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2)
        if d == 0:
            ice_surface[i] = ice_surface[i - 1]
            thickness[i] = thickness[i - 1]
            continue

        z_prev = bed_elevs[i - 1]
        z_curr = bed_elevs[i]
        H_prev = ice_surface[i - 1]
        F_curr = F[i] if isinstance(F, np.ndarray) else F

        b = -(z_prev + z_curr)
        c = H_prev * (z_curr - (H_prev - z_prev)) - (2 * d * tau_pa / F_curr) / (rho * g)

        discriminant = b ** 2 - 4 * c

        if discriminant >= 0:
            H_curr = (-b + np.sqrt(discriminant)) / 2
        else:
            H_curr = H_prev

        thick = H_curr - z_curr
        thick = np.clip(thick, 0, 2000)

        if np.isnan(thick) or np.isinf(thick):
            thick = thickness[i - 1]
            H_curr = z_curr + thick

        ice_surface[i] = H_curr
        thickness[i] = thick

    return thickness, ice_surface


# ============================================================================
# 层级处理
# ============================================================================

def build_hierarchy(flowlines_gdf):
    """
    构建流线层级关系

    Returns:
        levels: dict {flowline_id: level}
        children: dict {parent_id: [child_ids]}
    """
    children = defaultdict(list)
    levels = {}

    # 构建父子关系
    for idx, row in flowlines_gdf.iterrows():
        fl_id = row['FlowlineID']
        parent_id = row.get('ParentID', None)

        if pd.isna(parent_id) or parent_id is None:
            levels[fl_id] = 0  # 主流线
        else:
            children[parent_id].append(fl_id)

    # 计算层级
    def get_level(fl_id, visited=None):
        if visited is None:
            visited = set()
        if fl_id in visited:
            return 0  # 防止循环
        visited.add(fl_id)

        if fl_id in levels:
            return levels[fl_id]

        row = flowlines_gdf[flowlines_gdf['FlowlineID'] == fl_id]
        if len(row) == 0:
            return 0

        parent_id = row['ParentID'].values[0]
        if pd.isna(parent_id) or parent_id is None:
            levels[fl_id] = 0
            return 0

        parent_level = get_level(parent_id, visited)
        levels[fl_id] = parent_level + 1
        return levels[fl_id]

    for idx, row in flowlines_gdf.iterrows():
        fl_id = row['FlowlineID']
        if fl_id not in levels:
            get_level(fl_id)

    return levels, dict(children)


def find_parent_ice_surface(parent_id, junction_x, junction_y,
                            all_results, flowlines_utm):
    """
    找到父流线在交汇点的冰面高程

    Args:
        parent_id: 父流线ID
        junction_x, junction_y: 交汇点坐标（UTM）
        all_results: 已计算的所有流线结果 {fl_id: DataFrame}
        flowlines_utm: 流线几何 {fl_id: LineString}
    """
    if parent_id not in all_results:
        return None

    parent_df = all_results[parent_id]
    parent_coords = parent_df[['x_utm', 'y_utm']].values

    # 找最近点
    dists = np.sqrt((parent_coords[:, 0] - junction_x) ** 2 +
                    (parent_coords[:, 1] - junction_y) ** 2)
    nearest_idx = np.argmin(dists)

    if dists[nearest_idx] > 200:  # 如果最近点太远，可能有问题
        print(f"    警告: 交汇点距离父流线 {dists[nearest_idx]:.0f}m")

    return parent_df['z_ice_surface'].values[nearest_idx]


# ============================================================================
# 主流程
# ============================================================================

def run():
    """主函数"""
    print("=" * 70)
    print("新流线冰厚计算")
    print("=" * 70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # 1. 加载数据
    print("\n[1] 加载数据...")

    # 加载加密后的流线
    flowlines_gdf = gpd.read_file(CONFIG['densified_flowlines_path'])
    print(f"  加密流线总数: {len(flowlines_gdf)}")

    n_oggm = len(flowlines_gdf[flowlines_gdf['IsNew'] == False])
    n_new = len(flowlines_gdf[flowlines_gdf['IsNew'] == True])
    print(f"  OGGM流线: {n_oggm}, 新流线: {n_new}")

    # 加载原有冰厚数据
    original_thickness = None
    if os.path.exists(CONFIG['original_thickness_csv']):
        original_thickness = pd.read_csv(CONFIG['original_thickness_csv'])
        print(f"  原有冰厚点: {len(original_thickness)}")

    # 加载冰川边界
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    glacier_row = glacier_gdf[glacier_gdf['OrigRGIId'] == CONFIG['test_glacier_id']]
    glacier_geom = glacier_row.geometry.values[0]

    # 2. 坐标转换
    print("\n[2] 坐标转换...")
    centroid = glacier_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    print(f"  UTM Zone: {utm_zone}N")

    # 转换流线到UTM
    flowlines_utm = {}
    flowlines_info = {}

    for idx, row in flowlines_gdf.iterrows():
        fl_id = row['FlowlineID']
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == 'MultiLineString':
            geom = max(list(geom.geoms), key=lambda x: x.length)

        geom_utm = transform_geometry(geom, src_crs, utm_crs)
        flowlines_utm[fl_id] = geom_utm
        flowlines_info[fl_id] = {
            'is_new': row['IsNew'],
            'parent_id': row.get('ParentID', None),
            'length_m': geom_utm.length
        }

    # 3. 构建层级关系
    print("\n[3] 构建层级关系...")
    levels, children = build_hierarchy(flowlines_gdf)

    max_level = max(levels.values()) if levels else 0
    print(f"  最大层级: {max_level}")
    for level in range(max_level + 1):
        count = sum(1 for l in levels.values() if l == level)
        print(f"    Level {level}: {count} 条流线")

    # 4. 加载F因子插值器
    f_interp = None
    if CONFIG['use_dynamic_f'] and os.path.exists(CONFIG['f_factor_csv']):
        f_data = pd.read_csv(CONFIG['f_factor_csv'])
        if 'distance_m' in f_data.columns and 'F_area' in f_data.columns:
            valid = ~f_data['F_area'].isna()
            if valid.sum() >= 2:
                f_interp = interp1d(f_data['distance_m'][valid],
                                    f_data['F_area'][valid],
                                    bounds_error=False,
                                    fill_value=(f_data['F_area'][valid].iloc[0],
                                                f_data['F_area'][valid].iloc[-1]))
                print(f"  ✓ F因子插值器: {valid.sum()} 点")

    # 5. 按层级计算冰厚
    print("\n[4] 计算冰厚...")

    all_results = {}

    # 首先处理原有OGGM流线（使用已有数据或重新计算）
    oggm_flowlines = {fl_id: info for fl_id, info in flowlines_info.items()
                      if not info['is_new']}
    new_flowlines = {fl_id: info for fl_id, info in flowlines_info.items()
                     if info['is_new']}

    print(f"\n  处理OGGM流线 ({len(oggm_flowlines)} 条)...")

    # 如果有原有冰厚数据，直接使用
    if original_thickness is not None:
        print("    使用原有冰厚数据")
        for fl_id in oggm_flowlines:
            # 从原始ID提取
            if fl_id.startswith('OGGM_'):
                orig_idx = int(fl_id.split('_')[1])
                fl_data = original_thickness[original_thickness['flowline_id'] == orig_idx]
                if len(fl_data) > 0:
                    all_results[fl_id] = fl_data.copy()
                    all_results[fl_id]['flowline_id'] = fl_id
    else:
        # 重新计算OGGM流线
        print("    重新计算OGGM流线冰厚")
        for fl_id, info in oggm_flowlines.items():
            if info['parent_id'] is None or pd.isna(info['parent_id']):
                # 主流线
                level = 0
            else:
                level = levels.get(fl_id, 0)

    # 处理新流线
    print(f"\n  处理新流线 ({len(new_flowlines)} 条)...")

    # 按层级排序
    sorted_new_flowlines = sorted(new_flowlines.items(),
                                  key=lambda x: levels.get(x[0], 999))

    for fl_id, info in sorted_new_flowlines:
        fl_utm = flowlines_utm[fl_id]
        parent_id = info['parent_id']
        level = levels.get(fl_id, 0)

        print(f"\n    {fl_id} (Level {level}, 长度 {fl_utm.length:.0f}m)")

        # DEM采样
        sample = sample_dem_along_line(fl_utm, CONFIG['dem_path'], utm_crs,
                                       CONFIG['flowline_spacing_m'])

        distances = sample['distances']
        xs, ys = sample['xs_utm'], sample['ys_utm']
        elevs = sample['elevs']

        valid_count = np.sum(~np.isnan(elevs))
        if valid_count < 3:
            print(f"      跳过: 有效高程点不足 ({valid_count})")
            continue

        # 计算坡度和τ
        slopes = calculate_slope(elevs, distances)
        mean_slope = np.nanmean(slopes)

        if CONFIG['use_dynamic_tau']:
            # 简化：使用流线长度估算汇水面积
            area_km2 = (fl_utm.length / 1000) ** 1.5 * 0.5  # 简化估算
            tau = estimate_tau(area_km2, mean_slope)
        else:
            tau = DEFAULT_TAU

        # F因子
        if f_interp is not None:
            F = f_interp(distances)
            F = np.clip(F, 0.445, 1.0)
        else:
            F = DEFAULT_F

        # 计算冰厚
        if parent_id is None or pd.isna(parent_id) or parent_id not in all_results:
            # 无父流线或父流线未计算：从末端（索引0）开始
            print(f"      从末端开始 (H=0)")
            thickness, ice_surface = ice_thickness_from_terminus(
                elevs, xs, ys, tau, F)
        else:
            # 有父流线：从交汇点继承冰面高程
            junction_x, junction_y = xs[0], ys[0]
            parent_ice = find_parent_ice_surface(parent_id, junction_x, junction_y,
                                                 all_results, flowlines_utm)

            if parent_ice is not None:
                print(f"      从交汇点继承 (父冰面={parent_ice:.0f}m, 床面={elevs[0]:.0f}m)")
                thickness, ice_surface = ice_thickness_from_junction(
                    elevs, xs, ys, tau, F, parent_ice)
            else:
                print(f"      找不到父流线冰面，从末端开始")
                thickness, ice_surface = ice_thickness_from_terminus(
                    elevs, xs, ys, tau, F)

        # 转回WGS84
        transformer = Transformer.from_crs(utm_crs, src_crs, always_xy=True)
        lons, lats = transformer.transform(xs, ys)

        # 保存结果
        all_results[fl_id] = pd.DataFrame({
            'flowline_id': fl_id,
            'is_new': True,
            'level': level,
            'parent_id': parent_id,
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

        print(f"      平均冰厚: {np.mean(thickness[thickness > 0]):.1f}m, "
              f"最大: {np.max(thickness):.1f}m")

    # 6. 合并所有结果
    print("\n[5] 合并结果...")

    all_dfs = []
    for fl_id, df in all_results.items():
        if isinstance(df, pd.DataFrame):
            all_dfs.append(df)

    if not all_dfs:
        print("  ✗ 无有效结果")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # 保存
    output_csv = CONFIG['output_dir'] + 'densified_thickness_all.csv'
    combined_df.to_csv(output_csv, index=False)
    print(f"  ✓ 保存: {output_csv}")

    # 统计
    print("\n[6] 统计...")

    oggm_points = combined_df[combined_df['is_new'] == False] if 'is_new' in combined_df.columns else combined_df
    new_points = combined_df[combined_df['is_new'] == True] if 'is_new' in combined_df.columns else pd.DataFrame()

    print(f"  总冰厚点: {len(combined_df)}")
    if len(new_points) > 0:
        print(f"  OGGM流线点: {len(combined_df) - len(new_points)}")
        print(f"  新流线点: {len(new_points)}")

    valid_h = combined_df['thickness_m'].values
    valid_h = valid_h[valid_h > 0]
    print(f"  有效冰厚点: {len(valid_h)}")
    print(f"  平均冰厚: {np.mean(valid_h):.1f} m")
    print(f"  最大冰厚: {np.max(valid_h):.1f} m")
    print(f"  中位数冰厚: {np.median(valid_h):.1f} m")

    # 7. 可视化
    print("\n[7] 可视化...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 冰厚点分布
    ax1 = axes[0]
    scatter = ax1.scatter(combined_df['x_utm'], combined_df['y_utm'],
                          c=combined_df['thickness_m'], cmap='Blues',
                          s=5, alpha=0.7)
    plt.colorbar(scatter, ax=ax1, label='Ice Thickness (m)')
    ax1.set_xlabel('X (UTM)')
    ax1.set_ylabel('Y (UTM)')
    ax1.set_title('Ice Thickness Points')
    ax1.set_aspect('equal')

    # 冰厚直方图
    ax2 = axes[1]
    ax2.hist(valid_h, bins=50, color='steelblue', edgecolor='white')
    ax2.axvline(np.mean(valid_h), color='red', linestyle='--', label=f'Mean: {np.mean(valid_h):.1f}m')
    ax2.axvline(np.median(valid_h), color='orange', linestyle='--', label=f'Median: {np.median(valid_h):.1f}m')
    ax2.set_xlabel('Ice Thickness (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Ice Thickness Distribution')
    ax2.legend()

    plt.tight_layout()
    fig_path = CONFIG['output_dir'] + 'densified_thickness_overview.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fig_path}")

    print(f"\n{'=' * 70}")
    print("✓ 新流线冰厚计算完成!")
    print(f"{'=' * 70}")
    print(f"\n下一步:")
    print(f"  1. 检查输出: {output_csv}")
    print(f"  2. 修改 Step 5 配置，使用新的冰厚CSV")
    print(f"  3. 运行 Step 5 进行空间插值")

    return combined_df


if __name__ == "__main__":
    run()