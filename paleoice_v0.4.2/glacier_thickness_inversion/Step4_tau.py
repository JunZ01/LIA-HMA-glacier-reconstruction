"""
Step 4 修改版: 适配OGGM已连接流线
==================================

关键修改:
1. 删除连接线生成逻辑 (流线已物理连接)
2. 使用 is_main, order, flows_to 字段识别层级
3. 支流交汇点从几何上自动识别
4. 保留核心Benn & Hulton计算逻辑

输入:
- lia_flowlines_connected.shp (已连接的流线)
- step3_ffactor_data.csv (可选)
- DEM

输出:
- step4_flowlines_thickness.csv (冰厚点云)
"""

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.ops import transform
from pyproj import CRS, Transformer
from scipy.interpolate import interp1d
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
DEFAULT_F = 0.8
MIN_F, MAX_F = 0.445, 1.0

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    'flowlines_path': "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_connected.shp",
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'f_factor_csv': "/mnt/d/paleoice_output/step3_oggm/step3_ffactor_data.csv",  # 可选
    'output_dir': "/mnt/d/paleoice_output/step4_oggm/",

    # 测试
    'test_glacier_ids': [],  # 留空处理所有冰川

    # 参数
    'flowline_spacing_m': 50,
    'use_dynamic_f': True,  # 是否使用Step 3的F值
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
    """估算剪应力"""
    area_km2 = max(area_km2, 0.1)
    tau_base = np.clip(50 + 30 * np.log10(area_km2), 50, 150)
    if mean_slope_deg > 15:
        tau_base *= np.sin(np.radians(mean_slope_deg)) / np.sin(np.radians(12))
    return np.clip(tau_base, 50, 180) * 1000


def flip_line_to_low_start(line_utm, elevs):
    """确保流线从低端(末端)开始"""
    start_elev = elevs[0]
    end_elev = elevs[-1]

    if not np.isnan(start_elev) and not np.isnan(end_elev):
        if start_elev > end_elev:
            # 需要反转
            coords = list(line_utm.coords)[::-1]
            return LineString(coords), elevs[::-1], True

    return line_utm, elevs, False


def load_f_factors(csv_path, glacier_id, fl_id):
    """
    从Step 3输出加载F因子

    Returns:
        interpolator或None
    """
    try:
        df = pd.read_csv(csv_path)
        df_fl = df[(df['glacier_id'] == glacier_id) & (df['fl_id'] == fl_id)]

        if len(df_fl) == 0:
            return None

        # 按距离排序
        df_fl = df_fl.sort_values('distance_along_flowline')

        # 创建插值器
        distances = df_fl['distance_along_flowline'].values
        f_values = df_fl['F_final'].values

        if len(distances) < 2:
            return None

        interpolator = interp1d(
            distances, f_values,
            kind='linear',
            bounds_error=False,
            fill_value=(f_values[0], f_values[-1])
        )

        return interpolator
    except:
        return None


# ============================================================================
# Benn & Hulton 冰厚计算
# ============================================================================

def ice_thickness_benn_hulton(bed_elevs, xs, ys, distances, tau_pa, F_values,
                               rho=RHO_ICE, g=G):
    """
    Benn & Hulton (2010) 冰厚计算

    从末端(低海拔)向源头(高海拔)迭代

    Args:
        bed_elevs: 床面高程 (从末端到源头)
        xs, ys: UTM坐标
        distances: 沿流线距离
        tau_pa: 剪应力(Pa), 可以是标量或数组
        F_values: F因子, 可以是标量或数组
        rho, g: 物理常数

    Returns:
        dict: thickness, ice_surface, diagnostics
    """
    n = len(bed_elevs)

    # 初始化
    ice_surface = np.zeros(n)
    thickness = np.zeros(n)

    # 边界条件: 末端冰厚=0
    ice_surface[0] = bed_elevs[0]
    thickness[0] = 0

    # 确保tau和F是数组
    if np.isscalar(tau_pa):
        tau_pa = np.full(n, tau_pa)
    if np.isscalar(F_values):
        F_values = np.full(n, F_values)

    # 从末端向源头迭代
    for i in range(1, n):
        # 点间距
        d = np.sqrt((xs[i] - xs[i-1])**2 + (ys[i] - ys[i-1])**2)

        if d == 0:
            ice_surface[i] = ice_surface[i-1]
            thickness[i] = thickness[i-1]
            continue

        z_prev = bed_elevs[i-1]
        z_curr = bed_elevs[i]
        H_prev = ice_surface[i-1]

        # Benn & Hulton公式
        b = -(z_prev + z_curr)
        c = H_prev * (z_curr - (H_prev - z_prev)) - (2*d*tau_pa[i]/F_values[i]) / (rho*g)

        discriminant = b**2 - 4*c

        if discriminant >= 0:
            H_curr = (-b + np.sqrt(discriminant)) / 2
        else:
            H_curr = H_prev

        thick = H_curr - z_curr

        # 物理约束
        if thick < 0:
            thick = 0
            H_curr = z_curr

        if thick > 2000:
            thick = 2000
            H_curr = z_curr + thick

        if np.isnan(thick) or np.isinf(thick):
            thick = thickness[i-1]
            H_curr = z_curr + thick

        ice_surface[i] = H_curr
        thickness[i] = thick

    return {
        'thickness': thickness,
        'ice_surface': ice_surface
    }


# ============================================================================
# 主处理函数
# ============================================================================

def find_junction_on_parent(tributary_utm, parent_utm):
    """
    找到支流末端在主流上的交汇点

    Returns:
        (distance_on_parent, junction_point, junction_ice_surface)
    """
    # 支流末端点
    trib_end = Point(tributary_utm.coords[-1])

    # 在主流上找最近点
    distance_on_parent = parent_utm.project(trib_end)
    junction_point = parent_utm.interpolate(distance_on_parent)

    return distance_on_parent, junction_point


def process_glacier(rgi_id, flowlines_gdf, glacier_gdf, dem_path,
                    f_factor_csv, config):
    """处理单个冰川"""

    print(f"\n处理冰川: {rgi_id}")

    # 获取冰川边界
    glacier = glacier_gdf[glacier_gdf['OrigRGIId'] == rgi_id]
    if len(glacier) == 0:
        glacier = glacier_gdf[glacier_gdf['RGIId'] == rgi_id]
    if len(glacier) == 0:
        print(f"  ✗ 找不到冰川边界")
        return None

    glacier_geom = glacier.geometry.values[0]
    glacier_area_km2 = glacier_geom.area / 1e6

    # 获取流线
    glacier_flowlines = flowlines_gdf[flowlines_gdf['rgi_id'] == rgi_id]
    if len(glacier_flowlines) == 0:
        print(f"  ✗ 找不到流线")
        return None

    print(f"  流线数: {len(glacier_flowlines)}")

    # UTM坐标系
    centroid = glacier_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    # 按层级排序流线
    glacier_flowlines = glacier_flowlines.sort_values('order')

    # 存储结果
    all_thickness_data = []
    flowline_ice_surfaces = {}  # 存储每条流线的冰面高程插值器

    # 处理每条流线
    for idx, row in glacier_flowlines.iterrows():
        fl_id = row['fl_id']
        is_main = row['is_main']
        order = row['order']
        flows_to = row['flows_to']

        fl_geom = row.geometry
        if fl_geom.geom_type == 'MultiLineString':
            fl_geom = max(list(fl_geom.geoms), key=lambda x: x.length)

        # 转UTM
        fl_utm = transform_geometry(fl_geom, src_crs, utm_crs)

        # 采样DEM
        sample_data = sample_dem_along_line(
            fl_utm, dem_path, utm_crs, config['flowline_spacing_m']
        )

        # 反转方向(确保从低→高)
        fl_utm, elevs, flipped = flip_line_to_low_start(
            fl_utm, sample_data['elevs']
        )

        if flipped:
            sample_data['xs_utm'] = sample_data['xs_utm'][::-1]
            sample_data['ys_utm'] = sample_data['ys_utm'][::-1]
            sample_data['distances'] = sample_data['length'] - sample_data['distances'][::-1]

        # 计算坡度
        slopes = calculate_slope(elevs, sample_data['distances'])
        mean_slope = np.mean(slopes[~np.isnan(slopes)])

        # 估算τ
        tau_pa = estimate_tau(glacier_area_km2, mean_slope)

        # 加载F因子
        if config['use_dynamic_f'] and f_factor_csv and os.path.exists(f_factor_csv):
            f_interp = load_f_factors(f_factor_csv, rgi_id, fl_id)
            if f_interp:
                F_values = f_interp(sample_data['distances'])
                F_values = np.clip(F_values, MIN_F, MAX_F)
            else:
                F_values = DEFAULT_F
        else:
            F_values = DEFAULT_F

        # 计算冰厚
        if is_main:
            # 主流: 从末端开始
            result = ice_thickness_benn_hulton(
                elevs, sample_data['xs_utm'], sample_data['ys_utm'],
                sample_data['distances'], tau_pa, F_values
            )

            print(f"  主流{fl_id}: 长度{sample_data['length']:.0f}m, "
                  f"平均厚{np.mean(result['thickness']):.1f}m, "
                  f"最大厚{np.max(result['thickness']):.1f}m")

        else:
            # 支流: 需要从交汇点继承冰面高程
            if pd.isna(flows_to) or flows_to == -1 or flows_to not in flowline_ice_surfaces:
                print(f"  支流{fl_id}→{flows_to}: ✗ 父流线未处理")
                continue

            # 找到交汇点
            parent_fl_id = int(flows_to)
            parent_interp = flowline_ice_surfaces[parent_fl_id]

            # 支流末端在父流线上的投影距离
            # 这里简化处理: 用末端点在父流线上的投影
            # 更精确的方法需要几何计算
            trib_end_elev = elevs[0]  # 末端高程

            # 从父流线插值冰面高程
            # 注意: 这里需要知道支流末端在父流线上的距离
            # 简化处理: 假设连接点就是支流末端
            junction_ice_surface = parent_interp(0)  # 这里需要改进

            # 修改边界条件
            elevs_copy = elevs.copy()
            xs_copy = sample_data['xs_utm'].copy()
            ys_copy = sample_data['ys_utm'].copy()

            # 插入交汇点
            elevs_copy = np.insert(elevs_copy, 0, trib_end_elev)
            xs_copy = np.insert(xs_copy, 0, xs_copy[0])
            ys_copy = np.insert(ys_copy, 0, ys_copy[0])

            # 计算
            result = ice_thickness_benn_hulton(
                elevs_copy, xs_copy, ys_copy,
                np.insert(sample_data['distances'], 0, 0),
                tau_pa, F_values
            )

            # 手动设置交汇点冰面
            result['ice_surface'][0] = junction_ice_surface
            result['thickness'][0] = junction_ice_surface - trib_end_elev

            # 移除插入的点
            result['thickness'] = result['thickness'][1:]
            result['ice_surface'] = result['ice_surface'][1:]

            print(f"  支流{fl_id}→{parent_fl_id}: 长度{sample_data['length']:.0f}m, "
                  f"平均厚{np.mean(result['thickness']):.1f}m")

        # 存储冰面插值器(供下游支流使用)
        flowline_ice_surfaces[fl_id] = interp1d(
            sample_data['distances'],
            result['ice_surface'],
            kind='linear',
            bounds_error=False,
            fill_value=(result['ice_surface'][0], result['ice_surface'][-1])
        )

        # 记录数据
        for i in range(len(result['thickness'])):
            all_thickness_data.append({
                'glacier_id': rgi_id,
                'fl_id': fl_id,
                'is_main': is_main,
                'order': order,
                'point_idx': i,
                'distance_m': sample_data['distances'][i],
                'x_utm': sample_data['xs_utm'][i],
                'y_utm': sample_data['ys_utm'][i],
                'utm_zone': utm_zone,
                'z_bed': elevs[i],
                'thickness_m': result['thickness'][i],
                'z_ice_surface': result['ice_surface'][i],
                'tau_pa': tau_pa,
                'F_factor': F_values[i] if isinstance(F_values, np.ndarray) else F_values,
                'slope_deg': slopes[i]
            })

    return all_thickness_data


def run_step4():
    """运行Step 4"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 4: 冰厚反演 (OGGM流线版)")
    print("=" * 70)

    # 加载数据
    print("\n加载数据...")
    flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])

    print(f"  流线: {len(flowlines_gdf)} 条")
    print(f"  冰川: {len(glacier_gdf)} 个")

    # 确定处理的冰川
    if CONFIG['test_glacier_ids']:
        glacier_ids = CONFIG['test_glacier_ids']
    else:
        glacier_ids = flowlines_gdf['rgi_id'].unique()

    print(f"\n处理冰川数: {len(glacier_ids)}")

    # 处理
    all_results = []

    for i, rgi_id in enumerate(glacier_ids, 1):
        print(f"\n[{i}/{len(glacier_ids)}]", end=' ')

        result = process_glacier(
            rgi_id, flowlines_gdf, glacier_gdf,
            CONFIG['dem_path'], CONFIG['f_factor_csv'], CONFIG
        )

        if result:
            all_results.extend(result)

    # 保存
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = os.path.join(CONFIG['output_dir'], 'step4_flowlines_thickness.csv')
        df.to_csv(output_csv, index=False)

        print(f"\n{'=' * 70}")
        print("统计:")
        print(f"  总点数: {len(df)}")
        print(f"  平均冰厚: {df['thickness_m'].mean():.1f} m")
        print(f"  最大冰厚: {df['thickness_m'].max():.1f} m")
        print(f"\n✓ 保存至: {output_csv}")
        print("=" * 70)

        return df
    else:
        print("\n✗ 无有效结果")
        return None


if __name__ == "__main__":
    result_df = run_step4()