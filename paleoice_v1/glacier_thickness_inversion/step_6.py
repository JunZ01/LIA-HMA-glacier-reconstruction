"""
Step 6: 基于高程的冰厚空间插值填充
==================================

功能:
- 读取Step5输出的冰厚栅格（流线区域有值，其他区域=0）
- 基于高程-冰厚关系填充零值区域
- 输出完整的冰厚栅格

输入:
- step5_thickness_raster.tif (Step5输出)
- glacier_success.shp (LIA边界)
- Himalaya_bare_DEM.tif (床面DEM)

输出:
- step6_thickness_filled.tif (填充后的冰厚栅格)
- step6_ice_surface.tif (冰面高程栅格)
- step6_statistics.csv (统计数据)

方法:
1. 按高程分带（50m），统计每个高程带的流线冰厚平均值
2. 对零值像元，根据其高程查询对应冰厚
3. 不应用边界距离衰减（高程关系已隐含边界效应）
"""

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS as RioCRS
import geopandas as gpd
from scipy.interpolate import interp1d
import pandas as pd
import os
from datetime import datetime

# ============== 配置 ==============
CONFIG = {
    # 输入文件
    'thickness_raster': "/mnt/d/paleoice_output/step5_diagnostic/step5_thickness_raster.tif",
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",

    # 输出目录
    'output_dir': "/mnt/d/paleoice_output/step6/",

    # 测试冰川 (设为None则处理所有冰川)
    'test_glacier_id': 2692,  # UniqueID，整数类型

    # 高程带参数
    'elevation_bin_size': 50,  # 高程带宽度(m)

    # 插值方法: 'mean', 'median'
    'interpolation_method': 'mean',
}


def load_thickness_raster(raster_path):
    """加载Step5输出的冰厚栅格"""
    with rasterio.open(raster_path) as src:
        thickness = src.read(1)
        transform = src.transform
        crs = src.crs
        profile = src.profile.copy()
        bounds = src.bounds

    return {
        'thickness': thickness,
        'transform': transform,
        'crs': crs,
        'profile': profile,
        'bounds': bounds,
        'shape': thickness.shape
    }


def load_glacier_mask(glacier_path, glacier_id, raster_data):
    """加载并创建冰川掩膜"""
    glaciers = gpd.read_file(glacier_path)

    # 重投影到栅格坐标系
    if glaciers.crs != raster_data['crs']:
        glaciers = glaciers.to_crs(raster_data['crs'])

    # 获取指定冰川
    glacier = glaciers[glaciers['UniqueID'] == glacier_id].iloc[0]
    glacier_geom = glacier.geometry

    # 创建掩膜
    glacier_mask = ~geometry_mask(
        [glacier_geom],
        out_shape=raster_data['shape'],
        transform=raster_data['transform'],
        invert=False
    )

    return glacier_mask, glacier_geom, glaciers


def load_dem_resampled(dem_path, raster_data):
    """加载DEM并重采样到冰厚栅格网格"""
    dem_resampled = np.zeros(raster_data['shape'], dtype=np.float32)

    with rasterio.open(dem_path) as dem_src:
        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_resampled,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=raster_data['transform'],
            dst_crs=raster_data['crs'],
            resampling=Resampling.bilinear
        )

    return dem_resampled


def build_elevation_thickness_relationship(thickness, dem, glacier_mask, bin_size=50):
    """
    建立高程-冰厚关系

    Returns:
        stats: list of dicts, 每个高程带的统计信息
        elevation_bins: 高程带边界数组
    """
    flowline_mask = (thickness > 0) & glacier_mask

    # 获取流线区域的高程和冰厚
    flowline_elev = dem[flowline_mask]
    flowline_thick = thickness[flowline_mask]

    if len(flowline_elev) == 0:
        return [], np.array([])

    # 确定高程带范围
    elev_min = np.floor(dem[glacier_mask].min() / bin_size) * bin_size
    elev_max = np.ceil(dem[glacier_mask].max() / bin_size) * bin_size
    elevation_bins = np.arange(elev_min, elev_max + bin_size, bin_size)

    # 统计每个高程带
    stats = []
    for i in range(len(elevation_bins) - 1):
        e_low = elevation_bins[i]
        e_high = elevation_bins[i + 1]
        e_mid = (e_low + e_high) / 2

        band_mask = (flowline_elev >= e_low) & (flowline_elev < e_high)
        band_thickness = flowline_thick[band_mask]

        if len(band_thickness) > 0:
            stats.append({
                'elev_mid': e_mid,
                'elev_low': e_low,
                'elev_high': e_high,
                'count': len(band_thickness),
                'mean': band_thickness.mean(),
                'median': np.median(band_thickness),
                'std': band_thickness.std(),
                'max': band_thickness.max(),
            })

    return stats, elevation_bins


def create_elevation_thickness_function(stats, method='mean'):
    """
    创建高程→冰厚的插值函数
    """
    if len(stats) < 2:
        return None

    elevations = np.array([s['elev_mid'] for s in stats])
    thicknesses = np.array([s[method] for s in stats])

    func = interp1d(
        elevations, thicknesses,
        kind='linear',
        bounds_error=False,
        fill_value=(thicknesses[0], thicknesses[-1])
    )

    return func


def fill_zero_regions(thickness, dem, glacier_mask, elev_func):
    """
    填充零值区域

    使用高程-冰厚关系，不应用边界距离衰减
    """
    result = thickness.copy()
    zero_mask = (thickness == 0) & glacier_mask

    if not np.any(zero_mask):
        return result

    # 获取零值区域的高程
    zero_elevations = dem[zero_mask]

    # 用高程函数估算冰厚
    estimated_thickness = elev_func(zero_elevations)

    # 确保非负
    estimated_thickness = np.maximum(estimated_thickness, 0)

    # 填充
    result[zero_mask] = estimated_thickness

    return result


def calculate_ice_surface(thickness, dem, glacier_mask):
    """计算冰面高程"""
    ice_surface = np.where(glacier_mask, dem + thickness, dem)
    return ice_surface


def calculate_statistics(thickness, glacier_mask, resolution=30):
    """计算统计数据"""
    valid_thickness = thickness[glacier_mask]
    positive_thickness = valid_thickness[valid_thickness > 0]

    cell_area_m2 = resolution ** 2

    stats = {
        'area_km2': np.sum(glacier_mask) * cell_area_m2 / 1e6,
        'volume_km3': np.sum(thickness[glacier_mask]) * cell_area_m2 / 1e9,
        'mean_thickness_m': np.mean(valid_thickness) if len(valid_thickness) > 0 else 0,
        'max_thickness_m': np.max(valid_thickness) if len(valid_thickness) > 0 else 0,
        'median_thickness_m': np.median(valid_thickness) if len(valid_thickness) > 0 else 0,
        'positive_pixel_ratio': len(positive_thickness) / len(valid_thickness) if len(valid_thickness) > 0 else 0,
    }

    return stats


def save_raster(data, output_path, profile):
    """保存栅格"""
    profile = profile.copy()
    profile.update(dtype='float32', compress='lzw')

    # 处理NaN
    data = np.where(np.isnan(data), -9999, data)
    profile.update(nodata=-9999)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def process_single_glacier(glacier_id, raster_data, dem, glaciers, config):
    """处理单个冰川"""
    # 获取冰川
    glacier_row = glaciers[glaciers['UniqueID'] == glacier_id]
    if len(glacier_row) == 0:
        return None

    glacier_geom = glacier_row.iloc[0].geometry

    # 创建掩膜
    glacier_mask = ~geometry_mask(
        [glacier_geom],
        out_shape=raster_data['shape'],
        transform=raster_data['transform'],
        invert=False
    )

    thickness = raster_data['thickness']

    # 统计
    glacier_pixels = np.sum(glacier_mask)
    flowline_pixels = np.sum((thickness > 0) & glacier_mask)
    zero_pixels = np.sum((thickness == 0) & glacier_mask)

    if glacier_pixels == 0:
        return None

    flowline_ratio = flowline_pixels / glacier_pixels

    # 如果流线覆盖率太低，跳过
    if flowline_ratio < 0.1:
        print(f"    警告: 冰川 {glacier_id} 流线覆盖率仅 {flowline_ratio*100:.1f}%，跳过")
        return None

    # 建立高程-冰厚关系
    stats_list, _ = build_elevation_thickness_relationship(
        thickness, dem, glacier_mask, config['elevation_bin_size']
    )

    if len(stats_list) < 2:
        print(f"    警告: 冰川 {glacier_id} 高程带数据不足，跳过")
        return None

    # 创建插值函数
    elev_func = create_elevation_thickness_function(stats_list, config['interpolation_method'])

    # 填充零值区域
    filled_thickness = fill_zero_regions(thickness, dem, glacier_mask, elev_func)

    # 计算统计
    stats_before = calculate_statistics(thickness, glacier_mask)
    stats_after = calculate_statistics(filled_thickness, glacier_mask)

    return {
        'glacier_id': glacier_id,
        'glacier_mask': glacier_mask,
        'filled_thickness': filled_thickness,
        'stats_before': stats_before,
        'stats_after': stats_after,
        'flowline_ratio': flowline_ratio,
        'elev_stats': stats_list,
    }


def run_step6():
    """运行Step6"""
    print("=" * 70)
    print("Step 6: 基于高程的冰厚空间插值填充")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # ========== 1. 加载数据 ==========
    print("\n[1/5] 加载数据...")

    raster_data = load_thickness_raster(CONFIG['thickness_raster'])
    print(f"  冰厚栅格: {raster_data['shape']}")
    print(f"  栅格CRS: {raster_data['crs']}")

    # 加载冰川边界
    glaciers = gpd.read_file(CONFIG['glacier_path'])
    if glaciers.crs != raster_data['crs']:
        glaciers = glaciers.to_crs(raster_data['crs'])
    print(f"  冰川数量: {len(glaciers)}")

    # 加载DEM
    print("  加载DEM...")
    dem = load_dem_resampled(CONFIG['dem_path'], raster_data)
    print(f"  DEM高程范围: {dem.min():.0f} - {dem.max():.0f} m")

    # ========== 2. 处理冰川 ==========
    print("\n[2/5] 处理冰川...")

    if CONFIG['test_glacier_id'] is not None:
        # 单个测试冰川
        glacier_ids = [CONFIG['test_glacier_id']]
        print(f"  测试模式: 仅处理冰川 {CONFIG['test_glacier_id']}")
    else:
        # 所有冰川
        glacier_ids = glaciers['UniqueID'].tolist()
        print(f"  批量模式: 处理 {len(glacier_ids)} 个冰川")

    results = []
    filled_thickness_all = raster_data['thickness'].copy()

    for i, gid in enumerate(glacier_ids):
        if len(glacier_ids) > 1 and i % 50 == 0:
            print(f"  处理进度: {i}/{len(glacier_ids)}")

        result = process_single_glacier(gid, raster_data, dem, glaciers, CONFIG)

        if result is not None:
            results.append(result)
            # 更新全局栅格
            mask = result['glacier_mask']
            filled_thickness_all[mask] = result['filled_thickness'][mask]

    print(f"  成功处理: {len(results)} 个冰川")

    # ========== 3. 统计汇总 ==========
    print("\n[3/5] 统计汇总...")

    stats_rows = []
    for r in results:
        stats_rows.append({
            'glacier_id': r['glacier_id'],
            'area_km2': r['stats_after']['area_km2'],
            'volume_before_km3': r['stats_before']['volume_km3'],
            'volume_after_km3': r['stats_after']['volume_km3'],
            'volume_increase_pct': (r['stats_after']['volume_km3'] / r['stats_before']['volume_km3'] - 1) * 100 if r['stats_before']['volume_km3'] > 0 else 0,
            'mean_thickness_m': r['stats_after']['mean_thickness_m'],
            'max_thickness_m': r['stats_after']['max_thickness_m'],
            'flowline_coverage_pct': r['flowline_ratio'] * 100,
        })

    stats_df = pd.DataFrame(stats_rows)

    if len(stats_df) > 0:
        print(f"\n  总体统计:")
        print(f"    填充前总体积: {stats_df['volume_before_km3'].sum():.3f} km³")
        print(f"    填充后总体积: {stats_df['volume_after_km3'].sum():.3f} km³")
        print(f"    体积增加: {(stats_df['volume_after_km3'].sum() / stats_df['volume_before_km3'].sum() - 1) * 100:.1f}%")
        print(f"    平均流线覆盖率: {stats_df['flowline_coverage_pct'].mean():.1f}%")

    # ========== 4. 计算冰面高程 ==========
    print("\n[4/5] 计算冰面高程...")

    # 创建全局冰川掩膜
    all_glacier_mask = filled_thickness_all > 0
    ice_surface = calculate_ice_surface(filled_thickness_all, dem, all_glacier_mask)

    # ========== 5. 保存输出 ==========
    print("\n[5/5] 保存输出...")

    # 保存填充后的冰厚栅格
    thickness_path = os.path.join(CONFIG['output_dir'], 'step6_thickness_filled.tif')
    save_raster(filled_thickness_all, thickness_path, raster_data['profile'])
    print(f"  ✓ {thickness_path}")

    # 保存冰面高程栅格
    surface_path = os.path.join(CONFIG['output_dir'], 'step6_ice_surface.tif')
    save_raster(ice_surface, surface_path, raster_data['profile'])
    print(f"  ✓ {surface_path}")

    # 保存统计CSV
    stats_path = os.path.join(CONFIG['output_dir'], 'step6_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"  ✓ {stats_path}")

    # 保存高程-冰厚关系（仅测试模式）
    if CONFIG['test_glacier_id'] is not None and len(results) > 0:
        elev_stats = results[0]['elev_stats']
        elev_df = pd.DataFrame(elev_stats)
        elev_path = os.path.join(CONFIG['output_dir'], 'step6_elevation_thickness_relation.csv')
        elev_df.to_csv(elev_path, index=False)
        print(f"  ✓ {elev_path}")

    # ========== 报告 ==========
    print("\n" + "=" * 70)
    print("Step 6 完成!")
    print("=" * 70)

    if len(results) > 0:
        r = results[0]
        print(f"\n测试冰川 {r['glacier_id']} 结果:")
        print(f"  填充前体积: {r['stats_before']['volume_km3']:.3f} km³")
        print(f"  填充后体积: {r['stats_after']['volume_km3']:.3f} km³")
        print(f"  体积增加: {(r['stats_after']['volume_km3']/r['stats_before']['volume_km3']-1)*100:.1f}%")
        print(f"  平均冰厚: {r['stats_after']['mean_thickness_m']:.1f} m")
        print(f"  流线覆盖率: {r['flowline_ratio']*100:.1f}%")

    return filled_thickness_all, ice_surface, stats_df, results


if __name__ == "__main__":
    thickness, surface, stats, results = run_step6()