"""
Step5: 基于高程的冰厚空间插值
==============================
核心思路：
- 相同高程带的冰厚应该相似
- 对每个零值像元，找同高程带的流线冰厚作为参考
- 结合边界距离约束（边界处冰厚→0）
"""

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import os
from datetime import datetime

# ============== 配置 ==============
CONFIG = {
    'thickness_raster': "/mnt/d/paleoice_output/step5_diagnostic/step5_thickness_raster.tif",
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'test_glacier_id': 2692,
    'output_dir': "/mnt/d/paleoice_output/step5_diagnostic/",

    # 高程带参数
    'elevation_bin_size': 50,  # 高程带宽度(m)
}


def load_data(config):
    """加载所有数据"""
    print("=" * 60)
    print("加载数据...")

    # 1. 加载冰厚栅格
    with rasterio.open(config['thickness_raster']) as src:
        thickness = src.read(1)
        transform = src.transform
        raster_crs = src.crs
        profile = src.profile.copy()
        raster_bounds = src.bounds
        raster_shape = thickness.shape

    print(f"  冰厚栅格: {thickness.shape}")
    print(f"  栅格CRS: {raster_crs}")

    # 2. 加载并重投影冰川边界
    glaciers = gpd.read_file(config['glacier_path'])
    if glaciers.crs != raster_crs:
        glaciers = glaciers.to_crs(raster_crs)

    glacier = glaciers[glaciers['UniqueID'] == config['test_glacier_id']].iloc[0]
    glacier_geom = glacier.geometry

    # 3. 创建冰川掩膜
    glacier_mask = ~geometry_mask(
        [glacier_geom],
        out_shape=raster_shape,
        transform=transform,
        invert=False
    )

    # 4. 加载DEM并重采样到冰厚栅格网格
    print("  加载DEM...")
    with rasterio.open(config['dem_path']) as dem_src:
        # 创建目标数组
        dem_resampled = np.zeros(raster_shape, dtype=np.float32)

        reproject(
            source=rasterio.band(dem_src, 1),
            destination=dem_resampled,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=transform,
            dst_crs=raster_crs,
            resampling=Resampling.bilinear
        )

    # 5. 统计
    glacier_pixels = np.sum(glacier_mask)
    flowline_pixels = np.sum((thickness > 0) & glacier_mask)
    zero_pixels = np.sum((thickness == 0) & glacier_mask)

    print(f"  冰川面积: {glacier_pixels * 30 * 30 / 1e6:.2f} km²")
    print(f"  流线覆盖: {flowline_pixels} ({100*flowline_pixels/glacier_pixels:.1f}%)")
    print(f"  零值区域: {zero_pixels} ({100*zero_pixels/glacier_pixels:.1f}%)")

    # 高程统计
    glacier_dem = dem_resampled[glacier_mask]
    print(f"  高程范围: {glacier_dem.min():.0f} - {glacier_dem.max():.0f} m")

    return {
        'thickness': thickness,
        'dem': dem_resampled,
        'glacier_mask': glacier_mask,
        'transform': transform,
        'crs': raster_crs,
        'profile': profile
    }


def build_elevation_thickness_relationship(thickness, dem, glacier_mask, bin_size=50):
    """
    建立高程-冰厚关系

    对每个高程带，统计流线冰厚的分布
    """
    print(f"\n建立高程-冰厚关系 (带宽={bin_size}m)...")

    flowline_mask = (thickness > 0) & glacier_mask

    # 获取流线区域的高程和冰厚
    flowline_elev = dem[flowline_mask]
    flowline_thick = thickness[flowline_mask]

    # 确定高程带范围
    elev_min = np.floor(dem[glacier_mask].min() / bin_size) * bin_size
    elev_max = np.ceil(dem[glacier_mask].max() / bin_size) * bin_size
    elevation_bins = np.arange(elev_min, elev_max + bin_size, bin_size)

    print(f"  高程范围: {elev_min:.0f} - {elev_max:.0f} m")
    print(f"  高程带数量: {len(elevation_bins) - 1}")

    # 统计每个高程带
    stats = []
    for i in range(len(elevation_bins) - 1):
        e_low = elevation_bins[i]
        e_high = elevation_bins[i + 1]
        e_mid = (e_low + e_high) / 2

        # 该高程带的流线冰厚
        band_mask = (flowline_elev >= e_low) & (flowline_elev < e_high)
        band_thickness = flowline_thick[band_mask]

        if len(band_thickness) > 0:
            stats.append({
                'elev_mid': e_mid,
                'elev_low': e_low,
                'elev_high': e_high,
                'count': len(band_thickness),
                'mean': band_thickness.mean(),
                'std': band_thickness.std(),
                'median': np.median(band_thickness),
                'max': band_thickness.max(),
                'p25': np.percentile(band_thickness, 25),
                'p75': np.percentile(band_thickness, 75),
            })

    # 打印关键高程带
    print(f"\n  {'高程(m)':<12} {'样本数':<8} {'平均冰厚':<10} {'中位数':<10} {'最大':<10}")
    print("  " + "-" * 55)
    for s in stats[::max(1, len(stats)//10)]:  # 打印约10行
        print(f"  {s['elev_mid']:<12.0f} {s['count']:<8} {s['mean']:<10.1f} {s['median']:<10.1f} {s['max']:<10.1f}")

    return stats, elevation_bins


def create_elevation_thickness_function(stats, method='mean'):
    """
    创建高程→冰厚的插值函数

    method: 'mean', 'median', 'max', 'p75'
    """
    if len(stats) < 2:
        print("  警告：高程带数据不足，无法建立关系")
        return None

    elevations = np.array([s['elev_mid'] for s in stats])
    thicknesses = np.array([s[method] for s in stats])

    # 创建插值函数，边界外使用最近值
    func = interp1d(elevations, thicknesses,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(thicknesses[0], thicknesses[-1]))

    return func


def interpolate_by_elevation(thickness, dem, glacier_mask, elev_func, boundary_weight=True):
    """
    基于高程的冰厚插值

    Parameters:
    -----------
    elev_func : callable
        高程→冰厚的函数
    boundary_weight : bool
        是否应用边界距离权重（边界处→0）
    """
    print("\n执行高程插值...")

    result = thickness.copy()
    zero_mask = (thickness == 0) & glacier_mask

    # 获取零值区域的高程
    zero_elevations = dem[zero_mask]

    # 用高程函数估算冰厚
    estimated_thickness = elev_func(zero_elevations)

    print(f"  高程估算冰厚: [{estimated_thickness.min():.1f}, {estimated_thickness.max():.1f}] m")
    print(f"  高程估算平均: {estimated_thickness.mean():.1f} m")

    if boundary_weight:
        # 计算边界距离权重
        boundary_dist = ndimage.distance_transform_edt(glacier_mask)

        # 归一化：使用特征距离（如中位数距离的2倍）
        median_dist = np.median(boundary_dist[glacier_mask])
        characteristic_dist = median_dist * 2

        # 权重：边界处0，远离边界趋近1
        # 使用tanh函数使过渡更平滑
        weight = np.tanh(boundary_dist / characteristic_dist * 2)

        print(f"  边界距离特征值: {characteristic_dist:.1f} 像元")
        print(f"  权重范围: [{weight[zero_mask].min():.3f}, {weight[zero_mask].max():.3f}]")

        # 应用权重
        estimated_thickness = estimated_thickness * weight[zero_mask]

        print(f"  加权后冰厚: [{estimated_thickness.min():.1f}, {estimated_thickness.max():.1f}] m")
        print(f"  加权后平均: {estimated_thickness.mean():.1f} m")

    # 填充
    result[zero_mask] = estimated_thickness

    return result


def compare_interpolation_methods(thickness, dem, glacier_mask, stats):
    """比较不同统计量的插值效果"""
    print("\n" + "=" * 60)
    print("比较不同高程-冰厚关系...")

    methods = ['mean', 'median', 'p75', 'max']
    results = {}

    flowline_mask = (thickness > 0) & glacier_mask
    flowline_mean = thickness[flowline_mask].mean()
    original_volume = np.sum(thickness[glacier_mask]) * 30 * 30 / 1e9

    print(f"\n原始统计:")
    print(f"  流线平均冰厚: {flowline_mean:.1f} m")
    print(f"  原始体积: {original_volume:.3f} km³")

    print(f"\n{'方法':<10} {'填充均值':<12} {'填充中位数':<12} {'总体积':<12} {'平均冰厚':<12}")
    print("-" * 60)

    for method in methods:
        # 创建高程函数
        elev_func = create_elevation_thickness_function(stats, method=method)
        if elev_func is None:
            continue

        # 插值（带边界权重）
        result = interpolate_by_elevation(thickness, dem, glacier_mask, elev_func, boundary_weight=True)

        zero_mask = (thickness == 0) & glacier_mask
        filled_mean = result[zero_mask].mean()
        filled_median = np.median(result[zero_mask])
        total_volume = np.sum(result[glacier_mask]) * 30 * 30 / 1e9
        mean_thickness = result[glacier_mask].mean()

        print(f"{method:<10} {filled_mean:<12.1f} {filled_median:<12.1f} {total_volume:<12.3f} {mean_thickness:<12.1f}")

        results[method] = {
            'result': result,
            'elev_func': elev_func,
            'filled_mean': filled_mean,
            'total_volume': total_volume,
            'mean_thickness': mean_thickness
        }

    # 额外测试：不带边界权重
    print("\n--- 不带边界权重 ---")
    for method in ['mean', 'median']:
        elev_func = create_elevation_thickness_function(stats, method=method)
        result = interpolate_by_elevation(thickness, dem, glacier_mask, elev_func, boundary_weight=False)

        zero_mask = (thickness == 0) & glacier_mask
        filled_mean = result[zero_mask].mean()
        total_volume = np.sum(result[glacier_mask]) * 30 * 30 / 1e9
        mean_thickness = result[glacier_mask].mean()

        print(f"{method+'_no_bw':<10} {filled_mean:<12.1f} {'-':<12} {total_volume:<12.3f} {mean_thickness:<12.1f}")

    return results


def analyze_zero_region_elevation(thickness, dem, glacier_mask):
    """分析零值区域的高程分布"""
    print("\n" + "=" * 60)
    print("零值区域高程分析...")

    flowline_mask = (thickness > 0) & glacier_mask
    zero_mask = (thickness == 0) & glacier_mask

    flowline_elev = dem[flowline_mask]
    zero_elev = dem[zero_mask]

    print(f"\n  {'区域':<12} {'最低':<10} {'最高':<10} {'平均':<10} {'中位数':<10}")
    print("  " + "-" * 50)
    print(f"  {'流线区域':<12} {flowline_elev.min():<10.0f} {flowline_elev.max():<10.0f} {flowline_elev.mean():<10.0f} {np.median(flowline_elev):<10.0f}")
    print(f"  {'零值区域':<12} {zero_elev.min():<10.0f} {zero_elev.max():<10.0f} {zero_elev.mean():<10.0f} {np.median(zero_elev):<10.0f}")

    # 检查高程重叠
    overlap_low = max(flowline_elev.min(), zero_elev.min())
    overlap_high = min(flowline_elev.max(), zero_elev.max())

    if overlap_low < overlap_high:
        print(f"\n  高程重叠范围: {overlap_low:.0f} - {overlap_high:.0f} m")
        # 重叠范围内的零值像元数
        overlap_zeros = np.sum((zero_elev >= overlap_low) & (zero_elev <= overlap_high))
        print(f"  重叠范围内零值像元: {overlap_zeros} ({100*overlap_zeros/len(zero_elev):.1f}%)")
    else:
        print(f"\n  警告：流线和零值区域高程无重叠！")
        print(f"  流线高程: {flowline_elev.min():.0f} - {flowline_elev.max():.0f}")
        print(f"  零值高程: {zero_elev.min():.0f} - {zero_elev.max():.0f}")


def smooth_result(result, glacier_mask, sigma=1.5):
    """高斯平滑"""
    print(f"\n平滑处理 (sigma={sigma})...")

    smoothed = ndimage.gaussian_filter(result.astype(np.float64), sigma=sigma)
    smoothed[~glacier_mask] = 0

    # 边界处理
    boundary = glacier_mask & ~ndimage.binary_erosion(glacier_mask)
    smoothed[boundary] = 0

    return smoothed.astype(np.float32)


def save_result(result, data, config, suffix):
    """保存结果"""
    output_path = os.path.join(config['output_dir'], f"step5_thickness_{suffix}.tif")

    profile = data['profile'].copy()
    profile.update(dtype='float32', compress='lzw')

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(result.astype(np.float32), 1)

    print(f"保存: {output_path}")
    return output_path


def main():
    """主函数"""
    print("=" * 60)
    print("Step5: 基于高程的冰厚空间插值")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 加载数据
    data = load_data(CONFIG)

    # 2. 分析零值区域高程
    analyze_zero_region_elevation(data['thickness'], data['dem'], data['glacier_mask'])

    # 3. 建立高程-冰厚关系
    stats, elevation_bins = build_elevation_thickness_relationship(
        data['thickness'], data['dem'], data['glacier_mask'],
        bin_size=CONFIG['elevation_bin_size']
    )

    # 4. 比较不同方法
    results = compare_interpolation_methods(
        data['thickness'], data['dem'], data['glacier_mask'], stats
    )

    # 5. 选择最优结果：mean_no_bw（高程关系已隐含边界效应，不需双重衰减）
    print("\n" + "=" * 60)
    print("生成最终结果 (mean_no_bw)...")

    elev_func_mean = create_elevation_thickness_function(stats, method='mean')
    best_result = interpolate_by_elevation(
        data['thickness'], data['dem'], data['glacier_mask'],
        elev_func_mean, boundary_weight=False
    )

    # 6. 平滑
    smoothed = smooth_result(best_result, data['glacier_mask'], sigma=0.5)

    # 7. 最终统计
    print("\n" + "=" * 60)
    print("最终结果:")
    glacier_mask = data['glacier_mask']
    final_volume = np.sum(smoothed[glacier_mask]) * 30 * 30 / 1e9
    final_mean = smoothed[glacier_mask].mean()
    print(f"  总体积: {final_volume:.3f} km³")
    print(f"  平均冰厚: {final_mean:.1f} m")

    # 8. 保存
    save_result(smoothed, data, CONFIG, "elev_mean_no_bw")
    save_result(best_result, data, CONFIG, "elev_mean_no_bw_raw")
    save_result(results['median']['result'], data, CONFIG, "elev_median_raw")
    save_result(results['max']['result'], data, CONFIG, "elev_max_raw")

    print("\n完成！")
    return smoothed, results, stats


if __name__ == "__main__":
    result, all_results, stats = main()