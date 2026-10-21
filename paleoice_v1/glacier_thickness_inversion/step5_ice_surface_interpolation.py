"""
Step 5 诊断脚本: 冰面空间插值与栅格生成
==========================================

功能:
1. [数据融合] 合并Step 4的流线冰厚点与LIA边界零值点
2. [多流线处理] 处理主流线+支流线的交汇融合
3. [空间插值] 将稀疏点云插值为连续栅格 (Linear/Cubic)
4. [物理平滑] 应用高斯滤波模拟冰流的自然表面张力
5. [边界约束] 确保冰厚在LIA边界外为0
6. [地形叠加] 生成最终的古冰面高程 (ice_surface = bed + thickness)
7. [统计输出] 计算总体积、最大冰厚、平均冰厚等

输入:
- step4_final_flowline.csv (Step 4输出的冰厚点云)
- glacier_success.shp (LIA边界)
- Himalaya_bare_DEM.tif (床面DEM)

输出:
- step5_thickness_raster.tif (冰厚栅格)
- step5_ice_surface_raster.tif (冰面高程栅格)
- step5_statistics.csv (统计数据)
输入:
├── step4_final_flowline.csv (流线冰厚点)
├── glacier_success.shp (LIA边界)
└── Himalaya_bare_DEM.tif (床面DEM)

处理:
[1] 加载数据
[2] 坐标转换 (WGS84 → UTM)
[3] 准备插值点
    ├── 流线冰厚点 (Step 4输出)
    └── 边界零值点 (沿LIA边界采样, H=0)
[4] 创建插值网格 (与DEM分辨率对齐)
[5] 空间插值 (Linear/Cubic/Nearest)
[6] 后处理
    ├── 边界掩膜 (确保边界外H=0)
    ├── 物理约束 (0 ≤ H ≤ 1500m)
    └── 高斯平滑 (模拟冰流表面张力)
[7] 计算冰面高程 (ice_surface = bed + thickness)
[8] 保存输出

输出:
├── step5_thickness_raster.tif (冰厚栅格)
├── step5_ice_surface_raster.tif (冰面高程栅格)
├── step5_statistics.csv (体积、面积、冰厚统计)
└── step5_*.png (可视化)

使用方法:
    python step5_ice_surface_interpolation.py
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from rasterio.crs import CRS as RioCRS
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from shapely.geometry import Point, LineString, MultiLineString, Polygon
from shapely.ops import transform
from pyproj import CRS, Transformer
from scipy.interpolate import griddata, LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter, binary_dilation
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================
CONFIG = {
    # 输入文件
    'flowline_csv': "/mnt/d/paleoice_output/step4_v8/step4_v8_flowlines.csv",
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_connected.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",

    # 输出目录
    'output_dir': "/mnt/d/paleoice_output/step5_diagnostic/",

    # 测试冰川
    'test_glacier_id': "RGI60-15.06881",

    # 插值参数
    'interpolation_method': 'linear',  # 'linear', 'cubic', 'nearest'
    'output_resolution': 30,           # 输出栅格分辨率 (米)
    'boundary_sample_spacing': 150,     # LIA边界采样间距 (米)

    # 平滑参数
    'apply_smoothing': True,
    'gaussian_sigma': 1.5,             # 高斯平滑核大小 (像素)

    # 物理约束
    'max_thickness': 1500,             # 最大冰厚限制 (米)
    'min_thickness': 0,                # 最小冰厚 (米)
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


# ============================================================================
# 数据加载函数
# ============================================================================

def load_flowline_thickness_points(csv_path):
    """
    加载Step 4输出的流线冰厚点

    Returns:
        DataFrame with columns: x_utm, y_utm, thickness_m, z_bed, z_ice_surface
    """
    df = pd.read_csv(csv_path)

    required_cols = ['x_utm', 'y_utm', 'thickness_m', 'z_bed', 'z_ice_surface']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 过滤无效数据
    df = df[df['thickness_m'] >= 0].copy()
    df = df.dropna(subset=['x_utm', 'y_utm', 'thickness_m'])

    return df


def sample_boundary_points(glacier_geom_utm, spacing_m=50):
    """
    沿LIA边界采样点 (冰厚=0的约束点)

    Args:
        glacier_geom_utm: UTM坐标下的冰川几何
        spacing_m: 采样间距 (米)

    Returns:
        list of (x, y) tuples
    """
    boundary = glacier_geom_utm.boundary

    if boundary.geom_type == 'MultiLineString':
        # 处理多边形有孔洞的情况
        lines = list(boundary.geoms)
    else:
        lines = [boundary]

    points = []
    for line in lines:
        length = line.length
        num_points = max(int(length / spacing_m), 2)
        distances = np.linspace(0, length, num_points)

        for d in distances:
            pt = line.interpolate(d)
            points.append((pt.x, pt.y))

    return points


def load_all_flowlines_for_glacier(flowlines_path, glacier_id, dem_path, utm_crs, spacing_m=50):
    """
    加载指定冰川的所有流线并采样冰厚

    注意: 这里只加载坐标，冰厚需要从Step 4的CSV获取
    如果有多条支流线，需要分别处理
    """
    flowlines_gdf = gpd.read_file(flowlines_path)
    glacier_flowlines = flowlines_gdf[flowlines_gdf['MainRGI'] == glacier_id]

    if len(glacier_flowlines) == 0:
        return None

    return glacier_flowlines


# ============================================================================
# 插值函数
# ============================================================================

def create_interpolation_grid(glacier_geom_utm, resolution_m, buffer_m=100):
    """
    创建插值网格

    Args:
        glacier_geom_utm: UTM坐标下的冰川几何
        resolution_m: 栅格分辨率 (米)
        buffer_m: 边界缓冲区 (米)

    Returns:
        xi, yi: 网格坐标数组
        transform: rasterio transform
        shape: (rows, cols)
    """
    bounds = glacier_geom_utm.bounds
    xmin, ymin, xmax, ymax = bounds

    # 添加小缓冲区
    xmin -= buffer_m
    ymin -= buffer_m
    xmax += buffer_m
    ymax += buffer_m

    # 创建网格
    cols = int((xmax - xmin) / resolution_m)
    rows = int((ymax - ymin) / resolution_m)

    x = np.linspace(xmin + resolution_m/2, xmax - resolution_m/2, cols)
    y = np.linspace(ymax - resolution_m/2, ymin + resolution_m/2, rows)  # 注意y方向

    xi, yi = np.meshgrid(x, y)

    # 创建rasterio transform
    transform = from_bounds(xmin, ymin, xmax, ymax, cols, rows)

    return xi, yi, transform, (rows, cols)


def interpolate_thickness(points_x, points_y, points_z, xi, yi, method='linear'):
    """
    执行空间插值

    Args:
        points_x, points_y: 采样点坐标
        points_z: 采样点冰厚值
        xi, yi: 插值网格坐标
        method: 插值方法 ('linear', 'cubic', 'nearest')

    Returns:
        插值后的冰厚网格
    """
    points = np.column_stack((points_x, points_y))

    # 使用griddata进行插值
    zi = griddata(points, points_z, (xi, yi), method=method)

    # 对于linear和cubic方法，边界外会产生NaN
    # 使用nearest neighbor填充NaN
    if method in ['linear', 'cubic']:
        nan_mask = np.isnan(zi)
        if np.any(nan_mask):
            # 用最近邻填充
            zi_nearest = griddata(points, points_z, (xi, yi), method='nearest')
            zi[nan_mask] = zi_nearest[nan_mask]

    return zi


def apply_boundary_mask(thickness_grid, glacier_geom_utm, xi, yi):
    """
    应用边界掩膜，确保冰厚在LIA边界外为0

    Args:
        thickness_grid: 冰厚网格
        glacier_geom_utm: UTM坐标下的冰川几何
        xi, yi: 网格坐标

    Returns:
        掩膜后的冰厚网格
    """
    rows, cols = thickness_grid.shape

    # 创建掩膜
    mask = np.zeros((rows, cols), dtype=bool)

    for i in range(rows):
        for j in range(cols):
            pt = Point(xi[i, j], yi[i, j])
            if glacier_geom_utm.contains(pt):
                mask[i, j] = True

    # 应用掩膜
    masked_thickness = np.where(mask, thickness_grid, 0)

    return masked_thickness, mask


def apply_boundary_mask_fast(thickness_grid, glacier_geom_utm, transform, shape):
    """
    快速应用边界掩膜 (使用rasterize)
    """
    from rasterio.features import rasterize

    mask = rasterize(
        [(glacier_geom_utm, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    masked_thickness = np.where(mask == 1, thickness_grid, 0)

    return masked_thickness, mask.astype(bool)


def apply_physical_constraints(thickness_grid, min_val=0, max_val=1500):
    """
    应用物理约束

    - 负值修正为0
    - 超过最大值的修正为最大值
    """
    thickness_grid = np.clip(thickness_grid, min_val, max_val)
    return thickness_grid


def apply_gaussian_smoothing(thickness_grid, sigma=1.5, mask=None):
    """
    边界保持型高斯平滑
    - 内部区域：应用平滑
    - 边界带：保持原值，避免边界冰厚被削弱
    """
    if mask is None:
        return gaussian_filter(thickness_grid, sigma=sigma)

    from scipy.ndimage import binary_erosion

    # 腐蚀掩膜，得到内部区域（远离边界）
    erosion_iterations = max(int(sigma * 2), 1)
    interior = binary_erosion(mask, iterations=erosion_iterations)

    # 对整个网格平滑
    smoothed = gaussian_filter(thickness_grid, sigma=sigma)

    # 内部用平滑值，边界带保持原值
    result = np.where(interior, smoothed, thickness_grid)

    # 边界外设为0
    result = np.where(mask, result, 0)

    return result


# ============================================================================
# 冰面高程计算
# ============================================================================

def calculate_ice_surface(thickness_grid, dem_path, transform, shape, utm_crs):
    """
    计算冰面高程 = 床面高程 + 冰厚

    Args:
        thickness_grid: 冰厚网格
        dem_path: bare_dem路径
        transform: 栅格transform
        shape: 栅格形状
        utm_crs: UTM坐标系

    Returns:
        ice_surface: 冰面高程网格
        bed_elevation: 床面高程网格
    """
    rows, cols = shape

    # 创建网格坐标
    src_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(utm_crs, src_crs, always_xy=True)

    # 获取栅格中心点坐标
    xs = np.zeros((rows, cols))
    ys = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            x, y = transform * (j + 0.5, i + 0.5)
            xs[i, j] = x
            ys[i, j] = y

    # 采样DEM
    bed_elevation = np.zeros((rows, cols))

    with rasterio.open(dem_path) as src:
        for i in range(rows):
            for j in range(cols):
                lon, lat = transformer.transform(xs[i, j], ys[i, j])
                try:
                    row, col = src.index(lon, lat)
                    if 0 <= row < src.height and 0 <= col < src.width:
                        val = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                        bed_elevation[i, j] = val if val > -9999 else np.nan
                    else:
                        bed_elevation[i, j] = np.nan
                except:
                    bed_elevation[i, j] = np.nan

    # 计算冰面高程
    ice_surface = bed_elevation + thickness_grid

    return ice_surface, bed_elevation


def calculate_ice_surface_fast(thickness_grid, dem_path, glacier_geom_utm, transform, shape, utm_crs):
    """
    快速计算冰面高程 - 使用rasterio.warp.reproject一次性重投影
    """
    from rasterio.warp import reproject, Resampling

    rows, cols = shape

    # 转换CRS格式
    if isinstance(utm_crs, CRS):
        dst_crs = RioCRS.from_wkt(utm_crs.to_wkt())
    else:
        dst_crs = utm_crs

    # 创建目标数组
    bed_elevation = np.zeros((rows, cols), dtype=np.float32)

    # 一次性重投影DEM到目标网格
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=bed_elevation,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

    # 处理无效值
    bed_elevation = np.where((bed_elevation < -9000) | (bed_elevation > 9000), np.nan, bed_elevation)

    # 计算冰面高程
    ice_surface = bed_elevation + thickness_grid

    return ice_surface, bed_elevation


# ============================================================================
# 统计计算
# ============================================================================

def calculate_statistics(thickness_grid, mask, resolution_m):
    """
    计算冰川统计数据

    Returns:
        dict: 包含各种统计量
    """
    valid_thickness = thickness_grid[mask]
    valid_thickness = valid_thickness[~np.isnan(valid_thickness)]
    valid_thickness = valid_thickness[valid_thickness > 0]

    if len(valid_thickness) == 0:
        return {
            'volume_km3': 0,
            'area_km2': 0,
            'mean_thickness_m': 0,
            'max_thickness_m': 0,
            'std_thickness_m': 0,
            'median_thickness_m': 0
        }

    # 计算面积 (km²)
    cell_area_m2 = resolution_m ** 2
    glaciated_cells = np.sum(mask)
    area_km2 = glaciated_cells * cell_area_m2 / 1e6

    # 计算体积 (km³)
    # V = Σ(H × cell_area)
    volume_m3 = np.sum(thickness_grid[mask]) * cell_area_m2
    volume_km3 = volume_m3 / 1e9

    return {
        'volume_km3': volume_km3,
        'area_km2': area_km2,
        'mean_thickness_m': np.mean(valid_thickness),
        'max_thickness_m': np.max(valid_thickness),
        'std_thickness_m': np.std(valid_thickness),
        'median_thickness_m': np.median(valid_thickness)
    }


# ============================================================================
# 栅格保存函数
# ============================================================================

def save_raster(data, output_path, transform, crs, nodata=-9999):
    """
    保存栅格数据为GeoTIFF
    """
    rows, cols = data.shape

    # 处理NaN
    data = np.where(np.isnan(data), nodata, data)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=rows,
        width=cols,
        count=1,
        dtype=data.dtype,
        crs=crs.to_string(),
        transform=transform,
        nodata=nodata,
        compress='lzw'
    ) as dst:
        dst.write(data, 1)


# ============================================================================
# 可视化函数
# ============================================================================

def plot_interpolation_result(thickness_grid, ice_surface, bed_elevation, mask,
                               points_x, points_y, points_z,
                               glacier_id, output_path):
    """
    绘制插值结果
    """
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig)

    # 1. 冰厚分布
    ax1 = fig.add_subplot(gs[0, 0])
    thickness_masked = np.ma.masked_where(~mask, thickness_grid)
    im1 = ax1.imshow(thickness_masked, cmap='Blues', aspect='equal')
    ax1.scatter([], [], c='red', s=5, label='Flowline points')  # 图例占位
    plt.colorbar(im1, ax=ax1, label='Ice thickness (m)')
    ax1.set_title('Ice Thickness Distribution')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')

    # 2. 冰面高程
    ax2 = fig.add_subplot(gs[0, 1])
    ice_surface_masked = np.ma.masked_where(~mask, ice_surface)
    im2 = ax2.imshow(ice_surface_masked, cmap='terrain', aspect='equal')
    plt.colorbar(im2, ax=ax2, label='Ice surface elevation (m)')
    ax2.set_title('Ice Surface Elevation')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')

    # 3. 床面高程
    ax3 = fig.add_subplot(gs[0, 2])
    bed_masked = np.ma.masked_where(~mask, bed_elevation)
    im3 = ax3.imshow(bed_masked, cmap='terrain', aspect='equal')
    plt.colorbar(im3, ax=ax3, label='Bed elevation (m)')
    ax3.set_title('Bed Elevation (bare DEM)')
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')

    # 4. 冰厚直方图
    ax4 = fig.add_subplot(gs[1, 0])
    valid_h = thickness_grid[mask & (thickness_grid > 0)]
    ax4.hist(valid_h.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(np.mean(valid_h), color='red', linestyle='--',
                label=f'Mean = {np.mean(valid_h):.1f} m')
    ax4.axvline(np.median(valid_h), color='orange', linestyle='--',
                label=f'Median = {np.median(valid_h):.1f} m')
    ax4.set_xlabel('Ice thickness (m)')
    ax4.set_ylabel('Pixel count')
    ax4.set_title('Ice Thickness Histogram')
    ax4.legend()

    # 5. 采样点分布
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(points_x, points_y, c=points_z, cmap='Blues', s=5, alpha=0.7)
    plt.colorbar(scatter, ax=ax5, label='Thickness (m)')
    ax5.set_xlabel('X (UTM)')
    ax5.set_ylabel('Y (UTM)')
    ax5.set_title(f'Sample Points ({len(points_x)} points)')
    ax5.set_aspect('equal')

    # 6. 统计摘要
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats = calculate_statistics(thickness_grid, mask, CONFIG['output_resolution'])

    stats_text = f"""
    === Ice Thickness Statistics ===
    
    Glacier: {glacier_id}
    
    Volume:     {stats['volume_km3']:.3f} km³
    Area:       {stats['area_km2']:.2f} km²
    
    Mean H:     {stats['mean_thickness_m']:.1f} m
    Max H:      {stats['max_thickness_m']:.1f} m
    Median H:   {stats['median_thickness_m']:.1f} m
    Std H:      {stats['std_thickness_m']:.1f} m
    
    Resolution: {CONFIG['output_resolution']} m
    Method:     {CONFIG['interpolation_method']}
    Smoothing:  σ = {CONFIG['gaussian_sigma']} px
    """

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
            fontsize=11, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle(f'{glacier_id} - Ice Thickness Interpolation Result', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_cross_section(thickness_grid, bed_elevation, ice_surface, mask,
                       transform, glacier_id, output_path):
    """
    绘制横截面剖面图
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    rows, cols = thickness_grid.shape

    # 找到冰川中心行和列
    valid_rows, valid_cols = np.where(mask)
    center_row = int(np.median(valid_rows))
    center_col = int(np.median(valid_cols))

    # 1. 东西向剖面 (沿行)
    ax1 = axes[0, 0]
    row_data = thickness_grid[center_row, :]
    bed_row = bed_elevation[center_row, :]
    ice_row = ice_surface[center_row, :]
    mask_row = mask[center_row, :]

    x_coords = np.arange(cols) * CONFIG['output_resolution']

    ax1.fill_between(x_coords, np.nanmin(bed_row[mask_row]) - 100, bed_row,
                     where=mask_row, alpha=0.3, color='brown', label='Bed')
    ax1.plot(x_coords, bed_row, 'k-', linewidth=1)
    ax1.fill_between(x_coords, bed_row, ice_row,
                     where=mask_row & (row_data > 0), alpha=0.5, color='lightblue', label='Ice')
    ax1.plot(x_coords, ice_row, 'b-', linewidth=2, label='Ice surface')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.set_title(f'E-W Cross Section (Row {center_row})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 南北向剖面 (沿列)
    ax2 = axes[0, 1]
    col_data = thickness_grid[:, center_col]
    bed_col = bed_elevation[:, center_col]
    ice_col = ice_surface[:, center_col]
    mask_col = mask[:, center_col]

    y_coords = np.arange(rows) * CONFIG['output_resolution']

    ax2.fill_between(y_coords, np.nanmin(bed_col[mask_col]) - 100, bed_col,
                     where=mask_col, alpha=0.3, color='brown', label='Bed')
    ax2.plot(y_coords, bed_col, 'k-', linewidth=1)
    ax2.fill_between(y_coords, bed_col, ice_col,
                     where=mask_col & (col_data > 0), alpha=0.5, color='lightblue', label='Ice')
    ax2.plot(y_coords, ice_col, 'b-', linewidth=2, label='Ice surface')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Elevation (m)')
    ax2.set_title(f'N-S Cross Section (Col {center_col})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 厚度沿流线方向变化
    ax3 = axes[1, 0]
    # 找到厚度最大的点，绘制通过该点的剖面
    max_idx = np.unravel_index(np.argmax(thickness_grid * mask), thickness_grid.shape)

    diag_row = thickness_grid[max_idx[0], :]
    diag_mask = mask[max_idx[0], :]

    ax3.plot(x_coords[diag_mask], diag_row[diag_mask], 'b-', linewidth=2)
    ax3.fill_between(x_coords[diag_mask], 0, diag_row[diag_mask], alpha=0.3, color='lightblue')
    ax3.axhline(y=np.mean(diag_row[diag_mask & (diag_row > 0)]), color='red',
               linestyle='--', label='Mean')
    ax3.set_xlabel('Distance (m)')
    ax3.set_ylabel('Ice thickness (m)')
    ax3.set_title(f'Thickness Profile (Row {max_idx[0]}, through max)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 冰厚等值线
    ax4 = axes[1, 1]
    thickness_masked = np.ma.masked_where(~mask | (thickness_grid <= 0), thickness_grid)

    levels = np.linspace(0, np.nanmax(thickness_grid[mask]), 10)
    contour = ax4.contourf(thickness_masked, levels=levels, cmap='Blues')
    ax4.contour(thickness_masked, levels=levels, colors='white', linewidths=0.5, alpha=0.5)
    plt.colorbar(contour, ax=ax4, label='Ice thickness (m)')
    ax4.set_title('Ice Thickness Contours')
    ax4.set_aspect('equal')

    plt.suptitle(f'{glacier_id} - Cross Sections', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主诊断函数
# ============================================================================

def run_diagnostic():
    """运行Step 5诊断"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    print("=" * 70)
    print("Step 5 诊断: 冰面空间插值与栅格生成")
    print("=" * 70)

    # ========== 1. 加载数据 ==========
    print("\n[1/8] 加载数据...")

    # 加载流线冰厚点
    flowline_df = load_flowline_thickness_points(CONFIG['flowline_csv'])
    print(f"  流线冰厚点: {len(flowline_df)} 个")
    print(f"  冰厚范围: {flowline_df['thickness_m'].min():.1f} - {flowline_df['thickness_m'].max():.1f} m")

    # 加载冰川边界
    glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
    test_glacier = glacier_gdf[glacier_gdf['OrigRGIId'] == CONFIG['test_glacier_id']]

    if len(test_glacier) == 0:
        print(f"  ✗ 找不到测试冰川: {CONFIG['test_glacier_id']}")
        return

    glacier_geom = test_glacier.geometry.values[0]
    print(f"  测试冰川: {CONFIG['test_glacier_id']}")

    # ========== 2. 坐标转换 ==========
    print("\n[2/8] 坐标转换...")

    centroid = glacier_geom.centroid
    utm_crs, utm_zone = get_utm_crs(centroid.x, centroid.y)
    src_crs = CRS.from_epsg(4326)

    glacier_utm = transform_geometry(glacier_geom, src_crs, utm_crs)

    print(f"  UTM Zone: {utm_zone}N")
    print(f"  冰川面积: {glacier_utm.area / 1e6:.2f} km²")

    # ========== 3. 准备插值点 ==========
    print("\n[3/8] 准备插值点...")

    # 流线冰厚点 (已在UTM坐标)
    points_x = flowline_df['x_utm'].values
    points_y = flowline_df['y_utm'].values
    points_z = flowline_df['thickness_m'].values

    print(f"  流线点: {len(points_x)} 个")

    # 边界零值点
    boundary_points = sample_boundary_points(glacier_utm, CONFIG['boundary_sample_spacing'])
    boundary_x = np.array([p[0] for p in boundary_points])
    boundary_y = np.array([p[1] for p in boundary_points])
    boundary_z = np.zeros(len(boundary_points))

    print(f"  边界零值点: {len(boundary_points)} 个")

    # 合并所有点
    all_x = np.concatenate([points_x, boundary_x])
    all_y = np.concatenate([points_y, boundary_y])
    all_z = np.concatenate([points_z, boundary_z])

    print(f"  总插值点: {len(all_x)} 个")

    # ========== 4. 创建插值网格 ==========
    print("\n[4/8] 创建插值网格...")

    xi, yi, raster_transform, shape = create_interpolation_grid(
        glacier_utm, CONFIG['output_resolution']
    )

    print(f"  网格大小: {shape[0]} rows × {shape[1]} cols")
    print(f"  分辨率: {CONFIG['output_resolution']} m")

    # ========== 5. 执行插值 ==========
    print("\n[5/8] 执行空间插值...")
    print(f"  插值方法: {CONFIG['interpolation_method']}")

    thickness_grid = interpolate_thickness(
        all_x, all_y, all_z, xi, yi,
        method=CONFIG['interpolation_method']
    )

    print(f"  插值完成")
    print(f"  原始冰厚范围: {np.nanmin(thickness_grid):.1f} - {np.nanmax(thickness_grid):.1f} m")

    # ========== 6. 后处理 ==========
    print("\n[6/8] 后处理...")

    # 应用边界掩膜
    print("  - 应用边界掩膜...")
    thickness_grid, mask = apply_boundary_mask_fast(
        thickness_grid, glacier_utm, raster_transform, shape
    )

    # 应用物理约束
    print("  - 应用物理约束...")
    thickness_grid = apply_physical_constraints(
        thickness_grid,
        CONFIG['min_thickness'],
        CONFIG['max_thickness']
    )

    # 应用高斯平滑
    if CONFIG['apply_smoothing']:
        print(f"  - 应用高斯平滑 (σ={CONFIG['gaussian_sigma']})...")
        thickness_grid = apply_gaussian_smoothing(
            thickness_grid,
            CONFIG['gaussian_sigma'],
            mask
        )

    print(f"  最终冰厚范围: {np.min(thickness_grid[mask]):.1f} - {np.max(thickness_grid[mask]):.1f} m")

    # ========== 7. 计算冰面高程 ==========
    print("\n[7/8] 计算冰面高程...")

    ice_surface, bed_elevation = calculate_ice_surface_fast(
        thickness_grid, CONFIG['dem_path'], glacier_utm,
        raster_transform, shape, utm_crs
    )

    valid_ice = ice_surface[mask & ~np.isnan(ice_surface)]
    print(f"  冰面高程范围: {np.min(valid_ice):.1f} - {np.max(valid_ice):.1f} m")

    # ========== 8. 保存输出 ==========
    print("\n[8/8] 保存输出...")

    # 保存冰厚栅格
    thickness_path = CONFIG['output_dir'] + 'step5_thickness_raster.tif'
    save_raster(thickness_grid.astype(np.float32), thickness_path,
                raster_transform, utm_crs)
    print(f"  ✓ {thickness_path}")

    # 保存冰面高程栅格
    ice_surface_path = CONFIG['output_dir'] + 'step5_ice_surface_raster.tif'
    save_raster(ice_surface.astype(np.float32), ice_surface_path,
                raster_transform, utm_crs)
    print(f"  ✓ {ice_surface_path}")

    # 计算统计数据
    stats = calculate_statistics(thickness_grid, mask, CONFIG['output_resolution'])

    # 保存统计CSV
    stats_df = pd.DataFrame([{
        'glacier_id': CONFIG['test_glacier_id'],
        **stats,
        'resolution_m': CONFIG['output_resolution'],
        'interpolation_method': CONFIG['interpolation_method'],
        'smoothing_sigma': CONFIG['gaussian_sigma'] if CONFIG['apply_smoothing'] else 0,
        'n_flowline_points': len(points_x),
        'n_boundary_points': len(boundary_points),
        'n_total_points': len(all_x)
    }])

    stats_path = CONFIG['output_dir'] + 'step5_statistics.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"  ✓ {stats_path}")

    # 生成可视化
    print("\n[Extra] 生成可视化...")

    result_path = CONFIG['output_dir'] + 'step5_interpolation_result.png'
    plot_interpolation_result(thickness_grid, ice_surface, bed_elevation, mask,
                               all_x, all_y, all_z,
                               CONFIG['test_glacier_id'], result_path)
    print(f"  ✓ {result_path}")

    section_path = CONFIG['output_dir'] + 'step5_cross_sections.png'
    plot_cross_section(thickness_grid, bed_elevation, ice_surface, mask,
                       raster_transform, CONFIG['test_glacier_id'], section_path)
    print(f"  ✓ {section_path}")

    # ========== 生成报告 ==========
    report = []
    report.append("=" * 70)
    report.append("Step 5 诊断报告: 冰面空间插值与栅格生成")
    report.append("=" * 70)

    report.append("\n## 测试配置")
    report.append(f"- 测试冰川: {CONFIG['test_glacier_id']}")
    report.append(f"- 输出分辨率: {CONFIG['output_resolution']} m")
    report.append(f"- 插值方法: {CONFIG['interpolation_method']}")
    report.append(f"- 高斯平滑: σ = {CONFIG['gaussian_sigma']} px")

    report.append("\n## 插值点统计")
    report.append(f"- 流线冰厚点: {len(points_x)}")
    report.append(f"- 边界零值点: {len(boundary_points)}")
    report.append(f"- 总插值点: {len(all_x)}")

    report.append("\n## 输出栅格")
    report.append(f"- 栅格大小: {shape[0]} × {shape[1]} pixels")
    report.append(f"- 冰厚范围: {np.min(thickness_grid[mask]):.1f} - {np.max(thickness_grid[mask]):.1f} m")
    report.append(f"- 冰面高程范围: {np.min(valid_ice):.1f} - {np.max(valid_ice):.1f} m")

    report.append("\n## 冰川统计")
    report.append(f"- 体积: {stats['volume_km3']:.3f} km³")
    report.append(f"- 面积: {stats['area_km2']:.2f} km²")
    report.append(f"- 平均冰厚: {stats['mean_thickness_m']:.1f} m")
    report.append(f"- 最大冰厚: {stats['max_thickness_m']:.1f} m")
    report.append(f"- 中位数冰厚: {stats['median_thickness_m']:.1f} m")

    report.append("\n## 输出文件")
    report.append(f"- step5_thickness_raster.tif (冰厚栅格)")
    report.append(f"- step5_ice_surface_raster.tif (冰面高程栅格)")
    report.append(f"- step5_statistics.csv (统计数据)")

    report.append("\n## 结论")

    if stats['volume_km3'] > 0 and stats['mean_thickness_m'] > 50:
        report.append("✓ 插值成功，冰厚分布合理")
    else:
        report.append("⚠ 插值结果可能异常，请检查输入数据")

    if stats['max_thickness_m'] < CONFIG['max_thickness']:
        report.append(f"✓ 最大冰厚 {stats['max_thickness_m']:.1f}m 在物理限制范围内")

    report.append("\n✓ Step 5 验证完成")
    report.append("=" * 70)

    report_text = "\n".join(report)
    report_path = CONFIG['output_dir'] + 'step5_diagnostic_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n报告已保存: {report_path}")
    print(report_text)

    print("\n" + "=" * 70)
    print("诊断完成! 输出文件:")
    print(f"  1. {thickness_path}")
    print(f"  2. {ice_surface_path}")
    print(f"  3. {stats_path}")
    print(f"  4. {report_path}")
    print("=" * 70)

    return {
        'thickness_grid': thickness_grid,
        'ice_surface': ice_surface,
        'bed_elevation': bed_elevation,
        'mask': mask,
        'stats': stats
    }


if __name__ == "__main__":
    results = run_diagnostic()