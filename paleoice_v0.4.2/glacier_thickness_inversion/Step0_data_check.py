"""
Step 0 数据检查诊断脚本
========================

检查内容:
1. 文件存在性和可读性
2. CRS一致性
3. 字段完整性（关键字段是否存在）
4. 空间覆盖范围（DEM是否覆盖所有冰川）
5. 流线拓扑关系（支流-主流汇合点）
6. bare_dem质量检查（异常值、空洞）
7. 冰川-流线匹配度

使用方法:
    python step0_data_check.py
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box, Point, mapping
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置路径 - 请根据你的实际路径修改
# ============================================================================
CONFIG = {
    'glacier_path': "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp",
    'flowlines_path': "/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_id.shp",
    'dem_path': "/mnt/f/oggm_related/oggm_LIA_himalaya_great/Himalaya_bare_DEM.tif",
    'output_dir': "/mnt/d/paleoice_output/step0_diagnostic/",
}


# ============================================================================
# 检查函数
# ============================================================================

def check_file_exists(filepath, name):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return True, f"✓ {name}: 存在 ({size_mb:.1f} MB)"
    else:
        return False, f"✗ {name}: 文件不存在 - {filepath}"


def check_shapefile_complete(shp_path, name):
    """检查shapefile附属文件是否完整"""
    base = shp_path.replace('.shp', '')
    required_ext = ['.shp', '.shx', '.dbf']
    optional_ext = ['.prj', '.cpg']

    missing_required = []
    missing_optional = []

    for ext in required_ext:
        if not os.path.exists(base + ext):
            missing_required.append(ext)

    for ext in optional_ext:
        if not os.path.exists(base + ext):
            missing_optional.append(ext)

    if missing_required:
        return False, f"✗ {name}: 缺少必需文件 {missing_required}"
    elif missing_optional:
        return True, f"⚠ {name}: 缺少可选文件 {missing_optional} (可能影响CRS)"
    else:
        return True, f"✓ {name}: 所有附属文件完整"


def check_crs_consistency(glacier_gdf, flowlines_gdf, dem_crs):
    """检查CRS一致性"""
    results = []

    glacier_crs = glacier_gdf.crs
    flowlines_crs = flowlines_gdf.crs

    results.append(f"  冰川边界 CRS: {glacier_crs}")
    results.append(f"  流线 CRS: {flowlines_crs}")
    results.append(f"  DEM CRS: {dem_crs}")

    # 检查是否一致
    all_match = True

    if glacier_crs != flowlines_crs:
        results.append(f"  ⚠ 冰川边界与流线CRS不一致!")
        all_match = False

    if glacier_crs is not None and dem_crs is not None:
        # 简单比较EPSG代码
        g_epsg = glacier_crs.to_epsg() if hasattr(glacier_crs, 'to_epsg') else None
        d_epsg = dem_crs.to_epsg() if hasattr(dem_crs, 'to_epsg') else None

        if g_epsg and d_epsg and g_epsg != d_epsg:
            results.append(f"  ⚠ 冰川边界与DEM的EPSG不一致: {g_epsg} vs {d_epsg}")
            all_match = False

    if all_match:
        results.insert(0, "✓ CRS一致性检查通过")
    else:
        results.insert(0, "✗ CRS一致性检查: 存在不一致")

    return all_match, results


def check_required_fields(gdf, required_fields, name):
    """检查必需字段是否存在"""
    existing = list(gdf.columns)
    missing = [f for f in required_fields if f not in existing]

    if missing:
        return False, f"✗ {name}: 缺少字段 {missing}\n    现有字段: {existing}"
    else:
        return True, f"✓ {name}: 必需字段完整 {required_fields}"


def check_spatial_coverage(glacier_gdf, dem_bounds):
    """检查DEM是否覆盖所有冰川"""
    dem_box = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)

    total = len(glacier_gdf)
    covered = 0
    partial = 0
    outside = 0
    outside_ids = []

    for idx, row in glacier_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if dem_box.contains(geom):
            covered += 1
        elif dem_box.intersects(geom):
            partial += 1
        else:
            outside += 1
            # 获取ID
            if 'OrigRGIId' in glacier_gdf.columns:
                outside_ids.append(row['OrigRGIId'])
            else:
                outside_ids.append(idx)

    results = []
    results.append(f"  总冰川数: {total}")
    results.append(f"  完全覆盖: {covered}")
    results.append(f"  部分覆盖: {partial}")
    results.append(f"  完全在外: {outside}")

    if outside > 0:
        results.append(f"  ⚠ DEM范围外的冰川: {outside_ids[:10]}{'...' if len(outside_ids) > 10 else ''}")
        return False, results
    elif partial > 0:
        results.insert(0, "⚠ 空间覆盖检查: 部分冰川仅部分覆盖")
        return True, results
    else:
        results.insert(0, "✓ 空间覆盖检查: DEM完全覆盖所有冰川")
        return True, results


def check_flowline_topology(flowlines_gdf, glacier_gdf):
    """检查流线拓扑关系"""
    results = []

    # 检查关键拓扑字段
    topo_fields = ['OFID', 'NEAR_FID', 'FlowlineID', 'SegmentID']
    existing_topo = [f for f in topo_fields if f in flowlines_gdf.columns]

    results.append(f"  拓扑相关字段: {existing_topo if existing_topo else '无'}")

    # 检查流线-冰川匹配
    if 'MainRGI' in flowlines_gdf.columns:
        fl_ids = set(flowlines_gdf['MainRGI'].unique())
        id_field = 'MainRGI'
    elif 'GlacierID' in flowlines_gdf.columns:
        fl_ids = set(flowlines_gdf['GlacierID'].unique())
        id_field = 'GlacierID'
    else:
        results.append("  ⚠ 流线中没有冰川ID字段 (MainRGI/GlacierID)")
        return False, results

    if 'OrigRGIId' in glacier_gdf.columns:
        gl_ids = set(glacier_gdf['OrigRGIId'].unique())
    elif 'RGIId' in glacier_gdf.columns:
        gl_ids = set(glacier_gdf['RGIId'].unique())
    else:
        gl_ids = set()

    # 统计匹配情况
    matched = fl_ids & gl_ids
    fl_only = fl_ids - gl_ids
    gl_only = gl_ids - fl_ids

    results.append(f"  流线中的冰川ID数: {len(fl_ids)}")
    results.append(f"  冰川边界中的ID数: {len(gl_ids)}")
    results.append(f"  匹配的ID数: {len(matched)}")

    if fl_only:
        results.append(f"  ⚠ 仅在流线中的ID: {len(fl_only)} 个")
        results.append(f"      示例: {list(fl_only)[:5]}")

    if gl_only:
        results.append(f"  ⚠ 仅在冰川边界中的ID (无流线): {len(gl_only)} 个")
        results.append(f"      示例: {list(gl_only)[:5]}")

    # 检查每个冰川的流线数
    fl_counts = flowlines_gdf[id_field].value_counts()
    results.append(f"  每个冰川的流线数:")
    results.append(f"    最小: {fl_counts.min()}, 最大: {fl_counts.max()}, 平均: {fl_counts.mean():.1f}")

    # 检查是否有支流关系 (NEAR_FID)
    if 'NEAR_FID' in flowlines_gdf.columns:
        has_near = flowlines_gdf['NEAR_FID'].notna().sum()
        results.append(f"  有NEAR_FID的点数: {has_near} (用于支流汇合)")

    success = len(matched) > 0 and len(fl_only) < len(fl_ids) * 0.1
    if success:
        results.insert(0, "✓ 流线拓扑检查: 基本正常")
    else:
        results.insert(0, "✗ 流线拓扑检查: 存在问题")

    return success, results


def check_dem_quality(dem_data, dem_transform, glacier_gdf, sample_size=5):
    """检查bare_dem质量"""
    results = []

    # 基本统计
    valid_data = dem_data[dem_data > -9999]
    if len(valid_data) == 0:
        return False, ["✗ DEM质量检查: 无有效数据"]

    results.append(f"  DEM尺寸: {dem_data.shape}")
    results.append(f"  有效像元: {len(valid_data)} / {dem_data.size} ({100 * len(valid_data) / dem_data.size:.1f}%)")
    results.append(f"  高程范围: {np.min(valid_data):.1f} - {np.max(valid_data):.1f} m")
    results.append(f"  平均高程: {np.mean(valid_data):.1f} m")
    results.append(f"  标准差: {np.std(valid_data):.1f} m")

    # 检查NoData
    nodata_count = np.sum(dem_data <= -9999)
    nodata_pct = 100 * nodata_count / dem_data.size
    results.append(f"  NoData像元: {nodata_count} ({nodata_pct:.2f}%)")

    # 检查异常值
    q01 = np.percentile(valid_data, 1)
    q99 = np.percentile(valid_data, 99)
    extreme_low = np.sum(valid_data < q01 - 500)
    extreme_high = np.sum(valid_data > q99 + 500)

    if extreme_low > 0 or extreme_high > 0:
        results.append(f"  ⚠ 潜在异常值: 极低={extreme_low}, 极高={extreme_high}")

    # 在几个冰川区域采样检查
    results.append(f"\n  抽样检查 ({sample_size} 个冰川区域):")

    sample_glaciers = glacier_gdf.sample(min(sample_size, len(glacier_gdf)))
    glacier_stats = []

    for idx, row in sample_glaciers.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        gid = row.get('OrigRGIId', row.get('RGIId', idx))

        # 获取边界框内的DEM
        bounds = geom.bounds
        col_start, row_end = ~dem_transform * (bounds[0], bounds[1])
        col_end, row_start = ~dem_transform * (bounds[2], bounds[3])

        row_start = max(0, int(row_start))
        row_end = min(dem_data.shape[0], int(row_end))
        col_start = max(0, int(col_start))
        col_end = min(dem_data.shape[1], int(col_end))

        if row_start >= row_end or col_start >= col_end:
            continue

        subset = dem_data[row_start:row_end, col_start:col_end]
        valid_subset = subset[subset > -9999]

        if len(valid_subset) > 0:
            glacier_stats.append({
                'id': gid,
                'min': np.min(valid_subset),
                'max': np.max(valid_subset),
                'mean': np.mean(valid_subset),
                'nodata_pct': 100 * np.sum(subset <= -9999) / subset.size
            })
            results.append(f"    {gid}: {np.min(valid_subset):.0f}-{np.max(valid_subset):.0f}m, "
                           f"NoData={100 * np.sum(subset <= -9999) / subset.size:.1f}%")

    # 检查是否有高NoData比例的区域
    high_nodata = [s for s in glacier_stats if s['nodata_pct'] > 20]
    if high_nodata:
        results.append(f"  ⚠ 高NoData比例的冰川区域: {len(high_nodata)} 个")

    success = nodata_pct < 50 and len(high_nodata) < sample_size * 0.3
    if success:
        results.insert(0, "✓ DEM质量检查: 基本正常")
    else:
        results.insert(0, "⚠ DEM质量检查: 存在潜在问题")

    return success, results


def visualize_overview(glacier_gdf, flowlines_gdf, dem_data, dem_transform, dem_bounds, output_path):
    """生成数据概览图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. 空间范围图
    ax1 = axes[0]

    # DEM范围
    dem_box = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    gpd.GeoSeries([dem_box]).plot(ax=ax1, facecolor='lightblue', edgecolor='blue',
                                  alpha=0.3, linewidth=2, label='DEM范围')

    # 冰川边界
    glacier_gdf.plot(ax=ax1, facecolor='white', edgecolor='red', alpha=0.5, linewidth=0.5)

    # 流线
    flowlines_gdf.plot(ax=ax1, color='green', linewidth=0.3, alpha=0.7)

    ax1.set_title('空间覆盖范围')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')

    # 添加图例
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='blue', alpha=0.3, label='DEM范围'),
        Patch(facecolor='white', edgecolor='red', alpha=0.5, label='冰川边界'),
        Patch(facecolor='green', alpha=0.7, label='流线')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')

    # 2. DEM高程分布
    ax2 = axes[1]
    valid_dem = dem_data[dem_data > -9999].flatten()
    ax2.hist(valid_dem, bins=100, color='steelblue', edgecolor='none', alpha=0.7)
    ax2.axvline(np.mean(valid_dem), color='red', linestyle='--', label=f'均值: {np.mean(valid_dem):.0f}m')
    ax2.set_xlabel('高程 (m)')
    ax2.set_ylabel('像元数')
    ax2.set_title('DEM高程分布')
    ax2.legend()

    # 3. 冰川面积分布
    ax3 = axes[2]
    areas = glacier_gdf.geometry.area / 1e6  # km²
    ax3.hist(areas, bins=50, color='coral', edgecolor='none', alpha=0.7)
    ax3.axvline(areas.mean(), color='red', linestyle='--', label=f'均值: {areas.mean():.1f} km²')
    ax3.axvline(areas.median(), color='blue', linestyle='--', label=f'中位数: {areas.median():.1f} km²')
    ax3.set_xlabel('面积 (km²)')
    ax3.set_ylabel('冰川数')
    ax3.set_title(f'冰川面积分布 (n={len(areas)})')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def visualize_sample_glaciers(glacier_gdf, flowlines_gdf, dem_data, dem_transform,
                              sample_ids, output_path):
    """可视化抽样冰川的DEM和流线"""
    n = len(sample_ids)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 10))

    if n == 1:
        axes = axes.reshape(2, 1)

    for i, gid in enumerate(sample_ids):
        # 获取冰川边界
        if 'OrigRGIId' in glacier_gdf.columns:
            gl = glacier_gdf[glacier_gdf['OrigRGIId'] == gid]
        else:
            gl = glacier_gdf[glacier_gdf.index == gid]

        # 获取流线
        if 'MainRGI' in flowlines_gdf.columns:
            fl = flowlines_gdf[flowlines_gdf['MainRGI'] == gid]
        elif 'GlacierID' in flowlines_gdf.columns:
            fl = flowlines_gdf[flowlines_gdf['GlacierID'] == gid]
        else:
            fl = flowlines_gdf.iloc[:0]

        if len(gl) == 0:
            continue

        geom = gl.geometry.values[0]
        bounds = geom.bounds

        # 扩展边界
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        bounds = (bounds[0] - buffer, bounds[1] - buffer,
                  bounds[2] + buffer, bounds[3] + buffer)

        # 提取DEM子集
        col_start, row_end = ~dem_transform * (bounds[0], bounds[1])
        col_end, row_start = ~dem_transform * (bounds[2], bounds[3])

        row_start = max(0, int(row_start))
        row_end = min(dem_data.shape[0], int(row_end))
        col_start = max(0, int(col_start))
        col_end = min(dem_data.shape[1], int(col_end))

        subset = dem_data[row_start:row_end, col_start:col_end].copy()
        subset[subset <= -9999] = np.nan

        # 计算子集的地理范围
        x_min = dem_transform.c + col_start * dem_transform.a
        y_max = dem_transform.f + row_start * dem_transform.e
        x_max = dem_transform.c + col_end * dem_transform.a
        y_min = dem_transform.f + row_end * dem_transform.e

        extent = [x_min, x_max, y_min, y_max]

        # 上图: DEM + 边界
        ax1 = axes[0, i]
        im = ax1.imshow(subset, extent=extent, cmap='terrain', origin='upper')
        gl.boundary.plot(ax=ax1, color='red', linewidth=2)
        if len(fl) > 0:
            fl.plot(ax=ax1, color='blue', linewidth=1.5)
        ax1.set_title(f'{gid}\nDEM + 边界 + 流线')
        ax1.set_xlim(bounds[0], bounds[2])
        ax1.set_ylim(bounds[1], bounds[3])
        plt.colorbar(im, ax=ax1, label='高程 (m)', shrink=0.8)

        # 下图: 坡度图 (检查伪影)
        ax2 = axes[1, i]
        if subset.shape[0] > 2 and subset.shape[1] > 2:
            # 计算坡度
            dy, dx = np.gradient(subset)
            slope = np.sqrt(dx ** 2 + dy ** 2)

            im2 = ax2.imshow(slope, extent=extent, cmap='hot', origin='upper',
                             vmin=0, vmax=np.nanpercentile(slope, 95))
            gl.boundary.plot(ax=ax2, color='cyan', linewidth=2)
            ax2.set_title(f'{gid}\n坡度图 (检查伪影)')
            ax2.set_xlim(bounds[0], bounds[2])
            ax2.set_ylim(bounds[1], bounds[3])
            plt.colorbar(im2, ax=ax2, label='坡度', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# ============================================================================
# 主诊断函数
# ============================================================================

def run_diagnostic():
    """运行完整诊断"""
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    report = []
    report.append("=" * 70)
    report.append("Step 0 数据检查诊断报告")
    report.append("=" * 70)

    all_passed = True

    # ========== 1. 文件存在性检查 ==========
    print("\n[1/7] 检查文件存在性...")
    report.append("\n## 1. 文件存在性检查")

    files_ok = True
    for key, path in [('glacier_path', '冰川边界'),
                      ('flowlines_path', '流线'),
                      ('dem_path', 'DEM')]:
        exists, msg = check_file_exists(CONFIG[key], path)
        report.append(msg)
        print(f"  {msg}")
        if not exists:
            files_ok = False

    if not files_ok:
        report.append("\n✗ 文件检查失败，无法继续")
        print("\n✗ 文件检查失败，无法继续")
        save_report(report)
        return

    # 检查shapefile完整性
    for key, name in [('glacier_path', '冰川边界'), ('flowlines_path', '流线')]:
        ok, msg = check_shapefile_complete(CONFIG[key], name)
        report.append(msg)
        print(f"  {msg}")

    # ========== 2. 加载数据 ==========
    print("\n[2/7] 加载数据...")
    report.append("\n## 2. 数据加载")

    try:
        glacier_gdf = gpd.read_file(CONFIG['glacier_path'])
        report.append(f"✓ 冰川边界: {len(glacier_gdf)} 个")
        print(f"  ✓ 冰川边界: {len(glacier_gdf)} 个")
    except Exception as e:
        report.append(f"✗ 加载冰川边界失败: {e}")
        print(f"  ✗ 加载冰川边界失败: {e}")
        save_report(report)
        return

    try:
        flowlines_gdf = gpd.read_file(CONFIG['flowlines_path'])
        report.append(f"✓ 流线: {len(flowlines_gdf)} 条")
        print(f"  ✓ 流线: {len(flowlines_gdf)} 条")
    except Exception as e:
        report.append(f"✗ 加载流线失败: {e}")
        print(f"  ✗ 加载流线失败: {e}")
        save_report(report)
        return

    try:
        with rasterio.open(CONFIG['dem_path']) as src:
            dem_data = src.read(1)
            dem_transform = src.transform
            dem_crs = src.crs
            dem_bounds = src.bounds
        report.append(f"✓ DEM: {dem_data.shape}, 分辨率 {abs(dem_transform.a):.1f}m")
        print(f"  ✓ DEM: {dem_data.shape}, 分辨率 {abs(dem_transform.a):.1f}m")
    except Exception as e:
        report.append(f"✗ 加载DEM失败: {e}")
        print(f"  ✗ 加载DEM失败: {e}")
        save_report(report)
        return

    # ========== 3. CRS一致性检查 ==========
    print("\n[3/7] 检查CRS一致性...")
    report.append("\n## 3. CRS一致性检查")

    crs_ok, crs_results = check_crs_consistency(glacier_gdf, flowlines_gdf, dem_crs)
    for r in crs_results:
        report.append(r)
        print(f"  {r}")
    if not crs_ok:
        all_passed = False

    # ========== 4. 字段完整性检查 ==========
    print("\n[4/7] 检查字段完整性...")
    report.append("\n## 4. 字段完整性检查")

    # 冰川边界必需字段
    glacier_required = ['OrigRGIId']  # 或 'RGIId'
    glacier_alternative = ['RGIId']

    ok, msg = check_required_fields(glacier_gdf, glacier_required, '冰川边界')
    if not ok:
        ok, msg = check_required_fields(glacier_gdf, glacier_alternative, '冰川边界(备选)')
    report.append(msg)
    print(f"  {msg}")

    # 流线必需字段
    flowline_required = ['MainRGI']  # 或 'GlacierID'
    flowline_alternative = ['GlacierID']

    ok, msg = check_required_fields(flowlines_gdf, flowline_required, '流线')
    if not ok:
        ok, msg = check_required_fields(flowlines_gdf, flowline_alternative, '流线(备选)')
    report.append(msg)
    print(f"  {msg}")

    # 列出所有字段
    report.append(f"\n  冰川边界所有字段: {list(glacier_gdf.columns)}")
    report.append(f"  流线所有字段: {list(flowlines_gdf.columns)}")

    # ========== 5. 空间覆盖检查 ==========
    print("\n[5/7] 检查空间覆盖范围...")
    report.append("\n## 5. 空间覆盖检查")

    coverage_ok, coverage_results = check_spatial_coverage(glacier_gdf, dem_bounds)
    for r in coverage_results:
        report.append(r)
        print(f"  {r}")
    if not coverage_ok:
        all_passed = False

    # ========== 6. 流线拓扑检查 ==========
    print("\n[6/7] 检查流线拓扑关系...")
    report.append("\n## 6. 流线拓扑检查")

    topo_ok, topo_results = check_flowline_topology(flowlines_gdf, glacier_gdf)
    for r in topo_results:
        report.append(r)
        print(f"  {r}")
    if not topo_ok:
        all_passed = False

    # ========== 7. DEM质量检查 ==========
    print("\n[7/7] 检查DEM质量...")
    report.append("\n## 7. DEM质量检查")

    dem_ok, dem_results = check_dem_quality(dem_data, dem_transform, glacier_gdf)
    for r in dem_results:
        report.append(r)
        print(f"  {r}")
    if not dem_ok:
        all_passed = False

    # ========== 生成可视化 ==========
    print("\n生成可视化图表...")

    # 概览图
    overview_path = CONFIG['output_dir'] + 'step0_overview.png'
    try:
        visualize_overview(glacier_gdf, flowlines_gdf, dem_data, dem_transform,
                           dem_bounds, overview_path)
        report.append(f"\n✓ 概览图已保存: {overview_path}")
        print(f"  ✓ 概览图已保存: {overview_path}")
    except Exception as e:
        report.append(f"\n⚠ 生成概览图失败: {e}")
        print(f"  ⚠ 生成概览图失败: {e}")

    # 抽样冰川详图
    # 选择2个冰川
    if 'OrigRGIId' in glacier_gdf.columns:
        sample_ids = glacier_gdf['OrigRGIId'].sample(min(2, len(glacier_gdf))).tolist()
    elif 'RGIId' in glacier_gdf.columns:
        sample_ids = glacier_gdf['RGIId'].sample(min(2, len(glacier_gdf))).tolist()
    else:
        sample_ids = glacier_gdf.index[:2].tolist()

    sample_path = CONFIG['output_dir'] + 'step0_sample_glaciers.png'
    try:
        visualize_sample_glaciers(glacier_gdf, flowlines_gdf, dem_data, dem_transform,
                                  sample_ids, sample_path)
        report.append(f"✓ 抽样冰川图已保存: {sample_path}")
        report.append(f"  抽样冰川ID: {sample_ids}")
        print(f"  ✓ 抽样冰川图已保存: {sample_path}")
    except Exception as e:
        report.append(f"⚠ 生成抽样冰川图失败: {e}")
        print(f"  ⚠ 生成抽样冰川图失败: {e}")

    # ========== 总结 ==========
    report.append("\n" + "=" * 70)
    report.append("## 总结")

    if all_passed:
        report.append("✓ 所有检查通过，可以进入 Step 1")
        print("\n" + "=" * 70)
        print("✓ 所有检查通过，可以进入 Step 1")
    else:
        report.append("⚠ 部分检查未通过，请查看上述详情")
        print("\n" + "=" * 70)
        print("⚠ 部分检查未通过，请查看上述详情")

    report.append("=" * 70)

    # 保存报告
    save_report(report)

    print("\n" + "=" * 70)
    print("诊断完成! 请将以下文件发给我:")
    print(f"  1. {CONFIG['output_dir']}step0_diagnostic_report.txt")
    print(f"  2. {CONFIG['output_dir']}step0_overview.png")
    print(f"  3. {CONFIG['output_dir']}step0_sample_glaciers.png")
    print("=" * 70)


def save_report(report):
    """保存报告"""
    report_text = "\n".join(report)
    report_path = CONFIG['output_dir'] + 'step0_diagnostic_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    run_diagnostic()