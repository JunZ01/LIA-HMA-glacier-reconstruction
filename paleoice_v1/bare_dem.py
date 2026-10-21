"""
bare_dem_local.py - 使用本地Hugonnet物质平衡数据版本
从RGI冰川反演生成床面DEM (bare_dem = surface_dem - thickness)

完全本地运行，不依赖OGGM服务器下载

作者: Jun (Yunnan University)
日期: 2026-01
"""
import os
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from pathlib import Path
from tqdm import tqdm
os.environ['http_proxy'] = 'http://192.168.160.1:7890'
os.environ['https_proxy'] = 'http://192.168.160.1:7890'

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

from oggm import cfg, workflow, tasks, utils


# ============================================================
# 配置区域
# ============================================================

RGI_PATH = "/mnt/e/2Data/022_工作2/出图数据/himalaya_glaciers_complete/himalaya_glaciers_complete.shp"
DEM_PATH = "/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif"
WORKING_DIR = "/mnt/d/paleoice_working_rgi"
OUTPUT_PATH = "/mnt/d/paleoice_output/bare_dem.tif"

# Hugonnet 物质平衡数据
MB_PATHS = [
    "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_14/dh_14_rgi60_pergla_rates.csv",
    "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_15/dh_15_rgi60_pergla_rates.csv",
]
MB_PERIOD = "2000-01-01_2020-01-01"  # 20年时段

N_WORKERS = 12
BATCH_SIZE = 100


def load_mb_data(mb_paths, period):
    """
    加载并合并Hugonnet物质平衡数据
    """
    print(f"加载物质平衡数据，时段: {period}")

    dfs = []
    for path in mb_paths:
        wsl_path = path
        print(f"  读取: {wsl_path}")
        df = pd.read_csv(wsl_path)
        dfs.append(df)

    mb_all = pd.concat(dfs, ignore_index=True)
    print(f"  合并后总记录: {len(mb_all)}")

    # 筛选时段
    mb_period = mb_all[mb_all['period'] == period].copy()
    print(f"  时段 {period} 记录数: {len(mb_period)}")

    # 转换单位: dmdtda (m w.e. yr-1) -> kg m-2 yr-1 (乘1000)
    mb_period['dmdtda_kg'] = mb_period['dmdtda'] * 1000

    # 设置索引
    mb_period = mb_period.set_index('rgiid')

    return mb_period


def clean_geometries(gdf):
    """清理几何数据"""
    print(f"原始冰川数: {len(gdf)}")

    gdf = gdf[gdf.geometry.notna() & ~gdf.is_empty].copy()
    gdf['geometry'] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid].copy()

    def all_coords_finite(geom):
        try:
            if geom.geom_type == 'Polygon':
                all_coords = list(geom.exterior.coords)
                for interior in geom.interiors:
                    all_coords.extend(list(interior.coords))
            elif geom.geom_type == 'MultiPolygon':
                all_coords = []
                for poly in geom.geoms:
                    all_coords.extend(list(poly.exterior.coords))
                    for interior in poly.interiors:
                        all_coords.extend(list(interior.coords))
            else:
                all_coords = list(geom.coords)

            for coord in all_coords:
                for val in coord:
                    if not np.isfinite(val):
                        return False
            return True
        except:
            return False

    gdf = gdf[gdf.geometry.apply(all_coords_finite)].copy()
    print(f"清理后冰川数: {len(gdf)}")
    return gdf


def calibrate_mb_batch(gdirs, mb_data):
    """
    对一批冰川执行物质平衡校准
    """
    success = 0
    failed = []

    for gdir in gdirs:
        try:
            rgi_id = gdir.rgi_id
            if rgi_id not in mb_data.index:
                failed.append((rgi_id, "无MB数据"))
                continue

            ref_mb = mb_data.loc[rgi_id, 'dmdtda_kg']

            workflow.execute_entity_task(
                tasks.mb_calibration_from_scalar_mb,
                [gdir],
                ref_mb=ref_mb,
                ref_mb_years=(2000, 2020),
                overwrite_gdir=True,
                calibrate_param1='melt_f',
                calibrate_param2='temp_bias',
            )
            success += 1

        except Exception as e:
            failed.append((gdir.rgi_id, str(e)[:50]))

    return success, failed


def extract_thickness_batch(gdirs, thickness_total, dem_crs, dem_transform, dem_height, dem_width):
    """提取thickness并累积"""
    success = 0

    for gdir in gdirs:
        try:
            gridded_path = gdir.get_filepath('gridded_data')
            with utils.ncDataset(gridded_path) as nc:
                if 'distributed_thickness' not in nc.variables:
                    continue

                thick = nc.variables['distributed_thickness'][:]
                x = nc.variables['x'][:]
                y = nc.variables['y'][:]

            dx = x[1] - x[0]
            dy = y[0] - y[1]
            local_transform = from_bounds(
                x.min() - dx/2, y.min() - dy/2,
                x.max() + dx/2, y.max() + dy/2,
                len(x), len(y)
            )

            thick_reproj = np.zeros((dem_height, dem_width), dtype=np.float32)
            reproject(
                source=thick.astype(np.float32),
                destination=thick_reproj,
                src_transform=local_transform,
                src_crs=gdir.grid.proj,
                dst_transform=dem_transform,
                dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )

            np.maximum(thickness_total, thick_reproj, out=thickness_total)
            success += 1

        except Exception:
            continue

    return success


def generate_bare_dem(batch_size=BATCH_SIZE, n_workers=N_WORKERS):
    """
    主流程：本地数据 → OGGM反演 → bare_dem
    """
    working_dir = Path(WORKING_DIR)
    working_dir.mkdir(parents=True, exist_ok=True)

    # 1. 配置OGGM
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = str(working_dir)
    cfg.PATHS['dem_file'] = DEM_PATH
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['mp_processes'] = n_workers
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['border'] = 80
    # cfg.PARAMS['use_winter_prcp_factor'] = False
    # cfg.PARAMS['prcp_fac'] = 3.5

    # 2. 加载物质平衡数据
    mb_data = load_mb_data(MB_PATHS, MB_PERIOD)

    # 3. 读取RGI
    print(f"\n读取RGI: {RGI_PATH}")
    rgi = gpd.read_file(RGI_PATH)
    if rgi.crs and rgi.crs.to_epsg() != 4326:
        rgi = rgi.to_crs("EPSG:4326")
    rgi = clean_geometries(rgi)

    # 只保留有MB数据的冰川
    rgi_with_mb = rgi[rgi['RGIId'].isin(mb_data.index)].copy()
    print(f"有MB数据的冰川: {len(rgi_with_mb)}/{len(rgi)}")
    rgi = rgi_with_mb.reset_index(drop=True)

    # 4. 读取DEM模板
    print(f"\n读取DEM: {DEM_PATH}")
    with rasterio.open(DEM_PATH) as src:
        dem = src.read(1).astype(np.float32)
        dem_profile = src.profile.copy()
        dem_crs = src.crs
        dem_transform = src.transform
        dem_height, dem_width = dem.shape
    print(f"DEM尺寸: {dem_width} x {dem_height}")

    thickness_total = np.zeros_like(dem)

    # 5. 定义任务
    tasks_before_mb = [
        tasks.define_glacier_region,
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.catchment_area,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.process_climate_data,
    ]

    tasks_after_mb = [
        tasks.apparent_mb_from_any_mb,
        tasks.prepare_for_inversion,
        tasks.mass_conservation_inversion,
        tasks.filter_inversion_output,
        tasks.distribute_thickness_per_altitude,
    ]

    # 6. 分批处理
    n_glaciers = len(rgi)
    n_batches = (n_glaciers + batch_size - 1) // batch_size
    print(f"\n总计 {n_glaciers} 个冰川，分 {n_batches} 批处理")

    total_success = 0
    total_mb_failed = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_glaciers)
        rgi_batch = rgi.iloc[start_idx:end_idx].copy()

        print(f"\n{'='*60}")
        print(f"批次 {batch_idx + 1}/{n_batches}: 冰川 {start_idx}-{end_idx-1} (共 {len(rgi_batch)} 个)")
        print(f"{'='*60}")

        gdirs = None
        try:
            # 初始化
            print("  初始化glacier directories...")
            gdirs = workflow.init_glacier_directories(rgi_batch, reset=False)

            # MB校准前的任务
            for task in tasks_before_mb:
                print(f"  执行: {task.__name__}")
                workflow.execute_entity_task(task, gdirs)

            # MB校准
            print("  执行: mb_calibration_from_scalar_mb")
            mb_success, mb_failed = calibrate_mb_batch(gdirs, mb_data)
            print(f"    MB校准成功: {mb_success}/{len(gdirs)}")
            if mb_failed:
                total_mb_failed += len(mb_failed)

            # MB校准后的任务
            for task in tasks_after_mb:
                print(f"  执行: {task.__name__}")
                workflow.execute_entity_task(task, gdirs)

            # 提取thickness
            batch_success = extract_thickness_batch(
                gdirs, thickness_total, dem_crs, dem_transform, dem_height, dem_width
            )
            total_success += batch_success
            print(f"  提取thickness成功: {batch_success}/{len(gdirs)}")

        except Exception as e:
            print(f"  批次出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            if gdirs is not None:
                del gdirs
            del rgi_batch
            gc.collect()

    # 7. 保存结果
    print(f"\n{'='*60}")
    print(f"处理完成: 成功 {total_success}/{n_glaciers}")
    print(f"{'='*60}")

    bare_dem = dem - thickness_total

    # 统计
    thick_mask = thickness_total > 0
    print(f"\nthickness统计:")
    print(f"  有效像素: {thick_mask.sum()}")
    if thick_mask.sum() > 0:
        print(f"  范围: {thickness_total[thick_mask].min():.1f} - {thickness_total[thick_mask].max():.1f} m")
        print(f"  均值: {thickness_total[thick_mask].mean():.1f} m")

    # 保存
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    dem_profile.update(dtype=np.float32, compress='lzw')

    with rasterio.open(OUTPUT_PATH, 'w', **dem_profile) as dst:
        dst.write(bare_dem, 1)
    print(f"\nbare_dem已保存: {OUTPUT_PATH}")

    thickness_path = OUTPUT_PATH.replace('.tif', '_thickness.tif')
    with rasterio.open(thickness_path, 'w', **dem_profile) as dst:
        dst.write(thickness_total, 1)
    print(f"thickness已保存: {thickness_path}")

    return OUTPUT_PATH


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成bare_dem（本地Hugonnet数据版）')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--workers', type=int, default=N_WORKERS)
    args = parser.parse_args()

    print("="*60)
    print("bare_dem生成工具 (本地Hugonnet数据版)")
    print("="*60)

    generate_bare_dem(batch_size=args.batch_size, n_workers=args.workers)