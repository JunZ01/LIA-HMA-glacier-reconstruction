"""
LIA冰川流线提取 - 基于OGGM官方建议的智能混合版（修复ID映射 v3）

修复内容：
1. 使用行索引建立映射（cook_rgidf 保持行顺序）
2. 不依赖 RGIId，因为 cook_rgidf 会重新生成
3. 保留所有原始属性
"""
import numpy as np
import geopandas as gpd
from pathlib import Path
import shapely.geometry as shpg
from scipy.interpolate import RegularGridInterpolator
from datetime import datetime
import gc

from oggm import cfg, workflow, tasks, utils
from oggm.utils import line_interpol


def smooth_centerlines_no_truncation(gdir, dx=None):
    """对centerlines进行插值平滑，但不截断支流"""
    if dx is None:
        dx = cfg.PARAMS['flowline_dx']

    cls = gdir.read_pickle('centerlines')

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo_smoothed'][:]

    xy = (np.arange(0, gdir.grid.ny - 0.1, 1),
          np.arange(0, gdir.grid.nx - 0.1, 1))
    interpolator = RegularGridInterpolator(xy, topo.astype(np.float64))

    smoothed_lines = []

    for ic, cl in enumerate(cls):
        points = line_interpol(cl.line, dx)
        if len(points) < 2:
            continue

        smoothed_line = shpg.LineString(points)

        flows_to_idx = None
        if cl.flows_to is not None:
            for j, other_cl in enumerate(cls):
                if cl.flows_to is other_cl:
                    flows_to_idx = j
                    break

        smoothed_lines.append({
            'line': smoothed_line,
            'is_main': cl.flows_to is None,
            'order': cl.order,
            'flows_to_idx': flows_to_idx,
            'original_cl': cl
        })

    return smoothed_lines


def print_progress(message, level=0):
    """带时间戳的进度输出"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    indent = "  " * level
    print(f"[{timestamp}] {indent}{message}", flush=True)


def extract_lia_flowlines_smart(
    lia_shp_path: str,
    dem_path: str,
    working_dir: str,
    output_path: str,
    min_area_km2: float = 10.0,
    batch_size: int = 20,
    n_processes: int = 20,
    reset: bool = False,
    flowline_dx: float = 2,
    resume: bool = True,
):
    """
    智能混合模式：遵循OGGM官方设计
    """

    print_progress("=" * 60)
    print_progress("LIA冰川流线提取 - OGGM官方推荐模式 (v3)")
    print_progress("=" * 60)

    # ========== 1. 初始化OGGM ==========
    print_progress("步骤 1: 初始化OGGM配置")
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = working_dir
    cfg.PATHS['dem_file'] = dem_path

    cfg.PARAMS['border'] = 80
    cfg.PARAMS['use_multiprocessing'] = True
    cfg.PARAMS['mp_processes'] = n_processes
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_rgi_area'] = False
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_multiple_flowlines'] = True
    cfg.PARAMS['flowline_dx'] = flowline_dx

    print_progress(f"批大小: {batch_size} (小批次策略)", 1)
    if n_processes == -1:
        import multiprocessing
        actual_procs = multiprocessing.cpu_count()
        print_progress(f"并行进程: 全部CPU ({actual_procs} 核)", 1)
    else:
        print_progress(f"并行进程: {n_processes} 核", 1)
    print_progress("✓ 配置完成\n")

    # ========== 2. 加载LIA清单 ==========
    print_progress("步骤 2: 加载LIA冰川清单")
    lia_gdf_original = gpd.read_file(lia_shp_path)
    print_progress(f"原始记录数: {len(lia_gdf_original)}", 1)
    print_progress(f"原始列: {list(lia_gdf_original.columns)}", 1)

    # 面积过滤
    if 'LIAArea' in lia_gdf_original.columns:
        lia_gdf = lia_gdf_original[lia_gdf_original['LIAArea'] >= min_area_km2].copy()
    elif 'Area' in lia_gdf_original.columns:
        lia_gdf = lia_gdf_original[lia_gdf_original['Area'] >= min_area_km2].copy()
    else:
        lia_gdf = lia_gdf_original.copy()

    # 重置索引，确保连续
    lia_gdf = lia_gdf.reset_index(drop=True)

    print_progress(f"面积 >= {min_area_km2} km²: {len(lia_gdf)}", 1)

    if len(lia_gdf) == 0:
        print_progress("错误: 没有符合条件的冰川", 1)
        return None

    # ========== 关键：保存原始属性（按行索引） ==========
    # cook_rgidf 会保持行顺序，所以用行索引作为 key
    attr_columns = [col for col in lia_gdf.columns if col != 'geometry']
    print_progress(f"将保留的属性列: {attr_columns}", 1)

    # 保存原始数据副本（用于后续映射）
    original_attrs_list = []
    for idx in range(len(lia_gdf)):
        row = lia_gdf.iloc[idx]
        attrs = {col: row[col] for col in attr_columns}
        original_attrs_list.append(attrs)

    print_progress(f"已保存 {len(original_attrs_list)} 条原始属性", 1)

    # 验证
    print_progress(f"示例 - 第0行 UniqueID: {original_attrs_list[0].get('UniqueID', 'N/A')}", 1)
    print_progress(f"示例 - 第1行 UniqueID: {original_attrs_list[1].get('UniqueID', 'N/A')}", 1)

    # 准备 OGGM 需要的列
    lia_gdf['RGIId'] = [f"RGI60-15.{i:05d}" for i in range(len(lia_gdf))]  # 临时ID，会被cook覆盖
    if 'Area' not in lia_gdf.columns:
        if 'LIAArea' in lia_gdf.columns:
            lia_gdf['Area'] = lia_gdf['LIAArea']

    # cook_rgidf - 会重新生成 RGIId 但保持行顺序
    rgidf = utils.cook_rgidf(lia_gdf, o1_region='15')

    # ★★★ 关键：建立 cook 后的 RGIId -> 原始行索引的映射 ★★★
    # cook_rgidf 保持行顺序，所以第 i 行对应 original_attrs_list[i]
    rgiid_to_idx = {row['RGIId']: i for i, row in rgidf.reset_index(drop=True).iterrows()}

    print_progress(f"cook 后 RGIId 示例: {list(rgiid_to_idx.keys())[:5]}", 1)
    print_progress("✓ 清单加载完成\n")

    # ========== 3. 检查已处理 ==========
    processed_rgiids = set()
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_output = output_dir / "temp_results_v3.gpkg"

    all_records = []

    if resume and temp_output.exists():
        print_progress("步骤 3: 检测到临时结果")
        try:
            existing_gdf = gpd.read_file(temp_output)
            processed_rgiids = set(existing_gdf['rgi_id'].unique())
            print_progress(f"已处理: {len(processed_rgiids)} 个冰川", 1)

            # 加载已有记录
            for _, row in existing_gdf.iterrows():
                all_records.append(row.to_dict())
            print_progress(f"已加载 {len(all_records)} 条流线记录", 1)
        except Exception as e:
            print_progress(f"加载临时文件出错: {e}", 1)
            processed_rgiids = set()
    else:
        print_progress("步骤 3: 全新开始")

    print()

    # ========== 4. 批处理循环 ==========
    total_glaciers = len(rgidf)
    n_batches = (total_glaciers + batch_size - 1) // batch_size

    print_progress(f"步骤 4: 开始批处理 (共 {n_batches} 批)")
    print_progress(f"总冰川数: {total_glaciers}", 1)

    global_success = 0
    global_failed = 0

    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_glaciers)

        print_progress(f"\n{'='*50}")
        print_progress(f"批次 {batch_idx+1}/{n_batches}")
        print_progress(f"冰川范围: {start_idx+1}-{end_idx}/{total_glaciers}")
        print_progress(f"{'='*50}")

        batch_rgidf = rgidf.iloc[start_idx:end_idx].copy()

        # 过滤已处理
        if resume and len(processed_rgiids) > 0:
            batch_rgidf = batch_rgidf[~batch_rgidf['RGIId'].isin(processed_rgiids)]
            if len(batch_rgidf) == 0:
                print_progress("本批次已完成，跳过", 1)
                continue

        print_progress(f"本批待处理: {len(batch_rgidf)} 个", 1)

        try:
            # 初始化glacier directories
            print_progress("初始化glacier directories...", 1)
            gdirs = workflow.init_glacier_directories(batch_rgidf, reset=False)
            print_progress(f"✓ 初始化完成: {len(gdirs)} 个", 2)

            print_progress("定义冰川区域 (多进程)...", 1)
            workflow.execute_entity_task(tasks.define_glacier_region, gdirs, source='USER')
            print_progress("✓ 完成", 2)

            print_progress("生成掩膜 (多进程)...", 1)
            workflow.execute_entity_task(tasks.glacier_masks, gdirs)
            print_progress("✓ 完成", 2)

            print_progress("计算中心线 (多进程)...", 1)
            workflow.execute_entity_task(tasks.compute_centerlines, gdirs)
            print_progress("✓ 完成", 2)

            # 提取流线
            print_progress("提取并平滑流线...", 1)
            batch_records = []

            for gdir in gdirs:
                try:
                    # ★★★ 通过 rgi_id 找到原始行索引，获取所有属性 ★★★
                    original_idx = rgiid_to_idx.get(gdir.rgi_id)
                    if original_idx is not None and original_idx < len(original_attrs_list):
                        glacier_attrs = original_attrs_list[original_idx]
                    else:
                        glacier_attrs = {}
                        print_progress(f"警告: {gdir.rgi_id} 找不到原始属性", 3)

                    smoothed_cls = smooth_centerlines_no_truncation(gdir, dx=flowline_dx)

                    for i, cl_info in enumerate(smoothed_cls):
                        line = cl_info['line']
                        x, y = line.xy
                        coords_wgs84 = []
                        for xi, yi in zip(x, y):
                            lon, lat = gdir.grid.ij_to_crs(xi, yi, crs='EPSG:4326')
                            coords_wgs84.append((lon, lat))

                        if len(coords_wgs84) >= 2:
                            # 基础流线属性
                            record = {
                                'rgi_id': gdir.rgi_id,
                                'fl_id': i,
                                'is_main': cl_info['is_main'],
                                'order': cl_info['order'],
                                'flows_to': cl_info['flows_to_idx'],
                                'length_m': line.length * gdir.grid.dx,
                                'geometry': shpg.LineString(coords_wgs84),
                            }

                            # ★★★ 添加所有原始属性 ★★★
                            for attr_name, attr_value in glacier_attrs.items():
                                record[attr_name] = attr_value

                            batch_records.append(record)

                    global_success += 1
                    processed_rgiids.add(gdir.rgi_id)

                except Exception as e:
                    global_failed += 1
                    print_progress(f"✗ {gdir.rgi_id}: {str(e)[:50]}", 3)

            print_progress(f"✓ 本批完成: 成功 {len(batch_records)} 条流线", 2)

            # 验证属性
            if len(batch_records) > 0:
                sample = batch_records[0]
                print_progress(f"  验证 - UniqueID: {sample.get('UniqueID', 'N/A')}", 2)
                print_progress(f"  验证 - MainRGI: {sample.get('MainRGI', 'N/A')}", 2)

            # 合并并保存临时结果
            all_records.extend(batch_records)

            if len(all_records) > 0:
                temp_gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")
                temp_gdf.to_file(temp_output, driver='GPKG')
                print_progress(f"✓ 临时结果已保存 (累计 {len(all_records)} 条)", 2)

            # 清理内存
            del gdirs, batch_rgidf, batch_records
            gc.collect()

        except Exception as e:
            print_progress(f"✗ 批次失败: {e}", 1)
            import traceback
            print_progress(f"详细错误: {traceback.format_exc()[:200]}", 2)
            continue

    # ========== 5. 保存最终结果 ==========
    print_progress(f"\n{'='*60}")
    print_progress("步骤 5: 保存最终结果")

    if len(all_records) > 0:
        final_gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")

        # 保存为 GeoPackage
        output_gpkg = output_path.replace('.shp', '.gpkg')
        final_gdf.to_file(output_gpkg, driver='GPKG')

        # 保存为 Shapefile
        final_gdf.to_file(output_path)

        print_progress(f"✓ 总流线数: {len(all_records)}", 1)
        print_progress(f"输出文件: {output_path}", 1)
        print_progress(f"输出文件 (GPKG): {output_gpkg}", 1)
        print_progress(f"成功/失败: {global_success}/{global_failed}", 1)

        # 统计信息
        main_count = sum(1 for r in all_records if r.get('is_main'))
        trib_count = len(all_records) - main_count
        print_progress(f"主流线: {main_count}, 支流: {trib_count}", 1)

        # 检查属性
        print_progress(f"\n输出列: {list(final_gdf.columns)}", 1)

        if 'UniqueID' in final_gdf.columns:
            unique_ids = final_gdf['UniqueID'].dropna().unique()
            print_progress(f"唯一 UniqueID 数量: {len(unique_ids)}", 1)
            print_progress(f"UniqueID 示例: {sorted(unique_ids)[:10]}", 1)

        if 'MainRGI' in final_gdf.columns:
            print_progress(f"MainRGI 示例: {final_gdf['MainRGI'].head(5).tolist()}", 1)

        # 清理临时文件
        if temp_output.exists():
            temp_output.unlink()
            print_progress("✓ 临时文件已清理", 1)

        print_progress("\n" + "=" * 60)
        print_progress("✓ 程序执行完成")
        print_progress("=" * 60)

        return final_gdf
    else:
        print_progress("✗ 未提取到任何流线", 1)
        return None


if __name__ == "__main__":
    gdf = extract_lia_flowlines_smart(
        lia_shp_path="/mnt/e/2Data/022_工作2/出图数据/Himalaya_LIA_with_RGIID_Complete/Himalaya_LIA_with_RGIID_Complete.shp",
        dem_path="/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif",
        working_dir="/mnt/f/oggm_related/oggm_LIA_flowlines",
        output_path="/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_smooth.shp",
        min_area_km2=10.0,
        batch_size=50,
        n_processes=20,
        reset=False,
        flowline_dx=1,
        resume=False,  # 第一次运行设为 False
    )