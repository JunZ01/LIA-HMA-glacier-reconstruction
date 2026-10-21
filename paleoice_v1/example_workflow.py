"""
example_workflow.py - PaleoIce使用示例 (优化版)
================================================

优化功能:
1. 进度条显示
2. 断点续传支持
3. 详细错误日志
4. 时间估计
5. 内存优化
"""

import sys
import warnings
from pathlib import Path
import logging

# 抑制所有警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# 路径配置 (请根据实际情况修改)
# =============================================================================

# LIA边界文件
LIA_PATH = r"/mnt/e/2Data/022_工作2/出图数据/Himalaya_LIA_with_RGIID_Complete/Himalaya_LIA_with_RGIID_Complete.shp"

# DEM文件
DEM_PATH = r"/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif"

# 工作目录 (OGGM输出)
WORKING_DIR = r"/mnt/d/paleoice_working"

# 输出目录
OUTPUT_DIR = r"/mnt/d/paleoice_output"

# 并行进程数 (根据系统内存调整，建议: 内存GB数/4)
N_WORKERS = 12


# =============================================================================
# Step 1: 数据准备
# =============================================================================

def step1_prepare(force_rerun: bool = False):
    """准备LIA数据"""
    from paleoice.prepare import validate_inputs, prepare_lia_for_oggm

    print("\n" + "="*60)
    print("Step 1: 数据准备")
    print("="*60)

    output_path = Path(OUTPUT_DIR) / "lia_for_oggm.gpkg"

    # 检查是否已存在
    if output_path.exists() and not force_rerun:
        print(f"\n输出文件已存在: {output_path}")
        print("使用 --force 重新运行")
        return output_path

    # 验证输入
    print("\n[1.1] 验证输入数据...")
    result = validate_inputs(LIA_PATH, DEM_PATH)

    if not result['valid']:
        print("错误:", result['errors'])
        return None

    if result['warnings']:
        print("警告:")
        for w in result['warnings']:
            print(f"  - {w}")

    # 准备OGGM格式
    print("\n[1.2] 转换LIA边界为OGGM格式...")

    lia_gdf = prepare_lia_for_oggm(
        lia_path=LIA_PATH,
        dem_path=DEM_PATH,
        output_path=output_path,
        id_field='UniqueID',
        rgi_field='MainRGI',
        area_field='LIAArea',
        min_area_km2=0.01,
        fix_geometry=True,
        show_progress=True
    )

    print(f"\n准备完成: {len(lia_gdf)} 个冰川")
    print(f"输出文件: {output_path}")

    return output_path


# =============================================================================
# Step 2: OGGM Flowline生成
# =============================================================================

def step2_flowlines(outline_path, reset: bool = False, n_workers: int = 24, batch_size: int = None):
    """生成Flowlines"""
    from paleoice.flowline import FlowlineProcessor, batch_extract_flowlines

    print("\n" + "="*60)
    print("Step 2: Flowline生成")
    print("="*60)

    # 创建处理器
    processor = FlowlineProcessor(
        working_dir=WORKING_DIR,
        dem_path=DEM_PATH,
        n_workers=n_workers
    )

    # 运行
    print(f"\n[2.1] 运行OGGM (workers={n_workers}, batch={batch_size})...")
    gdirs = processor.run(outline_path, reset=reset, batch_size=batch_size)

    # 提取flowline数据
    print("\n[2.2] 提取Flowline数据...")
    flowlines_gdf = batch_extract_flowlines(
        gdirs=gdirs,
        output_dir=OUTPUT_DIR,
        show_progress=True,
        working_dir=WORKING_DIR  # 添加working_dir参数用于从磁盘恢复
    )

    print(f"\n完成: {len(flowlines_gdf)} 条flowlines")

    return gdirs, flowlines_gdf


# =============================================================================
# 测试模式
# =============================================================================

def run_test_subset(n_glaciers: int = 50, reset: bool = False, n_workers: int = 24):
    """用小样本测试"""
    from paleoice.prepare import prepare_lia_for_oggm
    import geopandas as gpd

    print("\n" + "="*60)
    print(f"测试模式: 使用 {n_glaciers} 个冰川")
    print("="*60)

    # 创建输出目录
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # 读取并采样
    print("\n[1] 读取并采样LIA数据...")
    lia_full = gpd.read_file(LIA_PATH)

    if 'LIAArea' in lia_full.columns:
        lia_sorted = lia_full.sort_values('LIAArea', ascending=False)
        start_idx = int(len(lia_sorted) * 0.1)
        lia_sample = lia_sorted.iloc[start_idx:start_idx + n_glaciers].copy()
    else:
        lia_sample = lia_full.sample(n=min(n_glaciers, len(lia_full)), random_state=42)

    sample_path = Path(OUTPUT_DIR) / "lia_sample.gpkg"
    lia_sample.to_file(sample_path, driver='GPKG')
    print(f"  样本保存至: {sample_path}")
    print(f"  样本数量: {len(lia_sample)}")

    # 准备OGGM格式
    print("\n[2] 转换为OGGM格式...")
    oggm_path = Path(OUTPUT_DIR) / "lia_sample_oggm.gpkg"

    lia_oggm = prepare_lia_for_oggm(
        lia_path=sample_path,
        dem_path=DEM_PATH,
        output_path=oggm_path,
        id_field='UniqueID',
        rgi_field='MainRGI',
        area_field='LIAArea' if 'LIAArea' in lia_sample.columns else None,
        min_area_km2=0.01,
        show_progress=True
    )

    # 生成Flowlines
    print("\n[3] 生成Flowlines...")
    gdirs, flowlines = step2_flowlines(oggm_path, reset=reset, n_workers=n_workers)

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print(f"  处理冰川: {len(lia_oggm)}")
    print(f"  成功flowlines: {len(flowlines)}")
    print(f"  结果目录: {OUTPUT_DIR}")

    return gdirs, flowlines


# =============================================================================
# 主函数
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='PaleoIce Workflow - LIA冰川重建',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试模式 (50个冰川)
  python example_workflow.py --test 50
  
  # 完整运行
  python example_workflow.py
  
  # 重新运行Step 1
  python example_workflow.py --step 1 --force
  
  # 断点续传Step 2
  python example_workflow.py --step 2
  
  # 重置并重新运行Step 2
  python example_workflow.py --step 2 --reset
        """
    )

    parser.add_argument('--test', type=int, default=0,
                       help='测试模式，指定冰川数量')
    parser.add_argument('--step', type=int, default=0,
                       help='运行指定步骤 (1=准备, 2=flowline, 0=全部)')
    parser.add_argument('--force', action='store_true',
                       help='强制重新运行 (覆盖已有结果)')
    parser.add_argument('--reset', action='store_true',
                       help='重置OGGM工作目录')
    parser.add_argument('--workers', type=int, default=N_WORKERS,
                        help=f'并行进程数 (默认: {N_WORKERS})')
    parser.add_argument('--batch', type=int, default=None,
                        help='分批处理大小，内存不足时使用 (如 --batch 200)')

    args = parser.parse_args()

    # 设置进程数
    n_workers = args.workers

    # 测试模式
    if args.test > 0:
        run_test_subset(n_glaciers=args.test, reset=args.reset, n_workers=n_workers)
        return

    # 完整流程
    outline_path = Path(OUTPUT_DIR) / "lia_for_oggm.gpkg"

    if args.step == 0 or args.step == 1:
        outline_path = step1_prepare(force_rerun=args.force)

    if args.step == 0 or args.step == 2:
        if outline_path and Path(outline_path).exists():
            step2_flowlines(outline_path, reset=args.reset, n_workers=n_workers, batch_size=args.batch)
        else:
            print("错误: 请先运行Step 1或提供边界文件")
            print(f"  期望文件: {outline_path}")


if __name__ == "__main__":
    main()