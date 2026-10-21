"""
==================================================

功能: 使用OGGM冰厚反演从表面DEM生成基岩DEM
      bare_dem = surface_dem - ice_thickness

适用: LIA (小冰期) 大型冰川 (≥10 km²) 及其关联的现代RGI冰川

主要步骤:
    1. 数据筛选 - 筛选LIA≥10km²冰川及关联的RGI冰川
    2. 数据准备 - 清理几何、匹配物质平衡数据
    3. OGGM初始化 - 配置工作目录、DEM、参数
    4. 预处理任务 - DEM处理、中心线计算、流域宽度
    5. 物质平衡校准 - 使用Hugonnet数据校准
    6. 冰厚反演 - 质量守恒反演
    7. 生成基岩DEM - 从表面DEM减去冰厚

作者: Jun (Yunnan University)
日期: 2026-01-18
版本: 0.1.0
bare_dem_himalaya_refactored.py - 喜马拉雅地区基岩DEM生成模块 (鲁棒版)
==================================================================

改进功能:
1. GlacierTracker - 统一追踪每个冰川的处理状态
2. 预筛选 - 运行前排除无效冰川
3. 错误收集 - 每步记录失败原因，失败的冰川不再运行后续步骤
4. 最终报告 - 输出完整的处理统计和错误汇总

作者: Jun (Yunnan University)
日期: 2026-01-21
版本: 0.3.0
"""

import os
import gc
import warnings
import logging
from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict, Any, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import xarray as xr

# 抑制警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='invalid value encountered')

# 进度条
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


    def tqdm(iterable, **kwargs):
        return iterable

# OGGM导入
try:
    from oggm import cfg, workflow, tasks, utils
    from oggm.workflow import execute_entity_task

    OGGM_AVAILABLE = True
except ImportError:
    OGGM_AVAILABLE = False

# pyproj 用于测地线计算
try:
    from pyproj import Geod

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置环境变量
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ============================================================
# 全局常量
# ============================================================

LOG_SEPARATOR = "=" * 60
ERROR_MSG_MAX_LEN = 80
GC_INTERVAL = 50  # 每处理多少个冰川执行一次垃圾回收

# OGGM 任务列表
TASKS_BEFORE_MB = [
    'define_glacier_region',
    'glacier_masks',
    'compute_centerlines',
    'initialize_flowlines',
    'compute_downstream_line',
    'compute_downstream_bedshape',
    'catchment_area',
    'catchment_intersections',
    'catchment_width_geom',
    'catchment_width_correction',
    'process_climate_data',
]

TASKS_AFTER_MB = [
    'apparent_mb_from_any_mb',
    'prepare_for_inversion',
    'mass_conservation_inversion',
    'filter_inversion_output',
    'distribute_thickness_per_altitude',
]


# ============================================================
# 冰川状态追踪
# ============================================================

class GlacierStatus(Enum):
    """冰川处理状态"""
    PENDING = "pending"
    NO_MB_DATA = "no_mb_data"
    GEOMETRY_INVALID = "geometry_invalid"
    INIT_FAILED = "init_failed"
    PREPROCESS_FAILED = "preprocess_failed"
    MB_CALIB_FAILED = "mb_calib_failed"
    INVERSION_FAILED = "inversion_failed"
    THICKNESS_FAILED = "thickness_failed"
    EXPORT_FAILED = "export_failed"
    SUCCESS = "success"
    WARNING = "warning"

    @classmethod
    def failed_statuses(cls) -> List['GlacierStatus']:
        """返回所有失败状态"""
        return [
            cls.NO_MB_DATA,
            cls.GEOMETRY_INVALID,
            cls.INIT_FAILED,
            cls.PREPROCESS_FAILED,
            cls.MB_CALIB_FAILED,
            cls.INVERSION_FAILED,
            cls.THICKNESS_FAILED,
            cls.EXPORT_FAILED,
        ]


@dataclass
class GlacierRecord:
    """单个冰川的处理记录"""
    rgi_id: str
    status: GlacierStatus = GlacierStatus.PENDING
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mb_value: Optional[float] = None
    has_thickness: bool = False
    exported: bool = False

    def add_error(self, step: str, message: str):
        self.errors.append(f"[{step}] {message[:ERROR_MSG_MAX_LEN]}")

    def add_warning(self, step: str, message: str):
        self.warnings.append(f"[{step}] {message[:ERROR_MSG_MAX_LEN]}")

    def to_dict(self) -> dict:
        return {
            'rgi_id': self.rgi_id,
            'status': self.status.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'mb_value': self.mb_value,
            'has_thickness': self.has_thickness,
            'exported': self.exported
        }


class GlacierTracker:
    """
    冰川处理追踪器

    统一管理所有冰川的处理状态，支持：
    - 记录每个冰川的状态和错误
    - 获取可继续处理的冰川列表
    - 生成最终报告
    """

    def __init__(self):
        self.records: Dict[str, GlacierRecord] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def register(self, rgi_id: str, mb_value: Optional[float] = None):
        """注册一个冰川"""
        self.records[rgi_id] = GlacierRecord(rgi_id=rgi_id, mb_value=mb_value)

    def register_batch(self, rgi_ids: List[str], mb_data: Optional[pd.DataFrame] = None):
        """批量注册冰川"""
        for rgi_id in rgi_ids:
            mb_val = None
            if mb_data is not None and rgi_id in mb_data.index:
                mb_val = mb_data.loc[rgi_id, 'dmdtda_kg']
            self.register(rgi_id, mb_val)

    def set_status(self, rgi_id: str, status: GlacierStatus, error_msg: str = None):
        """设置冰川状态"""
        if rgi_id in self.records:
            self.records[rgi_id].status = status
            if error_msg and status in GlacierStatus.failed_statuses():
                self.records[rgi_id].add_error(status.value, error_msg)

    def add_warning(self, rgi_id: str, step: str, message: str):
        """添加警告"""
        if rgi_id in self.records:
            self.records[rgi_id].add_warning(step, message)
            if self.records[rgi_id].status == GlacierStatus.PENDING:
                self.records[rgi_id].status = GlacierStatus.WARNING

    def mark_success(self, rgi_id: str, has_thickness: bool = True, exported: bool = False):
        """标记成功"""
        if rgi_id in self.records:
            rec = self.records[rgi_id]
            rec.has_thickness = has_thickness
            rec.exported = exported
            if not rec.warnings:
                rec.status = GlacierStatus.SUCCESS
            else:
                rec.status = GlacierStatus.WARNING

    def get_valid_ids(self) -> Set[str]:
        """获取有效（未失败）的冰川ID"""
        failed = GlacierStatus.failed_statuses()
        return {
            rgi_id for rgi_id, rec in self.records.items()
            if rec.status not in failed
        }

    def get_failed_ids(self) -> Set[str]:
        """获取失败的冰川ID"""
        failed = GlacierStatus.failed_statuses()
        return {
            rgi_id for rgi_id, rec in self.records.items()
            if rec.status in failed
        }

    def filter_gdirs(self, gdirs: List) -> List:
        """过滤掉已失败的glacier directories"""
        valid_ids = self.get_valid_ids()
        return [gdir for gdir in gdirs if gdir.rgi_id in valid_ids]

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total': len(self.records),
            'by_status': {},
            'success_rate': 0.0,
            'with_warnings': 0,
            'exported': 0,
        }

        for status in GlacierStatus:
            count = sum(1 for rec in self.records.values() if rec.status == status)
            if count > 0:
                stats['by_status'][status.value] = count

        success_count = sum(1 for rec in self.records.values()
                            if rec.status in [GlacierStatus.SUCCESS, GlacierStatus.WARNING])
        stats['success_rate'] = success_count / len(self.records) * 100 if self.records else 0

        stats['with_warnings'] = sum(1 for rec in self.records.values() if rec.warnings)
        stats['exported'] = sum(1 for rec in self.records.values() if rec.exported)

        return stats

    def generate_report(self, output_path: Path = None) -> str:
        """生成处理报告"""
        stats = self.get_statistics()

        lines = [
            LOG_SEPARATOR,
            "冰川处理报告",
            LOG_SEPARATOR,
            f"总冰川数: {stats['total']}",
            f"成功率: {stats['success_rate']:.1f}%",
            f"已导出: {stats['exported']}",
            "",
            "状态统计:",
        ]

        for status, count in stats['by_status'].items():
            lines.append(f"  - {status}: {count}")

        failed_ids = self.get_failed_ids()
        if failed_ids:
            lines.extend(["", f"失败的冰川 ({len(failed_ids)}):"])

            by_reason = {}
            for rgi_id in failed_ids:
                rec = self.records[rgi_id]
                reason = rec.status.value
                if reason not in by_reason:
                    by_reason[reason] = []
                by_reason[reason].append(rgi_id)

            for reason, ids in by_reason.items():
                lines.append(f"\n  [{reason}] ({len(ids)}个):")
                for rgi_id in ids[:10]:
                    rec = self.records[rgi_id]
                    error_msg = rec.errors[-1] if rec.errors else "未知错误"
                    lines.append(f"    - {rgi_id}: {error_msg[:60]}")
                if len(ids) > 10:
                    lines.append(f"    ... 还有 {len(ids) - 10} 个")

        if stats['with_warnings'] > 0:
            lines.extend(["", f"有警告的冰川: {stats['with_warnings']}"])

        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            lines.extend(["", f"处理耗时: {duration / 60:.1f} 分钟"])

        lines.append(LOG_SEPARATOR)

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)

            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'statistics': stats,
                    'records': {rgi_id: rec.to_dict() for rgi_id, rec in self.records.items()}
                }, f, indent=2, ensure_ascii=False)

            logger.info(f"报告已保存: {output_path}")
            logger.info(f"详细记录: {json_path}")

        return report


# ============================================================
# 配置类
# ============================================================

class BareDEMConfig:
    """基岩DEM生成配置"""

    def __init__(
            self,
            lia_path: str,
            rgi_path: str,
            dem_path: str,
            mb_paths: List[str],
            working_dir: str,
            output_dir: str,
            min_lia_area_km2: float = 10.0,
            mb_period: str = "2000-01-01_2020-01-01",
            border: int = 360,
            n_workers: int = 16,
            use_multiprocessing: bool = True,
            baseline_climate: str = 'W5E5',
    ):
        self.lia_path = Path(lia_path)
        self.rgi_path = Path(rgi_path)
        self.dem_path = Path(dem_path)
        self.mb_paths = [Path(p) for p in mb_paths]
        self.working_dir = Path(working_dir)
        self.output_dir = Path(output_dir)
        self.min_lia_area_km2 = min_lia_area_km2
        self.mb_period = mb_period
        self.border = border
        self.n_workers = n_workers
        self.use_multiprocessing = use_multiprocessing
        self.baseline_climate = baseline_climate

    def validate(self) -> Dict[str, Any]:
        """验证配置有效性"""
        result = {'valid': True, 'errors': [], 'warnings': []}
        for path, name in [
            (self.lia_path, 'LIA边界'),
            (self.rgi_path, 'RGI边界'),
            (self.dem_path, 'DEM'),
        ]:
            if not path.exists():
                result['valid'] = False
                result['errors'].append(f"{name}不存在: {path}")
        for path in self.mb_paths:
            if not path.exists():
                result['valid'] = False
                result['errors'].append(f"物质平衡数据不存在: {path}")
        return result

    @property
    def output_bare_dem_path(self) -> Path:
        return self.output_dir / "Himalaya_bare_DEM.tif"

    @property
    def output_thickness_path(self) -> Path:
        return self.output_dir / "Himalaya_thickness.tif"

    @property
    def output_per_glacier_dir(self) -> Path:
        return self.output_dir / "per_glacier"

    @property
    def output_report_path(self) -> Path:
        return self.output_dir / "processing_report.txt"


# ============================================================
# 数据加载模块
# ============================================================

class DataLoader:
    """数据加载器"""

    def __init__(self, config: BareDEMConfig):
        self.config = config

    def load_lia_boundaries(self) -> gpd.GeoDataFrame:
        """加载LIA冰川边界"""
        logger.info(f"加载LIA边界: {self.config.lia_path}")
        lia = gpd.read_file(self.config.lia_path)
        original_count = len(lia)
        lia_large = lia[lia['LIAArea'] >= self.config.min_lia_area_km2].copy()
        logger.info(f"  原始: {original_count}, 筛选后(≥{self.config.min_lia_area_km2} km²): {len(lia_large)}")
        return lia_large

    def load_rgi_glaciers(self) -> gpd.GeoDataFrame:
        """加载RGI现代冰川边界"""
        logger.info(f"加载RGI边界: {self.config.rgi_path}")
        rgi = gpd.read_file(self.config.rgi_path)
        logger.info(f"  冰川数: {len(rgi)}")
        return rgi

    def match_lia_rgi(
            self,
            lia_gdf: gpd.GeoDataFrame,
            rgi_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """匹配LIA冰川与现代RGI冰川"""
        logger.info("匹配LIA与RGI冰川...")
        all_rgi_ids = set()
        for rgi_str in lia_gdf['AllRGIs'].dropna():
            ids = [x.strip() for x in str(rgi_str).split(';')]
            all_rgi_ids.update(ids)
        rgi_matched = rgi_gdf[rgi_gdf['RGIId'].isin(all_rgi_ids)].copy()
        logger.info(f"  关联的RGI ID数: {len(all_rgi_ids)}")
        logger.info(f"  匹配的冰川数: {len(rgi_matched)}")
        return rgi_matched

    def load_mass_balance_data(self) -> pd.DataFrame:
        """加载Hugonnet物质平衡数据"""
        logger.info(f"加载物质平衡数据，时段: {self.config.mb_period}")
        dfs = []
        for path in self.config.mb_paths:
            logger.info(f"  读取: {path}")
            df = pd.read_csv(path)
            dfs.append(df)
        mb_all = pd.concat(dfs, ignore_index=True)
        logger.info(f"  合并后总记录: {len(mb_all)}")
        mb_data = mb_all[mb_all['period'] == self.config.mb_period].copy()
        logger.info(f"  时段 {self.config.mb_period} 记录数: {len(mb_data)}")
        mb_data['dmdtda_kg'] = mb_data['dmdtda'] * 1000
        mb_data = mb_data.set_index('rgiid')
        return mb_data


# ============================================================
# 几何处理模块
# ============================================================

def clean_geometries(gdf: gpd.GeoDataFrame, tracker: GlacierTracker = None) -> gpd.GeoDataFrame:
    """清理几何数据，并记录无效几何"""
    logger.info(f"清理几何数据，原始: {len(gdf)}")
    original_ids = set(gdf['RGIId'].tolist()) if 'RGIId' in gdf.columns else set()

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

    if tracker and 'RGIId' in gdf.columns:
        remaining_ids = set(gdf['RGIId'].tolist())
        removed_ids = original_ids - remaining_ids
        for rgi_id in removed_ids:
            tracker.set_status(rgi_id, GlacierStatus.GEOMETRY_INVALID, "几何无效或坐标异常")

    logger.info(f"清理后: {len(gdf)}")
    return gdf


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """确保数据为WGS84坐标系"""
    if gdf.crs is None:
        logger.warning("数据无坐标系，假设为WGS84")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        logger.info(f"转换坐标系: {gdf.crs} -> EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


# ============================================================
# OGGM处理器
# ============================================================

class OGGMProcessor:
    """OGGM处理器（带错误追踪）"""

    def __init__(self, config: BareDEMConfig, tracker: GlacierTracker):
        self.config = config
        self.tracker = tracker
        self.gdirs = None
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'start_time': None,
            'end_time': None,
        }

    def _check_oggm(self):
        """检查OGGM是否可用"""
        if not OGGM_AVAILABLE:
            raise ImportError("OGGM未安装。请运行: conda install -c conda-forge oggm")
        import oggm
        version = getattr(cfg, 'OGGM_VERSION', None) or getattr(oggm, '__version__', 'unknown')
        logger.info(f"OGGM版本: {version}")

    def initialize(self):
        """初始化OGGM配置"""
        self._check_oggm()
        logger.info(LOG_SEPARATOR)
        logger.info("初始化OGGM配置")
        logger.info(LOG_SEPARATOR)

        cfg.initialize(logging_level='WARNING')
        self.config.working_dir.mkdir(parents=True, exist_ok=True)
        cfg.PATHS['working_dir'] = str(self.config.working_dir)
        cfg.PATHS['dem_file'] = str(self.config.dem_path)
        cfg.PARAMS['border'] = self.config.border
        cfg.PARAMS['use_intersects'] = False
        cfg.PARAMS['use_rgi_area'] = False
        cfg.PARAMS['continue_on_error'] = True
        cfg.PARAMS['use_multiprocessing'] = self.config.use_multiprocessing
        cfg.PARAMS['mp_processes'] = self.config.n_workers
        cfg.PARAMS['use_multiple_flowlines'] = True
        cfg.PARAMS['baseline_climate'] = self.config.baseline_climate

        logger.info(f"  工作目录: {cfg.PATHS['working_dir']}")
        logger.info(f"  DEM文件: {cfg.PATHS['dem_file']}")
        logger.info(f"  并行进程: {cfg.PARAMS['mp_processes']}")

    def init_glacier_directories(self, rgi_gdf: gpd.GeoDataFrame, reset: bool = False) -> List:
        """初始化glacier directories"""
        logger.info("初始化glacier directories...")
        self.stats['total'] = len(rgi_gdf)
        self.stats['start_time'] = datetime.now()

        try:
            self.gdirs = workflow.init_glacier_directories(rgi_gdf, reset=reset)
        except Exception as e:
            logger.error(f"初始化glacier directories失败: {e}")
            raise

        initialized_ids = {gdir.rgi_id for gdir in self.gdirs}
        for rgi_id in rgi_gdf['RGIId']:
            if rgi_id not in initialized_ids:
                self.tracker.set_status(rgi_id, GlacierStatus.INIT_FAILED, "OGGM初始化失败")

        logger.info(f"  初始化完成: {len(self.gdirs)} 个冰川")
        return self.gdirs

    def run_preprocessing_tasks(self, show_progress: bool = True):
        """执行预处理任务"""
        logger.info(LOG_SEPARATOR)
        logger.info("执行预处理任务")
        logger.info(LOG_SEPARATOR)

        for task_name in TASKS_BEFORE_MB:
            valid_gdirs = self.tracker.filter_gdirs(self.gdirs)
            logger.info(f"  执行: {task_name} (有效冰川: {len(valid_gdirs)})")

            if not valid_gdirs:
                logger.warning("  无有效冰川，跳过")
                continue

            try:
                task_func = getattr(tasks, task_name)
                if task_name == 'define_glacier_region':
                    for gdir in valid_gdirs:
                        try:
                            task_func(gdir, source='USER')
                        except Exception as e:
                            self.tracker.set_status(
                                gdir.rgi_id,
                                GlacierStatus.PREPROCESS_FAILED,
                                f"{task_name}: {str(e)[:ERROR_MSG_MAX_LEN]}"
                            )
                else:
                    execute_entity_task(task_func, valid_gdirs)
            except Exception as e:
                logger.warning(f"    任务 {task_name} 部分失败: {str(e)[:ERROR_MSG_MAX_LEN]}")

            gc.collect()

    def calibrate_mass_balance(self, mb_data: pd.DataFrame, show_progress: bool = True):
        """执行物质平衡校准"""
        logger.info(LOG_SEPARATOR)
        logger.info("执行物质平衡校准")
        logger.info(LOG_SEPARATOR)

        valid_gdirs = self.tracker.filter_gdirs(self.gdirs)
        success = 0
        failed = 0

        iterator = tqdm(valid_gdirs, desc="MB校准", ncols=80) if show_progress and HAS_TQDM else valid_gdirs

        for i, gdir in enumerate(iterator):
            rgi_id = gdir.rgi_id
            try:
                if rgi_id not in mb_data.index:
                    self.tracker.set_status(rgi_id, GlacierStatus.NO_MB_DATA, "MB数据缺失")
                    failed += 1
                    continue

                ref_mb = mb_data.loc[rgi_id, 'dmdtda_kg']
                tasks.mb_calibration_from_scalar_mb(
                    gdir,
                    ref_mb=ref_mb,
                    ref_mb_years=(2000, 2020),
                    overwrite_gdir=True,
                    calibrate_param1='melt_f',
                    calibrate_param2='temp_bias',
                )
                success += 1
            except Exception as e:
                error_msg = str(e)
                if "ref mb not matched" in error_msg:
                    self.tracker.set_status(rgi_id, GlacierStatus.MB_CALIB_FAILED,
                                            "MB校准失败: 参数范围不足以匹配观测值")
                else:
                    self.tracker.set_status(rgi_id, GlacierStatus.MB_CALIB_FAILED,
                                            f"MB校准失败: {error_msg[:ERROR_MSG_MAX_LEN]}")
                failed += 1

            if (i + 1) % GC_INTERVAL == 0:
                gc.collect()

        logger.info(f"  MB校准成功: {success}, 失败: {failed}")

    def run_inversion_tasks(self, show_progress: bool = True):
        """执行冰厚反演任务"""
        logger.info(LOG_SEPARATOR)
        logger.info("执行冰厚反演任务")
        logger.info(LOG_SEPARATOR)

        for task_name in TASKS_AFTER_MB:
            valid_gdirs = self.tracker.filter_gdirs(self.gdirs)
            logger.info(f"  执行: {task_name} (有效冰川: {len(valid_gdirs)})")

            if not valid_gdirs:
                logger.warning("  无有效冰川，跳过")
                continue

            try:
                task_func = getattr(tasks, task_name)
                execute_entity_task(task_func, valid_gdirs)
            except Exception as e:
                logger.warning(f"    任务 {task_name} 部分失败: {str(e)[:ERROR_MSG_MAX_LEN]}")

            gc.collect()

        logger.info("检查冰厚反演结果...")
        valid_gdirs = self.tracker.filter_gdirs(self.gdirs)
        success_count = 0

        for gdir in valid_gdirs:
            rgi_id = gdir.rgi_id
            try:
                with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
                    if 'distributed_thickness' in ds:
                        success_count += 1
                        self.tracker.mark_success(rgi_id, has_thickness=True)
                        thick = ds['distributed_thickness'].values
                        if np.any(thick < 0):
                            self.tracker.add_warning(rgi_id, "inversion",
                                                     "存在负冰厚值，可能有strongly negative flux")
                    else:
                        self.tracker.set_status(rgi_id, GlacierStatus.THICKNESS_FAILED,
                                                "gridded_data中无冰厚数据")
            except FileNotFoundError:
                self.tracker.set_status(rgi_id, GlacierStatus.INVERSION_FAILED,
                                        "反演输出文件不存在")
            except Exception as e:
                self.tracker.set_status(rgi_id, GlacierStatus.THICKNESS_FAILED,
                                        f"读取冰厚数据失败: {str(e)[:ERROR_MSG_MAX_LEN]}")

        self.stats['success'] = success_count
        self.stats['failed'] = len(self.gdirs) - success_count
        self.stats['end_time'] = datetime.now()
        logger.info(f"  冰厚反演成功: {success_count}/{len(valid_gdirs)}")

    def get_valid_gdirs(self) -> List:
        """获取有效的glacier directories"""
        return self.tracker.filter_gdirs(self.gdirs)

    def print_stats(self):
        """打印处理统计"""
        logger.info(LOG_SEPARATOR)
        logger.info("处理统计")
        logger.info(LOG_SEPARATOR)
        logger.info(f"  总冰川数: {self.stats['total']}")
        logger.info(f"  成功反演: {self.stats['success']}")
        logger.info(f"  失败数: {self.stats['failed']}")
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
            logger.info(f"  耗时: {duration / 60:.1f} 分钟")


# ============================================================
# 基岩DEM生成器
# ============================================================

class BareDEMGenerator:
    """基岩DEM生成器"""

    def __init__(self, config: BareDEMConfig, tracker: GlacierTracker):
        self.config = config
        self.tracker = tracker
        self.dem_profile = None
        self.dem_transform = None
        self.dem_crs = None
        self.dem_width = None
        self.dem_height = None
        self._load_dem_metadata()

    def _load_dem_metadata(self):
        """加载 DEM 元数据"""
        try:
            with rasterio.open(self.config.dem_path) as src:
                self.dem_profile = src.profile.copy()
                self.dem_transform = src.transform
                self.dem_crs = src.crs
                self.dem_width = src.width
                self.dem_height = src.height
            logger.info(f"DEM元数据已加载: {self.dem_width}x{self.dem_height}, CRS={self.dem_crs}")
        except Exception as e:
            logger.warning(f"加载DEM元数据失败: {e}")

    def generate_per_glacier(self, gdirs: List, show_progress: bool = True) -> Tuple[int, int]:
        """生成逐冰川文件"""
        logger.info(LOG_SEPARATOR)
        logger.info("生成逐冰川DEM和冰厚")
        logger.info(LOG_SEPARATOR)

        per_glacier_dir = self.config.output_per_glacier_dir
        per_glacier_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  输出目录: {per_glacier_dir}")

        valid_gdirs = self.tracker.filter_gdirs(gdirs)
        logger.info(f"  有效冰川数: {len(valid_gdirs)}")

        saved_count = 0
        failed_count = 0

        iterator = tqdm(valid_gdirs, desc="保存逐冰川DEM", ncols=80) if show_progress and HAS_TQDM else valid_gdirs

        for i, gdir in enumerate(iterator):
            rgi_id = gdir.rgi_id
            try:
                if self._save_single_glacier(gdir, per_glacier_dir):
                    saved_count += 1
                    self.tracker.records[rgi_id].exported = True
                else:
                    self.tracker.set_status(rgi_id, GlacierStatus.EXPORT_FAILED, "保存文件失败")
                    failed_count += 1
            except Exception as e:
                self.tracker.set_status(rgi_id, GlacierStatus.EXPORT_FAILED,
                                        f"导出异常: {str(e)[:ERROR_MSG_MAX_LEN]}")
                failed_count += 1

            if (i + 1) % GC_INTERVAL == 0:
                gc.collect()

        logger.info(f"\n保存统计:")
        logger.info(f"  成功: {saved_count}")
        logger.info(f"  失败: {failed_count}")
        return saved_count, failed_count

    def _save_single_glacier(self, gdir, output_dir: Path) -> bool:
        """保存单个冰川的DEM和冰厚（重投影到WGS84）"""
        from rasterio.warp import calculate_default_transform, reproject, Resampling

        rgi_id = gdir.rgi_id
        glacier_dir = output_dir / rgi_id.replace('-', '_').replace('.', '_')
        glacier_dir.mkdir(parents=True, exist_ok=True)

        try:
            with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
                if 'distributed_thickness' not in ds:
                    return False
                thick = ds['distributed_thickness'].values.copy()
                topo = ds['topo'].values.copy()
                x = ds.x.values
                y = ds.y.values

            # 处理nodata
            nodata = -9999.0
            thick = np.where(np.isfinite(thick) & (np.abs(thick) < 1e10), thick, nodata)
            topo = np.where(np.isfinite(topo) & (np.abs(topo) < 1e10), topo, nodata)

            valid_mask = (thick != nodata) & (topo != nodata)
            bare = np.where(valid_mask, topo - thick, nodata)

            # 源坐标系参数
            dx = abs(x[1] - x[0])
            dy = abs(y[0] - y[1])
            src_transform = from_bounds(
                x.min() - dx / 2, y.min() - dy / 2,
                x.max() + dx / 2, y.max() + dy / 2,
                len(x), len(y)
            )
            src_crs = gdir.grid.proj.srs
            dst_crs = 'EPSG:4326'

            # 计算目标变换参数
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, len(x), len(y),
                left=x.min() - dx / 2,
                bottom=y.min() - dy / 2,
                right=x.max() + dx / 2,
                top=y.max() + dy / 2
            )

            profile = {
                'driver': 'GTiff',
                'dtype': np.float32,
                'width': dst_width,
                'height': dst_height,
                'count': 1,
                'crs': dst_crs,
                'transform': dst_transform,
                'compress': 'lzw',
                'nodata': nodata,
            }

            safe_id = rgi_id.replace('-', '_').replace('.', '_')

            for data, filename in [
                (thick, f"{safe_id}_thickness.tif"),
                (bare, f"{safe_id}_bare_dem.tif"),
                (topo, f"{safe_id}_surface_dem.tif"),
            ]:
                dst_array = np.full((dst_height, dst_width), nodata, dtype=np.float32)
                reproject(
                    source=data.astype(np.float32),
                    destination=dst_array,
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=nodata,
                    dst_nodata=nodata
                )
                with rasterio.open(glacier_dir / filename, 'w', **profile) as dst:
                    dst.write(dst_array, 1)

            return True

        except Exception as e:
            logger.debug(f"  保存 {rgi_id} 失败: {e}")
            return False

    def _extract_and_accumulate_thickness(self, gdir, thickness_total: np.ndarray):
        """提取单个冰川的冰厚并累积到总数组（用于合并DEM扩展）"""
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            if 'distributed_thickness' not in ds:
                raise ValueError("无冰厚数据")
            thick = ds['distributed_thickness'].values
            x = ds.x.values
            y = ds.y.values

        dx = x[1] - x[0]
        dy = y[0] - y[1]

        local_transform = from_bounds(
            x.min() - dx / 2, y.min() - dy / 2,
            x.max() + dx / 2, y.max() + dy / 2,
            len(x), len(y)
        )

        thick_reproj = np.zeros((self.dem_height, self.dem_width), dtype=np.float32)

        reproject(
            source=thick.astype(np.float32),
            destination=thick_reproj,
            src_transform=local_transform,
            src_crs=gdir.grid.proj,
            dst_transform=self.dem_transform,
            dst_crs=self.dem_crs,
            resampling=Resampling.bilinear
        )

        np.maximum(thickness_total, thick_reproj, out=thickness_total)

    def save(self, bare_dem: np.ndarray, thickness: np.ndarray):
        """保存合并结果（用于合并DEM扩展）"""
        logger.info(LOG_SEPARATOR)
        logger.info("保存结果")
        logger.info(LOG_SEPARATOR)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_profile = self.dem_profile.copy()
        output_profile.update(dtype=np.float32, compress='lzw')

        output_path = self.config.output_bare_dem_path
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(bare_dem, 1)
        logger.info(f"  基岩DEM: {output_path}")

        thickness_path = self.config.output_thickness_path
        with rasterio.open(thickness_path, 'w', **output_profile) as dst:
            dst.write(thickness, 1)
        logger.info(f"  冰厚数据: {thickness_path}")


# ============================================================
# Flowline导出器
# ============================================================

class FlowlineExporter:
    """Flowline导出器"""

    @staticmethod
    def _geodesic_length(line) -> float:
        """计算测地线长度（米）"""
        if HAS_PYPROJ:
            geod = Geod(ellps='WGS84')
            coords = list(line.coords)
            total = 0.0
            for i in range(len(coords) - 1):
                _, _, dist = geod.inv(
                    coords[i][0], coords[i][1],
                    coords[i + 1][0], coords[i + 1][1]
                )
                total += dist
            return total
        else:
            return line.length * 111000

    @staticmethod
    def extract_and_save(
            gdirs: List,
            output_dir: Union[str, Path],
            tracker: GlacierTracker = None,
            show_progress: bool = True
    ) -> gpd.GeoDataFrame:
        """提取并保存flowlines"""
        from shapely.geometry import LineString

        logger.info("提取Flowlines...")
        records = []

        iterator = tqdm(gdirs, desc="提取flowlines", ncols=80) if show_progress and HAS_TQDM else gdirs

        for gdir in iterator:
            try:
                fls = gdir.read_pickle('inversion_flowlines')
                for i, fl in enumerate(fls):
                    try:
                        x, y = fl.line.xy
                    except AttributeError:
                        continue

                    coords = []
                    for xi, yi in zip(x, y):
                        lon, lat = gdir.grid.ij_to_crs(xi, yi, crs='EPSG:4326')
                        coords.append((lon, lat))

                    if len(coords) >= 2:
                        line = LineString(coords)
                        length_m = getattr(fl, 'length_m', None)
                        if length_m is None:
                            length_m = FlowlineExporter._geodesic_length(line)

                        records.append({
                            'rgi_id': gdir.rgi_id,
                            'fl_id': i,
                            'geometry': line,
                            'length_m': length_m,
                        })
            except Exception as e:
                logger.debug(f"  {gdir.rgi_id}: {e}")
                continue

        logger.info(f"  提取了 {len(records)} 条flowlines")

        flowlines_gdf = gpd.GeoDataFrame(records, geometry='geometry', crs="EPSG:4326")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        gpkg_path = output_dir / "flowlines.gpkg"
        flowlines_gdf.to_file(gpkg_path, driver="GPKG")
        logger.info(f"  保存至: {gpkg_path}")

        shp_path = output_dir / "flowlines.shp"
        flowlines_gdf.to_file(shp_path, driver="ESRI Shapefile")
        logger.info(f"  保存至: {shp_path}")

        return flowlines_gdf


# ============================================================
# 主工作流
# ============================================================

class BareDEMWorkflow:
    """基岩DEM生成完整工作流（鲁棒版）"""

    def __init__(self, config: BareDEMConfig):
        self.config = config
        self.tracker = GlacierTracker()
        self.data_loader = DataLoader(config)
        self.oggm_processor = OGGMProcessor(config, self.tracker)
        self.dem_generator = BareDEMGenerator(config, self.tracker)
        self.lia_gdf = None
        self.rgi_gdf = None
        self.mb_data = None

    def run(
            self,
            reset: bool = False,
            export_flowlines: bool = True,
            show_progress: bool = True
    ) -> Path:
        """运行完整工作流"""
        logger.info(LOG_SEPARATOR)
        logger.info("开始基岩DEM生成工作流 (鲁棒版 v0.3.0)")
        logger.info(LOG_SEPARATOR)

        self.tracker.start_time = datetime.now()

        validation = self.config.validate()
        if not validation['valid']:
            for error in validation['errors']:
                logger.error(error)
            raise ValueError("配置验证失败")

        # Step 1: 加载数据
        logger.info("\n[Step 1] 加载数据")

        # 加载原始数据
        lia_full = gpd.read_file(self.config.lia_path)
        rgi_full = gpd.read_file(self.config.rgi_path)
        logger.info(f"  LIA 总数: {len(lia_full)}, RGI 总数: {len(rgi_full)}")

        # 筛选 RGI >= 10 km²
        rgi_large = rgi_full[rgi_full['Area'] >= 10].copy()
        large_rgi_ids = set(rgi_large['RGIId'])
        logger.info(f"  RGI >= 10 km²: {len(rgi_large)} 个")

        # 找包含大冰川的 LIA
        def contains_large_rgi(rgi_str):
            if pd.isna(rgi_str):
                return False
            ids = [x.strip() for x in str(rgi_str).split(';')]
            return bool(set(ids) & large_rgi_ids)

        lia_matched = lia_full[lia_full['AllRGIs'].apply(contains_large_rgi)].copy()
        logger.info(f"  包含大冰川的 LIA: {len(lia_matched)} 个")

        self.lia_gdf = lia_matched
        self.rgi_gdf = rgi_large
        self.mb_data = self.data_loader.load_mass_balance_data()

        # Step 2: 数据预处理和预筛选
        logger.info("\n[Step 2] 数据预处理和预筛选")
        self.lia_gdf = ensure_wgs84(self.lia_gdf)
        self.rgi_gdf = ensure_wgs84(self.rgi_gdf)

        self.tracker.register_batch(self.rgi_gdf['RGIId'].tolist(), self.mb_data)

        rgi_ids_with_mb = set(self.mb_data.index)
        no_mb_count = 0
        for rgi_id in self.rgi_gdf['RGIId']:
            if rgi_id not in rgi_ids_with_mb:
                self.tracker.set_status(rgi_id, GlacierStatus.NO_MB_DATA, "无物质平衡数据")
                no_mb_count += 1

        if no_mb_count > 0:
            logger.info(f"  预筛选: {no_mb_count} 个冰川无MB数据，已标记跳过")

        self.rgi_gdf = self.rgi_gdf[self.rgi_gdf['RGIId'].isin(rgi_ids_with_mb)].copy()
        logger.info(f"  有MB数据的冰川: {len(self.rgi_gdf)}")

        self.rgi_gdf = clean_geometries(self.rgi_gdf, self.tracker)

        # Step 3: OGGM初始化
        logger.info("\n[Step 3] OGGM初始化")
        self.oggm_processor.initialize()
        self.oggm_processor.init_glacier_directories(self.rgi_gdf, reset=reset)

        # Step 4: 预处理任务
        logger.info("\n[Step 4] 预处理任务")
        self.oggm_processor.run_preprocessing_tasks(show_progress)

        # Step 5: 物质平衡校准
        logger.info("\n[Step 5] 物质平衡校准")
        self.oggm_processor.calibrate_mass_balance(self.mb_data, show_progress)

        # Step 6: 冰厚反演
        logger.info("\n[Step 6] 冰厚反演")
        self.oggm_processor.run_inversion_tasks(show_progress)
        self.oggm_processor.print_stats()

        # Step 7: 生成逐冰川DEM
        logger.info("\n[Step 7] 生成逐冰川DEM")
        valid_gdirs = self.oggm_processor.get_valid_gdirs()
        self.dem_generator.generate_per_glacier(valid_gdirs, show_progress)

        # Step 8: 导出flowlines (可选)
        if export_flowlines:
            logger.info("\n[Step 8] 导出Flowlines")
            FlowlineExporter.extract_and_save(
                valid_gdirs,
                self.config.output_dir,
                self.tracker,
                show_progress
            )

        self.tracker.end_time = datetime.now()
        report = self.tracker.generate_report(self.config.output_report_path)

        logger.info("\n" + report)
        logger.info(f"\n输出目录: {self.config.output_dir}")

        return self.config.output_per_glacier_dir


# ============================================================
# 命令行接口
# ============================================================

def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='喜马拉雅地区基岩DEM生成工具 (鲁棒版 v0.3.0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python bare_dem_himalaya_robust.py
  python bare_dem_himalaya_robust.py --reset
  python bare_dem_himalaya_robust.py --workers 8
  python bare_dem_himalaya_robust.py --no-flowlines
        """
    )

    parser.add_argument('--reset', action='store_true', help='重置OGGM工作目录')
    parser.add_argument('--workers', type=int, default=16, help='并行进程数 (默认: 16)')
    parser.add_argument('--no-flowlines', action='store_true', help='不导出flowlines')
    parser.add_argument('--quiet', action='store_true', help='静默模式')

    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    config = BareDEMConfig(
        lia_path="/mnt/e/2Data/022_工作2/出图数据/Himalaya_LIA_with_RGIID_Complete/Himalaya_LIA_with_RGIID_Complete.shp",
        rgi_path="/mnt/e/2Data/022_工作2/出图数据/himalaya_glaciers_complete/himalaya_glaciers_complete.shp",
        dem_path="/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif",
        mb_paths=[
            "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_14/dh_14_rgi60_pergla_rates.csv",
            "/mnt/d/01 文件整理/04 数据/03 物质平衡数据/hugonnet time_series_15/dh_15_rgi60_pergla_rates.csv",
        ],
        working_dir="/mnt/f/oggm_related/oggm_LIA_himalaya_large_rgi",
        output_dir="/mnt/f/oggm_related/oggm_LIA_himalaya_large_rgi/output",
        n_workers=args.workers,
    )

    workflow = BareDEMWorkflow(config)
    workflow.run(
        reset=args.reset,
        export_flowlines=not args.no_flowlines,
    )


if __name__ == "__main__":
    main()