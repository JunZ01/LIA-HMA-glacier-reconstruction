"""
flowline.py - OGGM Flowline生成模块 (优化版)
============================================

优化功能:
1. tqdm进度条 + 时间估计
2. 断点续传 - 跳过已完成的冰川
3. 详细错误日志
4. 内存优化批处理
5. 实时统计
"""

import os
import sys
import shutil
import logging
import warnings
import json
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
from datetime import datetime
import time

import numpy as np
import geopandas as gpd

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

# OGGM imports
try:
    import oggm
    from oggm import cfg, utils, workflow, tasks
    OGGM_AVAILABLE = True
except ImportError:
    OGGM_AVAILABLE = False

# 设置HDF5文件锁定模式 - 解决多进程NetCDF写入冲突
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'  # 禁用HDF5文件锁

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FlowlineProcessor:
    """Flowline处理器 - 带进度跟踪和断点续传"""
    
    def __init__(
        self,
        working_dir: Union[str, Path],
        dem_path: Union[str, Path],
        n_workers: int = 4,
        continue_on_error: bool = True
    ):
        self.working_dir = Path(working_dir)
        self.dem_path = Path(dem_path)
        self.n_workers = n_workers
        self.continue_on_error = continue_on_error
        
        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': None,
            'end_time': None
        }
        
        # 错误日志文件
        self.error_log_path = self.working_dir / 'error_log.json'
        self.errors = []
        
        # 进度文件
        self.progress_file = self.working_dir / 'progress.json'
        
    def _check_oggm(self):
        """检查OGGM是否可用"""
        if not OGGM_AVAILABLE:
            raise ImportError("OGGM未安装。请运行: conda install -c conda-forge oggm")
        logger.info(f"OGGM版本: {oggm.__version__}")
        
    def _init_config(self):
        """初始化OGGM配置"""
        cfg.initialize(logging_level='WARNING')
        cfg.PATHS['working_dir'] = str(self.working_dir)
        cfg.PATHS['dem_file'] = str(self.dem_path)
        cfg.PARAMS['use_rgi_area'] = False
        cfg.PARAMS['use_multiprocessing'] = self.n_workers > 1
        cfg.PARAMS['mp_processes'] = self.n_workers
        cfg.PARAMS['continue_on_error'] = self.continue_on_error
        cfg.PARAMS['use_intersects'] = False
        cfg.PARAMS['border'] = 360

        
    def _load_progress(self) -> set:
        """加载已完成的冰川ID"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('completed', []))
            except:
                pass
        return set()
    
    def _save_progress(self, completed: set):
        """保存进度"""
        with open(self.progress_file, 'w') as f:
            json.dump({
                'completed': list(completed),
                'last_update': datetime.now().isoformat(),
                'stats': self.stats
            }, f, indent=2)
    
    def _log_error(self, rgi_id: str, task: str, error: str):
        """记录错误"""
        self.errors.append({
            'rgi_id': rgi_id,
            'task': task,
            'error': str(error),
            'time': datetime.now().isoformat()
        })
        
    def _save_errors(self):
        """保存错误日志"""
        if self.errors:
            with open(self.error_log_path, 'w') as f:
                json.dump(self.errors, f, indent=2)
            logger.info(f"错误日志已保存: {self.error_log_path}")
    
    def init_glacier_dirs(
        self,
        outline_path: Union[str, Path],
        reset: bool = False
    ) -> List:
        """初始化glacier directories (带断点续传)"""
        self._check_oggm()
        self._init_config()
        
        logger.info("="*60)
        logger.info("初始化Glacier Directories")
        logger.info("="*60)
        
        # 读取边界
        logger.info(f"读取冰川边界: {outline_path}")
        rgidf = gpd.read_file(outline_path)
        self.stats['total'] = len(rgidf)
        logger.info(f"  冰川数量: {len(rgidf)}")
        
        # 检查RGIId
        if 'RGIId' not in rgidf.columns:
            raise ValueError("边界文件缺少'RGIId'字段")
        
        duplicates = rgidf['RGIId'].duplicated().sum()
        if duplicates > 0:
            raise ValueError(f"RGIId存在 {duplicates} 个重复")
        
        # 断点续传检查
        completed_ids = self._load_progress() if not reset else set()
        if completed_ids:
            logger.info(f"  发现已完成: {len(completed_ids)} 个冰川")
            rgidf_todo = rgidf[~rgidf['RGIId'].isin(completed_ids)]
            self.stats['skipped'] = len(completed_ids)
            logger.info(f"  待处理: {len(rgidf_todo)} 个冰川")
        else:
            rgidf_todo = rgidf
        
        # 重置目录
        if reset:
            gdir_path = self.working_dir / 'per_glacier'
            if gdir_path.exists():
                logger.info(f"重置目录: {gdir_path}")
                shutil.rmtree(gdir_path)
            completed_ids = set()
        
        # 初始化gdirs
        if len(rgidf_todo) > 0:
            logger.info("创建glacier directories...")
            self.stats['start_time'] = datetime.now().isoformat()
            
            gdirs = workflow.init_glacier_directories(
                rgidf_todo,
                from_prepro_level=None,
                prepro_border=80,
                reset=False
            )
            
            # 更新已完成列表
            for gdir in gdirs:
                completed_ids.add(gdir.rgi_id)
            self._save_progress(completed_ids)
            
            logger.info(f"  成功初始化: {len(gdirs)} 个")
        else:
            # 全部已完成，重新加载
            gdirs = workflow.init_glacier_directories(
                rgidf,
                from_prepro_level=None,
                prepro_border=80,
                reset=False
            )
        
        return gdirs
    
    def process_dem(self, gdirs: List) -> List:
        """处理DEM数据"""
        logger.info("处理DEM数据...")
        
        task_list = [
            tasks.define_glacier_region,
            tasks.glacier_masks,
        ]
        
        for task in task_list:
            logger.info(f"  执行: {task.__name__}")
            try:
                workflow.execute_entity_task(task, gdirs)
            except Exception as e:
                logger.warning(f"    部分失败: {str(e)[:100]}")
        
        return gdirs
    
    def compute_centerlines(self, gdirs: List) -> List:
        """计算中心线/flowlines (带进度条)"""
        logger.info("="*60)
        logger.info("计算Flowlines")
        logger.info("="*60)

        centerline_tasks = [
            ('compute_centerlines', tasks.compute_centerlines),
            ('initialize_flowlines', tasks.initialize_flowlines),
            ('compute_downstream_line', tasks.compute_downstream_line),
            ('compute_downstream_bedshape', tasks.compute_downstream_bedshape),  # 添加
            ('catchment_area', tasks.catchment_area),
            ('catchment_intersections', tasks.catchment_intersections),  # 添加
            ('catchment_width_geom', tasks.catchment_width_geom),
            ('catchment_width_correction', tasks.catchment_width_correction),
        ]
        for task_name, task_func in centerline_tasks:
            logger.info(f"执行: {task_name}")
            try:
                workflow.execute_entity_task(task_func, gdirs)
            except Exception as e:
                logger.warning(f"  部分失败: {str(e)[:100]}")
        
        # 统计成功率
        success_count = 0
        for gdir in gdirs:
            try:
                fls = gdir.read_pickle('inversion_flowlines')
                if fls:
                    success_count += 1
                    self.stats['success'] += 1
                else:
                    self.stats['failed'] += 1
                    self._log_error(gdir.rgi_id, 'flowline', 'Empty flowlines')
            except Exception as e:
                self.stats['failed'] += 1
                self._log_error(gdir.rgi_id, 'flowline', str(e))
        
        logger.info(f"Flowline生成完成: {success_count}/{len(gdirs)} 成功")
        
        return gdirs
    
    def run(
        self,
        outline_path: Union[str, Path],
        reset: bool = False,
        batch_size: int = None
    ) -> List:
        """运行完整流程
        
        Parameters
        ----------
        outline_path : Path
            冰川边界文件
        reset : bool
            是否重置
        batch_size : int, optional
            分批处理的批次大小。None表示不分批（默认行为）
        """
        self.stats['start_time'] = datetime.now().isoformat()
        
        # 如果指定了batch_size，使用分批处理
        if batch_size is not None:
            return self._run_batched(outline_path, reset, batch_size)
        
        # 原有逻辑
        try:
            # Step 1: 初始化gdirs
            gdirs = self.init_glacier_dirs(outline_path, reset)
            
            # Step 2: 处理DEM
            gdirs = self.process_dem(gdirs)
            
            # Step 3: 计算flowlines
            gdirs = self.compute_centerlines(gdirs)
            
        finally:
            self.stats['end_time'] = datetime.now().isoformat()
            self._save_errors()
            self._save_progress(set())  # 保存最终统计
        
        # 打印统计
        self._print_stats()
        
        return gdirs
    
    def _run_batched(
        self,
        outline_path: Union[str, Path],
        reset: bool = False,
        batch_size: int = 200
    ) -> List:
        """分批处理模式（内存受限时使用）"""
        import gc
        
        all_gdirs = []
        
        try:
            # 读取全部边界
            logger.info(f"读取边界文件: {outline_path}")
            rgidf = gpd.read_file(outline_path)
            total = len(rgidf)
            self.stats['total'] = total
            logger.info(f"  总冰川数: {total}")
            
            # 计算批次
            n_batches = (total + batch_size - 1) // batch_size
            logger.info(f"  分 {n_batches} 批处理，每批 {batch_size} 个")
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total)
                
                logger.info("="*60)
                logger.info(f"批次 {batch_idx+1}/{n_batches}: 冰川 {start_idx+1}-{end_idx}")
                logger.info("="*60)
                
                # 提取当前批次
                batch_df = rgidf.iloc[start_idx:end_idx].copy()
                
                # 保存临时文件
                batch_file = self.working_dir / f"batch_{batch_idx}.gpkg"
                batch_df.to_file(batch_file, driver='GPKG')
                
                # 处理当前批次
                self._check_oggm()
                self._init_config()
                
                # Step 1: 初始化gdirs
                gdirs = workflow.init_glacier_directories(
                    batch_df,
                    from_prepro_level=None,
                    prepro_border=80,
                    reset=reset if batch_idx == 0 else False
                )
                
                # Step 2: 处理DEM
                gdirs = self.process_dem(gdirs)
                
                # Step 3: 计算flowlines
                gdirs = self.compute_centerlines(gdirs)
                
                all_gdirs.extend(gdirs)
                
                # 清理内存
                del batch_df, gdirs
                gc.collect()
                
                # 删除临时文件
                if batch_file.exists():
                    batch_file.unlink()
                
                logger.info(f"批次 {batch_idx+1} 完成，累计成功: {self.stats['success']}")
            
        finally:
            self.stats['end_time'] = datetime.now().isoformat()
            self._save_errors()
            self._save_progress(set())
        
        self._print_stats()
        
        return all_gdirs
    
    def _print_stats(self):
        """打印统计信息"""
        logger.info("="*60)
        logger.info("处理统计:")
        logger.info(f"  - 总数: {self.stats['total']}")
        logger.info(f"  - 成功: {self.stats['success']}")
        logger.info(f"  - 失败: {self.stats['failed']}")
        logger.info(f"  - 跳过: {self.stats['skipped']}")
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            duration = (end - start).total_seconds()
            logger.info(f"  - 耗时: {duration/60:.1f} 分钟")
        if self.errors:
            logger.info(f"  - 错误日志: {self.error_log_path}")
        logger.info("="*60)


def compute_flowlines(
    outline_path: Union[str, Path],
    dem_path: Union[str, Path],
    working_dir: Union[str, Path],
    reset: bool = False,
    parallel: bool = True,
    n_workers: int = 4
) -> List:
    """
    完整的flowline生成流程 (简化接口)
    """
    processor = FlowlineProcessor(
        working_dir=working_dir,
        dem_path=dem_path,
        n_workers=n_workers if parallel else 1
    )
    
    return processor.run(outline_path, reset)


def extract_flowline_data(gdir) -> Dict[str, Any]:
    """从单个glacier directory提取flowline数据"""
    result = {
        'rgi_id': gdir.rgi_id,
        'success': False,
        'flowlines': [],
        'error': None
    }
    
    try:
        fls = gdir.read_pickle('inversion_flowlines')
        
        for i, fl in enumerate(fls):
            fl_data = {
                'index': i,
                'length_m': fl.length_m,
                'dx_m': fl.dx_meter,
                'n_points': fl.nx,
                'surface_h': fl.surface_h.copy(),
                'widths_m': fl.widths_m.copy(),
                'x_coords': fl.line.coords.xy[0] if hasattr(fl, 'line') else None,
                'y_coords': fl.line.coords.xy[1] if hasattr(fl, 'line') else None,
            }
            result['flowlines'].append(fl_data)
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def batch_extract_flowlines(
    gdirs: List,
    output_dir: Union[str, Path] = None,
    show_progress: bool = True,
    working_dir: Union[str, Path] = None
) -> gpd.GeoDataFrame:
    """批量提取所有冰川的flowline数据
    
    Parameters
    ----------
    gdirs : List
        Glacier directories列表
    output_dir : Path, optional
        输出目录
    show_progress : bool
        是否显示进度条
    working_dir : Path, optional
        如果gdirs为空，从此目录扫描已处理的冰川
    
    Returns
    -------
    gpd.GeoDataFrame
        包含所有flowlines的GeoDataFrame
    """
    from shapely.geometry import LineString
    
    logger.info("批量提取Flowline数据...")
    
    records = []
    success = 0
    failed = 0
    
    # 如果gdirs为空或全部失败，尝试从磁盘扫描
    if (not gdirs or len(gdirs) == 0) and working_dir:
        logger.info("从磁盘扫描已处理的冰川目录...")
        working_dir = Path(working_dir)
        per_glacier = working_dir / 'per_glacier'
        if per_glacier.exists():
            # 重新初始化gdirs
            from oggm import GlacierDirectory
            gdirs = []
            for region_dir in per_glacier.iterdir():
                if region_dir.is_dir():
                    for subdir in region_dir.iterdir():
                        if subdir.is_dir():
                            for gdir_path in subdir.iterdir():
                                if gdir_path.is_dir():
                                    try:
                                        gdir = GlacierDirectory(gdir_path.name, base_dir=str(working_dir))
                                        gdirs.append(gdir)
                                    except:
                                        pass
            logger.info(f"  从磁盘加载了 {len(gdirs)} 个glacier directories")
    
    if not gdirs:
        logger.warning("没有可用的glacier directories!")
        return gpd.GeoDataFrame()
    
    if show_progress and HAS_TQDM:
        iterator = tqdm(gdirs, desc="提取Flowlines", unit="冰川", ncols=80)
    else:
        iterator = gdirs
    
    for gdir in iterator:
        data = extract_flowline_data(gdir)
        
        if not data['success']:
            failed += 1
            continue
        
        success += 1
        for fl in data['flowlines']:
            if fl['x_coords'] is not None and fl['y_coords'] is not None:
                coords = list(zip(fl['x_coords'], fl['y_coords']))
                if len(coords) >= 2:
                    record = {
                        'RGIId': data['rgi_id'],
                        'fl_index': fl['index'],
                        'length_m': fl['length_m'],
                        'n_points': fl['n_points'],
                        'mean_width': np.mean(fl['widths_m']),
                        'min_elev': np.min(fl['surface_h']),
                        'max_elev': np.max(fl['surface_h']),
                        'geometry': LineString(coords)
                    }
                    records.append(record)
    
    logger.info(f"  成功: {success}, 失败: {failed}")
    
    if not records:
        logger.warning("没有成功提取任何flowline!")
        return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame(records)
    
    if gdirs:
        try:
            gdf.crs = gdirs[0].grid.proj
        except:
            pass
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "flowlines.gpkg"
        gdf.to_file(output_file, driver='GPKG')
        logger.info(f"  保存至: {output_file}")
    
    return gdf


# 保持向后兼容的函数
def init_glacier_dirs(
    outline_path: Union[str, Path],
    working_dir: Union[str, Path],
    dem_path: Union[str, Path],
    reset: bool = False
) -> List:
    """向后兼容接口"""
    processor = FlowlineProcessor(working_dir, dem_path)
    return processor.init_glacier_dirs(outline_path, reset)


def check_oggm():
    """检查OGGM是否可用"""
    if not OGGM_AVAILABLE:
        raise ImportError("OGGM未安装")
    logger.info(f"OGGM版本: {oggm.__version__}")
    return True


def init_oggm_config(
    working_dir: Union[str, Path],
    dem_path: Union[str, Path],
    use_multiprocessing: bool = True,
    n_workers: int = None,
    use_rgi_area: bool = False
) -> Path:
    """初始化OGGM配置 (向后兼容)"""
    check_oggm()
    
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    
    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = str(working_dir)
    cfg.PATHS['dem_file'] = str(dem_path)
    cfg.PARAMS['use_rgi_area'] = use_rgi_area
    cfg.PARAMS['use_multiprocessing'] = use_multiprocessing
    if n_workers:
        cfg.PARAMS['mp_processes'] = n_workers
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['continue_on_error'] = True
    cfg.PARAMS['use_intersects'] = True
    
    return working_dir
