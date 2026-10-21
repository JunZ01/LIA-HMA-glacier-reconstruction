"""
prepare.py - 数据准备模块 (优化版)
==================================

将LIA冰川边界转换为OGGM兼容格式

优化功能:
1. 进度条显示
2. 坐标系自动转换
3. 详细日志
4. 内存优化批处理
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, box
from pathlib import Path
import logging
import warnings
from typing import Union, Optional, Tuple, List

# 抑制shapely警告
warnings.filterwarnings('ignore', category=RuntimeWarning, module='shapely')
warnings.filterwarnings('ignore', message='invalid value encountered')

# 进度条
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_inputs(
    lia_path: Union[str, Path],
    dem_path: Union[str, Path],
    rgi_path: Optional[Union[str, Path]] = None
) -> dict:
    """
    验证输入数据的有效性 (修复坐标系检测)
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    # 1. 检查LIA边界
    logger.info(f"验证LIA边界: {lia_path}")
    try:
        lia_gdf = gpd.read_file(lia_path)
        result['info']['lia'] = {
            'count': len(lia_gdf),
            'crs': str(lia_gdf.crs),
            'bounds': lia_gdf.total_bounds.tolist(),
            'columns': list(lia_gdf.columns)
        }
        
        # 检查几何有效性
        invalid_geoms = (~lia_gdf.geometry.is_valid).sum()
        if invalid_geoms > 0:
            result['warnings'].append(f"LIA有 {invalid_geoms} 个无效几何 (将自动修复)")
            
        logger.info(f"  - 冰川数量: {len(lia_gdf)}")
        logger.info(f"  - 坐标系: {lia_gdf.crs}")
        
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"读取LIA边界失败: {str(e)}")
        return result
    
    # 2. 检查DEM
    logger.info(f"验证DEM: {dem_path}")
    try:
        with rasterio.open(dem_path) as src:
            result['info']['dem'] = {
                'crs': str(src.crs),
                'resolution': src.res,
                'bounds': list(src.bounds),
                'shape': src.shape,
                'nodata': src.nodata
            }
            logger.info(f"  - 分辨率: {src.res}")
            logger.info(f"  - 坐标系: {src.crs}")
            logger.info(f"  - 尺寸: {src.shape}")
            dem_crs = src.crs
            dem_bounds = box(*src.bounds)
            
    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"读取DEM失败: {str(e)}")
        return result
    
    # 3. 检查坐标系一致性并计算覆盖率 (修复版)
    lia_crs = lia_gdf.crs
    
    # 关键修复: 先转换坐标系再计算覆盖率
    if lia_crs != dem_crs:
        result['warnings'].append(
            f"坐标系不一致: LIA={lia_crs.to_epsg() or 'custom'}, DEM={dem_crs.to_epsg() or 'custom'}. 将自动转换."
        )
        # 转换LIA到DEM坐标系后再计算覆盖
        lia_gdf_reproj = lia_gdf.to_crs(dem_crs)
        lia_bounds = box(*lia_gdf_reproj.total_bounds)
    else:
        lia_bounds = box(*lia_gdf.total_bounds)
    
    # 计算覆盖率
    if not dem_bounds.intersects(lia_bounds):
        result['warnings'].append("警告: DEM与LIA边界不相交，请检查数据")
    else:
        intersection_area = dem_bounds.intersection(lia_bounds).area
        lia_area = lia_bounds.area
        if lia_area > 0:
            coverage = intersection_area / lia_area * 100
            if coverage < 100:
                result['warnings'].append(f"DEM覆盖LIA边界: {coverage:.1f}%")
            logger.info(f"  - DEM覆盖率: {coverage:.1f}%")
    
    return result


def prepare_lia_for_oggm(
    lia_path: Union[str, Path],
    dem_path: Union[str, Path],
    output_path: Union[str, Path],
    rgi_path: Optional[Union[str, Path]] = None,
    id_field: str = 'UniqueID',
    rgi_field: str = 'MainRGI',
    area_field: str = 'LIAArea',
    min_area_km2: float = 0.01,
    fix_geometry: bool = True,
    show_progress: bool = True
) -> gpd.GeoDataFrame:
    """
    将LIA边界转换为OGGM兼容格式 (优化版)
    """
    logger.info("="*60)
    logger.info("开始准备LIA数据用于OGGM")
    logger.info("="*60)
    
    # 1. 读取LIA边界
    logger.info(f"读取LIA边界: {lia_path}")
    lia_gdf = gpd.read_file(lia_path)
    original_count = len(lia_gdf)
    logger.info(f"  原始冰川数量: {original_count}")
    
    # 2. 修复几何
    if fix_geometry:
        logger.info("修复无效几何...")
        invalid_mask = ~lia_gdf.geometry.is_valid
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            lia_gdf.loc[invalid_mask, 'geometry'] = lia_gdf.loc[invalid_mask, 'geometry'].buffer(0)
            logger.info(f"  修复了 {invalid_count} 个无效几何")
        
        # 移除空几何
        empty_mask = lia_gdf.geometry.is_empty | lia_gdf.geometry.isna()
        if empty_mask.sum() > 0:
            logger.info(f"  移除 {empty_mask.sum()} 个空几何")
            lia_gdf = lia_gdf[~empty_mask].copy()
    
    # 3. 过滤小冰川
    if area_field in lia_gdf.columns:
        before_filter = len(lia_gdf)
        lia_gdf = lia_gdf[lia_gdf[area_field] >= min_area_km2].copy()
        logger.info(f"  过滤面积 < {min_area_km2} km²: {before_filter} -> {len(lia_gdf)}")
    
    # 重置索引
    lia_gdf = lia_gdf.reset_index(drop=True)
    
    # 4. 构建OGGM所需字段
    logger.info("构建OGGM所需字段...")
    
    # RGIId - 使用唯一索引生成ID
    lia_gdf['RGIId'] = [f"LIA60-14.{i:05d}" for i in range(len(lia_gdf))]
    
    # 保留原始RGI关联信息
    if rgi_field in lia_gdf.columns:
        lia_gdf['OrigRGIId'] = lia_gdf[rgi_field]
    
    # 保留原始ID
    if id_field in lia_gdf.columns:
        lia_gdf['OriginalID'] = lia_gdf[id_field]
    
    # 5. 中心点坐标
    logger.info("  计算中心点坐标...")
    centroids = lia_gdf.geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids, crs=lia_gdf.crs).to_crs("EPSG:4326")
    lia_gdf['CenLon'] = centroids_wgs84.x
    lia_gdf['CenLat'] = centroids_wgs84.y
    
    # 6. 面积计算
    if area_field in lia_gdf.columns:
        lia_gdf['Area'] = lia_gdf[area_field]
    else:
        logger.info("  计算面积...")
        if lia_gdf.crs.is_geographic:
            utm_zone = int((lia_gdf['CenLon'].mean() + 180) / 6) + 1
            hemisphere = 'north' if lia_gdf['CenLat'].mean() >= 0 else 'south'
            utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"
            lia_utm = lia_gdf.to_crs(utm_crs)
            lia_gdf['Area'] = lia_utm.geometry.area / 1e6
        else:
            lia_gdf['Area'] = lia_gdf.geometry.area / 1e6
    
    # 7. 提取高程统计
    logger.info("  提取高程统计 (Zmin, Zmax, Zmed)...")
    lia_gdf = _extract_elevation_stats(lia_gdf, dem_path, show_progress=show_progress)
    
    # 8. 补充其他OGGM必需字段
    logger.info("  补充OGGM字段...")
    lia_gdf['GLIMSId'] = lia_gdf['RGIId'].apply(lambda x: f"G{x.split('.')[-1]}E00000N")
    lia_gdf['BgnDate'] = '19000101'
    lia_gdf['EndDate'] = '19000101'
    lia_gdf['O1Region'] = '14'
    lia_gdf['O2Region'] = '3'
    lia_gdf['GlacType'] = '0000'
    lia_gdf['Status'] = 0
    lia_gdf['Connect'] = 0
    lia_gdf['Form'] = 0
    lia_gdf['TermType'] = 0
    lia_gdf['Surging'] = 9
    lia_gdf['Linkages'] = 9
    lia_gdf['Name'] = ''
    
    # 9. 计算几何属性
    logger.info("  计算几何属性...")
    if lia_gdf.crs.is_geographic:
        utm_zone = int((lia_gdf['CenLon'].mean() + 180) / 6) + 1
        hemisphere = 'north' if lia_gdf['CenLat'].mean() >= 0 else 'south'
        utm_crs = f"+proj=utm +zone={utm_zone} +{hemisphere} +datum=WGS84"
        lia_utm = lia_gdf.to_crs(utm_crs)
    else:
        lia_utm = lia_gdf
    
    lia_gdf['Lmax'] = lia_utm.geometry.apply(
        lambda g: int(max(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1])) if g and not g.is_empty else 0
    )
    lia_gdf['Slope'] = 20.0
    lia_gdf['Aspect'] = 0
    
    # 10. 转换到WGS84 (OGGM需要)
    logger.info("  转换坐标系到WGS84...")
    if lia_gdf.crs.to_epsg() != 4326:
        lia_gdf = lia_gdf.to_crs("EPSG:4326")
    
    # 11. 选择输出字段
    output_columns = [
        'RGIId', 'GLIMSId', 'BgnDate', 'EndDate',
        'CenLon', 'CenLat', 'O1Region', 'O2Region',
        'Area', 'Zmin', 'Zmax', 'Zmed', 'Slope', 'Aspect', 'Lmax',
        'Status', 'Connect', 'Form', 'TermType', 'Surging', 'Linkages', 'Name',
        'geometry'
    ]
    
    if 'OriginalID' in lia_gdf.columns:
        output_columns.insert(1, 'OriginalID')
    if 'OrigRGIId' in lia_gdf.columns:
        output_columns.insert(2, 'OrigRGIId')
    
    output_columns = [c for c in output_columns if c in lia_gdf.columns]
    lia_gdf_out = lia_gdf[output_columns].copy()
    
    # 12. 移除高程提取失败的冰川
    valid_mask = lia_gdf_out['Zmin'].notna() & lia_gdf_out['Zmax'].notna()
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"  移除 {invalid_count} 个高程提取失败的冰川")
        lia_gdf_out = lia_gdf_out[valid_mask].copy()
    
    # 13. 保存输出
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"保存输出: {output_path}")
    if output_path.suffix.lower() == '.gpkg':
        lia_gdf_out.to_file(output_path, driver='GPKG')
    else:
        lia_gdf_out.to_file(output_path)
    
    # 14. 打印统计
    logger.info("="*60)
    logger.info(f"完成! 统计信息:")
    logger.info(f"  - 原始冰川: {original_count}")
    logger.info(f"  - 输出冰川: {len(lia_gdf_out)}")
    logger.info(f"  - 面积范围: {lia_gdf_out['Area'].min():.2f} - {lia_gdf_out['Area'].max():.2f} km²")
    logger.info(f"  - 高程范围: {lia_gdf_out['Zmin'].min():.0f} - {lia_gdf_out['Zmax'].max():.0f} m")
    logger.info(f"  - 输出文件: {output_path}")
    logger.info("="*60)
    
    return lia_gdf_out


def _extract_elevation_stats(
    gdf: gpd.GeoDataFrame, 
    dem_path: Union[str, Path],
    batch_size: int = 100,
    show_progress: bool = True
) -> gpd.GeoDataFrame:
    """从DEM提取高程统计 (优化版)"""
    gdf = gdf.copy()
    gdf['Zmin'] = np.nan
    gdf['Zmax'] = np.nan
    gdf['Zmed'] = np.nan
    
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        nodata = src.nodata
        
        # 转换坐标系以匹配DEM
        if gdf.crs != dem_crs:
            logger.info(f"    转换坐标系: {gdf.crs.to_epsg() or 'custom'} -> {dem_crs.to_epsg() or 'custom'}")
            gdf_reproj = gdf.to_crs(dem_crs)
        else:
            gdf_reproj = gdf
        
        total = len(gdf)
        success_count = 0
        fail_count = 0
        
        # 进度条
        if show_progress and HAS_TQDM:
            iterator = tqdm(range(total), desc="    提取高程", unit="冰川", ncols=80)
        else:
            iterator = range(total)
        
        for i in iterator:
            try:
                geom = gdf_reproj.geometry.iloc[i]
                
                if geom is None or geom.is_empty:
                    fail_count += 1
                    continue
                
                if not box(*src.bounds).intersects(geom):
                    fail_count += 1
                    continue
                
                out_image, _ = mask(src, [geom], crop=True, all_touched=True)
                data = out_image[0]
                
                if nodata is not None:
                    data = data[data != nodata]
                
                data = data[(data > -500) & (data < 9000)]
                
                if len(data) > 0:
                    gdf.loc[gdf.index[i], 'Zmin'] = int(np.nanmin(data))
                    gdf.loc[gdf.index[i], 'Zmax'] = int(np.nanmax(data))
                    gdf.loc[gdf.index[i], 'Zmed'] = int(np.nanmedian(data))
                    success_count += 1
                else:
                    fail_count += 1
                    
            except Exception:
                fail_count += 1
                continue
        
        logger.info(f"    高程提取完成: 成功 {success_count}, 失败 {fail_count}")
    
    return gdf


def subset_by_region(
    gdf: gpd.GeoDataFrame,
    region_bounds: Tuple[float, float, float, float] = None,
    region_polygon: gpd.GeoDataFrame = None,
    n_sample: int = None
) -> gpd.GeoDataFrame:
    """按区域或数量子集化数据"""
    if region_bounds is not None:
        minx, miny, maxx, maxy = region_bounds
        gdf = gdf.cx[minx:maxx, miny:maxy]
    
    if region_polygon is not None:
        gdf = gpd.sjoin(gdf, region_polygon, how='inner', predicate='within')
    
    if n_sample is not None and len(gdf) > n_sample:
        gdf = gdf.sample(n=n_sample, random_state=42)
    
    return gdf.copy()
