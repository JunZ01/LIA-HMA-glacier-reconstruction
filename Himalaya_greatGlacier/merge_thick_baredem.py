"""
只运行合并步骤 - 从已有OGGM结果生成bare DEM（分块处理版）
"""
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.transform import from_bounds
from rasterio.windows import Window
from pathlib import Path
import xarray as xr
import tempfile
import shutil
import math
from tqdm import tqdm
import gc

# ============================================================
# 配置
# ============================================================
working_dir = Path("/mnt/f/oggm_related/oggm_LIA_himalaya_great")
dem_path = Path("/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif")
output_dir = working_dir / "output"
output_dir.mkdir(exist_ok=True)

# 分块大小（越大越快，但内存占用越高）
# 8192 x 8192 x 4字节 x 3数组 ≈ 800MB
CHUNK_SIZE = 8192

# ============================================================
# Step 1: 找到所有已处理的冰川目录
# ============================================================
print("查找已处理的冰川...")

per_glacier_dir = working_dir / "per_glacier"
glacier_dirs = []

for region_dir in per_glacier_dir.iterdir():
    if not region_dir.is_dir() or not region_dir.name.startswith('RGI'):
        continue
    for sub_dir in region_dir.iterdir():
        if not sub_dir.is_dir():
            continue
        for gdir in sub_dir.iterdir():
            if gdir.is_dir():
                gridded_file = gdir / "gridded_data.nc"
                if gridded_file.exists():
                    glacier_dirs.append(gdir)

print(f"找到 {len(glacier_dirs)} 个冰川目录")

# ============================================================
# Step 2: 读取DEM元数据
# ============================================================
print(f"读取DEM元数据: {dem_path}")

with rasterio.open(dem_path) as src:
    dem_crs = src.crs
    dem_transform = src.transform
    dem_height = src.height
    dem_width = src.width
    dem_nodata = src.nodata if src.nodata is not None else -9999.0
    dem_profile = src.profile.copy()

print(f"  尺寸: {dem_height} x {dem_width}, CRS: {dem_crs}")

# ============================================================
# Step 3: 导出每个冰川的冰厚tif
# ============================================================
print("导出冰厚tif...")

temp_dir = Path(tempfile.mkdtemp(prefix="glacier_thickness_"))
print(f"  临时目录: {temp_dir}")

thickness_files = []
success = 0
failed = 0

for gdir in tqdm(glacier_dirs, desc="导出冰厚"):
    try:
        gridded_file = gdir / "gridded_data.nc"

        with xr.open_dataset(gridded_file) as ds:
            if 'distributed_thickness' not in ds:
                failed += 1
                continue

            thick = ds['distributed_thickness'].values.copy()
            x = ds.x.values
            y = ds.y.values

            # 读取投影信息
            if 'pyproj_srs' in ds.attrs:
                src_crs = ds.attrs['pyproj_srs']
            else:
                failed += 1
                continue

        thick = np.where(np.isfinite(thick) & (thick > 0), thick, 0)

        if np.sum(thick > 0) == 0:
            failed += 1
            continue

        # 源变换
        dx = abs(x[1] - x[0])
        dy = abs(y[0] - y[1])
        src_transform = from_bounds(
            x.min() - dx/2, y.min() - dy/2,
            x.max() + dx/2, y.max() + dy/2,
            len(x), len(y)
        )

        # 计算目标变换
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dem_crs, len(x), len(y),
            left=x.min() - dx/2,
            bottom=y.min() - dy/2,
            right=x.max() + dx/2,
            top=y.max() + dy/2
        )

        # 重投影
        dst_array = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=thick.astype(np.float32),
            destination=dst_array,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0
        )

        # 保存
        out_path = temp_dir / f"{gdir.name}_thickness.tif"
        profile = {
            'driver': 'GTiff',
            'dtype': np.float32,
            'width': dst_width,
            'height': dst_height,
            'count': 1,
            'crs': dem_crs,
            'transform': dst_transform,
            'nodata': 0,
        }

        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(dst_array, 1)

        thickness_files.append(out_path)
        success += 1

    except Exception as e:
        failed += 1
        continue

print(f"  成功: {success}, 失败: {failed}")

# ============================================================
# Step 4: 合并冰厚
# ============================================================
print(f"合并 {len(thickness_files)} 个冰厚文件...")

src_files = [rasterio.open(f) for f in thickness_files]

try:
    merged_data, merged_transform = merge(src_files, method='max')
    merged_crs = src_files[0].crs
finally:
    for src in src_files:
        src.close()

merged_data = merged_data[0]
print(f"  合并后尺寸: {merged_data.shape}")

# 保存临时合并文件
merged_path = temp_dir / "merged_thickness.tif"
merged_profile = {
    'driver': 'GTiff',
    'dtype': np.float32,
    'width': merged_data.shape[1],
    'height': merged_data.shape[0],
    'count': 1,
    'crs': merged_crs,
    'transform': merged_transform,
    'nodata': 0,
    'compress': 'lzw',
}

with rasterio.open(merged_path, 'w', **merged_profile) as dst:
    dst.write(merged_data.astype(np.float32), 1)

del merged_data
gc.collect()
print("  合并完成")

# ============================================================
# Step 5: 重采样冰厚到DEM网格（写入临时文件）
# ============================================================
print("重采样冰厚到DEM网格...")

resampled_thick_path = temp_dir / "resampled_thickness.tif"

resample_profile = dem_profile.copy()
resample_profile.update(
    dtype=np.float32,
    nodata=0,
    compress='lzw',
    tiled=True,
    blockxsize=512,
    blockysize=512
)

with rasterio.open(merged_path) as src_thick:
    with rasterio.open(resampled_thick_path, 'w', **resample_profile) as dst_thick:
        reproject(
            source=rasterio.band(src_thick, 1),
            destination=rasterio.band(dst_thick, 1),
            src_transform=src_thick.transform,
            src_crs=src_thick.crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear,
            src_nodata=0,
            dst_nodata=0
        )

print("  重采样完成")

# ============================================================
# Step 6: 分块计算并保存基岩DEM
# ============================================================
print("分块计算基岩DEM...")

n_rows = math.ceil(dem_height / CHUNK_SIZE)
n_cols = math.ceil(dem_width / CHUNK_SIZE)
total_chunks = n_rows * n_cols

print(f"  DEM尺寸: {dem_height} x {dem_width}")
print(f"  分块大小: {CHUNK_SIZE} x {CHUNK_SIZE}")
print(f"  分块数: {n_rows} x {n_cols} = {total_chunks}")

# 输出文件配置
output_profile = dem_profile.copy()
output_profile.update(
    dtype=np.float32,
    compress='lzw',
    tiled=True,
    blockxsize=512,
    blockysize=512
)

bare_dem_path = output_dir / "Himalaya_bare_DEM.tif"
thickness_out_path = output_dir / "Himalaya_thickness.tif"

thick_out_profile = output_profile.copy()
thick_out_profile['nodata'] = 0

# 统计变量
ice_pixels = 0
max_thickness = 0.0
thickness_sum = 0.0
thickness_count = 0

with rasterio.open(dem_path) as dem_src, \
     rasterio.open(resampled_thick_path) as thick_src, \
     rasterio.open(bare_dem_path, 'w', **output_profile) as bare_dst, \
     rasterio.open(thickness_out_path, 'w', **thick_out_profile) as thick_dst:

    for chunk_idx in tqdm(range(total_chunks), desc="分块处理"):
        row_idx = chunk_idx // n_cols
        col_idx = chunk_idx % n_cols

        row_start = row_idx * CHUNK_SIZE
        col_start = col_idx * CHUNK_SIZE
        row_end = min(row_start + CHUNK_SIZE, dem_height)
        col_end = min(col_start + CHUNK_SIZE, dem_width)

        window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

        # 读取数据块
        dem_chunk = dem_src.read(1, window=window).astype(np.float32)
        thick_chunk = thick_src.read(1, window=window).astype(np.float32)

        # 计算掩膜
        dem_valid = (dem_chunk != dem_nodata) & np.isfinite(dem_chunk)
        has_ice = thick_chunk > 0
        ice_mask = dem_valid & has_ice

        # 计算基岩DEM
        bare_chunk = dem_chunk.copy()
        bare_chunk[ice_mask] = dem_chunk[ice_mask] - thick_chunk[ice_mask]

        # 写入输出
        bare_dst.write(bare_chunk, 1, window=window)
        thick_dst.write(thick_chunk, 1, window=window)

        # 更新统计
        ice_pixels += int(np.sum(ice_mask))
        if np.any(has_ice):
            max_thickness = max(max_thickness, float(np.max(thick_chunk[has_ice])))
            thickness_sum += float(np.sum(thick_chunk[has_ice]))
            thickness_count += int(np.sum(has_ice))

# 输出统计
mean_thickness = thickness_sum / thickness_count if thickness_count > 0 else 0

print(f"\n统计结果:")
print(f"  冰川覆盖像素: {ice_pixels:,}")
print(f"  最大冰厚: {max_thickness:.1f} m")
print(f"  平均冰厚: {mean_thickness:.1f} m")

# ============================================================
# Step 7: 清理临时文件
# ============================================================
print("\n清理临时文件...")
shutil.rmtree(temp_dir, ignore_errors=True)

print(f"\n完成!")
print(f"  基岩DEM: {bare_dem_path}")
print(f"  冰厚: {thickness_out_path}")