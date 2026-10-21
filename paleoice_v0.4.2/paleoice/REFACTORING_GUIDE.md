# 基岩DEM生成代码重构说明

## 概述

本文档说明了将原始 `bare_dem_himalaya.ipynb` Jupyter notebook 重构为符合 PaleoIce 风格的模块化 Python 代码的过程。

---

## 原始代码问题分析

原始notebook存在以下问题：

1. **代码结构混乱** - 所有代码都在一个notebook中，缺乏模块化
2. **逻辑不清晰** - 处理步骤交织在一起，难以理解整体流程
3. **缺乏配置管理** - 路径和参数硬编码在代码中
4. **错误处理不完善** - 部分失败冰川的ID分散在多处
5. **缺乏文档和类型注解** - 代码可读性差
6. **不符合PaleoIce风格** - 与项目其他模块风格不一致

---

## 重构后的代码架构

```
bare_dem_himalaya.py
│
├── BareDEMConfig          # 配置类
│   ├── 输入/输出路径
│   ├── 筛选参数
│   ├── OGGM参数
│   └── validate()         # 配置验证
│
├── DataLoader             # 数据加载器
│   ├── load_lia_boundaries()
│   ├── load_rgi_glaciers()
│   ├── match_lia_rgi()
│   └── load_mass_balance_data()
│
├── 几何处理函数
│   ├── clean_geometries()
│   └── ensure_wgs84()
│
├── OGGMProcessor          # OGGM处理器
│   ├── initialize()
│   ├── init_glacier_directories()
│   ├── run_preprocessing_tasks()
│   ├── calibrate_mass_balance()
│   ├── run_inversion_tasks()
│   └── get_valid_gdirs()
│
├── BareDEMGenerator       # 基岩DEM生成器
│   ├── generate()
│   └── save()
│
├── FlowlineExporter       # Flowline导出器
│   └── extract_and_save()
│
└── BareDEMWorkflow        # 完整工作流
    └── run()
```

---

## 处理流程对比

### 原始代码流程 (分散在多个cell中)

```
Cell 1: 读取LIA和RGI，筛选LIA≥10km²
Cell 2: 读取MB数据，清理几何
Cell 3: OGGM初始化
Cell 4: CRU气候数据 (后来改为W5E5)
Cell 5: 各种参数设置
Cell 6: define_glacier_region
Cell 7: 预处理任务链
Cell 8: 气候数据处理
Cell 9: MB校准 + 反演任务链
...
Cell N: 生成bare_dem
```

### 重构后流程 (清晰的8步)

```
Step 1: 加载数据
    ├── 加载LIA边界，筛选≥10km²
    ├── 加载RGI边界
    ├── 匹配LIA-RGI关联
    └── 加载Hugonnet MB数据

Step 2: 数据预处理
    ├── 坐标系转换为WGS84
    ├── 清理无效几何
    └── 筛选有MB数据的冰川

Step 3: OGGM初始化
    ├── 配置工作目录、DEM路径
    ├── 设置参数 (border, workers等)
    └── 初始化glacier directories

Step 4: 预处理任务
    ├── define_glacier_region (USER DEM)
    ├── glacier_masks
    ├── compute_centerlines
    ├── initialize_flowlines
    ├── compute_downstream_line
    ├── compute_downstream_bedshape
    ├── catchment_area/intersections
    ├── catchment_width_geom/correction
    └── process_climate_data

Step 5: 物质平衡校准
    └── mb_calibration_from_scalar_mb
        (使用Hugonnet 2000-2020数据)

Step 6: 冰厚反演
    ├── apparent_mb_from_any_mb
    ├── prepare_for_inversion
    ├── mass_conservation_inversion
    ├── filter_inversion_output
    └── distribute_thickness_per_altitude

Step 7: 生成基岩DEM
    ├── 读取原始DEM
    ├── 逐冰川提取冰厚
    ├── 重投影到DEM坐标系
    ├── 累积冰厚 (取最大值)
    ├── 计算 bare_dem = surface_dem - thickness
    └── 保存结果

Step 8: 导出Flowlines (可选)
    ├── 提取flowline坐标
    ├── 转换为WGS84
    └── 保存为GPKG和SHP
```

---

## PaleoIce风格特征

重构后的代码遵循PaleoIce项目的以下风格特征：

### 1. 模块化设计

```python
# 每个功能封装为独立的类
class DataLoader:
    """数据加载器"""
    
class OGGMProcessor:
    """OGGM处理器"""
    
class BareDEMGenerator:
    """基岩DEM生成器"""
```

### 2. 配置类集中管理

```python
class BareDEMConfig:
    """集中管理所有配置"""
    def __init__(self, lia_path, rgi_path, dem_path, ...):
        ...
    def validate(self) -> Dict:
        """验证配置有效性"""
```

### 3. 完整的类型注解

```python
def load_mass_balance_data(self) -> pd.DataFrame:
    """
    加载Hugonnet物质平衡数据
    
    Returns
    -------
    pd.DataFrame
        处理后的物质平衡数据，以rgiid为索引
    """
```

### 4. 日志和进度显示

```python
# 统一的日志格式
logger.info("="*60)
logger.info("执行预处理任务")
logger.info("="*60)

# tqdm进度条
iterator = tqdm(gdirs, desc="MB校准", ncols=80)
```

### 5. 错误处理和统计

```python
self.stats = {
    'total': 0,
    'success': 0,
    'failed': [],
    'no_thickness': [],
    'start_time': None,
    'end_time': None,
}
```

### 6. 命令行接口

```python
def main():
    parser = argparse.ArgumentParser(...)
    parser.add_argument('--reset', ...)
    parser.add_argument('--workers', ...)
    ...
```

---

## 使用方法

### 方式1: 作为模块导入

```python
from bare_dem_himalaya import BareDEMConfig, BareDEMWorkflow

config = BareDEMConfig(
    lia_path="path/to/lia.shp",
    rgi_path="path/to/rgi.shp",
    dem_path="path/to/dem.tif",
    mb_paths=["path/to/mb1.csv", "path/to/mb2.csv"],
    working_dir="path/to/working",
    output_dir="path/to/output",
    n_workers=16,
)

workflow = BareDEMWorkflow(config)
output_path = workflow.run(reset=False, export_flowlines=True)
```

### 方式2: 命令行运行

```bash
# 默认配置运行
python bare_dem_himalaya.py

# 重置工作目录
python bare_dem_himalaya.py --reset

# 指定并行进程数
python bare_dem_himalaya.py --workers 8

# 不导出flowlines
python bare_dem_himalaya.py --no-flowlines
```

### 方式3: 单独使用组件

```python
from bare_dem_himalaya import (
    BareDEMConfig, DataLoader, OGGMProcessor, 
    BareDEMGenerator, clean_geometries
)

# 仅加载数据
config = BareDEMConfig(...)
loader = DataLoader(config)
lia = loader.load_lia_boundaries()
rgi = loader.load_rgi_glaciers()

# 仅执行OGGM处理
processor = OGGMProcessor(config)
processor.initialize()
processor.init_glacier_directories(rgi)
processor.run_preprocessing_tasks()
...

# 仅生成基岩DEM
generator = BareDEMGenerator(config)
bare_dem, thickness = generator.generate(valid_gdirs)
generator.save(bare_dem, thickness)
```

---

## 与原始代码的主要差异

| 方面 | 原始代码 | 重构后代码 |
|------|---------|-----------|
| 结构 | 单一notebook | 模块化类和函数 |
| 配置 | 硬编码 | BareDEMConfig类 |
| 日志 | print语句 | logging模块 |
| 进度 | 部分有tqdm | 统一tqdm封装 |
| 错误处理 | 分散的try-except | 集中的stats统计 |
| 类型 | 无类型注解 | 完整类型注解 |
| 文档 | 少量注释 | 完整docstring |
| 运行方式 | 仅notebook | 支持CLI和模块导入 |

---

## 输出文件

重构后的代码生成以下输出：

```
output_dir/
├── Himalaya_bare_DEM.tif     # 基岩DEM
├── Himalaya_thickness.tif    # 冰厚数据
├── flowlines.gpkg            # Flowlines (GeoPackage)
└── flowlines.shp             # Flowlines (Shapefile)
```

---

## 注意事项

1. **DEM源设置**: 代码使用 `source='USER'` 确保使用本地DEM而非在线数据
2. **坐标系**: 所有GIS数据统一转换为WGS84 (EPSG:4326)
3. **失败处理**: 失败的冰川ID被记录在stats中，不影响其他冰川处理
4. **内存管理**: 大规模处理时建议适当降低n_workers或分批处理
5. **气候数据**: 默认使用W5E5气候数据，可通过配置修改

---

## 扩展建议

1. 添加分批处理功能以支持更大规模数据
2. 添加断点续传功能
3. 添加结果验证模块
4. 添加可视化输出功能
