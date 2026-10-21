"""
诊断脚本: 检查流线和冰川边界的ID匹配关系
"""

import geopandas as gpd
import pandas as pd

# 路径
FLOWLINES_PATH = "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_connected.shp"
GLACIER_PATH = "/mnt/d/paleoice_output/arcgis_export_v6/glacier_success.shp"

print("=" * 70)
print("ID字段诊断")
print("=" * 70)

# 加载流线
print("\n1. 流线数据")
flowlines_gdf = gpd.read_file(FLOWLINES_PATH)
print(f"   总数: {len(flowlines_gdf)}")
print(f"   字段: {list(flowlines_gdf.columns)}")
print(f"\n   前5个rgi_id示例:")
unique_rgi = flowlines_gdf['rgi_id'].unique()[:5]
for rid in unique_rgi:
    print(f"   - {rid} (类型: {type(rid).__name__})")

# 加载冰川边界
print("\n2. 冰川边界数据")
glacier_gdf = gpd.read_file(GLACIER_PATH)
print(f"   总数: {len(glacier_gdf)}")
print(f"   字段: {list(glacier_gdf.columns)}")

# 检查ID字段
id_fields = [col for col in glacier_gdf.columns if 'RGI' in col.upper() or 'ID' in col.upper()]
print(f"\n   可能的ID字段: {id_fields}")

for field in id_fields:
    print(f"\n   {field} 示例:")
    print(f"   - {list(glacier_gdf[field].unique()[:5])}")

# 尝试匹配
print("\n3. 匹配测试")
flowline_ids = set(flowlines_gdf['rgi_id'].unique())
print(f"   流线中唯一ID数: {len(flowline_ids)}")

for field in id_fields:
    glacier_ids = set(glacier_gdf[field].dropna().unique())
    print(f"\n   {field}:")
    print(f"   - 冰川中唯一ID数: {len(glacier_ids)}")

    # 交集
    intersection = flowline_ids & glacier_ids
    print(f"   - 直接匹配数: {len(intersection)}")

    if len(intersection) > 0:
        print(f"   - ✓ 可以直接匹配!")
        print(f"   - 匹配示例: {list(intersection)[:3]}")
    else:
        # 检查是否只是格式差异
        # 示例: "RGI60-15.00001" vs "RGI60-15.00001" (可能有空格或前缀)
        flowline_sample = list(flowline_ids)[0]
        glacier_sample = list(glacier_ids)[0]
        print(f"   - ✗ 无直接匹配")
        print(f"   - 流线ID示例: '{flowline_sample}' (长度{len(flowline_sample)})")
        print(f"   - 冰川ID示例: '{glacier_sample}' (长度{len(glacier_sample)})")

# 检查几何匹配
print("\n4. 几何匹配测试")
print("   检查流线是否在冰川边界内...")

sample_flowline = flowlines_gdf.iloc[0]
fl_geom = sample_flowline.geometry
fl_centroid = fl_geom.centroid
fl_rgi_id = sample_flowline['rgi_id']

print(f"   测试流线: {fl_rgi_id}")
print(f"   流线中心点: ({fl_centroid.x:.6f}, {fl_centroid.y:.6f})")

# 找包含该中心点的冰川
containing_glaciers = glacier_gdf[glacier_gdf.contains(fl_centroid)]

if len(containing_glaciers) > 0:
    print(f"   ✓ 找到包含该流线的冰川:")
    for idx, row in containing_glaciers.iterrows():
        for field in id_fields:
            if pd.notna(row[field]):
                print(f"      {field}: {row[field]}")
else:
    # 找最近的冰川
    distances = glacier_gdf.geometry.distance(fl_centroid)
    nearest_idx = distances.idxmin()
    nearest_glacier = glacier_gdf.loc[nearest_idx]

    print(f"   ✗ 流线不在任何冰川内")
    print(f"   最近的冰川 (距离{distances[nearest_idx]:.1f}度):")
    for field in id_fields:
        if pd.notna(nearest_glacier[field]):
            print(f"      {field}: {nearest_glacier[field]}")

print("\n" + "=" * 70)
print("建议:")
print("1. 如果有直接匹配的字段 → 使用该字段")
print("2. 如果ID格式不同 → 需要建立映射表")
print("3. 如果完全不匹配 → 使用几何关系匹配")
print("=" * 70)