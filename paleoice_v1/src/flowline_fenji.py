import geopandas as gpd
from shapely.geometry import Point

flowlines = gpd.read_file("/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_id.shp")
flowlines_utm = flowlines.to_crs(epsg=32644)
flowlines_utm['len_m'] = flowlines_utm.geometry.length

# 初始化字段
flowlines_utm['is_main_new'] = 0
flowlines_utm['level'] = -1
flowlines_utm['parent'] = -1

# 按冰川分组处理
glaciers = flowlines_utm['MainRGI'].unique()
print(f"共 {len(glaciers)} 个冰川")

for i, glacier_id in enumerate(glaciers):
    fl = flowlines_utm[flowlines_utm['MainRGI'] == glacier_id]

    if len(fl) == 0:
        continue

    # 主流线：最长
    main_idx = fl['len_m'].idxmax()
    main_fl_index = fl.loc[main_idx, 'fl_index']

    flowlines_utm.loc[main_idx, 'is_main_new'] = 1
    flowlines_utm.loc[main_idx, 'level'] = 0
    flowlines_utm.loc[main_idx, 'parent'] = -1

    # 分析支流汇入
    for idx, row in fl.iterrows():
        if idx == main_idx:
            continue

        end_pt = Point(list(row.geometry.coords)[-1])

        min_dist = float('inf')
        parent_fl = -1

        for idx2, row2 in fl.iterrows():
            if idx == idx2 or row2['len_m'] <= row['len_m']:
                continue

            d = row2.geometry.distance(end_pt)
            if d < min_dist and d < 2500:
                min_dist = d
                parent_fl = row2['fl_index']

        flowlines_utm.loc[idx, 'parent'] = parent_fl

    # 确定层级
    for idx, row in fl.iterrows():
        if idx == main_idx:
            continue

        parent = flowlines_utm.loc[idx, 'parent']
        if parent == main_fl_index:
            flowlines_utm.loc[idx, 'level'] = 1
        elif parent != -1:
            parent_row = fl[fl['fl_index'] == parent]
            if len(parent_row) > 0 and parent_row.iloc[0]['fl_index'] == main_fl_index:
                flowlines_utm.loc[idx, 'level'] = 2
            else:
                flowlines_utm.loc[idx, 'level'] = 2
        else:
            flowlines_utm.loc[idx, 'level'] = -1  # 孤立

    if (i + 1) % 100 == 0:
        print(f"已处理 {i + 1}/{len(glaciers)}")

# 保存
flowlines_out = flowlines_utm.to_crs(epsg=4326)
flowlines_out.to_file("/mnt/d/paleoice_output/arcgis_export_v6/flowlines_with_hierarchy.shp")
print(f"\n已保存: flowlines_with_hierarchy.shp")
print(f"新增字段: len_m, is_main_new, level, parent")