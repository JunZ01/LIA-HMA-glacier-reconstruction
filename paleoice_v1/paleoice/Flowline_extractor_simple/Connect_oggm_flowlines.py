"""
OGGM流线连接器 - 适配版（修复ID映射 v2）
将支流延伸连接到主流线

修复内容：
1. 保留所有原始属性（UniqueID, MainRGI, region, LIAArea 等）
2. 支持 GeoPackage 格式

输入：extract_lia_flowlines输出的shapefile/gpkg
输出：带连接线的完整流线网络
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import transform
from pyproj import CRS, Transformer
import warnings

warnings.filterwarnings('ignore')


class OGGMFlowlineConnector:
    """OGGM流线连接器"""

    def __init__(self, dem_path, search_distance=2500, step_size=30):
        self.dem_path = dem_path
        self.search_distance = search_distance
        self.step_size = step_size

    def get_utm_crs(self, lon, lat):
        """获取UTM坐标系"""
        zone = int((lon + 180) / 6) + 1
        if lat >= 0:
            return CRS.from_epsg(32600 + zone)
        else:
            return CRS.from_epsg(32700 + zone)

    def transform_geometry(self, geom, src_crs, dst_crs):
        """坐标转换"""
        transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
        return transform(transformer.transform, geom)

    def get_elevation_at_point(self, x_utm, y_utm, utm_crs):
        """获取UTM坐标点的高程"""
        transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)
        lon, lat = transformer.transform(x_utm, y_utm)

        with rasterio.open(self.dem_path) as src:
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < src.height and 0 <= col < src.width:
                    return src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
            except:
                pass
        return np.nan

    def flip_line_to_upstream(self, line_utm, utm_crs):
        """翻转流线方向，确保从上游(高海拔)到下游(低海拔)"""
        coords = list(line_utm.coords)
        start_elev = self.get_elevation_at_point(coords[0][0], coords[0][1], utm_crs)
        end_elev = self.get_elevation_at_point(coords[-1][0], coords[-1][1], utm_crs)

        if not np.isnan(start_elev) and not np.isnan(end_elev):
            if start_elev < end_elev:
                return LineString(coords[::-1])

        return line_utm

    def trace_downslope_to_parent(self, start_x, start_y, parent_line, utm_crs, max_steps=500):
        """从支流末端沿坡度下降追踪到父流线"""
        transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

        directions = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        path = [(start_x, start_y)]
        x, y = start_x, start_y
        visited = set()

        with rasterio.open(self.dem_path) as src:
            for step in range(max_steps):
                visited.add((round(x, 0), round(y, 0)))

                current_pt = Point(x, y)
                dist_to_parent = parent_line.distance(current_pt)

                if dist_to_parent < self.step_size * 2:
                    nearest_pt = parent_line.interpolate(parent_line.project(current_pt))
                    path.append((nearest_pt.x, nearest_pt.y))
                    return LineString(path), "success"

                lon, lat = transformer.transform(x, y)
                try:
                    row, col = src.index(lon, lat)
                    current_elev = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                except:
                    break

                if current_elev < -9999:
                    break

                nearest_on_parent = parent_line.interpolate(parent_line.project(current_pt))
                target_dx = nearest_on_parent.x - x
                target_dy = nearest_on_parent.y - y
                target_dist = np.sqrt(target_dx ** 2 + target_dy ** 2)

                if target_dist > 0:
                    target_dx /= target_dist
                    target_dy /= target_dist

                best_score = float('inf')
                best_x, best_y = None, None

                for dx, dy in directions:
                    nx = x + dx * self.step_size
                    ny = y + dy * self.step_size

                    if (round(nx, 0), round(ny, 0)) in visited:
                        continue

                    lon_n, lat_n = transformer.transform(nx, ny)
                    try:
                        row_n, col_n = src.index(lon_n, lat_n)
                        if 0 <= row_n < src.height and 0 <= col_n < src.width:
                            elev_n = src.read(1, window=((row_n, row_n + 1), (col_n, col_n + 1)))[0, 0]

                            if elev_n < -9999:
                                continue

                            norm = np.sqrt(dx ** 2 + dy ** 2)
                            dir_score = -(dx / norm * target_dx + dy / norm * target_dy)

                            elev_diff = elev_n - current_elev
                            if elev_diff > 0:
                                elev_penalty = elev_diff * 5
                            else:
                                elev_penalty = elev_diff * 0.5

                            dist_n = parent_line.distance(Point(nx, ny))
                            score = elev_penalty + dist_n * 0.3 + dir_score * 50

                            if score < best_score:
                                best_score = score
                                best_x, best_y = nx, ny
                    except:
                        continue

                if best_x is None:
                    nearest_pt = parent_line.interpolate(parent_line.project(current_pt))
                    path.append((nearest_pt.x, nearest_pt.y))
                    return LineString(path), f"stuck_at_step_{step}"

                x, y = best_x, best_y
                path.append((x, y))

        current_pt = Point(x, y)
        nearest_pt = parent_line.interpolate(parent_line.project(current_pt))
        path.append((nearest_pt.x, nearest_pt.y))
        return LineString(path), "max_steps"

    def connect_flowlines_for_glacier(self, flowlines_gdf_subset, rgi_id):
        """为单个冰川连接流线网络"""
        print(f"\n处理冰川: {rgi_id}")
        print(f"  流线数: {len(flowlines_gdf_subset)}")

        # ★★★ 获取该冰川的所有原始属性（从第一条流线获取，因为同一冰川属性相同）★★★
        first_row = flowlines_gdf_subset.iloc[0]

        # 识别原始属性列（排除流线特有的列）
        flowline_specific_cols = ['geometry', 'rgi_id', 'fl_id', 'is_main', 'order', 'flows_to',
                                   'length_m', 'type', 'conn_len', 'conn_status', 'direct_dist']
        glacier_attrs = {}
        for col in flowlines_gdf_subset.columns:
            if col not in flowline_specific_cols:
                glacier_attrs[col] = first_row[col]

        print(f"  保留属性: {list(glacier_attrs.keys())}")

        # 获取中心点确定UTM区
        centroid = flowlines_gdf_subset.union_all().centroid
        utm_crs = self.get_utm_crs(centroid.x, centroid.y)
        src_crs = CRS.from_epsg(4326)

        # 转换到UTM
        flowlines_utm = {}
        flowlines_info = {}

        for idx, row in flowlines_gdf_subset.iterrows():
            geom = row.geometry

            if geom.geom_type == 'MultiLineString':
                geom = max(list(geom.geoms), key=lambda x: x.length)

            geom_utm = self.transform_geometry(geom, src_crs, utm_crs)
            geom_utm = self.flip_line_to_upstream(geom_utm, utm_crs)

            fl_id = row['fl_id']
            flowlines_utm[fl_id] = geom_utm

            flows_to = row['flows_to']
            if pd.isna(flows_to) or flows_to == -1:
                flows_to = None
            else:
                flows_to = int(flows_to)

            flowlines_info[fl_id] = {
                'is_main': row['is_main'],
                'order': row['order'],
                'flows_to': flows_to
            }

        print(f"  有效流线: {len(flowlines_utm)}")

        main_flowlines = [fl_id for fl_id, info in flowlines_info.items() if info['is_main']]
        tributary_flowlines = [fl_id for fl_id, info in flowlines_info.items() if not info['is_main']]

        print(f"  主流线: {len(main_flowlines)}, 支流: {len(tributary_flowlines)}")

        connected_tributaries = {}
        connection_stats = {
            'success': 0,
            'stuck': 0,
            'max_steps': 0,
            'too_far': 0,
            'parent_not_found': 0
        }

        for trib_id in tributary_flowlines:
            parent_id = flowlines_info[trib_id]['flows_to']

            if parent_id is None:
                print(f"    FL{trib_id}: 无父流线")
                connection_stats['parent_not_found'] += 1
                continue

            if parent_id not in flowlines_utm:
                print(f"    FL{trib_id}: 父流线{parent_id}不存在")
                connection_stats['parent_not_found'] += 1
                continue

            trib_line = flowlines_utm[trib_id]
            trib_end = Point(trib_line.coords[-1])
            parent_line = flowlines_utm[parent_id]

            dist_to_parent = parent_line.distance(trib_end)

            if dist_to_parent > self.search_distance:
                print(f"    FL{trib_id}→{parent_id}: 距离{dist_to_parent:.0f}m > {self.search_distance}m")
                connection_stats['too_far'] += 1
                continue

            conn_line, status = self.trace_downslope_to_parent(
                trib_end.x, trib_end.y,
                parent_line, utm_crs
            )

            coords_combined = list(trib_line.coords) + list(conn_line.coords)[1:]
            connected_line = LineString(coords_combined)

            connected_tributaries[trib_id] = {
                'geometry_utm': connected_line,
                'status': status,
                'connection_length': conn_line.length,
                'direct_distance': dist_to_parent
            }

            if status == 'success':
                connection_stats['success'] += 1
                print(f"    FL{trib_id}→{parent_id}: ✓ {dist_to_parent:.0f}m → 延长{conn_line.length:.0f}m")
            elif 'stuck' in status:
                connection_stats['stuck'] += 1
                print(f"    FL{trib_id}→{parent_id}: ✗ stuck (延长{conn_line.length:.0f}m)")
            else:
                connection_stats['max_steps'] += 1
                print(f"    FL{trib_id}→{parent_id}: ✗ max_steps (延长{conn_line.length:.0f}m)")

        # 转回WGS84并组合
        result_features = []

        for fl_id, geom_utm in flowlines_utm.items():
            if fl_id in connected_tributaries:
                geom_utm = connected_tributaries[fl_id]['geometry_utm']
                conn_length = connected_tributaries[fl_id]['connection_length']
                conn_status = connected_tributaries[fl_id]['status']
                direct_dist = connected_tributaries[fl_id]['direct_distance']
            else:
                conn_length = 0
                conn_status = 'no_connection'
                direct_dist = 0

            geom_wgs84 = self.transform_geometry(geom_utm, utm_crs, src_crs)
            info = flowlines_info[fl_id]

            # ★★★ 构建记录，包含所有原始属性 ★★★
            record = {
                'geometry': geom_wgs84,
                'rgi_id': rgi_id,
                'fl_id': fl_id,
                'is_main': info['is_main'],
                'order': info['order'],
                'flows_to': info['flows_to'],
                'type': 'flowline',
                'conn_len': conn_length,
                'conn_status': conn_status,
                'direct_dist': direct_dist
            }

            # ★★★ 添加所有原始属性 ★★★
            for attr_name, attr_value in glacier_attrs.items():
                record[attr_name] = attr_value

            result_features.append(record)

        result_gdf = gpd.GeoDataFrame(result_features, crs="EPSG:4326")

        return result_gdf, connection_stats


def connect_all_flowlines(flowlines_path, dem_path, output_path,
                          search_distance=2500, step_size=30):
    """连接所有冰川的流线网络"""
    print("=" * 60)
    print("OGGM流线网络连接器 (v2)")
    print("=" * 60)

    # 读取流线（支持 shp 和 gpkg）
    flowlines_gdf = gpd.read_file(flowlines_path)
    print(f"\n总流线数: {len(flowlines_gdf)}")
    print(f"输入列: {list(flowlines_gdf.columns)}")

    # ★★★ 检查输入数据的属性列 ★★★
    if 'UniqueID' in flowlines_gdf.columns:
        unique_ids = flowlines_gdf['UniqueID'].dropna().unique()
        print(f"UniqueID 数量: {len(unique_ids)}")
        print(f"UniqueID 示例: {sorted(unique_ids)[:5]}")
    else:
        print("警告: 输入数据不包含 UniqueID 列")

    # 创建连接器
    connector = OGGMFlowlineConnector(
        dem_path=dem_path,
        search_distance=search_distance,
        step_size=step_size
    )

    # 按冰川分组处理
    all_results = []
    global_stats = {
        'success': 0,
        'stuck': 0,
        'max_steps': 0,
        'too_far': 0,
        'parent_not_found': 0
    }

    glacier_ids = flowlines_gdf['rgi_id'].unique()
    print(f"冰川数: {len(glacier_ids)}\n")

    for i, rgi_id in enumerate(glacier_ids, 1):
        print(f"[{i}/{len(glacier_ids)}] ", end='')

        glacier_flowlines = flowlines_gdf[flowlines_gdf['rgi_id'] == rgi_id]

        result_gdf, stats = connector.connect_flowlines_for_glacier(
            glacier_flowlines, rgi_id
        )

        all_results.append(result_gdf)

        for key in global_stats:
            global_stats[key] += stats.get(key, 0)

    # 合并所有结果
    final_gdf = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=True), crs="EPSG:4326")

    # 保存为 GeoPackage（推荐）和 Shapefile
    output_gpkg = output_path.replace('.shp', '.gpkg')
    final_gdf.to_file(output_gpkg, driver='GPKG')
    final_gdf.to_file(output_path)

    print(f"\n{'=' * 60}")
    print("全局统计:")
    print(f"  成功连接: {global_stats['success']}")
    print(f"  卡住: {global_stats['stuck']}")
    print(f"  超步数: {global_stats['max_steps']}")
    print(f"  距离过远: {global_stats['too_far']}")
    print(f"  父流线缺失: {global_stats['parent_not_found']}")

    # ★★★ 输出属性统计 ★★★
    print(f"\n输出列: {list(final_gdf.columns)}")
    if 'UniqueID' in final_gdf.columns:
        unique_ids = final_gdf['UniqueID'].dropna().unique()
        print(f"UniqueID 数量: {len(unique_ids)}")
        print(f"UniqueID 示例: {sorted(unique_ids)[:10]}")

    print(f"\n✓ 保存至: {output_path}")
    print(f"✓ 保存至: {output_gpkg}")
    print("=" * 60)

    return final_gdf, global_stats


if __name__ == "__main__":
    # 配置参数 - 优先使用 gpkg 作为输入
    FLOWLINES_INPUT = "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_smooth.gpkg"
    DEM_PATH = "/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif"
    OUTPUT_PATH = "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_connected.shp"

    # 如果 gpkg 不存在，使用 shp
    import os
    if not os.path.exists(FLOWLINES_INPUT):
        FLOWLINES_INPUT = FLOWLINES_INPUT.replace('.gpkg', '.shp')
        print(f"使用 shapefile: {FLOWLINES_INPUT}")

    # 执行连接
    result_gdf, stats = connect_all_flowlines(
        flowlines_path=FLOWLINES_INPUT,
        dem_path=DEM_PATH,
        output_path=OUTPUT_PATH,
        search_distance=1500,
        step_size=30
    )