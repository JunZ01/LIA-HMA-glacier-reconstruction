"""
OGGM流线连接器 - 适配版
将支流延伸连接到主流线

输入：extract_lia_flowlines输出的shapefile
输出：带连接线的完整流线网络
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
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
        """
        翻转流线方向，确保从上游(高海拔)到下游(低海拔)
        """
        coords = list(line_utm.coords)
        start_elev = self.get_elevation_at_point(coords[0][0], coords[0][1], utm_crs)
        end_elev = self.get_elevation_at_point(coords[-1][0], coords[-1][1], utm_crs)

        # 如果起点海拔低于终点，则翻转
        if not np.isnan(start_elev) and not np.isnan(end_elev):
            if start_elev < end_elev:
                return LineString(coords[::-1])

        return line_utm

    def trace_downslope_to_parent(self, start_x, start_y, parent_line, utm_crs, max_steps=500):
        """
        从支流末端沿坡度下降追踪到父流线

        Parameters
        ----------
        start_x, start_y : float
            支流末端坐标(UTM)
        parent_line : LineString
            父流线几何(UTM)
        utm_crs : CRS
            UTM坐标系
        max_steps : int
            最大追踪步数

        Returns
        -------
        LineString : 连接路径
        str : 状态 ('success', 'stuck', 'max_steps')
        """
        transformer = Transformer.from_crs(utm_crs, CRS.from_epsg(4326), always_xy=True)

        # 8个搜索方向
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

                # 检查是否已到达父流线
                current_pt = Point(x, y)
                dist_to_parent = parent_line.distance(current_pt)

                if dist_to_parent < self.step_size * 2:
                    # 找到最近点
                    nearest_pt = parent_line.interpolate(parent_line.project(current_pt))
                    path.append((nearest_pt.x, nearest_pt.y))
                    return LineString(path), "success"

                # 获取当前高程
                lon, lat = transformer.transform(x, y)
                try:
                    row, col = src.index(lon, lat)
                    current_elev = src.read(1, window=((row, row + 1), (col, col + 1)))[0, 0]
                except:
                    break

                if current_elev < -9999:  # 无效值
                    break

                # 计算目标方向(朝向父流线)
                nearest_on_parent = parent_line.interpolate(parent_line.project(current_pt))
                target_dx = nearest_on_parent.x - x
                target_dy = nearest_on_parent.y - y
                target_dist = np.sqrt(target_dx ** 2 + target_dy ** 2)

                if target_dist > 0:
                    target_dx /= target_dist
                    target_dy /= target_dist

                # 搜索8个方向
                best_score = float('inf')
                best_x, best_y = None, None

                for dx, dy in directions:
                    nx = x + dx * self.step_size
                    ny = y + dy * self.step_size

                    # 避免回头
                    if (round(nx, 0), round(ny, 0)) in visited:
                        continue

                    # 获取新位置高程
                    lon_n, lat_n = transformer.transform(nx, ny)
                    try:
                        row_n, col_n = src.index(lon_n, lat_n)
                        if 0 <= row_n < src.height and 0 <= col_n < src.width:
                            elev_n = src.read(1, window=((row_n, row_n + 1), (col_n, col_n + 1)))[0, 0]

                            if elev_n < -9999:
                                continue

                            # 评分系统
                            # 1. 方向一致性
                            norm = np.sqrt(dx ** 2 + dy ** 2)
                            dir_score = -(dx / norm * target_dx + dy / norm * target_dy)

                            # 2. 高程变化(上坡重罚，下坡奖励)
                            elev_diff = elev_n - current_elev
                            if elev_diff > 0:
                                elev_penalty = elev_diff * 5
                            else:
                                elev_penalty = elev_diff * 0.5

                            # 3. 距离父流线
                            dist_n = parent_line.distance(Point(nx, ny))

                            # 综合得分(越小越好)
                            score = elev_penalty + dist_n * 0.3 + dir_score * 50

                            if score < best_score:
                                best_score = score
                                best_x, best_y = nx, ny
                    except:
                        continue

                # 如果找不到更好的点，直接连到父流线
                if best_x is None:
                    nearest_pt = parent_line.interpolate(parent_line.project(current_pt))
                    path.append((nearest_pt.x, nearest_pt.y))
                    return LineString(path), f"stuck_at_step_{step}"

                x, y = best_x, best_y
                path.append((x, y))

        # 超过最大步数，强制连接
        current_pt = Point(x, y)
        nearest_pt = parent_line.interpolate(parent_line.project(current_pt))
        path.append((nearest_pt.x, nearest_pt.y))
        return LineString(path), "max_steps"

    def connect_flowlines_for_glacier(self, flowlines_gdf_subset, rgi_id):
        """
        为单个冰川连接流线网络

        Parameters
        ----------
        flowlines_gdf_subset : GeoDataFrame
            单个冰川的所有流线 (WGS84坐标)
        rgi_id : str
            冰川RGI ID

        Returns
        -------
        GeoDataFrame : 包含原始流线和连接线的完整网络(WGS84坐标)
        dict : 连接统计信息
        """
        print(f"\n处理冰川: {rgi_id}")
        print(f"  流线数: {len(flowlines_gdf_subset)}")

        # 获取中心点确定UTM区
        centroid = flowlines_gdf_subset.union_all().centroid
        utm_crs = self.get_utm_crs(centroid.x, centroid.y)
        src_crs = CRS.from_epsg(4326)

        # 转换到UTM
        flowlines_utm = {}
        flowlines_info = {}

        for idx, row in flowlines_gdf_subset.iterrows():
            geom = row.geometry

            # 处理MultiLineString
            if geom.geom_type == 'MultiLineString':
                geom = max(list(geom.geoms), key=lambda x: x.length)

            # 转UTM
            geom_utm = self.transform_geometry(geom, src_crs, utm_crs)

            # 翻转方向(上游→下游)
            geom_utm = self.flip_line_to_upstream(geom_utm, utm_crs)

            fl_id = row['fl_id']
            flowlines_utm[fl_id] = geom_utm

            # 存储层级信息
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

        # 识别主流线和支流
        main_flowlines = [fl_id for fl_id, info in flowlines_info.items() if info['is_main']]
        tributary_flowlines = [fl_id for fl_id, info in flowlines_info.items() if not info['is_main']]

        print(f"  主流线: {len(main_flowlines)}, 支流: {len(tributary_flowlines)}")

        # 连接支流到父流线
        connection_lines = []
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

            # 支流末端(下游端点)
            trib_line = flowlines_utm[trib_id]
            trib_end = Point(trib_line.coords[-1])

            # 父流线
            parent_line = flowlines_utm[parent_id]

            # 检查距离
            dist_to_parent = parent_line.distance(trib_end)

            if dist_to_parent > self.search_distance:
                print(f"    FL{trib_id}→{parent_id}: 距离{dist_to_parent:.0f}m > {self.search_distance}m")
                connection_stats['too_far'] += 1
                continue

            # 追踪连接
            conn_line, status = self.trace_downslope_to_parent(
                trib_end.x, trib_end.y,
                parent_line, utm_crs
            )

            connection_lines.append({
                'geometry_utm': conn_line,
                'child_fl_id': trib_id,
                'parent_fl_id': parent_id,
                'status': status,
                'direct_distance': dist_to_parent,
                'path_length': conn_line.length
            })

            # 统计
            if status == 'success':
                connection_stats['success'] += 1
                print(f"    FL{trib_id}→{parent_id}: ✓ {dist_to_parent:.0f}m → {conn_line.length:.0f}m")
            elif 'stuck' in status:
                connection_stats['stuck'] += 1
                print(f"    FL{trib_id}→{parent_id}: ✗ stuck")
            else:
                connection_stats['max_steps'] += 1
                print(f"    FL{trib_id}→{parent_id}: ✗ max_steps")

        # 转回WGS84并组合
        result_features = []

        # 添加原始流线
        for fl_id, geom_utm in flowlines_utm.items():
            geom_wgs84 = self.transform_geometry(geom_utm, utm_crs, src_crs)
            info = flowlines_info[fl_id]

            result_features.append({
                'geometry': geom_wgs84,
                'rgi_id': rgi_id,
                'fl_id': fl_id,
                'is_main': info['is_main'],
                'order': info['order'],
                'flows_to': info['flows_to'],
                'type': 'flowline'
            })

        # 添加连接线
        for conn in connection_lines:
            geom_wgs84 = self.transform_geometry(conn['geometry_utm'], utm_crs, src_crs)

            result_features.append({
                'geometry': geom_wgs84,
                'rgi_id': rgi_id,
                'fl_id': conn['child_fl_id'],
                'is_main': False,
                'order': -1,  # 标记为连接线
                'flows_to': conn['parent_fl_id'],
                'type': 'connection',
                'status': conn['status'],
                'path_length': conn['path_length']
            })

        result_gdf = gpd.GeoDataFrame(result_features, crs="EPSG:4326")

        return result_gdf, connection_stats


def connect_all_flowlines(flowlines_path, dem_path, output_path,
                          search_distance=2500, step_size=30):
    """
    连接所有冰川的流线网络

    Parameters
    ----------
    flowlines_path : str
        OGGM输出的流线shapefile路径
    dem_path : str
        DEM路径
    output_path : str
        输出shapefile路径
    search_distance : float
        最大搜索距离(米)
    step_size : float
        追踪步长(米)
    """
    print("=" * 60)
    print("OGGM流线网络连接器")
    print("=" * 60)

    # 读取流线
    flowlines_gdf = gpd.read_file(flowlines_path)
    print(f"\n总流线数: {len(flowlines_gdf)}")

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

        # 获取该冰川的所有流线
        glacier_flowlines = flowlines_gdf[flowlines_gdf['rgi_id'] == rgi_id]

        # 连接
        result_gdf, stats = connector.connect_flowlines_for_glacier(
            glacier_flowlines, rgi_id
        )

        all_results.append(result_gdf)

        # 累计统计
        for key in global_stats:
            global_stats[key] += stats.get(key, 0)

    # 合并所有结果
    final_gdf = gpd.GeoDataFrame(pd.concat(all_results, ignore_index=True), crs="EPSG:4326")

    # 保存
    final_gdf.to_file(output_path)

    print(f"\n{'=' * 60}")
    print("全局统计:")
    print(f"  成功连接: {global_stats['success']}")
    print(f"  卡住: {global_stats['stuck']}")
    print(f"  超步数: {global_stats['max_steps']}")
    print(f"  距离过远: {global_stats['too_far']}")
    print(f"  父流线缺失: {global_stats['parent_not_found']}")
    print(f"\n✓ 保存至: {output_path}")
    print("=" * 60)

    return final_gdf, global_stats


if __name__ == "__main__":
    # 配置参数
    FLOWLINES_INPUT = "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_smooth.shp"
    DEM_PATH = "/mnt/d/01 文件整理/04 数据/HMA_newregion/DEM_clip/Himalaya_3regions_DEM.tif"
    OUTPUT_PATH = "/mnt/f/oggm_related/oggm_LIA_flowlines/output/lia_flowlines_connected.shp"

    # 执行连接
    result_gdf, stats = connect_all_flowlines(
        flowlines_path=FLOWLINES_INPUT,
        dem_path=DEM_PATH,
        output_path=OUTPUT_PATH,
        search_distance=1500,  # 最大搜索距离1.5km
        step_size=30  # 追踪步长30m
    )