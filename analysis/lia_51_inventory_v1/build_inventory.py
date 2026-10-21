"""Read-only source analysis: 32 retained systems + 19 legacy-review candidates."""
from pathlib import Path
import gzip
import hashlib
import html
import json
import pickle
import sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Geod, Transformer
from scipy.spatial import cKDTree
from shapely.geometry import LineString, mapping

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'validation_data/results/paleo_surface_validation_v0'
MODERN = ROOT / 'Himalaya_greatGlacier/With_Rec_climate_organized/outputs/019f730c-ee32-74d0-8b77-fdb7832f6a06/s1_turnover_f19_full_v0/oggm_workdir/per_glacier'
GEOD = Geod(ellps='WGS84')
SOURCES = {}


def source(path):
    path = Path(path)
    key = str(path.relative_to(ROOT)).replace('\\', '/')
    if key not in SOURCES:
        SOURCES[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return key


def read_json(path):
    source(path)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    source(path)
    return pd.read_csv(path)


def read_pickle(path):
    source(path)
    with Path(path).open('rb') as f:
        compressed = f.read(2) == b'\x1f\x8b'
    with (gzip.open(path, 'rb') if compressed else Path(path).open('rb')) as f:
        return pickle.load(f)


def gdir(rgi):
    return MODERN / rgi[:8] / rgi[:11] / rgi


def orientation(xy, crs):
    line = LineString(xy)
    tr = Transformer.from_crs(crs, 4326, always_xy=True)
    lon, lat = tr.transform(xy[:, 0], xy[:, 1])
    overall, _, straight = GEOD.inv(lon[0], lat[0], lon[-1], lat[-1])
    reach = min(1000., line.length * .1)
    near = line.interpolate(line.length - reach)
    near_lon, near_lat = tr.transform(near.x, near.y)
    terminal, _, _ = GEOD.inv(near_lon, near_lat, lon[-1], lat[-1])
    return dict(length_km=line.length / 1000, head_lon=float(lon[0]), head_lat=float(lat[0]),
                terminus_lon=float(lon[-1]), terminus_lat=float(lat[-1]),
                azimuth_deg=overall % 360, terminal_azimuth_deg=terminal % 360,
                terminal_reach_m=reach, sinuosity=line.length / straight), np.column_stack([lon, lat])


def num(value):
    try:
        v = float(value)
        return v if np.isfinite(v) and v > -9999 else None
    except (ValueError, TypeError):
        return None


def run():
    retained = read_json(BASE / 'lia_accepted32_research26_v1/summary.json')
    search = read_json(BASE / 'lia_direct_continuous_v1/completion-local-v1.json')
    kept = set(retained['retained_system_ids'])
    candidates = set(search['combined_candidate_systems'])
    assert len(kept) == 32 and len(candidates) == 19 and not kept & candidates
    targets = read_csv(BASE / 'lia_design_b_targets_and_inputs_v1/lia_design_b_system_targets.csv').set_index('lia_unique_id')
    members = read_csv(BASE / 'lia_design_b_targets_and_inputs_v1/lia_design_b_input_audit.csv').set_index('lia_unique_id')
    reg = read_csv(BASE / 'lia_accepted32_research26_v1/retained_32_registry.csv').set_index('system_id')
    gallery = {int(g['system']): g for g in read_json(BASE / 'lia_new38_gallery_v1/data.json')}
    rows, features, qa = [], [], []
    for sid in sorted(kept | candidates):
        t = targets.loc[sid]
        rgi = str(t.rgi_id)
        state = '已认可（32条）' if sid in kept else '旧复核候选（19条）'
        ref_lia = float(reg.loc[sid, 'lia_km']) if sid in kept else gallery[sid]['target_lia_km']
        ref_modern = float(reg.loc[sid, 'rgi_km']) if sid in kept else gallery[sid]['target_modern_km']
        row = dict(lia_unique_id=sid, status=state, main_rgi_id=rgi,
                   lia_reference_length_km=ref_lia, modern_reference_length_km=ref_modern,
                   reference_length_difference_km=ref_lia-ref_modern,
                   lia_registered_target_year=int(t.lia_target_year_ce))
        # Select the component by the frozen outlet crosswalk, then its stored main branch.
        nets = []
        for p in sorted((BASE / 'lia_gdir_anchor_reroute_core191_v3' / str(sid)).glob('component_*/lia_centerline_network.pkl.gz')):
            n = read_pickle(p)
            branch = [b for b in n['branches'] if b['branch_index'] == n['main_branch_index']]
            assert len(branch) == 1, (sid, 'ambiguous stored main')
            xy = np.asarray(branch[0]['coords_projected_m'], float)
            d = np.linalg.norm(xy[-1] - [t.target_outlet_x_m, t.target_outlet_y_m])
            nets.append((d, p, n, branch[0], xy))
        matched = [n for n in nets if n[0] < 1.0]
        if len(matched) != 1:
            raise ValueError((sid, 'registered outlet component crosswalk', [(a[0], str(a[1])) for a in nets]))
        distance, p, n, branch, xy = matched[0]
        metrics, ll = orientation(xy, n['crs'])
        row.update({'lia_' + k: v for k, v in metrics.items()})
        row.update(lia_component=p.parent.name, lia_main_branch=int(n['main_branch_index']),
                   lia_network_source=source(p), lia_outlet_crosswalk_error_m=float(distance))
        stored_elev = np.asarray(branch['reference_surface_m'], float)
        features.extend([
            dict(type='Feature', properties=dict(system=sid, status=state, kind='LIA_mainline_v3'), geometry=dict(type='LineString', coordinates=ll.tolist())),
            dict(type='Feature', properties=dict(system=sid, status=state, kind='LIA_mainline_terminus_v3'), geometry=dict(type='Point', coordinates=ll[-1].tolist()))])
        grids = list((BASE / 'core191_anchor_reroute_f19_v3').glob(f'lia_{sid}_*_system_grid_v1.nc'))
        assert len(grids) == 1
        gp = grids[0]
        source(gp)
        with xr.open_dataset(gp) as ds:
            mi = list(ds.member_id.values.astype(str)).index('farinotti2019')
            h = ds.thickness_m.isel(member=mi).values
            z = ds.surface_elevation_m.isel(member=mi).values
            bed = ds.bed_elevation_m.isel(member=mi).values
            lia_mask = ds.lia_mask.values > 0
            valid = (ds.valid_mask.isel(member=mi).values > 0) & lia_mask & np.isfinite(h) & np.isfinite(z) & np.isfinite(bed)
            ice = valid & (h > 1e-6)
            area = abs(float(np.median(np.diff(ds.x.values))) * float(np.median(np.diff(ds.y.values))))
            yy, xx = np.where(ice)
            tr = Transformer.from_crs(ds.spatial_ref.attrs['crs_wkt'], 4326, always_xy=True)
            clon, clat = tr.transform(float(ds.x.values[xx].mean()), float(ds.y.values[yy].mean()))
            row.update(lia_source_grid=source(gp), lia_positive_ice_area_km2=float(ice.sum()*area/1e6),
                       lia_reconstructed_volume_km3=float(h[ice].sum()*area/1e9),
                       lia_mean_thickness_m=float(h[ice].mean()), lia_max_thickness_m=float(h[ice].max()),
                       lia_surface_min_m=float(z[ice].min()), lia_surface_max_m=float(z[ice].max()),
                       lia_surface_mean_m=float(z[ice].mean()), lia_surface_median_m=float(np.median(z[ice])),
                       lia_positive_ice_centroid_lon=clon, lia_positive_ice_centroid_lat=clat,
                       lia_outline_mask_area_km2=float(lia_mask.sum()*area/1e6),
                       lia_positive_coverage_fraction=float(ice.sum()/lia_mask.sum()))
            # The stored reference_surface array is not consistently aligned with the
            # rerouted V3 coordinates. Read elevations from the paired V3 raster instead.
            vy, vx = np.where(valid)
            tree = cKDTree(np.column_stack([ds.x.values[vx], ds.y.values[vy]]))
            distances, indices = tree.query(xy[[0, -1]])
            sampled_z = z[vy[indices], vx[indices]].astype(float)
            allowed = 1.5 * np.hypot(np.median(np.diff(ds.x.values)), np.median(np.diff(ds.y.values)))
            sampled_z[distances > allowed] = np.nan
            row.update(lia_head_surface_m=num(sampled_z[0]), lia_terminus_surface_m=num(sampled_z[-1]),
                       lia_head_elevation_sample_distance_m=float(distances[0]),
                       lia_terminus_elevation_sample_distance_m=float(distances[-1]),
                       lia_elevation_sample_limit_m=float(allowed),
                       lia_elevation_source='nearest valid paired V3 raster cell; bounded at 1.5 cell diagonals')
            row['lia_mainline_mean_gradient_deg'] = num(np.degrees(np.arctan2(sampled_z[0]-sampled_z[-1], row['lia_length_km']*1000)))
            row['lia_stored_endpoint_reference_difference_m'] = num(stored_elev[-1]-sampled_z[-1])
            if np.isfinite(sampled_z[-1]) and abs(stored_elev[-1]-sampled_z[-1]) > 100:
                qa.append(dict(system=sid, note='网络reference_surface末端与成对V3栅格差异超过100 m，表中使用栅格采样；未修改原始文件', difference_m=float(stored_elev[-1]-sampled_z[-1])))
        # Modern endpoint is the unique flows_to=None root in the mapped MAIN RGI gdir.
        md = gdir(rgi)
        grid = read_json(md / 'glacier_grid.json')
        lines = read_pickle(md / 'centerlines.pkl')
        roots = [i for i, l in enumerate(lines) if l.flows_to is None]
        assert len(roots) == 1, (sid, 'ambiguous modern root')
        line = lines[roots[0]]
        pixels = np.asarray(line.line.coords, float)
        modern_xy = pixels * np.asarray(grid['dxdy']) + np.asarray(grid['x0y0'])
        metrics, ll = orientation(modern_xy, grid['proj'])
        row.update({'modern_' + k: v for k, v in metrics.items()})
        row.update(modern_root_index=roots[0], modern_centerline_source=source(md / 'centerlines.pkl'))
        features.extend([
            dict(type='Feature', properties=dict(system=sid, status=state, kind='modern_main_RGI_root'), geometry=dict(type='LineString', coordinates=ll.tolist())),
            dict(type='Feature', properties=dict(system=sid, status=state, kind='modern_main_RGI_root_terminus'), geometry=dict(type='Point', coordinates=ll[-1].tolist()))])
        _, _, separation = GEOD.inv(row['lia_terminus_lon'], row['lia_terminus_lat'], row['modern_terminus_lon'], row['modern_terminus_lat'])
        row['terminus_point_separation_m'] = float(separation)
        ids = str(members.loc[sid, 'member_rgi_ids']).split(';')
        all_area, names, dates, bounds, missing = 0., [], [], [], []
        for member in ids:
            op = gdir(member) / 'outlines.tar.gz'
            if not op.exists():
                missing.append(member)
                continue
            source(op)
            outline = gpd.read_file('/vsitar/' + op.as_posix() + '/outlines.shp')
            all_area += float(outline.geometry.area.sum())/1e6
            metadata = outline.iloc[0]
            dates.append(str(metadata.get('BgnDate', '')))
            if member == rgi:
                row.update(modern_main_rgi_area_km2=float(outline.geometry.area.sum())/1e6,
                           modern_main_rgi_inventory_aspect_deg=num(metadata.get('Aspect')),
                           modern_main_rgi_inventory_slope_deg=num(metadata.get('Slope')),
                           modern_main_rgi_inventory_zmin_m=num(metadata.get('Zmin')),
                           modern_main_rgi_inventory_zmax_m=num(metadata.get('Zmax')),
                           modern_main_rgi_outline_date=str(metadata.get('BgnDate', '')),
                           main_rgi_name='' if pd.isna(metadata.get('Name')) else str(metadata.get('Name')))
            geo = outline.to_crs(4326)
            for geom in geo.geometry:
                features.append(dict(type='Feature', properties=dict(system=sid, status=state, kind='modern_RGI_outline', member=member), geometry=mapping(geom)))
        row.update(modern_member_rgi_ids=';'.join(ids), modern_member_count=len(ids),
                   modern_all_members_area_km2=all_area if not missing else None,
                   modern_member_outline_dates=';'.join(sorted(set(dates))), missing_modern_members=';'.join(missing))
        row['lia_mainline_reference_difference_m'] = row['lia_length_km']*1000 - ref_lia*1000
        row['modern_mainline_reference_difference_m'] = row['modern_length_km']*1000 - ref_modern*1000
        row['lia_direction'] = ['N','NE','E','SE','S','SW','W','NW'][int((row['lia_azimuth_deg']+22.5)//45)%8]
        if abs(row['lia_mainline_reference_difference_m']) > 1:
            qa.append(dict(system=sid, note='V3空间主流线与冻结长度目标不同；分列保留，不替换目标', difference_m=row['lia_mainline_reference_difference_m']))
        if abs(row['modern_mainline_reference_difference_m']) > 1:
            qa.append(dict(system=sid, note='现代空间主流线与冻结长度目标不同；分列保留', difference_m=row['modern_mainline_reference_difference_m']))
        rows.append(row)
        print(f'{sid}: complete', flush=True)
    assert len(rows) == 51
    df = pd.DataFrame(rows)
    for col in df.select_dtypes(include='number'):
        assert np.isfinite(df[col].dropna()).all(), col
    summary = dict(count=51, retained=32, candidates=19,
                   longitude_range=[float(df.lia_positive_ice_centroid_lon.min()), float(df.lia_positive_ice_centroid_lon.max())],
                   latitude_range=[float(df.lia_positive_ice_centroid_lat.min()), float(df.lia_positive_ice_centroid_lat.max())],
                   lia_area_km2_range=[float(df.lia_positive_ice_area_km2.min()), float(df.lia_positive_ice_area_km2.max())],
                   lia_area_km2_median=float(df.lia_positive_ice_area_km2.median()),
                   lia_length_km_range=[float(df.lia_length_km.min()), float(df.lia_length_km.max())],
                   directions=df.lia_direction.value_counts().to_dict(), quality_notes=qa,
                   missing_fields=df.isna().sum()[df.isna().sum()>0].to_dict())
    (OUT/'inventory.json').write_text(df.to_json(orient='records', force_ascii=False, indent=2), encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT/'glaciers.geojson').write_text(json.dumps(dict(type='FeatureCollection', features=features), ensure_ascii=False), encoding='utf-8')
    (OUT/'sources.json').write_text(json.dumps(SOURCES, indent=2, ensure_ascii=False), encoding='utf-8')
    make_plot(df)
    make_report(df, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_plot(df):
    colors = np.where(df.status.str.startswith('已'), '#087f8c', '#d47a35')
    fig = plt.figure(figsize=(13, 9), layout='constrained')
    ax = fig.add_subplot(221)
    ax.scatter(df.lia_positive_ice_centroid_lon, df.lia_positive_ice_centroid_lat, s=15+df.lia_positive_ice_area_km2*1.3, c=colors, alpha=.8, edgecolors='white', linewidth=.5)
    ax.set(xlabel='Longitude (E)', ylabel='Latitude (N)', title='51 LIA systems: reconstructed positive-ice centroids')
    ax.grid(alpha=.15)
    ax = fig.add_subplot(222)
    for status, color in [('已认可（32条）','#087f8c'),('旧复核候选（19条）','#d47a35')]:
        d = df[df.status==status]
        ax.scatter(d.lia_length_km, d.modern_length_km, c=color, s=35, label='Retained (32)' if status.startswith('已') else 'Legacy-review candidates (19)')
    lim=max(df.lia_length_km.max(),df.modern_length_km.max())*1.05
    ax.plot([0,lim],[0,lim], '--', color='#888', linewidth=1)
    ax.set(xlabel='LIA V3 spatial mainline (km)', ylabel='Modern main-RGI spatial root (km)', xlim=(0,lim), ylim=(0,lim), title='Mapped mainline lengths (not simulation lengths)')
    ax.legend(frameon=False, fontsize=9)
    ax = fig.add_subplot(223, projection='polar')
    ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
    theta = np.arange(8)*np.pi/4
    bins=np.arange(-22.5,360,45)
    for i,(status,color) in enumerate([('已认可（32条）','#087f8c'),('旧复核候选（19条）','#d47a35')]):
        a=df[df.status==status].lia_azimuth_deg.to_numpy(); a=(a+22.5)%360
        counts=np.histogram(a,np.linspace(0,360,9))[0]
        ax.bar(theta+(i-.5)*.28,counts,width=.27,color=color,alpha=.85)
    ax.set_xticks(theta,['N','NE','E','SE','S','SW','W','NW'])
    ax.set_title('LIA head-to-terminus azimuth / counts', pad=22)
    ax = fig.add_subplot(224)
    ax.scatter(df.lia_positive_ice_area_km2,df.lia_mean_thickness_m,c=colors,s=35)
    ax.set(xlabel='LIA V3 reconstructed positive-ice area (km²)',ylabel='LIA V3 reconstructed mean thickness (m)',title='Source geometry, not selected spinup geometry')
    ax.grid(alpha=.15)
    fig.suptitle('LIA / OGGM — spatial and geometric inventory • 32 retained + 19 candidates', fontsize=15)
    fig.savefig(OUT/'overview.png',dpi=180)
    plt.close(fig)


def make_report(df, summary):
    labels = {
        'lia_unique_id':'LIA编号','status':'认可状态','main_rgi_id':'主RGI编号',
        'lia_positive_ice_centroid_lon':'LIA中心经度','lia_positive_ice_centroid_lat':'LIA中心纬度',
        'lia_terminus_lon':'LIA末端经度','lia_terminus_lat':'LIA末端纬度','lia_terminus_surface_m':'LIA末端表面高程 m',
        'modern_terminus_lon':'现代末端经度','modern_terminus_lat':'现代末端纬度',
        'lia_positive_ice_area_km2':'LIA重建正冰面积 km²','modern_all_members_area_km2':'现代全部成员面积 km²',
        'lia_length_km':'LIA V3空间主流线 km','modern_length_km':'现代空间主流线 km',
        'lia_azimuth_deg':'LIA整体方位角 °','lia_direction':'八方位','lia_terminal_azimuth_deg':'LIA末端段方位角 °',
        'modern_main_rgi_inventory_aspect_deg':'RGI原始Aspect °','modern_main_rgi_inventory_slope_deg':'RGI原始Slope °',
        'lia_surface_min_m':'LIA最低表面 m','lia_surface_max_m':'LIA最高表面 m','lia_surface_median_m':'LIA表面中位高程 m',
        'lia_mean_thickness_m':'LIA重建平均冰厚 m','lia_reconstructed_volume_km3':'LIA重建体积 km³',
        'terminus_point_separation_m':'两末端直线间距 m','modern_member_count':'现代成员数',
        'modern_main_rgi_outline_date':'主RGI轮廓日期','lia_main_branch':'LIA主支编号','modern_root_index':'现代根支编号',
        'lia_reference_length_km':'冻结LIA长度目标 km','modern_reference_length_km':'冻结现代长度目标 km',
        'lia_mainline_reference_difference_m':'V3主线减冻结目标 m','lia_positive_coverage_fraction':'LIA正冰覆盖比例'
    }
    view=df[list(labels)].copy()
    for c in view.select_dtypes(include='number'):
        if c not in ['lia_unique_id','modern_member_count','lia_main_branch','modern_root_index']:
            view[c]=view[c].map(lambda v: '' if pd.isna(v) else f'{v:.5f}' if 'lon' in c or 'lat' in c else f'{v:.2f}')
    table=view.rename(columns=labels).to_html(index=False,escape=True,table_id='inventory')
    notes=''.join('<li>'+html.escape(str(q))+'</li>' for q in summary['quality_notes']) or '<li>51条的V3主流线和现代根流线均完成唯一身份核对。</li>'
    content='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>51条冰川属性清单</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;margin:32px;color:#20323c;background:#f6f8f9}h1{font-size:27px}p{line-height:1.8;max-width:1100px}.cards{display:flex;gap:20px}.card{padding:16px 26px;background:white;border-left:4px solid #087f8c;font-size:21px}.wrap{overflow:auto;max-height:75vh;background:white}table{border-collapse:collapse;font-size:12px;white-space:nowrap}th,td{padding:9px 11px;border-bottom:1px solid #e4e8ea;text-align:right}th{position:sticky;top:0;background:#174b58;color:white}td:first-child{font-weight:bold}tr:hover{background:#eaf4f5}input,select{padding:10px;margin:14px 8px 14px 0;border:1px solid #bccbd1;border-radius:5px}img{max-width:1100px;width:100%}a{color:#086f87}details{margin-top:22px}li{line-height:1.7}</style>
<h1>51条冰川：空间位置、末端、几何与朝向</h1><p>名单来自“1—2条冰川的曲线诊断与参数搜寻原型”对话对应的项目证据。按LIA系统编号去重；本次整理沿用已有认可状态。</p>
<div class="cards"><div class="card">51 条系统</div><div class="card">32 条已认可</div><div class="card" style="border-color:#d47a35">19 条旧复核候选</div></div>
<p><a href="glaciers.geojson">GIS数据（GeoJSON）</a> · <a href="inventory.json">完整属性数据（JSON）</a> · <a href="sources.json">来源与校验值</a></p>
<img src="overview.png" alt="空间位置、长度、朝向和重建冰厚概览">
<p><b>指标定义：</b>空间几何取成对的V3 anchor-reroute主流线与F19重建栅格。末端是主流线最后一点，现代末端对应主RGI成员的唯一根流线。整体朝向为上游起点指向末端的真北顺时针方位角；末端段取末端前10%线长、最多1 km。RGI原始Aspect另列，不能与主流线方位角互换。</p>
<p><b>解释范围：</b>LIA面积、冰厚、体积和高程是重建源几何属性，并非所选模拟曲线的初态或2000年结果。末端高程从配套V3栅格最近有效像元采样，限制在1.5个像元对角线内；采样距离见完整数据。网络reference_surface字段存在与V3栅格不一致的记录，未用来填写末端高程。现代面积累加该系统的全部映射成员，现代主流线只代表主RGI成员。末端直线间距不是沿同一流线的退缩量。现代轮廓有各自制图日期，不能一律称为2000年实测末端。冻结长度目标与V3空间线长分列，未替换任何评分目标。</p>
<input id="query" placeholder="搜索冰川编号、RGI或方位"><select id="status"><option value="">全部51条</option value="已认可">已认可32条</option value="旧复核候选">旧复核候选19条</option></select><span id="count"></span><div class="wrap">'''+table+'''</div><details><summary>质量核对与来源说明</summary><ul>'''+notes+'''</ul><p>所有系统均按已登记末端坐标匹配LIA网络组件，再使用存储的main_branch_index；现代主线要求flows_to为None的根唯一。原始文件只读，输出单独保存；未运行新的OGGM模拟。</p></details>
<script>const rows=[...document.querySelectorAll('tbody tr')];function filter(){let q=document.querySelector('#query').value.toLowerCase(),s=document.querySelector('#status').value,n=0;rows.forEach(r=>{let ok=r.textContent.toLowerCase().includes(q)&&r.children[1].textContent.includes(s);r.hidden=!ok;if(ok)n++});document.querySelector('#count').textContent=`显示 ${n} 条`}document.querySelector('#query').oninput=filter;document.querySelector('#status').onchange=filter;filter();</script></html>'''
    (OUT/'index.html').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    run()
