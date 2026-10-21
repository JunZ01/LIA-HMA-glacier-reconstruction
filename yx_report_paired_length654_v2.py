"""Report the disclosed all-system paired stored-axis metric, preserving old studies."""
from pathlib import Path
import os, json, hashlib, subprocess, time, html
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['GDAL_DATA'] = 'D:/Anaconda/envs/oggm_env2/Library/share/gdal'
os.environ['PROJ_LIB'] = 'D:/Anaconda/envs/oggm_env2/Library/share/proj'
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from pyproj import Geod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent
BASE = Path('H:/ToYangxin/result_yx')
SELECT = BASE / 'selected_reconstruction_20261010_v1_654'
OLD = BASE / 'length_change_selected654_20261010_v1'
OUT = BASE / 'paired_axis_length_selected654_20261010_v2'
METHOD = 'all654-two-endpoint-maximum-complete-stored-reference-axis-length-v2'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def bools(s):
    return s.astype(str).str.lower().eq('true')

def desc(s):
    return {'n': len(s), 'mean': float(s.mean()), 'median': float(s.median()),
            'p25': float(s.quantile(.25)), 'p75': float(s.quantile(.75)),
            'minimum': float(s.min()), 'maximum': float(s.max())}

def main():
    receipt = []
    for name in ['LIA_PROJECT_FACTS.md', 'LIA_EXECUTION_CONTRACT.md', 'LIA_CURRENT_STATE.json']:
        p = REPO / name
        s = p.read_text(encoding='utf-8-sig')
        receipt.append({'path': str(p), 'fully_read': True, 'chars': len(s), 'sha256': sha(p)})
    state = json.loads((REPO / 'LIA_CURRENT_STATE.json').read_text(encoding='utf-8-sig'))
    assert state['current_focus']['state'] != 'decision-required'
    branch = state['research_branches']['yx_selected654_paired_axis_length_v2']
    assert branch['status'] == 'registered-all654-paired-stored-axis-length-analysis'
    validator = REPO / 'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
    v = subprocess.run([os.sys.executable, str(validator)], cwd=REPO, capture_output=True, text=True, encoding='utf-8', check=True)
    assert json.loads(v.stdout)['status'] == 'pass'
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'paired_axis_lengths654.csv').exists(), 'Immutable report outputs already exist'
    save(OUT / 'report_canonical_read_receipt.json', receipt)
    save(OUT / 'report_validator.json', json.loads(v.stdout))
    old_manifest = json.loads((OLD / 'length_analysis_sha256_manifest.json').read_text(encoding='utf-8'))
    old_entries = {r['relative_path']: r for r in old_manifest['files']}
    for f in ['source_axes/root_axes.csv', 'length_diagnostics_selected654.csv']:
        assert sha(OLD / f) == old_entries[f]['sha256']
    selected_manifest = json.loads((SELECT / 'selection_sha256_manifest.json').read_text(encoding='utf-8'))
    selected_entries = {r['relative_path']: r for r in selected_manifest['files']}
    for f in ['selected_system_status.csv', 'local_10_mountain_summary.csv', 'selected_reconstruction_results.gpkg']:
        assert sha(SELECT / f) == selected_entries[f]['sha256']

    lia_all = pd.read_csv(OLD / 'source_axes/root_axes.csv')
    modern_all = pd.read_csv(OUT / 'modern_axes/root_axes.csv')
    statuses = pd.read_csv(OUT / 'modern_axes/extraction_status654.csv')
    assert len(statuses) == 654 and statuses.record_id.nunique() == 654
    assert statuses.status.eq('extracted').all(), statuses[~statuses.status.eq('extracted')].to_dict('records')
    selected = pd.read_csv(SELECT / 'selected_system_status.csv')
    old_diag = pd.read_csv(OLD / 'length_diagnostics_selected654.csv')
    lia = lia_all[bools(lia_all.is_system_reference)].set_index('record_id')
    modern = modern_all[bools(modern_all.is_system_reference)].set_index('record_id')
    expected = set(selected.record_id)
    assert len(lia) == len(modern) == 654 and set(lia.index) == set(modern.index) == expected
    assert lia.index.is_unique and modern.index.is_unique
    geod = Geod(ellps='WGS84')
    rows = []
    for rec in selected.record_id:
        a, b = lia.loc[rec], modern.loc[rec]
        L, M = float(a.axis_full_length_m), float(b.axis_full_length_m)
        assert np.isfinite(L) and np.isfinite(M) and L > 0 and M > 0
        la, lb = wkt.loads(a.geometry_wgs84_wkt), wkt.loads(b.geometry_wgs84_wkt)
        ha, hb, ta, tb = la.coords[0], lb.coords[0], la.coords[-1], lb.coords[-1]
        head_distance = float(geod.inv(*ha, *hb)[2])
        tail_distance = float(geod.inv(*ta, *tb)[2])
        threshold = 2 * max(float(a.native_dx_m), float(b.native_dx_m))
        row = {'record_id': rec, 'lia_unique_id': int(a.lia_unique_id),
               'LIA_max_stored_axis_m': L, 'modern_max_stored_axis_m': M,
               'axis_length_decrease_m': L - M, 'axis_length_decrease_relative_LIA_percent': 100 * (L - M) / L,
               'LIA_reference_root_key': a.root_key, 'modern_reference_root_key': b.root_key,
               'LIA_source_family': a.source_family, 'modern_source_family': b.source_family,
               'LIA_component_count': int(a.component_count), 'modern_component_count': int(b.component_count),
               'LIA_root_count': int(a.system_root_count), 'modern_root_count': int(b.system_root_count),
               'LIA_reference_component_index': int(a.component_index), 'modern_reference_component_index': int(b.component_index),
               'LIA_reference_head_id': a.head_node_id, 'modern_reference_head_id': b.head_node_id,
               'reference_heads_separation_m': head_distance, 'reference_tails_separation_m': tail_distance,
               'head_shift_description_threshold_m': threshold,
               'head_shift_exceeds_two_coarser_grid_cells_description_only': head_distance > threshold,
               'head_shift_is_not_verified_branch_switch': True,
               'LIA_head_crop_displacement_m': a.source_head_crop_distance_m,
               'modern_head_crop_displacement_m': b.source_head_crop_distance_m,
               'LIA_head_crop_known': bool(str(a.source_head_crop_known).lower() == 'true'),
               'modern_head_crop_known': bool(str(b.source_head_crop_known).lower() == 'true'),
               'LIA_head_distance_to_endpoint_outline_m': a.head_boundary_distance_m,
               'modern_head_distance_to_endpoint_outline_m': b.head_boundary_distance_m,
               'LIA_tail_distance_to_endpoint_outline_m': a.tail_boundary_distance_m,
               'modern_tail_distance_to_endpoint_outline_m': b.tail_boundary_distance_m,
               'LIA_source_pickle_path': a.source_pickle_path, 'modern_source_pickle_path': b.source_pickle_path,
               'LIA_source_pickle_sha256': a.source_pickle_sha256, 'modern_source_pickle_sha256': b.source_pickle_sha256,
               'all654_statistical_inclusion': True, 'same_branch_observed_terminus_retreat': False,
               'complete_physical_boundary_to_boundary_length': False}
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.merge(selected[['record_id', 'source_record_ids', 'source_record_count', 'mountain', 'selected_model_version']], on='record_id', validate='one_to_one')
    df = df.merge(old_diag[['record_id', 'length_diagnostic_status', 'primary_unambiguous_status', 'lia_area_geodesic_km2', 'modern_area_geodesic_km2', 'area_decrease_relative_LIA_percent', 'lia_volume_deduplicated_km3', 'modern_volume_km3', 'volume_loss_deduplicated_percent']], on='record_id', validate='one_to_one')
    df = df.rename(columns={'length_diagnostic_status': 'previous_fixed_LIA_axis_diagnostic_status', 'primary_unambiguous_status': 'previous_485_subset_membership'})
    df.to_csv(OUT / 'paired_axis_lengths654.csv', index=False, encoding='utf-8-sig')
    assert len(df) == 654 and np.isfinite(df.axis_length_decrease_relative_LIA_percent).all()
    delta = df.axis_length_decrease_m
    fraction = df.axis_length_decrease_relative_LIA_percent
    eps = 1e-6
    summary = {'status': 'computed-all654-disclosed-paired-stored-axis-length-index', 'method': METHOD,
               'selected_systems': 654, 'source_records': 655, 'statistical_systems': 654,
               'systems_missing_either_endpoint': 0, 'LIA_roots': len(lia_all), 'modern_roots': len(modern_all),
               'LIA_total_stored_axis_km': float(df.LIA_max_stored_axis_m.sum() / 1000),
               'modern_total_stored_axis_km': float(df.modern_max_stored_axis_m.sum() / 1000),
               'total_stored_axis_decrease_km': float(delta.sum() / 1000),
               'length_weighted_decrease_percent': float(100 * delta.sum() / df.LIA_max_stored_axis_m.sum()),
               'individual_decrease_m': desc(delta), 'individual_decrease_percent': desc(fraction),
               'sign_counts': {'shorter': int((delta > eps).sum()), 'equal_within_numerical_1e_6m': int((delta.abs() <= eps).sum()), 'longer': int((delta < -eps).sum())},
               'previous_485_subset_does_not_filter_this_metric': True,
               'former_169_all_included': int((~bools(df.previous_485_subset_membership)).sum()),
               'head_shift_description_only_count': int(df.head_shift_exceeds_two_coarser_grid_cells_description_only.sum()),
               'no_model_recalculations_or_new_routes': True, 'area_and_volume_results_unchanged': True,
               'not_same_branch_observed_terminus_retreat': True, 'not_complete_boundary_to_boundary_glacier_length': True,
               'no_annual_rate_formal_CI_or_HMA_extrapolation': True,
               'endpoint_maxima_definition': 'maximum complete stored head-to-root planform axis across all saved components and roots; actual connectors; each endpoint chosen independently',
               'difference_definition': 'LIA maximum stored axis minus modern maximum stored axis',
               'relative_definition': '100 times difference divided by LIA maximum stored axis',
               'numerical_sign_classification_tolerance_m': eps}
    save(OUT / 'summary.json', summary)
    pd.crosstab(df.LIA_source_family, df.modern_source_family).to_csv(OUT / 'paired_source_family_counts.csv', encoding='utf-8-sig')
    previous = df.groupby('previous_fixed_LIA_axis_diagnostic_status', dropna=False).agg(systems=('record_id', 'size'), paired_complete=('all654_statistical_inclusion', 'sum'))
    previous.to_csv(OUT / 'previous_169_inclusion_check.csv', encoding='utf-8-sig')
    mts = pd.read_csv(SELECT / 'local_10_mountain_summary.csv')
    mountain_col = 'mountain'
    assert mountain_col in mts.columns
    region_rows = []
    for mountain in mts[mountain_col]:
        z = df[df.mountain == mountain]
        vals = {'mountain': mountain, 'systems': len(z)}
        if len(z):
            vals.update(LIA_axis_sum_km=float(z.LIA_max_stored_axis_m.sum()/1000), modern_axis_sum_km=float(z.modern_max_stored_axis_m.sum()/1000),
                        axis_decrease_sum_km=float(z.axis_length_decrease_m.sum()/1000),
                        length_weighted_decrease_percent=float(100*z.axis_length_decrease_m.sum()/z.LIA_max_stored_axis_m.sum()),
                        individual_mean_decrease_percent=float(z.axis_length_decrease_relative_LIA_percent.mean()),
                        individual_median_decrease_percent=float(z.axis_length_decrease_relative_LIA_percent.median()),
                        individual_median_decrease_m=float(z.axis_length_decrease_m.median()),
                        shorter_systems=int((z.axis_length_decrease_m>eps).sum()), equal_systems=int((z.axis_length_decrease_m.abs()<=eps).sum()), longer_systems=int((z.axis_length_decrease_m< -eps).sum()))
        region_rows.append(vals)
    regions = pd.DataFrame(region_rows)
    assert len(regions) == 10 and regions.systems.sum() == 654
    regions.to_csv(OUT / 'local10_paired_axis_length_summary.csv', index=False, encoding='utf-8-sig')

    polygons = gpd.read_file(SELECT / 'selected_reconstruction_results.gpkg', layer='lia_selected654')
    if 'record_id' not in polygons:
        raise ValueError('Selected GIS lacks record_id')
    polygons = polygons[['record_id', 'geometry']].merge(df, on='record_id', validate='one_to_one')
    gp = OUT / 'paired_axis_lengths654.gpkg'
    polygons.to_file(gp, layer='paired_length_systems654', driver='GPKG')
    for label, all_axes in [('LIA', lia_all), ('modern', modern_all)]:
        keep = ['root_key', 'record_id', 'lia_unique_id', 'source_family', 'component_index', 'system_root_count', 'root_node_id', 'head_node_id', 'axis_full_length_m', 'is_system_reference', 'native_dx_m']
        axes_gdf = gpd.GeoDataFrame(all_axes[keep], geometry=[wkt.loads(v) for v in all_axes.geometry_wgs84_wkt], crs=4326)
        axes_gdf.to_file(gp, layer=f'{label}_all_root_axes', driver='GPKG')

    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax[0].hist(delta/1000, bins=36, color='#376b91', edgecolor='white')
    ax[0].axvline(0, color='#333333', lw=1)
    ax[0].set(xlabel='小冰期−现代 最大保存参考轴长度（km）', ylabel='冰川系统数', title='全部654个系统：有符号长度差')
    ax[1].hist(fraction, bins=36, color='#4a8d76', edgecolor='white')
    ax[1].axvline(0, color='#333333', lw=1)
    ax[1].set(xlabel='长度差 / 小冰期参考轴长度（%）', ylabel='冰川系统数', title='全部654个系统：逐系统比例')
    fig.suptitle('两期最大保存参考轴长度变化；各期独立选最长路径', fontsize=13)
    fig.text(.5, .015, '正值表示现代参考轴较短；负值表示较长。不是同一末端的实测退缩距离。', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, .06, 1, .93])
    fig.savefig(OUT/'paired_axis_change_distribution654.png', dpi=180)
    plt.close(fig)
    examples = ['YX-00006', 'YX-20287', 'YX-29308', 'YX-04312']
    lia_polys = gpd.read_file(SELECT/'selected_reconstruction_results.gpkg', layer='lia_selected654').set_index('record_id')
    modern_polys = gpd.read_file(SELECT/'selected_reconstruction_results.gpkg', layer='modern_selected654').set_index('record_id')
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, rec in zip(axes.ravel(), examples):
        a,b=lia.loc[rec],modern.loc[rec]
        crs=a.native_crs_wkt
        for geometry,color in [(lia_polys.loc[rec].geometry,'#ad7138'),(modern_polys.loc[rec].geometry,'#347eb0')]:
            gpd.GeoSeries([geometry],crs=4326).to_crs(crs).boundary.plot(ax=ax,color=color,linewidth=.7)
        for row,color in [(a,'#d2914c'),(b,'#2e72aa')]:
            line=gpd.GeoSeries([wkt.loads(row.geometry_wgs84_wkt)],crs=4326).to_crs(crs).iloc[0]
            x,y=np.asarray(line.coords).T
            ax.plot(x,y,color=color,lw=1.3)
            ax.scatter(x[0],y[0],s=13,color=color)
        z=df[df.record_id.eq(rec)].iloc[0]
        ax.set_title(f'{rec}  LIA {z.LIA_max_stored_axis_m/1000:.2f} → 现代 {z.modern_max_stored_axis_m/1000:.2f} km\n长度差比例 {z.axis_length_decrease_relative_LIA_percent:.2f}%')
        ax.set_aspect('equal');ax.ticklabel_format(style='sci',axis='both',scilimits=(0,0));ax.tick_params(labelsize=7)
    fig.suptitle('两期参考主轴示例：棕色小冰期，蓝色现代；圆点为保存轴起点',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(OUT/'paired_axis_examples654.png',dpi=180);plt.close(fig)

    display = regions.rename(columns={'mountain':'山脉','systems':'系统数','LIA_axis_sum_km':'小冰期参考轴合计(km)','modern_axis_sum_km':'现代参考轴合计(km)','axis_decrease_sum_km':'长度差合计(km)','length_weighted_decrease_percent':'长度加权比例(%)','individual_mean_decrease_percent':'逐系统平均比例(%)','individual_median_decrease_percent':'逐系统中位比例(%)','individual_median_decrease_m':'长度差中位数(m)','shorter_systems':'现代较短','equal_systems':'数值相同','longer_systems':'现代较长'})
    nointer = df[df.record_id.isin(['YX-20287','YX-29308'])][['record_id','LIA_max_stored_axis_m','modern_max_stored_axis_m','axis_length_decrease_relative_LIA_percent']]
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>654个系统的两期参考主轴长度变化</title><style>body{{max-width:1160px;margin:32px auto;padding:0 20px;font:16px/1.7 Arial,"Microsoft YaHei",sans-serif;color:#24313c}}h1{{font-size:27px}}h2{{font-size:21px;margin-top:32px}}.call{{background:#edf4f7;padding:18px;border-left:5px solid #467894}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:7px;border:1px solid #cbd6dc;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto}}.scroll{{overflow-x:auto}}a{{color:#216287}}</style>
    <h1>654个冰川系统：两期最大保存参考轴长度变化</h1>
    <p>当前保留样本654个系统、655条源记录，全部纳入本指标。前一版485个明确正末端空段样本仍保留为另一种指标；其余169个本次全部纳入，没有将现代与旧轴不相交误记为冰川消失。</p>
    <div class="call">全部654个系统的长度加权差比例 <b>{summary['length_weighted_decrease_percent']:.2f}%</b>；逐系统中位数 <b>{fraction.median():.2f}%</b>，平均值 <b>{fraction.mean():.2f}%</b>。长度差中位数 <b>{delta.median()/1000:.3f} km</b>，平均值 <b>{delta.mean()/1000:.3f} km</b>。现代参考轴较短{summary['sign_counts']['shorter']}个、数值相同{summary['sign_counts']['equal_within_numerical_1e_6m']}个、较长{summary['sign_counts']['longer']}个；负值完整保留。</div>
    <h2>统一计算定义</h2><p>分别提取小冰期与现代保存路线网络中各起点至出口的完整路径，使用原网络实际连接线拼接，在原投影坐标中计算水平长度。各期所有保存组件和出口中的最长路径，定义为该期最大保存参考轴长度。长度差＝小冰期参考轴长度−现代参考轴长度；逐系统比例＝100×长度差÷小冰期参考轴长度；整体长度加权比例＝100×全部长度差合计÷全部小冰期参考轴长度合计。各期独立选轴，现代不沿旧轴取交集。</p>
    <p><b>指标边界：</b>这些路线已初始化，部分源头经过裁剪、端部未完全贴合真实轮廓；最长路径也可能在两期对应不同支流或出口。因此，这是两期最大保存参考轴的几何长度差，不能直接称完整边界到边界的实测冰川长度，也不能称同一末端的实测退缩距离。正、零、负值均保留；面积、冰量及原报告未改变，没有重新运行冰川模型。多组件、多出口、候选路线、起点裁剪量、距轮廓距离、起终点位移及原长度诊断均逐条记录，作为可靠性信息，不作为筛除条件。起点位移超过两倍较粗网格间距仅为描述性提示，不是已验证的支流切换。</p>
    <p>概念上参考RGI使用最长几何中心线描述最大长度的做法，但这里保存的初始化路线并非完整RGI中心线产品，不能直接称RGI lmax或使用其精度。<a href="https://www.glims.org/rgi_user_guide/products/centerlines_product.html">RGI中心线定义</a>；<a href="https://docs.oggm.org/en/stable/flowlines.html">OGGM中心线与初始化流线说明</a>。</p>
    <h2>全部样本分布</h2><img src="paired_axis_change_distribution654.png" alt="654个系统的有符号长度差和比例分布">
    <h2>本地10山脉分区</h2><p>无保留样本的三个喜马拉雅分区留空，不填零变化。</p><div class="scroll">{display.to_html(index=False,na_rep='—',float_format=lambda v:f'{v:.3f}',border=0)}</div>
    <h2>此前旧轴未相交的两个系统</h2><p>现代轮廓仍有自身的保存主轴，采用真实现代源路径计算，不记为零现代长度或100%消失。</p>{nointer.to_html(index=False,float_format=lambda v:f'{v:.3f}',border=0)}
    <h2>地图核对示例</h2><img src="paired_axis_examples654.png" alt="四个系统的两期主轴地图">
    <h2>文件</h2><ul><li><a href="paired_axis_lengths654.csv">654个系统的两期长度、比例和可靠性信息</a></li><li><a href="local10_paired_axis_length_summary.csv">本地10山脉汇总</a></li><li><a href="paired_axis_lengths654.gpkg">GIS：654个系统与两期全部出口轴</a></li><li><a href="summary.json">完整统计摘要</a></li><li><a href="modern_axes/root_axes.csv">现代全部出口参考轴</a></li><li><a href="previous_169_inclusion_check.csv">原169个诊断系统纳入核对</a></li><li><a href="../length_change_selected654_20261010_v1/report.html">此前485个同轴末端空段研究（不同指标）</a></li><li><a href="../selected_reconstruction_20261010_v1_654/report.html">654个系统面积与冰量</a></li></ul>
    <p>所有统计均限当前保留样本；未计算年度长度变化率、正式置信区间或全HMA外推。独立核对和封存记录另见同目录审计文件。</p></html>'''
    (OUT/'report.html').write_text(text,encoding='utf-8')
    save(OUT/'report_input_hashes.json',{str(p):sha(p) for p in [OLD/'source_axes/root_axes.csv',OLD/'length_analysis_sha256_manifest.json',OLD/'length_diagnostics_selected654.csv',SELECT/'selection_sha256_manifest.json',SELECT/'selected_system_status.csv',SELECT/'selected_reconstruction_results.gpkg',SELECT/'local_10_mountain_summary.csv',OUT/'modern_axes/root_axes.csv',OUT/'modern_axes/extraction_status654.csv',Path(__file__)]})
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
