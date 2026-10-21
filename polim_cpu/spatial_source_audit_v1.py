"""Offline source reconciliation and spatial input export. No dynamics or climate.

Rebuild the existing elevation-band mapping only to establish source identity;
the exported 2-D source is not a newly selected scientific flowline.
"""
import base64
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import xarray as xr
from .v3.runner import PROJECT, digest, write_json
from .thermal_attribution_v1 import validate, ref

BASE=PROJECT/'validation_data/results/paleo_surface_validation_v0'
OUT=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-v1'
EVIDENCE=BASE/'polim_lia_system2387_spatial_source_audit_v1'
REPORT=PROJECT/'polim_cpu/reports/spatial-source-audit-v1'
DRIVERS=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers'


def read(p): return json.loads(p.read_text(encoding='utf-8'))


def decode(path):
    with xr.open_dataset(path) as d:
        i=list(map(str,d.member_id.values)).index('farinotti2019')
        arrays={k:np.asarray(d[k].isel(member=i).values,dtype=float) for k in ('thickness_m','surface_elevation_m','bed_elevation_m')}
        arrays.update(x=np.asarray(d.x.values,dtype=float),y=np.asarray(d.y.values,dtype=float),
                      source_valid_mask=np.asarray(d.valid_mask.isel(member=i).values),lia_mask=np.asarray(d.lia_mask.values),
                      excluded_component_mask=np.asarray(d.excluded_component_mask.values))
        metadata={'source':ref(path),'attributes':{k:(v.item() if isinstance(v,np.generic) else v) for k,v in d.attrs.items()},'spatial_ref':dict(d.spatial_ref.attrs),'bed_scenario':'farinotti2019'}
    arrays['valid_mask']=(arrays['source_valid_mask']>0)&(arrays['lia_mask']>0)
    for key in ('thickness_m','surface_elevation_m','bed_elevation_m'): arrays['valid_mask'] &= np.isfinite(arrays[key])
    return arrays,metadata


def build():
    controller=validate(); OUT.mkdir(parents=True,exist_ok=True); EVIDENCE.mkdir(parents=True,exist_ok=True); REPORT.mkdir(parents=True,exist_ok=True)
    if (EVIDENCE/'summary-completed.json').exists(): raise RuntimeError('Audit already complete')
    before=digest(PROJECT/'LIA_CURRENT_STATE.json'); facts=digest(PROJECT/'LIA_PROJECT_FACTS.md')
    source_audit=BASE/'lia_elevation_band_core191_length_corrected_v1/2387/elevation_band_audit.json'
    source=read(source_audit)
    frozen=PROJECT/'polim_cpu/runs/lia-2387-v3-fixed-a-control-v1'
    with np.load(frozen/'step-000000/state.npz',allow_pickle=False) as d: state={k:d[k].copy() for k in d.files}
    candidates=[]; data={}
    for folder in ('core191_production_candidate_v01','core191_production_candidate_v2','core191_anchor_reroute_f19_v2','core191_anchor_reroute_f19_v3'):
        path=BASE/folder/'lia_2387_RGI60-15.04813_system_grid_v1.nc'
        a,m=decode(path); dx=abs(float(np.median(np.diff(a['x'])))); dy=abs(float(np.median(np.diff(a['y'])))); wet=a['valid_mask']&(a['thickness_m']>0)
        volume=float(np.sum(a['thickness_m'][wet],dtype=np.float64)*dx*dy); area=float(wet.sum()*dx*dy)
        candidates.append({'folder':folder,**m,'area_m2':area,'volume_m3':volume,
                           'area_matches_existing':bool(np.isclose(area,source['source_area_m2'],rtol=1e-12)),
                           'volume_matches_existing':bool(np.isclose(volume,source['source_volume_m3'],rtol=1e-12)),
                           'volume_relative_difference':volume/source['source_volume_m3']-1})
        data[folder]=a
    matches=[r for r in candidates if r['area_matches_existing'] and r['volume_matches_existing']]
    if not matches: raise RuntimeError('No matching spatial source; do not select by filename')
    selected=matches[0]; a=data[selected['folder']]
    for m in matches[1:]:
        if any(not np.array_equal(v,data[m['folder']][k],equal_nan=True) for k,v in a.items()):
            raise RuntimeError('Equal global statistics conceal different spatial arrays; resolve source identity')
    # Independent static replay of the documented construction; no OGGM import.
    module_path=DRIVERS/'lia_elevation_band.py'; spec=importlib.util.spec_from_file_location('_polim_source_replay',module_path)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    dx=abs(float(np.median(np.diff(a['x'])))); dy=abs(float(np.median(np.diff(a['y']))))
    rebuilt=module.build_elevation_band_state(a['surface_elevation_m'],a['bed_elevation_m'],a['thickness_m'],a['valid_mask'],
        cell_dx_m=dx,cell_dy_m=dy,elevation_bin_size_m=30.,node_dx_m=50.,target_length_m=source['target_main_flowline_length_m'])
    n=len(rebuilt.node_thickness_m)
    errors={k:float(np.max(abs(value-state[k][:n]))) for k,value in [('h',rebuilt.node_thickness_m),('bed',rebuilt.node_bed_h_m),('width',rebuilt.node_width_m)]}
    if max(errors.values())>1e-8 or n!=94: raise RuntimeError('Spatial source does not reproduce frozen initial geometry')
    if not (OUT/'geometry.npz').exists():
        with (OUT/'geometry.npz').open('xb') as f: np.savez_compressed(f,**a)
    with np.load(OUT/'geometry.npz',allow_pickle=False) as d:
        if any(not np.array_equal(v,d[k],equal_nan=True) for k,v in a.items()): raise RuntimeError('Spatial export changed arrays')
    x=state['x']; slope=np.gradient(state['bed']+state['h'],x); speed=abs(state['u']).max(axis=1); wet=state['h']>0
    rows=[]
    for i in np.flatnonzero(wet):
        rows.append({'index':int(i),'x_pseudo_m':float(x[i]),'bed_m':float(state['bed'][i]),'thickness_m':float(state['h'][i]),
                     'equivalent_width_m':float(state['width'][i]),'surface_slope':float(slope[i]),
                     'apparent_profile_angle_degrees':float(np.degrees(np.arctan(abs(slope[i])))),
                     'fixed_A_initial_column_max_speed_m_year':float(speed[i])})
    if not (REPORT/'profile.csv').exists():
        with (REPORT/'profile.csv').open('x',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    matrix=BASE/'lia_191_all191_oggm_flowline_1d_v10_203_transaction_v1/level1_matrix.csv'
    with matrix.open(encoding='utf-8-sig',newline='') as f:
        entries={int(r['lia_unique_id']):float(r['glen_a_pa_3_s_1']) for r in csv.DictReader(f)}
    peak=max(rows,key=lambda r:r['fixed_A_initial_column_max_speed_m_year'])
    package={'schema':'polim-spatial-source-package-v1','system':2387,'status':'source-reconciled-profile-design-pending',
             'source':selected['source'],'identical_matching_sources':[m['source'] for m in matches],
             'metadata':selected['spatial_ref'],'grid_shape':list(a['thickness_m'].shape),'grid_spacing_m':[dx,dy],
             'array_archive':ref(OUT/'geometry.npz'),'existing_geometry_static_replay_max_errors_m':errors,
             'selection_rule':'Match frozen source area/volume, require matching candidate arrays identical, then reproduce all 94 original wet nodes',
             'excluded_components_preserved':True,'nan_and_valid_masks_preserved':True,'climate_fields_decoded':False,
             'profile_ready':False,'launch_authorized':False,'scientific_acceptance':False,
             'remaining':'Define actual geographic centreline and tributary/catchment accounting; verify geometric mapping and register new model transaction before dynamics'}
    if (OUT/'input-package.json').exists():
        if read(OUT/'input-package.json')!=package: raise RuntimeError('Existing package changed')
    else: write_json(OUT/'input-package.json',package)
    result={'schema':'polim-spatial-source-audit-v1','status':'readonly-completed','controller':controller,
            'facts_sha256':facts,'source_audit':ref(source_audit),'source_candidates':candidates,'package':ref(OUT/'input-package.json'),
            'construction_code':ref(module_path),'parameter_matrix':ref(matrix),
            'parameter_origin':'stage2_glen_a inherited via v10.189/v10.203, not calibrated to the v3 BP model',
            'supplied_A':entries[2387],'A_rank_descending_in_191':1+sum(v>entries[2387] for v in entries.values()),
            'peak_fixed_A_initial_cell':peak,'maximum_equivalent_width_m':float(state['width'][wet].max()),
            'mean_equivalent_width_m':float(source['initial_flowline_area_m2']/source['pseudo_flowline_length_m']),
            'longitudinal_scale_factor':source['longitudinal_scale_factor'],
            'source_zero_MB_volume_drift_fraction':source['zero_mb_checkpoints'][0]['volume_change_fraction_from_initial'],
            'historical_zero_MB_result_scope':'Different prior OGGM solver/configuration; not a failed mass-conservation result for v3',
            'causal_limit':'This source audit does not partition speed into slope/width/A contributions and does not validate a replacement profile',
            'next_implementation':'Prepare georeferenced profile/network from the matched 2-D source; preserve source pixels, topology and area/volume accounting. No arbitrary slope smoothing, width rescaling or A fitting to suppress speed.',
            'new_simulations':0,'accepted_count_effect':0,'scientific_acceptance':False}
    render(a,state,result,source)
    result['figure']=ref(REPORT/'geometry.png'); result['report']=ref(REPORT/'index.html')
    if digest(PROJECT/'LIA_CURRENT_STATE.json')!=before or digest(PROJECT/'LIA_PROJECT_FACTS.md')!=facts: raise RuntimeError('Controller/facts changed concurrently')
    write_json(EVIDENCE/'summary-completed.json',result)
    print(json.dumps({'status':result['status'],'matching_sources':[m['folder'] for m in matches],'static_replay_errors_m':errors,'peak':peak,'A_rank':result['A_rank_descending_in_191'],'package':str(OUT),'report':str(REPORT/'index.html')},indent=2))


def render(a,s,r,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    xx=(a['x']-a['x'].min())/1000; yy=(a['y']-a['y'].min())/1000
    wet=a['valid_mask']&(a['thickness_m']>0)
    for ax,key,title in [(axes[0,0],'thickness_m','Matched geographic source: ice thickness (m)'),(axes[0,1],'surface_elevation_m','Matched geographic source: surface elevation (m)')]:
        im=ax.pcolormesh(xx,yy,np.ma.masked_where(~wet,a[key]),shading='nearest',cmap='viridis'); fig.colorbar(im,ax=ax)
        ax.set(title=title,xlabel='Local easting (km)',ylabel='Local northing (km)',aspect='equal')
    count=94; x=s['x'][:count]/1000; ax=axes[1,0]
    ax.plot(x,s['bed'][:count],label='Equivalent bed'); ax.plot(x,(s['bed']+s['h'])[:count],label='Equivalent surface')
    peak=r['peak_fixed_A_initial_cell']; ax.axvline(peak['x_pseudo_m']/1000,color='#bc552b',ls=':',label='Fixed-A initial speed peak')
    ax.set(title='Elevation-sorted pseudo-flowline',xlabel='Pseudo distance (km)',ylabel='Elevation (m)'); ax.legend()
    ax=axes[1,1]; ax.plot(x,s['width'][:count]/1000,color='#206b9b'); ax.set(title='Equivalent width preserves total area',xlabel='Pseudo distance (km)',ylabel='Equivalent width (km)')
    fig.suptitle('System 2387 | source reconciliation only | no new simulation',fontsize=15)
    fig.savefig(REPORT/'geometry.png',dpi=150); plt.close(fig)
    png=base64.b64encode((REPORT/'geometry.png').read_bytes()).decode()
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 输入来源与几何</title><style>body{{max-width:1120px;margin:36px auto;padding:0 22px;background:#f3f6f9;color:#183047;font:16px/1.75 system-ui}}section{{background:white;padding:24px;margin:18px 0;border-radius:14px}}img{{width:100%}}a{{color:#206b9b}}h1{{line-height:1.35}}</style>
<h1>下一步：从空间源几何建立可解释的剖面</h1><p>本次已完成来源核对与二维输入打包。没有新增模拟，也没有修改第三版的原有结果。</p>
<section><h2>高流速所在位置</h2><p>固定 A 对照在初始伪流线 {peak['x_pseudo_m']/1000:.2f} km 处达到 {peak['fixed_A_initial_column_max_speed_m_year']/1000:.2f} km/年；该格冰厚 {peak['thickness_m']:.1f} m、等效宽度 {peak['equivalent_width_m']/1000:.2f} km、表面坡度绝对值 {abs(peak['surface_slope']):.2f}（伪剖面表观角度 {peak['apparent_profile_angle_degrees']:.1f}°）。这些是伪流线坐标上的计算值，不能定位为地图上同一距离的实测断面。</p>
<p>构造将按高程排序的冰体由 {source['unscaled_pseudo_flowline_length_m']/1000:.2f} km 缩放到 {source['target_main_flowline_length_m']/1000:.2f} km，离散后为 4.70 km；通过等效宽度保留面积、体积。最大等效宽度为 {r['maximum_equivalent_width_m']/1000:.2f} km。因此总体积一致并不足以证明局部坡度、宽度与 BP 动量模型具有物理等价性。</p></section>
<img src="data:image/png;base64,{png}" alt="匹配二维冰厚、表面高程与当前伪流线及等效宽度">
<section><h2>这次已经交付</h2><p>448 × 613、间距 15.75 m 的二维源包包含冰厚、表面与底部高程、地理投影、坐标和掩膜。候选版本先按来源面积和体积匹配，再核对空间数组，并用原构造函数复现全部 94 个有冰网格。具有不同冰厚的其他版本另列保留，没有静默替换。</p><p>继承的 A 为 {r['supplied_A']:.6g} Pa⁻³ s⁻¹，在 191 个来源标量中从大到小排第 {r['A_rank_descending_in_191']}。它来自原 Stage 2 参数链，尚无本次 BP 模型的独立校准。</p></section>
<section><h2>接续工作与解释范围</h2><p>下一阶段需要在匹配的二维源上确定带真实距离的中心线及支流／汇流区归属，保留像元和面积、体积对应关系，然后登记新几何试算。此包目前是空间源输入，尚不是可直接运行的修正剖面。</p><p>项目曾记录 2387 在另一套 OGGM 配置下五年零质量平衡体积增加 3.565%，这是已有几何动力学背景，不能直接当作第三版的质量不守恒结论。本次也没有把高速归因比例分配给坡度、宽度和 A，更没有据此调低参数。</p><p><a href="../../lia-inputs-v1/system-2387/spatial-source-v1/input-package.json">二维输入说明</a> · <a href="../../../validation_data/results/paleo_surface_validation_v0/polim_lia_system2387_spatial_source_audit_v1/summary-completed.json">来源核对证据</a> · <a href="../thermal-attribution-v1/index.html">前次温度反馈对照</a></p></section></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__': build()
