"""Independent checks built from immutable selected654 sources, before new results."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd
import xarray as xr

SOURCE=Path(r'H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
OUT=Path(r'H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/selected654_independent_qc')
OUT.mkdir(exist_ok=True)
v=pd.read_csv(SOURCE/'paired_volume_changes.csv').set_index('record_id')
a=pd.read_csv(SOURCE/'selected_inventory_area_and_modern_volume.csv').set_index('record_id')
assert len(v)==len(a)==654 and v.index.is_unique and set(a.index)==set(v.index)
assert set(v.index).isdisjoint({'YX-03055','YX-03621','YX-05103','YX-05176','YX-14077','YX-28236'})
baseline={'population':654,'model_families':v.groupby('selected_model_version').size().to_dict(),
 'lia_volume_deduplicated_km3':float(v.lia_volume_deduplicated_km3.sum()),'modern_volume_km3':float(v.modern_volume_km3.sum()),
 'volume_loss_deduplicated_km3':float(v.volume_loss_deduplicated_km3.sum()),
 'lia_geodesic_area_km2':float(a.lia_area_geodesic_km2.sum()),'modern_geodesic_area_km2':float(a.modern_area_geodesic_km2.sum()),
 'negative_volume_loss_systems':int(v.volume_loss_deduplicated_km3.lt(0).sum()),
 'overlap_removal_total_km3':float(v.lia_overlap_removed_km3.sum())}
assert abs(baseline['lia_volume_deduplicated_km3']-3351.2161671466)<1e-7
assert abs(baseline['modern_volume_km3']-3022.6269297718)<1e-7
assert abs(baseline['volume_loss_deduplicated_km3']-328.5892373748)<1e-7
assert abs(baseline['lia_geodesic_area_km2']-27209.3228041975)<1e-7
assert abs(baseline['modern_geodesic_area_km2']-25954.3288930462)<1e-7
rows=[]
selected=['YX-00006','YX-02442','YX-00568','YX-11977','YX-25071','YX-12988','YX-20287','YX-20368']
for rid in selected:
    row=v.loc[rid]; result=json.loads(Path(row.selected_result_path).read_text('utf-8'));files={}
    for name,expected in result['output_hashes'].items():
        if not name.endswith('.nc'):continue
        path=Path(name);assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        endpoint='LIA' if '_lia_' in name else 'modern';files[endpoint]=path
    with xr.open_dataset(files['LIA']) as src_l, xr.open_dataset(files['modern']) as src_m:
        l,m=xr.align(src_l[['thickness_m','bed_elevation_m','surface_elevation_m','valid_mask','lia_mask']],src_m[['thickness_m','bed_elevation_m','surface_elevation_m','valid_mask','lia_mask']],join='outer')
        dx=float(np.median(np.diff(src_l.x)));dy=float(np.median(np.diff(src_l.y)));cell=abs(dx*dy)
        assert abs(float(np.median(np.diff(src_m.x)))-dx)<1e-9 and abs(float(np.median(np.diff(src_m.y)))-dy)<1e-9
        hl=l.thickness_m.values.astype(float);hm=m.thickness_m.values.astype(float)
        # Filled Boolean masks must explicitly reject alignment NaNs.
        ml=(l.valid_mask.values==1)&(l.lia_mask.values==1)&np.isfinite(hl)
        mm=(m.valid_mask.values==1)&(m.lia_mask.values==1)&np.isfinite(hm)
        for support in ['valid_mask_and_lia_mask','positive_thickness_inside_masks']:
            ll=ml.copy();mn=mm.copy()
            if support=='positive_thickness_inside_masks':ll&=hl>0;mn&=hm>0
            common=ll&mn;only_l=ll&~mn;only_m=mn&~ll
            vl=np.sum(hl[ll],dtype=np.float64)*cell/1e9;vm=np.sum(hm[mn],dtype=np.float64)*cell/1e9
            vc=np.sum(hl[common]-hm[common],dtype=np.float64)*cell/1e9
            retired=np.sum(hl[only_l],dtype=np.float64)*cell/1e9;modern_only=np.sum(hm[only_m],dtype=np.float64)*cell/1e9
            db=l.bed_elevation_m.values.astype(float)-m.bed_elevation_m.values.astype(float)
            eligible=common&np.isfinite(db)&(abs(db)<=.002)
            ineligible=common&~eligible
            ds=l.surface_elevation_m.values.astype(float)-m.surface_elevation_m.values.astype(float)
            rec={'record_id':rid,'selected_model_version':int(row.selected_model_version),'support_definition':support,
              'dx_m':dx,'dy_m':dy,'cell_area_m2':cell,'aligned_shape':str(hl.shape),
              'LIA_native_float32_integral_km3':vl,'modern_native_float32_integral_km3':vm,
              'LIA_native_minus_reference_km3':vl-row.lia_volume_km3,'modern_native_minus_reference_km3':vm-row.modern_volume_km3,
              'common_support_area_km2':int(common.sum())*cell/1e6,'LIA_only_support_area_km2':int(only_l.sum())*cell/1e6,'modern_only_support_area_km2':int(only_m.sum())*cell/1e6,
              'common_thickness_change_volume_km3':vc,'LIA_only_volume_km3':retired,'modern_only_volume_km3':modern_only,
              'raw_volume_change_identity_residual_km3':(vl-vm)-(vc+retired-modern_only),
              'bed_consistent_common_area_km2':int(eligible.sum())*cell/1e6,
              'bed_consistent_common_thickness_change_volume_km3':np.sum((hl-hm)[eligible],dtype=np.float64)*cell/1e9,
              'bed_inconsistent_common_thickness_change_volume_km3':np.sum((hl-hm)[ineligible],dtype=np.float64)*cell/1e9,
              'bed_consistent_surface_change_volume_km3':np.sum(ds[eligible],dtype=np.float64)*cell/1e9,
              'bed_consistent_surface_change_mean_m':float(np.mean(ds[eligible])),
              'overlap_correction_km3':float(row.lia_overlap_removed_km3),
              'reference_deduplicated_volume_loss_km3':float(row.volume_loss_deduplicated_km3),
              'native_deduplicated_volume_loss_km3':vl-vm-row.lia_overlap_removed_km3,
              'source_LIA_path':str(files['LIA']),'source_modern_path':str(files['modern'])}
            assert abs(rec['raw_volume_change_identity_residual_km3'])<1e-10
            assert abs(rec['LIA_native_minus_reference_km3'])<1e-7 and abs(rec['modern_native_minus_reference_km3'])<1e-7
            rows.append(rec)
pd.DataFrame(rows).to_csv(OUT/'source_independent_raster_expectations.csv',index=False,encoding='utf-8-sig')
(OUT/'source_baseline_checks.json').write_text(json.dumps({'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'source_checks_pass_new_analysis_comparison_pending','canonical':baseline,'predetermined_sample':selected,'source_checks':'Checks use source selected654 tables and endpoint NetCDFs, without importing producer code. Both outline-mask and positive-thickness alternatives retained to compare to disclosed producer definition.'},ensure_ascii=False,indent=2),'utf-8')
print(json.dumps(baseline,ensure_ascii=False,indent=2));print('SAMPLE_SOURCE_PASS',len(selected))
