"""Read-only modern RGI7 covariates for the frozen selected654 cohort."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, subprocess
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT = Path('E:/3spyderProject/1_LIA_himalaya_oggm')
OUT = Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/climate_geometry_exploration')
SEL = Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
SRC = Path('H:/ToYangxin/result_yx/surface_reconstruction_20261009_v4/modern_rgi7_members.gpkg')
INPUTS = [SRC, SEL/'selected_modern_member_native_product_baseline.csv', SEL/'selected_inventory_area_and_modern_volume.csv']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n, v): (OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'geometry_summary.json').exists(): raise RuntimeError('Completed immutable output exists')
    canonical = {}
    for n in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
        p = ROOT/n; t = p.read_text(encoding='utf-8-sig')
        canonical[n] = {'sha256':sha(p),'full_read_characters':len(t)}
    r = subprocess.run([sys.executable,str(ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True,text=True,encoding='utf-8')
    v = json.loads(r.stdout); assert r.returncode == 0 and v['status'] == 'pass'
    canonical['validator'] = {k:v[k] for k in ['status','launch_authorized','counts','failures']}
    hashes = {str(p):sha(p) for p in INPUTS}
    raw = gpd.read_file(SRC).drop(columns='geometry')
    members = pd.read_csv(INPUTS[1]); inv = pd.read_csv(INPUTS[2])
    assert len(members) == 1256 and members.rgi_id.is_unique
    assert len(inv) == 654 and inv.record_id.is_unique
    assert raw.rgi_id.is_unique and set(members.rgi_id).issubset(set(raw.rgi_id))
    fields = ['zmin_m','zmax_m','zmean_m','zmed_m','slope_deg','aspect_deg']
    extra = ['cenlon','cenlat','src_date','dem_source','area_km2','lia_unique_id']
    m = members.merge(raw[['rgi_id']+fields+extra],on='rgi_id',how='left',validate='one_to_one',suffixes=('','_rgi7'))
    assert (m.lia_unique_id == m.lia_unique_id_rgi7).all()
    m['modern_member_relief_m'] = m.zmax_m-m.zmin_m
    m['valid_elevation'] = m[fields[:4]].apply(np.isfinite).all(axis=1) & (m.zmin_m>-500) & (m.zmax_m<10000) & (m.zmin_m<=m.zmed_m) & (m.zmed_m<=m.zmax_m) & (m.zmin_m<=m.zmean_m) & (m.zmean_m<=m.zmax_m)
    m['valid_slope'] = np.isfinite(m.slope_deg) & m.slope_deg.between(0,90)
    m['valid_aspect'] = np.isfinite(m.aspect_deg) & m.aspect_deg.between(0,360,inclusive='left')
    m['member_weight'] = m.member_area_geodesic_km2/m.groupby('system_record_id').member_area_geodesic_km2.transform('sum')
    assert (m.member_area_geodesic_km2>0).all()
    records = []
    for sid,g in m.groupby('system_record_id',sort=True):
        w = g.member_weight.to_numpy(); a = float(g.member_area_geodesic_km2.sum())
        b = inv.set_index('record_id').loc[sid]
        row = {'record_id':sid,'lia_unique_id':int(b.lia_unique_id),'mountain':b.mountain,
               'modern_member_count':len(g),'modern_summed_member_area_km2':a,'modern_system_union_area_km2':float(b.modern_area_geodesic_km2),
               'LIA_area_geodesic_km2':float(b.lia_area_geodesic_km2),
               'modern_largest_member_area_fraction':float(g.member_area_geodesic_km2.max()/a),
               'modern_member_weight_sum':float(w.sum()),
               'modern_source_date_min':g.src_date.min(),'modern_source_date_max':g.src_date.max(),
               'modern_dem_sources':';'.join(sorted(g.dem_source.unique())),
               'modern_elevation_available_area_fraction':float(w[g.valid_elevation.to_numpy()].sum()),
               'modern_slope_available_area_fraction':float(w[g.valid_slope.to_numpy()].sum()),
               'modern_aspect_available_area_fraction':float(w[g.valid_aspect.to_numpy()].sum()),
               'modern_all_members_elevation_valid':bool(g.valid_elevation.all()),
               'modern_all_members_slope_valid':bool(g.valid_slope.all()),
               'modern_all_members_aspect_valid':bool(g.valid_aspect.all())}
        for f in fields[:4]+['modern_member_relief_m']:
            row['modern_'+f.replace('modern_','')+'_member_area_weighted'] = float(np.dot(w,g[f])) if g.valid_elevation.all() else None
        row['modern_slope_deg_member_area_weighted'] = float(np.dot(w,g.slope_deg)) if g.valid_slope.all() else None
        row['modern_system_zmin_m'] = float(g.zmin_m.min()) if g.valid_elevation.all() else None
        row['modern_system_zmax_m'] = float(g.zmax_m.max()) if g.valid_elevation.all() else None
        row['modern_system_elevation_range_m'] = row['modern_system_zmax_m']-row['modern_system_zmin_m'] if g.valid_elevation.all() else None
        if g.valid_aspect.all():
            north = float(np.dot(w,np.cos(np.deg2rad(g.aspect_deg))))
            east = float(np.dot(w,np.sin(np.deg2rad(g.aspect_deg))))
            row.update(modern_aspect_northness_member_area_weighted=north,modern_aspect_eastness_member_area_weighted=east,
                       modern_aspect_member_direction_resultant_length=float(np.hypot(north,east)),
                       modern_aspect_member_direction_deg=float(np.degrees(np.arctan2(east,north))%360) if np.hypot(north,east)>1e-12 else None)
        else:
            for c in ['northness_member_area_weighted','eastness_member_area_weighted','member_direction_resultant_length','member_direction_deg']: row['modern_aspect_'+c]=None
        records.append(row)
    d = pd.DataFrame(records)
    assert len(d)==654 and set(d.record_id)==set(inv.record_id)
    assert d.modern_member_count.sum()==1256
    assert np.allclose(d.modern_member_weight_sum,1,rtol=0,atol=1e-12)
    assert np.allclose(d.modern_summed_member_area_km2,d.modern_system_union_area_km2,rtol=0,atol=1e-4)
    fields_dict = [
        {'source_fields':'zmin_m,zmax_m,zmed_m,zmean_m','source':str(SRC),'output':'modern_*_member_area_weighted','operator':'sum(member_geodesic_area * attribute)/sum(member_geodesic_area); strict all-member validity','units':'m','caveat':'Mean of member minima/maxima/medians is not the system minimum/maximum/median; true min/max also supplied.'},
        {'source_fields':'zmin_m,zmax_m','source':str(SRC),'output':'modern_member_relief_m_member_area_weighted','operator':'area weighted mean of (member zmax-member zmin)','units':'m','caveat':'Modern DEM-derived member relief; not LIA relief.'},
        {'source_fields':'slope_deg','source':str(SRC),'output':'modern_slope_deg_member_area_weighted','operator':'member-area-weighted source mean slope','units':'degree','caveat':'Modern DEM surface context. Do not assume unchanged LIA slope or independent causal exposure.'},
        {'source_fields':'aspect_deg','source':str(SRC),'output':'modern_aspect_northness/eastness_member_area_weighted','operator':'sum(w*cos(aspect)),sum(w*sin(aspect)); degree clockwise from north','units':'dimensionless','caveat':'Weighted member direction proxy; member internal resultant magnitudes unavailable, so not full system pixel-pooled mean aspect.'},
        {'source_fields':'member_area_geodesic_km2','source':str(INPUTS[1]),'output':'modern_summed_member_area_km2,modern_largest_member_area_fraction,modern_member_count','operator':'sum, maximum/sum, count; all1256 frozen members retained','units':'km2,fraction,count','caveat':'No missing-thickness member dropped; tiny source member overlaps retained. Separate frozen union area supplied.'},
        {'source_fields':'lia_area_geodesic_km2,modern_area_geodesic_km2,mountain','source':str(INPUTS[2]),'output':'LIA_area_geodesic_km2,modern_system_union_area_km2,mountain','operator':'exact frozen record_id join','units':'km2,km2,label','caveat':'LIA area is shared with reconstructed volume geometry; predictive comparison cannot establish independent physical mechanism.'},
    ]
    protocol = {'created_utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'canonical_full_read':canonical,
        'sample':'selected654,1256 unique modern members; all retained; no physical model run',
        'sources':hashes,'official_attribute_documentation':'https://www.glims.org/rgi_user_guide/products/glacier_product.html',
        'geometry_epoch':'RGI7 source outlines1993–2008; DEM products modern; no LIA-invariance assumption',
        'weights':'Frozen member WGS84 geodesic areas divided by system sum. This averages member summaries, not original DEM cells.',
        'missing_policy':'Never impute source covariates in extraction; strict all-member validity by field; unavailable-area fractions supplied.',
        'validity_rules':'Elevations finite, -500<zmin<=zmean/zmed<=zmax<10000 m; slope0–90 inclusive; aspect0–360 excluding360; zero denominators forbidden.',
        'aspect':'Use weighted cos and sin of member source mean directions. Angle derived only if resultant magnitude>1e-12. Not pooled DEM mean direction.',
        'identity':'Every source rgi_id matched exactly once against frozen selected member list; source-only excluded7members are not reintroduced.',
        'CV_authorization':'Root authorized statistical descriptive leave-one-mountain prediction comparison; frozen separate cv_protocol.json before fit.'}
    dump('geometry_protocol.json',protocol)
    dump('geometry_fieldpaths.json',fields_dict)
    d.to_csv(OUT/'geometry_covariates654.csv',index=False,encoding='utf-8-sig')
    m.to_csv(OUT/'geometry_source_members1256.csv',index=False,encoding='utf-8-sig')
    unchanged = {p:sha(p)==h for p,h in hashes.items()}; assert all(unchanged.values())
    summary = {'status':'complete','systems':len(d),'modern_members':len(m),'mountain_counts':{str(k):int(v) for k,v in d.mountain.value_counts().items()},
       'missing_elevation_systems':int((~d.modern_all_members_elevation_valid).sum()),'missing_slope_systems':int((~d.modern_all_members_slope_valid).sum()),'missing_aspect_systems':int((~d.modern_all_members_aspect_valid).sum()),
       'max_abs_weight_sum_error':float(abs(d.modern_member_weight_sum-1).max()),'max_abs_summed_vs_union_area_difference_km2':float(abs(d.modern_summed_member_area_km2-d.modern_system_union_area_km2).max()),
       'source_files_unchanged':unchanged,'modern_dem_source_counts':{str(k):int(v) for k,v in m.dem_source.value_counts().items()},
       'date_min':str(m.src_date.min()),'date_max':str(m.src_date.max()),'no_model_runs':True,
       'limitations':['All terrain descriptors are modern inventory attributes, not measured LIA geometry.','Geometric predictors partly enter the reconstruction, so model performance is not independent mechanistic evidence.','Weighted member median elevation is not the system pooled median.','Area-weighted member aspect vector is a direction proxy, not pixel-pooled aspect.']}
    dump('geometry_summary.json',summary)
    dump('geometry_outputs_sha256.json',{p.name:sha(p) for p in OUT.glob('geometry_*') if p.name!='geometry_outputs_sha256.json'})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
