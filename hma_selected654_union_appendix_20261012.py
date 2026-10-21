"""Read-only polygon union sensitivity; frozen system-sum results untouched."""
from hma_modern_identity_audit_20261012 import *

def main():
    final=OUT/'harmonized899_spatial_union_area_scenarios.csv'
    assert not final.exists()
    path=SEL/'selected_reconstruction_results.gpkg'
    lia=load_geo(path,'record_id','lia_selected654')
    modern=load_geo(path,'record_id','modern_selected654')
    mountains=gpd.read_file(path,layer='local_mountains_10')[['NAME','Chinese']]
    order=mountains.NAME.tolist();zh=dict(zip(mountains.NAME,mountains.Chinese))
    frozen=pd.read_csv(SEL/'selected_inventory_area_and_modern_volume.csv')
    rows=[]
    for name in order+['ALL_SELECTED654']:
        l=lia if name=='ALL_SELECTED654' else lia[lia.mountain==name]
        m=modern if name=='ALL_SELECTED654' else modern[modern.mountain==name]
        f=frozen if name=='ALL_SELECTED654' else frozen[frozen.mountain==name]
        la=area(unary_union(l.geometry));ma=area(unary_union(m.geometry))
        ls=sum(area(x) for x in l.geometry);ms=sum(area(x) for x in m.geometry)
        fl=f.lia_area_geodesic_km2.sum();fm=f.modern_area_geodesic_km2.sum()
        assert abs(ls-fl)<1e-5 and abs(ms-fm)<1e-5
        rows.append({'region':name,'region_zh':zh.get(name,'654全部'),'selected654_systems':len(l),'lia_summed_system_area_km2':fl,'lia_spatial_union_area_km2':la,'lia_sum_minus_spatial_union_km2':fl-la,'modern_summed_system_union_area_km2':fm,'modern_spatial_union_area_km2':ma,'modern_sum_minus_spatial_union_km2':fm-ma,'spatial_union_signed_area_reduction_km2':la-ma,'spatial_union_area_reduction_relative_lia_pct':100*(la-ma)/la if la else None,'union_role':'overlap_sensitivity_diagnostic; frozen per-system sums unchanged'})
    selected=csv('selected654_spatial_union_area_diagnostic.csv',rows).set_index('region')
    old=pd.read_csv(OUT/'legacy245_spatial_union_area_scenarios.csv')
    allrows=[]
    for scenario,sc in old.groupby('scenario'):
        sc=sc.set_index('region')
        for name in order+['ALL_899_SOURCE_SYSTEMS']:
            allflag=name=='ALL_899_SOURCE_SYSTEMS';s=selected.loc['ALL_SELECTED654' if allflag else name]
            oldname='ALL_LEGACY245' if allflag else name
            o=sc.loc[oldname] if oldname in sc.index else None
            def ov(k):return 0 if o is None else o[k]
            la=s.lia_spatial_union_area_km2+ov('lia_spatial_union_area_km2');ma=s.modern_spatial_union_area_km2+ov('modern_spatial_union_area_km2')
            ls=s.lia_summed_system_area_km2+ov('lia_summed_system_area_km2');ms=s.modern_summed_system_union_area_km2+ov('modern_summed_system_union_area_km2')
            allrows.append({'legacy_scenario':scenario,'region':name,'region_zh':'899来源系统' if allflag else zh[name],'selected654_systems':int(s.selected654_systems),'legacy245_systems':int(ov('systems')),'source_system_records':int(s.selected654_systems+ov('systems')),'selected654_lia_union_km2':s.lia_spatial_union_area_km2,'selected654_modern_union_km2':s.modern_spatial_union_area_km2,'legacy245_lia_union_km2':ov('lia_spatial_union_area_km2'),'legacy245_modern_union_km2':ov('modern_spatial_union_area_km2'),'combined_lia_summed_system_area_km2':ls,'combined_lia_spatial_union_area_km2':la,'combined_lia_sum_minus_union_km2':ls-la,'combined_modern_summed_system_union_area_km2':ms,'combined_modern_spatial_union_area_km2':ma,'combined_modern_sum_minus_union_km2':ms-ma,'combined_spatial_union_signed_area_reduction_km2':la-ma,'combined_spatial_union_area_reduction_relative_lia_pct':100*(la-ma)/la if la else None,'modern_reference_year_range':'1993-2010; unequal source epochs','population_definition':'899 source systems; heterogeneous size-selection rules; not an HMA census','aggregation_definition':'sum of cross-source disjoint cohort spatial unions; diagnostics only; no system reassignment or volume deduplication'})
    d=csv(final.name,allrows)
    assert len(d)==33 and set(d.groupby('legacy_scenario').source_system_records.sum())=={1798}
    totals=d[d.region=='ALL_899_SOURCE_SYSTEMS'];assert (totals.source_system_records==899).all()
    report={'status':'complete-selected654-and899-spatial-union-sensitivity','source_systems_unchanged':True,'canonical_values_replaced':False,'model_runs':0,'selected654_total':{k:(None if pd.isna(v) else v.item() if hasattr(v,'item') else v) for k,v in selected.loc['ALL_SELECTED654'].items()},'combined899_totals':totals.to_dict('records'),'cross_cohort_zero_overlap_basis':'previous coverage cross_source_overlap_candidates.csv for LIA; modern_identity cross_version_member_geometry_intersections.csv and cross_cohort_modern_system_intersections.csv for modern','additional_geometry_repair_events':REPAIRS,'notes':['Per-mountain whole-system unions retain existing mountain assignment; polygons are not clipped to mountain boundaries.','All654 and all899 totals use whole-cohort unions, not unverified sums of mountain union rows.','Legacy modern original1159/confirmed1196/download1205 scenarios remain separate, including2692 candidate uncertainty.','Unknown glacier-specific LIA dates and modern1993-2010 source-date differences preclude annual rates.']}
    jdump('selected654_and899_union_appendix_summary.json',report)
    jdump('selected654_and899_union_appendix_sha256.json',{f:sha(OUT/f) for f in ['selected654_spatial_union_area_diagnostic.csv',final.name,'selected654_and899_union_appendix_summary.json']})
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
