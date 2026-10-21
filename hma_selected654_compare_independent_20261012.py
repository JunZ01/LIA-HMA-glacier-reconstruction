"""Independent arithmetic and prespecified source-raster comparison."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import pandas as pd

B=Path(r'H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1')
P=B/'selected654_analysis';O=B/'selected654_independent_qc';S=Path(r'H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
v=pd.read_csv(S/'paired_volume_changes.csv').set_index('record_id')
a=pd.read_csv(S/'selected_inventory_area_and_modern_volume.csv').set_index('record_id')
l=pd.read_csv(r'H:/ToYangxin/result_yx/paired_axis_length_selected654_20261010_v2/paired_axis_lengths654.csv').set_index('record_id')
d=pd.read_csv(P/'spatial_volume_decomposition654.csv').set_index('record_id');new=pd.read_csv(P/'selected654_merged_analysis.csv').set_index('record_id')
reg=pd.read_csv(P/'mountain_volume_area_length_budget654.csv').set_index('mountain');summary=json.loads((P/'summary.json').read_text('utf-8'))
assert set(v.index)==set(a.index)==set(l.index)==set(d.index)==set(new.index) and len(d)==654
checks={}
def check(name,got,expected,tol=1e-9):
    diff=float(np.max(np.abs(np.asarray(got,dtype=float)-np.asarray(expected,dtype=float))))
    checks[name]={'max_absolute_difference':diff,'tolerance':tol,'pass':diff<=tol}
    assert diff<=tol,(name,diff,tol)
for col in ['lia_volume_deduplicated_km3','modern_volume_km3','volume_loss_deduplicated_km3','lia_overlap_removed_km3']:
    check('all654_frozen_'+col,new.loc[v.index,col],v[col],1e-10)
for col in ['lia_area_geodesic_km2','modern_area_geodesic_km2']:
    check('all654_frozen_'+col,new.loc[a.index,col],a[col],1e-10)
for col in ['LIA_max_stored_axis_m','modern_max_stored_axis_m','axis_length_decrease_m']:
    check('all654_frozen_'+col,new.loc[l.index,col],l[col],1e-8)
five=d.LIA_only_positive_volume_km3+d.common_positive_thickness_loss_km3-d.modern_only_positive_volume_km3+d.stored_negative_thickness_residual_km3-d.overlap_correction_km3
check('all654_five_term_identity',five,d.lia_integral_km3-d.modern_integral_km3-d.overlap_correction_km3,1e-10)
check('all654_saved_decomposed_identity',five,d.decomposed_published_net_loss_km3,1e-10)
check('all654_common_bed_split',d.common_positive_thickness_loss_km3,d.common_bed_consistent_thickness_loss_km3+d.common_bed_inconsistent_or_unknown_thickness_loss_km3,1e-10)
check('all654_overlap_correction',d.overlap_correction_km3,v.loc[d.index,'lia_overlap_removed_km3'],1e-15)
check('all654_native_raw_loss_to_frozen',five,v.loc[d.index,'volume_loss_deduplicated_km3'],1e-7)
assert int(new.volume_loss_deduplicated_km3.lt(0).sum())==36
for region,gg in v.groupby('mountain'):
    ii=gg.index; rr=reg.loc[region]
    expected={'systems':len(gg),'lia_area_km2':a.loc[ii].lia_area_geodesic_km2.sum(),'modern_area_km2':a.loc[ii].modern_area_geodesic_km2.sum(),
      'lia_volume_km3':gg.lia_volume_deduplicated_km3.sum(),'modern_volume_km3':gg.modern_volume_km3.sum(),'net_volume_loss_km3':gg.volume_loss_deduplicated_km3.sum(),
      'area_loss_percent':100*(a.loc[ii].lia_area_geodesic_km2.sum()-a.loc[ii].modern_area_geodesic_km2.sum())/a.loc[ii].lia_area_geodesic_km2.sum(),
      'volume_loss_percent':100*gg.volume_loss_deduplicated_km3.sum()/gg.lia_volume_deduplicated_km3.sum(),
      'axis_length_index_loss_percent':100*(l.loc[ii].LIA_max_stored_axis_m.sum()-l.loc[ii].modern_max_stored_axis_m.sum())/l.loc[ii].LIA_max_stored_axis_m.sum(),
      'volume_increase_systems':int(gg.volume_loss_deduplicated_km3.lt(0).sum())}
    for col,x in expected.items():check('region_'+region+'_'+col,rr[col],x,1e-8)
    for col in ['LIA_only_positive_volume_km3','common_positive_thickness_loss_km3','modern_only_positive_volume_km3','overlap_correction_km3']:
        check('region_'+region+'_'+col,rr[col],d.loc[ii,col].sum(),1e-9)
check('canonical1299_total_volume',summary['total']['net_volume_loss_km3'],328.5892373748,1e-7)
check('canonical1305_all654_length',summary['total']['axis_length_index_loss_percent'],10.3454864936,1e-8)

expected=pd.read_csv(O/'source_independent_raster_expectations.csv')
expected=expected[expected.support_definition.eq('positive_thickness_inside_masks')].set_index('record_id')
mapping={'LIA_only_positive_volume_km3':'LIA_only_volume_km3','modern_only_positive_volume_km3':'modern_only_volume_km3',
 'common_positive_thickness_loss_km3':'common_thickness_change_volume_km3','common_positive_area_km2':'common_support_area_km2',
 'LIA_only_positive_area_km2':'LIA_only_support_area_km2','modern_only_positive_area_km2':'modern_only_support_area_km2',
 'common_bed_consistent_area_km2':'bed_consistent_common_area_km2',
 'common_bed_consistent_thickness_loss_km3':'bed_consistent_common_thickness_change_volume_km3',
 'common_bed_inconsistent_or_unknown_thickness_loss_km3':'bed_inconsistent_common_thickness_change_volume_km3'}
for result_col,source_col in mapping.items():check('eight_native_NC_to_result_'+result_col,d.loc[expected.index,result_col],expected[source_col],1e-10)
edge=json.loads((O/'negative_thickness_edgecase_independent_v2.json').read_text('utf-8'))
check('negative_edgecase_cells',d.loc['YX-00134','stored_negative_thickness_cells'],edge['modern_negative_cells'],0)
check('negative_edgecase_signed_residual',d.loc['YX-00134','stored_negative_thickness_residual_km3'],edge['signed_negative_numerical_residual_km3'],1e-25)
check('negative_edgecase_common',d.loc['YX-00134','common_positive_thickness_loss_km3'],edge['common_positive_thickness_change_km3'],1e-10)

order=v.sort_values('lia_volume_deduplicated_km3',ascending=False)
cc=json.loads((P/'concentration_summary.json').read_text('utf-8'))
for row in cc:
    n=row['n_largest_by_initial_volume'];g=order.iloc[:n]
    check(f'concentration_{n}_volume_share',row['initial_volume_share_percent'],100*g.lia_volume_deduplicated_km3.sum()/v.lia_volume_deduplicated_km3.sum())
    check(f'concentration_{n}_net_loss_share',row['net_loss_share_percent'],100*g.volume_loss_deduplicated_km3.sum()/v.volume_loss_deduplicated_km3.sum())
    check(f'concentration_{n}_gross_positive_share',row['positive_loss_share_percent'],100*g.volume_loss_deduplicated_km3.clip(lower=0).sum()/v.volume_loss_deduplicated_km3.clip(lower=0).sum())

mp=pd.read_csv(S/'modern_product_support.csv');contrast=pd.read_csv(P/'modern_product_contrast_diagnostic.csv').set_index('record_id');prod=[]
for rid,g in mp.groupby('system_record_id'):
    positive=g.far_volume_km3.gt(0)&g.millan_volume_km3.gt(0)
    strict=positive & g.far_native_positive_area_fraction.ge(.9) & g.millan_positive_coverage_fraction.ge(.9)
    prod.append({'record_id':rid,'strict':bool(strict.all()),'f19':g.far_volume_km3.sum(),'millan':g.millan_volume_km3.sum()})
prod=pd.DataFrame(prod).set_index('record_id');assert set(prod.index)==set(v.index)
check('modern_product_strict_flags',prod.strict,contrast.loc[prod.index,'both_products_area90_all_members'],0)
check('modern_product_F19_sum',prod.f19,contrast.loc[prod.index,'F19_supported_modern_volume_km3'],1e-9)
check('modern_product_Millan_sum',prod.millan,contrast.loc[prod.index,'Millan_supported_modern_volume_km3'],1e-9)
strict=prod[prod.strict];higher=((strict.millan-strict.f19).abs()>v.loc[strict.index].volume_loss_deduplicated_km3.abs()).sum()
check('modern_product_strict_sample',len(strict),511,0);check('modern_product_difference_exceeds_change',higher,421,0)
hashes=json.loads((P/'input_table_sha256.json').read_text('utf-8'))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
result={'schema':'selected654-independent-result-review-v1','status':'pass','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'independence':'Separate implementation; frozen source tables, xarray coordinate alignment of source NCs and prespecified native raster expected values. Producer implementation not imported/read.',
 'all654_id_and_frozen_value_reconciliation':True,'all654_five_term_identity_and_regional_aggregation':True,'source_tables_unchanged':True,
 'predetermined_source_raster_samples':expected.index.tolist(),'additional_numerical_edgecase':'YX-00134','checks':checks,
 'max_full654_native_vs_reference_volume_difference_km3':float((five-v.loc[d.index,'volume_loss_deduplicated_km3']).abs().max()),
 'interpretation_review':['654 retained, 36 signed modeled gains retained; no excluded six reintroduced.',
 'Seven occupied mountains in new budget table; three Himalaya subdivisions have no selected654 sample and are not zero-change observations.',
 'Common-positive-support contribution is model-based thickness-volume difference; bed-inconsistent contribution stays separate.',
 'Surface-lowering mean is inherited common-bed-valid frozen footprint and differs from the new strictly positive common-support footprint; their denominators must not be exchanged.',
 'Modern F19-Millan comparison is native-supported product contrast, not physical uncertainty bound or alternative LIA result.']}
(O/'independent_analysis_QC.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2));print('CHECKS',len(checks))
