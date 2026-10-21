"""Companion union-area diagnostics, preserving source system definitions."""
from hma_modern_identity_audit_20261012 import *

def main():
    if (OUT/'union_supplement_summary.json').exists():raise RuntimeError('Immutable supplement already exists')
    prev=json.loads((OUT/'input_sha256.json').read_text())
    assert all(sha(p)==s for p,s in prev.items())
    lia=load_geo(OLD/'Him_10_lia_with_large.shp','UniqueID').sort_values('UniqueID').reset_index(drop=True)
    oldorig=load_geo(OLD/'Him_10_rgi_related_lia.shp','RGIId').set_index('RGIId')
    oldfull=load_geo(BASE/'velocity_work/target_rgi60_1205_esri102027.gpkg','RGIId').set_index('RGIId')
    scenarios=pd.read_csv(OUT/'legacy245_endpoint_area_scenarios.csv')
    reg=pd.read_csv(OUT/'all2461_modern_member_registry.csv');oldreg=reg[reg.cohort=='legacy245'].set_index('rgi_id')
    ints=pd.read_csv(OUT/'legacy1205_modern_members_internal_intersections.csv')
    ints['left_lia_id']=ints.left_id.map(oldreg.lia_unique_id);ints['right_lia_id']=ints.right_id.map(oldreg.lia_unique_id)
    ints['same_lia_system']=ints.left_lia_id==ints.right_lia_id
    ints['fraction_left_member']=ints.intersection_km2/ints.left_id.map(oldreg.area_geodesic_km2)
    ints['fraction_right_member']=ints.intersection_km2/ints.right_id.map(oldreg.area_geodesic_km2)
    csv('legacy1205_internal_overlap_system_attribution.csv',ints)
    ilia=internal_overlaps(lia,'UniqueID');csv('legacy245_lia_internal_intersections.csv',ilia,['left_id','right_id','intersection_km2'])
    ldict=lia.set_index('UniqueID').geometry.to_dict();agg=[]
    for scenario,d in scenarios.groupby('scenario'):
        source=oldorig if scenario=='original_1159' else oldfull
        gs={int(r.lia_unique_id):unary_union([source.loc[x].geometry for x in r.modern_member_ids.split(';')]) for _,r in d.iterrows()}
        for label,sub in [('ALL_LEGACY245',d)]+list(d.groupby('mountain10')):
            u=unary_union([gs[int(k)] for k in sub.lia_unique_id]);lu=unary_union([ldict[int(k)] for k in sub.lia_unique_id]);a=area(lu);ma=area(u)
            agg.append({'scenario':scenario,'region':label,'systems':len(sub),'lia_summed_system_area_km2':sub.lia_area_geodesic_km2.sum(),'lia_spatial_union_area_km2':a,'lia_sum_minus_spatial_union_km2':sub.lia_area_geodesic_km2.sum()-a,'modern_summed_system_union_area_km2':sub.modern_union_area_geodesic_km2.sum(),'modern_spatial_union_area_km2':ma,'modern_sum_minus_spatial_union_km2':sub.modern_union_area_geodesic_km2.sum()-ma,'spatial_union_signed_area_reduction_km2':a-ma,'spatial_union_area_reduction_relative_lia_pct':100*(a-ma)/a})
    csv('legacy245_spatial_union_area_scenarios.csv',agg)
    crossed=ints[~ints.same_lia_system];n=reg.groupby(['cohort','system_record_id']).size()
    summary={'status':'complete-geometry-union-supplement','source_systems_reassigned_or_dropped':False,'old245_internal_same_RGI_id_shared_between_systems':bool(oldreg.index.duplicated().any()),'old245_modern_different_RGI6_cross_system_positive_area_pairs':len(crossed),'old245_modern_cross_system_pair_intersection_sum_km2':crossed.intersection_km2.sum(),'old245_modern_cross_system_member_fraction_max':float(max(crossed.fraction_left_member.max(),crossed.fraction_right_member.max())),'old245_lia_internal_positive_area_pairs':len(ilia),'old245_lia_internal_pair_intersection_sum_km2':sum(x['intersection_km2'] for x in ilia),'all245_union_rows':[r for r in agg if r['region']=='ALL_LEGACY245'],'selected654_multi_modern_member_systems':int((n.loc['selected654']>1).sum()),'old245_download_multi_modern_member_systems':int((n.loc['legacy245']>1).sum()),'interpretation':['RGI identifiers are exclusive within each cohort, but distinct old RGI6 footprints can partially overlap across LIA systems.','Polygon spatial unions give footprint-area totals without assigning shared ice to a specific system; they do not deduplicate modeled ice volume or establish that partial overlaps are duplicate glacier systems.','Main scenario CSV keeps unchanged individual system definitions and summed system areas. This supplement explicitly provides cross-system spatial-union totals as a distinct metric.','Membership scenarios and the2692 candidate ambiguity remain; no automatic member removal or split/merge decision was made.','The selected654 and old245 modern footprints have zero cross-cohort positive-area intersection, unchanged from first audit.'],'source_hashes_unchanged':all(sha(p)==s for p,s in prev.items())}
    jdump('union_supplement_summary.json',summary)
    jdump('union_supplement_sha256.json',{f:sha(OUT/f) for f in ['legacy1205_internal_overlap_system_attribution.csv','legacy245_lia_internal_intersections.csv','legacy245_spatial_union_area_scenarios.csv','union_supplement_summary.json']})
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
