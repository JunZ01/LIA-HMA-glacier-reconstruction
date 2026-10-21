"""Frozen-result postprocessing only. No glacier model construction or advancement."""
from pathlib import Path
import json,hashlib,math,shutil,time
import numpy as np,pandas as pd,rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/selected654_analysis')
OUT.mkdir(parents=True,exist_ok=True)
SRC=Path('H:/ToYangxin/result_yx/selected_reconstruction_20261010_v1_654')
LEN=Path('H:/ToYangxin/result_yx/paired_axis_length_selected654_20261010_v2/paired_axis_lengths654.csv')
def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8')
def rec(d):return json.loads(d.to_json(orient='records',force_ascii=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol={
 'scope':'read-only postprocessing of frozen selected654; no new simulation or acceptance',
 'population':654,'excluded_six_remain_excluded':True,
 'area':'WGS84 source per-system areas, sum across systems as in frozen delivery; not spatial-union census',
 'volume':'use lia_volume_deduplicated_km3 and volume_loss_deduplicated_km3; keep signed gains; modern model calibration is not independent validation',
 'length':'all654 independently maximum stored head-to-root endpoint axes; not physical terminus retreat',
 'regional_percent':'100*sum(LIA-modern)/sum(LIA), never unweighted mean of individual percentages',
 'raster_alignment':'identical CRS, pixel size and rotation; exact integer pixel offsets onto already published difference-raster grid; no interpolation',
 'partition_operator':'positive thickness >0 on published valid difference domain: LIA-positive/modern-zero; both-positive; modern-positive/LIA-zero. Zero-thickness boundary pixels contribute no ice. These are modeled ice-support partitions, not vector retreat zones.',
 'nodata':'published difference nodata cells excluded, never turned into observed zero; nonpositive or absent endpoint support is zero only in published known difference domain',
 'decomposition_identity':'raw deltaV = LIA-only positive volume + common-positive thickness difference - modern-only positive volume + stored-negative-thickness signed residual; frozen overlap correction is separate for published net deltaV',
 'technical_amendment':'Initial strict nonnegative assertion found 15 cells in YX-00134 modern source at minimum -2.2667188e-9 m (NC and TIFF agree). Original protocol retained. Source raw thickness and sums unchanged; positive-support partition now carries all stored negative values as an explicit algebraic residual, never hidden/clipped from endpoint totals.',
 'bed_consistency':'common-positive thickness difference always algebraic; interpret as surface lowering only on paired finite bed agreement<=0.002 m, following frozen common-bed criterion; others separate',
 'numerical_tolerances':'integer offset and spacing 1e-7 pixels; volume max(1e-7 km3,16*float32eps*(Vlia+Vmodern)); decomposition arithmetic 1e-10 km3',
 'concentration':'order by initial LIA volume, report top10/topceil(10%N) shares; gross-positive and net losses separately; no response-selected eligibility',
 'sensitivity':'retain full654 primary; additional native-thickness-complete and common-bed>=95% and dominant v0.2 family summaries are diagnostic comparisons, not new sample exclusions',
 'uncertainty':'no formal physical confidence interval; spatial quantiles and sensitivity subsets are not uncertainty bounds',
 'climate':'no climate causal attribution or annual rate in this postprocessing',
 'source_dates':'selected modern outlines 1993-2008; LIA dates heterogeneous',
}
if (OUT/'analysis_protocol.json').exists() and not (OUT/'analysis_protocol_v1_initial.json').exists():shutil.copy2(OUT/'analysis_protocol.json',OUT/'analysis_protocol_v1_initial.json')
dump('analysis_protocol.json',protocol)
inputs=['paired_volume_changes.csv','selected_inventory_area_and_modern_volume.csv','common_ice_surface_change_statistics.csv','common_grid_bed_alignment_diagnostics.csv','gis_raster_index.csv','thickness_change_raster_QC.csv','modern_product_support.csv']
sourcehash={str(SRC/n):sha(SRC/n) for n in inputs};sourcehash[str(LEN)]=sha(LEN);dump('input_table_sha256.json',sourcehash)
v=pd.read_csv(SRC/'paired_volume_changes.csv')
a=pd.read_csv(SRC/'selected_inventory_area_and_modern_volume.csv')
l=pd.read_csv(LEN)
c=pd.read_csv(SRC/'common_ice_surface_change_statistics.csv')
b=pd.read_csv(SRC/'common_grid_bed_alignment_diagnostics.csv')
gi=pd.read_csv(SRC/'gis_raster_index.csv');tq=pd.read_csv(SRC/'thickness_change_raster_QC.csv')
assert len(v)==654 and v.record_id.is_unique
d=v.merge(a[['record_id','lia_area_geodesic_km2','modern_area_geodesic_km2','area_difference_LIA_minus_modern_km2','area_decrease_relative_LIA_percent','modern_F19_all_members_have_positive_native_support','modern_F19_missing_or_unrepresented_member_count','modern_containment_fraction']],on='record_id',validate='one_to_one')
d=d.merge(l[['record_id','LIA_max_stored_axis_m','modern_max_stored_axis_m','axis_length_decrease_m','axis_length_decrease_relative_LIA_percent','previous_485_subset_membership']],on='record_id',validate='one_to_one')
d=d.merge(c[['record_id','common_ice_area_km2','bed_consistent_common_area_fraction','mean_modeled_ice_surface_lowering_m']],on='record_id',validate='one_to_one')
d=d.merge(b,on='record_id',validate='one_to_one')

def read_aligned(path,grid):
 with rasterio.open(path) as q:
  assert q.crs==grid.crs and np.allclose([q.transform.a,q.transform.b,q.transform.d,q.transform.e],[grid.transform.a,grid.transform.b,grid.transform.d,grid.transform.e],rtol=0,atol=1e-7)
  col,row=(~grid.transform)*(q.transform.c,q.transform.f)
  assert abs(col-round(col))<1e-7 and abs(row-round(row))<1e-7,(path,col,row)
  col,row=int(round(col)),int(round(row))
  assert row>=0 and col>=0 and row+q.height<=grid.height and col+q.width<=grid.width
  arr=q.read(1,masked=True);target=np.full(grid.shape,np.nan,dtype=np.float64)
  target[row:row+q.height,col:col+q.width]=arr.filled(np.nan)
  return target

decomp=[];rasterchecks=[];start=time.monotonic()
for n,r in enumerate(d.itertuples(),1):
 paths=gi[gi.record_id==r.record_id]
 def find(suffix):
  x=paths[paths.delivery_path.str.endswith(suffix)]
  assert len(x)==1,(r.record_id,suffix,len(x))
  return x.iloc[0]
 diff=find('_LIA_minus_modern_thickness_change_m.tif')
 endpoint={key:find(suf) for key,suf in [('L','_lia_farinotti2019_thickness_m.tif'),('M','_modern_farinotti2019_thickness_m.tif'),('LB','_lia_farinotti2019_bed_elevation_m.tif'),('MB','_modern_farinotti2019_bed_elevation_m.tif')]}
 for x in [diff,*endpoint.values()]:
  actual=sha(x.delivery_path);assert actual==x.sha256,(r.record_id,x.delivery_path,'SHA mismatch')
  rasterchecks.append({'record_id':r.record_id,'path':x.delivery_path,'sha256':actual})
 with rasterio.open(diff.delivery_path) as grid:
  z=grid.read(1,masked=True);valid=~np.ma.getmaskarray(z);z=z.filled(0).astype(float)
  L0=read_aligned(endpoint['L'].delivery_path,grid);M0=read_aligned(endpoint['M'].delivery_path,grid)
  L=np.nan_to_num(L0,nan=0);M=np.nan_to_num(M0,nan=0)
  rawL=L.copy();rawM=M.copy();L=np.maximum(L,0);M=np.maximum(M,0)
  cell=abs(grid.transform.a*grid.transform.e-grid.transform.b*grid.transform.d)
  con=cell/1e9
  common=valid&(L>0)&(M>0);lost=valid&(L>0)&(M==0);new=valid&(L==0)&(M>0)
  LB=read_aligned(endpoint['LB'].delivery_path,grid);MB=read_aligned(endpoint['MB'].delivery_path,grid)
  bedok=common&np.isfinite(LB)&np.isfinite(MB)&(np.abs(LB-MB)<=.002)
  def integ(x,mask):return float(np.sum(x[mask],dtype=np.float64)*con)
  lv=integ(rawL,valid);mv=integ(rawM,valid);commonloss=integ(L-M,common);lostv=integ(L,lost);newv=integ(M,new)
  residual=integ(np.minimum(rawL,0)-np.minimum(rawM,0),valid)
  raw=integ(z,valid);calc=lostv+commonloss-newv+residual;corrected=calc-r.lia_overlap_removed_km3
  tol=max(1e-7,16*np.finfo(np.float32).eps*(r.lia_volume_km3+r.modern_volume_km3))
  assert abs(lv-r.lia_volume_km3)<=tol and abs(mv-r.modern_volume_km3)<=tol
  assert abs(calc-raw)<=tol and abs(calc-r.volume_loss_km3)<=tol
  assert abs(corrected-r.volume_loss_deduplicated_km3)<=tol
  assert abs((lv-mv)-calc)<1e-10
  decomp.append({'record_id':r.record_id,'mountain':r.mountain,'lia_integral_km3':lv,'modern_integral_km3':mv,'LIA_only_positive_volume_km3':lostv,'common_positive_thickness_loss_km3':commonloss,'modern_only_positive_volume_km3':newv,'stored_negative_thickness_residual_km3':residual,'stored_minimum_LIA_thickness_m':float(rawL.min()),'stored_minimum_modern_thickness_m':float(rawM.min()),'stored_negative_thickness_cells':int(((rawL<0)&valid).sum()+((rawM<0)&valid).sum()),'overlap_correction_km3':r.lia_overlap_removed_km3,'decomposed_published_net_loss_km3':corrected,'published_net_loss_km3':r.volume_loss_deduplicated_km3,'closure_error_km3':corrected-r.volume_loss_deduplicated_km3,'common_positive_area_km2':float(common.sum()*cell/1e6),'LIA_only_positive_area_km2':float(lost.sum()*cell/1e6),'modern_only_positive_area_km2':float(new.sum()*cell/1e6),'common_bed_consistent_area_km2':float(bedok.sum()*cell/1e6),'common_bed_consistent_thickness_loss_km3':integ(L-M,bedok),'common_bed_inconsistent_or_unknown_thickness_loss_km3':integ(L-M,common&~bedok),'numerical_volume_tolerance_km3':tol,'grid_shape':str(grid.shape),'grid_dx_m':grid.transform.a,'unknown_excluded_component_cells':int(tq.loc[tq.record_id==r.record_id,'unknown_excluded_component_cells'].iloc[0])})
 if n%50==0:print(json.dumps({'processed':n,'of':654,'elapsed_s':round(time.monotonic()-start,1)}),flush=True)
dd=pd.DataFrame(decomp);dd.to_csv(OUT/'spatial_volume_decomposition654.csv',index=False,encoding='utf-8-sig')
dump('raster_input_sha256.json',rasterchecks)
d=d.merge(dd.drop(columns='mountain'),on='record_id',validate='one_to_one')

def stats(s):
 al=s.lia_area_geodesic_km2.sum();am=s.modern_area_geodesic_km2.sum();vl=s.lia_volume_deduplicated_km3.sum();vm=s.modern_volume_km3.sum();net=vl-vm
 ll=s.LIA_max_stored_axis_m.sum();lm=s.modern_max_stored_axis_m.sum()
 x={'systems':len(s),'lia_area_km2':al,'modern_area_km2':am,'area_loss_km2':al-am,'area_loss_percent':100*(al-am)/al,'lia_volume_km3':vl,'modern_volume_km3':vm,'net_volume_loss_km3':net,'volume_loss_percent':100*net/vl,'axis_length_index_loss_percent':100*(ll-lm)/ll,'median_individual_volume_loss_percent':s.volume_loss_deduplicated_percent.median(),'median_individual_area_loss_percent':s.area_decrease_relative_LIA_percent.median(),'median_individual_axis_loss_percent':s.axis_length_decrease_relative_LIA_percent.median(),'volume_increase_systems':int((s.volume_loss_deduplicated_km3<0).sum()),'volume_loss_systems':int((s.volume_loss_deduplicated_km3>0).sum()),'gross_positive_loss_km3':s.volume_loss_deduplicated_km3.clip(lower=0).sum(),'gross_modeled_gain_km3':-s.volume_loss_deduplicated_km3.clip(upper=0).sum(),'equivalent_loss_over_LIA_area_m':1000*net/al,'native_thickness_missing_systems':int((~s.modern_F19_all_members_have_positive_native_support).sum()),'common_surface_lowering_weighted_m':float(np.average(s.mean_modeled_ice_surface_lowering_m,weights=s.common_ice_area_km2))}
 for k in ['LIA_only_positive_volume_km3','common_positive_thickness_loss_km3','modern_only_positive_volume_km3','stored_negative_thickness_residual_km3','overlap_correction_km3','common_bed_consistent_thickness_loss_km3','common_bed_inconsistent_or_unknown_thickness_loss_km3']:
  x[k]=float(s[k].sum())
 x['common_positive_share_of_net_loss_percent']=100*x['common_positive_thickness_loss_km3']/net
 return x
total=stats(d);regs=pd.DataFrame([{'mountain':k,**stats(s)} for k,s in d.groupby('mountain')])
regs['net_loss_share_percent']=100*regs.net_volume_loss_km3/total['net_volume_loss_km3']
regs=regs.sort_values('net_volume_loss_km3',ascending=False)
regs.to_csv(OUT/'mountain_volume_area_length_budget654.csv',index=False,encoding='utf-8-sig')
sens=[]
selections=[('全654（主结果）',np.ones(len(d),bool)),('原生冰厚成员无缺项（诊断）',d.modern_F19_all_members_have_positive_native_support),('共同冰区床一致面积≥95%（诊断）',d.bed_consistent_common_area_fraction>=.95),('v0.2主模型族（诊断）',d.model_family=='v0.2 raw-product bed modern2D volume closure')]
for label,mask in selections:sens.append({'selection':label,**stats(d[mask])})
pd.DataFrame(sens).to_csv(OUT/'sensitivity_summaries.csv',index=False,encoding='utf-8-sig')
families=pd.DataFrame([{'model_family':k,**stats(s)} for k,s in d.groupby('model_family')]);families.to_csv(OUT/'model_family_summaries.csv',index=False,encoding='utf-8-sig')
rank=d.sort_values('lia_volume_deduplicated_km3',ascending=False).copy();rank['size_rank']=np.arange(1,len(rank)+1);rank['cumulative_initial_volume_share']=rank.lia_volume_deduplicated_km3.cumsum()/total['lia_volume_km3'];rank['cumulative_net_loss_share']=rank.volume_loss_deduplicated_km3.cumsum()/total['net_volume_loss_km3'];rank['cumulative_positive_loss_share']=rank.volume_loss_deduplicated_km3.clip(lower=0).cumsum()/total['gross_positive_loss_km3']
rank[['record_id','mountain','size_rank','lia_volume_deduplicated_km3','volume_loss_deduplicated_km3','cumulative_initial_volume_share','cumulative_net_loss_share','cumulative_positive_loss_share']].to_csv(OUT/'size_concentration.csv',index=False,encoding='utf-8-sig')
conc=[]
for n in [1,10,math.ceil(.1*len(rank))]:
 s=rank.head(n);conc.append({'n_largest_by_initial_volume':n,'initial_volume_share_percent':100*s.lia_volume_deduplicated_km3.sum()/total['lia_volume_km3'],'net_loss_share_percent':100*s.volume_loss_deduplicated_km3.sum()/total['net_volume_loss_km3'],'positive_loss_share_percent':100*s.volume_loss_deduplicated_km3.clip(lower=0).sum()/total['gross_positive_loss_km3'],'remaining_net_loss_percent':100*(rank.iloc[n:].volume_loss_deduplicated_km3.sum()/rank.iloc[n:].lia_volume_deduplicated_km3.sum())})
dump('concentration_summary.json',conc)

# Independent modern product contrast is a diagnostic, not propagated LIA uncertainty.
prod=pd.read_csv(SRC/'modern_product_support.csv');pr=[]
for rid,s in prod.groupby('system_record_id'):
 both=(s.far_volume_km3>0)&(s.millan_volume_km3>0)
 covered=both&(s.far_native_positive_area_fraction>=.9)&(s.millan_positive_coverage_fraction>=.9)
 pr.append({'record_id':rid,'native_members':len(s),'members_both_products':int(both.sum()),'both_products_positive_all_members':bool(both.all()),'both_products_area90_all_members':bool(covered.all()),'F19_supported_modern_volume_km3':float(s.far_volume_km3.sum()),'Millan_supported_modern_volume_km3':float(s.millan_volume_km3.sum())})
pr=pd.DataFrame(pr).merge(d[['record_id','volume_loss_deduplicated_km3']],on='record_id',validate='one_to_one');pr['modern_product_difference_km3']=pr.Millan_supported_modern_volume_km3-pr.F19_supported_modern_volume_km3;pr['abs_product_difference_exceeds_abs_LIA_modern_loss']=pr.modern_product_difference_km3.abs()>pr.volume_loss_deduplicated_km3.abs()
pr.to_csv(OUT/'modern_product_contrast_diagnostic.csv',index=False,encoding='utf-8-sig')
strict=pr[pr.both_products_area90_all_members]
productsummary={'strict_scope':'all members positive in both products and each product native positive coverage>=90%; diagnostic only, not full co-registered common-support comparison','strict_n':len(strict),'n_abs_modern_product_difference_exceeds_abs_modeled_change':int(strict.abs_product_difference_exceeds_abs_LIA_modern_loss.sum()),'F19_sum_km3':float(strict.F19_supported_modern_volume_km3.sum()),'Millan_sum_km3':float(strict.Millan_supported_modern_volume_km3.sum()),'interpretation':'difference of native supported integrals; not formal error bound, not alternate LIA reconstruction'}
dump('modern_product_contrast_summary.json',productsummary)
d.to_csv(OUT/'selected654_merged_analysis.csv',index=False,encoding='utf-8-sig')

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
names=regs.mountain.tolist();colors=plt.cm.tab10(np.arange(len(names)))
fig,axs=plt.subplots(2,2,figsize=(15,10),layout='constrained')
ax=axs[0,0];ax.barh(names,regs.net_volume_loss_km3,color=colors);ax.invert_yaxis();ax.set_xlabel('Modeled net ice-volume loss (km³)');ax.set_title('A  Distribution of loss across seven sampled ranges',loc='left')
ax=axs[0,1];x=np.arange(len(names));w=.24
for off,col,color,label in [(-w,'area_loss_percent','#4c9bb0','Area'),(0,'volume_loss_percent','#b7523a','Volume'),(w,'axis_length_index_loss_percent','#637fb5','Stored-axis index')]:ax.bar(x+off,regs[col],w,label=label,color=color)
ax.set_xticks(x,[{'Tibetan Interior Mountains':'Interior','Western Kunlun Shan':'W. Kunlun','Western Tien Shan':'Local Tien Shan'}.get(z,z) for z in names],rotation=35,ha='right');ax.set_ylabel('Reduction relative to LIA (%)');ax.legend(fontsize=9);ax.set_title('B  Ratios of summed endpoint quantities',loc='left')
ax=axs[1,0]
for name,color in zip(names,colors):
 s=d[d.mountain==name];ax.scatter(s.area_decrease_relative_LIA_percent,s.volume_loss_deduplicated_percent,s=14,alpha=.65,color=color,label=name)
ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set(xlabel='Area reduction (%)',ylabel='Modeled volume reduction (%)');ax.set_title('C  All 654 systems; signed changes retained',loc='left');ax.legend(fontsize=7,ncol=2)
ax=axs[1,1];xx=100*rank.size_rank/len(rank);ax.plot(xx,100*rank.cumulative_initial_volume_share,label='Initial ice volume',lw=2);ax.plot(xx,100*rank.cumulative_net_loss_share,label='Net volume loss',lw=2);ax.plot(xx,100*rank.cumulative_positive_loss_share,label='Gross positive loss',lw=1.5,ls='--');ax.plot([0,100],[0,100],color='grey',lw=.8,ls=':');ax.set(xlabel='Systems ranked by initial volume (%)',ylabel='Cumulative share (%)',xlim=(0,100));ax.legend(fontsize=9);ax.set_title('D  Concentration by glacier size',loc='left')
fig.suptitle('Selected 654: LIA-to-modern geometry and modeled ice-volume change',fontsize=17)
fig.savefig(OUT/'selected654_regional_budget.png',dpi=300);plt.close(fig)
fig,ax=plt.subplots(figsize=(13,6),layout='constrained');x=np.arange(len(names))
vals=[('LIA_only_positive_volume_km3','LIA-only positive ice support','#db9b42',1),('common_positive_thickness_loss_km3','Common positive support: thickness change','#197d94',1),('modern_only_positive_volume_km3','Modern-only positive ice support','#7e70ad',-1)]
pos=np.zeros(len(names));neg=np.zeros(len(names))
for key,label,color,sign in vals:
 v0=sign*regs[key].to_numpy();bot=np.where(v0>=0,pos,neg);ax.bar(x,v0,bottom=bot,label=label,color=color);pos+=np.maximum(v0,0);neg+=np.minimum(v0,0)
ax.scatter(x,regs.net_volume_loss_km3,color='#222',s=22,zorder=3,label='Published net loss (includes overlap correction)')
ax.axhline(0,color='grey',lw=.7);ax.set_xticks(x,names,rotation=22,ha='right');ax.set_ylabel('Modeled volume difference (km³)');ax.set_title('Spatial accounting on frozen endpoint thickness grids',loc='left',fontsize=16);ax.legend(fontsize=9)
fig.savefig(OUT/'selected654_spatial_volume_partition.png',dpi=300);plt.close(fig)

qc={'count':len(d),'unique_record_ids':bool(d.record_id.is_unique),'frozen_area_total_difference_km2':float(total['lia_area_km2']-27209.3228041975),'frozen_volume_total_difference_km3':float(total['net_volume_loss_km3']-328.5892373748),'all654_length_index_ratio_percent':total['axis_length_index_loss_percent'],'raster_rows':len(dd),'raster_source_files_hash_verified':len(rasterchecks),'raster_max_volume_closure_error_km3':float(dd.closure_error_km3.abs().max()),'unknown_excluded_component_cells':int(dd.unknown_excluded_component_cells.sum()),'regional_sums_match':bool(np.isclose(regs.net_volume_loss_km3.sum(),total['net_volume_loss_km3'],atol=1e-10,rtol=0)),'signed_volume_gain_count':total['volume_increase_systems'],'no_models_run':True,'all_input_table_hashes_unchanged':all(sha(p)==s for p,s in sourcehash.items())}
assert len(d)==654 and abs(qc['frozen_volume_total_difference_km3'])<1e-7 and qc['regional_sums_match'] and qc['all_input_table_hashes_unchanged'] and qc['signed_volume_gain_count']==36
dump('QC.json',qc);dump('summary.json',{'scope':protocol['scope'],'total':total,'regional':rec(regs),'concentration':conc,'modern_product_contrast':productsummary,'caveats':['655source rows/654systems retained; six user exclusions unchanged','positive-support spatial partition is not vector terminus retreat','modern-volume matching calibrates the model; no independent LIA validation','Himalaya legacy191 not combined because model and bed/calibration differ','no formal uncertainty interval or uniform LIA epoch'],'elapsed_s':round(time.monotonic()-start,1)})
shutil.copy2(__file__,OUT/'reproduce_analysis.py')
print(json.dumps({'done':True,'output':str(OUT),'total':total,'QC':qc},ensure_ascii=False),flush=True)
