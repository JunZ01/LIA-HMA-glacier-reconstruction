from pathlib import Path
import json, hashlib, shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('E:/3spyderProject/1_LIA_himalaya_oggm')
OUT=Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/climate_geometry_exploration')
if (OUT/'cv_interpretation.json').exists():raise RuntimeError('Completed immutable notes exist')
# Full reads precede this plotting-only derivative, consistent with the project contract.
for fn in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
    (ROOT/fn).read_text(encoding='utf-8-sig')
import subprocess,sys
p=subprocess.run([sys.executable,str(ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True,text=True,encoding='utf-8')
assert p.returncode==0 and json.loads(p.stdout)['status']=='pass'

f=pd.read_csv(OUT/'cv_folds21.csv');o=pd.read_csv(OUT/'cv_overall3.csv');inc=pd.read_csv(OUT/'cv_incremental_comparisons.csv');grid=pd.read_csv(OUT/'cv_shared_climate_cells_by_fold.csv')
notes={'finding_zh':'在现有654样本的7山脉留一验证中，现代几何与LIA面积组成的6项描述变量降低了各山脉的平均绝对误差；加入3项1981–2010气候背景变量没有改善总体误差。',
       'geometry_lower_MAE_regions':int((inc[(inc.comparison=='add_geometry')&(inc.scope!='ALL_SYSTEM_POOLED')].improvement_MAE_percentage_points>0).sum()),
       'climate_lower_MAE_regions':int((inc[(inc.comparison=='add_climate')&(inc.scope!='ALL_SYSTEM_POOLED')].improvement_MAE_percentage_points>0).sum()),
       'actual_grid_sharing_result':'All7 actual held-out mountains have zero climate point IDs shared with their training mountains; this resolves the pre-fit possibility warning in cv_protocol.json. Reuse within regions and spatial correlation among cells remain.',
       'climate_interpretation_zh':'该结果仅说明这3项粗格网现代气候背景指标在固定线性预测形式下未增加跨山脉预测能力，不代表气候对冰川变化没有作用。',
       'geometry_interpretation_zh':'几何参数也参与冰量重建；小冰期面积与响应构建存在结构耦合，预测改善不能作为独立机制验证。现代地形属性不等同于小冰期地形。',
       'independence_zh':'654是系统记录数；164是所用气候源格点数，7是验证折数。三者均不得直接替代正式统计有效样本量。',
       'followup_zh':'保留既定结果；后续关键证据是覆盖小冰期至编目端点的气候变化及独立观测验证，而不是通过调参抬高现有拟合度。',
       'physical_model_runs':0,'negative_volume_loss_responses_retained':36}
(OUT/'cv_interpretation.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf-8')
models=['training_mean_baseline','geometry_ridge','geometry_climate_ridge']
labels=['Training mean','Geometry (6 predictors)','Geometry + climate (9 predictors)']
colors=['#878c94','#207a88','#cd713d']
region_order=['Hindu Kush','Pamir','Karakoram','Western Tien Shan','Western Kunlun Shan','Tibetan Interior Mountains','Nyainqentanglha']
fig,axes=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[1,2.3]})
for i,(mo,la,co) in enumerate(zip(models,labels,colors)):
    mm=o[o.model==mo].iloc[0]
    axes[0].barh(2-i,mm.MAE_percentage_points,color=co,height=.58)
    axes[0].text(mm.MAE_percentage_points+.13,2-i,f'{mm.MAE_percentage_points:.2f}',va='center',fontsize=10)
    vals=f[f.model==mo].set_index('held_out_mountain').loc[region_order,'MAE_percentage_points']
    axes[1].plot(vals,np.arange(7)+(i-1)*.16,'o',color=co,ms=6,label=la)
axes[0].set_yticks([2,1,0],['Mean','Geometry','Geometry\n+ climate'])
axes[0].set_xlim(0,11);axes[0].set_title('All 654 systems',loc='left',fontweight='bold')
axes[1].set_yticks(range(7),['Hindu Kush','Pamir','Karakoram','Western Tien Shan','Western Kunlun Shan','Tibetan Interior','Nyainqentanglha'])
axes[1].invert_yaxis();axes[1].set_xlim(0,14);axes[1].set_title('Each held-out mountain range',loc='left',fontweight='bold')
for ax in axes:
    ax.set_xlabel('Out-of-fold MAE (percentage points)')
    ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[1].legend(loc='upper center',bbox_to_anchor=(.4,-.16),ncol=1,frameon=False,fontsize=9)
fig.suptitle('Cross-mountain prediction of model-derived volume-loss fraction',x=.05,y=.97,ha='left',fontsize=14,fontweight='bold')
fig.text(.05,.02,'All 654 retained; 36 negative responses. Fixed ridge alpha = 1; training-fold standardization.\nClimate: 1981–2010 grid background; no causal attribution or physical-volume validation.',fontsize=9,color='#555555')
fig.subplots_adjust(left=.12,right=.97,top=.83,bottom=.32,wspace=.8)
fig.savefig(OUT/'cv_mountain_prediction_comparison.png',dpi=300,bbox_inches='tight')
fig.savefig(OUT/'cv_mountain_prediction_comparison.pdf',bbox_inches='tight')
plt.close(fig)
for name in ['hma_geometry_covariates654_20261012.py','hma_covariate_lomo_cv654_20261012.py','hma_cv654_notes_and_plot_20261012.py']:
    dst=OUT/name
    assert not dst.exists()
    shutil.copy2(ROOT/name,dst)
(OUT/'covariate_analysis_sourcecode_sha256.json').write_text(json.dumps({n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['hma_geometry_covariates654_20261012.py','hma_covariate_lomo_cv654_20261012.py','hma_cv654_notes_and_plot_20261012.py']},indent=2),encoding='utf-8')
print(json.dumps(notes,ensure_ascii=False,indent=2))
