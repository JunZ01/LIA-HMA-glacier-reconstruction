"""Frozen descriptive LOMO ridge comparison; no physical reconstruction/model run."""
from pathlib import Path
from datetime import datetime, timezone
import json, hashlib, sys, subprocess
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

ROOT=Path('E:/3spyderProject/1_LIA_himalaya_oggm')
OUT=Path('H:/ToYangxin/result_yx/hma_harmonized_analysis_20261012_v1/climate_geometry_exploration')
RESPONSE=OUT.parent/'selected654_analysis/selected654_merged_analysis.csv'
SOURCES=[OUT/'geometry_covariates654.csv',OUT/'climate_background654.csv',RESPONSE]
GEOM=['ln_LIA_area_km2','modern_zmean_m_member_area_weighted','modern_member_relief_m_member_area_weighted','modern_slope_deg_member_area_weighted','ln1p_modern_member_count','modern_largest_member_area_fraction']
CLIM=['annual_mean_temperature_C','ln1p_annual_precipitation_mm','mjjas_precipitation_fraction']
ALPHA=1.0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,v): (OUT/n).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def canonical():
    d={}
    for f in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
        p=ROOT/f;t=p.read_text(encoding='utf-8-sig');d[f]={'sha256':sha(p),'read_complete_characters':len(t)}
    cp=subprocess.run([sys.executable,str(ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True,text=True,encoding='utf-8')
    v=json.loads(cp.stdout);assert cp.returncode==0 and v['status']=='pass'
    d['validator']={k:v[k] for k in ['status','launch_authorized','counts','failures']};return d

def freeze():
    if (OUT/'cv_protocol.json').exists():raise RuntimeError('CV protocol already exists; do not silently replace')
    p={'created_utc':datetime.now(timezone.utc).isoformat(),'protocol_frozen_before_first_fit':True,
       'analysis_type':'Descriptive cross-mountain prediction of existing model-derived signed LIA-to-modern fractional volume loss',
       'canonical_read':canonical(),'source_sha256':{str(p):sha(p) for p in SOURCES},
       'population':'All654 frozen systems across7 occupied local mountain ranges; retain all36 negative responses; no tuning-based filtering',
       'response_field':'volume_loss_deduplicated_percent','response_unit':'percentage points','response_definition':'100*(deduplicated LIA modeled volume-modern modeled volume)/(deduplicated LIA modeled volume)',
       'models':{'training_mean_baseline':[],'geometry_ridge':GEOM,'geometry_climate_ridge':GEOM+CLIM},
       'transforms':{'ln_LIA_area_km2':'natural log of positive frozen LIA area','ln1p_modern_member_count':'natural log(1+count)','ln1p_annual_precipitation_mm':'natural log(1+annual climatological precipitation in mm)'},
       'excluded_predictors':['area change','length change','modeled volume','modeled thickness','retreat fraction','mountain label or one-hot region'],
       'split':'Deterministic leave one occupied mountain range out,7folds; all other6ranges train; exactly one fold-out prediction per system per model',
       'estimator':'sklearn StandardScaler fit only on trainingfold (population variance ddof0), then Ridge(alpha=1.0, fit_intercept=True, solver=cholesky); constant baseline is training y mean',
       'regularization':'fixedalpha1; squared residual sum+alpha*sum(squared standardized coefficients); no intercept penalization; no hyperparameter search',
       'missing_policy':'All654 required source descriptors currently complete. Fail if missing rather than drop, impute, or change model inputs.',
       'metrics':'Perfold and pooled out-of-fold MAE(pp),MSE(pp²),RMSE(pp),R²; Spearman rho descriptive only. Equal system weights. Also unweighted mean of7fold MAEs; no training R² used.',
       'increment_sign':'Baseline minus geometry error and geometry minus geometry+climate error; positive means lower error after added variables',
       'effective_sample_reporting':'Report654system records,7region folds,unique climate source cells and fold train/test shared cells; do not label cell count as formal effective sample size.',
       'grid_overlap':'Use complete set of source climate point IDs per system; report test systems sharing any point with training systems, and fraction of test system climate-area weights in shared points.',
       'independent_numerical_check':'For each Ridge fold, rebuild training means/std and centered closed-form ridge in NumPy; compare all test predictions to sklearn with1e-8pp absolute tolerance.',
       'scope_limits':['Existing samples only; LOMO does not validate transfer to unsampled HMA mountain ranges.','No causal attribution, physical parameter inference, climate-change reconstruction, statistical significance, confidence interval, or uncertainty calibration.','Modern climatology1981–2010 is a background proxy, not LIA-to-modern forcing difference.','Grid temperature is uncorrected for glacier elevation and is not glacier surface temperature.','Source climate cells may cross mountain boundaries; LOMO is not independent climate-cell holdout.','Volume response is model-based and geometry also enters reconstruction. Predictability can arise from shared construction; it is not independent mechanistic validation.','Unequal and partly undated inventory epochs prevent interpreting response as annual rate.']}
    dump('cv_protocol.json',p);print('FROZEN',sha(OUT/'cv_protocol.json'))

def metrics(y,p):
    return {'MAE_percentage_points':float(mean_absolute_error(y,p)),'MSE_percentage_points_squared':float(mean_squared_error(y,p)),
            'RMSE_percentage_points':float(np.sqrt(mean_squared_error(y,p))),'out_of_fold_R2':float(r2_score(y,p))}

def fit():
    if (OUT/'cv_summary.json').exists():raise RuntimeError('Completed immutable CV output exists')
    prot=json.loads((OUT/'cv_protocol.json').read_text(encoding='utf-8'));p_hash=sha(OUT/'cv_protocol.json');c=canonical()
    assert all(sha(p)==h for p,h in prot['source_sha256'].items())
    g=pd.read_csv(SOURCES[0]);cl=pd.read_csv(SOURCES[1],dtype={'climate_point_indices':str,'climate_grid_area_weights':str});r=pd.read_csv(RESPONSE)
    assert len(g)==len(cl)==len(r)==654 and g.record_id.is_unique and cl.record_id.is_unique and r.record_id.is_unique
    d=g.merge(cl.drop(columns=['lia_unique_id','modern_member_count']),on=['record_id','mountain'],validate='one_to_one').merge(r[['record_id','volume_loss_deduplicated_percent','model_family']],on='record_id',validate='one_to_one').sort_values('record_id').reset_index(drop=True)
    assert len(d)==654 and d.all_members_complete_360_months.all()
    d['ln_LIA_area_km2']=np.log(d.LIA_area_geodesic_km2)
    d['ln1p_modern_member_count']=np.log1p(d.modern_member_count)
    d['ln1p_annual_precipitation_mm']=np.log1p(d.annual_precipitation_mm)
    assert np.isfinite(d[GEOM+CLIM+['volume_loss_deduplicated_percent']]).all().all()
    y=d.volume_loss_deduplicated_percent.to_numpy()
    assert (y<0).sum()==36
    regions=sorted(d.mountain.unique()); assert len(regions)==7
    gridsets=[set(int(q) for q in str(v).split(';')) for v in d.climate_point_indices]
    gridweights=[{int(a):float(b) for a,b in (q.split(':') for q in v.split(';'))} for v in d.climate_grid_area_weights]
    predictions=[];foldrows=[];scaling=[];coeffs=[];gridrows=[];independentmax=0.0
    for region in regions:
        te=d.mountain.to_numpy()==region;tr=~te
        trainpts=set().union(*[gridsets[i] for i in np.where(tr)[0]])
        testpts=set().union(*[gridsets[i] for i in np.where(te)[0]])
        shared=trainpts & testpts
        sharedweights=[sum(w for k,w in gridweights[i].items() if k in shared) for i in np.where(te)[0]]
        gridrow={'held_out_mountain':region,'training_systems':int(tr.sum()),'test_systems':int(te.sum()),'training_unique_climate_cells':len(trainpts),'test_unique_climate_cells':len(testpts),'shared_train_test_climate_cells':len(shared),'test_systems_with_shared_climate_cell':int(sum(bool(gridsets[i]&trainpts) for i in np.where(te)[0])),'test_system_share_with_shared_climate_cell':float(np.mean([bool(gridsets[i]&trainpts) for i in np.where(te)[0]])),'mean_test_system_climate_area_fraction_shared':float(np.mean(sharedweights)),'shared_climate_point_indices':';'.join(map(str,sorted(shared)))}
        gridrows.append(gridrow)
        for name,cols in prot['models'].items():
            if not cols: pred=np.full(te.sum(),y[tr].mean());err=0.0
            else:
                X=d[cols].to_numpy();model=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA,fit_intercept=True,solver='cholesky'))
                model.fit(X[tr],y[tr]);pred=model.predict(X[te])
                sc=model.named_steps['standardscaler'];reg=model.named_steps['ridge']
                mu=X[tr].mean(axis=0);sd=X[tr].std(axis=0);sd[sd==0]=1
                xs=(X[tr]-mu)/sd;xt=(X[te]-mu)/sd;ym=y[tr].mean()
                beta=np.linalg.solve(xs.T@xs+ALPHA*np.eye(len(cols)),xs.T@(y[tr]-ym));pred2=xt@beta+ym
                err=float(abs(pred-pred2).max());assert err<1e-8;independentmax=max(independentmax,err)
                for j,col in enumerate(cols):
                    scaling.append({'held_out_mountain':region,'model':name,'predictor':col,'training_mean':float(sc.mean_[j]),'training_std_population':float(sc.scale_[j]),'training_systems':int(tr.sum())})
                    coeffs.append({'held_out_mountain':region,'model':name,'predictor':col,'coefficient_per_training_std':float(reg.coef_[j]),'interpretation':'numerical reproducibility only; not causal importance'})
                coeffs.append({'held_out_mountain':region,'model':name,'predictor':'intercept','coefficient_per_training_std':float(reg.intercept_),'interpretation':'numerical reproducibility only'})
            foldrows.append({'held_out_mountain':region,'model':name,'training_systems':int(tr.sum()),'test_systems':int(te.sum()),'test_negative_responses':int((y[te]<0).sum()),'closed_form_max_abs_prediction_difference_pp':err,**metrics(y[te],pred)})
            for i,p in zip(np.where(te)[0],pred): predictions.append({'record_id':d.record_id.iloc[i],'held_out_mountain':region,'model':name,'observed_model_derived_signed_volume_loss_percent':float(y[i]),'out_of_fold_predicted_percent':float(p),'residual_observed_minus_prediction_pp':float(y[i]-p),'training_systems':int(tr.sum()),'shares_any_climate_cell_with_training':bool(gridsets[i]&trainpts),'test_climate_area_fraction_shared_with_training':sum(w for k,w in gridweights[i].items() if k in trainpts)})
    pp=pd.DataFrame(predictions);ff=pd.DataFrame(foldrows); overall=[]
    assert len(pp)==654*3 and len(ff)==7*3
    for name,s in pp.groupby('model'):
        assert len(s)==654 and s.record_id.is_unique
        mm=metrics(s.observed_model_derived_signed_volume_loss_percent,s.out_of_fold_predicted_percent)
        mm.update(model=name,systems=654,region_folds=7,negative_responses=36,unique_climate_cells=len(set().union(*gridsets)),spearman_rho_descriptive=float(spearmanr(s.observed_model_derived_signed_volume_loss_percent,s.out_of_fold_predicted_percent).statistic),mean_of_7_fold_MAE_percentage_points=float(ff.loc[ff.model==name,'MAE_percentage_points'].mean()))
        overall.append(mm)
    oo=pd.DataFrame(overall);inc=[]
    for scope,sub in [('ALL_SYSTEM_POOLED',oo)]+[(rr,ff[ff.held_out_mountain==rr]) for rr in regions]:
        ss=sub.set_index('model')
        for a,b,label in [('training_mean_baseline','geometry_ridge','add_geometry'),('geometry_ridge','geometry_climate_ridge','add_climate')]:
            inc.append({'scope':scope,'comparison':label,'base_model':a,'added_model':b,**{'improvement_'+metric:float(ss.loc[a,metric]-ss.loc[b,metric]) for metric in ['MAE_percentage_points','MSE_percentage_points_squared','RMSE_percentage_points']}})
    for filename,frame in [('cv_predictions1962.csv',pp),('cv_folds21.csv',ff),('cv_overall3.csv',oo),('cv_incremental_comparisons.csv',pd.DataFrame(inc)),('cv_shared_climate_cells_by_fold.csv',pd.DataFrame(gridrows)),('cv_training_fold_scaling.csv',pd.DataFrame(scaling)),('cv_coefficients_for_reproduction_only.csv',pd.DataFrame(coeffs)),('covariates_response654.csv',d)]: frame.to_csv(OUT/filename,index=False,encoding='utf-8-sig')
    sources_unchanged={p:sha(p)==h for p,h in prot['source_sha256'].items()};assert all(sources_unchanged.values());assert sha(OUT/'cv_protocol.json')==p_hash
    summary={'status':'complete-descriptive-leave-one-mountain-out','completed_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':p_hash,'canonical_read_at_fit':c,'sources_unchanged':sources_unchanged,'systems':654,'folds':7,'negative_responses_retained':36,'climate_unique_cells':len(set().union(*gridsets)),'climate_cells_are_not_formal_effective_sample_size':True,'geometry_feature_count':6,'climate_added_feature_count':3,'alpha':ALPHA,'hyperparameter_search':False,'source_model_runs':0,'new_ice_reconstruction_runs':0,'fold_predictions_unique_complete':True,'max_sklearn_vs_independent_closed_form_difference_pp':independentmax,'overall_out_of_fold_metrics':overall,'fold_climate_sharing':gridrows,'limits':prot['scope_limits']}
    dump('cv_summary.json',summary)
    dump('cv_QC.json',{'status':'pass','all654_exactly_once_per_model':True,'negative36_retained':True,'all_predictors_and_predictions_finite':bool(np.isfinite(pp.out_of_fold_predicted_percent).all()),'source_hashes_unchanged':all(sources_unchanged.values()),'protocol_unchanged':True,'training_only_scaling':True,'independent_closed_form_pass':independentmax<1e-8,'no_tuning':True})
    dump('cv_output_sha256.json',{p.name:sha(p) for p in OUT.glob('cv_*') if p.name!='cv_output_sha256.json'})
    print(json.dumps({'overall':overall,'max_closed_form_difference_pp':independentmax,'grid_sharing':gridrows},ensure_ascii=False,indent=2))

if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ['freeze','fit']:raise SystemExit('Use freeze then fit in separate invocations')
    freeze() if sys.argv[1]=='freeze' else fit()
