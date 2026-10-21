"""Read-only scientific postprocessing. Writes only this versioned analysis folder."""
from pathlib import Path
import importlib.util
import json
import html
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'validation_data/results/paleo_surface_validation_v0'
spec = importlib.util.spec_from_file_location('evolution', ROOT/'analysis/lia_51_evolution_v2/build_report.py')
evo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo)
old = evo.previous
YEARS = np.arange(1500,2001)
COLORS = ['#2678a2','#d99436','#ae4265']
LABELS = {'low':'低高程 <4000 m','middle':'中高程 4000–<4500 m','high':'高高程 ≥4500 m',
          'west_<80':'经度 <80°E','middle_80_86':'经度 80–<86°E','east_>=86':'经度 ≥86°E',
          'legacy_constant':'已认可·恒定校正','synthetic_timevarying':'已认可·时变校正',
          'candidate_constant':'候选·恒定校正','retained32':'主样本32条','candidate19':'候选19条','all51':'合并51条',
          'retained_legacy_constant':'主样本·恒定校正','retained_synthetic_timevarying':'主样本·时变校正',
          'retained_no_boundary_volume_peak':'剔除体积边界峰值','retained_peak_after1550':'体积峰值晚于1550年'}
COHORTS = [('retained','已认可32条'),('candidate','候选19条'),('all','全部51条（补充）')]
METRICS = [('length','length_km','长度','km'),('volume','volume_km3','体积','km³'),('area','area_km2','面积','km²')]
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus':False,'font.size':10,'figure.dpi':140})

def clean(x):
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)): return [clean(v) for v in x]
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    return x

def save(name,data):
    (OUT/name).write_text(json.dumps(clean(data),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def read(path): return old.read_json(path)
def reg(path,expected=None): return old.register(path,expected)
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
def scalar(x): return float(x) if x is not None else np.nan
def resolve_root(item):
    return (ROOT/'validation_data' if item.get('anchor')=='validation' else ROOT)/item['sub']

def rho(x,y):
    a,b=np.asarray(x,float),np.asarray(y,float)
    keep=np.isfinite(a)&np.isfinite(b)
    a,b=a[keep],b[keep]
    return dict(n=len(a),rho=float(spearmanr(a,b).statistic) if len(a)>=4 and np.ptp(a)>0 and np.ptp(b)>0 else None)

def partial(frame,x,y,covariates):
    d=frame[[x,y]+covariates].dropna()
    n=len(d)
    if n<max(10,len(covariates)+5): return dict(n=n,rho=None,reason='insufficient sample',covariates=covariates)
    ranked=d.apply(rankdata)
    design=np.column_stack([np.ones(n),ranked[covariates].values])
    if np.linalg.matrix_rank(design)<design.shape[1]: return dict(n=n,rho=None,reason='rank deficient',covariates=covariates)
    residuals=[ranked[k].values-design@np.linalg.lstsq(design,ranked[k].values,rcond=None)[0] for k in [x,y]]
    r=float(np.corrcoef(*residuals)[0,1]) if all(np.std(v)>1e-10 for v in residuals) else None
    return dict(n=n,rho=r,covariates=covariates,interpretation='conditional rank association, not causation')

def peak(y,x,start):
    y,x=np.asarray(y),np.asarray(x)
    keep=(y>=start)&(y<=2000)
    y,x=y[keep],x[keep]
    ids=np.flatnonzero(np.isclose(x,x.max(),rtol=1e-10,atol=1e-12))
    return dict(value=float(x.max()),first_year=int(y[ids[0]]),last_year=int(y[ids[-1]]),
                tied_years=y[ids].astype(int).tolist(),left_boundary=bool(ids[0]==0),right_boundary=bool(ids[-1]==len(y)-1))

def metadata(m,c):
    """Track the actual selected parent and distinguish physical parameters from multipliers."""
    path=ROOT/m['source']; parent=path.parent
    p=dict(system=m['system'],model_route=m['model_route'],spin_delta_c=None,forward_start_c=None,forward_end_c=None,
           ramp_exponent=None,pulse=None,melt_f=None,prcp_fac=None,glen_a=None,base_temp_bias_c=None,
           melt_multiplier=None,prcp_multiplier=None,glen_multiplier=None,source_files=[],
           policy='unknown',numerical_reproduction='not established by this audit',spin_status=None)
    spin=None; hist_initial=None
    if m['status']=='candidate':
        q=m['parameters']; p.update(spin_delta_c=q.get('delta'),forward_start_c=q.get('constant_offset_degC'),
            forward_end_c=q.get('constant_offset_degC'),melt_multiplier=q.get('melt'),
            prcp_multiplier=q.get('prcp'),glen_multiplier=q.get('glen'),policy='constant saved GPU correction')
        catalog=path.parents[3]/'catalog.json'
        data=read(catalog); p['source_files'].append(catalog.relative_to(ROOT).as_posix())
        rows=[r for r in data['rows'] if (r.get('forward') or {}).get('state',{}).get('path')==m['source']]
        assert len(rows)==1,(m['system'],'selected candidate parent')
        record=rows[0]; assert record['candidate']==q
        for side in ['spin','forward']:
            for artifact in ['result','state']:
                proof=record[side][artifact]; reg(proof['path'],proof['sha256']); p['source_files'].append(proof['path'])
        p['spin_status']=record.get('spin_status')
        with np.load(ROOT/record['spin']['state']['path'],allow_pickle=False) as a:
            last=a['history'][-1]
            spin=dict(length_m=float(last[0]),area_m2=float(last[1]),volume_m3=float(last[2]))
        with np.load(path,allow_pickle=False) as a:
            first=a['history'][0]
            hist_initial=dict(length_m=float(first[0]),area_m2=float(first[1]),volume_m3=float(first[2]))
        p['physical_parameter_note']='multipliers only; do not interpret multiplier 1 as a physical parameter value'
    elif m['model_route']=='frozen-legacy':
        summary=read(parent/'summary.json'); par=summary.get('parameters',{})
        task=read(parent/'task.json') if (parent/'task.json').exists() else {}
        policy=summary.get('temperature_offset_policy',{})
        delta=par.get('delta_t_c',task.get('delta_t_c'))
        p.update(spin_delta_c=par.get('spinup_delta_t_c',task.get('spinup_delta_t_c',policy.get('spinup_delta_t_c',delta))),
                 forward_start_c=par.get('forward_delta_t_c',task.get('forward_delta_t_c',policy.get('forward_delta_t_c',delta))),
                 policy='constant legacy correction',melt_f=par.get('melt_f'),prcp_fac=par.get('prcp_fac'),
                 glen_a=par.get('glen_a',par.get('glen_a_pa_3_s_1')),base_temp_bias_c=par.get('base_temp_bias_c'))
        p['forward_end_c']=p['forward_start_c']
        d=pd.read_csv(reg(path)); f=d[d.phase.eq('forward')].sort_values('year'); s=d[d.phase.eq('spinup')].sort_values('year')
        if len(s): spin={k:float(s.iloc[-1][k]) for k in ['length_m','area_m2','volume_m3']}
        hist_initial={k:float(f.iloc[0][k]) for k in ['length_m','area_m2','volume_m3']}
        sp=parent/'source_spinup_summary.json'
        if sp.exists():
            ss=read(sp); p['spin_status']=ss.get('status'); p['spinup_control']=ss.get('spinup_control')
            if spin is None:
                sm=ss.get('metrics',{})
                if all(k in sm for k in ['raw_l_spinup_m','a_spinup_m2','v_spinup_m3']):
                    spin=dict(length_m=sm['raw_l_spinup_m'],area_m2=sm['a_spinup_m2'],volume_m3=sm['v_spinup_m3'])
                    p['spin_endpoint_source']='saved source_spinup_summary metrics; raw length operator'
        p['source_files']=[str((parent/n).relative_to(ROOT)).replace('\\','/') for n in ['summary.json','task.json'] if (parent/n).exists()]
    else:
        config=read(parent/'config_snapshot.json'); hs=read(parent/'historical_summary.json')
        par=config['stage2']; cl=hs.get('climate',{})
        sr=resolve_root(config['steady_root']); ss=read(sr/'steady_50y_summary.json')
        sc=read(sr/'config_snapshot.json'); spin=ss['final']; hist_initial=hs['initial']
        p.update(spin_delta_c=sc.get('stage2',{}).get('delta_t_c'),spin_status=ss.get('status'),
            forward_start_c=hs.get('forward_delta_t_start_c',cl.get('effective_delta_t_start_c')),
            forward_end_c=hs.get('forward_delta_t_end_c',cl.get('effective_delta_t_end_c')),
            ramp_exponent=cl.get('ramp_shape_exponent',config.get('ramp_shape_exponent',1.0)),
            pulse=cl.get('early_warm_pulse'),policy=cl.get('temperature_bias_policy','unknown'),
            melt_f=par.get('melt_f'),prcp_fac=par.get('prcp_fac'),glen_a=par.get('glen_a'),
            base_temp_bias_c=par.get('base_temp_bias_c'),climate_source=cl.get('source'),
            spin_window=ss.get('spinup_window'),spin_volume_change_fraction=ss.get('volume_change_fraction_50y'),
            initial_thickness_modification=cl.get('initial_thickness_modification'))
        p['source_files']=[str(v.relative_to(ROOT)).replace('\\','/') for v in [parent/'config_snapshot.json',parent/'historical_summary.json',sr/'config_snapshot.json',sr/'steady_50y_summary.json']]
        rp=parent/'reproduction_comparison.json'
        if rp.exists():
            rc=read(rp); p['numerical_reproduction']=rc.get('status','comparison present')
            p['reproduction_evidence']=rp.relative_to(ROOT).as_posix()
    p['route_family']='candidate_constant' if m['status']=='candidate' else ('legacy_constant' if m['model_route']=='frozen-legacy' else 'synthetic_timevarying')
    p['forward_correction_change_c']=None if p['forward_start_c'] is None or p['forward_end_c'] is None else p['forward_end_c']-p['forward_start_c']
    audit=dict(system=m['system'],status=m['status'],spin_endpoint=spin,historical_initial=hist_initial,
               length_operator=m['length_operator'],handoff_basis='raw model L/A/V at transition; plotted length may use a different observation operator',
               full_initial_array_equality='not tested',independent_observational_validation='not established',
               endpoint_role='calibration target; not independent validation')
    if spin and hist_initial:
        audit['handoff_differences']={k:hist_initial[k]-spin[k] for k in ['length_m','area_m2','volume_m3']}
        audit['scalar_handoff_match']=all(np.isclose(spin[k],hist_initial[k],rtol=1e-8,atol=1e-5) for k in audit['handoff_differences'])
    else: audit['scalar_handoff_match']=None
    # Compare audited initial geometry with the plotted series only where the operators agree.
    for key,unit in [('area_m2',1e6),('volume_m3',1e9)]:
        curve_key='native_area_km2' if key=='area_m2' else 'native_volume_km3'
        audit[key+'_initial_curve_match']=bool(hist_initial and np.isclose(hist_initial[key]/unit,c[curve_key][0],rtol=1e-8,atol=1e-10))
    return p,audit

def hypsometry(inv):
    bins=np.arange(2500,8501,200)
    with xr.open_dataset(reg(inv['lia_source_grid'])) as ds:
        mi=list(ds.member_id.values.astype(str)).index('farinotti2019')
        h=ds.thickness_m.isel(member=mi).values; z=ds.surface_elevation_m.isel(member=mi).values
        bed=ds.bed_elevation_m.isel(member=mi).values
        mask=(ds.valid_mask.isel(member=mi).values>0)&(ds.lia_mask.values>0)&np.isfinite(h)&np.isfinite(z)&np.isfinite(bed)&(h>1e-6)
        cell=abs(float(np.median(np.diff(ds.x.values))*np.median(np.diff(ds.y.values))))
        zz=z[mask]; hh=h[mask]
        assert zz.min()>=bins[0] and zz.max()<=bins[-1]
        hist=np.histogram(zz,bins)[0]*cell/1e6
        area=float(mask.sum()*cell/1e6); volume=float(hh.sum()*cell/1e9)
        assert np.isclose(area,inv['lia_positive_ice_area_km2'],rtol=1e-8)
        assert np.isclose(volume,inv['lia_reconstructed_volume_km3'],rtol=1e-7)
        return dict(system=inv['lia_unique_id'],elevation_bin_edges_m=bins.tolist(),area_km2=hist.tolist(),
                    total_area_km2=area,volume_km3=volume,relief_m=float(np.ptp(zz)),
                    normalized_mean_height=float((zz.mean()-zz.min())/np.ptp(zz)),
                    lower_half_area_fraction=float(np.mean(zz<(zz.max()+zz.min())/2)),
                    mean_thickness_m=float(hh.mean()),mask_rule='finite paired V3 F19, valid LIA mask, H > 1e-6 m')

def build_features(curves,members):
    inventory={r['lia_unique_id']:r for r in read(ROOT/'analysis/lia_51_inventory_v1/inventory.json')}
    pixels=pd.read_csv(reg(BASE/'lia_191_controller_v1/core191_climate_pixels_ensemble_rec_ref.csv')).set_index('lia_unique_id')
    rows=[]; audits=[]; parameters=[]; hyps=[]; diagnostics=[]
    for c,m in zip(curves,members):
        sid=m['system']; inv=inventory[sid]
        par,audit=metadata(m,c); hyp=hypsometry(inv); hyps.append(hyp); parameters.append(par)
        pix=pixels.loc[sid]; assert not isinstance(pix,pd.DataFrame)
        row=dict(system=sid,status=m['status'],longitude=m['longitude'],latitude=m['latitude'],
            terminus_elevation_m=m['terminus_elevation_m'],elevation_group=m['elevation_group'],
            longitude_group='west_<80' if m['longitude']<80 else ('middle_80_86' if m['longitude']<86 else 'east_>=86'),
            log_area=np.log10(inv['lia_positive_ice_area_km2']),log_length=np.log10(inv['lia_length_km']),
            static_area_km2=inv['lia_positive_ice_area_km2'],static_length_km=inv['lia_length_km'],
            gradient_deg=inv['lia_mainline_mean_gradient_deg'],relief_m=hyp['relief_m'],
            normalized_mean_height=hyp['normalized_mean_height'],lower_half_area_fraction=hyp['lower_half_area_fraction'],
            mean_thickness_m=hyp['mean_thickness_m'],azimuth_deg=inv['lia_azimuth_deg'],
            terminal_azimuth_deg=inv['lia_terminal_azimuth_deg'],sinuosity=inv['lia_sinuosity'],
            northness=np.cos(np.deg2rad(inv['lia_azimuth_deg'])),eastness=np.sin(np.deg2rad(inv['lia_azimuth_deg'])),
            modern_outline_dates=inv['modern_member_outline_dates'],modern_member_count=inv['modern_member_count'],
            modern_terminus_lon=inv['modern_terminus_lon'],modern_terminus_lat=inv['modern_terminus_lat'],
            lia_terminus_lon=inv['lia_terminus_lon'],lia_terminus_lat=inv['lia_terminus_lat'],
            terminus_separation_m=inv['terminus_point_separation_m'],
            climate_pixel=f"{int(pix.climate_lat_index)}:{int(pix.climate_lon_index)}",
            climate_pixel_lon=float(pix.climate_pixel_longitude),climate_pixel_lat=float(pix.climate_pixel_latitude),
            climate_pixel_distance_m=float(pix.climate_pixel_distance_m),
            native_interval_years=m['native_step_years_max'],route_family=par['route_family'],
            timevarying=int(par['route_family']=='synthetic_timevarying'))
        for k in ['spin_delta_c','forward_start_c','forward_end_c','forward_correction_change_c','melt_f','prcp_fac','glen_a','base_temp_bias_c']:
            row[k]=scalar(par.get(k))
        diag=dict(system=sid,status=m['status'])
        for short,key,_,unit in METRICS:
            a=np.asarray(c[key]); y=np.asarray(c['native_years']); native=np.asarray(c['native_'+key])
            p=peak(y,native,1500)
            row[short+'_peak_year']=p['first_year']; row[short+'_peak_boundary']=p['left_boundary'] or p['right_boundary']
            row[short+'_interior_peak_year']=np.nan if row[short+'_peak_boundary'] else p['first_year']
            row[short+'_loss_peak_pct']=100*(p['value']-a[-1])/p['value']
            row[short+'_loss_1900_pct']=100*(1-a[-1]/a[400])
            row[short+'_initial_growth_pct']=100*(p['value']/a[0]-1)
            row[short+'_growth_1470_1500_pct']=100*(a[0]/native[0]-1)
            periods=[]
            for start,end in [(1500,1850),(1850,1900),(1900,2000)]:
                diff=a[end-1500]-a[start-1500]
                periods.append(dict(start=start,end=end,absolute_change=float(diff),change_pct=float(100*diff/a[start-1500]),net_rate_per_year=float(diff/(end-start)),unit=unit))
            declines={str(w):evo.decline(a,w) for w in [5,10,20]}
            for w in [5,10,20]: declines[str(w)]['native_resolution_supported']=m['native_step_years_max']<=w
            diag[short]=dict(peak=p,peaks_by_window={str(s):peak(y,native,s) for s in [1470,1500,1550]},
                 common_decadal_peak=peak(np.arange(1500,2001,10),np.interp(np.arange(1500,2001,10),y,native),1500),
                 final_decline=declines,periods=periods,
                 growth_to_peak_pct=row[short+'_initial_growth_pct'],loss_peak_to_2000_pct=row[short+'_loss_peak_pct'])
        row['length_curve_type']='left-boundary peak' if diag['length']['peak']['left_boundary'] else ('right-boundary peak' if diag['length']['peak']['right_boundary'] else 'interior peak with later retreat')
        audit.update(static_v3_volume_km3=inv['lia_reconstructed_volume_km3'],
            dynamic_initial_volume_km3=c['native_volume_km3'][0],
            initial_to_static_volume_ratio=c['native_volume_km3'][0]/inv['lia_reconstructed_volume_km3'],
            historical_peak_length_1470_2000_km=max(c['native_length_km']),
            lia_length_target_km=m['frozen_lia_target_km'],
            peak_target_residual_pct=100*(max(c['native_length_km'])/m['frozen_lia_target_km']-1),
            modern_length_target_km=m['frozen_modern_target_km'],
            length_2000_km=c['length_km'][-1],modern_target_residual_pct=100*(c['length_km'][-1]/m['frozen_modern_target_km']-1),
            modern_inventory_dates=inv['modern_member_outline_dates'],
            caveat='geographical lengths and model observation operators are not identical; outline dates need not be year 2000')
        rows.append(row); audits.append(audit); diagnostics.append(diag)
    return pd.DataFrame(rows),audits,parameters,hyps,diagnostics

def associations(frame):
    predictors=['terminus_elevation_m','gradient_deg','log_area','log_length','relief_m','normalized_mean_height',
                'lower_half_area_fraction','mean_thickness_m','sinuosity','northness','eastness','longitude','latitude',
                'spin_delta_c','forward_start_c','forward_end_c','forward_correction_change_c','melt_f','prcp_fac']
    responses=['length_loss_peak_pct','volume_loss_1900_pct','length_interior_peak_year','volume_interior_peak_year']
    out=[]
    for cohort,_ in COHORTS:
        sub=frame if cohort=='all' else frame[frame.status.eq(cohort)]
        for x in predictors:
            for y in responses: out.append(dict(cohort=cohort,predictor=x,response=y,**rho(sub[x],sub[y])))
    return out

def robustness(frame,curves,diags):
    f=frame[frame.status.eq('retained')]; x='terminus_elevation_m'; y='volume_loss_1900_pct'
    subsets={'retained32':f,'candidate19':frame[frame.status.eq('candidate')],'all51':frame,
        'retained_legacy_constant':f[f.route_family.eq('legacy_constant')],
        'retained_synthetic_timevarying':f[f.route_family.eq('synthetic_timevarying')],
        'retained_no_boundary_volume_peak':f[~f.volume_peak_boundary],
        'retained_peak_after1550':f[f.volume_peak_year.gt(1550)]}
    result=dict(target='terminus elevation versus positive volume loss 1900–2000',
        subsets={k:rho(v[x],v[y]) for k,v in subsets.items()},
        partial=[partial(f,x,y,c) for c in [['longitude'],['log_area'],['timevarying'],['longitude','log_area','timevarying'],['normalized_mean_height'],['longitude','normalized_mean_height','timevarying']]],
        leave_one_glacier_out=[dict(omitted=int(s),**rho(f.loc[f.system.ne(s),x],f.loc[f.system.ne(s),y])) for s in f.system],
        leave_one_pixel_out=[dict(omitted=p,**rho(f.loc[f.climate_pixel.ne(p),x],f.loc[f.climate_pixel.ne(p),y])) for p in sorted(f.climate_pixel.unique())])
    grouped=f.groupby('climate_pixel')[[x,y]].median()
    result['equal_pixel_median_association']=rho(grouped[x],grouped[y])
    legacy=f[f.route_family.eq('legacy_constant')]
    result['constant_route_parameter_adjustment']=partial(legacy,x,y,['forward_start_c','log_area'])
    result['pixel_counts']={co:int((frame if co=='all' else frame[frame.status.eq(co)]).climate_pixel.nunique()) for co,_ in COHORTS}
    result['elevation_boundary_sensitivity']=[]
    for shift in [-100,0,100]:
        labels=np.digitize(f[x],[4000+shift,4500+shift])
        groups=[dict(group=i,n=int(np.sum(labels==i)),median_loss_pct=float(f.loc[labels==i,y].median())) for i in range(3)]
        result['elevation_boundary_sensitivity'].append(dict(shift_m=shift,groups=groups,high_minus_low_pp=groups[2]['median_loss_pct']-groups[0]['median_loss_pct']))
    primary_ids=set(f.system); d=[r for r in diags if r['system'] in primary_ids]
    result['peak_sensitivity']={short:dict(
        left_boundary_by_window={str(s):sum(r[short]['peaks_by_window'][str(s)]['left_boundary'] for r in d) for s in [1470,1500,1550]},
        changed_first_peak_year_1470_vs1500=sum(r[short]['peaks_by_window']['1470']['first_year']!=r[short]['peak']['first_year'] for r in d),
        common_decadal_changed_first_peak=sum(r[short]['common_decadal_peak']['first_year']!=r[short]['peak']['first_year'] for r in d)) for short in ['length','volume']}
    # Common-sampling sensitivity of amplitudes, not just of peak dates.
    result['baseline_sensitivity']=[]
    for baseline in [1500,1850,1900]:
        data=[]
        for c in curves:
            if c['system'] in primary_ids: data.append((c['system'],100*(1-c['volume_km3'][-1]/c['volume_km3'][baseline-1500])))
        aligned=f[['system',x]].merge(pd.DataFrame(data,columns=['system','loss']),on='system')
        result['baseline_sensitivity'].append(dict(baseline_year=baseline,**rho(aligned[x],aligned.loss)))
    return result

def saved_alternatives(members):
    """Existing reviewed alternatives only. No reranking, no simulations, no ensemble probabilities."""
    gallery=read(BASE/'lia_new38_legacy_review_recalculation_v1/data.json')
    objects=gallery['glaciers'] if isinstance(gallery,dict) else gallery
    selected={m['system']:m for m in members if m['status']=='candidate'}
    out=[]
    for g in objects:
        sid=g['system']
        if sid not in selected: continue
        cases=[]
        for case in g['cases']:
            proof=case['proof']; rp=reg(proof['result']['path'],proof['result']['sha256']); sp=reg(proof['state']['path'],proof['state']['sha256'])
            result=read(rp); assert result['candidate']==case['candidate']
            with np.load(sp,allow_pickle=False) as z:
                years=z['years']; history=z['history']; v=history[:,2]/1e9; l=history[:,0]/1000
                p=peak(years,l,1500)
                cases.append(dict(parameters=case['candidate'],result_source=proof['result']['path'],
                    selected=proof['result']['path']==selected[sid]['result_source'],
                    volume_loss_1900_pct=float(100*(1-v[-1]/np.interp(1900,years,v))),
                    length_peak_year=p['first_year'],length_peak_boundary=p['left_boundary'] or p['right_boundary'],
                    length_loss_peak_pct=float(100*(1-l[-1]/p['value']))))
        out.append(dict(system=sid,n=len(cases),cases=cases,
            volume_loss_1900_range=[min(c['volume_loss_1900_pct'] for c in cases),max(c['volume_loss_1900_pct'] for c in cases)],
            scope='saved reviewed alternatives only; configuration range, not confidence interval'))
    return out

def aggregate(frame,curves):
    byid={c['system']:c for c in curves}; results=[]
    for cohort,_ in COHORTS:
        sub=frame if cohort=='all' else frame[frame.status.eq(cohort)]
        entry=dict(cohort=cohort,n=len(sub),systems=sub.system.tolist(),years=YEARS.tolist(),climate_pixels=int(sub.climate_pixel.nunique()))
        for short,key,_,unit in METRICS[1:]:
            arrays=np.asarray([byid[s][key] for s in sub.system]); total=arrays.sum(axis=0)
            p=peak(YEARS,total,1500)
            losses=arrays[:,400]-arrays[:,-1]
            entry[short]=dict(total=total.tolist(),unit=unit,year1500=float(total[0]),year1900=float(total[400]),year2000=float(total[-1]),
                simultaneous_peak=p,common_decadal_peak=peak(YEARS[::10],total[::10],1500),
                peak_precision_note='annual alignment mixes native annual and decadal series; first annual maximum is not annual physical dating precision',
                sum_individual_max=float(arrays.max(axis=1).sum()),
                loss_1900_2000=float(total[400]-total[-1]),loss_1900_2000_pct=float(100*(1-total[-1]/total[400])),
                loss_peak_to_2000_pct=float(100*(1-total[-1]/p['value'])),
                contributions_1900_2000=sorted([dict(system=int(s),loss=float(v),share_pct=float(100*v/losses.sum()) if abs(losses.sum())>1e-12 else None) for s,v in zip(sub.system,losses)],key=lambda r:r['loss'],reverse=True))
        results.append(entry)
    return results

def group_statistics(frame,curves,diags):
    byid={c['system']:c for c in curves}; ds={d['system']:d for d in diags}; out=[]
    for cohort,_ in COHORTS:
        sub=frame if cohort=='all' else frame[frame.status.eq(cohort)]
        for grouping in ['elevation_group','longitude_group','route_family']:
            for name,f in sub.groupby(grouping):
                entry=dict(cohort=cohort,grouping=grouping,group=name,n=len(f),systems=f.system.tolist(),years=YEARS.tolist())
                for short,key,_,unit in METRICS:
                    arrays=np.array([byid[s][key] for s in f.system]); norm=100*(arrays/arrays[:,-1,None]-1)
                    interior=f.loc[~f[short+'_peak_boundary'],short+'_peak_year']
                    entry[short]=dict(unit=unit,absolute_q25_median_q75=np.quantile(arrays,[.25,.5,.75],axis=0).tolist(),
                        relative2000_q25_median_q75=np.quantile(norm,[.25,.5,.75],axis=0).tolist(),
                        peak_interior_n=len(interior),peak_interior_median_year=float(interior.median()) if len(interior) else None,
                        boundary_peak_n=int(f[short+'_peak_boundary'].sum()),
                        median_loss_1900_pct=float(f[short+'_loss_1900_pct'].median()),
                        median_loss_peak_pct=float(f[short+'_loss_peak_pct'].median()),
                        fixed_period_median_net_rates=[float(np.median([ds[s][short]['periods'][i]['net_rate_per_year'] for s in f.system])) for i in range(3)])
                out.append(entry)
    return out

def figsave(name):
    plt.savefig(OUT/(name+'.png'),bbox_inches='tight'); plt.savefig(OUT/(name+'.svg'),bbox_inches='tight'); plt.close()

def figures(frame,curves,hyps,params,assoc,rob,aggs,diags):
    primary=frame[frame.status.eq('retained')]; byid={c['system']:c for c in curves}
    # Map columns use the same scale for both cohorts; coast/background not needed for coordinate inspection.
    fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    columns=[('terminus_elevation_m','LIA末端高程 / m'),('length_loss_peak_pct','历史峰值→2000 长度损失 / %'),('volume_loss_1900_pct','1900→2000 体积损失 / %')]
    for j,(key,label) in enumerate(columns):
        lo,hi=frame[key].min(),frame[key].max()
        for i,(co,title) in enumerate(COHORTS[:2]):
            s=frame[frame.status.eq(co)]; ax=axs[i,j]
            sc=ax.scatter(s.longitude,s.latitude,c=s[key],vmin=lo,vmax=hi,cmap='viridis',s=25+2*s.static_area_km2,edgecolors='white',linewidth=.5,marker='o' if i==0 else '^')
            ax.set_title(title+' · '+label); ax.grid(alpha=.2); ax.set_xlabel('经度 / °E'); ax.set_ylabel('纬度 / °N')
            ax.set_aspect(1/np.cos(np.deg2rad(31)))
            fig.colorbar(sc,ax=ax,shrink=.7)
    fig.suptitle('01 空间差异｜点为冰川系统质心；大小表示静态面积；不是区域连续场')
    fig.tight_layout(); figsave('01_spatial')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for i,(group,labels) in enumerate([('elevation_group',['low','middle','high']),('longitude_group',['west_<80','middle_80_86','east_>=86'])]):
        for j,(short,key,title,unit) in enumerate(METRICS):
            ax=axs[i,j]
            for color,label in zip(COLORS,labels):
                ids=primary.loc[primary[group].eq(label),'system']
                if len(ids)==0: continue
                array=np.array([100*(np.array(byid[s][key])/byid[s][key][-1]-1) for s in ids])
                q=np.quantile(array,[.25,.5,.75],axis=0)
                ax.plot(YEARS,q[1],color=color,label=f'{LABELS[label]}, n={len(ids)}'); ax.fill_between(YEARS,q[0],q[2],color=color,alpha=.14)
            ax.axhline(0,color='gray',lw=.5); ax.set_title(title+'｜'+('末端高程组' if i==0 else '经度带（非气候区）')); ax.set_ylabel('相对各自模拟2000年 / %'); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('02 历史曲线｜仅已认可32条；等冰川中位数与四分位范围；年际插值不增加时间精度')
    fig.tight_layout(); figsave('02_grouped_history')
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,(x,label) in zip(axs.flat,[('terminus_elevation_m','末端高程 / m'),('gradient_deg','主线平均坡度 / °'),('static_area_km2','静态面积 / km²'),('relief_m','表面高差 / m'),('normalized_mean_height','归一化平均高程'),('northness','流线方向北向分量')]):
        for route,col,mark in [('legacy_constant',COLORS[0],'o'),('synthetic_timevarying',COLORS[2],'s')]:
            s=primary[primary.route_family.eq(route)]; ax.scatter(s[x],s.volume_loss_1900_pct,color=col,marker=mark,label=LABELS[route],alpha=.85)
        r=rho(primary[x],primary.volume_loss_1900_pct); ax.set_title(f"ρ={r['rho']:.2f}, n={r['n']}"); ax.set_xlabel(label); ax.set_ylabel('1900→2000 体积损失 / %'); ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7); fig.suptitle('03 地形关系｜已认可32条；颜色区分模型设置；相关不等于因果')
    fig.tight_layout(); figsave('03_geometry')
    fig,axs=plt.subplots(1,2,figsize=(12,4.7))
    for label,col in zip(['low','middle','high'],COLORS):
        ids=set(primary.loc[primary.elevation_group.eq(label),'system']); items=[h for h in hyps if h['system'] in ids]
        arr=np.asarray([np.array(h['area_km2'])/h['total_area_km2'] for h in items]); centers=(np.array(items[0]['elevation_bin_edges_m'])[:-1]+100)
        axs[0].plot(centers,np.mean(arr,axis=0)*100,color=col,label=f'{LABELS[label]}, n={len(items)}')
    axs[0].set(xlabel='表面高程 / m',ylabel='每200 m带平均面积比例 / %',title='等冰川权重高程—面积分布'); axs[0].legend()
    sc=axs[1].scatter(primary.azimuth_deg,primary.volume_loss_1900_pct,c=primary.terminus_elevation_m,cmap='viridis'); axs[1].set(xlabel='头部→末端方位角 / °（顺时针，自北起）',ylabel='1900→2000 体积损失 / %',title='朝向作为方向描述；不能替代辐射量',xticks=[0,90,180,270,360]); fig.colorbar(sc,ax=axs[1],label='末端高程 / m')
    fig.tight_layout(); figsave('04_hypsometry_direction')
    fig,axs=plt.subplots(1,3,figsize=(14,4.5))
    for p in params:
        if p['forward_start_c'] is None or p['forward_end_c'] is None: continue
        years=np.arange(1470,2001); gamma=p['ramp_exponent'] or 1
        values=p['forward_start_c']+(p['forward_end_c']-p['forward_start_c'])*((years-1470)/530)**gamma
        if p['pulse']:
            q=p['pulse']; values+=q['amplitude_c']*np.maximum(0,np.minimum((years-q['start_year'])/(q['peak_year']-q['start_year']),(q['end_year']-years)/(q['end_year']-q['peak_year'])))
        j=['legacy_constant','synthetic_timevarying','candidate_constant'].index(p['route_family'])
        axs[j].plot(years,values,alpha=.6,lw=1)
    for ax,title in zip(axs,['已认可：恒定校正','已认可：时变校正','候选：恒定校正']): ax.set(title=title,xlabel='历史年份',ylabel='附加温度校正 δ / °C'); ax.grid(alpha=.2)
    fig.suptitle('05 参数策略示意｜按保存策略计算年整点校正；不含基准偏差、原始气候异常或月内取样')
    fig.tight_layout(); figsave('05_parameter_policy')
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    items=[(k,v) for k,v in rob['subsets'].items() if v['rho'] is not None]
    axs[0].barh([f"{LABELS[k]} (n={v['n']})" for k,v in items],[v['rho'] for k,v in items],color=COLORS[0]); axs[0].axvline(0,color='gray'); axs[0].set(xlim=(-1,1),xlabel='末端高程—1900至2000体积损失 Spearman ρ',title='样本与路线敏感性')
    for j,key in enumerate(['leave_one_glacier_out','leave_one_pixel_out']):
        vals=[v['rho'] for v in rob[key] if v['rho'] is not None]; axs[1].scatter(vals,np.ones(len(vals))*j,alpha=.5,s=20,label=['逐冰川剔除','逐气候像元剔除'][j])
    axs[1].axvline(rob['subsets']['retained32']['rho'],color='black',ls='--',label='32条全样本'); axs[1].set(yticks=[0,1],yticklabels=['逐冰川剔除','逐气候像元剔除'],xlim=(-1,1),xlabel='Spearman ρ',title='删除敏感性；不是置信区间'); axs[1].legend(fontsize=8)
    fig.tight_layout(); figsave('06_robustness')
    fig,axs=plt.subplots(1,3,figsize=(14,4.8))
    for a,col,(_,title) in zip(aggs,COLORS,COHORTS):
        axs[0].plot(YEARS,a['volume']['total'],label=title,color=col)
        v=np.array(a['volume']['total']); axs[1].plot(YEARS,100*(v/v[-1]-1),label=title,color=col)
    top=aggs[0]['volume']['contributions_1900_2000'][:10]
    axs[2].barh([str(r['system']) for r in top][::-1],[r['loss'] for r in top][::-1],color=COLORS[0])
    axs[0].set(title='固定样本同步总体积',xlabel='年份',ylabel='km³'); axs[1].set(title='同一样本相对模拟2000年',xlabel='年份',ylabel='%'); axs[2].set(title='已认可32条：1900→2000损失前10',xlabel='模拟体积损失 / km³',ylabel='系统编号')
    axs[0].legend(fontsize=8); fig.suptitle('07 样本总体积｜候选不并入主结论；不外推191条或喜马拉雅总体')
    fig.tight_layout(); figsave('07_sample_volume')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,key,title in zip(axs,['length','volume'],['长度','体积']):
        interior=primary[~primary[key+'_peak_boundary']]; boundary=primary[primary[key+'_peak_boundary']]
        sc=ax.scatter(interior.longitude,interior.latitude,c=interior[key+'_peak_year'],vmin=1500,vmax=2000,cmap='plasma',s=65)
        ax.scatter(boundary.longitude,boundary.latitude,marker='x',s=65,c='black',label=f'窗口边界峰值 n={len(boundary)}')
        ax.set(title=title+'最早并列峰值年份',xlabel='经度 / °E',ylabel='纬度 / °N'); ax.legend(fontsize=8); ax.grid(alpha=.2)
        fig.colorbar(sc,ax=ax,label='原生时间步年份')
    fig.suptitle('08 峰值时间的空间分布｜仅主样本；1500—2000窗口；不是独立定年的LIA最大期')
    fig.tight_layout(); figsave('08_peak_timing')
    for c in curves:
        row=frame[frame.system.eq(c['system'])].iloc[0]
        fig,axs=plt.subplots(1,3,figsize=(12,3.2))
        for ax,(short,key,title,unit),col in zip(axs,METRICS,COLORS):
            ax.plot(YEARS,c[key],color=col,lw=1.3)
            y=np.asarray(c['native_years']); a=np.asarray(c['native_'+key]); ok=y>=1500
            if row.native_interval_years>1: ax.scatter(y[ok],a[ok],s=5,color=col)
            p=peak(y,a,1500); ax.axvline(p['first_year'],color='gray',ls=':',lw=.8)
            ax.set(title=title,xlabel='历史年份',ylabel=unit); ax.grid(alpha=.2)
        fig.suptitle(f"系统 {c['system']}｜{'已认可' if row.status=='retained' else '候选'}｜末端 {row.terminus_elevation_m:.0f} m｜原生间隔 {row.native_interval_years:g} 年")
        fig.tight_layout(); plt.savefig(OUT/f"glacier_{c['system']}.png",bbox_inches='tight'); plt.close()

def table(frame,columns=None):
    return frame.to_html(index=False,columns=columns,escape=True,na_rep='未提供',float_format=lambda v:f'{v:.3g}' if 0<abs(v)<.001 else f'{v:.3f}',border=0,classes='data')

def make_html(frame,audits,params,rob,aggs,summary,alts):
    a=aggs[0]['volume']; r=rob['subsets']['retained32']; legacy=rob['subsets']['retained_legacy_constant']
    primary=frame[frame.status.eq('retained')]
    elevation_results='；'.join(f"{LABELS[g]}：{s.volume_loss_1900_pct.median():.2f}%（n={len(s)}）" for g,s in primary.groupby('elevation_group',sort=False))
    longitude_results='；'.join(f"{LABELS[g]}：{s.volume_loss_1900_pct.median():.2f}%（n={len(s)}）" for g,s in primary.groupby('longitude_group',sort=False))
    route_results='；'.join(f"{LABELS[g]}：{s.volume_loss_1900_pct.median():.2f}%（n={len(s)}）" for g,s in primary.groupby('route_family',sort=False))
    interior=primary.loc[~primary.length_peak_boundary,'length_peak_year']
    hyp_r=rho(primary.normalized_mean_height,primary.volume_loss_1900_pct)['rho']
    north_r=rho(primary.northness,primary.volume_loss_1900_pct)['rho']
    east_r=rho(primary.eastness,primary.volume_loss_1900_pct)['rho']
    deletions=[v['rho'] for v in rob['leave_one_pixel_out'] if v['rho'] is not None]
    intro=f"在既有已认可32条中，1900—2000年模拟总体积由 {a['year1900']:.3f} 降至 {a['year2000']:.3f} km³（损失 {a['loss_1900_2000_pct']:.2f}%）。末端高程与该时段单条冰川体积损失比例的描述性相关为 ρ={r['rho']:.2f}；恒定校正分支内为 ρ={legacy['rho']:.2f}（n={legacy['n']}）。这不是高程的独立因果效应。"
    sections=[]
    def section(num,title,text,figs='',extra=''):
        sections.append(f'<section id="s{num}"><h2>{num}. {title}</h2><p>{text}</p>'+''.join(f'<a href="{f}.svg"><img src="{f}.png" loading="lazy" alt="{html.escape(title)}"></a>' for f in figs.split() if f)+extra+'</section>')
    auditdf=pd.DataFrame(audits)
    section(1,'初态—历史—终点：先确认研究对象',
        f"固定保留32条主样本和19条候选，各系统沿用原先代表曲线。{summary['scalar_handoff_matches']}/51 条初态交接的长度、面积、体积标量一致；{summary['scalar_handoff_mismatches']} 条存在不一致，{summary['scalar_handoff_unknown']} 条证据不足。此检查不等同于完整初始厚度数组相等。静态LIA重建与动力初态之比只说明几何转换，不能当作历史损失。历史从1470开始；1400为登记年份，不能从这些曲线推回1400的真实最大值。",extra='<details><summary>逐冰川来源与目标残差</summary>'+table(auditdf,['system','status','scalar_handoff_match','initial_to_static_volume_ratio','peak_target_residual_pct','modern_target_residual_pct','modern_inventory_dates'])+'</details><p>长度目标残差为样本内校准诊断：地图主线与模型长度算子不同，现代轮廓日期也未必是2000年。spinup仅用于交接核查，不作为历史演变曲线。</p>')
    section(2,'历史演变：增长、峰值与退缩',
        f"主窗口1500—2000年。32条中有 {len(interior)} 条长度峰值位于窗口内部，最早并列峰值年份的中位数为 {interior.median():.0f}；另 {int(primary.length_peak_boundary.sum())} 条触及窗口边界，不能赋予明确最大期。1900—2000年各高程组体积损失比例中位数为：{elevation_results}。", '02_grouped_history',extra='<p>长度、面积、体积均保留绝对值与相对模拟2000年的曲线。峰值从原生时间步提取，记录并列年份及边界；阶段变化用固定端点计算。最后持续下降沿用5/10/20年分箱定义，10年原生曲线不支持5年分箱推断。阴影为冰川间四分位范围，不是置信区间。逐条峰值、增长幅度、净变化速率、分箱退缩诊断见 <a href="diagnostics.json">历史诊断数据</a>；<a href="group_statistics.json">分组统计</a>；<a href="curves.json">曲线数据</a>。年度插值不能用于推断年度峰值精度。</p>')
    section(3,'空间位置与终点',
        f"32条主样本按经度带的1900—2000年体积损失比例中位数为：{longitude_results}。这是选定样本的空间对比，不能据此把经度带命名为气候区或外推连续区域趋势。分别展示32条主样本与19条候选；颜色和大小区分历史变化与静态规模。逐条清单保留LIA与现代末端坐标、末端间直线距离、主线长度及方向；末端间直线距离不等于沿谷退缩长度。",'01_spatial 08_peak_timing')
    section(4,'地形几何：哪些属性与变化共同出现',
        f"除末端高程外，归一化平均高程与1900—2000年体积损失也有描述性关系（ρ={hyp_r:.2f}）。流线方向的北向、东向分量相关分别为 {north_r:.2f}、{east_r:.2f}；这不构成朝向效应的独立检验。归一化平均高程定义为（平均表面高程−最低高程）/表面高差，描述面积在高程范围内的分配。",'03_geometry 04_hypsometry_direction',extra='<p>使用配对V3正冰厚栅格计算高程—面积分布、高差、厚度和面积；使用主线计算平均坡度与方向。末端高程不是平衡线高程；主线方向不是坡面辐射；静态几何可能也参与初态构建，不能视为完全独立的观测解释变量。相关分析为探索性描述，不筛选显著结果、不作因果归因。<a href="associations.json">全部相关结果（含样本数与无可计算结果）</a> · <a href="hypsometry.json">逐条高程—面积数据</a></p>')
    section(5,'气候与参数：分开外部强迫、校正策略和冰川响应',
        f"已有主样本分支的1900—2000年体积损失比例中位数为：{route_results}。两组冰川并非相同对象，不能将差值解释成校正策略效应。恒定校正分支内，控制历史校正值和静态面积后的高程—损失条件秩相关为 {rob['constant_route_parameter_adjustment']['rho']:.2f}（n=22）。",'05_parameter_policy',extra='<p>时变分支包含不同幂指数及一条早期脉冲。图为保存策略的年整点示意，不是重建温度序列。物理参数与候选搜索倍率分列；倍率1不代表物理参数值1。本次仅读取气候像元登记与配置，不打开原始气候数组，因此不声称完成温度—降水独立归因或响应滞后识别。降水异常关闭不等于 prcp_fac 未参与校准。</p><details><summary>逐条参数与复现证据</summary>'+table(pd.DataFrame(params),['system','route_family','spin_delta_c','forward_start_c','forward_end_c','ramp_exponent','melt_f','prcp_fac','glen_a','melt_multiplier','prcp_multiplier','numerical_reproduction'])+'</details>'+f'<p>另核查 {len(alts)} 条旧候选的已保存复核组合：<a href="saved_alternative_sensitivity.json">条件性参数范围</a>。这些组合不是随机样本或完整不确定性集合；新增4条未补搜其他组合。</p>')
    partials=pd.DataFrame([dict(控制变量=' + '.join(v['covariates']),n=v['n'],条件秩相关=v['rho']) for v in rob['partial']])
    section(6,'稳健性与验证：结论能走多远',
        f"主样本对应 {rob['pixel_counts']['retained']} 个登记气候像元；不能把32条当作32份完全独立气候信息。逐像元剔除后高程—损失相关范围为 {min(deletions):.2f}—{max(deletions):.2f}；等像元权重中位数相关为 {rob['equal_pixel_median_association']['rho']:.2f}。加入经度、高程—面积分布及模型路线控制后，条件秩相关降为 {rob['partial'][-1]['rho']:.2f}。因此正向联系在这些检查中保留，但解释应同时考虑几何分布和模型设置。",'06_robustness',extra='<p>检验已认可/候选、恒定/时变路线、边界峰值剔除、1470/1500/1550窗口、统一10年取样、基准年、高程阈值±100m，以及逐冰川和逐像元剔除。条件秩相关用于提示混杂，不能消除参数择优和小样本偏差。</p>'+table(partials)+'<p>验证层级：①源文件哈希与算术闭合；②保存的数值复现证据；③独立观测。前两者不等于第三者。LIA长度、现代长度及曾参与参数校准的质量平衡不能再次当作独立检验。本次未建立独立定年冰碛、未参与拟合的历史末端或同空间范围体积观测的联合验证；因此历史峰值日期与体积变化仍是模型诊断。详见 <a href="robustness.json">完整稳健性数据</a>。</p>')
    section(7,'固定样本总体积：把综合量与区域推断分开',
        f"主样本1900—2000年模拟体积净损失 {a['loss_1900_2000']:.3f} km³，占1900年体积 {a['loss_1900_2000_pct']:.2f}%。总体峰值按同一年各冰川体积相加后识别，不能把不同年份的单条最大值相加当作区域峰值。结果仅代表既有选定系统；不外推191条，也不换算整个喜马拉雅总量。",'07_sample_volume',extra=table(pd.DataFrame([dict(样本=co[1],n=g['n'],V1500=g['volume']['year1500'],V1900=g['volume']['year1900'],V2000=g['volume']['year2000'],损失1900_2000百分比=g['volume']['loss_1900_2000_pct']) for co,g in zip(COHORTS,aggs)])))
    head='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>LIA 51条系统分析 v1</title><style>body{font-family:Microsoft YaHei,Arial,sans-serif;background:#f2f5f7;color:#183447;margin:0;line-height:1.8}main{max-width:1280px;margin:auto;padding:26px}header,section{background:white;padding:26px;margin-bottom:20px;border-radius:12px}h1{font-size:30px}h2{font-size:23px}img{width:100%;height:auto}a{color:#12678a}nav{display:flex;flex-wrap:wrap;gap:18px}.lead{font-size:18px;border-left:4px solid #237c9e;padding-left:18px}.note{background:#fff6df;padding:14px}details{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}th{background:#eaf1f4}input{padding:10px;margin:12px 0;min-width:300px}.scroll{overflow:auto}</style><main>'''
    content=head+'<header><h1>LIA冰川历史变化：51条系统分析</h1><p>主分析：32条已认可｜补充：19条候选｜全部结果来自已保存模拟</p><p class="lead">'+intro+'</p><p class="note">本报告完成七类后处理与证据盘点，不意味着完成新的独立观测验证。没有启动新模拟，没有改变候选身份或源文件。</p><nav>'+''.join(f'<a href="#s{i}">{i}. {t}</a>' for i,t in enumerate(['一致性','历史曲线','空间','地形','参数','稳健性','总体积'],1))+'</nav></header>'+''.join(sections)
    content+='<section><h2>51条冰川清单</h2><input id="filter" placeholder="搜索系统编号、分组或路线"><div class="scroll" id="inventory">'+table(frame,['system','status','longitude','latitude','terminus_elevation_m','lia_terminus_lon','lia_terminus_lat','modern_terminus_lon','modern_terminus_lat','terminus_separation_m','static_length_km','static_area_km2','gradient_deg','azimuth_deg','terminal_azimuth_deg','route_family','length_peak_year','volume_loss_1900_pct'])+'</div><p><a href="features.json">全部属性</a> · <a href="methods.json">方法定义</a> · <a href="source_manifest.json">来源哈希</a> · <a href="validation.json">检查结果</a> · <a href="summary.json">摘要数据</a></p></section>'
    content+='<section><h2>逐冰川绝对历史曲线</h2><p>每条均为1500—2000年的长度、体积和面积；虚线表示窗口内最早并列峰值，可能存在多个峰值或边界峰值。10年原生输出标出取样点，连线部分是插值。展开后查看。</p>'+''.join(f'<details><summary>系统 {int(r.system)}｜{"已认可" if r.status=="retained" else "候选"}｜末端 {r.terminus_elevation_m:.0f} m</summary><a href="glacier_{int(r.system)}.png"><img loading="lazy" src="glacier_{int(r.system)}.png" alt="系统{int(r.system)}绝对历史曲线"></a></details>' for r in frame.itertuples())+'</section>'
    content+='<section><h2>与既有研究方法的衔接</h2><p><a href="https://tc.copernicus.org/articles/16/3739/2022/">Yang等（2022）</a>将历史曲线与气候情景及几何响应联系起来；这里借鉴历史阶段、归一化和响应差异的组织方式，不照搬其初始化方案。<a href="https://tc.copernicus.org/articles/14/3135/2020/">Parkes与Goosse（2020）</a>提示初始化影响需要单独检查，因此本报告保留早期窗口敏感性。<a href="https://tc.copernicus.org/articles/19/5913/2025/">Henz等（2025）</a>的静态LIA重建与本项目瞬态历史模拟属于不同问题；长度匹配不能自动验证体积，朝向也不能直接替代辐射。</p><p>下一项能真正提升科学约束的工作，是建立独立观测及其日期、空间范围和是否参与校准的登记表；若要进行气候归因，应另行明确授权、冻结对照实验方案后再运行。</p></section></main><script>document.getElementById("filter").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll("#inventory tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q));});</script></html>'
    (OUT/'index.html').write_text(content,encoding='utf-8')

def main():
    validator=ROOT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'
    result=subprocess.run([sys.executable,'-X','utf8',str(validator)],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    if result.returncode: raise RuntimeError(result.stdout+result.stderr)
    preflight=json.loads(result.stdout); assert preflight['status'].lower()=='pass',preflight.get('failures')
    save('controller_validation.json',preflight)
    for name in ['LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json']:
        reg(ROOT/name)
    reg(Path(__file__)); reg(OUT/'methods.json'); reg(ROOT/'analysis/lia_51_evolution_v2/build_report.py')
    curves,members,sampling=old.load_sources()
    print('loaded immutable representatives: 32 retained + 19 candidates',flush=True)
    frame,audits,params,hyps,diags=build_features(curves,members)
    print('completed source handoff, paired V3 hypsometry, historical diagnostics',flush=True)
    assoc=associations(frame); rob=robustness(frame,curves,diags); alts=saved_alternatives(members); aggs=aggregate(frame,curves)
    groups=group_statistics(frame,curves,diags)
    summary=dict(version='v1',created_at=datetime.now(timezone.utc).isoformat(),n=51,retained=32,candidates=19,
        scalar_handoff_matches=sum(a['scalar_handoff_match'] is True for a in audits),
        scalar_handoff_mismatches=sum(a['scalar_handoff_match'] is False for a in audits),
        scalar_handoff_unknown=sum(a['scalar_handoff_match'] is None for a in audits),
        numerical_reproduction_metadata_present=sum('reproduction_evidence' in p for p in params),
        independent_observational_validation='not established; explicit evidence gap, not implied by source closure',
        route_counts=frame.route_family.value_counts().to_dict(),pixel_counts=rob['pixel_counts'],
        primary_elevation_association=rob['subsets']['retained32'],primary_volume_summary={k:v for k,v in aggs[0]['volume'].items() if k not in ['total','contributions_1900_2000']},
        simulations_launched=False,selection_changed=False,modules_completed=7)
    for name,data in [('members.json',members),('curves.json',curves),('features.json',frame.to_dict('records')),
        ('source_audit.json',audits),('parameters.json',params),('hypsometry.json',hyps),('diagnostics.json',diags),
        ('associations.json',assoc),('robustness.json',rob),('saved_alternative_sensitivity.json',alts),('aggregate.json',aggs),('group_statistics.json',groups),('summary.json',summary)]: save(name,data)
    figures(frame,curves,hyps,params,assoc,rob,aggs,diags)
    make_html(frame,audits,params,rob,aggs,summary,alts)
    save('source_manifest.json',old.SOURCES)
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__': main()
