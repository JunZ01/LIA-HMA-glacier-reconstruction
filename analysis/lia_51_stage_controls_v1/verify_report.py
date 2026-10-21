"""Independent duration/integration, precision-matrix and block-PRESS checks."""
from collections import Counter
from datetime import datetime,timezone
from html.parser import HTMLParser
from pathlib import Path
import hashlib,json
import numpy as np
from scipy.stats import rankdata,spearmanr

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
ATLAS=ROOT/'analysis/lia_51_stage_reliability_v1'
COUNTS=Counter()
CORE=['terminus_elevation_m','gradient_deg','log_area','normalized_mean_height','longitude',
      'base_temp_bias_c','q_mean_1500_2000_c','melt_f','prcp_fac','log_glen_a']


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def close(a,b):assert np.allclose(a,b,rtol=2e-9,atol=2e-9),(a,b)


def integrate(y,x,start,end,threshold):
    area=0.;duration=0.
    for i in range(len(y)-1):
        left=max(start,y[i]);right=min(end,y[i+1])
        if right<=left:continue
        slope=(x[i+1]-x[i])/(y[i+1]-y[i])
        a=x[i]+slope*(left-y[i]);b=x[i]+slope*(right-y[i])
        area+=(a+b)*(right-left)/2
        if min(a,b)>=threshold:duration+=right-left
        elif max(a,b)<threshold:continue
        else:
            crossing=left+(threshold-a)/slope
            duration+=(crossing-left) if slope<0 else (right-crossing)
    return area/(end-start),duration


def independent_profile(years,values,step=10,start=1500,offset=0,fraction=.01):
    y=np.array(sorted(set([1470]+list(range(1480,2001,step)))),float)
    x=np.interp(y,years,values)
    maximum=max([float(np.interp(start,y,x))]+x[y>=start].tolist())
    early,ne=integrate(y,x,1600+offset,1700+offset,maximum*(1-fraction))
    late,nl=integrate(y,x,1800+offset,1900+offset,maximum*(1-fraction))
    return dict(maximum=maximum,early_mean=early,late_mean=late,contrast_pp=100*(late-early)/maximum,
                early_near_percent=ne,late_near_percent=nl,near_contrast_pp=nl-ne),y,x


def check_saved_profile(years,values,d):
    result,y,x=independent_profile(years,values,d['step'],d['normalization_start'],d['offset'],d['fraction'])
    for k,v in result.items():close(d[k],v)
    for p in d['periods']:
        mean,dur=integrate(y,x,p['start'],p['end'],result['maximum']*(1-d['fraction']))
        close(p['mean'],mean);close(p['mean_relative_to_max_pct'],100*mean/result['maximum']);close(p['near_duration_years'],dur)
        assert 0<=dur<=100+1e-9
        a,b=np.interp([p['start'],p['end']],y,x);close(p['net_change_pct'],100*(b-a)/a)
        COUNTS['period_integrals']+=1
    COUNTS['profiles']+=1


def subcohort(rows,name):
    if name=='all51':return rows
    if name=='constant41':return [r for r in rows if r['policy_group']=='constant']
    return [r for r in rows if r['status']==('retained' if name=='original32' else 'candidate')]


def joint_controls(rows,x):return [k for k in CORE if k!=x]+[k for k in ['route_candidate','route_synthetic'] if len({r[k] for r in rows})>1]


def independent_rho(rows,x,y,controls=None):
    controls=controls or [];data=np.round([[r[k] for k in [x,y]+controls] for r in rows],10)
    minimum=max(12,3*(len(controls)+2)) if controls else 4
    if len(data)<minimum or min(np.std(data,axis=0))<1e-12:return None
    if not controls:return float(spearmanr(data[:,0],data[:,1]).statistic)
    ranks=np.column_stack([rankdata(data[:,i]) for i in range(data.shape[1])])
    correlation=np.corrcoef(ranks.T)
    precision=np.linalg.inv(correlation)
    return float(-precision[0,1]/np.sqrt(precision[0,0]*precision[1,1]))


def check_corr(rows,x,y,record):
    expected=independent_rho(rows,x,y,record.get('controls'))
    if expected is None:assert record['rho'] is None
    else:close(record['rho'],expected)
    COUNTS['correlations']+=1


class LinkParser(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img' and 'src' in a:self.links.append(a['src'])


def main():
    validation=read(OUT/'validation.json')
    for path,expected in read(OUT/'source_manifest.json').items():assert sha(ROOT/path)==expected,path
    for path,expected in validation['authority_hashes'].items():assert sha(ROOT/path)==expected,path
    for name,expected in read(OUT/'output_manifest.json').items():
        if name in [Path(__file__).name,'visual_review.json']:continue
        assert sha(OUT/name)==expected,name
    rows=read(OUT/'features_51.json');byid={r['system']:r for r in rows}
    curves={c['system']:c for c in read(ATLAS/'reference_curves.json')}
    profiles=read(OUT/'period_profiles.json')
    for p in profiles:
        c=curves[p['system']]
        for key in ['length_km','volume_km3']:
            check_saved_profile(np.array(c['native_years']),np.array(c['native_'+key]),p['metrics'][key])
            close(byid[p['system']][key.split('_')[0]+'_contrast_pp'],p['metrics'][key]['contrast_pp'])
    for v in read(OUT/'metric_variants.json'):
        for r in v['records']:
            c=curves[r['system']]
            for key in ['length_km','volume_km3']:
                d,_,_=independent_profile(np.array(c['native_years']),np.array(c['native_'+key]),v['step'],v['start'],v['offset'],v['fraction'])
                for ending in ['contrast_pp','near_contrast_pp']:close(r[key.split('_')[0]+'_'+ending],d[ending])
                COUNTS['variant_metric_checks']+=1
    case_profiles=read(OUT/'case_profiles.json')
    for c in case_profiles:
        with np.load(ROOT/c['state_source'],allow_pickle=False) as z:y=z['years'].copy().astype(float);v=z['history'].copy()
        for key,i,scale in [('length_km',0,1e3),('volume_km3',2,1e9)]:check_saved_profile(y,v[:,i]/scale,c['metrics'][key])
    for record in read(OUT/'associations.json'):
        check_corr(subcohort(rows,record['cohort']),record['predictor'],record['response'],record)
    for record in read(OUT/'pixel_robustness.json'):
        sub=subcohort(rows,record['cohort']);x=record['predictor'];y=record['response']
        for d in record['deletions']:
            remain=[r for r in sub if r['climate_pixel']!=d['pixel']]
            for mode in ['raw','joint']:check_corr(remain,x,y,d[mode])
    variants=read(OUT/'metric_variants.json')
    variant_lookup={(v['offset'],v['step'],v['start'],v['fraction']):v['records'] for v in variants}
    for record in read(OUT/'definition_sensitivity.json'):
        selected=variant_lookup[(record['offset'],record['step'],record['start'],record['fraction'])]
        sub=[{**r,**d} for r,d in zip(rows,selected)]
        for mode in ['raw','joint']:check_corr(sub,record['predictor'],record['response'],record[mode])
    cases={c['case_id']:c for c in case_profiles}
    for replacement in read(OUT/'alternative_replacements.json'):
        c=cases[replacement['case_id']];assert not c['selected']
        sub=[dict(r) for r in rows]
        target=next(r for r in sub if r['system']==c['system'])
        target['q_mean_1500_2000_c']=c['parameters']['constant_offset_degC']
        for key in ['length_km','volume_km3']:target[key.split('_')[0]+'_contrast_pp']=c['metrics'][key]['contrast_pp']
        for record in replacement['correlations']:
            x,y=record['predictor'],record['response']
            for mode in ['raw','joint']:
                cc=[] if mode=='raw' else joint_controls(sub,x)
                value=independent_rho(sub,x,y,cc);close(record[mode],value)
                COUNTS['replacement_correlations']+=1
    # A separate block-PRESS identity verifies every leave-pixel-out prediction.
    for model in read(OUT/'cross_validation.json'):
        sub=subcohort(rows,model['cohort']);y=np.array([r[model['response']] for r in sub]);keys=model['predictors']
        if keys:
            x=np.array([[r[k] for k in keys] for r in sub]);x=(x-x.mean(axis=0))/x.std(axis=0)
            z=np.c_[np.ones(len(sub)),x]
        else:z=np.ones((len(sub),1))
        hat=z@np.linalg.solve(z.T@z,z.T);residual=y-hat@y
        close(model['training_r2'],1-sum(residual**2)/sum((y-y.mean())**2))
        sse=0.;sse0=0.
        for pixel in sorted({r['climate_pixel'] for r in sub}):
            ix=[i for i,r in enumerate(sub) if r['climate_pixel']==pixel]
            hold_residual=np.linalg.solve(np.eye(len(ix))-hat[np.ix_(ix,ix)],residual[ix])
            prediction=y[ix]-hold_residual
            mean=float(np.mean([r[model['response']] for r in sub if r['climate_pixel']!=pixel]))
            for i,value in zip(ix,prediction):
                actual=next(p for p in model['predictions'] if p['system']==sub[i]['system'])
                close(actual['predicted'],value);close(actual['training_mean'],mean)
                sse+=(y[i]-value)**2;sse0+=(y[i]-mean)**2
                COUNTS['held_out_predictions']+=1
        assert not model['failed_folds'] and model['held_out_n']==len(sub)
        close(model['q2'],1-sse/sse0);close(model['rmse_pp'],np.sqrt(sse/len(sub)))
        COUNTS['prediction_models']+=1
    summary=read(OUT/'summary.json')
    multi=[p for p in {r['climate_pixel'] for r in rows} if sum(r['climate_pixel']==p for r in rows)>1]
    assert len(multi)==11 and sum(r['climate_pixel'] in multi for r in rows)==29
    assert all(len({round(r['prcp_fac'],10) for r in rows if r['climate_pixel']==p})==1 for p in multi)
    assert summary['precipitation_factor_constant_within_all_multi_system_pixels']
    assert COUNTS['held_out_predictions']==920 and COUNTS['prediction_models']==20
    parser=LinkParser();parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:
        if '://' not in link and not link.startswith('#'):assert (OUT/link).exists(),link
    COUNTS['html_links']=len(parser.links)
    validation['independent_verification']=dict(status='pass',checked_utc=datetime.now(timezone.utc).isoformat(),counts=dict(COUNTS),
        method='Direct clipped-segment integration/duration; precision-matrix partial correlations; independent block-PRESS held-out predictions. Not observational validation.')
    validation['status']='numerical-pass-visual-review-pending'
    if (OUT/'visual_review.json').exists():
        review=read(OUT/'visual_review.json')
        for name,digest in review['figure_hashes'].items():assert sha(OUT/name)==digest,name
        assert review['status']=='pass';validation['visual_review']=review;validation['status']='pass'
    save('validation.json',validation)
    save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(validation['independent_verification'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
