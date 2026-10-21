"""Independent frozen-input, parameter, arithmetic and partial-correlation verification."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import json,hashlib,importlib.util
from datetime import datetime,timezone
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=ROOT/'analysis/lia_51_systematic_analysis_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
def same(a,b,msg):assert np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True),msg
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def subset(f,name):
    if name=='constant41':return f[f.policy_group.eq('constant')]
    if name=='timevarying10':return f[f.policy_group.eq('timevarying')]
    if name=='original32_sensitivity':return f[f.status.eq('retained')]
    if name=='original19_sensitivity':return f[f.status.eq('candidate')]
    assert name=='all51';return f
def independent_partial(f,x,y,controls):
    d=f[[x,y]+controls].dropna().round(10)
    # Precision-matrix identity is independent of the report's least-squares residual calculation.
    ranks=d.rank(method='average').to_numpy();r=np.corrcoef(ranks.T);precision=np.linalg.inv(r)
    return float(-precision[0,1]/np.sqrt(precision[0,0]*precision[1,1]))

class Links(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.images=0
    def handle_starttag(self,t,attrs):
        a=dict(attrs)
        if t=='a':self.refs.append(a.get('href',''))
        if t=='img':self.refs.append(a['src']);self.images+=1;assert a.get('alt')

def main():
    for p in OUT.glob('*.json'):read(p)
    sources=read(OUT/'source_manifest.json')
    for name,digest in sources.items():assert sha(ROOT/name)==digest,(name,'source mutated')
    f=pd.DataFrame(read(OUT/'parameters_51.json'));assert len(f)==51 and f.system.nunique()==51
    assert f.policy_group.value_counts().to_dict()=={'constant':41,'timevarying':10}
    oldparams={r['system']:r for r in read(OLD/'parameters.json')};oldfeatures={r['system']:r for r in read(OLD/'features.json')}
    curves={r['system']:r for r in read(OLD/'curves.json')};policies=read(OUT/'temperature_policies.json')
    design=read(ROOT/'validation_data/results/paleo_surface_validation_v0/lia_191_gpu_four_group_design_v10_262_v1/four_group_parameter_matrix.json')
    baseline=pd.read_csv(ROOT/design['baseline_source']['path'])
    assert sha(ROOT/design['baseline_source']['path'])==design['baseline_source']['sha256']
    fields=design['baseline_source']['per_system_fields']
    for row,policy in zip(f.to_dict('records'),policies):
        sid=row['system'];p=oldparams[sid];c=curves[sid];assert policy['system']==sid
        assert row['status']==oldfeatures[sid]['status']
        if row['status']=='candidate':
            b=baseline.loc[baseline.lia_unique_id.eq(sid),fields].drop_duplicates();assert len(b)==1;b=b.iloc[0]
            for key,bkey,mult in [('melt_f','melt_f','melt_multiplier'),('prcp_fac','prcp_fac','prcp_multiplier'),('glen_a','glen_a_pa_3_s_1','glen_multiplier')]:
                # Relative-only comparison matters for tiny Glen-A values.
                assert np.isclose(row[key],float(b[bkey])*p[mult],rtol=1e-12,atol=0)
            same(row['base_temp_bias_c'],float(b.base_temp_bias_degC),'restored base bias')
        else:
            for key in ['base_temp_bias_c','melt_f','prcp_fac','glen_a']:assert np.isclose(row[key],p[key],rtol=1e-12,atol=0)
        assert row['temp_melt_c']==-1
        y=np.array(policy['years']);q=np.array(policy['q_c']);assert np.array_equal(y,np.arange(1470,2001))
        exponent=p['ramp_exponent'] if p['ramp_exponent'] is not None else 1
        expected=p['forward_start_c']+(p['forward_end_c']-p['forward_start_c'])*((y-1470)/530.)**exponent
        if p.get('pulse'):
            pulse=p['pulse']
            addition=np.interp(y,[pulse['start_year'],pulse['peak_year'],pulse['end_year']],[0,pulse['amplitude_c'],0],left=0,right=0)
            expected=expected+addition
        same(q,expected,'policy formula including pulse')
        same(policy['base_plus_q_c'],q+row['base_temp_bias_c'],'combined policy bias')
        same(row['q_mean_1500_2000_c'],np.mean(q[y>=1500]),'q average')
        same(row['q_change_1900_2000_c'],q[-1]-q[y==1900][0],'20th century q change')
        same(row['volume_loss_1900_pct'],100*(1-c['volume_km3'][-1]/c['volume_km3'][400]),'response closure')
    associations=read(OUT/'associations.json')
    for a in associations:
        d=subset(f,a['cohort'])[[a['parameter'],a['response']]].dropna().round(10);assert len(d)==a['n']
        if a['rho'] is not None:same(a['rho'],spearmanr(d.iloc[:,0],d.iloc[:,1]).statistic,'Spearman')
        else:assert len(d)<4 or any(d[k].nunique()==1 for k in d)
    adjusted=read(OUT/'adjusted_associations.json');tested=0
    for a in adjusted:
        if a['rho'] is not None:
            same(a['rho'],independent_partial(subset(f,a['cohort']),a['parameter'],a['response'],a['controls']),'partial via precision matrix');tested+=1
        else:assert a['reason']
    deletions=0
    for item in read(OUT/'robustness.json'):
        sub=subset(f,item['cohort']);x=item['parameter'];y='volume_loss_1900_pct'
        assert set(d['omitted'] for d in item['deletions'])==set(sub.climate_pixel)
        for d in item['deletions']:
            keep=sub[sub.climate_pixel.ne(d['omitted'])];a=d['joint']
            if a['rho'] is not None:same(a['rho'],independent_partial(keep,x,y,a['controls']),'deletion partial')
            else:assert a['reason']
            assert len(keep)==d['raw']['n'];deletions+=1
        pixel=sub.groupby('climate_pixel')[[x,y]].median().round(10)
        same(item['equal_pixel_medians']['rho'],spearmanr(pixel[x],pixel[y]).statistic,'pixel equal weight')
    groups=read(OUT/'group_statistics.json')
    for x in set(g['parameter'] for g in groups):
        sets=[set(g['systems']) for g in groups if g['parameter']==x]
        assert set.union(*sets)==set(f.system) and sum(map(len,sets))==51
    for g in groups:
        assert g['n']==len(g['systems'])
        sub=f[f.system.isin(g['systems'])];bounds=np.quantile(f[g['parameter']].round(10),[1/3,2/3]);same(g['bounds'],bounds,'group bounds')
        assert np.all(np.digitize(sub[g['parameter']].round(10),bounds)+1==g['group'])
        for key in ['length_km','volume_km3']:
            a=np.array([curves[s][key] for s in g['systems']]);expected=np.percentile(100*(a/a[:,-1,None]-1),[25,50,75],axis=0)
            same(g[key]['q25_median_q75'],expected,'group normalized quantiles')
    for s in read(OUT/'parameter_summary.json'):
        a=subset(f,s['cohort'])[s['parameter']]
        assert s['n']==len(a)
        same([s['minimum'],s['median'],s['maximum']],[a.min(),a.median(),a.max()],'parameter summary')
    spec=importlib.util.spec_from_file_location('parameter_report',OUT/'build_report.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture=pd.DataFrame({'x':np.arange(40),'y':np.arange(40),'z':np.tile([0,1],20)})
    same(module.partial(fixture,'x','y',['z'])['rho'],1,'perfect residual fixture')
    assert module.partial(fixture.iloc[:10],'x','y',['z'])['rho'] is None
    fixture['constant']=1;assert module.corr(fixture,'constant','x')['rho'] is None
    # Floating point differences do not invent distinct ranks for equal parameters.
    fixture['near_constant']=1+np.arange(40)*1e-14;assert module.corr(fixture,'near_constant','x')['rho'] is None
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for ref in parser.refs:
        if ref=='validation.json' or ref.startswith(('https://','#')):continue
        assert (OUT/ref).is_file(),ref
    assert parser.images==8
    summary=read(OUT/'summary.json');assert summary['unified_sample']==51 and summary['climate_pixels']==33
    assert not any(summary[k] for k in ['simulations_launched','raw_climate_accessed','acceptance_ledger_modified'])
    output_manifest={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['validation.json','output_manifest.json']}
    (OUT/'output_manifest.json').write_text(json.dumps(output_manifest,indent=2),encoding='utf-8')
    result=dict(status='PASS',verified_at=datetime.now(timezone.utc).isoformat(),systems=51,restored_physical_parameter_rows=19,
        source_hashes=len(sources),association_checks=len(associations),partial_precision_matrix_checks=tested,
        pixel_deletion_checks=deletions,parameter_curve_groups=len(groups),figures=8,fixtures=4,
        formal_acceptance_unchanged=True,independent_climate_attribution='not established by this descriptive analysis')
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
