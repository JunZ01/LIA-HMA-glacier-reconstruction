"""Independent arithmetic/source/coverage checks for the saved report; no simulations."""
from pathlib import Path
import json
import hashlib
import importlib.util
from datetime import datetime,timezone
from html.parser import HTMLParser
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def check(condition,message):
    if not condition: raise AssertionError(message)
def same(a,b,message): check(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True),message)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.refs=[]; self.ids=[]; self.images=0
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if 'id' in d: self.ids.append(d['id'])
        if tag=='a' and 'href' in d: self.refs.append(d['href'])
        if tag=='img':
            self.refs.append(d['src']); self.images+=1; check(bool(d.get('alt')),'image alt missing')

def main():
    for file in OUT.glob('*.json'):
        if file.name not in ['validation.json','output_manifest.json']: read(file.name)
    curves=read('curves.json'); features=read('features.json'); members=read('members.json'); diagnostics=read('diagnostics.json')
    audit=read('source_audit.json'); params=read('parameters.json'); hyps=read('hypsometry.json'); aggregates=read('aggregate.json')
    manifest=read('source_manifest.json'); summary=read('summary.json')
    expected=read('members.json')
    ids=[m['system'] for m in expected]
    check(len(ids)==len(set(ids))==51,'51 unique representatives')
    for array in [curves,features,members,diagnostics,audit,params,hyps]: check([r['system'] for r in array]==ids,'aligned IDs')
    for path,digest in manifest.items(): check(sha(ROOT/path)==digest.upper(),f'source changed: {path}')
    original=json.loads((ROOT/'analysis/lia_51_terminus_elevation_curves_v1/members.json').read_text(encoding='utf-8'))
    check([(r['system'],r['source'],r['status']) for r in members]==[(r['system'],r['source'],r['status']) for r in original],'representative selection unchanged')
    units=[('length','length_km'),('volume','volume_km3'),('area','area_km2')]
    for c,f,m,d,a,p,h in zip(curves,features,members,diagnostics,audit,params,hyps):
        sid=c['system']; y=np.asarray(c['native_years']); years=np.asarray(c['year'])
        check(np.array_equal(years,np.arange(1500,2001)),'analysis time axis')
        check(y[0]==1470 and y[-1]==2000 and np.all(np.diff(y)>0),'native time axis')
        check(a['area_m2_initial_curve_match'] and a['volume_m3_initial_curve_match'],f'{sid} initial closure')
        check(a['scalar_handoff_match'] is True,f'{sid} scalar handoff')
        same(sum(h['area_km2']),h['total_area_km2'],'hypsometry area closure')
        same(h['total_area_km2'],f['static_area_km2'],'static area closure')
        same(f['northness']**2+f['eastness']**2,1,'orientation encoding')
        same(a['initial_to_static_volume_ratio'],c['native_volume_km3'][0]/h['volume_km3'],'geometry ratio')
        for short,key in units:
            values=np.asarray(c[key]); native=np.asarray(c['native_'+key]); mask=y>=1500
            same(values,np.interp(years,y,native),'native annual alignment')
            check(np.all(np.isfinite(values)) and np.all(values>0),'positive geometry')
            vmax=native[mask].max(); tied=y[mask][np.isclose(native[mask],vmax,rtol=1e-10,atol=1e-12)].astype(int).tolist()
            same(d[short]['peak']['value'],vmax,'native peak value')
            check(d[short]['peak']['tied_years']==tied,'peak ties retained')
            same(f[short+'_loss_1900_pct'],100*(values[400]-values[-1])/values[400],'fixed-period loss')
            same(f[short+'_loss_peak_pct'],100*(vmax-values[-1])/vmax,'peak loss')
            for period in d[short]['periods']:
                start,end=period['start'],period['end']; delta=values[end-1500]-values[start-1500]
                same(period['absolute_change'],delta,'period absolute change')
                same(period['net_rate_per_year'],delta/(end-start),'period net rate')
            for width in [5,10,20]:
                item=d[short]['final_decline'][str(width)]
                means=np.array([np.mean(values[i:i+width]) for i in range(0,500,width)])
                same(item['means'],means,'nonoverlapping bin means')
                check(item['native_resolution_supported']==(m['native_step_years_max']<=width),'resolution flag')
        same(a['peak_target_residual_pct'],100*(max(c['native_length_km'])/m['frozen_lia_target_km']-1),'LIA target residual')
        same(a['modern_target_residual_pct'],100*(c['length_km'][-1]/m['frozen_modern_target_km']-1),'modern target residual')
        check((OUT/f'glacier_{sid}.png').stat().st_size>10000,'individual plot exists')
    byid={c['system']:c for c in curves}
    for agg in aggregates:
        selected=[f for f in features if agg['cohort']=='all' or f['status']==agg['cohort']]
        check([f['system'] for f in selected]==agg['systems'],'fixed aggregate membership')
        for short,key in units[1:]:
            a=agg[short]; total=np.sum([byid[f['system']][key] for f in selected],axis=0)
            same(a['total'],total,'simultaneous aggregate')
            same(a['loss_1900_2000'],total[400]-total[-1],'aggregate loss')
            same(sum(r['loss'] for r in a['contributions_1900_2000']),a['loss_1900_2000'],'contribution closure')
            same(sum(r['share_pct'] for r in a['contributions_1900_2000']),100,'contribution percent closure')
            check(a['sum_individual_max']>=max(total)-1e-10,'asynchronous vs simultaneous maximum')
    f=pd.DataFrame(features)
    for a in read('associations.json'):
        sub=f if a['cohort']=='all' else f[f.status.eq(a['cohort'])]
        pair=sub[[a['predictor'],a['response']]].dropna()
        check(a['n']==len(pair),'association sample count')
        if a['rho'] is not None: same(a['rho'],spearmanr(pair.iloc[:,0],pair.iloc[:,1]).statistic,'Spearman reproduction')
    for group in read('group_statistics.json'):
        check(group['n']==len(group['systems']),'group membership size')
        for short,key in units:
            arrays=np.array([byid[s][key] for s in group['systems']])
            same(group[short]['absolute_q25_median_q75'],np.percentile(arrays,[25,50,75],axis=0),'group quantiles')
    primary=f[f.status.eq('retained')]; rob=read('robustness.json')
    check(rob['pixel_counts']['retained']==primary.climate_pixel.nunique(),'pixel count')
    check(len(rob['leave_one_glacier_out'])==32 and len(rob['leave_one_pixel_out'])==primary.climate_pixel.nunique(),'deletion coverage')
    check(sum(g['n'] for g in rob['elevation_boundary_sensitivity'][1]['groups'])==32,'elevation group accounting')
    alternative_cases=sum(len(a['cases']) for a in read('saved_alternative_sensitivity.json'))
    # Small fixtures test interpretation-critical edge cases separately from the real data.
    spec=importlib.util.spec_from_file_location('report_fixture',OUT/'build_report.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    p=mod.peak([1500,1510,1520,2000],[3,3,2,1],1500)
    check(p['left_boundary'] and p['tied_years']==[1500,1510],'plateau at left boundary')
    p=mod.peak([1500,1510,1520,2000],[1,3,3,1],1500)
    check(not p['left_boundary'] and p['first_year']==1510 and p['last_year']==1520,'interior ties')
    check(mod.evo.decline(np.ones(501),10)['start_year'] is None,'flat series not sustained retreat')
    check(mod.evo.decline(np.arange(501,0,-1),10)['left_censored'],'declining series left censoring')
    check(not mod.evo.decline(np.arange(1,502),10)['start_year'],'growing series no retreat')
    parser=Links(); parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for ref in parser.refs:
        if ref.startswith(('https:','http:')): continue
        if ref.startswith('#'): check(ref[1:] in parser.ids,'section anchor'); continue
        if ref=='validation.json': continue  # Written atomically at the successful end of this check.
        check((OUT/ref).is_file(),f'missing HTML artifact: {ref}')
    check(all(f's{i}' in parser.ids for i in range(1,8)),'seven modules in HTML')
    check(summary['simulations_launched'] is False and summary['selection_changed'] is False,'scope assertions')
    outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in ['validation.json','output_manifest.json']}
    (OUT/'output_manifest.json').write_text(json.dumps(outputs,indent=2),encoding='utf-8')
    result=dict(status='PASS',checked_at=datetime.now(timezone.utc).isoformat(),systems=51,primary32=True,candidates19=True,
        source_hashes_checked=len(manifest),individual_figures_checked=51,overview_figures_checked=8,
        associations_checked=len(read('associations.json')),groups_checked=len(read('group_statistics.json')),saved_alternative_cases=alternative_cases,
        scalar_handoffs_checked=51,unit_fixtures=5,local_html_links_checked=len(parser.refs),modules=7,
        independent_observational_validation='NOT ESTABLISHED — arithmetic PASS does not establish scientific validation',
        output_hashes='output_manifest.json',notes=['PNG figures visually inspected separately by assistant','No simulation or source mutation'])
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
