"""Independent arithmetic/source verification of the saved-output atlas."""
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import csv
import hashlib
import itertools
import json
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
KEYS=[('length_km',0,1e3),('area_km2',1,1e6),('volume_km3',2,1e9)]
COUNTS=Counter()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def distance(a,b):
    if not a or not b:return None
    return max(max(min(abs(x-y) for y in b) for x in a),max(min(abs(y-x) for x in a) for y in b))


def close(a,b):
    assert np.allclose(a,b,atol=1e-10,rtol=1e-11),(a,b)


def independent_label(r,reference,t,a,f):
    if r['case_count']<2:return 'untested'
    if r['length_km']['boundary_any'] or r['volume_km3']['boundary_any']:return 'boundary'
    if any(v>a for v in r['amplitude_ranges_pp'].values()):return 'amplitude_sensitive'
    for key in ['length_km','volume_km3']:
        if r[key]['near_set_pairwise_distances'][str(f)]>t:return 'timing_sensitive'
        variants=reference['metrics'][key]['sensitivity']
        primary=next(v['near']['years'] for v in variants if (v['start'],v['step'],v['fraction'])==(1500,10,f))
        for v in variants:
            if v['fraction']!=f or (v['step']!=10 and (v['step'],v['start'])!=(20,1500)):continue
            overlap=max(v['start'],1500)
            dd=distance([x for x in primary if x>=overlap],[x for x in v['near']['years'] if x>=overlap])
            if dd is None or dd>t:return 'timing_sensitive'
    return 'stable'


def check_metric(years,values,d):
    mask=(years>=1500)&(years<=2000)
    y,x=years[mask],values[mask]
    p=d['native_peak'];close(p['value'],max(x))
    tied=y[np.isclose(x,max(x),rtol=1e-10,atol=1e-12)].tolist()
    represented=[t for t in y.tolist() if any(a<=t<=b for a,b in p['segments'])]
    assert tied==represented and p['first_year']==tied[0] and p['last_year']==tied[-1]
    assert p['left_boundary']==(y[0] in tied) and p['right_boundary']==(y[-1] in tied)
    close(d['loss_peak_to_2000_pct'],100*(max(x)-values[-1])/max(x))
    v1900=np.interp(1900,years,values)
    close(d['loss_1900_2000_pct'],100*(v1900-values[-1])/v1900)
    for s in d['sensitivity']:
        grid=np.unique(np.r_[np.arange(s['start'],2001,s['step']),2000]).astype(float)
        v=np.interp(grid,years,values);peak=max(v)
        expected=[t for t,z in zip(grid,v) if z>=peak*(1-s['fraction'])-1e-12*max(1,peak)]
        assert expected==s['near']['years']
        represented=[t for t in grid.tolist() if any(a<=t<=b for a,b in s['near']['segments'])]
        assert represented==expected
        for local in s['local_peaks']:
            i=grid.tolist().index(local['year'])
            assert 0<i<len(grid)-1 and v[i]>=v[i-1] and v[i]>=v[i+1]
            assert local['prominence_pct']+1e-10>=s['fraction']*100
        COUNTS['near_maximum_variants']+=1
    for k in ['interval','interval20']:
        d2=d[k];yy=np.array(d2['years']);vv=np.interp(yy,years,values)
        close(vv,d2['values'])
        signs=[1 if b-a>.001*max(vv) else -1 if b-a<-.001*max(vv) else 0 for a,b in zip(vv[:-1],vv[1:])]
        assert signs==d2['states']
        runs=d2['runs'];assert runs[0]['start']==1500 and runs[-1]['end']==2000
        assert all(a['end']==b['start'] for a,b in zip(runs[:-1],runs[1:]))
        final=d2['final_decline_start']
        if final is not None:
            idx=yy.tolist().index(final)
            assert 2000-final>=30 and 1 not in signs[idx:] and vv[idx]-vv[-1]>=.01*max(vv)
        COUNTS['interval_diagnostics']+=1
    for p in d['periods']:
        start,end=np.interp([p['start'],p['end']],years,values)
        close(p['absolute'],end-start);close(p['change_pct'],100*(end-start)/start)
    COUNTS['metric_diagnostics']+=1


def raw_reference(d):
    path=ROOT/d['source']
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:
            return z['years'].copy().astype(float),{k:z['history'][:,i].copy()/scale for k,i,scale in KEYS}
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    rows=[r for r in rows if ('phase' not in r or r['phase']=='forward') and 1470<=float(r['year'])<=2000]
    rows.sort(key=lambda r:float(r['year']))
    years=np.array([float(r['year']) for r in rows])
    raw={}
    for k,col,scale in KEYS:
        source=d['length_operator'] if k=='length_km' else ('area_m2' if k=='area_km2' else 'volume_m3')
        raw[k]=np.array([float(r.get(source) or r['length_m']) for r in rows])/scale
    return years,raw


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.rows=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img' and 'src' in a:self.links.append(a['src'])
        if tag=='tr':self.rows+=1


def main():
    atlas=read(OUT/'atlas.json');cases=read(OUT/'case_diagnostics.json');rels=read(OUT/'conditional_reliability.json')
    summary=read(OUT/'summary.json');activity=read(OUT/'regional_activity.json')
    validation=read(OUT/'validation.json');sources=read(OUT/'source_manifest.json')
    for p,expected in sources.items():assert digest(ROOT/p)==expected,p
    for p,expected in validation['canonical_hashes'].items():assert digest(ROOT/p)==expected,p
    for p,expected in read(OUT/'output_manifest.json').items():
        if p==Path(__file__).name:
            if digest(OUT/p)!=expected:
                validation.setdefault('verifier_code_revisions',[]).append(dict(previous=expected,current=digest(OUT/p)))
            continue
        assert digest(OUT/p)==expected,p
    COUNTS['source_hashes']=len(sources)
    assert len(atlas)==51 and len({d['system'] for d in atlas})==51 and len(cases)==113
    assert len({d['climate_pixel'] for d in atlas})==33
    lookup={d['system']:d for d in atlas}
    for d in atlas:
        years,values=raw_reference(d)
        assert years[0]==1470 and years[-1]==2000 and np.all(np.diff(years)>0)
        for key,_,_ in KEYS:check_metric(years,values[key],d['metrics'][key])
    for c in cases:
        with np.load(ROOT/c['state_source'],allow_pickle=False) as z:
            years=z['years'].copy().astype(float);raw=z['history'].copy()
        for key,col,scale in KEYS:check_metric(years,raw[:,col]/scale,c['metrics'][key])
        assert c['parent_state_sha256'] is not None
    for r in rels:
        subset=[c for c in cases if c['system']==r['system']]
        assert len(subset)==r['case_count']
        for key,_,_ in KEYS:
            for f in [.005,.01,.02]:
                sets=[next(s['near']['years'] for s in c['metrics'][key]['sensitivity'] if s['start']==1500 and s['step']==10 and s['fraction']==f) for c in subset]
                expected=max([distance(a,b) for a,b in itertools.combinations(sets,2)]+[0.])
                assert expected==r[key]['near_set_pairwise_distances'][str(f)]
                COUNTS['pairwise_set_range_checks']+=1
        for field,key,measure in [('length_km_peak_loss','length_km','loss_peak_to_2000_pct'),('volume_km3_peak_loss','volume_km3','loss_peak_to_2000_pct'),('volume_loss_1900','volume_km3','loss_1900_2000_pct')]:
            vals=[c['metrics'][key][measure] for c in subset]
            close(r['amplitude_ranges_pp'][field],max(vals)-min(vals))
        assert independent_label(r,lookup[r['system']],30,2,.01)==r['label']
        for v in r['threshold_sensitivity']:
            assert independent_label(r,lookup[r['system']],v['timing_years'],v['amplitude_pp'],v['fraction'])==v['label']
            COUNTS['classification_sensitivity_checks']+=1
    assert summary['label_counts']==dict(Counter(d['reliability_label'] for d in atlas))
    for r in activity:
        subset=[d for d in atlas if r['group']=='all' or d['elevation_group']==r['group']]
        pixels=sorted({d['climate_pixel'] for d in subset});key=r['metric']
        expected=[];pixel_expected=[]
        for t in r['years']:
            indicators={d['system']:int(t in d['metrics'][key]['near']['years']) for d in subset}
            expected.append(sum(indicators.values())/len(subset))
            pixel_expected.append(sum(sum(indicators[d['system']] for d in subset if d['climate_pixel']==p)/sum(d['climate_pixel']==p for d in subset) for p in pixels)/len(pixels))
        close(r['near_equal_system'],expected);close(r['near_equal_pixel'],pixel_expected)
        close(np.array(r['growing_equal_system'])+r['shrinking_equal_system']+np.array(r['small_equal_system']),np.ones(50))
        assert np.all(np.array(r['conditional_lower'])<=np.array(expected)+1e-12)
        assert np.all(np.array(r['conditional_upper'])>=np.array(expected)-1e-12)
        lower=[];upper=[]
        for t in r['years']:
            lows=[];highs=[]
            for d in subset:
                aa=[c for c in cases if c['system']==d['system']]
                indicators=[int(t in c['metrics'][key]['near']['years']) for c in aa] if len(aa)>=2 else [int(t in d['metrics'][key]['near']['years'])]
                lows.append(min(indicators));highs.append(max(indicators))
            lower.append(sum(lows)/len(subset));upper.append(sum(highs)/len(subset))
        close(lower,r['conditional_lower']);close(upper,r['conditional_upper'])
        COUNTS['regional_groups']+=1
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:
        if '://' not in link and not link.startswith('#'):assert (OUT/link).exists(),link
    assert parser.rows==52
    assert sum(1 for p in (OUT/'glaciers').glob('*.png'))==51
    COUNTS['html_local_links']=len(parser.links)
    # Confirm the numerical claims displayed in the report against the delivered data.
    assert summary['repeated_exact_length_peak_systems']==36
    assert summary['disjoint_exact_length_peak_systems']==6
    assert summary['multi_near_length_interval_systems']==21
    ll=next(r for r in activity if r['metric']=='length_km' and r['group']=='all')
    vv=next(r for r in activity if r['metric']=='volume_km3' and r['group']=='all')
    assert ll['years'][np.argmax(ll['near_equal_system'])]==1880 and round(max(ll['near_equal_system'])*51)==24
    assert vv['years'][np.argmax(vv['near_equal_system'])]==1870 and round(max(vv['near_equal_system'])*51)==34
    loose=next(r for r in vv['near_threshold_sensitivity'] if r['fraction']==.02)
    assert vv['years'][np.argmax(loose['equal_system'])]==1770
    validation['independent_verification']=dict(status='pass',checked_utc=datetime.now(timezone.utc).isoformat(),counts=dict(COUNTS),
        notes='Independent script reread original geometry time series, rebuilt masks and time-set distances, checked weighting, links and report claims. Numerical agreement is not independent observational validation.')
    validation['status']='numerical-pass-visual-review-pending'
    review_path=OUT/'visual_review.json'
    if review_path.exists():
        review=read(review_path)
        assert review['status']=='pass'
        for name,expected in review['figure_hashes'].items():assert digest(OUT/name)==expected,name
        validation['visual_review']=review
        validation['status']='pass'
    authority_path=OUT/'authority_review.json'
    if authority_path.exists():
        authority=read(authority_path)
        assert authority['final_facts_sha256']==digest(ROOT/'LIA_PROJECT_FACTS.md')
        validation['authority_review']=authority
    write(OUT/'validation.json',validation)
    write(OUT/'output_manifest.json',{p.relative_to(OUT).as_posix():digest(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(validation['independent_verification'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
