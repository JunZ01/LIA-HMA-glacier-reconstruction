"""Independent synthesis arithmetic with native saved alternatives; no builder import."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
from bisect import bisect_left
from html.parser import HTMLParser
import hashlib,json,math,itertools
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];ATLAS=ROOT/'analysis/lia_51_stage_reliability_v1'
KEYS=['length_km','area_km2','volume_km3'];COUNTS=Counter()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def close(a,b):assert np.allclose(a,b,rtol=1e-10,atol=1e-9),(a,b)
def save(name,x):(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def interp(t,x,target):
    i=bisect_left(t,target)
    if i<len(t) and t[i]==target:return float(x[i])
    assert 0<i<len(t)
    return float(x[i-1]+(x[i]-x[i-1])*(target-t[i-1])/(t[i]-t[i-1]))
def quantiles(v):
    v=sorted(v);out=[]
    for f in [0,.25,.5,.75,1]:
        at=(len(v)-1)*f;i=math.floor(at);j=math.ceil(at);out.append(v[i]+(v[j]-v[i])*(at-i))
    return out
def pp(c,key,a,b):
    t=c['native_years'];x=c['native_'+key];s=interp(t,x,a);e=interp(t,x,b)
    return s,e,e-s,100*(e-s)/s
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        d=dict(attrs)
        if tag=='img' and 'src' in d:self.links.append(d['src'])
        if tag=='a' and 'href' in d:self.links.append(d['href'])

def main():
    methods=read(OUT/'methods.json');authority=methods['authority']
    for reconciliation in sorted(OUT.glob('authority_reconciliation_v*.json')):
        rec=read(reconciliation);assert authority['LIA_PROJECT_FACTS.md']==rec['facts_previous_sha256']
        assert hashlib.sha256((ROOT/'LIA_PROJECT_FACTS.md').read_bytes()[:rec['facts_previous_bytes']]).hexdigest().upper()==rec['facts_previous_sha256']
        authority=rec['current_authority_hashes']
    for p,h in authority.items():assert sha(ROOT/p)==h,p
    source=read(OUT/'source_manifest.json')
    for p,h in source.items():
        assert 'inputs.npz' not in p and not any(s in p.lower() for s in ['20cr','jionglacuo','s7t4zc34sr','temperature reconstruction.xlsx','s41612-025-01254-y'])
        assert sha(ROOT/p)==h,p
    for p,h in read(OUT/'output_manifest.json').items():
        if p not in ['verify_report.py','visual_review.json']:assert sha(OUT/p)==h,p
    curves={r['system']:r for r in read(ATLAS/'reference_curves.json')};atlas={r['system']:r for r in read(ATLAS/'atlas.json')}
    features={r['system']:r for r in read(ROOT/'analysis/lia_51_stage_controls_v1/features_51.json')}
    individual=read(OUT/'individual_changes.json');byid={r['system']:r for r in individual};cases=defaultdict(list)
    # Reopen only original saved geometry history fields for113 cases, never raw forcing.
    for c in read(ATLAS/'case_diagnostics.json'):
        path=ROOT/c['state_source'];assert c['state_source'] in source
        with np.load(path,allow_pickle=False) as z:
            t=z['years'].copy().astype(float);h=z['history'].copy().astype(float)
        assert t.tolist()==list(range(1470,2001)) and h.shape[1]>=3
        r=dict(native_years=t.tolist())
        for k,col,scale in zip(KEYS,[0,1,2],[1000,1e6,1e9]):
            r['native_'+k]=(h[:,col]/scale).tolist()
            close([interp(t,r['native_'+k],y) for y in range(1500,2001,10)],c['metrics'][k]['interval']['values'])
            if c['selected']:close(r['native_'+k],curves[c['system']]['native_'+k])
        cases[c['system']].append(r);COUNTS['native_saved_cases']+=1
    # Fixtures explicitly distinguish paired-case changes from independent endpoint bounds.
    choices=[[(100,90),(200,190)],[(10,8),(10,7)]]
    sums=[math.fsum(e-s for s,e in combo) for combo in itertools.product(*choices)]
    close([min(sums),max(sums)],[-13,-12]);assert min(e for s,e in choices[0])-max(s for s,e in choices[0])==-110
    close(interp([0,10],[4,8],5),6);close(quantiles([1,2,3,4]),[1,1.75,2.5,3.25,4]);COUNTS['fixtures']=3
    assert set(byid)==set(curves) and len(byid)==51
    for r in individual:
        sid=r['system'];c=curves[sid];options=cases.get(sid) or [c]
        assert r['case_count']==len(cases.get(sid,[])) and r['reliability_label']==atlas[sid]['reliability_label']
        for key in KEYS:
            for p in r['metrics'][key]['periods']:
                a,b,d,pct=pp(c,key,p['start'],p['end']);close([p['value_start'],p['value_end'],p['delta'],p['change_pct']],[a,b,d,pct]);close(p['net_rate_per_year'],d/(p['end']-p['start']))
                alternatives=[pp(v,key,p['start'],p['end']) for v in options]
                close(p['saved_case_delta_range'],[min(v[2] for v in alternatives),max(v[2] for v in alternatives)])
                close(p['saved_case_pct_range'],[min(v[3] for v in alternatives),max(v[3] for v in alternatives)])
                assert p['conditional_range_available']==(len(options)>1);COUNTS['individual_period_metrics']+=1
            t=c['native_years'];x=c['native_'+key];peak=max(v for y,v in zip(t,x) if 1500<=y<=2000)
            m=r['metrics'][key];close(m['native_peak']['value'],peak);close(m['peak_to_2000_loss'],peak-x[-1]);close(m['peak_to_2000_loss_pct'],100*(peak-x[-1])/peak)
            assert m['native_peak']==atlas[sid]['metrics'][key]['native_peak'];COUNTS['native_peak_checks']+=1
    agg=read(OUT/'cohort_aggregates.json')
    for name,g in agg.items():
        ids=g['systems'];assert len(ids)==g['n']
        if name=='all51':assert set(ids)==set(curves)
        if name=='constant41':assert set(ids)=={s for s,f in features.items() if not f['timevarying']}
        if name=='synthetic10':assert set(ids)=={s for s,f in features.items() if f['timevarying']}
        if name=='original32':assert set(ids)=={s for s,f in features.items() if f['status']=='retained'}
        if name=='original19':assert set(ids)=={s for s,f in features.items() if f['status']=='candidate'}
        for key,m in g['metrics'].items():
            for p in m['periods']:
                start,end=p['start'],p['end'];vv=[pp(curves[s],key,start,end) for s in ids]
                close(p['individual_change_pct_quantiles'],quantiles([v[3] for v in vv]))
                assert p['shrinking_count']==sum(v[2]<-1e-10 for v in vv) and p['growing_count']==sum(v[2]>1e-10 for v in vv) and p['zero_count']==sum(abs(v[2])<=1e-10 for v in vv)
                if key!='length_km':
                    total=p['total'];a=math.fsum(v[0] for v in vv);b=math.fsum(v[1] for v in vv);close([total['value_start'],total['value_end'],total['delta'],total['change_pct']],[a,b,b-a,100*(b-a)/a])
                    lo=[];hi=[]
                    for sid in ids:
                        changes=[pp(c,key,start,end)[2] for c in (cases.get(sid) or [curves[sid]])];lo.append(min(changes));hi.append(max(changes))
                    close(total['conditional_delta_range'],[math.fsum(lo),math.fsum(hi)]);assert total['conditional_delta_range'][0]-1e-9<=total['delta']<=total['conditional_delta_range'][1]+1e-9
                    COUNTS['aggregate_paired_change_bounds']+=1
                COUNTS['cohort_period_metrics']+=1
            if key=='length_km':assert 'totals' not in m;continue
            totals=[]
            for y,val,lo,hi in zip(m['years'],m['totals'],m['pointwise_conditional_lower'],m['pointwise_conditional_upper']):
                total=math.fsum(interp(curves[s]['native_years'],curves[s]['native_'+key],y) for s in ids);totals.append(total);close(val,total)
                choices=[[interp(c['native_years'],c['native_'+key],y) for c in (cases.get(s) or [curves[s]])] for s in ids]
                close([lo,hi],[math.fsum(min(v) for v in choices),math.fsum(max(v) for v in choices)]);COUNTS['pointwise_total_bounds']+=1
            peak=max(totals);close(m['common_grid_maximum'],peak);assert m['common_grid_maximum_years']==[y for y,v in zip(m['years'],totals) if math.isclose(v,peak,rel_tol=1e-10,abs_tol=1e-12)]
            assert m['near_1pct_years']==[y for y,v in zip(m['years'],totals) if v>=.99*peak]
    times=read(OUT/'timing_summary.json');assert len(times)==18
    for r in times:
        ids=agg[r['group']]['systems'];pixels=sorted({features[s]['climate_pixel'] for s in ids});t=r['years'];key=r['metric']
        values={s:[interp(curves[s]['native_years'],curves[s]['native_'+key],y) for y in t] for s in ids}
        def weights(ind):return math.fsum(math.fsum(ind[s] for s in ids if features[s]['climate_pixel']==p)/sum(features[s]['climate_pixel']==p for s in ids) for p in pixels)/len(pixels)
        for v in r['near_variants']:
            eq=[];wp=[]
            for i,y in enumerate(t):
                mask={s:int(values[s][i]>=max(values[s])*(1-v['fraction'])-1e-12*max(1,max(values[s]))) for s in ids}
                eq.append(sum(mask.values())/len(ids));wp.append(weights(mask))
            close(v['equal_system'],eq);close(v['equal_pixel'],wp);close(v['max_fraction'],max(eq));close(v['max_pixel_fraction'],max(wp))
            assert v['max_fraction_years']==[y for y,x in zip(t,eq) if np.isclose(x,max(eq))] and v['max_pixel_years']==[y for y,x in zip(t,wp) if np.isclose(x,max(wp))];COUNTS['near_maximum_series']+=1
        for i in range(len(t)-1):
            mask={s:int(values[s][i+1]-values[s][i]<-.001*max(values[s])) for s in ids};close(r['shrinking_equal_system'][i],sum(mask.values())/len(ids));close(r['shrinking_equal_pixel'][i],weights(mask));COUNTS['interval_states']+=1
    summary=read(OUT/'summary.json');assert summary['multiple_case_systems']==sum(len(v)>1 for v in cases.values())==14 and summary['untested_systems']==37
    assert summary['reliability_label_counts']==dict(Counter(r['reliability_label'] for r in individual))
    for key,r in summary['conditional_1900_2000_direction'].items():
        tested=[s for s in cases if len(cases[s])>1];assert r['all_saved_cases_shrink']==sum(all(pp(c,key,1900,2000)[2]<-1e-10 for c in cases[s]) for s in tested)
        assert r['selected_shrink']==sum(pp(c,key,1900,2000)[2]<-1e-10 for c in curves.values())
    losses=sorted([(s,-pp(c,'volume_km3',1900,2000)[2]) for s,c in curves.items()],key=lambda r:r[1],reverse=True)
    top=summary['top5_volume_loss_contribution'];assert [r['system'] for r in top['rows']]==[s for s,v in losses[:5]];close(top['share_of_net_loss'],math.fsum(v for s,v in losses[:5])/math.fsum(v for s,v in losses))
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf8'))
    for link in parser.links:
        if '://' not in link and not link.startswith('#'):assert (OUT/link).exists(),link
    COUNTS['local_html_links']=len(parser.links);assert len(list(OUT.glob('*.png')))==3
    validation=read(OUT/'validation.json');validation['status']='numerical-pass-visual-review-pending'
    validation['independent_verification']=dict(status='pass',checked_utc=datetime.now(timezone.utc).isoformat(),counts=dict(COUNTS),scope='Independent scalar interpolation, native113-case histories, sorted quantiles, math.fsum aggregation, exact paired-case extrema and Boolean timing/weights; no builder import. Not independent observational validation.')
    if (OUT/'visual_review.json').exists():
        review=read(OUT/'visual_review.json');assert review['status']=='pass'
        for p,h in review['figure_hashes'].items():assert sha(OUT/p)==h,p
        validation['visual_review']=review;validation['status']='pass'
    for p,h in source.items():assert sha(ROOT/p)==h,p
    save('validation.json',validation);save('output_manifest.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='output_manifest.json'})
    print(json.dumps(validation['independent_verification'],ensure_ascii=False))

if __name__=='__main__':main()
