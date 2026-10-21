"""Independent numerical check of saved-output paired diagnostics (no model)."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
import hashlib, itertools, json
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
COUNTS=Counter();KEYS=['length_km','volume_km3']
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def close(a,b):assert np.allclose(a,b,atol=3e-9,rtol=3e-9),(a,b)


def segment_mean(t,x,start,end):
    area=0.
    for i in range(len(t)-1):
        l,r=max(t[i],start),min(t[i+1],end)
        if r<=l:continue
        slope=(x[i+1]-x[i])/(t[i+1]-t[i]);u=x[i]+slope*(l-t[i]);v=x[i]+slope*(r-t[i])
        area+=(u+v)*(r-l)/2
    return area/(end-start)


def independent_events(t,g,threshold,minimum):
    # Subdivide at every +/- crossing, classify interval midpoints, then merge same-sign runs.
    knots={1700.,1900.}|{float(y) for y in t if 1700<y<1900}
    for i in range(len(t)-1):
        if g[i+1]==g[i]:continue
        for target in [-threshold,threshold]:
            f=(target-g[i])/(g[i+1]-g[i])
            x=t[i]+f*(t[i+1]-t[i])
            if 0<f<1 and 1700<x<1900:knots.add(float(x))
    knots=sorted(knots);runs=[]
    for l,r in zip(knots[:-1],knots[1:]):
        v=np.interp((l+r)/2,t,g)
        sign=1 if v>=threshold-1e-12 else (-1 if v<=-threshold+1e-12 else 0)
        if not sign:continue
        if runs and runs[-1][2]==sign and abs(runs[-1][1]-l)<1e-8:runs[-1][1]=r
        else:runs.append([l,r,sign])
    return [r for r in runs if r[1]-r[0]>=minimum-1e-8]


def check_events(actual,t,g,threshold,minimum):
    expected=independent_events(t,g,threshold,minimum);assert len(actual)==len(expected),(actual,expected)
    step=t[1]-t[0]
    for a,(left,right,sign) in zip(actual,expected):
        close([a['start_estimate'],a['end_estimate'],a['duration_years']],[left,right,right-left]);assert a['sign']==sign
        assert a['left_censored']==(abs(left-1700)<1e-8) and a['right_censored']==(abs(right-1900)<1e-8)
        for value,bracket in [(left,a['start_bracket']),(right,a['end_bracket'])]:
            assert bracket[0]-1e-8<=value<=bracket[1]+1e-8 and bracket[1]-bracket[0]<=step+1e-8
            assert all(abs((b-1500)/step-round((b-1500)/step))<1e-8 for b in bracket)
    COUNTS['event_lists']+=1


def independent_gap(ca,cb,key,step=10,baseline=(1600,1700)):
    t=np.array(list(range(1500,2001,step)),float)
    a=np.interp(t,ca['native_years'],ca['native_'+key]);b=np.interp(t,cb['native_years'],cb['native_'+key])
    ra=segment_mean(t,a,*baseline);rb=segment_mean(t,b,*baseline)
    ac=(a-ra)*100/max(a);bc=(b-rb)*100/max(b)
    return t,bc-ac,dict(normalized_a=a*100/max(a),normalized_b=b*100/max(b),centered_a=ac,centered_b=bc,
                         normalization_a=max(a),normalization_b=max(b),reference_a=ra,reference_b=rb)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set()
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag=='img' and 'src' in a:self.links.append(a['src'])


def main():
    methods=read(OUT/'methods.json');validation=read(OUT/'validation.json')
    for p,h in read(OUT/'source_manifest.json').items():
        assert not any(s in p.lower() for s in ['20cr','jionglacuo','s7t4zc34sr','temperature reconstruction.xlsx']) and not p.endswith('inputs.npz')
        assert sha(ROOT/p)==h,p
    authority=methods['authority']
    if (OUT/'authority_reconciliation_v1.json').exists():
        rec=read(OUT/'authority_reconciliation_v1.json');assert rec['initial_authority_hashes']==authority
        assert hashlib.sha256((ROOT/'LIA_PROJECT_FACTS.md').read_bytes()[:rec['facts_previous_bytes']]).hexdigest().upper()==rec['facts_previous_sha256']
        assert sha(ROOT/rec['current_route']['path'])==rec['current_route']['sha256']
        authority=rec['current_authority_hashes']
    for p,h in authority.items():assert sha(ROOT/p)==h,p
    assert validation['authority_hashes']==authority
    for p,h in read(OUT/'output_manifest.json').items():
        if p in [Path(__file__).name,'visual_review.json']:continue
        assert sha(OUT/p)==h,p
    # Fixtures exercise constant/short/disjoint/two-sign and censored detection.
    t=np.arange(1500,2001,10,dtype=float)
    assert independent_events(t,np.zeros(len(t)),2,30)==[]
    assert independent_events(t,np.full(len(t),3.),2,30)==[[1700.,1900.,1]]
    assert independent_events(t,np.full(len(t),-3.),2,30)==[[1700.,1900.,-1]]
    g=(t-1800)/10;events=independent_events(t,g,2,30)
    close(events,[[1700,1780,-1],[1820,1900,1]])
    impulse=np.zeros(len(t));impulse[t==1800]=5;assert independent_events(t,impulse,2,30)==[]
    wave=np.where(((t>=1720)&(t<=1770))|((t>=1820)&(t<=1870)),3.,0.);assert len(independent_events(t,wave,2,30))==2
    close(segment_mean(np.array([0,10]),np.array([0,10]),2,8),5)
    COUNTS['fixtures']=7
    rows=read(ROOT/'analysis/lia_51_stage_controls_v1/features_51.json');byid={r['system']:r for r in rows}
    curves={c['system']:c for c in read(ROOT/'analysis/lia_51_stage_reliability_v1/reference_curves.json')}
    pairs=read(OUT/'pairs.json');pairbyid={p['pair_id']:p for p in pairs};groups=defaultdict(list)
    for r in rows:groups[r['climate_pixel']].append(r)
    expected=set()
    for pixel,rs in groups.items():
        for a,b in itertools.combinations(sorted(rs,key=lambda r:(r['terminus_elevation_m'],r['system'])),2):expected.add((pixel,a['system'],b['system']))
    assert {(p['pixel'],p['a'],p['b']) for p in pairs}==expected and len(pairs)==27
    selection=read(OUT/'pair_selection.json');assert selection['methods_sha256']==sha(OUT/'methods.json')
    paramkeys=['base_temp_bias_c','q_mean_1500_2000_c','log_melt','log_glen_a'];scales={}
    for key in paramkeys:
        values=np.array([np.log10(r['melt_f']) if key=='log_melt' else r[key] for r in rows]);q25,q75=np.quantile(values,[.25,.75])
        scales[key]=(q75-q25) if q75-q25>1e-10 else values.std()
        close(scales[key],selection['parameter_distance_scales'][key])
    for p in pairs:
        a,b=byid[p['a']],byid[p['b']]
        assert a['climate_pixel']==b['climate_pixel']==p['pixel'];close(a['prcp_fac'],b['prcp_fac'])
        same=a['route_family']==b['route_family'];assert p['same_route']==same
        assert p['same_route_constant']==(same and not a['timevarying'] and not b['timevarying'])
        for k,v in p['differences'].items():close(v,b[k]-a[k])
        for field,key in [('area_ratio','static_area_km2'),('melt_ratio','melt_f'),('glen_ratio','glen_a')]:close(p[field],max(a[key]/b[key],b[key]/a[key]))
        close(p['azimuth_separation_deg'],min(abs(a['azimuth_deg']-b['azimuth_deg']),360-abs(a['azimuth_deg']-b['azimuth_deg'])))
        scores=[abs(np.log10(b['melt_f']/a['melt_f']))/scales[k] if k=='log_melt' else abs(b[k]-a[k])/scales[k] for k in paramkeys if scales[k]>1e-10]
        close(p['parameter_distance'],np.mean(scores))
        for key in KEYS:
            d=p['metrics'][key];t,g,extra=independent_gap(curves[p['a']],curves[p['b']],key)
            close(t,d['years']);close(g,d['gap_pp'])
            for k,x in extra.items():close(d[k],x)
            close(segment_mean(t,g,1600,1700),0)
            late=segment_mean(t,g,1800,1900);close(d['late_contrast_pp'],late)
            close(d['modern_mean_gap_pp'],segment_mean(t,g,1900,2000))
            # Algebraically bridge to the previous independently verified whole-sample report.
            oldkey=key.split('_')[0]+'_contrast_pp';close(late,b[oldkey]-a[oldkey])
            for per in d['periods']:
                left,right=per['start'],per['end'];close(per['mean_gap_pp'],segment_mean(t,g,left,right))
                for side in ['a','b']:
                    val=np.interp([left,right],t,extra['normalized_'+side]);close(per[side+'_net_change_pp'],val[1]-val[0])
                COUNTS['periods']+=1
            assert d['largest_premodern_absolute_bin']==max(d['periods'][:8],key=lambda x:abs(x['mean_gap_pp']))
            check_events(d['episodes'],t,g,2,30);COUNTS['pair_metric_profiles']+=1
    restricted=[p for p in pairs if p['same_route_constant']];assert len(restricted)==11
    h=min([p for p in restricted if abs(p['differences']['gradient_deg'])<=2 and p['area_ratio']<=2],key=lambda p:(-p['differences']['terminus_elevation_m'],p['pair_id']))
    s=min([p for p in restricted if p['differences']['terminus_elevation_m']<=600 and p['area_ratio']<=2],key=lambda p:(-abs(p['differences']['gradient_deg']),p['pair_id']))
    third=min([p for p in restricted if p['pair_id'] not in [h['pair_id'],s['pair_id']] and (p['differences']['terminus_elevation_m']>=400 or abs(p['differences']['gradient_deg'])>=3)],key=lambda p:(p['parameter_distance'],p['pair_id']))
    assert [r['pair_id'] for r in selection['spotlights']]==[h['pair_id'],s['pair_id'],third['pair_id']]
    variants=read(OUT/'definition_sensitivity.json');assert len(variants)==1944
    for s in variants:
        p=pairbyid[s['pair_id']];t,g,_=independent_gap(curves[p['a']],curves[p['b']],s['metric'],s['step'],s['baseline'])
        close(s['late_contrast_pp'],segment_mean(t,g,1800,1900));close(s['modern_mean_gap_pp'],segment_mean(t,g,1900,2000))
        check_events(s['episodes'],t,g,s['threshold_pp'],s['minimum_duration_years']);COUNTS['definition_profiles']+=1
    for p in pairs:
        for key in KEYS:
            sub=[v for v in variants if v['pair_id']==p['pair_id'] and v['metric']==key];assert len(sub)==36
            assert len({(tuple(v['baseline']),v['step'],v['threshold_pp'],v['minimum_duration_years']) for v in sub})==36
            detected=[v for v in sub if v['episodes']];summary=p['definition_sensitivity'][key]
            assert summary['detected']==len(detected) and summary['definitions']==36
            span=[min(v['episodes'][0]['start_bracket'][0] for v in detected),max(v['episodes'][0]['start_bracket'][1] for v in detected)] if detected else None
            assert summary['first_start_bracket_span']==span
            assert summary['first_signs']==sorted({v['episodes'][0]['sign'] for v in detected})
            assert summary['left_censored_definitions']==sum(v['episodes'][0]['left_censored'] for v in detected)
    replacements=read(OUT/'alternative_replacements.json');cache={}
    registered={c['case_id']:c for c in read(ROOT/'analysis/lia_51_stage_reliability_v1/case_diagnostics.json')}
    for alt in replacements:
        c=registered[alt['case_id']];assert not c['selected'] and c['system']==alt['replaced_system'] and c['parameters']==alt['parameters']
        assert c['state_source']==alt['state_source'] and c['parent_state_sha256']==alt['parent_state_sha256']
        assert all(c['parameters'][k]==1 for k in ['melt','prcp','glen'])
        if alt['case_id'] not in cache:
            with np.load(ROOT/alt['state_source'],allow_pickle=False) as z:
                cache[alt['case_id']]=dict(native_years=z['years'].copy(),native_length_km=z['history'][:,0].copy()/1000,native_volume_km3=z['history'][:,2].copy()/1e9)
        p=pairbyid[alt['pair_id']];ca=cache[alt['case_id']] if p['a']==alt['replaced_system'] else curves[p['a']];cb=cache[alt['case_id']] if p['b']==alt['replaced_system'] else curves[p['b']]
        t,g,_=independent_gap(ca,cb,alt['metric']);close(g,alt['gap_pp']);close(alt['late_contrast_pp'],segment_mean(t,g,1800,1900))
        check_events(alt['episodes'],t,g,2,30);COUNTS['replacement_profiles']+=1
    for p in pairs:
        for key in KEYS:
            alts=[a for a in replacements if a['pair_id']==p['pair_id'] and a['metric']==key];cond=p['conditional'][key]
            expected={(c['system'],c['case_id']) for c in registered.values() if c['system'] in [p['a'],p['b']] and not c['selected']}
            assert {(a['replaced_system'],a['case_id']) for a in alts}==expected
            paths=np.array([p['metrics'][key]['gap_pp']]+[a['gap_pp'] for a in alts]);late=[p['metrics'][key]['late_contrast_pp']]+[a['late_contrast_pp'] for a in alts]
            assert cond['nonselected_replacements']==len(alts);close(cond['gap_min_pp'],paths.min(axis=0));close(cond['gap_max_pp'],paths.max(axis=0))
            close([cond['late_min_pp'],cond['late_max_pp']],[min(late),max(late)])
            assert cond['replacement_detections']==sum(bool(a['episodes']) for a in alts)
            assert cond['reference_detection']==bool(p['metrics'][key]['episodes'])
            assert set(cond['tested_endpoints'])=={a['replaced_system'] for a in alts}
    for agg in read(OUT/'aggregates.json'):
        sub=pairs if agg['cohort']=='all_pairs' else restricted;key=agg['metric'];pg=defaultdict(list);pl=defaultdict(list)
        for p in sub:
            sign=1 if agg['orientation']=='height' else np.sign(p['differences']['gradient_deg'])
            if sign==0:continue
            pg[p['pixel']].append(np.array(p['metrics'][key]['gap_pp'])*sign);pl[p['pixel']].append(p['metrics'][key]['late_contrast_pp']*sign)
        matrix=np.array([np.median(pg[pixel],axis=0) for pixel in sorted(pg)]);late=np.array([np.median(pl[pixel]) for pixel in sorted(pl)])
        close(agg['median_gap_pp'],np.median(matrix,axis=0));close(agg['q25_gap_pp'],np.quantile(matrix,.25,axis=0));close(agg['q75_gap_pp'],np.quantile(matrix,.75,axis=0))
        close(agg['median_late_contrast_pp'],np.median(late));assert agg['pixel_count']==len(pg) and agg['pair_count']==len(sub)
        assert [agg['positive_pixels'],agg['negative_pixels'],agg['zero_pixels']]==[sum(late>1e-8),sum(late<-1e-8),sum(abs(late)<=1e-8)]
        loo=[np.median(np.delete(late,i)) for i in range(len(late))];close(agg['leave_one_pixel_median_range_pp'],[min(loo),max(loo)])
        for saved,pixel in zip(agg['pixel_results'],sorted(pg)):
            assert saved['pixel']==pixel and saved['pair_count']==len(pg[pixel]);close(saved['median_gap_pp'],np.median(pg[pixel],axis=0));close(saved['median_late_contrast_pp'],np.median(pl[pixel]))
        COUNTS['aggregate_profiles']+=1
    coverage=read(OUT/'coverage_51.json');assert len(coverage)==51 and sum(c['has_peer'] for c in coverage)==29
    for c in coverage:assert set(c['peer_ids'])=={r['system'] for r in groups[c['pixel']] if r['system']!=c['system']} and c['has_peer']==bool(c['peer_ids'])
    summary=read(OUT/'summary.json');assert len(summary['without_peer_ids'])==22 and summary['pairs']==27 and summary['same_route_constant_pixels']==7
    assert summary['paired_systems_with_alternatives']==len({a['replaced_system'] for a in replacements})==7
    assert summary['saved_cases_used']==sum(c['system'] in {r['system'] for r in coverage if r['has_peer']} for c in registered.values())==73
    parser=Links();parser.feed((OUT/'index.html').read_text(encoding='utf-8'))
    for link in parser.links:
        if link.startswith('#'):assert link[1:] in parser.ids,link
        elif '://' not in link:assert (OUT/link).exists(),link
    COUNTS['html_links']=len(parser.links);assert len(list(OUT.glob('*.png')))==17
    validation['independent_verification']=dict(status='pass',checked_utc=datetime.now(timezone.utc).isoformat(),counts=dict(COUNTS),method='Separate clipped-segment integration and threshold-knot midpoint classification; full pair enumeration, metadata-only ranking, all36 definitions, all saved one-endpoint replacements and equal-pixel aggregation. Not independent observational validation.')
    validation['status']='numerical-pass-visual-review-pending'
    if (OUT/'visual_review.json').exists():
        review=read(OUT/'visual_review.json');assert review['status']=='pass'
        for p,h in review['figure_hashes'].items():assert sha(OUT/p)==h,p
        validation['visual_review']=review;validation['status']='pass'
    save('validation.json',validation)
    save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(validation['independent_verification'],ensure_ascii=False),flush=True)


if __name__=='__main__':main()
