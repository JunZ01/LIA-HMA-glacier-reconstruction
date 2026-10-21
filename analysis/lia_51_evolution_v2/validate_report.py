"""Independent arithmetic, source-integrity, and report-link checks."""
from pathlib import Path
from html.parser import HTMLParser
import importlib.util
import json
import hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


class CheckHTML(HTMLParser):
    def __init__(self):
        super().__init__();self.links=[];self.in_members=False;self.rows=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        for key in ['src','href']:
            if key in a:self.links.append(a[key])
        if tag=='table' and a.get('id')=='members':self.in_members=True
        if tag=='tr' and self.in_members:self.rows+=1
    def handle_endtag(self,tag):
        if tag=='table':self.in_members=False


spec=importlib.util.spec_from_file_location('analysis_builder',OUT/'build_report.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for w in [5,10,20]:
    assert module.decline(np.arange(501,dtype=float)+1,w)['start_year'] is None
    a=module.decline(600-np.arange(501,dtype=float),w)
    assert a['start_year']==1500 and a['left_censored']
    assert module.decline(np.ones(501),w)['start_year'] is None
testmeans=np.r_[np.arange(41),np.arange(39,30,-1)]
assert len(testmeans)==50
testseries=np.r_[np.repeat(testmeans,10),29]+100
assert module.decline(testseries,10)['start_year']==1900
p=module.native_peak(np.array([1500,1510,1520,1530,1540]),np.array([1.,3.,3.,2.,3.]))
assert p['tied_native_segments']==[[1510,1520],[1540,1540]]

curves={c['system']:c for c in read('curves.json')}
members=read('members.json');diagnostics=read('diagnostics.json');summary=read('summary.json')
assert len(curves)==len(members)==len(diagnostics)==51
assert sum(m['status']=='retained' for m in members)==32
assert sum(m['status']=='candidate' for m in members)==19
assert [sum(d['group']==g for d in diagnostics) for g in ['low','middle','high']]==[21,14,16]
for d in diagnostics:
    c=curves[d['system']]
    assert c['year']==list(range(1500,2001))
    for metric,key in [('length','length_km'),('volume','volume_km3')]:
        a=np.array(c[key]);y=np.array(c['native_years']);raw=np.array(c['native_'+key])
        assert np.isfinite(a).all() and (a>0).all()
        assert np.allclose(c[metric+'_relative_2000_pct'],100*(a/a[-1]-1),atol=1e-12,rtol=0)
        assert c[metric+'_relative_2000_pct'][-1]==0
        peak=raw[(y>=1500)&(y<=2000)].max()
        assert d[metric]['peak']['value']==peak
        assert abs(d[metric]['loss_peak_to_2000_pct']-100*(peak-a[-1])/peak)<1e-10
        for w in [5,10,20]:
            v=d[metric]['sensitivity'][str(w)]
            assert np.allclose(v['means'],a[:500].reshape(-1,w).mean(axis=1))
            if v['start_year'] is not None:
                idx=(v['start_year']-1500)//w
                diffs=np.diff(v['means'][idx:])
                tol=1e-10*max(1,max(v['means']))
                assert np.all(diffs<=tol)
                assert np.sum(diffs < -tol)>=int(np.ceil(30/w))
for s in summary:
    subset=[d for d in diagnostics if d['group']==s['group'] and (s['cohort']=='all' or d['status']==s['cohort'])]
    assert s['n']==len(subset)
    vals=[d[s['metric']]['loss_peak_to_2000_pct'] for d in subset]
    assert s['loss_peak_to_2000_pct_median']==float(np.median(vals))
manifest=read('source_manifest.json')
for rel,expected in manifest.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest().upper()==expected.upper(),rel
parser=CheckHTML();source=(OUT/'index.html').read_text(encoding='utf-8');parser.feed(source)
assert parser.rows==52 and '\ufffd' not in source
for link in parser.links:
    if not link.startswith(('https:','http:','#')):assert (OUT/link).exists(),link
result=read('validation.json')
result['independent_checks']={'unit_fixtures':'pass','all_51_curve_arithmetic':'pass','group_statistics':'pass',
                              'source_hashes':'pass','local_report_links':'pass','html_member_rows':51}
(OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
