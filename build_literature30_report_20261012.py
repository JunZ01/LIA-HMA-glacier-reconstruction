from pathlib import Path
import json, html, hashlib, collections, re

ROOT = Path('E:/3spyderProject/1_LIA_himalaya_oggm')
OUT = Path('H:/ToYangxin/result_yx/literature30_research_20261012_v1')

def esc(x):
    if isinstance(x, (list, dict)):
        x = json.dumps(x, ensure_ascii=False)
    return html.escape(str(x))

def main():
    papers=[]
    reviews={}
    for group in 'ABC':
        src=ROOT/f'literature30_20261012_{group}.json'
        data=json.loads(src.read_text(encoding='utf-8-sig'))
        assert len(data['papers'])==10, (group,len(data['papers']))
        reviews[group]=data.get('synthesis')
        for p in data['papers']:
            p['group']=group
            papers.append(p)
    assert len(papers)==30
    assert len({p['id'] for p in papers})==30
    assert len({p['doi'].lower().removeprefix('https://doi.org/') for p in papers})==30
    required=['title','authors','year','journal','doi','url','access_level','sections_read','question','design_and_independence','key_finding','not_demonstrated','implication_for_654','selection_reason']
    for p in papers:
        assert all(p.get(k) for k in required),(p['id'],[k for k in required if not p.get(k)])
    decision=json.loads((OUT/'research_decision.json').read_text(encoding='utf-8'))
    evidence={'review_date':'2026-10-12','core_count':30,'papers':papers,'group_reviews':reviews,'decision':decision}
    (OUT/'evidence30.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    for group in 'ABC':
        (OUT/f'group_{group}_review.json').write_bytes((ROOT/f'literature30_20261012_{group}.json').read_bytes())
    bib=[]
    for p in papers:
        authors=p['authors']
        if isinstance(authors,list):authors=' and '.join(authors)
        def clean(v):return str(v).replace('{','').replace('}','')
        bib.append('@article{'+p['id']+',\n'+',\n'.join('  '+k+' = {'+clean(v)+'}' for k,v in [('title',p['title']),('author',authors),('year',p['year']),('journal',p['journal']),('doi',p['doi'].removeprefix('https://doi.org/')),('url',p['url'])])+'\n}')
    (OUT/'references30.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
    groups={'A':'物理理论、几何反馈与响应时间','B':'高亚洲独立观测与区域机制','C':'小冰期重建、厚度反演与可识别性'}
    fields={'question':'科学问题','design_and_independence':'设计与证据独立性','key_finding':'关键发现','not_demonstrated':'没有证明什么','implication_for_654':'对本项目的意义','selection_reason':'入选理由','sections_read':'实际核读范围'}
    access_labels={'primary_full_text_author_repository_pdf':'作者机构库正文PDF；核读下列关键章节','primary_full_text_author_uploaded_article':'作者公开的论文正文；核读下列关键章节','primary_full_text_publisher':'出版社正文；核读下列关键章节','primary_full_text_publisher_pdf':'出版社正文PDF；核读下列关键章节','partial_primary_abstract_and_metadata_only':'正文受限：仅出版社摘要与书目元数据','partial_primary_abstract_data_availability_and_author_correction':'正文受限：仅摘要、数据可用性及作者更正'}
    cards=[]
    for p in papers:
        auth='; '.join(p['authors']) if isinstance(p['authors'],list) else p['authors']
        parts=''.join(f'<div class="field"><b>{label}</b><p>{esc("；".join(p[key]) if isinstance(p[key],list) else p[key])}</p></div>' for key,label in fields.items())
        extra=''
        for key in ['fulltext_url','full_text_url','source_urls','correction_url','correction','access_gap','notes','version_note','metadata_note']:
            if key in p:extra+=f'<p class="small">{esc(key)}：{esc(p[key])}</p>'
        cards.append(f'<article class="paper" id="{esc(p["id"])}" data-group="{p["group"]}"><div class="eyebrow">{p["id"]} · {esc(p["journal"])} · {p["year"]}</div><h3><a href="{esc(p["url"])}">{esc(p["title"])}</a></h3><p class="authors">{esc(auth)}</p><p class="access">读取层级：{esc(access_labels.get(p["access_level"],p["access_level"]))}</p><p class="doi"><a href="https://doi.org/{esc(p["doi"].removeprefix("https://doi.org/"))}">DOI：{esc(p["doi"])}</a></p><details><summary>展开逐篇评议</summary>{parts}{extra}</details></article>')
    rows=''.join('<tr>'+''.join(f'<td>{esc(q[k])}</td>' for k in ['question','literature_status','unique_information','critical_test','decision'])+'</tr>' for q in decision['candidate_comparison'])
    conclusions=''.join(f'<li>{esc(x)}</li>' for x in decision['conclusions'])
    gates=''.join(f'<li>{esc(x)}</li>' for x in decision['next_evidence_tests'])
    limits=''.join(f'<li>{esc(x)}</li>' for x in decision['project_limits'])
    extra=''.join(f'<li><a href="{esc(x["url"])}">{esc(x["title"])}</a> — {esc(x["status"])}。{esc(x["implication"])}</li>' for x in decision.get('supplemental_screening',[]))
    read_levels=collections.Counter(p['access_level'] for p in papers)
    partial_ids={'B09','C03','C06'}
    access=f'<li>27篇取得正文，核读各卡片注明的关键方法、结果及讨论。</li><li>3篇正文访问受限：'+ '；'.join(f'<a href="#{p["id"]}">{p["id"]} · {esc(p["title"])}</a>' for p in papers if p['id'] in partial_ids)+'</li><li>预印本及额外竞争筛查另列，不计入30篇。</li>'
    index=''.join(f'<li><a href="#{p["id"]}">{p["id"]} · {esc(p["title"])}</a> <span>{p["year"]}</span></li>' for p in papers)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>30篇核心文献评议与科学问题选择</title><style>
    :root{--ink:#142a36;--muted:#526776;--accent:#087d7f;--line:#d7e2e4;--bg:#f4f7f7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.8 'Microsoft YaHei',system-ui,sans-serif}header{background:#143746;color:white;padding:58px max(24px,calc((100vw - 1080px)/2));border-bottom:7px solid #47b4ab}header p{max-width:900px;color:#cde2e7}h1{font-size:36px;line-height:1.35;max-width:900px;margin:12px 0 20px}main{max-width:1128px;margin:auto;padding:28px 24px 70px}section{background:white;border:1px solid var(--line);border-radius:12px;padding:28px 32px;margin:24px 0}h2{font-size:25px;margin:0 0 18px}h3{font-size:19px;line-height:1.5;margin:8px 0}a{color:#006f79;text-decoration:none}a:hover{text-decoration:underline}header a{color:white}p{margin:10px 0}li{margin:10px 0}.lead{font-size:22px;font-weight:bold;border-left:4px solid var(--accent);padding-left:20px}.muted,.small,.authors,.doi{color:var(--muted);font-size:14px}.eyebrow{font-size:13px;font-weight:bold;letter-spacing:.06em;color:var(--accent)}header .eyebrow{color:#7ce0d0}.actions{display:flex;gap:12px;flex-wrap:wrap}.button{display:inline-block;padding:8px 16px;border:1px solid var(--line);border-radius:7px;background:#fff}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;min-width:850px}td,th{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:13px 12px}th{background:#e9f2f2}td:first-child{font-weight:bold;width:18%}.paper{padding:24px 0;border-top:1px solid var(--line)}.paper:first-child{border-top:0}.access{font-size:14px;color:#825518}.field{display:grid;grid-template-columns:150px 1fr;gap:18px;margin:14px 0}.field b{font-size:14px;color:var(--muted)}.field p{margin:0;font-size:15px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}input,select,button{font:inherit;padding:9px 12px;border:1px solid #aebfc3;border-radius:6px;background:white}input{flex:1;min-width:240px}button,summary{cursor:pointer}summary{color:var(--accent);font-weight:bold;font-size:15px}.index{columns:2;column-gap:30px;font-size:14px;padding-left:22px}.index li{break-inside:avoid}.index span{color:var(--muted)}.banner{padding:18px 22px;background:#edf5f3;border-radius:8px}.hide{display:none}footer{font-size:13px;color:var(--muted);padding:20px 0}@media(max-width:750px){h1{font-size:28px}section{padding:22px 18px}.field{display:block}.index{columns:1}.lead{font-size:19px}}@media print{body{background:white;font-size:11pt}header{background:white;color:#142a36;padding:20px;border-bottom:2px solid}header p{color:#526776}main{padding:0}section{border:0;padding:18px 0;break-inside:auto}.controls,.actions{display:none}.paper{break-inside:avoid}.field p{font-size:10pt}.index{columns:1}a{color:inherit}}
    </style><header><div class="eyebrow">LIA / 654 SELECTED GLACIER SYSTEMS / 12 OCT 2026</div><h1>30篇核心文献评议<br>先检验证据，再选择科学问题</h1><p>理论、观测与重建三条证据线；逐篇记录科学问题、证据独立性、实际读取范围和对本项目的约束。</p></header><main>'''
    page+=f'<div class="actions"><a class="button" href="#decision">科学问题选择</a><a class="button" href="#papers">30篇逐篇评议</a><a class="button" href="evidence30.json">完整证据JSON</a><a class="button" href="references30.bib">引用文件BibTeX</a></div><section id="decision"><h2>调研后的决定</h2><p class="lead">{esc(decision["recommended_question"])}</p><p>{esc(decision["decision_status"])}</p><ul>{conclusions}</ul></section>'
    if decision.get('research_design'):
        design=decision['research_design']
        page+='<section><h2>把科学问题变成可证伪的研究</h2>'+''.join(f'<div class="field"><b>{esc(x["label"])}</b><p>{esc(x["text"])}</p></div>' for x in design)+'</section>'
    page+=f'<section><h2>候选问题的取舍</h2><div class="table-wrap"><table><thead><tr><th>候选问题</th><th>已有研究做到哪里</th><th>本数据可能增加的信息</th><th>决定性检验</th><th>取舍</th></tr></thead><tbody>{rows}</tbody></table></div></section>'
    page+=f'<section><h2>什么证据会改变决定</h2><ol>{gates}</ol><div class="banner">{esc(decision["stop_rule"])}</div></section>'
    page+=f'<section><h2>阅读范围与遴选依据</h2><p>{esc(decision["selection_statement"])}</p><ul>{access}</ul><p>{esc(decision["access_caveat"])}</p><details><summary>30篇完整索引</summary><ol class="index">{index}</ol></details></section>'
    page+=f'<section id="papers"><h2>逐篇证据与批判性评议</h2><div class="controls"><input id="q" placeholder="搜索作者、题名、问题或方法" aria-label="搜索文献"><select id="group" aria-label="研究组"><option value="">全部30篇</option value="A">A 理论与几何反馈</option><option value="B">B 高亚洲观测与机制</option><option value="C">C 重建与可识别性</option></select><button id="expand">展开全部评议</button></div><p id="count" class="small">显示30篇</p>{"".join(cards)}</section>'
    page+=f'<section><h2>项目证据边界</h2><ul>{limits}</ul></section><section><h2>核心30篇之外的前沿筛查</h2><p class="small">以下内容单列，不计入30篇已发表核心研究。</p><ul>{extra}</ul></section><footer>本报告是面向选题的定向批判性调研，不是穷尽式系统综述。未计算新的冰川指标，未运行模型，未改变既有654样本或项目状态。书目、读取层级与评议可在JSON中逐项追溯。</footer></main>'
    page+='''<script>const papers=[...document.querySelectorAll('.paper')];function filter(){const q=document.querySelector('#q').value.toLowerCase(),g=document.querySelector('#group').value;let n=0;papers.forEach(p=>{const ok=(!g||p.dataset.group===g)&&p.textContent.toLowerCase().includes(q);p.classList.toggle('hide',!ok);n+=ok;});document.querySelector('#count').textContent='显示 '+n+' / 30 篇';}document.querySelector('#q').addEventListener('input',filter);document.querySelector('#group').addEventListener('change',filter);document.querySelector('#expand').addEventListener('click',()=>{let opening=papers.some(p=>!p.querySelector('details').open);papers.forEach(p=>p.querySelector('details').open=opening);document.querySelector('#expand').textContent=opening?'收起全部评议':'展开全部评议';});addEventListener('beforeprint',()=>papers.forEach(p=>p.querySelector('details').open=true));</script></html>'''
    (OUT/'report.html').write_text(page,encoding='utf-8')
    qc={'paper_count':len(papers),'unique_dois':30,'required_fields_complete':True,'group_counts':dict(collections.Counter(p['group'] for p in papers)),'body_key_sections_reviewed':27,'partial_access_ids':sorted(partial_ids),'access_levels':dict(read_levels),'new_glacier_metrics':False,'source_deliveries_modified':False,'browser_render_verified':False}
    (OUT/'content_QC.json').write_text(json.dumps(qc,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(qc,ensure_ascii=False))

if __name__=='__main__':main()
