"""Post-process saved historical outputs only. Never launches glacier dynamics."""
from pathlib import Path
import importlib.util
import json
from datetime import datetime, timezone
import html
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'analysis/lia_51_terminus_elevation_curves_v1'
spec = importlib.util.spec_from_file_location('previous_analysis', OLD/'build_grouped_curves.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
BANDS = previous.BANDS
YEARS = previous.YEARS
COHORTS = [('retained', '32条已认可'), ('candidate', '19条候选'), ('all', '全部51条（补充）')]
METRICS = [('length', 'length_km', '长度', 'km'), ('volume', 'volume_km3', '体积', 'km³')]


def save_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def decline(values, width):
    """Operational final sustained descent, not an inferred climatic event."""
    starts = np.arange(1500, 2000, width)
    means = np.asarray(values[:500]).reshape(-1, width).mean(axis=1)
    epsilon = 1e-10 * max(1., float(np.max(np.abs(means))))
    diffs = np.diff(means)
    increases = np.flatnonzero(diffs > epsilon)
    index = int(increases[-1]+1) if len(increases) else 0
    tail = diffs[index:]
    declines = int(np.count_nonzero(tail < -epsilon))
    required = int(np.ceil(30/width))
    identified = declines >= required and len(tail) >= required
    return dict(bin_width_years=width, start_year=int(starts[index]) if identified else None,
                start_bin_end_year=int(starts[index]+width-1) if identified else None,
                left_censored=bool(identified and index == 0), strictly_declining_intervals=declines,
                required_declining_intervals=required, bin_start_years=starts.tolist(), means=means.tolist())


def native_peak(years, values):
    mask = (years >= 1500) & (years <= 2000)
    y, x = years[mask], values[mask]
    value = float(x.max())
    tied = np.flatnonzero(np.isclose(x, value, rtol=1e-10, atol=1e-12))
    blocks = np.split(tied, np.flatnonzero(np.diff(tied)>1)+1)
    segments = [[int(y[b[0]]), int(y[b[-1]])] for b in blocks]
    return dict(value=value, first_year=int(y[tied[0]]), last_year=int(y[tied[-1]]),
                tied_native_segments=segments, at_left_boundary=bool(y[tied[0]]==1500),
                at_right_boundary=bool(y[tied[-1]]==2000))


def prepare():
    methods = previous.read_json(OUT/'methods.json')
    previous.register(Path(__file__))
    previous.register(OLD/'build_grouped_curves.py')
    curves, members, sampling = previous.load_sources()
    inventory = {r['lia_unique_id']: r for r in previous.read_json(ROOT/'analysis/lia_51_inventory_v1/inventory.json')}
    diagnostics = []
    for c, m in zip(curves, members):
        assert c['system'] == m['system']
        inv = inventory[m['system']]
        d = dict(system=m['system'], status=m['status'], status_zh=m['status_zh'],
                 group=m['elevation_group'], elevation_m=m['terminus_elevation_m'],
                 longitude=m['longitude'], latitude=m['latitude'],
                 mainline_gradient_deg=inv['lia_mainline_mean_gradient_deg'],
                 static_area_km2=m['static_lia_area_km2'], native_max_interval=m['native_step_years_max'],
                 registered_lia_target_year=inv['lia_registered_target_year'])
        for metric, key, _, _ in METRICS:
            a = np.asarray(c[key])
            c[metric+'_relative_2000_pct'] = (100*(a/a[-1]-1)).tolist()
            p = native_peak(np.asarray(c['native_years']), np.asarray(c['native_'+key]))
            sensitivity = {str(w): decline(a, w) for w in [5, 10, 20]}
            for w in [5, 10, 20]:
                sensitivity[str(w)]['source_resolution_supported'] = m['native_step_years_max'] <= w
            supported = [v['start_year'] for v in sensitivity.values()
                         if v['source_resolution_supported'] and v['start_year'] is not None]
            d[metric] = dict(peak=p, terminal_decline=sensitivity['10'], sensitivity=sensitivity,
                             supported_onset_range=[min(supported),max(supported)] if supported else None,
                             loss_peak_to_2000_pct=float(100*(p['value']-a[-1])/p['value']),
                             periods=[dict(start=s,end=e,change_pct=float(100*(a[e-1500]/a[s-1500]-1)),
                                           absolute_change=float(a[e-1500]-a[s-1500]))
                                      for s,e in methods['fixed_periods']])
        diagnostics.append(d)
    return curves, members, diagnostics, methods, sampling


def subset(rows, cohort, band):
    return [r for r in rows if (cohort=='all' or r['status']==cohort) and r['group']==band]


def axis_style(ax, xlabel='年份'):
    ax.grid(axis='y', alpha=.16)
    ax.set_xlabel(xlabel)
    if xlabel=='年份':
        ax.set_xlim(1500, 2000)
        ax.set_xticks(np.arange(1500,2001,100))
        labels = ax.get_xticklabels()
        if labels:
            labels[0].set_ha('left'); labels[-1].set_ha('right')


def plot_curves(curves, diagnostics, cohort, raw=False):
    lookup={c['system']:c for c in curves}
    fig,axs=plt.subplots(2,3,figsize=(15,8.4),sharex=True,sharey='row')
    for j,(band,label,color) in enumerate(BANDS):
        rows=subset(diagnostics,cohort,band)
        for i,(metric,key,name,unit) in enumerate(METRICS):
            ax=axs[i,j]
            ck=key if raw else metric+'_relative_2000_pct'
            a=np.array([lookup[d['system']][ck] for d in rows])
            q=np.quantile(a,[.25,.5,.75],axis=0)
            ax.fill_between(YEARS,q[0],q[2],color=color,alpha=.16,lw=0)
            for d in rows:
                c=lookup[d['system']]
                y=np.asarray(c['native_years']); x=np.asarray(c['native_'+key])
                use=(y>=1500)&(y<=2000)
                z=x if raw else 100*(x/c[key][-1]-1)
                ax.plot(y[use],z[use],color='#627784',lw=.85,alpha=.44,
                        ls='-' if d['status']=='retained' else '--')
            ax.plot(YEARS,q[1],color=color,lw=2.6)
            if not raw: ax.axhline(0,color='#666666',lw=.65)
            axis_style(ax)
            if i==0:ax.set_title(f'{label} · n={len(rows)}',fontsize=12,pad=14)
            if j==0:ax.set_ylabel(f'{name} / {unit}' if raw else f'{name}相对模拟2000年 / %')
    cohortname=dict(COHORTS)[cohort]
    fig.suptitle(f'1500—2000年历史演变｜{cohortname}',fontsize=17,y=.982)
    subtitle='绝对模拟值；不将1500年自动视为LIA重建状态' if raw else '2000年是显示参照，不是LIA基准；归一化不能消除初态影响'
    fig.text(.5,.93,subtitle,ha='center',fontsize=11,color='#465767')
    fig.text(.5,.025,'细线：单条历史曲线；粗线：等权中位数；阴影：冰川间25%–75%范围，并非置信区间。spinup不在此图中。',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.84,bottom=.10,wspace=.10,hspace=.20)
    suffix='_absolute' if raw else '_relative2000'
    stem=cohort+suffix
    fig.savefig(OUT/(stem+'.png'),dpi=160)
    if cohort=='retained' and not raw:fig.savefig(OUT/(stem+'.svg'))
    plt.close(fig)


def plot_timing(diagnostics,cohort):
    fig,axs=plt.subplots(2,3,figsize=(15,8.8),sharex=True)
    for j,(band,label,color) in enumerate(BANDS):
        rows=sorted(subset(diagnostics,cohort,band),key=lambda d:d['elevation_m'])
        for i,(metric,_,name,_) in enumerate(METRICS):
            ax=axs[i,j]
            for k,d in enumerate(rows):
                p=d[metric]['peak']; onset=d[metric]['terminal_decline']['start_year']
                ax.plot([p['first_year'],p['last_year']],[k,k],color=color,lw=2.2,alpha=.7)
                ax.scatter(p['first_year'],k,color=color,s=25,marker='o')
                if p['at_left_boundary'] or p['at_right_boundary']:
                    ax.scatter(p['first_year'],k,s=85,facecolors='none',edgecolors='#b12626',lw=1.1)
                if onset is not None:ax.scatter(onset,k,color='#293743',s=28,marker='v')
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([str(d['system']) for d in rows],fontsize=8)
            ax.set_ylim(-.75,len(rows)-.25)
            axis_style(ax)
            if i==0:ax.set_title(f'{label} · n={len(rows)}',fontsize=12,pad=10)
            ax.set_ylabel(f'{name}｜LIA编号（由低到高）',fontsize=10)
    handles=[Line2D([],[],color='#286c9f',marker='o',label='窗口内峰值；线段为并列峰值最早—最晚（可不连续）'),
             Line2D([],[],color='#293743',marker='v',ls='',label='末段持续下降起点（十年尺度）')]
    fig.suptitle(f'峰值与末段下降时间｜{dict(COHORTS)[cohort]}',fontsize=17,y=.985)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.955),ncol=2,fontsize=9,frameon=False)
    fig.text(.5,.021,'红圈：峰值落在分析窗口边界；未画三角表示未满足≥30年持续下降规则。不是LIA定年，也不是已验证的气候转折点。',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.86,bottom=.09,wspace=.27,hspace=.23)
    fig.savefig(OUT/f'{cohort}_timing.png',dpi=170);plt.close(fig)


def plot_amplitudes(diagnostics,cohort):
    fig,axs=plt.subplots(2,4,figsize=(15,7.0),sharex=True)
    titles=['1500→1850','1850→1900','1900→2000','窗口峰值→2000损失']
    for i,(metric,_,name,_) in enumerate(METRICS):
        for j in range(4):
            ax=axs[i,j]
            for k,(band,_,color) in enumerate(BANDS):
                rows=subset(diagnostics,cohort,band)
                vals=[d[metric]['periods'][j]['change_pct'] if j<3 else d[metric]['loss_peak_to_2000_pct'] for d in rows]
                offsets=np.linspace(-.16,.16,len(vals))
                ax.scatter(k+offsets,vals,color=color,alpha=.65,s=23)
                q=np.quantile(vals,[.25,.5,.75])
                ax.plot([k,k],[q[0],q[2]],color=color,lw=5)
                ax.plot([k-.22,k+.22],[q[1],q[1]],color='#263c49',lw=1.8)
            ax.axhline(0,color='#999999',lw=.7)
            ax.set_xticks(range(3));ax.set_xticklabels(['低','中','高'])
            axis_style(ax,'LIA末端高程组')
            if i==0:ax.set_title(titles[j],fontsize=12)
            if j==0:ax.set_ylabel(f'{name}变化 / %')
    fig.suptitle(f'变化幅度｜{dict(COHORTS)[cohort]}',fontsize=16,y=.98)
    fig.text(.5,.027,'前三列：相对各时段起点，负值为减少；最后一列：以各自峰值为分母，正值为损失。点：单冰川；横线：中位数；粗竖线：四分位范围。',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.87,bottom=.13,wspace=.27,hspace=.18)
    fig.savefig(OUT/f'{cohort}_amplitudes.png',dpi=170);plt.close(fig)


def summaries(diagnostics):
    out=[]
    for cohort,_ in COHORTS:
        for band,label,_ in BANDS:
            rows=subset(diagnostics,cohort,band)
            for metric,_,name,_ in METRICS:
                values=[d[metric] for d in rows]
                starts=[v['terminal_decline']['start_year'] for v in values if v['terminal_decline']['start_year'] is not None]
                interior=[v['peak']['first_year'] for v in values if not(v['peak']['at_left_boundary'] or v['peak']['at_right_boundary'])]
                out.append(dict(cohort=cohort,group=band,label=label,metric=metric,n=len(rows),
                                peak_first_year_median=float(np.median(interior)) if interior else None,
                                interior_peak_n=len(interior),boundary_peak_n=len(rows)-len(interior),
                                decline_start_median=float(np.median(starts)) if starts else None,
                                decline_identified_n=len(starts),
                                loss_peak_to_2000_pct_median=float(np.median([v['loss_peak_to_2000_pct'] for v in values])),
                                period_change_pct_medians=[float(np.median([v['periods'][j]['change_pct'] for v in values])) for j in range(3)]))
    return out


def fmt_year(value):
    return '未识别' if value is None else f'{value:g}'


def make_report(diagnostics,summary):
    rows=[]
    for d in sorted(diagnostics,key=lambda x:x['elevation_m']):
        r={'LIA编号':d['system'],'状态':d['status_zh'],'末端高程 m':round(d['elevation_m']),
           '原生间隔 年':d['native_max_interval']}
        for metric,_,name,_ in METRICS:
            v=d[metric];p=v['peak']
            r[name+'峰值年']=str(p['first_year'])+(f'—{p["last_year"]}（并列）' if p['last_year']!=p['first_year'] else '')+('＊边界' if p['at_left_boundary'] or p['at_right_boundary'] else '')
            r[name+'末段下降']=fmt_year(v['terminal_decline']['start_year'])
            r[name+'峰后损失 %']=f'{v["loss_peak_to_2000_pct"]:.1f}'
            s=v['supported_onset_range']
            eligible=[a for a in v['sensitivity'].values() if a['source_resolution_supported']]
            identified=sum(a['start_year'] is not None for a in eligible)
            r[name+'分辨率检查']=(f'{s[0]}—{s[1]}' if s else '未识别')+f'（识别{identified}/{len(eligible)}）'
        rows.append(r)
    stats=[]
    for s in summary:
        if s['cohort']!='retained':continue
        stats.append({'分组':s['label'],'指标':'长度' if s['metric']=='length' else '体积','条数':s['n'],
                      '内部峰值年中位数':fmt_year(s['peak_first_year_median']),
                      '边界峰值条数':s['boundary_peak_n'],'末段下降年中位数':fmt_year(s['decline_start_median']),
                      '下降识别条数':s['decline_identified_n'],'峰后损失中位数 %':f'{s["loss_peak_to_2000_pct_median"]:.1f}'})
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LIA历史演变 · 第二版</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#243b49;background:#f6f8fa}h1{font-size:28px}h2{margin-top:35px;font-size:22px}p,li{line-height:1.85}p{max-width:1200px}.note{background:#e9f0f4;border-left:4px solid #347da0;padding:15px 20px}img{width:100%;background:white;border:1px solid #dee5e9}a{color:#126b90}.scroll{overflow:auto;max-height:640px}table{border-collapse:collapse;background:white;font-size:13px;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #e1e7eb;text-align:right}th{background:#284e63;color:white;position:sticky;top:0}details{margin:20px 0}summary{cursor:pointer;font-size:17px}input{padding:10px;width:260px;margin:14px 0}small{color:#536573}</style>
<h1>LIA以来的历史演变：时间、阶段与幅度</h1>
<p>第二版｜32条已认可为主分析；19条旧复核候选单独展示。沿用上一版的代表曲线和末端高程分组，没有新增模拟、调参或修改认可状态。</p>
<div class="note"><b>这次研究什么：</b>1500—2000年模拟历史中的推进、窗口峰值及最后持续下降阶段。每条曲线以其<b>模拟2000年</b>为显示参照，spinup不进入历史曲线。换参照年不能消除初态影响；这里不把1500年值当成LIA重建状态，也不把窗口峰值称为整个LIA的最大值。</div>
<p><a href="retained_relative2000.png">主图PNG</a> · <a href="retained_relative2000.svg">主图SVG</a> · <a href="diagnostics.json">逐条诊断数据</a> · <a href="summary.json">分组统计</a> · <a href="methods.json">方法定义</a> · <a href="source_manifest.json">来源校验</a></p>
<h2>1. 历史演变：32条已认可</h2><img src="retained_relative2000.png" alt="32条已认可冰川的历史演变，相对模拟2000年">
<p>相对量 = 100 × [X(t)/X(2000) − 1]。同一条冰川的正值表示当年比其模拟2000年更大，<b>不表示当年正在增长</b>；增长与退缩要看曲线斜率。粗线和阴影分别是冰川等权中位数和四分位范围，并非区域总体积或置信区间。</p>
<details><summary>查看绝对长度与体积，避免归一化掩盖规模差异</summary><img src="retained_absolute.png" alt="绝对模拟长度体积"></details>
<h2>2. 何时达到峰值，何时进入末段下降？</h2><img src="retained_timing.png" alt="已认可冰川的峰值与末段下降时间">
<p>峰值取1500—2000年原生保存点的最大值，并列峰值保留最早和最晚年份。红圈表示最大值落在窗口边界，实际峰值可能在窗口之外；统计内部峰值年份时排除这些边界情况。线段两端之间不一定始终处于峰值。</p>
<p>倒三角是<b>末段持续下降的操作性指标</b>：将1500—1999年划为连续十年段，要求从该段起一直到1990年代均值不再上升，且至少有三个下降间隔。未满足则不标记。它不是首次退缩、不是冰碛定年，也不是已证实的气候转折点。</p>
<p class="note">这批结果的下降起点对分段尺度敏感，部分冰川在某个尺度下未能识别。先作为诊断保留，不据此宣称统一的LIA结束年代。逐条表同时列出支持的尺度中成功识别的比例。</p>
<h2>3. 变化幅度：分时段，而非只看1500与2000两个端点</h2><img src="retained_amplitudes.png" alt="已认可冰川的分阶段变化幅度">
<p>1850年只是固定的比较切分点，不代表每条冰川的LIA终止年代。峰后损失以各自峰值为分母，与主图以2000年为分母的百分比不同。</p>
<h2>4. 已认可组统计</h2><div class="scroll">'''+pd.DataFrame(stats).to_html(index=False,escape=True)+'''</div>
<p>年份中位数是样本描述，并不具有一年级别的定年精度。峰值统计仅包含内部峰值，末段下降统计仅包含成功识别者；各自有效条数需一起看。</p>
<h2>5. 候选曲线单独比较</h2><details><summary>展开19条候选的演变、时间与幅度</summary><img src="candidate_relative2000.png" alt="候选历史演变"><img src="candidate_timing.png" alt="候选时间指标"><img src="candidate_amplitudes.png" alt="候选变化幅度"></details>
<details><summary>全部51条混合展示（仅作补充）</summary><img src="all_relative2000.png" alt="全部51条混合演变"></details>
<h2>6. 逐条诊断</h2><input id="q" placeholder="搜索LIA编号或状态"><span id="count"></span><div class="scroll">'''+pd.DataFrame(rows).to_html(index=False,table_id='members',escape=True)+'''</div>
<h2>7. 限制与方法来源</h2><ul>
<li>末端高程是成对V3/F19重建栅格上的静态LIA主流线终点高程，不是现代最低高程，也不是各年移动的末端高程。分界4000和4500 m保持不变。</li>
<li>部分原始曲线每10年保存一次。原生线不平滑；统计所需共同年度坐标沿用线性插值，插值不增加真实时间分辨率。峰值用原生点，不用插值制造新峰。</li>
<li>末段下降另以5年、20年分段检查；原生间隔大于5年的冰川不把5年结果当作有效敏感性证据。表中的范围只总结成功识别且时间分辨率受支持的结果，不是置信区间；未识别情况完整保留在诊断数据中。</li>
<li>尚未完成逐条初态记忆的归因检验。因此早期升降不能自动归因于气候，也不能仅凭曲线平滑就认为初态影响消失。</li>
<li>既有清单中的LIA目标登记年为1400；本次时间窗从1500开始，不能据此寻找1400年或更早的峰值。该登记值不等于每条冰川独立定年的最大范围年代。</li>
<li>本次没有把重建栅格体积直接作为模型曲线的归一化分母，也没有把“长度满足约束”当作“体积已被独立验证”。</li>
<li>尚未控制地区、规模、坡度和参数路线的混杂。高程组比较是描述性的，不能单独证明高程效应。没有自动划分LIA子阶段或对气候机制进行归因。</li></ul>
<p>参考：<a href="https://tc.copernicus.org/articles/16/3739/2022/">Yang等（2022）</a>使用1950年参照的长度变化，研究推进峰值并检验初态影响；<a href="https://tc.copernicus.org/articles/14/3135/2020/">Parkes与Goosse（2020）</a>同时检查归一化演变与绝对误差；<a href="https://tc.copernicus.org/articles/19/5913/2025/">Henz等（2025）</a>区分LIA几何重建与连续历史演变。本页的2000年参照和末段下降规则是此次明确声明的描述性选择，不是照搬上述论文的全部方法。</p>
<script>const rows=[...document.querySelectorAll('#members tbody tr')];function f(){const q=document.querySelector('#q').value.toLowerCase();let n=0;rows.forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q);if(!r.hidden)n++});document.querySelector('#count').textContent=' 显示 '+n+' 条'}document.querySelector('#q').oninput=f;f();</script></html>'''
    (OUT/'index.html').write_text(doc,encoding='utf-8')


def main():
    curves,members,diagnostics,methods,sampling=prepare()
    summary=summaries(diagnostics)
    for cohort,_ in COHORTS:plot_curves(curves,diagnostics,cohort)
    plot_curves(curves,diagnostics,'retained',raw=True)
    for cohort in ['retained','candidate']:
        plot_timing(diagnostics,cohort);plot_amplitudes(diagnostics,cohort)
    make_report(diagnostics,summary)
    for path,digest in previous.SOURCES.items():
        assert previous.sha(ROOT/path)==digest, f'Changed source: {path}'
    save_json('curves.json',curves)
    save_json('members.json',members)
    save_json('diagnostics.json',diagnostics)
    save_json('summary.json',summary)
    save_json('source_manifest.json',previous.SOURCES)
    assert len({c['system'] for c in curves})==51
    for c in curves:
        for metric,_,_,_ in METRICS:
            assert c[metric+'_relative_2000_pct'][-1]==0
    for s in summary:assert s['n']==len(subset(diagnostics,s['cohort'],s['group']))
    save_json('validation.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),status='pass',
              unique_systems=51,retained=32,candidates=19,source_hashes_checked=len(previous.SOURCES),
              native_history_closure=True,baseline_2000_zero=True,new_simulations=0,
              original_results_modified=False,acceptance_changed=False))
    print(json.dumps([s for s in summary if s['cohort']=='retained'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
