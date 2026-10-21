"""Descriptive altitude grouping of saved curves; no OGGM model execution."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'validation_data/results/paleo_surface_validation_v0'
OUT = Path(__file__).resolve().parent
YEARS = np.arange(1500, 2001)
BANDS = [('low', '低末端：< 4000 m', '#286c9f'),
         ('middle', '中末端：4000–<4500 m', '#cc8a27'),
         ('high', '高末端：≥ 4500 m', '#945494')]
SOURCES = {}
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def register(path, expected=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    digest = sha(path)
    if expected is not None:
        assert digest == expected.upper(), f'Source hash mismatch: {path}'
    SOURCES[path.relative_to(ROOT).as_posix()] = digest
    return path


def read_json(path):
    return json.loads(register(path).read_text(encoding='utf-8-sig'))


def json_write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def band_of(z):
    return 'low' if z < 4000 else 'middle' if z < 4500 else 'high'


def load_sources():
    inventory = {int(r['lia_unique_id']): r for r in read_json(ROOT / 'analysis/lia_51_inventory_v1/inventory.json')}
    approved = read_json(BASE / 'lia_accepted32_research26_v1/summary.json')
    keep = set(approved['retained_system_ids'])
    completion = read_json(BASE / 'lia_direct_continuous_v1/completion-local-v1.json')
    candidates = set(completion['combined_candidate_systems'])
    assert len(keep) == 32 and len(candidates) == 19 and not keep & candidates
    assert set(inventory) == keep | candidates
    meta = pd.read_csv(register(BASE / 'lia_191_retained_32_mean_geometry_mass_balance_v2/source_manifest.csv')).set_index('system_id')
    annual = pd.read_csv(register(BASE / 'lia_191_retained_32_mean_geometry_mass_balance_v2/retained_32_annual_geometry_and_net_mass_balance.csv'))
    old_gallery = {int(g['system']): g for g in read_json(BASE / 'lia_new38_legacy_review_recalculation_v1/data.json')}
    old_summary = read_json(BASE / 'lia_new38_legacy_review_recalculation_v1/summary.json')
    new_gallery = {int(g['system']): g for g in read_json(BASE / 'lia_direct_new4_gallery_v1/data.json')}
    assert set(old_gallery) == set(completion['original_candidate_systems'])
    assert set(new_gallery) == set(completion['direct_search_candidate_systems'])
    # Verify the paired V3 elevation provenance used by the preceding inventory.
    elevation_manifest = read_json(ROOT / 'analysis/lia_51_inventory_v1/sources.json')
    for sid, info in inventory.items():
        for key in ['lia_source_grid', 'lia_network_source']:
            p = info[key]
            register(p, elevation_manifest[p])
        assert info['lia_terminus_elevation_sample_distance_m'] <= info['lia_elevation_sample_limit_m']

    curves, selection, sampling = [], [], []
    for sid in sorted(inventory):
        info = inventory[sid]
        z = float(info['lia_terminus_surface_m'])
        item = dict(system=sid, rgi=info['main_rgi_id'], status='retained' if sid in keep else 'candidate',
                    status_zh='已认可' if sid in keep else '旧复核候选',
                    terminus_elevation_m=z, elevation_group=band_of(z),
                    longitude=info['lia_positive_ice_centroid_lon'], latitude=info['lia_positive_ice_centroid_lat'],
                    elevation_sample_distance_m=info['lia_terminus_elevation_sample_distance_m'],
                    elevation_source=info['lia_source_grid'],
                    static_lia_area_km2=info['lia_positive_ice_area_km2'],
                    frozen_lia_target_km=info['lia_reference_length_km'],
                    frozen_modern_target_km=info['modern_reference_length_km'])
        if sid in keep:
            record = meta.loc[sid]
            p = register(record.source)
            d = pd.read_csv(p)
            if 'phase' in d:
                d = d.loc[d.phase.eq('forward')].copy()
            d = d.loc[d.year.between(1470, 2000)].sort_values('year')
            assert not d.year.duplicated().any(), (sid, 'duplicate native year')
            y = d.year.to_numpy(float)
            key = 'observation_length_m' if record.route == 'frozen-legacy' else 'length_m'
            if key not in d.columns:
                key = 'length_m'
            length = d[key].fillna(d['length_m']).to_numpy(float)/1000
            volume = d.volume_m3.to_numpy(float)/1e9
            area = d.area_m2.to_numpy(float)/1e6
            item.update(selection='既有保留32条登记曲线；不重新择优', source=p.relative_to(ROOT).as_posix(),
                        model_route=str(record.route), length_operator=key, curve_page=None)
        else:
            if sid in new_gallery:
                g = new_gallery[sid]
                case = g['selected']
                choice = g['selection']
                proof = choice['forward']
                params, score = choice['candidate'], choice['legacy_score']
                page = BASE / choice['batch'] / 'curves.html'
                selection_rule = '沿用新增4条展示页已选定的最高旧复核评分组合'
            else:
                g = old_gallery[sid]
                # Preserve source ordering for exact score ties; no refitting or rescoring.
                case = max(g['cases'], key=lambda c: c['legacy_review_score'])
                proof = case['proof']
                params, score = case['candidate'], case['legacy_review_score']
                page = (BASE / 'lia_new38_legacy_review_recalculation_v1' / g['source_page']).resolve()
                selection_rule = '已有旧复核通过组合中取最高既存评分；同分沿用原页顺序'
            assert score >= old_summary['threshold']
            rp = register(proof['result']['path'], proof['result']['sha256'])
            sp = register(proof['state']['path'], proof['state']['sha256'])
            result = read_json(rp)
            assert result['candidate'] == params, (sid, 'result/selection mismatch')
            with np.load(sp, allow_pickle=False) as a:
                y = a['years'].copy().astype(float)
                history = a['history'].copy()
            assert np.array_equal(y, np.arange(1470, 2001)), sid
            length, area, volume = history[:, 0]/1000, history[:, 1]/1e6, history[:, 2]/1e9
            assert np.array_equal(y, case['years'])
            assert np.allclose(length, case['length_km'], atol=1e-12, rtol=0)
            assert np.allclose(volume, case['volume_km3'], atol=1e-12, rtol=0)
            item.update(selection=selection_rule, source=sp.relative_to(ROOT).as_posix(),
                        result_source=rp.relative_to(ROOT).as_posix(), model_route='saved-GPU-research-candidate',
                        length_operator='native history[:,0] / 1000', parameters=params,
                        existing_legacy_score=float(score), curve_page=page.relative_to(ROOT).as_posix())
        assert y[0] <= 1500 and y[-1] == 2000
        assert np.all(np.diff(y) > 0) and np.isfinite([length, volume, area]).all()
        assert np.all(length > 0) and np.all(volume > 0) and np.all(area > 0)
        values = {'length_km': length, 'volume_km3': volume, 'area_km2': area}
        curve = dict(system=sid, native_years=y.tolist(), year=YEARS.tolist())
        for key, raw in values.items():
            aligned = np.interp(YEARS, y, raw)
            curve['native_' + key] = raw.tolist()
            curve[key] = aligned.tolist()
            short = key.split('_')[0]
            curve[short + '_change_percent'] = (100*(aligned/aligned[0]-1)).tolist()
            item[short + '_1500'] = float(aligned[0])
            item[short + '_2000'] = float(aligned[-1])
            item[short + '_change_1500_2000_percent'] = float(curve[short + '_change_percent'][-1])
            item[short + '_change_1900_2000_percent'] = float(100*(aligned[-1]/aligned[400]-1))
            if sid in keep:
                old = annual.loc[annual.system_id.eq(sid)].sort_values('year')
                assert np.array_equal(old.year, YEARS)
                assert np.allclose(aligned, old[key], rtol=1e-11, atol=1e-11), (sid, key, 'prior-table closure')
        curve['length_fraction_lia_target'] = (np.asarray(curve['length_km'])/item['frozen_lia_target_km']).tolist()
        native_l = np.asarray(curve['length_km'])
        item['first_length_peak_year_1500_2000'] = int(YEARS[np.argmax(native_l)])
        item['native_step_years_min'] = float(np.diff(y).min())
        item['native_step_years_max'] = float(np.diff(y).max())
        item['interpolated_to_annual'] = bool(np.diff(y).max() > 1.001)
        item['distance_to_nearest_group_boundary_m'] = min(abs(z-4000), abs(z-4500))
        curves.append(curve)
        selection.append(item)
        sampling.append(dict(system=sid, native_points=len(y), interval_years=np.unique(np.diff(y)).tolist(),
                             annual_alignment='linear interpolation' if item['interpolated_to_annual'] else 'native annual'))
    assert len(selection) == 51
    return curves, selection, sampling


def group_statistics(curves, members):
    by_id = {c['system']: c for c in curves}
    stats, endpoints = [], []
    for cohort in ['all', 'retained', 'candidate']:
        for code, label, color in BANDS:
            rows = [m for m in members if m['elevation_group'] == code and (cohort == 'all' or m['status'] == cohort)]
            ids = [m['system'] for m in rows]
            assert ids
            stat = dict(cohort=cohort, band=code, label=label, color=color, n=len(ids), systems=ids, year=YEARS.tolist())
            for key in ['length_change_percent', 'volume_change_percent', 'length_km', 'volume_km3', 'length_fraction_lia_target']:
                a = np.array([by_id[sid][key] for sid in ids])
                stat[key] = {name: f(a, axis=0).tolist() for name, f in [
                    ('median', np.median), ('mean', np.mean),
                    ('q25', lambda a, axis: np.quantile(a, .25, axis=axis)),
                    ('q75', lambda a, axis: np.quantile(a, .75, axis=axis))]}
            stats.append(stat)
            endpoint = dict(cohort=cohort, band=code, label=label, n=len(ids), systems=ids,
                            retained=sum(m['status']=='retained' for m in rows),
                            candidates=sum(m['status']=='candidate' for m in rows),
                            elevation_min_m=min(m['terminus_elevation_m'] for m in rows),
                            elevation_max_m=max(m['terminus_elevation_m'] for m in rows))
            for metric in ['length','volume']:
                arr = np.array([m[metric+'_change_1500_2000_percent'] for m in rows])
                endpoint[metric+'_change_1500_2000_percent_median'] = float(np.median(arr))
                endpoint[metric+'_change_1500_2000_percent_iqr'] = np.quantile(arr,[.25,.75]).tolist()
                endpoint[metric+'_change_1900_2000_percent_median'] = float(np.median([m[metric+'_change_1900_2000_percent'] for m in rows]))
            endpoints.append(endpoint)
    return stats, endpoints


def style(ax, baseline=0):
    ax.grid(axis='y', alpha=.18)
    if baseline is not None:
        ax.axhline(baseline, color='#999999', linewidth=.7, zorder=0)
    ax.set_xlim(1500, 2000)
    ax.set_xticks(np.arange(1500, 2001, 100))
    ax.set_xlabel('年份')
    # Keep the two endpoint tick labels within each panel.
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_ha('left')
        labels[-1].set_ha('right')


def grouped_plot(curves, members, stats, raw=False):
    by_id = {c['system']: c for c in curves}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), sharex=True, sharey='row')
    keys = [('length_km','长度 / km'),('volume_km3','体积 / km³')] if raw else [
        ('length_change_percent','长度相对1500年的变化 / %'),('volume_change_percent','体积相对1500年的变化 / %')]
    for col, (code, label, color) in enumerate(BANDS):
        ms = [m for m in members if m['elevation_group']==code]
        st = next(s for s in stats if s['cohort']=='all' and s['band']==code)
        retained = sum(m['status']=='retained' for m in ms)
        for row,(key,ylabel) in enumerate(keys):
            ax=axes[row,col]
            q=st[key]
            ax.fill_between(YEARS,q['q25'],q['q75'],color=color,alpha=.16,lw=0)
            for m in ms:
                c=by_id[m['system']]
                if raw:
                    y=np.array(c['native_years']); val=np.array(c['native_'+key])
                else:
                    metric=key.split('_')[0]
                    rawkey='length_km' if metric=='length' else 'volume_km3'
                    y=np.array(c['native_years']); val=100*(np.array(c['native_'+rawkey])/m[metric+'_1500']-1)
                ax.plot(y,val,color='#607582',alpha=.46,lw=.85,ls='-' if m['status']=='retained' else '--')
            ax.plot(YEARS,q['median'],color=color,lw=2.7,zorder=5)
            style(ax,baseline=None if raw else 0)
            if col==0: ax.set_ylabel(ylabel)
            if row==0: ax.set_title(f'{label}\n{len(ms)} 条：已认可 {retained} / 候选 {len(ms)-retained}',fontsize=12,pad=12)
    handles=[Line2D([],[],color='#607582',lw=1.2,label='单条已认可曲线'),
             Line2D([],[],color='#607582',lw=1.2,ls='--',label='单条候选曲线'),
             Line2D([],[],color='#286c9f',lw=2.7,label='组内中位数；阴影为25%–75%范围')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.944),ncol=3,frameon=False,fontsize=10)
    fig.suptitle('按 LIA 末端高程分组：51条冰川的'+('原始长度与体积' if raw else '长度与体积变化'),fontsize=17,y=.995)
    fig.text(.5,.013,'每条冰川一条代表曲线、等权；按静态LIA重建末端高程分组。阴影表示冰川间差异，并非置信区间。',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.84,bottom=.09,wspace=.08,hspace=.2)
    name='grouped_absolute' if raw else 'grouped_normalized'
    fig.savefig(OUT/(name+'.png'),dpi=170)
    if not raw: fig.savefig(OUT/(name+'.svg'))
    plt.close(fig)


def cohort_comparison(stats):
    fig,axes=plt.subplots(2,3,figsize=(15,8.3),sharex=True,sharey='row')
    for col,(cohort,title) in enumerate([('all','全部51条'),('retained','仅32条已认可'),('candidate','仅19条候选')]):
        for row,(key,ylabel) in enumerate([('length_change_percent','长度相对1500年的变化 / %'),('volume_change_percent','体积相对1500年的变化 / %')]):
            ax=axes[row,col]
            for code,label,color in BANDS:
                st=next(s for s in stats if s['cohort']==cohort and s['band']==code)
                q=st[key]
                ax.fill_between(YEARS,q['q25'],q['q75'],color=color,alpha=.09,lw=0)
                ax.plot(YEARS,q['median'],color=color,lw=2,label=f'{label} · n={st["n"]}')
            style(ax)
            if row==0:
                ax.set_title(title,fontsize=13,pad=10)
                ax.legend(frameon=False,fontsize=8,loc='upper left')
            if col==0: ax.set_ylabel(ylabel)
    fig.suptitle('高程组比较，并分别检查已认可与候选曲线',fontsize=17,y=.99)
    fig.text(.5,.018,'实线：组内中位数；阴影：25%–75%范围。三列共享坐标范围。分组结果是描述性比较，不分离地区、初态或参数方案的影响。',ha='center',fontsize=9,color='#555555')
    fig.subplots_adjust(top=.90,bottom=.10,hspace=.18,wspace=.08)
    fig.savefig(OUT/'cohort_comparison.png',dpi=170); plt.close(fig)


def individual_group_plots(curves,members):
    by_id={c['system']:c for c in curves}
    for code,label,_ in BANDS:
        ms=sorted([m for m in members if m['elevation_group']==code],key=lambda m:m['terminus_elevation_m'])
        fig,axes=plt.subplots(1,2,figsize=(15,6))
        colors=plt.get_cmap('turbo')(np.linspace(.04,.96,len(ms)))
        handles=[]
        for color,m in zip(colors,ms):
            c=by_id[m['system']]
            ls='-' if m['status']=='retained' else '--'
            for ax,key,metric in [(axes[0],'length_km','length'),(axes[1],'volume_km3','volume')]:
                y=np.asarray(c['native_years']); a=100*(np.asarray(c['native_'+key])/m[metric+'_1500']-1)
                ax.plot(y,a,color=color,lw=1.25,ls=ls)
                style(ax)
            handles.append(Line2D([],[],color=color,lw=1.5,ls=ls,label=f'{m["system"]} · {m["terminus_elevation_m"]:.0f} m'+(' · 候选' if m['status']=='candidate' else '')))
        axes[0].set_ylabel('长度相对1500年的变化 / %');axes[1].set_ylabel('体积相对1500年的变化 / %')
        fig.suptitle(f'{label}｜逐条曲线 · {len(ms)}条',fontsize=16,y=.99)
        fig.legend(handles=handles,ncol=5,loc='lower center',fontsize=8,frameon=False,bbox_to_anchor=(.5,.012))
        fig.subplots_adjust(top=.90,bottom=.28,wspace=.18)
        fig.savefig(OUT/f'{code}_individual.png',dpi=170);plt.close(fig)


def write_report(members,summary):
    df=pd.DataFrame(members).sort_values(['terminus_elevation_m','system'])
    display=df[['system','status_zh','terminus_elevation_m','elevation_group','length_1500','length_2000',
                'length_change_1500_2000_percent','volume_1500','volume_2000','volume_change_1500_2000_percent',
                'native_step_years_max','distance_to_nearest_group_boundary_m']].copy()
    display.elevation_group=display.elevation_group.map(dict((k,l) for k,l,_ in BANDS))
    for c in display.select_dtypes(include='number'):
        if c!='system': display[c]=display[c].map(lambda v:f'{v:.2f}')
    display.columns=['LIA编号','状态','LIA末端高程 m','分组','1500长度 km','2000长度 km','长度变化 %',
                     '1500体积 km³','2000体积 km³','体积变化 %','原始最大间隔 年','距分组边界 m']
    table=display.to_html(index=False,table_id='members',escape=True)
    ep=pd.DataFrame([r for r in summary['endpoints'] if r['cohort']=='all'])
    ep=ep[['label','n','retained','candidates','length_change_1500_2000_percent_median','volume_change_1500_2000_percent_median']]
    ep.columns=['高程组','条数','已认可','候选','1500–2000长度变化中位数 %','1500–2000体积变化中位数 %']
    for c in ep.columns[-2:]:ep[c]=ep[c].map(lambda v:f'{v:+.1f}')
    near=summary['within_50m_of_boundary']
    neartext='、'.join(str(x['system']) for x in near) or '无'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>末端高程分组曲线 · 51条冰川</title>
<style>body{font-family:"Microsoft YaHei",sans-serif;background:#f7f9fa;color:#243743;margin:32px auto;max-width:1500px;padding:0 20px}h1{font-size:27px}h2{font-size:20px;margin-top:36px}p{line-height:1.85;max-width:1180px}img{width:100%;background:white;border:1px solid #dde4e8}table{border-collapse:collapse;background:white;font-size:13px;white-space:nowrap}th,td{padding:10px 12px;border-bottom:1px solid #e1e7eb;text-align:right}th{background:#254d62;color:white;position:sticky;top:0}.scroll{overflow:auto;max-height:630px}a{color:#086d97}input{padding:10px;width:240px;margin:12px 0;border:1px solid #aabac5;border-radius:4px}details{margin:18px 0}li{line-height:1.85}</style>
<h1>按 LIA 末端高程分组看曲线</h1><p>分析范围：32条已认可曲线＋19条达到旧复核门槛的候选，共51条。每条冰川只取一条代表曲线，组内等权。固定分界为4000 m和4500 m，没有依据曲线表现调整分界。</p>
<p><a href="grouped_normalized.png">下载主图 PNG</a> · <a href="grouped_normalized.svg">主图矢量 SVG</a> · <a href="members.json">分组名单及逐条指标</a> · <a href="curves.json">完整绘图数据</a> · <a href="source_manifest.json">来源校验清单</a></p>
<img src="grouped_normalized.png" alt="三组冰川长度和体积变化曲线">
<p>纵轴为100×[X(t)/X(1500)−1]；每条曲线的1500年值为零。细实线是已认可冰川，细虚线是候选；粗线是组内中位数，阴影是冰川间25%–75%范围，<b>不是置信区间或参数不确定性</b>。这里的组是静态LIA末端高程分组，未随模拟年份改变。</p>
<h2>1500–2000年变化</h2>'''+ep.to_html(index=False,escape=True)+'''
<h2>分开检查32条已认可与19条候选</h2><img src="cohort_comparison.png" alt="三高程组在全部、已认可和候选中的曲线比较">
<p>初步观察：三组内部差异明显；目前没有稳定的“末端越低、退缩越强”的单调关系。仅看32条已认可曲线，低、中、高组1500–2000年长度变化中位数分别约为−9.6%、−15.3%、−9.5%，体积变化约为−10.5%、−15.4%、−15.5%。这与混合51条后的比较不同，因此样本状态需要保留。</p>
<p>这批曲线存在地区、几何、初态及参数方案差异。高程组间的形状差别是描述性结果，不能单独归因于末端高程。候选曲线也不因参与分组而自动成为正式接受曲线。</p>
<h2>逐条查看</h2>'''
    for code,label,_ in BANDS:
        doc+=f'<details><summary>{label}：显示有编号的逐条曲线</summary><img src="{code}_individual.png" alt="{label}逐条曲线"></details>'
    doc+='''<details><summary>显示原始长度（km）与体积（km³）</summary><img src="grouped_absolute.png" alt="原始长度体积分组曲线"></details>
<h2>分组名单与指标</h2><input id="q" placeholder="搜索冰川编号或状态"><span id="count"></span><div class="scroll">'''+table+'''</div>
<h2>数据口径与核对</h2><ul>
<li>末端高程沿用上一版属性清单：对LIA空间主流线终点，在成对V3/F19重建栅格上取最近有效像元；不是现代RGI最低高程，也不是各年模拟末端高程。没有使用存在不一致记录的网络reference_surface字段。</li>
<li>32条采用既有保留名单对应的登记曲线；15条候选采用原旧复核通过组合中评分最高的一条；另外4条沿用已有展示页的已选曲线。没有重算接受门槛、重新拟合参数或改变认可状态。</li>
<li>候选曲线与原生保存状态的年份、长度和体积逐项核对，来源哈希验证通过。32条从原始时间序列重新对齐，与既有年度汇总闭合。</li>
<li>部分旧曲线原生间隔为10年；单条线按原始点连接，组中位数与四分位数在线性插值到共同年度坐标后计算，不把插值当作新增模拟输出；不额外平滑。</li>
<li>距4000或4500 m分界不足50 m的冰川：'''+neartext+'''。50 m只是边界敏感性标记，并非测量误差估计；逐条高程和距边界距离列在表中。</li>
<li>本次只分析已保存结果，没有启动新OGGM模拟，也没有修改控制器、认可名单或原始结果。</li></ul>
<script>const rows=[...document.querySelectorAll('#members tbody tr')];function f(){let q=document.querySelector('#q').value.toLowerCase(),n=0;rows.forEach(r=>{let ok=r.textContent.toLowerCase().includes(q);r.hidden=!ok;if(ok)n++});document.querySelector('#count').textContent=' 显示 '+n+' 条'}document.querySelector('#q').oninput=f;f();</script></html>'''
    (OUT/'index.html').write_text(doc,encoding='utf-8')


def main():
    curves,members,sampling=load_sources()
    stats,endpoints=group_statistics(curves,members)
    summary=dict(created_at=datetime.now(timezone.utc).isoformat(),system_count=51,retained=32,candidates=19,
                 elevation_definition='static LIA mapped-mainline terminus elevation sampled from paired V3/F19 raster',
                 bin_edges_m=[4000,4500],bin_intervals=['z < 4000','4000 <= z < 4500','z >= 4500'],
                 baseline_year=1500,final_year=2000,weighting='one representative curve per glacier; equal glacier weights',
                 endpoints=endpoints,interpolated_source_systems=[m['system'] for m in members if m['interpolated_to_annual']],
                 within_50m_of_boundary=[dict(system=m['system'],elevation_m=m['terminus_elevation_m'],distance_m=m['distance_to_nearest_group_boundary_m']) for m in members if m['distance_to_nearest_group_boundary_m']<50],
                 new_simulations=0,acceptance_changes=0,source_files_modified=0)
    grouped_plot(curves,members,stats)
    grouped_plot(curves,members,stats,raw=True)
    cohort_comparison(stats)
    individual_group_plots(curves,members)
    write_report(members,summary)
    for path,digest in SOURCES.items():
        assert sha(ROOT/path)==digest, f'Source changed during analysis: {path}'
    json_write('members.json',members)
    json_write('curves.json',curves)
    json_write('group_statistics.json',stats)
    json_write('summary.json',summary)
    json_write('native_sampling.json',sampling)
    json_write('source_manifest.json',SOURCES)
    print(json.dumps(dict(status='complete',source_count=len(SOURCES),endpoints=endpoints,
                          interpolated_systems=summary['interpolated_source_systems'],
                          boundary_systems=summary['within_50m_of_boundary']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
