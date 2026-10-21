"""Offline moving-front report. Zoom affects plotting only, never model extent."""
import base64
import html
import json
from pathlib import Path
import numpy as np


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve(); folders=sorted(root.glob('step-*'))
    with np.load(folders[0]/'state.npz') as a: initial={k:a[k].copy() for k in a.files}
    with np.load(folders[-1]/'state.npz') as a: final={k:a[k].copy() for k in a.files}
    meta=json.loads((folders[-1]/'checkpoint.json').read_text()); manifest=json.loads((root/'run.json').read_text())
    history=meta['history']; last=history[-1]; config=manifest['payload']['config']
    active=np.flatnonzero((initial['h']>0)|(final['h']>=1.))
    last_index=min(len(final['h'])-1,(int(active[-1]) if len(active) else 1)+15)
    take=slice(0,last_index+1); x=final['x'][take]/1000; bed=final['bed'][take]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    ax=axes[0,0]; ax.plot(x,bed,color='#8b7765',label='Bed')
    ax.plot(x,bed+initial['h'][take],'--',label='Initial surface'); ax.plot(x,bed+final['h'][take],label='Final surface')
    ax.set(title='Full source geometry retained; ice-region zoom',xlabel='Distance (km)',ylabel='Elevation (m)'); ax.legend()
    sigma=np.linspace(0,1,final['E'].shape[1]); z=bed[:,None]+final['h'][take,None]*sigma
    xx=np.broadcast_to(x[:,None],z.shape)
    for ax,key,title,cmap in ((axes[0,1],'u','Velocity (m/year)','viridis'),(axes[1,0],'temperature','Ice temperature (C)','coolwarm')):
        values=final[key][take].copy()-(273.15 if key=='temperature' else 0)
        values=np.ma.masked_where(np.broadcast_to(final['h'][take,None]==0,values.shape),values)
        mesh=ax.pcolormesh(xx,z,values,shading='gouraud',cmap=cmap); fig.colorbar(mesh,ax=ax)
        ax.set(title=title,xlabel='Distance (km)',ylabel='Elevation (m)')
    ax=axes[1,1]; t=[r['clock_years'] for r in history]
    ax.plot(t,[r['ice_cells'] for r in history],label='H > 0',marker='.')
    ax.plot(t,[r['ice_cells_ge_1m'] for r in history],label='H >= 1 m',marker='.')
    ax.set(title='Ice support (thin numerical tails shown separately)',xlabel='Time (years)',ylabel='Ice-covered cells'); ax.legend()
    fig.suptitle(config['case_id']+' | '+manifest['payload']['runtime']['device'].upper(),fontsize=15)
    fig.savefig(root/'overview.png',dpi=150); plt.close(fig)
    png=base64.b64encode((root/'overview.png').read_bytes()).decode()
    maxmass=max(abs(r['mass_error_m3']) for r in history)
    wet=int(last['ice_cells']); whole=len(final['h'])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>第三版冰缘耦合报告</title><style>body{{max-width:1150px;margin:40px auto;padding:0 24px;background:#f4f7fb;color:#17233b;font:16px/1.7 system-ui}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:white;border-radius:12px;padding:20px}}b{{display:block;font-size:25px}}img{{width:100%;margin:24px 0;border-radius:12px}}a{{color:#17689a}}code{{overflow-wrap:anywhere}}</style>
<h1>第三版 · 冰缘与热力耦合</h1><p>{html.escape(config['case_id'])} · {manifest['payload']['runtime']['device']} · {meta['step']} 步</p>
<div class="cards"><div class="card"><b>{last['clock_years']:.6g} 年</b>实际模拟时间</div><div class="card"><b>{whole} 格</b>原始物理域全部保留</div><div class="card"><b>{last['bp_nodes']} 格</b>当前 BP 动量求解域</div><div class="card"><b>{wet} 格</b>含冰网格（包括薄冰尾部）</div><div class="card"><b>{maxmass:.3e} m³</b>最大累计质量误差</div><div class="card"><b>{last['max_speed_m_year']:.4g} m/年</b>最大流速</div></div>
<img src="data:image/png;base64,{png}" alt="冰川地形、流速、冰温及有冰区域演化">
<p>冰厚不设人为下限，原有无冰区和地形保留。消融受可用冰量限制，无法实现的潜在消融单独记账。小于 1 m 的薄冰及不足三格的小冰块采用局地 SIA；薄冰温度采用表面温度极限。图中区分 H&gt;0 与 H≥1 m，薄冰数值尾部不当作观测冰川长度。</p>
<p>气候推导的初始冰温与 0.05 W/m² 地热通量是用户确认的实验假设。水平焓通量随冰量守恒运输，表面积累采用列内混合近似。该短程技术试算不等于平衡态、历史重建或 LIA 科学验收。</p>
<p>完整结果：<code>{html.escape(str(folders[-1]))}</code></p></html>'''
    (root/'index.html').write_text(text,encoding='utf-8'); return str(root/'index.html')
