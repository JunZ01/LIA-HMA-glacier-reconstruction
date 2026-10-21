"""Offline scientific plots; optional and independent of committed model state."""
from pathlib import Path
import base64
import html
import json
import numpy as np


def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve(); folder=sorted(root.glob('step-*'))[-1]
    manifest=json.loads((root/'run.json').read_text()); meta=json.loads((folder/'checkpoint.json').read_text())
    with np.load(folder/'state.npz') as data: state={k:data[k] for k in data.files}
    history=meta['history']; last=history[-1]
    x=state['x']; h=state['h']; bed=state['bed']; sigma=np.linspace(0,1,state['E'].shape[1])
    z=bed[:,None]+h[:,None]*sigma[None,:]; xx=np.broadcast_to(x[:,None],z.shape)/1000
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for ax,key,title,cmap in zip(axes.flat[:3],('u','temperature','water_fraction'),
                                ('Velocity (m/year)','Temperature (C)','Liquid water fraction'),('viridis','coolwarm','Blues')):
        values=state[key]-(273.15 if key=='temperature' else 0)
        limits={'vmin':0.,'vmax':.03} if key=='water_fraction' else {}
        mesh=ax.pcolormesh(xx,z,values,shading='gouraud',cmap=cmap,**limits)
        ax.plot(x/1000,bed,color='#49352c',lw=1); ax.set(xlabel='Distance (km)',ylabel='Elevation (m)',title=title)
        fig.colorbar(mesh,ax=ax)
    ax=axes[1,1]; times=[v['clock_years'] for v in history]
    ax.plot(times,[v['max_speed_m_per_year'] for v in history],marker='o',color='#2563eb')
    ax.set(xlabel='Time (years)',ylabel='Maximum speed (m/year)',title='Coupled evolution')
    fig.suptitle(f"{manifest['payload']['config']['case_id']} | {manifest['payload']['runtime']['backend'].upper()} | step {meta['step']}",fontsize=16)
    fig.savefig(root/'overview.png',dpi=150); plt.close(fig)
    png=base64.b64encode((root/'overview.png').read_bytes()).decode()
    unit=manifest['payload']['geometry_provenance'].get('unit_width',False)
    note=manifest['payload']['geometry_provenance'].get('note','Synthetic development case.')
    cards=[('模拟时间',f"{last['clock_years']:.6g} 年"),('最大流速',f"{last['max_speed_m_per_year']:.6g} m/年"),
           ('动量残差',f"{last['momentum_relative_residual']:.3e}"),('累计质量误差',f"{last['mass_error_m3']:.3e} m³"),
           ('计算体积'+('（单位宽度）' if unit else ''),f"{last['volume_m3']:.8g} m³"),('最大含水率',f"{last['maximum_water_fraction']:.3%}")]
    cells=''.join(f'<div class="card"><small>{html.escape(k)}</small><strong>{html.escape(v)}</strong></div>' for k,v in cards)
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PoLIM 第二版运行报告</title>
<style>body{{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#17223b;background:#f5f7fb}}h1{{margin-bottom:4px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{padding:18px;background:white;border-radius:12px}}small{{display:block;color:#566377}}strong{{font-size:23px}}img{{width:100%;background:white;border-radius:12px;margin:24px 0}}p{{max-width:1000px}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{.cards{{grid-template-columns:1fr 1fr}}}}</style>
<h1>冰川热力与流动耦合 · 第二版</h1><p>{html.escape(manifest['payload']['config']['case_id'])} · {html.escape(manifest['payload']['runtime']['backend'])} · 已完成 {meta['step']} 步</p>
<div class="cards">{cells}</div><img src="data:image/png;base64,{png}" alt="冰川剖面流速、温度、含水率与演化曲线">
<p>{html.escape(note)}</p><p>本报告来自可运行的实验模型。使用新的 Q1 有限元动量离散、焓输运和温度反馈；端点封闭。尚未获得 MATLAB 全模型等价验证或 LIA 科学验收。</p>
<p>完整状态包含焓、温度、含水率、速度、冰厚、融化和模型时钟。断点：<code>{html.escape(str(folder))}</code></p></html>'''
    (root/'index.html').write_text(content,encoding='utf-8')
    return str(root/'index.html')
