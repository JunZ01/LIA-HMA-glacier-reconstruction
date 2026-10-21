"""Offline report generation: never needed for numerical continuation."""
from pathlib import Path
import html
import json
import numpy as np

def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root)
    latest=sorted(root.glob('step-*'))[-1]
    meta=json.loads((latest/'checkpoint.json').read_text())
    manifest=json.loads((root/'run.json').read_text())
    with np.load(latest/'state.npz',allow_pickle=False) as d:
        x,bed,h,u=d['x'],d['bed'],d['thickness'],d['u']
    xs=np.r_[x[0],(x[:-1]+x[1:])/2,x[-1]]
    bs=np.r_[bed[0],(bed[:-1]+bed[1:])/2,bed[-1]]
    hs=np.r_[h[0],(h[:-1]+h[1:])/2,h[-1]]
    z=bs[None,:]+np.linspace(0,1,u.shape[0])[:,None]*hs
    fig,axes=plt.subplots(2,1,figsize=(10,7),constrained_layout=True)
    contour=axes[0].contourf(np.broadcast_to(xs/1000,u.shape),z,u,24,cmap='viridis')
    axes[0].plot(x/1000,bed,color='#543e2c',lw=2)
    axes[0].set(xlabel='Along-flow distance (km)',ylabel='Elevation (m)',title='Synthetic PoLIM CPU prototype | horizontal velocity')
    fig.colorbar(contour,ax=axes[0],label='m/year')
    history=meta['history']
    axes[1].plot([r['clock_years'] for r in history],[r['volume_m3']/1e6 for r in history],'o-',color='#126f86')
    axes[1].set(xlabel='Time (years)',ylabel='Ice volume (million m3)',title='Saved trajectory')
    fig.savefig(root/'result.png',dpi=150); plt.close(fig)
    rows=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ('step','clock_years','iterations','velocity_converged','momentum_relative_residual','cumulative_mass_error_m3'))+'</tr>' for r in history)
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PoLIM CPU 原型</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#203040}}img{{width:100%}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;width:100%;font-size:13px}}small{{color:#526575}}</style>
<h1>PoLIM CPU 原型</h1><p>{html.escape(manifest['config']['case_id'])} · 已保存 {meta['step']} 个演化步 · {meta['clock_years']:.6g} 年</p>
<p>当前范围：等温、恒定宽度、x–z 高阶速度、封闭两端、无滑动或线性底摩擦。每步可恢复。数值原型结果，尚未通过真实冰川科学验证。</p>
<img src="result.png" alt="速度剖面与冰体积"><h2>集中检查</h2><table><tr><th>步</th><th>年</th><th>迭代</th><th>速度收敛</th><th>动量相对残差</th><th>累计质量差 m³</th></tr>{rows}</table>
<p><small>温度、焓、水文、变宽流带、真实冰川映射与 GPU 尚未接入。参考 PoLIM 提交 {manifest['upstream_commit']}。迭代控制及守恒演化是本原型新增实现。</small></p></html>'''
    (root/'index.html').write_text(content,encoding='utf-8')
    return root/'index.html'
