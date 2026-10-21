"""Read-only cost and accuracy comparison of the six-hour BP update."""
from .bp_cadence_6h_v1 import R,I,RUN,REPORT,BASE,read,arrays
from .v3.runner import write_json,digest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
 r=read(RUN/'result.json');c=r['comparison'];assert r['status']=='completed'
 REPORT.mkdir(parents=True,exist_ok=False)
 speedup=c['baseline_wall_seconds']/r['numerical_wall_seconds'];yearhours=r['numerical_wall_seconds']*365/3600
 summary=dict(comparison=c,baseline_seconds=c['baseline_wall_seconds'],six_hour_seconds=r['numerical_wall_seconds'],total_seconds_including_comparison=r['wall_seconds'],observed_speedup=speedup,linear_year_hours=yearhours,checks=r['checks'],final_max_speed_m_year=r['timeline'][-1]['max_speed_m_year'],final_H_error_m=c['all_hours'][-1]['H_max_abs_m'],final_velocity_vector_error_m_year=c['all_hours'][-1]['surface_velocity_max_vector_error_m_year'],future_working_BP_interval_hours=6 if c['status']=='pass' else None,timing_scope='Same25hourly sealed outputs;new numerical time excludes extra paired comparison and final source rehash,baseline archived full run timing includes its verification. Different launch times;not a controlled benchmark. Annual time is linear extrapolation,not measured.')
 write_json(REPORT/'summary.json',summary)
 plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
 rows=c['all_hours'];hours=[v['hour'] for v in rows]
 fig,axs=plt.subplots(1,3,figsize=(15,4.6),layout='constrained')
 axs[0].plot(hours,[v['H_max_abs_m']*1000 for v in rows],marker='.',color='#087e9e');axs[0].set(title='冰厚最大差值',ylabel='毫米',xlabel='模拟时间（小时）')
 axs[1].plot(hours,[v['surface_velocity_max_vector_error_m_year'] for v in rows],marker='.',color='#087e9e');axs[1].set(title='表面速度向量最大差值',ylabel='米/年',xlabel='模拟时间（小时）')
 for h in [6,12,18,24]:axs[1].axvline(h,color='#a5b2ba',lw=.7,ls='--')
 values=[c['baseline_wall_seconds'],r['numerical_wall_seconds']];bars=axs[2].bar(['逐小时 BP','每 6 小时 BP'],values,color=['#8796a9','#087e9e']);axs[2].set(title='模拟一天的实际耗时',ylabel='秒');axs[2].set_ylim(0,max(values)*1.18)
 for bar,val in zip(bars,values):axs[2].text(bar.get_x()+bar.get_width()/2,val+5,f'{val:.1f} 秒',ha='center')
 for ax in axs[:2]:ax.grid(alpha=.2);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
 fig.savefig(REPORT/'comparison.png',dpi=145);plt.close(fig)
 mesh=arrays(I/'mesh.npz');g=arrays(I/'geometry.npz');a=arrays(RUN/'real/frame-024/state.npz');b=arrays(BASE/'real/frame-024/state.npz');diff=a['h']-b['h'];z=np.full(g['lia_mask'].shape,np.nan);z[mesh['pixel_rows'],mesh['pixel_cols']]=diff[mesh['pixel_center_nodes']]*1000;lim=float(np.nanmax(abs(z)))
 fig,ax=plt.subplots(figsize=(10,7),layout='constrained');dx=abs(g['x'][1]-g['x'][0]);ext=[0,len(g['x'])*dx/1000,0,len(g['y'])*dx/1000];im=ax.imshow(z,origin='upper',extent=ext,vmin=-lim,vmax=lim,cmap='RdBu_r',interpolation='nearest');plt.colorbar(im,ax=ax,label='6 小时方案 − 逐小时方案（毫米）');ax.set(title='24 小时终态 · 冰厚差值',xlabel='东西距离（km）',ylabel='南北距离（km）');fig.savefig(REPORT/'thickness-error.png',dpi=130);plt.close(fig)
 outcome='通过本次单日误差标准，可作为下一阶段的工作设置' if c['status']=='pass' else '未通过本次误差标准，保留逐小时设置'
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BP 更新频率与速度优化</title><style>body{{font:17px Microsoft YaHei,sans-serif;margin:0;background:#eef3f8;color:#20384b}}main{{max-width:1350px;padding:28px;margin:auto}}p{{line-height:1.8}}img{{width:100%;border-radius:12px;background:white;margin:16px 0}}.cards{{display:flex;gap:15px;flex-wrap:wrap}}.card{{background:white;padding:20px;border-radius:12px;flex:1;min-width:240px}}b{{color:#087e9e}}.card b{{display:block;font-size:26px;margin:12px 0}}a{{color:#087e9e}}small{{color:#5d7181}}</style><main><h1>BP 从每天 24 次，减少到 4 次</h1><p>{outcome}。冰厚仍逐小时输出，并受原 CFL 稳定性条件约束；两次 BP 更新之间沿用上次速度。</p><div class="cards"><div class="card">模拟一天实际耗时<b>{c['baseline_wall_seconds']:.1f} → {r['numerical_wall_seconds']:.1f} 秒</b>观测加速约 {speedup:.2f} 倍</div><div class="card">全部时刻最大冰厚偏差<b>{c['maximum_H_error_m']*1000:.3f} 毫米</b>本次预设上限 10 毫米</div><div class="card">全部时刻最大速度向量偏差<b>{c['maximum_surface_velocity_vector_error_m_year']:.5f} 米/年</b>本次预设上限 0.05 米/年</div></div><img src="comparison.png" alt="误差时间曲线与模拟耗时"><p>蓝线逐小时对照已保存的参考结果。速度差呈锯齿形：沿用旧速度时差异增大，重新求解 BP 后回落。24 小时终态冰厚最大差值为 {summary['final_H_error_m']*1000:.3f} 毫米，面积加权速度向量相对误差在全部时刻最高为 {c['maximum_velocity_area_relative_L2']*100:.5f}%。材料与冰床相同，积累消融为零，质量守恒通过。</p><p>若按这次耗时线性换算，模拟一年约 <b>{yearhours:.1f} 小时</b>。这是单日结果的估算；长期冰面变化后，求解难度和误差可能变化。</p><small>耗时对比均包含 25 个逐小时状态保存；新耗时不含额外的新旧结果逐点对照，完整运行含该对照为 {r['wall_seconds']:.1f} 秒。基准采用此前运行记录，未重复计算。此次改变的是速度更新时间间隔，未更改 BP 方程或材料参数；暂未启用自适应更新。</small><details><summary>查看终态冰厚差值分布</summary><img src="thickness-error.png" alt="24小时终态冰厚差值"></details><p><a href="../independent-outlet-day1-v1/index.html">原逐小时演化动画</a> · <a href="../system2387-working-v1/index.html">三维冰面工作页</a> · <a href="summary.json">完整数值摘要</a></p></main></html>'''
 (REPORT/'index.html').write_text(page,encoding='utf-8')
 write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},source_result_sha256=digest(RUN/'result.json')))
 print({k:v for k,v in summary.items() if k!='comparison'})

if __name__=='__main__':main()
