"""Read-only grid/time-step evidence and conditional millennial cost report."""
from .millennium_cost_v1 import R,RUN,REPORT,I,SEED,read,arrays
from .v3.runner import write_json,digest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
 r=read(RUN/'result.json');REPORT.mkdir(parents=True,exist_ok=False);cases=r['cases'];native=next((c for c in cases if c['status']=='completed' and c['grid']['cell_m']==15.75),None);coarse=[c for c in cases if c['status']=='completed' and c['grid']['cell_m']!=15.75]
 plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
 rows=[]
 for c in coarse:
  g=c['grid'];m=c['monthly_vs_daily_error'];reasons=[]
  if abs(g['initial_area_change_relative'])>.05:reasons.append('面积变化超过 5%')
  if g['spatial_velocity_comparison']['velocity_relative_L2']>.1:reasons.append('速度加权差异超过 10%')
  if g['peak_relative_error']>.2:reasons.append('峰值差异超过 20%')
  if g['components']!=g['native_components']:reasons.append('连通分量改变')
  rows.append(dict(grid_m=g['cell_m'],nodes=g['nodes'],area_change_percent=g['initial_area_change_relative']*100,volume_remap_relative=g['volume_remap_relative'],spatial_velocity_error_percent=g['spatial_velocity_comparison']['velocity_relative_L2']*100,spatial_projected_peak_error_percent=g['peak_relative_error']*100,spatial_screen_pass=g['spatial_screen_pass'],spatial_fail_reasons=reasons,initial_CFL_days=c['daily']['initial_CFL_days'],daily_30d_seconds=c['daily']['wall_seconds'],monthly_30d_seconds=c['monthly']['wall_seconds'],monthly_H_error_m=m['H_max_m'],monthly_velocity_error_percent=m['velocity_relative_L2']*100,monthly_screen_pass=c['monthly_screen_pass'],millennium_daily_hours=c['millennium_daily_hours'],millennium_monthly_hours=c['millennium_monthly_hours'],final_daily_max_speed=c['daily']['timeline'][-1]['max_speed_m_year'],daily_budget=c['daily']['checks']['mass_relative'],monthly_budget=c['monthly']['checks']['mass_relative']))
 summary=dict(status=r['status'],native_daily_window=native,coarse=rows,failures=[c for c in cases if c['status']!='completed'],wall_seconds=r['wall_seconds'],cost_interpretation='1000year columns extrapolate the measured30day case,including checkpoints and BP,excluding initial grid solve,climate/thermal coupling and long-term geometry changes. Monthly cadence comparison uses same-grid daily reference,not exact truth. Uniform coarse grids are not accepted unless all frozen spatial gates pass.',selection='Retain native geometry and passed daily-window result. Use coarse cost evidence to design boundary-preserving coarsening;no failed spatial gate waived.' if not any(x['spatial_screen_pass'] for x in rows) else 'A grid passed the declared spatial screen;long-time settings remain bounded by temporal evidence.')
 write_json(REPORT/'summary.json',summary)
 if rows:
  fig,ax=plt.subplots(1,3,figsize=(15,5),layout='constrained');xx=np.arange(len(rows));labels=[f"{v['grid_m']:g} m" for v in rows]
  ax[0].bar(xx-.17,[v['millennium_daily_hours'] for v in rows],width=.34,label='每日更新 BP',color='#6686a3');ax[0].bar(xx+.17,[v['millennium_monthly_hours'] for v in rows],width=.34,label='每 30 天更新 BP',color='#0795a1');ax[0].set_yscale('log');ax[0].set(xticks=xx,xticklabels=labels,ylabel='估算计算小时（对数轴）',title='1000 年条件成本估算');ax[0].legend(fontsize=9)
  ax[1].bar(labels,[v['spatial_velocity_error_percent'] for v in rows],color='#0795a1');ax[1].axhline(10,color='#d0524b',ls='--',label='预设 10%');ax[1].set(ylabel='面积加权速度向量差异（%）',title='粗网格与细网格投影比较');ax[1].legend(fontsize=9)
  ax[2].bar(labels,[v['monthly_H_error_m'] for v in rows],color='#0795a1');ax[2].axhline(.1,color='#d0524b',ls='--',label='预设 0.1 m');ax[2].set(ylabel='30 天终态冰厚最大差值（m）',title='月更新与日更新比较');ax[2].legend(fontsize=9)
  fig.savefig(REPORT/'comparison.png',dpi=145);plt.close(fig)
 # Same-scale actual BP speed maps; no interpolation across grid boundaries.
 meshes=[arrays(I/'mesh.npz')];vels=[arrays(SEED/'frame-000/state.npz')['u']];titles=['原始 15.75 m']
 for c in coarse:
  folder=RUN/f"grid-{c['grid']['cell_m']:g}m";meshes.append(arrays(folder/'mesh.npz'));vels.append(arrays(folder/'frame-000/state.npz')['u']);titles.append(f"均匀粗化 {c['grid']['cell_m']:g} m")
 fig,axs=plt.subplots(1,len(meshes),figsize=(6*len(meshes),5.5),layout='constrained');axs=np.atleast_1d(axs);vmax=max(float(np.linalg.norm(u[:,-1],axis=1).max()) for u in vels)
 x0=float(meshes[0]['xy'][:,0].min());y0=float(meshes[0]['xy'][:,1].min())
 for ax,mesh,u,title in zip(axs,meshes,vels,titles):
  im=ax.tripcolor((mesh['xy'][:,0]-x0)/1000,(mesh['xy'][:,1]-y0)/1000,mesh['triangles'],np.linalg.norm(u[:,-1],axis=1),cmap='viridis',vmin=0,vmax=vmax,shading='gouraud',rasterized=True);ax.set(aspect='equal',title=title,xlabel='东西距离（km）',ylabel='南北距离（km）');fig.colorbar(im,ax=ax,fraction=.035,label='表面速度（m/年）')
 fig.savefig(REPORT/'speed-maps.png',dpi=125);plt.close(fig)
 table=''
 for v in rows:
  spatial='通过' if v['spatial_screen_pass'] else '；'.join(v['spatial_fail_reasons']);temporal='通过' if v['monthly_screen_pass'] else '未通过'
  table+=f"<tr><td>{v['grid_m']:g} m</td><td>{v['nodes']:,}</td><td>{v['area_change_percent']:+.2f}%</td><td>{v['spatial_velocity_error_percent']:.2f}%</td><td>{v['monthly_H_error_m']:.4f} m</td><td>{v['monthly_velocity_error_percent']:.3f}%</td><td>{v['monthly_30d_seconds']:.2f} 秒</td><td>{v['millennium_monthly_hours']:.1f} 小时</td><td>{spatial}<br>时间步长：{temporal}</td></tr>"
 native_text=''
 if native:
  ne=native['native_day_error'];native_text=f"细网格的初始输运 CFL 上限为 {native['daily_window']['initial_CFL_days']:.2f} 天。改为一天一个输运窗口和一次 BP 后，最大冰厚偏差 {ne['H_max_m']*1000:.3f} 毫米、最大速度偏差 {ne['velocity_max_m_year']:.6f} 米/年；对照预设标准{'通过' if native['native_day_pass'] else '未通过'}。CFL 约束保证的是输运步长限制，并不独立保证耦合时间精度。"
 verdict='均匀粗网格尚未通过全部空间标准。下一步应保留细网格边界、采用较粗内部网格；这次数据用于确定计算成本范围，不能直接当作已验证的千年结果。' if not any(v['spatial_screen_pass'] for v in rows) else '已有网格通过本次空间标准；继续延长时段时，根据几何变化检查更新间隔。'
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>千年模拟 · 网格与时间步长成本</title><style>body{{font:17px Microsoft YaHei,sans-serif;background:#eef3f8;color:#20384b;margin:0}}main{{max-width:1550px;margin:auto;padding:28px}}p{{line-height:1.9}}img{{width:100%;background:white;border-radius:12px;margin:18px 0}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:13px;border-bottom:1px solid #dce4ed;text-align:left}}.scroll{{overflow:auto}}a{{color:#087e9e}}.note{{background:white;border-left:5px solid #0795a1;padding:18px}}small{{color:#566d7e}}</style><main><h1>面向 1000 年：先测算网格与更新时间的成本</h1><p>{native_text}</p><p class="note">{verdict}</p><div class="scroll"><table><tr><th>网格</th><th>节点</th><th>面积变化</th><th>空间速度差异</th><th>月/日冰厚差</th><th>月/日速度差</th><th>模拟 30 天<br>月更新耗时</th><th>1000 年<br>条件估算</th><th>判定</th></tr>{table}</table></div><img src="comparison.png" alt="千年成本、空间速度差异及时间步长误差"><img src="speed-maps.png" alt="细网格与粗网格速度分布对照"><p>粗化保持初始总体积，冰厚采用分块体积汇总；床面、材料和初始速度同时按已登记规则投影。覆盖任何冰区的粗格都会被保留，因此边缘会扩展。本次明确统计了面积变化，速度比较使用细网格投影到粗格的参考场。</p><p>所有千年耗时均从 30 天试算线性外推，包含此次保存与求解开销；未计入气候、热力耦合和长期几何变化造成的成本变化。日更新仅作为月更新的时间精度参考。当前模型边界范围固定，冰可流出；千年推进还需要覆盖可能的冰川进退范围。</p><p><a href="summary.json">完整数值与各项标准</a> · <a href="../bp-cadence-6h-v1/index.html">此前六小时对照</a> · <a href="../system2387-working-v1/index.html">三维冰面工作页</a></p></main></html>'''
 (REPORT/'index.html').write_text(page,encoding='utf-8');write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},result_sha256=digest(RUN/'result.json')));print(json_safe(summary))

def json_safe(s):
 return {k:v for k,v in s.items() if k!='native_daily_window'}

if __name__=='__main__':main()
