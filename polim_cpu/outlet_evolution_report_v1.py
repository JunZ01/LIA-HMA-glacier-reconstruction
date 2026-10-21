"""Read-only hourly maps and time series for the independent-outlet continuation."""
from .outlet_evolution_v1 import R,I,RUN,REPORT,arrays,read
from .v3.runner import digest,write_json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
 result=read(RUN/'result.json');assert result['status']=='completed'
 REPORT.mkdir(parents=True,exist_ok=False)
 mesh=arrays(I/'mesh.npz');g=arrays(I/'geometry.npz');tl=result['timeline'];states=[arrays(RUN/f'real/frame-{k:03d}/state.npz') for k in range(25)]
 rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];shape=g['lia_mask'].shape
 def grid(a):
  z=np.full(shape,np.nan);z[rows,cols]=a[nodes];return z
 H=[grid(s['h']) for s in states];V=[grid(np.linalg.norm(s['u'][:,-1],axis=1)) for s in states];D=[h-H[0] for h in H]
 limit=max(float(np.nanmax(abs(d))) for d in D);vlim=max(float(np.nanmax(v)) for v in V);hlim=max(float(np.nanmax(h)) for h in H)
 plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
 dx=abs(g['x'][1]-g['x'][0]);x0=g['x'].min()-dx/2;y0=g['y'].min()-dx/2;extent=[0,(g['x'].max()+dx/2-x0)/1000,0,(g['y'].max()+dx/2-y0)/1000]
 X=(g['x']-x0)/1000;Y=(g['y']-y0)/1000;A=np.array([-2139.7096677023073,3159517.8405094133]);axp=(A[0]-x0)/1000;ayp=(A[1]-y0)/1000
 def show(ax,z,title,cmap,vmin,vmax,label):
  im=ax.imshow(z,origin='upper',extent=extent,cmap=cmap,vmin=vmin,vmax=vmax,interpolation='nearest');ax.contour(X,Y,g['lia_mask'].astype(int),levels=[.5],colors='#596879',linewidths=.55)
  ax.scatter([axp],[ayp],s=38,marker='*',c='yellow',edgecolors='black',linewidths=.5);ax.set(title=title,xlabel='东西距离（km）',ylabel='南北距离（km）');plt.colorbar(im,ax=ax,fraction=.03,pad=.025,label=label)
 for k in range(25):
  fig,axs=plt.subplots(1,2,figsize=(13,6.5),layout='constrained',facecolor='#f5f8fc')
  show(axs[0],D[k],f'{k:02d} 小时 · 冰厚变化','RdBu_r',-limit,limit,'相对初始冰厚变化（m）')
  show(axs[1],V[k],f'{k:02d} 小时 · 表面流速','viridis',0,vlim,'流速（m/年）')
  fig.suptitle('2387 独立出口修正版 · 单日动力演化',fontsize=18);fig.supxlabel('所有时刻使用同一色标；星号为原异常高速点 A。固定材料、零积累消融。',fontsize=10)
  fig.savefig(REPORT/f'hour-{k:02d}.png',dpi=115);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(13,12),layout='constrained')
 show(axs[0,0],H[0],'初始冰厚','Blues',0,hlim,'冰厚（m）');show(axs[0,1],H[-1],'24 小时后冰厚','Blues',0,hlim,'冰厚（m）')
 show(axs[1,0],D[-1],'24 小时冰厚变化','RdBu_r',-limit,limit,'变化（m）');show(axs[1,1],V[-1],'24 小时后表面流速','viridis',0,vlim,'流速（m/年）')
 fig.savefig(REPORT/'comparison.png',dpi=125);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');hours=[t['elapsed_hours'] for t in tl]
 for ax,key,title,unit in [(axs[0],'max_speed_m_year','全场最高流速','m/年'),(axs[1],'old_anomaly_A_speed_m_year','原高速点 A','m/年'),(axs[2],'maximum_H_change_m','最大节点冰厚变化','m')]:
  ax.plot(hours,[t[key] for t in tl],color='#087e9e',marker='.',lw=2);ax.set(xlabel='演化时间（小时）',ylabel=unit,title=title);ax.grid(alpha=.2);ax.ticklabel_format(axis='y',useOffset=False,style='plain')
 fig.savefig(REPORT/'timeseries.png',dpi=140);plt.close(fig)
 summary=dict(status=result['status'],duration_hours=24,initial_max_speed=tl[0]['max_speed_m_year'],final_max_speed=tl[-1]['max_speed_m_year'],initial_old_A_speed=tl[0]['old_anomaly_A_speed_m_year'],final_old_A_speed=tl[-1]['old_anomaly_A_speed_m_year'],maximum_nodal_H_change_m=tl[-1]['maximum_H_change_m'],maximum_pixel_center_H_change_m=limit,volume_change_m3=tl[-1]['volume_m3']-tl[0]['volume_m3'],exported_volume_m3=result['exported_volume_m3'],mass_error_m3=result['mass_error_m3'],checks=result['checks'],wall_seconds=result['wall_seconds'],model='BP plus conservative transport;fixed material,bed and footprint;zero SMB,no thermal feedback',all_map_values='Native pixel-center node fields,not rasterized triangle averages;extrema reported for all model nodes.',source_result_sha256=digest(RUN/'result.json'))
 write_json(REPORT/'summary.json',summary)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 · 24小时动力演化</title><style>body{margin:0;background:#eef3f8;color:#20384b;font:17px "Microsoft YaHei",sans-serif}main{max-width:1300px;margin:auto;padding:26px}h1{font-size:30px}p{line-height:1.8}.cards{display:flex;gap:15px;flex-wrap:wrap}.card{background:white;border-radius:12px;padding:18px;flex:1;min-width:220px}.card b{display:block;font-size:25px;color:#067e98;margin:10px 0}img{width:100%;display:block;background:white;border-radius:12px;margin:15px 0}button{padding:10px 20px;border:0;border-radius:8px;background:#087e9e;color:white;font-size:17px;cursor:pointer}input{width:65%;vertical-align:middle}a{color:#087e9e}.control{position:sticky;top:0;background:#eef3f8;padding:12px 0;z-index:2}small{color:#586d7c}</style><main><h1>2387 · 冰川已经向前演化了 24 小时</h1><p>从独立出口修正版接着计算，每小时更新冰厚与 BP 流速。材料和冰床保持固定，积累、消融为零。</p><div class="cards"><div class="card">全场最高流速<b>__V0__ → __V1__ m/年</b></div><div class="card">原异常高速点 A<b>__A0__ → __A1__ m/年</b></div><div class="card">最大节点冰厚变化<b>__DH__ m</b></div></div><div class="control"><button id="play">播放</button> <input id="time" type="range" min="0" max="24" value="24" step="1" aria-label="演化小时"><b id="label">24 小时</b></div><img id="frame" src="hour-24.png" alt="逐小时冰厚变化和表面速度"><p>红色表示变厚，蓝色表示变薄；星号为原异常高速点 A。所有时刻使用相同色标。地图显示网格中心值，卡片中的极值统计包含全部计算节点。</p><img src="timeseries.png" alt="最高流速、原高速点流速和最大冰厚变化曲线"><p>24 个小时状态均已保存。全局体积收支相对误差为 __MASS__，包含外边界流出；全部状态冰厚非负，材料参数逐位保持一致。单日结果表现平稳，下一阶段可据此延长演化时段。</p><details><summary>初始与一天后的冰厚对照</summary><img src="comparison.png" alt="初始与24小时后冰厚对照"></details><p><a href="../system2387-working-v1/index.html">三维冰面工作页</a> · <a href="../branch0-independent-outlet-v1/index.html">连接修正对照</a> · <a href="summary.json">数值摘要</a></p><small>本次为固定计算范围内的单日动力演化，尚未开启气候或冰温随时间反馈。无新的冰厚重建与参数调节。</small></main><script>const slider=document.getElementById('time'),pic=document.getElementById('frame'),label=document.getElementById('label'),btn=document.getElementById('play');let timer=null;function draw(){const k=Number(slider.value);pic.src='hour-'+String(k).padStart(2,'0')+'.png';label.textContent=k+' 小时'}slider.addEventListener('input',draw);function stop(){clearInterval(timer);timer=null;btn.textContent='播放'}btn.onclick=()=>{if(timer){stop();return}if(Number(slider.value)>=24){slider.value=0;draw()}btn.textContent='暂停';timer=setInterval(()=>{slider.value=Number(slider.value)+1;draw();if(Number(slider.value)>=24)stop()},600)};</script></html>'''
 for key,value in {'__V0__':f"{summary['initial_max_speed']:.2f}",'__V1__':f"{summary['final_max_speed']:.2f}",'__A0__':f"{summary['initial_old_A_speed']:.5f}",'__A1__':f"{summary['final_old_A_speed']:.5f}",'__DH__':f"{summary['maximum_nodal_H_change_m']:.3f}",'__MASS__':f"{result['checks']['relative_mass_budget']:.2e}"}.items():page=page.replace(key,value)
 (REPORT/'index.html').write_text(page,encoding='utf-8')
 write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},source_result_sha256=digest(RUN/'result.json'),builder_sha256=digest(__file__)))
 print(summary)

if __name__=='__main__':main()
