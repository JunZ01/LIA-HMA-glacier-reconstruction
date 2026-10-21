"""Playback of saved one-day relaxation; no browser simulation."""
import json
from pathlib import Path
import numpy as np
from .registration import RUN,MESH
from ..v3.runner import write_json,digest
REPORT=RUN.parents[1]/'reports/fixed-material-relaxation-v1'


def render():
    result=json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    if REPORT.exists():return REPORT/'index.html'
    REPORT.mkdir(parents=True);(REPORT/'frames').mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    with np.load(MESH,allow_pickle=False) as z:mesh={k:z[k] for k in z.files}
    rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];xy=mesh['xy'][nodes];shape=(int(rows.max()+1),int(cols.max()+1));frames=[]
    for record in result['timeline']:
        with np.load(RUN/f'real/frame-{record["frame"]:03d}/state.npz',allow_pickle=False) as z:frames.append((z['h'][nodes],z['u'][nodes,-1]))
    h0=frames[0][0];hmax=max(h.max() for h,u in frames);smax=max(np.linalg.norm(u,axis=1).max() for h,u in frames);limit=max(.001,float(np.quantile(np.concatenate([abs(h-h0) for h,u in frames]),.995)))
    dx=15.75;x0=float((xy[:,0]-cols*dx).mean())-dx/2;y0=float((xy[:,1]+rows*dx).mean())+dx/2;origin=xy[:,1].min();extent=[x0/1000,(x0+shape[1]*dx)/1000,(y0-shape[0]*dx-origin)/1000,(y0-origin)/1000]
    def raster(v):
        a=np.full(shape,np.nan);a[rows,cols]=v;return a
    images=[]
    for record,(h,u) in zip(result['timeline'],frames):
        speed=np.linalg.norm(u,axis=1);fig,axes=plt.subplots(1,3,figsize=(15,5.7),constrained_layout=True)
        for ax,v,title,cmap,norm,lo,hi in zip(axes,[h,speed,h-h0],['冰厚（m）','表面水平速度（m/年）','相对初始冰厚变化（m）'],['Blues','viridis','RdBu'],[None,LogNorm(.01,max(smax,.02)),None],[0,None,-limit],[hmax,None,limit]):
            im=ax.imshow(raster(v),extent=extent,origin='upper',interpolation='nearest',cmap=cmap,norm=norm,vmin=lo,vmax=hi);fig.colorbar(im,ax=ax,shrink=.75);ax.set(title=title,xlabel='投影东向（km）',ylabel='相对北向（km）')
        chosen=(rows%12==0)&(cols%12==0)&(speed>.1);den=speed[chosen]
        axes[1].quiver(xy[chosen,0]/1000,(xy[chosen,1]-origin)/1000,u[chosen,0]/den,u[chosen,1]/den,color='white',scale=45,width=.003)
        fig.suptitle(f'2387 冰川 · 固定材料动力松弛 {record["elapsed_hours"]:.0f} 小时',fontsize=17)
        filename=f'frames/frame-{record["frame"]:03d}.png';fig.savefig(REPORT/filename,dpi=120);plt.close(fig);images.append(filename)
    timeline=result['timeline'];hours=[r['elapsed_hours'] for r in timeline];fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    axs[0].plot(hours,[r['max_speed_m_year'] for r in timeline],'-o',ms=3,label='全域最高速度');axs[0].plot(hours,[r['original_peak_speed_m_year'] for r in timeline],label='原最高速度位置');axs[0].set(ylabel='速度（m/年）',title='速度如何变化');axs[0].legend(fontsize=8)
    axs[1].plot(hours,[r['maximum_H_change_m'] for r in timeline],color='#d55e00');axs[1].set(ylabel='最大绝对冰厚改变量（m）',title='冰厚重新分布')
    axs[2].plot(hours,[result['initial_volume_m3']-r['volume_m3'] for r in timeline],color='#0072b2');axs[2].set(ylabel='体积减少（m³）',title='仅通过边界流出减少')
    for ax in axs:ax.set_xlabel('已计算时间（小时）');ax.grid(alpha=.2)
    fig.savefig(REPORT/'history.png',dpi=140);plt.close(fig)
    first,last=timeline[0],timeline[-1];data=dict(timeline=[{k:v for k,v in r.items() if k!='momentum'} for r in timeline],images=images,checks=result['checks'],export_m3=result['exported_volume_m3'],mass_error_m3=result['mass_error_m3'],junction_error_m3=result['junction_budget_error_m3'],wall_seconds=result['wall_seconds'],color_limit_m=limit,peak_change_percent=100*(last['max_speed_m_year']/first['max_speed_m_year']-1),hot_change_percent=100*(last['initial_hot_region_mean_speed_m_year']/first['initial_hot_region_mean_speed_m_year']-1))
    (REPORT/'data.js').write_text('window.RELAXATION_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')
    html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 一天动力松弛</title><style>*{box-sizing:border-box}body{margin:0;background:#edf3f6;color:#193646;font-family:Microsoft YaHei,sans-serif}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px}p{line-height:1.8}.panel,.stats,.controls{background:white;padding:20px;border-radius:14px;margin-top:18px}.controls{display:flex;gap:20px;align-items:center}button{background:#116a98;color:white;border:0;border-radius:8px;padding:12px 20px;cursor:pointer}input{flex:1;accent-color:#116a98}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}.label,.note{color:#607584;font-size:13px}.value{font-size:25px;margin-top:8px}img{width:100%;height:auto}a{color:#116a98}table{width:100%;border-collapse:collapse}td{padding:9px;border-bottom:1px solid #eee}@media(max-width:700px){main{padding:12px}.stats{grid-template-columns:1fr 1fr}h1{font-size:24px}}</style><main><h1>2387 冰川 · 一天动力松弛</h1><p>固定原始冰床和材料流动性，关闭积累、消融、底融和温度反馈。只让冰流重新分配冰厚，边界流出的冰量单独记录。以下均为实际保存的计算结果。</p><div class="panel"><p id="outcome"></p><p class="note">一天的变化用于诊断初始冰厚的动力响应，不能证明已经达到平衡或恢复了真实冰面。计算域仍允许冰流出，未模拟域外冰缘扩展。</p></div><div class="controls"><button id="play">播放结果</button><input id="time" aria-label="已计算时间" type="range" min="0" max="24" value="0"><output id="clock">0 小时</output></div><div class="stats"><div><div class="label">冰体积</div><div class="value" id="volume"></div></div><div><div class="label">最大冰厚</div><div class="value" id="thick"></div></div><div><div class="label">最大表面速度</div><div class="value" id="speed"></div></div><div><div class="label">最大冰厚改变量</div><div class="value" id="change"></div></div></div><div class="panel"><img id="map" alt="冰厚、速度和冰厚变化"><p class="note" id="colors"></p></div><div class="panel"><img src="history.png" alt="一天内速度、冰厚和边界流出变化"></div><details class="panel"><summary>收支与计算条件</summary><table id="budget"></table><p class="note">15.75米原生网格，BP采用3个垂向节点；每小时重新求解速度，期间按CFL稳定步长保守输运。材料流动性固定在初始水平位置与相对高度层，不随冰厚、温度或输运更新。汇流区收支取原主要汇流点周围126米范围的网格控制体积；共享网格内通量单次计入。</p></details><p><a href="../material-rheology-v1/index.html">当前热力耦合模型</a> · <a href="../constrained-bed-seam-v1/index.html">之前的几何对照</a></p></main><script src="data.js"></script><script>const D=window.RELAXATION_DATA,$=id=>document.getElementById(id);let timer=null;const slider=$(\'time\');slider.max=D.timeline.length-1;function show(){const i=+slider.value,r=D.timeline[i];$(\'clock\').textContent=r.elapsed_hours.toFixed(0)+\' 小时\';$(\'map\').src=D.images[i];$(\'volume\').textContent=(r.volume_m3/1e6).toFixed(3)+\' 百万 m³\';$(\'thick\').textContent=r.max_thickness_m.toFixed(2)+\' m\';$(\'speed\').textContent=r.max_speed_m_year.toLocaleString(\'zh-CN\',{maximumFractionDigits:0})+\' m/年\';$(\'change\').textContent=r.maximum_H_change_m.toFixed(3)+\' m\'}slider.addEventListener(\'input\',show);$(\'play\').onclick=()=>{if(timer){clearInterval(timer);timer=null;$(\'play\').textContent=\'播放结果\';return}if(+slider.value===+slider.max)slider.value=0;show();$(\'play\').textContent=\'暂停\';timer=setInterval(()=>{slider.value=+slider.value+1;show();if(+slider.value===+slider.max){clearInterval(timer);timer=null;$(\'play\').textContent=\'重新播放\'}},700)};const first=D.timeline[0],last=D.timeline.at(-1);$(\'outcome\').textContent=\'24 小时内最高速度 \'+first.max_speed_m_year.toFixed(0)+\' → \'+last.max_speed_m_year.toFixed(0)+\' m/年，变化 \'+D.peak_change_percent.toFixed(2)+\'%；初始最快 1% 区域的面积加权平均速度变化 \'+D.hot_change_percent.toFixed(2)+\'%。\';$(\'colors\').textContent=\'各时刻共用色标，速度为对数色标。冰厚变化色标 ±\'+D.color_limit_m.toFixed(2)+\' m；超出范围颜色饱和，原始数据不截断。地图显示源像元中心，统计包括所有网格节点。\';const entries=[[\'积累、消融及底融\',\'全部为零\'],[\'累计边界流出\',D.export_m3.toFixed(3)+\' m³\'],[\'全域冰量收支误差\',D.mass_error_m3.toExponential(3)+\' m³\'],[\'汇流区冰量收支误差\',D.junction_error_m3.toExponential(3)+\' m³\'],[\'材料流动性\',\'全程固定\'],[\'运行耗时\',(D.wall_seconds/60).toFixed(2)+\' 分钟\']];$(\'budget\').innerHTML=entries.map(([a,b])=>\'<tr><td>\'+a+\'</td><td>\'+b+\'</td></tr>\').join(\'\');show();</script></html>'
    (REPORT/'index.html').write_text(html,encoding='utf-8')
    write_json(REPORT/'delivery.json',dict(status=result['status'],result_sha256=digest(RUN/'result.json'),files={p.relative_to(REPORT).as_posix():digest(p) for p in REPORT.rglob('*') if p.is_file()}))
    return REPORT/'index.html'
