"""A playable timeline of actual computed states, no browser-side simulation."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.font_manager import FontProperties
from .registration import RUN,MESH
from ..v3.runner import write_json,digest

REPORT=RUN.parents[1]/'reports/constrained-bed-seam-v1'


def render():
    result=json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    from .comparison import compare
    if REPORT.exists():return REPORT/'index.html'
    REPORT.mkdir(parents=True);(REPORT/'frames').mkdir()
    comparison=compare(result)
    from .profiles import render_profiles
    render_profiles(REPORT)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    with np.load(MESH,allow_pickle=False) as z:mesh={k:z[k] for k in z.files}
    rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];xy=mesh['xy'][nodes]
    nr,nc=int(rows.max()+1),int(cols.max()+1);shape=(nr,nc);snapshots=[]
    for frame in result['timeline']:
        with np.load(RUN/f'real/frame-{frame["frame"]:03d}/state.npz',allow_pickle=False) as z:
            snapshots.append((z['h'][nodes],np.linalg.norm(z['u'][nodes,-1],axis=1),z['u'][nodes,-1]))
    h0=snapshots[0][0];hmax=max(float(h.max()) for h,s,u in snapshots);smax=max(float(s.max()) for h,s,u in snapshots)
    delta_limit=max(.001,float(np.quantile(np.concatenate([abs(h-h0) for h,s,u in snapshots]),.995)))
    # Coordinates come from frozen native source centres; display does not resample data.
    dx=15.75;minx=float((xy[:,0]-cols*dx).mean())-dx*.5;maxy=float((xy[:,1]+rows*dx).mean())+dx*.5
    extent=[minx/1000,(minx+nc*dx)/1000,(maxy-nr*dx)/1000,maxy/1000]
    centre_y=float(xy[:,1].min())/1000;extent[2]-=centre_y;extent[3]-=centre_y
    def raster(v):
        a=np.full(shape,np.nan);a[rows,cols]=v;return a
    imagefiles=[]
    for frame,(h,s,u) in zip(result['timeline'],snapshots):
        fig,axs=plt.subplots(1,3,figsize=(15.5,5.8),constrained_layout=True)
        configs=[(h,'冰厚（m）','Blues',None,0,hmax),(s,'表面水平速度（m/年）','viridis',LogNorm(vmin=.01,vmax=max(smax,.02)),None,None),
                 (h-h0,'相对初始冰厚的变化（m）','RdBu',None,-delta_limit,delta_limit)]
        for ax,(values,title,cmap,norm,vmin,vmax) in zip(axs,configs):
            im=ax.imshow(raster(values),extent=extent,origin='upper',cmap=cmap,norm=norm,vmin=vmin,vmax=vmax,interpolation='nearest')
            ax.set(title=title,xlabel='投影东向（km）',ylabel='相对北向（km）');fig.colorbar(im,ax=ax,shrink=.75)
        select=(rows%12==0)&(cols%12==0)&(s>.1);norm=np.maximum(s[select],1e-30)
        axs[1].quiver(xy[select,0]/1000,xy[select,1]/1000-centre_y,u[select,0]/norm,u[select,1]/norm,color='white',scale=45,width=.003,alpha=.65)
        fig.suptitle(f'2387 多支流冰川 · 已计算 {frame["elapsed_hours"]*60:.0f} 分钟',fontsize=17,fontweight='bold')
        filename=f'frames/frame-{frame["frame"]:03d}.png';fig.savefig(REPORT/filename,dpi=120);plt.close(fig);imagefiles.append(filename)
    fig,axs=plt.subplots(1,2,figsize=(12,3.6),constrained_layout=True);times=[r['elapsed_hours']*60 for r in result['timeline']]
    axs[0].plot(times,[(r['volume_m3']-result['initial_volume_m3'])/1e6 for r in result['timeline']],'-o',color='#0072b2');axs[0].set(xlabel='已计算时间（分钟）',ylabel='相对初始体积变化（百万 m³）',title='冰体积变化')
    axs[1].plot(times,[r['mean_temperature_C'] for r in result['timeline']],'-o',color='#d55e00');axs[1].set(xlabel='已计算时间（分钟）',ylabel='体积加权平均冰温（°C）',title='温度反馈');axs[1].ticklabel_format(useOffset=False,axis='y')
    for ax in axs:ax.grid(alpha=.2)
    fig.savefig(REPORT/'history.png',dpi=140);plt.close(fig)
    data=dict(timeline=[{k:v for k,v in r.items() if k!='momentum'} for r in result['timeline']],images=imagefiles,
        budget={k:result[k] for k in ['initial_volume_m3','final_volume_m3','exported_volume_m3','added_volume_m3','removed_volume_m3','mass_error_m3','wall_seconds','total_substeps']},checks=result['checks'],comparison=comparison,delta_color_limit=delta_limit)
    (REPORT/'data.js').write_text('window.MODEL_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 冰川模型</title><style>
*{box-sizing:border-box}body{margin:0;background:#edf3f6;color:#183344;font-family:"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:28px}header{display:flex;justify-content:space-between;align-items:center;gap:20px}h1{font-size:30px;margin:0 0 10px}p{line-height:1.8}.pill{border-radius:30px;background:#deefe7;color:#176744;padding:10px 18px;white-space:nowrap}.intro{color:#4e6675;margin:4px 0 20px}.controls,.panel,.stats{background:white;border-radius:14px;padding:18px 22px;margin-top:18px;box-shadow:0 3px 16px #18334408}.controls{display:flex;align-items:center;gap:20px}button{border:0;border-radius:9px;background:#116a98;color:white;padding:12px 22px;font-size:15px;cursor:pointer}input{flex:1;accent-color:#116a98}output{width:110px;text-align:right}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}.label{font-size:13px;color:#607483}.value{font-size:25px;margin-top:8px;font-weight:600}.map img,.history{width:100%;height:auto;display:block}.note{font-size:13px;color:#607483;line-height:1.8}.details{display:grid;grid-template-columns:1fr 1fr;gap:20px}summary{cursor:pointer;font-weight:600}table{width:100%;border-collapse:collapse}td{padding:8px;border-bottom:1px solid #e8edf0}a{color:#116a98}@media(max-width:700px){main{padding:15px}header{display:block}h1{font-size:24px}.stats{grid-template-columns:1fr 1fr}.details{display:block}.controls{gap:8px;padding:12px}button{padding:10px}output{width:82px}.value{font-size:20px}}</style>
<main><header><div><h1>2387 多支流冰川模型</h1><p class="intro">原生重建几何 · BP 速度 · 冰厚更新 · 温度反馈</p></div><span class="pill">冰床转接已计算</span></header>
<p class="intro">针对现代冰川边缘的冰床拼接突跳，完成一次 63 米源尺度内的约束转接试算。初始冰厚、冰量、支流约束和材料参数保持一致。</p>
<div class="panel"><h3>同一初始冰厚的冰床转接前后对比</h3><p id="repair"></p><img class="history" src="comparison.png" alt="同一原始冰面修复前后的表面速度对比"><p class="note">仅调整冰床拼接带，支流约束处保持原值。初始冰厚和冰温完全一致，旧结果直接读取。采用共同对数色标，速度没有截断。这是明确假设下的转接试算，尚未恢复观测冰床。</p></div><div class="controls"><button id="play">播放结果</button><input id="time" aria-label="计算时间" type="range" min="0" max="6" value="0"><output id="clock">0 分钟</output></div>
<div class="stats"><div><div class="label">冰体积</div><div class="value" id="volume"></div></div><div><div class="label">最大冰厚</div><div class="value" id="thick"></div></div><div><div class="label">最大表面速度</div><div class="value" id="speed"></div></div><div><div class="label">平均冰温</div><div class="value" id="temp"></div></div></div>
<div class="panel map"><img id="map" alt="冰厚、速度与冰厚变化的实际计算图"><p class="note" id="colors"></p></div>
<div class="panel"><h3>高速区与汇流处剖面</h3><img class="history" src="profiles.png" alt="原始地形、冰床、冰面与支流约束剖面"><p class="note">P 为原最高速度位置，J 为主要汇流点。曲线直接采样原始及修正栅格，支流约束使用原有采样位置。</p></div><div class="panel"><img class="history" src="history.png" alt="完整时间序列中的冰体积与冰温变化"></div>
<div class="details"><div class="panel"><h3>这版已经连通</h3><p>所有支流在同一网格内交换应力、冰量与热量。温度改变冰的流动性，更新后的速度推动冰厚变化。状态会逐段保存，可从断点继续开发。</p><p class="note">每 10 分钟更新一次整体速度，期间按稳定步长推进冰厚和温度。图中播放的是保存结果，并非浏览器现场求解。</p></div>
<div class="panel"><h3>当前使用范围</h3><p>这是可运行的研究原型。当前计算域允许冰流出，不提供域外冰流入；尚未处理域外冰缘扩展。材料流变沿用上一版。现代冰缘转接保留了源地形、原始厚度和支流约束；真实冰崖、分水岭与外边界尚未逐段验证。</p><p class="note">保持 15.75 m 原生几何；BP 采用 3 个垂向节点，温度采用 9 个节点。材料公式借鉴 <a href="https://www.pism.io/docs/manual/modeling-choices/dynamics/rheology.html">PISM GPBLD</a>，含水量对软化的作用采用其 1% 上限；热量模块仍保留原来的 3% 排水上限。地热通量假设为 0.05 W/m²。采用项目 1441 年气候初始化并固定本小时表面条件。</p></div></div>
<details class="panel"><summary>守恒结果与运行信息</summary><table id="budget"></table><p class="note">冰量收支同时记录表面增减、底融和边界流出。热量采用保守水平输运与简化垂向扩散；尚未解析垂向平流。数值完整运行不等于科学验收。</p></details>
</main><script src="data.js"></script><script>
const D=window.MODEL_DATA,$=id=>document.getElementById(id);let timer=null;const slider=$('time');slider.max=D.timeline.length-1;
function show(){let i=+slider.value,r=D.timeline[i];$('clock').textContent=(r.elapsed_hours*60).toFixed(0)+' 分钟';$('map').src=D.images[i];$('volume').textContent=(r.volume_m3/1e6).toFixed(3)+' 百万 m³';$('thick').textContent=r.max_thickness_m.toFixed(2)+' m';$('speed').textContent=r.max_speed_m_year.toLocaleString('zh-CN',{maximumFractionDigits:0})+' m/年';$('temp').textContent=r.mean_temperature_C.toFixed(4)+' °C'}
slider.addEventListener('input',show);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放结果';return}if(+slider.value===+slider.max)slider.value=0;show();$('play').textContent='暂停';timer=setInterval(()=>{slider.value=+slider.value+1;show();if(+slider.value===+slider.max){clearInterval(timer);timer=null;$('play').textContent='重新播放'}},900)};
$('colors').textContent='各时刻使用相同色标。速度为对数色标；冰厚变化色标为 ±'+D.delta_color_limit.toFixed(3)+' m，超过此范围的颜色饱和，原始数值未截断。地图显示源像元中心值，统计包含所有网格节点。';
$('repair').textContent='初始最高速度：'+D.comparison.old_initial_max_speed_m_year.toLocaleString('zh-CN',{maximumFractionDigits:0})+' → '+D.comparison.new_initial_max_speed_m_year.toLocaleString('zh-CN',{maximumFractionDigits:0})+' m/年，降低 '+D.comparison.peak_reduction_percent.toFixed(2)+'%。完整一小时计算后最高速度为 '+D.timeline.at(-1).max_speed_m_year.toLocaleString('zh-CN',{maximumFractionDigits:0})+' m/年。';
const B=D.budget;const entries=[['累计表面增加',(B.added_volume_m3).toFixed(3)+' m³'],['累计表面消融及底融',B.removed_volume_m3.toFixed(3)+' m³'],['累计边界流出',B.exported_volume_m3.toFixed(3)+' m³'],['冰量收支误差',B.mass_error_m3.toExponential(3)+' m³'],['输运子步数',B.total_substeps],['计算耗时',(B.wall_seconds/60).toFixed(2)+' 分钟']];$('budget').innerHTML=entries.map(([a,b])=>'<tr><td>'+a+'</td><td>'+b+'</td></tr>').join('');D.images.forEach(src=>{let im=new Image();im.src=src});show();
</script></html>'''
    (REPORT/'index.html').write_text(html,encoding='utf-8')
    (RUN.parents[1]/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=reports/constrained-bed-seam-v1/index.html"><a href="reports/constrained-bed-seam-v1/index.html">打开冰川模型</a>',encoding='utf-8')
    write_json(REPORT/'delivery.json',dict(result_sha256=digest(RUN/'result.json'),files={p.relative_to(REPORT).as_posix():digest(p) for p in REPORT.rglob('*') if p.is_file()},status=result['status']))
    return REPORT/'index.html'
