"""Saved-result comparison for the surface-first geometry correction."""
import json
from pathlib import Path
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine
from .v13r1.registration import RUN,REPORT
from .v13.registration import PACKAGE,PROJECT
from .v3.runner import write_json,digest

def read(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def stats(v):
    v=np.asarray(v);v=v[np.isfinite(v)];return dict(n=len(v),median=float(np.median(v)) if len(v) else None,p95=float(np.quantile(v,.95)) if len(v) else None,max=float(v.max()) if len(v) else None)

def render():
    result=json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    if result['status']!='completed':raise ValueError('Run incomplete')
    REPORT.mkdir(parents=True,exist_ok=False);(REPORT/'frames').mkdir()
    oldroot=PROJECT/'polim_cpu/runs/v12-fixed-material-relaxation-v1';old=json.loads((oldroot/'result.json').read_text(encoding='utf-8'))
    mesh=read(RUN/'mesh.npz');g=read(RUN/'geometry.npz');obsroot=PROJECT/'polim_cpu/reports/observed-velocity-v1';obs=read(obsroot/'comparison-fields-v2.npz');obsinfo=json.loads((obsroot/'summary-v2.json').read_text(encoding='utf-8'))
    rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];shape=g['thickness_m'].shape;mask=g['mask'];x,y=g['x'],g['y'];dx=x[1]-x[0];origin=y.min();extent=[(x[0]-dx/2)/1000,(x[-1]+dx/2)/1000,(y[-1]-dx/2-origin)/1000,(y[0]+dx/2-origin)/1000]
    def raster(v):
        a=np.full(shape,np.nan);a[rows,cols]=v;return a
    frames=[];peak_diagnostics=[]
    from scipy.spatial import cKDTree
    ar,ac=np.nonzero(g['anchors']&mask);anchor_tree=cKDTree(np.c_[x[ac],y[ar]])
    for i in range(25):
        a=read(oldroot/f'real/frame-{i:03d}/state.npz');b=read(RUN/f'real/frame-{i:03d}/state.npz');frames.append(dict(oldH=raster(a['h'][nodes]),H=raster(b['h'][nodes]),oldv=raster(np.linalg.norm(a['u'][nodes,-1],axis=1)),v=raster(np.linalg.norm(b['u'][nodes,-1],axis=1))))
    for label,folder,index in [('old_initial',oldroot,0),('new_initial',RUN,0),('old_day1',oldroot,24),('new_day1',RUN,24)]:
        z=read(folder/f'real/frame-{index:03d}/state.npz');speed=np.linalg.norm(z['u'][:,-1],axis=1);node=int(np.argmax(speed));t=mesh['triangles'][np.any(mesh['triangles']==node,axis=1)];xy=mesh['xy'][t];S=mesh['bed'][t]+z['h'][t];edges=xy[:,1:]-xy[:,:1];gradient=np.linalg.solve(edges,(S[:,1:]-S[:,:1])[...,None])[...,0];slopes=np.degrees(np.arctan(np.linalg.norm(gradient,axis=1)));distance=float(anchor_tree.query(mesh['xy'][node])[0])
        peak_diagnostics.append(dict(case=label,node=node,xy_m=mesh['xy'][node].tolist(),speed_m_year=float(speed[node]),H_m=float(z['h'][node]),incident_triangle_surface_slope_degrees=slopes.tolist(),nearest_fixed_centreline_raster_m=distance,interpretation='Local geometric diagnostic; proximity to an input constraint does not prove that constraint is erroneous.'))
    write_json(REPORT/'peak-diagnostics.json',peak_diagnostics)
    package=json.loads(PACKAGE.read_text(encoding='utf-8'));transformer=Transformer.from_crs(package['metadata']['crs_wkt'],obsinfo['observation_grid']['crs'],always_xy=True);X,Y=np.meshgrid(x,y);ox,oy=transformer.transform(X,Y);ot=Affine(*obsinfo['observation_grid']['transform'][:6]);c,r=(~ot)*(ox,oy);idx=(np.floor(r).astype(int)*obs['observed'].shape[1]+np.floor(c).astype(int));N=obs['observed'].size
    def aggregate(v):
        m=mask&np.isfinite(v);n=np.bincount(idx[m],minlength=N);s=np.bincount(idx[m],weights=v[m],minlength=N);return np.divide(s,n,out=np.full(N,np.nan),where=n>0).reshape(obs['observed'].shape)
    core=(obs['lia_fraction']>=.5)&(obs['modern_fraction']>=.5)&obs['valid'];j=(np.hypot(obs['local_x']+134.49697636368546,obs['local_y']-3158715.5030236365)<=240)&core
    new0,new1=aggregate(frames[0]['v']),aggregate(frames[-1]['v']);comparison=dict(common=dict(observed=stats(obs['observed'][core]),old_initial=stats(obs['model_initial'][core]),new_initial=stats(new0[core]),old_day1=stats(obs['model_day1'][core]),new_day1=stats(new1[core])),junction=dict(observed=stats(obs['observed'][j]),old_day1=stats(obs['model_day1'][j]),new_day1=stats(new1[j])))
    comparison['peak_change_initial_percent']=100*(result['timeline'][0]['max_speed_m_year']/old['timeline'][0]['max_speed_m_year']-1);comparison['peak_change_day1_percent']=100*(result['timeline'][-1]['max_speed_m_year']/old['timeline'][-1]['max_speed_m_year']-1)
    comparison['surface_gradient_energy_change_percent']=100*(result['geometry']['new_surface_gradient_energy']/result['geometry']['old_surface_gradient_energy']-1)
    write_json(REPORT/'comparison.json',comparison);np.savez_compressed(REPORT/'comparison-fields.npz',old_speed=obs['model_day1'],new_speed=new1,observed_speed=obs['observed'],common_mask=core,old_H=frames[-1]['oldH'],new_H=frames[-1]['H'],x=x,y=y)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    vmax=max(float(np.nanmax(f[k])) for f in frames for k in ['v','oldv']);hmax=max(float(np.nanmax(f[k])) for f in frames for k in ['H','oldH']);images=[]
    for i,f in enumerate(frames):
        fig,axs=plt.subplots(2,2,figsize=(12.5,10),constrained_layout=True)
        for ax,k,title in zip(axs.ravel(),['oldv','v','oldH','H'],['原冰厚插值 · 表面速度','修正为冰面插值 · 表面速度','原模型冰厚','修正后冰厚']):
            velocity=k in ['v','oldv'];kw={'norm':LogNorm(.01,max(vmax,.02))} if velocity else {'vmin':0,'vmax':hmax}
            im=ax.imshow(f[k],extent=extent,origin='upper',interpolation='nearest',cmap='viridis' if velocity else 'Blues',**kw);fig.colorbar(im,ax=ax,shrink=.8,label='米/年' if velocity else '米');ax.set(title=title,xlabel='投影东向（km）',ylabel='相对北向（km）')
        fig.suptitle(f'相同 BP 与材料参数 · 初始几何修正对照 · 第 {i} 小时',fontsize=15);fn=f'frames/frame-{i:03d}.png';fig.savefig(REPORT/fn,dpi=120);plt.close(fig);images.append(fn)
    oldS=np.where(mask,g['bed_elevation_m']+frames[0]['oldH'],np.nan);newS=np.where(mask,g['surface_elevation_m'],np.nan)
    def slope(S):
        sy,sx=np.gradient(S,y,x);return np.degrees(np.arctan(np.hypot(sx,sy)))
    oldSlope,newSlope=slope(oldS),slope(newS);commonSlope=mask&np.isfinite(oldSlope)&np.isfinite(newSlope);deltaH=frames[0]['H']-frames[0]['oldH'];limit=float(np.nanmax(abs(deltaH)))
    audit=dict(scope='Initial geometry; centred raster gradients on shared valid support',old_slope_degrees=stats(oldSlope[commonSlope]),new_slope_degrees=stats(newSlope[commonSlope]),old_slope_above45_fraction=float(np.mean(oldSlope[commonSlope]>45)),new_slope_above45_fraction=float(np.mean(newSlope[commonSlope]>45)),thickness_difference_m=stats(deltaH[mask]))
    write_json(REPORT/'geometry-audit.json',audit);fig,axs=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for ax,field,title in zip(axs,[oldSlope,newSlope,deltaH],['原冰面坡度','重建冰面坡度','初始冰厚差：重建−原值']):
        isdelta=title.startswith('初始');im=ax.imshow(field,extent=extent,origin='upper',interpolation='nearest',cmap='RdBu_r' if isdelta else 'magma',vmin=-limit if isdelta else 0,vmax=limit if isdelta else 90);fig.colorbar(im,ax=ax,shrink=.8,label='米' if isdelta else '度');ax.set(title=title,xlabel='投影东向（km）',ylabel='相对北向（km）')
    ar,ac=np.nonzero(g['anchors']&mask);axs[2].scatter(x[ac]/1000,(y[ar]-origin)/1000,s=2,c='black',label='固定中心线栅格');axs[2].legend(fontsize=8);fig.suptitle('新初始几何：床面相同，中心线约束相同，冰厚与总体积发生改变');fig.savefig(REPORT/'geometry.png',dpi=140);plt.close(fig)
    profiles=json.loads((obsroot/'profiles-v2.json').read_text(encoding='utf-8'));fig,axs=plt.subplots(2,4,figsize=(16,7),constrained_layout=True);branchstats=[]
    for ax,p in zip(axs.ravel(),profiles):
        bx,by=np.array(p['x']),np.array(p['y']);xx,yy=transformer.transform(bx,by);cc,rr=(~ot)*(xx,yy);rr=np.floor(rr).astype(int);cc=np.floor(cc).astype(int);station=np.array(p['station_m'])/1000;ov=np.array([np.nan if v is None else v for v in p['observed']]);before=obs['model_day1'][rr,cc];after=new1[rr,cc]
        ax.plot(station,np.maximum(before,.001),color='#9ca3af',label='原模型');ax.plot(station,np.maximum(after,.001),color='#c2410c',label='冰面修正');ax.step(station,np.maximum(ov,.001),where='mid',color='#0369a1',label='现代观测');ax.set(yscale='log',ylim=(.001,max(4000,vmax*1.1)),title='分支 '+str(p['branch']),xlabel='沿分支距离（km）',ylabel='米/年');ax.grid(alpha=.2);branchstats.append(dict(branch=p['branch'],old=stats(before),new=stats(after)))
    axs[0,0].legend(fontsize=8);fig.suptitle('第1天分支流速 · 同一120米采样尺度；现代观测仅作跨年代参照');fig.savefig(REPORT/'branches.png',dpi=150);plt.close(fig);write_json(REPORT/'branches.json',branchstats)
    fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True);hours=np.arange(25)
    for ax,key,title,ylabel in zip(axs,['max_speed_m_year','maximum_H_change_m','volume_m3'],['最高速度','各自初始几何后的最大冰厚变化','冰体积'],['米/年','米','百万立方米']):
        scale=1e6 if key=='volume_m3' else 1;ax.plot(hours,[r[key]/scale for r in old['timeline']],label='原模型',color='#9ca3af');ax.plot(hours,[r[key]/scale for r in result['timeline']],label='冰面修正',color='#c2410c');ax.set(title=title,xlabel='小时',ylabel=ylabel);ax.grid(alpha=.2)
    axs[0].legend();fig.savefig(REPORT/'history.png',dpi=140);plt.close(fig)
    data=dict(old=[{k:r[k] for k in ['max_speed_m_year','max_thickness_m','volume_m3','maximum_H_change_m']} for r in old['timeline']],new=[{k:r[k] for k in ['max_speed_m_year','max_thickness_m','volume_m3','maximum_H_change_m']} for r in result['timeline']],images=images,comparison=comparison,geometry=result['geometry'],checks=result['checks'],export_m3=result['exported_volume_m3'],wall_seconds=result['wall_seconds']);(REPORT/'data.js').write_text('window.SURFACE_DATA='+json.dumps(data,ensure_ascii=False)+';',encoding='utf-8')
    def row(name,a,b,c):return f'<tr><td>{name}</td><td>{a:.3g}</td><td>{b:.3g}</td><td>{c:.3g}</td></tr>'
    co=comparison['common'];ju=comparison['junction'];table=row('共同范围中位数',co['observed']['median'],co['old_day1']['median'],co['new_day1']['median'])+row('共同范围最大值',co['observed']['max'],co['old_day1']['max'],co['new_day1']['max'])+row('汇流区中位数',ju['observed']['median'],ju['old_day1']['median'],ju['new_day1']['median'])
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 冰面重建修正</title><style>*{box-sizing:border-box}body{font-family:Microsoft YaHei,sans-serif;background:#edf3f6;color:#193646;margin:0}main{max-width:1400px;margin:auto;padding:28px}p{line-height:1.8}.panel,.controls,.stats{background:white;padding:22px;border-radius:14px;margin:18px 0}.controls{display:flex;gap:18px;align-items:center}input{flex:1}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.value{font-size:25px;color:#12638a}.note{font-size:13px;color:#607584}button{background:#116a98;color:white;border:0;padding:12px 18px;border-radius:8px;cursor:pointer}img{width:100%;height:auto}td,th{text-align:left;padding:12px;border-bottom:1px solid #e5edf2}table{border-collapse:collapse;width:100%}a{color:#116a98}@media(max-width:700px){main{padding:12px}.stats{grid-template-columns:1fr}td,th{padding:6px;font-size:12px}}</style><main><h1>2387 冰川 · 从冰厚插值改为冰面重建</h1><p>借鉴 PalaeoIce 2.0：先建立连续冰面，再减去冰床得到冰厚。新算法约束冰面不低于床面，保留原床面、中心线栅格约束和共同网格。两组使用相同 BP 与固定材料参数，均已完成一天计算。</p><div class="panel"><p id="outcome"></p><p id="geometry"></p><p id="spatialOutcome"></p><p class="note">这是新的初始几何，重建体积改变不是演化收支错误。现代观测不用于拟合参数；尚不能把几何修正等同于真实古冰面的恢复。原中心线高程仍是输入假设。</p></div><div class="controls"><button id="play">播放结果</button><input id="time" type="range" min="0" max="24" value="0" aria-label="计算时间"><output id="clock">0 小时</output></div><div class="stats"><div>原模型最高速度<div class="value" id="oldSpeed"></div></div><div>修正后最高速度<div class="value" id="newSpeed"></div></div><div>修正后最大冰厚<div class="value" id="thick"></div></div></div><div class="panel"><img id="map" alt="原模型与冰面修正模型的速度和冰厚对比"><p class="note">各时刻速度共用对数色标，冰厚共用线性色标。地图显示15.75米源像元中心，最高速度统计包含所有网格节点。</p></div><div class="panel"><img src="geometry.png" alt="初始冰面坡度与冰厚变化"><p class="note">坡度由15.75米栅格中心差分计算；仅比较两侧都有有效邻点的共同范围。黑点是保持原值的中心线栅格，不代表观测古冰面。</p></div><div class="panel"><img src="history.png" alt="一天速度冰厚与体积变化"></div><div class="panel"><h2>同尺度空间参照</h2><table><tr><th>米/年</th><th>现代观测</th><th>原模型第1天</th><th>修正后第1天</th></tr>TABLE</table><img src="branches.png" alt="各支流流速对比"><p class="note">共同范围953个独立120米观测像元。观测为ITS_LIVE2014—2024气候态，模型为古冰川几何；该表用于定位空间差异，不是现代条件下的误差评分。分支图显示下限0.001米/年，原数据不截断。</p></div><details class="panel"><summary>方法、计算收支与来源</summary><p>在原生四邻域网格上最小化冰面相邻高差平方和，同时满足H≥0。中心线及原正冰厚域边界圈的厚度保持原值。它是独立实现的带床面约束的冰面插值，未调用ArcGIS，也不是PalaeoIce的完整移植。允许内部无冰位置，不添加人为薄冰层，不强制保留原重建体积，不调低速度。</p><p>一个平面冰面解析算例与床面障碍算例验证新算子；已有BP、守恒输运和独立续算检查复用。第一小时后的极微量湿单元触发线性失衡，已修复低于高程浮点分辨率的动量干湿判定；原厚度和冰量保留，从第一小时继续，未重算已完成部分。一天中每小时更新BP，期间保守输运，材料固定，积累、消融、底融和热反馈关闭。</p><p id="budget"></p><p><a href="https://github.com/yingkui2003/PalaeoIce">PalaeoIce 2.0 原作者方法说明</a> · <a href="https://www.sciencedirect.com/science/article/pii/S0169555X22004160">PalaeoIce 论文</a> · <a href="comparison.json">对照统计</a> · <a href="comparison-fields.npz">对照空间数据</a></p></details><p><a href="../observed-velocity-v1/index.html">之前的观测对照</a> · <a href="../../index.html">项目首页</a></p></main><script src="data.js"></script><script>const D=window.SURFACE_DATA,$=id=>document.getElementById(id);let timer=null;const slider=$('time');function show(){let i=+slider.value;$('clock').textContent=i+' 小时';$('map').src=D.images[i];$('oldSpeed').textContent=D.old[i].max_speed_m_year.toFixed(1)+' 米/年';$('newSpeed').textContent=D.new[i].max_speed_m_year.toFixed(1)+' 米/年';$('thick').textContent=D.new[i].max_thickness_m.toFixed(2)+' 米'}slider.addEventListener('input',show);$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放结果';return}if(+slider.value===24)slider.value=0;show();$('play').textContent='暂停';timer=setInterval(()=>{slider.value=+slider.value+1;show();if(+slider.value===24){clearInterval(timer);timer=null;$('play').textContent='重新播放'}},700)};$('outcome').textContent='一天后的最高速度：原模型 '+D.old[24].max_speed_m_year.toFixed(0)+' → 修正后 '+D.new[24].max_speed_m_year.toFixed(0)+' 米/年，变化 '+D.comparison.peak_change_day1_percent.toFixed(2)+'%。';$('geometry').textContent='初始冰面高差平方和变化 '+D.comparison.surface_gradient_energy_change_percent.toFixed(2)+'%；重建体积变化 '+D.geometry.volume_change_percent.toFixed(2)+'%；固定栅格约束误差 '+D.geometry.constraint_error_m.toExponential(1)+' 米。';$('spatialOutcome').textContent='汇流区中位速度：原模型 '+D.comparison.junction.old_day1.median.toFixed(2)+' → 修正后 '+D.comparison.junction.new_day1.median.toFixed(2)+' 米/年；现代观测参照 '+D.comparison.junction.observed.median.toFixed(2)+' 米/年。局部高速仍在，几何插值修正尚未解决全部空间差异。';$('budget').textContent='新几何一天边界流出 '+D.export_m3.toFixed(3)+' 立方米；全域相对冰量收支误差 '+D.checks.relative_mass_budget.toExponential(2)+'，汇流区 '+D.checks.relative_junction_budget.toExponential(2)+'；计算耗时 '+(D.wall_seconds/60).toFixed(2)+' 分钟。';show();</script></html>'''
    (REPORT/'index.html').write_text(html.replace('TABLE',table),encoding='utf-8');write_json(REPORT/'delivery.json',dict(status='completed',result_sha256=digest(RUN/'result.json'),files={p.relative_to(REPORT).as_posix():digest(p) for p in REPORT.rglob('*') if p.is_file()},report_code_sha256=digest(Path(__file__))));print(json.dumps(comparison,ensure_ascii=False))

if __name__=='__main__':render()
