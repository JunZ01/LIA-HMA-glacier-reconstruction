"""Render completed native real BP diagnostic; never generate substitute fields."""
import json,shutil,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,colors,tri as mtri
from matplotlib.patches import Rectangle
from .v3.runner import PROJECT,digest,write_json

ROOT=PROJECT/'polim_cpu'
RUN=ROOT/'runs/v7-native-global-bp-r1'
REPORT=ROOT/'reports/native-global-bp-v1'


def main():
    result=json.loads((RUN/'result.json').read_text(encoding='utf-8'))
    if result['status']!='completed' or not result['converged']:raise ValueError('Completed converged fields required')
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':11,
                         'axes.titlepad':15,'figure.facecolor':'white','axes.facecolor':'#eff3f5'})
    REPORT.mkdir(parents=True,exist_ok=False)
    with np.load(ROOT/'runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz',allow_pickle=False) as z:mesh={k:z[k] for k in z.files}
    with np.load(RUN/'fields.npz',allow_pickle=False) as z:velocity=z['velocity_m_year'];stress=z['triangle_centroid_stress']
    with np.load(ROOT/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz',allow_pickle=False) as z:
        x=z['x'];y=z['y'];mask=z['valid_mask'];H=z['thickness_m']
    with np.load(ROOT/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/branch-07.npz',allow_pickle=False) as z:
        junction=np.array([z['easting_m'][23],z['northing_m'][23]])
    origin=np.array([x.min(),y.min()]);xy=(mesh['xy']-origin)/1000
    extent=[(x[0]-7.875-origin[0])/1000,(x[-1]+7.875-origin[0])/1000,
            (y[-1]-7.875-origin[1])/1000,(y[0]+7.875-origin[1])/1000]
    centers=mesh['pixel_center_nodes'];rows=mesh['pixel_rows'];cols=mesh['pixel_cols']
    speeds=np.linalg.norm(velocity[:,-1],axis=1)
    speedmap=np.full(mask.shape,np.nan);speedmap[rows,cols]=speeds[centers]
    selected=(rows%12==0)&(cols%12==0)&(speeds[centers]>1e-5)
    center_xy=xy[centers];vectors=velocity[centers,-1]
    dirs=vectors/np.maximum(np.linalg.norm(vectors,axis=1)[:,None],1e-30)
    jxy=(junction-origin)/1000;half=.65
    q95,q99=np.quantile(speedmap[mask],[.95,.99]);vmax=max(float(q99),1.);vmin=min(.1,vmax/1000)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#eff3f5')
    fig,axes=plt.subplots(1,2,figsize=(14,9.8),layout='constrained')
    im=axes[0].imshow(np.where(mask,H,np.nan),extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=400)
    fig.colorbar(im,ax=axes[0],shrink=.72,pad=.025,label='冰厚（m）')
    im=axes[1].imshow(np.where(speedmap>0,speedmap,np.nan),extent=extent,origin='upper',cmap=cmap,norm=colors.LogNorm(vmin=vmin,vmax=vmax))
    fig.colorbar(im,ax=axes[1],shrink=.72,pad=.025,label='表面速度（m/年；对数色标）',extend='both')
    fastest_xy=xy[int(speeds.argmax())]
    axes[1].plot(*fastest_xy,marker='*',markersize=12,color='#ff58ae',markeredgecolor='white',zorder=5)
    axes[1].annotate('最高速度',fastest_xy,xytext=(12,9),textcoords='offset points',color='#923264',fontsize=10)
    axes[1].quiver(center_xy[selected,0],center_xy[selected,1],dirs[selected,0],dirs[selected,1],
                   angles='xy',scale_units='xy',scale=8,color='white',width=.003,headwidth=3.5)
    for ax,title in zip(axes,['配套源数据冰厚','真实几何联合求解的表面速度']):
        ax.set_title(title,fontweight='bold');ax.set_xlabel('相对东向距离（km）');ax.set_ylabel('相对北向距离（km）')
        ax.add_patch(Rectangle(jxy-half,2*half,2*half,fill=False,edgecolor='#df4b35',linewidth=1.5))
        ax.set_aspect('equal')
    fig.suptitle('系统 2387 · 真实冰川固定几何诊断',fontsize=18,fontweight='bold')
    fig.savefig(REPORT/'surface-velocity.png',dpi=155);plt.close(fig)
    tri=mesh['triangles'];centroid=xy[tri].mean(axis=1)
    selection=(abs(centroid[:,0]-jxy[0])<half+.04)&(abs(centroid[:,1]-jxy[1])<half+.04)
    tau=stress[:,:,:6]
    effective=np.sqrt(.5*(tau[:,:,0]**2+tau[:,:,1]**2+tau[:,:,5]**2+2*(tau[:,:,2]**2+tau[:,:,3]**2+tau[:,:,4]**2))).mean(axis=1)/1000
    shear=tau[:,:,2].mean(axis=1)/1000
    shear_limit=max(float(np.quantile(abs(shear[selection]),.98)),1e-6)
    effective_limit=max(float(np.quantile(effective[selection],.98)),1e-6)
    triangulation=mtri.Triangulation(xy[:,0],xy[:,1],tri[selection])
    fig,axes=plt.subplots(1,2,figsize=(14,7.6),layout='constrained')
    im=axes[0].tripcolor(triangulation,facecolors=effective[selection],shading='flat',cmap='magma',vmin=0,vmax=effective_limit)
    fig.colorbar(im,ax=axes[0],shrink=.8,pad=.025,label='有效偏应力（kPa）',extend='max')
    im=axes[1].tripcolor(triangulation,facecolors=shear[selection],shading='flat',cmap='RdBu_r',norm=colors.TwoSlopeNorm(0,-shear_limit,shear_limit))
    fig.colorbar(im,ax=axes[1],shrink=.8,pad=.025,label='水平剪切应力 τxy（kPa）',extend='both')
    near=(abs(center_xy[:,0]-jxy[0])<half)&(abs(center_xy[:,1]-jxy[1])<half)&(rows%6==0)&(cols%6==0)&(speeds[centers]>1e-5)
    for ax,title in zip(axes,['主支流汇入区：应力强度','同一区域：水平剪切应力']):
        ax.quiver(center_xy[near,0],center_xy[near,1],dirs[near,0],dirs[near,1],angles='xy',scale_units='xy',scale=17,color='#33d4b5',width=.004)
        ax.set_xlim(jxy[0]-half,jxy[0]+half);ax.set_ylim(jxy[1]-half,jxy[1]+half)
        ax.set_title(title,fontweight='bold');ax.set_xlabel('相对东向距离（km）');ax.set_ylabel('相对北向距离（km）');ax.set_aspect('equal')
    fig.suptitle('汇流区应力 · 同一个方程组中的支流与主干',fontsize=18,fontweight='bold')
    fig.savefig(REPORT/'confluence-stress.png',dpi=155);plt.close(fig)
    np.savez_compressed(REPORT/'map-data.npz',x_m=x,y_m=y,valid_mask=mask,surface_speed_m_year=speedmap,
                        confluence_xy_m=junction,triangle_effective_stress_kPa=effective,triangle_shear_xy_kPa=shear)
    stats=dict(pixel_center_speed_quantiles_m_year=dict(zip(['min','p50','p90','p95','p99','max'],map(float,np.nanquantile(speedmap,[0,.5,.9,.95,.99,1])))),
               speed_color_range_m_year=[vmin,vmax],pixels_above_speed_color_max=int(np.count_nonzero(speedmap>vmax)),
               arrow_length='normalized; direction only',origin_xy_m=origin.tolist(),confluence_xy_m=junction.tolist(),
               stress_sampling='mean of sigma .25 and .75 triangle-centroid stresses; no nodal smoothing',
               confluence_shear_color_abs_limit_kPa=shear_limit,confluence_effective_color_max_kPa=effective_limit,
               global_effective_stress_max_kPa=float(effective.max()),global_shear_abs_max_kPa=float(abs(shear).max()))
    write_json(REPORT/'plot-metadata.json',stats)
    fastest=int(speeds.argmax());verts=mesh['xy'][tri]
    jac=np.stack([verts[:,1]-verts[:,0],verts[:,2]-verts[:,0]],axis=2)
    grad=np.einsum('ij,tjk->tik',np.array([[-1.,-1.],[1.,0.],[0.,1.]]),np.linalg.inv(jac))
    surface_grad=np.einsum('ti,tij->tj',(mesh['bed']+mesh['thickness'])[tri],grad)
    adjacent=np.any(tri==fastest,axis=1)
    diagnostic=dict(status='read-only-diagnostic-not-causal-attribution',max_node=fastest,xy_m=mesh['xy'][fastest].tolist(),
        thickness_m=float(mesh['thickness'][fastest]),max_surface_speed_m_year=float(speeds[fastest]),
        adjacent_surface_slopes_m_per_m=np.linalg.norm(surface_grad[adjacent],axis=1).tolist(),
        pixel_fraction_above1000_m_year=float(np.mean(speedmap[mask]>1000)),
        pixel_fraction_above100000_m_year=float(np.mean(speedmap[mask]>100000)),
        boundary_thickness_quantiles_m=np.quantile(mesh['thickness'][np.unique(mesh['boundary_edges'])],[.5,.9,.99,1]).tolist(),
        interpretation='Very steep reconstructed surface coincides with high velocity. Does not isolate geometry, A, vertical resolution or boundary-condition causes.')
    write_json(REPORT/'high-speed-diagnostic.json',diagnostic)
    p=stats['pixel_center_speed_quantiles_m_year']
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>真实冰川 · 联合速度与汇流应力</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#eef2f5;color:#173344;font:16px/1.8 "Microsoft YaHei",sans-serif}}main{{max-width:1250px;margin:auto;padding:35px 24px}}h1{{font-size:34px;line-height:1.4}}h2{{font-size:23px}}.tag{{color:#007e77;font-weight:bold}}.card{{background:white;border:1px solid #dce5e9;border-radius:14px;padding:24px;margin:22px 0}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.stat{{background:#dbefeb;padding:18px;border-radius:12px}}strong.big{{font-size:28px;display:block}}img{{width:100%;height:auto}}small,.muted{{color:#56707e}}a{{color:#006c80}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce5e9}}@media(max-width:700px){{.stats{{grid-template-columns:1fr}}h1{{font-size:27px}}.card{{padding:15px}}}}</style>
<main><div class="tag">真实固定几何 · 首次全域联合速度诊断</div><h1>支流和主干，已在真实网格上共同求解</h1><p>系统 2387 的 {len(xy):,} 个平面节点、{len(tri):,} 个三角形全部参与求解。每个位置求解两个水平速度分量；汇流区域共享速度节点，通过整体动量方程联系。下图使用本次收敛结果。</p>
<div class="stats"><div class="stat"><strong class="big">{result['free_unknowns']:,}</strong>底部固定后的速度未知量</div><div class="stat"><strong class="big">{result['relative_residual']:.2e}</strong>最终动量相对残差</div><div class="stat"><strong class="big">{result['iterations']} 轮</strong>真实几何非线性迭代</div></div>
<section class="card"><h2>本次暴露的问题：局部速度过高</h2><p>最高表面速度约 13.7 万 m/年，不能将其直接用于真实冰川演化。高速节点附近的重建表面坡度约为 1.9–3.3 m/m，说明真实动力学几何仍需核查；流变参数移用及侧边界的影响也尚未分离。本次保留完整原始结果，供定位这些问题。</p></section><section class="card"><h2>全冰川冰厚、流速与流向</h2><p>88,274 个正式正冰厚像元全部保留，最低冰厚约为 1 微米；没有抬高冰厚下限或降低水平分辨率。面积为 21.897469 km²，总体积为 0.830676874 km³。</p>
<a href="../../reports/native-global-bp-v1/surface-velocity.png"><img src="../../reports/native-global-bp-v1/surface-velocity.png" alt="真实冰川冰厚和收敛后的二维表面速度分布及流向"></a>
<p>源像元中心的表面速度中位数为 <b>{p['p50']:.3g} m/年</b>，95% 分位数为 <b>{p['p95']:.3g} m/年</b>，最大值为 <b>{p['max']:.3g} m/年</b>。这些是模型诊断值，尚未与观测速率比较。</p>
<p><small>等长箭头仅表示方向；颜色表示速度大小。对数色标上限取源像元速度的 99% 分位数（{vmax:.3g} m/年），共有 {stats['pixels_above_speed_color_max']:,} 个像元超过色标上限，原始数值完整保留。红框是下方放大区域。</small></p></section>
<section class="card"><h2>汇流区域的应力分布</h2><a href="../../reports/native-global-bp-v1/confluence-stress.png"><img src="../../reports/native-global-bp-v1/confluence-stress.png" alt="真实主汇流区的有效偏应力和水平剪切应力分布"></a>
<p>左图显示应力强度，右图显示东—北坐标中的水平剪切应力及其正负。两幅图均从共同求解的速度梯度计算，取两个垂向单元中心的平均，三角形间没有做应力平滑。</p>
<p><small>色标分别截取区域 98% 分位数，超出部分显示端点色；没有修改数据。共享速度节点实现离散弱形式中的内部受力平衡，但逐界面应力通量误差尚未独立评估，图像本身不能证明点态应力连续。</small></p></section>
<section class="card"><h2>这次采用的物理条件</h2><table><tr><th>项目</th><th>本次设定</th></tr>
<tr><td>流变</td><td>Glen n = 3；A = 7.53880198 × 10⁻²³ Pa⁻³ s⁻¹，沿用项目已有常数值</td></tr>
<tr><td>底部</td><td>无滑动（fs = 0），底部两个水平速度严格为零</td></tr>
<tr><td>表面及侧边界</td><td>BP 弱形式的自然有效应力边界；原始像元阶梯外边界保留</td></tr>
<tr><td>垂向</td><td>3 个节点，2 层单元；共 {result['unknowns']:,} 个速度未知量，底部约束后 {result['free_unknowns']:,} 个</td></tr>
<tr><td>热力及时间</td><td>常数 A，不反馈温度；几何固定，推进时间 0 年</td></tr></table>
<p>本次速度求解已经收敛，但源重建几何、仅两层垂向单元及侧边界条件的影响仍需评估。A 来自已有一维模型标定，尚未针对新的空间模型重新标定。这里交付的是可检查的真实动力学诊断，尚未达到科学验收。</p></section>
<section class="card"><h2>合并检查与可恢复结果</h2><p>初版在 80 轮上限停止，残差未达原定阈值，旧结果与失败记录已保留。技术修正版处理总能量变化过小时的步长判定，并检查缩放前的线性残差；几何、物理参数和最终收敛阈值均未改变。</p><p>4 项检查全部通过：新旧离散方程一致、极薄冰缩放有限、合成算例 AMG 求解与独立进程恢复逐位一致，以及真实源配对、网格身份和体积核对。真实计算耗时约 {result['wall_seconds']/60:.1f} 分钟，每轮迭代均保存速度状态。</p>
<p><a href="../../runs/v7-native-global-bp-r1/result.json">真实计算记录</a> · <a href="../../runs/v7-native-global-bp-r1/fixture/verification.json">合并检查</a> · <a href="../../runs/v7-native-global-bp-r1/fields.npz">完整速度与应力数组</a> · <a href="../../reports/native-global-bp-v1/plot-metadata.json">色标与统计定义</a></p>
<p>下一步：先核查高速区的重建表面坡度、流变参数移用及侧边界影响，再评估主汇流区的垂向分辨率，之后接入守恒输运与热力更新。</p></section><footer class="muted">PoLIM 扩展研究分支 · 不改变主线验收数量。<a href="../../reports/global-bp-v1/index.html">上一版合成应力传递结果</a></footer></main></html>'''
    (REPORT/'index.html').write_text(content,encoding='utf-8')
    for filename in ('index.html','DELIVERY.md','delivery-evidence.json'):
        src=ROOT/filename;dst=src.with_name(src.stem+'-global-bp-v1'+src.suffix)
        if src.exists() and not dst.exists():shutil.copy2(src,dst)
    (ROOT/'index.html').write_text(content.replace('../../',''),encoding='utf-8')
    (ROOT/'DELIVERY.md').write_text(f'''# 真实冰川联合速度与汇流应力\n\n系统 2387 原生网格固定几何求解完成，残差 {result['relative_residual']:.5g}，{result['iterations']} 轮迭代。\n\n入口：reports/native-global-bp-v1/index.html。真实原始结果：runs/v7-native-global-bp-r1/result.json 和 fields.npz。\n\n常数项目 A、n3、底部无滑动、自然侧边界；无厚度下限、无水平粗化。未推进时间或反馈温度，未通过观测验证或科学验收。\n''',encoding='utf-8')
    (ROOT/'delivery-evidence.json').write_text(json.dumps(dict(status='real-fixed-geometry-diagnostic-completed',report=str(REPORT.relative_to(PROJECT)/'index.html'),
                result=dict(path=str((RUN/'result.json').relative_to(PROJECT)),sha256=digest(RUN/'result.json')),
                scientific_acceptance=False,real_time_advanced=False),ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='rendered',report=str(REPORT/'index.html'),statistics=stats)))


if __name__=='__main__':main()
