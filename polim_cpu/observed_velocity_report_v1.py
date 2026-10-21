"""Complete the read-only product from saved arrays, including null-safe profiles."""
import json,math
from pathlib import Path
import numpy as np
from pyproj import Transformer
from rasterio.transform import Affine
from .observed_velocity_v1 import OUT,PACKAGE,PROJECT,readnp,stats
from .v3.runner import write_json,digest

def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    return value

def main():
    a=readnp(OUT/'comparison-fields.npz');s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    p=json.loads(PACKAGE.read_text(encoding='utf-8'));local=p['metadata']['crs_wkt'];forward=Transformer.from_crs(local,s['observation_grid']['crs'],always_xy=True);back=Transformer.from_crs(s['observation_grid']['crs'],local,always_xy=True)
    ot=Affine(*s['observation_grid']['transform'][:6]);v=a['observed'];err=a['error'];core=(a['lia_fraction']>=.5)&(a['modern_fraction']>=.5);valid=a['valid'];positive_error=valid&np.isfinite(err)&(err>0);snr=positive_error&(v>2*err)
    # Zero-valued integer errors are retained as quantized values, never treated as exact certainty.
    s['quality_note']='v_error is uint16 at1m/year storage precision with scale1. Zero does not establish zero uncertainty. No SNR qualification is assigned at error0; all valid velocity values retained.'
    for name,mask in [('modern_overlap',core),('lia_footprint',a['lia_fraction']>=.5)]:
        row=s['subsets'][name];row.pop('resolved_cells');row.pop('resolved_direction_difference_degrees');row['zero_stored_error_cells']=int((mask&valid&(err==0)).sum());row['positive_error_snr_gt2_cells']=int((mask&snr).sum())
    s['method']['quality']='Finite nonnegative v,finite vx/vy,count>0. Preserve low velocities. Integer error0 is uncertainty-unresolved; only positive errors permit an auxiliary v/error>2 screen. No scientific acceptance threshold.'
    s['method']['profiles']='Sample the saved120m aggregated model and120m observed fields on original LIA branch paths; show observations only within modern mask. Repeated cells are visual samples,not independent measurements.'
    s['status']='completed-read-only-comparison';s['supersedes_summary']='summary.json,whose zero-error SNR count is withdrawn; primary speeds,coverage and geometry statistics unchanged.'
    s['export_recovery']='Original profile JSON serialization stopped on NaN after all spatial arrays and summary were saved. This exporter reuses saved arrays, encodes missing values as null,and does not rerun a model or alter source files. Empty original profiles.json retained.'
    x,y=a['x'],a['y'];dx=x[1]-x[0];dy=y[1]-y[0];profiles=[]
    for f in sorted((PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired').glob('branch-*.npz')):
        b=readnp(f);bx,by=b['easting_m'],b['northing_m'];ox,oy=forward.transform(bx,by);cc,rr=(~ot)*(ox,oy);r=np.floor(rr).astype(int);c=np.floor(cc).astype(int);sr=np.rint((by-y[0])/dy).astype(int);sc=np.rint((bx-x[0])/dx).astype(int)
        modern=a['modern_mask'][sr,sc];obs=v[r,c].copy();obs[~valid[r,c]|~modern]=np.nan;error=err[r,c].copy();error[~np.isfinite(obs)]=np.nan
        profiles.append(dict(branch=int(b['branch_index'].flat[0]),station_m=b['station_m'].tolist(),x=bx.tolist(),y=by.tolist(),observed=obs.tolist(),error=error.tolist(),model_day1=a['model_day1'][r,c].tolist(),H=a['H_day1'][r,c].tolist(),slope=a['slope_day1_deg'][r,c].tolist(),modern=modern.tolist()))
    write_json(OUT/'profiles-v2.json',clean(profiles))
    ox,oy=forward.transform(a['local_x'],a['local_y']);rx,ry=back.transform(ox,oy);roundtrip=float(np.nanmax(np.hypot(rx-a['local_x'],ry-a['local_y'])))
    if roundtrip>.01:raise ValueError('Coordinate roundtrip failed')
    checks=dict(original_observation_hashes_verified=True,coordinate_roundtrip_error_m=roundtrip,native_observation_cells_not_upsampled=True,all_common_cells_have_positive_count=bool(np.all(a['count'][core&valid]>0)),source_fields_unchanged=True,new_simulations=0,missing_profile_values_encoded_as_null=True)
    write_json(OUT/'combined-check.json',checks);s['checks']=checks;s['report_code_sha256']=digest(Path(__file__));write_json(OUT/'summary-v2.json',s)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    origin=float(y.min());lx=a['local_x']/1000;ly=(a['local_y']-origin)/1000;X,Y=np.meshgrid(x/1000,(y-origin)/1000)
    extent=[(x[0]-dx/2)/1000,(x[-1]+dx/2)/1000,(y[-1]+dy/2-origin)/1000,(y[0]-dy/2-origin)/1000]
    peak=np.array(s['peak']['xy']);jx,jy=-134.49697636368546,3158715.5030236365
    def decorate(ax,labels=False):
        ax.contour(X,Y,a['lia_mask'],levels=[.5],colors='#8093a0',linewidths=.6)
        ax.contour(X,Y,a['modern_mask'],levels=[.5],colors='#d97706',linewidths=.75)
        ax.plot(peak[0]/1000,(peak[1]-origin)/1000,'*',ms=10,color='#dc2626',markeredgecolor='white',markeredgewidth=.4);ax.text(peak[0]/1000+.1,(peak[1]-origin)/1000,'P',fontsize=9)
        ax.plot(jx/1000,(jy-origin)/1000,'o',ms=6,mfc='none',mec='#dc2626');ax.text(jx/1000+.1,(jy-origin)/1000,'J',fontsize=9)
        ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='投影东向（km）',ylabel='相对北向（km）');ax.set_aspect('equal');ax.set_facecolor('#f4f6f8')
        if labels:
            for b in profiles:
                xx=np.array(b['x'])/1000;yy=(np.array(b['y'])-origin)/1000;ax.plot(xx,yy,color='#111827',lw=.6,alpha=.7);i=len(xx)//3;ax.text(xx[i],yy[i],str(b['branch']),fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
    observed=np.where(core&valid,v,np.nan);model=np.where(a['lia_fraction']>=.5,a['model_day1'],np.nan);norm=LogNorm(.01,3000)
    fig,axs=plt.subplots(2,2,figsize=(13,10.5),constrained_layout=True)
    for ax,z,title,cmap,n in zip(axs.ravel(),[observed,model,a['H_day1'],a['slope_day1_deg']],['现代卫星流速 · 2014—2024 气候态','古冰川几何 BP · 松弛第 1 天','模型冰厚（m）','模型表面坡度（°）'],['viridis','viridis','Blues','magma'],[norm,norm,None,None]):
        z=np.where(a['lia_fraction']>=.5,z,np.nan);kw={} if n else ({'vmin':0,'vmax':400} if '冰厚' in title else {'vmin':0,'vmax':70})
        im=ax.pcolormesh(lx,ly,z,shading='nearest',cmap=cmap,norm=n,**kw);fig.colorbar(im,ax=ax,shrink=.8,label='米/年' if n else None);ax.set_title(title);decorate(ax,labels=ax is axs[0,0])
    fig.suptitle('同一坐标与 120 米观测尺度 · 橙线为现代边界，灰线为古冰川范围\nP：模型最高速度位置；J：主要汇流点；数字：分支编号',fontsize=13)
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,5.4),constrained_layout=True)
    for ax,z,title,limit in zip(axs,[a['modern_dem_slope_deg'],a['slope_day1_deg'],a['H_day1']],['现代 DEM 坡度（°）','古冰川模型表面坡度（°）','古冰川模型冰厚（m）'],[70,70,400]):
        im=ax.pcolormesh(lx,ly,np.where(a['lia_fraction']>=.5,z,np.nan),shading='nearest',cmap='magma' if limit==70 else 'Blues',vmin=0,vmax=limit);fig.colorbar(im,ax=ax,shrink=.75);decorate(ax);ax.set_title(title)
    fig.savefig(OUT/'geometry.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
    for ax,b in zip(axs.ravel(),profiles):
        st=np.array(b['station_m'])/1000;obs=np.array(b['observed']);mod=np.array(b['model_day1']);ax.plot(st,np.maximum(mod,.001),color='#c2410c',label='模型第1天');ax.step(st,np.maximum(obs,.001),where='mid',color='#0369a1',label='现代观测');ax.set_yscale('log');ax.set_ylim(.001,4000);ax.set_title('分支 '+str(b['branch']));ax.set_xlabel('沿分支距离（km）');ax.set_ylabel('表面速度（米/年）');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=8);fig.suptitle('各分支同位置对照 · 观测缺口表示现代冰川范围外或缺测\n对数坐标显示下限 0.001 米/年，原始值保存在数据中',fontsize=13);fig.savefig(OUT/'branches.png',dpi=150);plt.close(fig)
    for b in profiles:
        fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True);st=np.array(b['station_m'])/1000;obs=np.array(b['observed']);mod=np.array(b['model_day1']);ax=axs[0];ax.plot(st,np.maximum(mod,.001),label='古冰川模型第1天',color='#c2410c');ax.step(st,np.maximum(obs,.001),where='mid',label='现代卫星观测',color='#0369a1');ax.set(yscale='log',ylim=(.001,4000),ylabel='米/年',title='表面速度');ax.legend(fontsize=8)
        axs[1].plot(st,b['H'],color='#0369a1');axs[1].set(ylabel='m',title='模型冰厚');axs[2].plot(st,b['slope'],color='#9333ea');axs[2].set(ylabel='度',title='模型表面坡度')
        for ax in axs:ax.set_xlabel('沿分支距离（km）');ax.grid(alpha=.2)
        fig.suptitle('分支 '+str(b['branch'])+' · 120米尺度抽样');fig.savefig(OUT/f'branch-{b["branch"]:02d}.png',dpi=130);plt.close(fig)
    m=s['subsets']['modern_overlap'];j=s['junction'];memberhtml=''.join(f'<tr><td>{r["rgi_id"]}</td><td>{r["valid_cells"]}</td><td>{r["observed"]["median"]:.2f}</td><td>{r["observed"]["max"]:.2f}</td><td>{r["model_day1"]["median"]:.3g}</td><td>{r["model_day1"]["max"]:.2f}</td></tr>' for r in s['members'])
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 观测流速对照</title><style>*{box-sizing:border-box}body{font-family:Microsoft YaHei,sans-serif;background:#eff4f7;color:#183546;margin:0}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px}p{line-height:1.8}.panel{background:white;border-radius:14px;padding:22px;margin:20px 0}.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}.value{font-size:28px;color:#12638a}.label,.note{color:#5c7281;font-size:13px}img{width:100%;height:auto}select{padding:10px;border-radius:6px;border:1px solid #c7d2dc;font:inherit}td,th{padding:12px;text-align:left;border-bottom:1px solid #e5edf2}table{border-collapse:collapse;width:100%}a{color:#116b9a}button{font:inherit} @media(max-width:750px){main{padding:12px}.stats{grid-template-columns:1fr}table{font-size:12px}td,th{padding:6px}}</style><main><h1>2387 冰川 · 现代观测与古冰川试算对照</h1><p>ITS_LIVE 2014—2024 气候态流速，对照原始古冰川几何松弛一天的 BP 结果。模型汇总至 120 米观测像元；各像元只统计一次。此次没有重跑模型或调整参数。</p>
<div class="panel"><b>主要发现：速度分布不只是整体偏快。</b><p>共同范围内，观测中位数约 5.2 米/年，模型中位数仅约 0.017 米/年；但模型局部又达到约 2768 米/年。主要汇流点周围，观测中位数约 14.7 米/年，模型约 0.28 米/年。高速区和过慢区同时存在，不能靠统一调低材料流动性解决。</p><p class="note">这里是跨年代的空间参照，不是现代几何模拟的误差检验，也不能直接证明古冰川速度错误。下一步应优先核对各支流的冰厚分配、表面坡度和底部条件。</p></div>
<div class="panel stats"><div><div class="label">共同范围观测最高速度</div><div class="value">114.6 米/年</div></div><div><div class="label">同尺度模型最高速度</div><div class="value">2768.3 米/年</div></div><div><div class="label">共同范围有效观测像元</div><div class="value">953 / 953</div></div></div>
<div class="panel"><img src="comparison.png" alt="现代观测速度、古冰川模拟速度、冰厚和坡度同坐标对照"><p class="note">两幅速度图共用对数色标。观测只显示现代与古冰川共同覆盖范围；模型显示古冰川范围。120米汇总后的最大值与原网格节点最大值2874米/年不同。覆盖完整表示有数据，并不表示所有数据都同样准确。</p></div>
<div class="panel"><b>最高速度点 P 位于现代边界外，所在120米像元约47%属于现代冰川。</b><p>该点模型冰厚约290米，模型表面局部坡度约62°，现代DEM局部坡度约30°。这是需要重点核对的几何差异。P像元是冰川边缘混合像元，因此不把这里的观测数值直接当作该古冰川点的真值。</p><img src="geometry.png" alt="现代DEM坡度与模型坡度和冰厚"></div>
<div class="panel"><label for="branch">查看分支：</label><select id="branch">OPTIONS</select><img id="profile" src="branch-00.png" alt="所选支流的速度冰厚坡度"><p class="note">使用原有古冰川分支路径。蓝线缺口表示现代冰川范围外或缺测；低于0.001米/年的模型值只在对数图底部显示，原值保留。</p><details><summary>同时查看全部分支</summary><img src="branches.png" alt="八个分支的流速对照"></details></div>
<div class="panel"><b>按现代冰川成员对照（米/年）</b><table><tr><th>成员</th><th>像元数</th><th>观测中位数</th><th>观测最大值</th><th>模型中位数</th><th>模型最大值</th></tr>MEMBERTABLE</table><p class="note">成员按现代轮廓独立筛选，边界混合像元可能在成员之间重叠；成员行数不能相加代替系统共同范围953像元。</p></div>
<details class="panel"><summary>数据与限制</summary><p>观测：ITS_LIVE v2.1，120米，2014—2024气候态，时间截距为2018年1月1日；并非2024单年平均。数据直接复用项目已下载且哈希一致的官方栅格。</p><p>误差层为整数米/年，零值不能解释为零不确定性。保留所有有效低流速，仅在误差为正时附带计算信噪比诊断。现代边界来自项目原有RGI轮廓，不是2024年专门测绘的冰缘。</p><p>模型：原始古冰川几何、固定材料与冰床、底部无滑动、一天零质量源项松弛。坡度为网格差分结果，陡坡与边缘对分辨率敏感。这张对照图不能独立区分冰厚、温度和底部滑动的贡献。</p><p><a href="https://its-live.jpl.nasa.gov/">NASA ITS_LIVE 产品说明</a> · <a href="summary-v2.json">统计与来源记录</a> · <a href="comparison-fields.npz">空间数据</a> · <a href="profiles-v2.json">分支数据</a></p></details><p><a href="../fixed-material-relaxation-v1/index.html">返回一天动力松弛</a> · <a href="../../index.html">项目首页</a></p></main><script>const selector=document.getElementById('branch');selector.addEventListener('change',()=>{document.getElementById('profile').src='branch-'+selector.value.padStart(2,'0')+'.png'});</script></html>'''
    html=html.replace('OPTIONS',''.join(f'<option value="{i}">分支 {i}</option>' for i in range(8))).replace('MEMBERTABLE',memberhtml)
    (OUT/'index.html').write_text(html,encoding='utf-8')
    write_json(OUT/'delivery.json',dict(status='completed',summary_sha256=digest(OUT/'summary-v2.json'),files={f.relative_to(OUT).as_posix():digest(f) for f in OUT.rglob('*') if f.is_file()},new_simulations=0))
    print(json.dumps(dict(status='delivered',path=str(OUT/'index.html'),common=m,junction=j),ensure_ascii=False))

if __name__=='__main__':main()
