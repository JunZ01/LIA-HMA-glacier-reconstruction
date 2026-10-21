from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parent;O=R/'reports/ice-bed-profiles-v1';O.mkdir(exist_ok=False)
gpath=R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz';g=np.load(gpath);a=np.load(R/'reports/geometry-source-audit-v1/audit-fields.npz');info=json.loads((R/'reports/geometry-source-audit-v1/audit.json').read_text(encoding='utf-8'));c=np.load(R/'reports/branch0-production-trace-v1/connector-0.npz')
x,y=g['x'],g['y'];B=g['bed_elevation_m'];H=g['thickness_m'];S=g['surface_elevation_m'];D=a['dem'];r,col=238,222;A=np.array(info['peak_xy']);Ap=np.array([x[col],y[r]])
xy=c['xy'];v=np.diff(xy,axis=0);t=np.clip(np.sum((A-xy[:-1])*v,axis=1)/np.sum(v*v,axis=1),0,1);q=xy[:-1]+t[:,None]*v;dist=np.linalg.norm(q-A,axis=1);j=int(dist.argmin())
s=c['station'][-1]-c['station'][::-1];bb=c['B'][::-1];hh=c['H'][::-1];ss=bb+hh;dd=c['DEM'][::-1]
rr=np.arange(r-32,r+33);off=y[rr]-y[r];sort=np.argsort(off);rr=rr[sort];off=off[sort]
assert hashlib.sha256(gpath.read_bytes()).hexdigest()=='59140635db4b67ef14099b273d5fdbc2dc6e091a4460c54720c68748223e4768'
assert np.isfinite(B[rr,col]).all() and np.isfinite(S[rr,col]).all()
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
fig=plt.figure(figsize=(14,12),facecolor='#f7f9fc');gs=fig.add_gridspec(2,2,height_ratios=[.9,1.15],hspace=.38,wspace=.28,left=.08,right=.96,bottom=.10,top=.88)
ax=fig.add_subplot(gs[0,:]);ax.fill_between(s,bb,ss,color='#bde8f4',alpha=.85);ax.plot(s,ss,color='#087ea4',lw=2.4,label='重建冰面');ax.plot(s,bb,color='#695844',lw=2.2,label='推算冰床');ax.plot(s,dd,color='#969696',lw=1.3,ls='--',label='区域 DEM（背景参照）');ax.set_aspect('equal',adjustable='box');ax.set_xlim(-40,s[-1]+40);ax.set_ylim(min(bb.min(),dd.min())-45,max(ss.max(),dd.max())+115);ax.grid(alpha=.2);ax.set_xlabel('沿原连接线的水平累计距离：T → P（m）');ax.set_ylabel('海拔（m）');ax.set_title('① 原连接线纵剖面：冰面整体缓降，冰床却向 P 升高',loc='left',fontsize=14,pad=14)
ax.text(20,ss[0]+45,'T 支流末端',color='#087ea4',weight='bold');ax.text(s[-1]-20,ss[-1]+45,'P 主干连接点',ha='right',color='#087ea4',weight='bold');ax.annotate('',(20,ss[0]),(20,bb[0]),arrowprops=dict(arrowstyle='<->',color='#35495d'));ax.text(55,(ss[0]+bb[0])/2,'冰厚约 396 m',va='center');ax.text(s[-1]-70,bb[-1]-65,'约 13 m',ha='right');ax.legend(loc='lower center',ncol=3,frameon=False,fontsize=10)
ax=fig.add_subplot(gs[1,0]);ax.fill_between(off,B[rr,col],S[rr,col],color='#bde8f4',alpha=.85)
for name,F,color,ls in [('冰面',S,'#087ea4','-'),('冰床',B,'#695844','-'),('区域 DEM',D,'#969696','--')]:ax.plot(off,F[rr,col],color=color,ls=ls,lw=2,label=name)
ax.set_aspect('equal',adjustable='box');ax.axvline(0,color='#ce334b',ls=':',lw=1.3);ax.axvspan(0,15.75,color='#ce334b',alpha=.15);ax.scatter([0,15.75],[S[r,col],S[r-1,col]],s=24,color='#ce334b',zorder=8)
ax.annotate('单个网格间隔 15.75 m\n冰面高差 43.6 m',(7.875,(S[r,col]+S[r-1,col])/2),xytext=(145,S[r,col]+90),fontsize=10,color='#a82c41',arrowprops=dict(arrowstyle='->',color='#a82c41'))
ax.set_xlim(-530,530);ax.set_ylim(min(B[rr,col].min(),D[rr,col].min())-45,max(S[rr,col].max(),D[rr,col].max())+100);ax.grid(alpha=.2);ax.set_xlabel('距 A′ 的南北距离（m；正值向北）');ax.set_ylabel('海拔（m）');ax.set_title('② 高速点旁的南北剖面\nA′ 为最近源网格中心',loc='left',fontsize=14,pad=12);ax.legend(frameon=False,fontsize=9,loc='lower left')
ax=fig.add_subplot(gs[1,1]);X,Y=np.meshgrid(x,y);im=ax.pcolormesh((X-Ap[0])/1000,(Y-Ap[1])/1000,np.ma.masked_where(~g['valid_mask'].astype(bool)|(H<=0),H),shading='nearest',cmap='viridis',vmin=0,vmax=400,rasterized=True)
ax.plot((xy[:,0]-Ap[0])/1000,(xy[:,1]-Ap[1])/1000,color='#f65b63',lw=2,label='原连接线');ax.plot([0,0],[-.504,.504],color='#f4a72c',lw=2,label='南北剖面');ax.scatter((A[0]-Ap[0])/1000,(A[1]-Ap[1])/1000,marker='*',s=180,color='#fa354f',edgecolors='white',linewidths=.7,label='原高速点 A',zorder=6)
for k,l in [(0,'P'),(-1,'T')]:ax.annotate(l,((xy[k,0]-Ap[0])/1000,(xy[k,1]-Ap[1])/1000),xytext=(9,-15 if k==-1 else 7),textcoords='offset points',color='#bf213b',weight='bold',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=1.5))
ax.set_xlim(-.75,1.75);ax.set_ylim(-1.4,.9);ax.set_aspect('equal');ax.set_xlabel('相对 A′ 的东西距离（km）');ax.set_ylabel('相对 A′ 的南北距离（km）');ax.set_title('③ 两条剖面的位置\n高速点不在原连接线上',loc='left',fontsize=14,pad=12);ax.legend(loc='upper right',framealpha=.9,fontsize=9);cb=fig.colorbar(im,ax=ax,shrink=.65,pad=.025);cb.set_label('冰厚（m）')
fig.suptitle('冰面是否陡峭？沿线剖面与局部坡面分开看',x=.08,ha='left',y=.97,fontsize=21);fig.text(.08,.923,'原始古冰川重建 · 两个高程剖面均横纵 1∶1，无垂直夸大 · 未平滑、未运行新模拟',fontsize=12,color='#526174')
fig.text(.08,.04,'① 冰面由原连接线 B + H 恢复；② 使用存档二维冰面与冰床。A′ 距 A %.1f m。\n区域 DEM 含现代冰面，不能当作独立实测冰床；局部坡差也不能直接换算成冰速成因比例。'%np.linalg.norm(A-Ap),fontsize=10,color='#526174')
fig.savefig(O/'profiles.png',dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
summary=dict(new_simulations=0,source_sha256=hashlib.sha256(gpath.read_bytes()).hexdigest(),vertical_exaggeration=1,connector_length_m=float(s[-1]),T_to_P_surface_change_m=float(ss[-1]-ss[0]),T_to_P_bed_change_m=float(bb[-1]-bb[0]),T_H_m=float(hh[0]),P_H_m=float(hh[-1]),endpoint_mean_surface_angle_deg=float(np.degrees(np.arctan(abs(ss[-1]-ss[0])/s[-1]))),A_xy=A.tolist(),sample_grid_xy=Ap.tolist(),A_to_sample_m=float(np.linalg.norm(A-Ap)),A_to_original_connector_m=float(dist[j]),local_north_edge=dict(horizontal_m=15.75,DEM_change_m=float(D[r-1,col]-D[r,col]),bed_change_m=float(B[r-1,col]-B[r,col]),thickness_change_m=float(H[r-1,col]-H[r,col]),surface_change_m=float(S[r-1,col]-S[r,col])),notes=['Endpoint angle is not maximum local slope.','Grid edge is not full 2D gradient.','Source geometry remains unchanged.'])
(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(O/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>冰床与冰面等比例剖面</title><style>body{margin:0;background:#f7f9fc;font:16px sans-serif}img{display:block;width:100%;max-width:1600px;margin:auto}p{max-width:1100px;margin:20px auto;padding:0 25px;line-height:1.8}</style><img src="profiles.png" alt="冰床与冰面等比例剖面"><p>原连接线长约 1.78 km，两端冰面高差约 127 m，平均倾角约 4.1°。高速点附近另有短距离陡坡。两者是不同位置、不同方向和尺度的几何特征。</p></html>',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
