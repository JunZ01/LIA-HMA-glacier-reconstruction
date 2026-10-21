"""Read-only decomposition of saved A-point geometry. No reconstruction or BP solve."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parent
O=R/'reports/peak-geometry-analysis-v1'
O.mkdir(exist_ok=False)
paths=[R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz',R/'reports/geometry-source-audit-v1/audit-fields.npz',R/'reports/geometry-source-audit-v1/audit.json',R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/branch-00.npz',R/'reports/constrained-bed-seam-v1/data.js']
g=np.load(paths[0],allow_pickle=False);a=np.load(paths[1],allow_pickle=False);info=json.loads(paths[2].read_text(encoding='utf-8'));p=np.load(paths[3],allow_pickle=False)
r,c=info['nearest_source_row'],info['nearest_source_col'];x,y=g['x'],g['y'];dx=x[1]-x[0];dy=y[1]-y[0];B=g['bed_elevation_m'].astype(float);H=g['thickness_m'].astype(float);S=g['surface_elevation_m'].astype(float);D=a['dem'];cor=B-D
scales=[]
for k in [1,2,4,8,16]:
 v={n:np.array([(f[r,c+k]-f[r,c-k])/(2*k*dx),(f[r+k,c]-f[r-k,c])/(2*k*dy)]) for n,f in [('DEM',D),('bed_correction',cor),('H',H),('S',S)]};u=v['S']/np.linalg.norm(v['S']);scales.append(dict(span_m=float(2*k*dx),surface_slope_degrees=float(np.degrees(np.arctan(np.linalg.norm(v['S'])))),projected_gradient={n:float(np.dot(q,u)) for n,q in v.items()},gradient_identity_error=float(np.linalg.norm(v['DEM']+v['bed_correction']+v['H']-v['S']))))
delta={n:float(f[r,c]-f[r-1,c]) for n,f in [('DEM',D),('bed_correction',cor),('bed',B),('H',H),('S',S)]};delta['spacing_m']=float(abs(dy));delta['surface_angle_degrees']=float(np.degrees(np.arctan(abs(delta['S']/dy))))
d=np.hypot(p['easting_m']-info['peak_xy'][0],p['northing_m']-info['peak_xy'][1]);j=int(np.argmin(d));near=(d<300)
summary=dict(status='completed-read-only-spatial-analysis',new_simulations=0,source_hashes={str(v.relative_to(R)):hashlib.sha256(v.read_bytes()).hexdigest() for v in paths},source_grid_point=dict(row=r,col=c,x=float(x[c]),y=float(y[r]),H=float(H[r,c]),modern=bool(a['modern_mask'][r,c])),edge=delta,gradient_scales=scales,nearby_path_samples=dict(branch=0,nearest_distance_m=float(d[j]),nearest_H_m=float(p['centreline_thickness_m'][j]),max_H_within300m=float(p['centreline_thickness_m'][near].max()),meaning='source-raster samples along saved original and connector route, not independent observations'),interpretation=['Boundary bed correction adds a one-cell step; H gradient adds in same direction.','Large-scale surface gradient persists beyond the seam.','Existing bed-seam BP control did not lower old-peak speed; smoothing-only solution is not supported.','Nearby high H is already in the saved source along branch0/connector; absolute thickness error is not established.','Projected gradient contributions are geometric decomposition, not causal velocity percentages.'])
(O/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False})
fig,axs=plt.subplots(2,2,figsize=(12,9));rr=np.arange(r-32,r+33);station=(y[rr]-y[r]);order=np.argsort(station);rr=rr[order];station=station[order]
ax=axs[0,0]
for name,F,color in [('区域 DEM',D,'#a16207'),('推算冰床',B,'#475569'),('古冰面',S,'#0369a1')]:ax.plot(station,F[rr,c],label=name,color=color,lw=1.8)
ax.axvline(0,color='#dc2626',ls='--');ax.set(title='穿过 A 最近源网格的南北剖面',ylabel='高程（米）',xlabel='相对位置（米；正值向北）');ax.legend(fontsize=9);ax.grid(alpha=.2)
ax=axs[0,1];ax.plot(station,H[rr,c],color='#0369a1');ax.axvline(0,color='#dc2626',ls='--');ax.set(title='同一剖面的古冰厚：变化延续数百米',ylabel='冰厚（米）',xlabel='相对位置（米；正值向北）');ax.grid(alpha=.2)
ax=axs[1,0];ri=np.arange(r-2,r+3);ss=y[ri]-y[r]
for name,F,color in [('区域 DEM',D,'#a16207'),('推算冰床',B,'#475569'),('古冰面',S,'#0369a1')]:ax.plot(ss,F[ri,c],'-o',label=name,color=color)
ax.axvspan(0,abs(dy),color='#fee2e2',alpha=.6);ax.set(title='跨冰缘 15.75 米：冰面高差 %.1f 米'%abs(delta['S']),ylabel='高程（米）',xlabel='相对位置（米）');ax.legend(fontsize=8);ax.grid(alpha=.2)
ax=axs[1,1];bottom=np.zeros(len(scales))
for n,label,color in [('DEM','区域地形','#a16207'),('bed_correction','冰床修正','#64748b'),('H','古冰厚变化','#0284c7')]:
 vals=np.array([v['projected_gradient'][n] for v in scales]);ax.bar(np.arange(len(scales)),vals,bottom=bottom,label=label,color=color);bottom+=vals
ax.set_xticks(range(len(scales)),[str(v['span_m']) for v in scales]);ax.set(title='冰面坡度分解（不代表流速因果比例）',xlabel='计算高差的水平跨度（米）',ylabel='沿冰面梯度方向的贡献（米/米）');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('A 点：冰缘接缝与宽范围厚度梯度叠加',fontsize=15);fig.tight_layout();fig.savefig(O/'diagnosis.png',dpi=150)
print(json.dumps(summary,ensure_ascii=False))
