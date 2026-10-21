from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parent;O=R/'reports/junction-terrain-check-v1';d=json.loads((O/'summary.json').read_text(encoding='utf-8'));g=np.load(R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz');a=np.load(R/'reports/geometry-source-audit-v1/audit-fields.npz');m=np.load(R/'lia-inputs-v1/system-2387/modern-raw-f19-bp-control-v1/geometry.npz');obs=np.load(R/'reports/observed-velocity-v1/comparison-fields-v2.npz');net=json.loads((R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json').read_text(encoding='utf-8'));old=np.load(R/'reports/branch0-production-trace-v1/connector-0.npz');tr=np.array(next(b for b in net['branches'] if b['branch_index']==7)['coords_projected_m']);pi=d['old_main_index'];qi=d['candidates']['raw_F19_bed']['main_native_index'];ds=float(np.linalg.norm(np.diff(tr[pi:qi+1],axis=0),axis=1).sum());P=np.array(d['old_main_xy']);T=np.array(d['tributary_endpoint']);Q=np.array(d['candidates']['raw_F19_bed']['main_xy']);origin=g['y'].min();X,Y=np.meshgrid(g['x']/1000,(g['y']-origin)/1000);F=np.where(m['mask'],m['bed'],a['dem']);rc=old['rc'][::-1];ox=old['xy'][::-1];os=np.r_[0,np.cumsum(np.linalg.norm(np.diff(ox,axis=0),axis=1))];oh=F[rc[:,0],rc[:,1]];candidate=np.load(O/'raw_F19_bed.npz');checks=[]
for name in d['candidates']:
 z=np.load(O/(name+'.npz'));rr,cc=z['rc'].T;steps=np.diff(z['rc'],axis=0);diagonal=np.all(steps!=0,axis=1);s=z['rc'][:-1][diagonal];delta=steps[diagonal];corners=np.all(g['lia_mask'][s[:,0]+delta[:,0],s[:,1]]) and np.all(g['lia_mask'][s[:,0],s[:,1]+delta[:,1]]);checks.append(dict(field=name,inside_LIA=bool(np.all(g['lia_mask'][rr,cc])),eight_neighbour=bool(np.all(np.abs(steps)<=1)),no_corner_cut=bool(corners),finite=bool(np.all(np.isfinite(z['height'])))))
assert all(all(v for k,v in c.items() if k!='field') for c in checks)
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False});fig,axs=plt.subplots(1,2,figsize=(13,6));ax=axs[0];im=ax.pcolormesh(X,Y,np.where(g['lia_mask'],a['dem'],np.nan),cmap='terrain',vmin=4800,vmax=6100,shading='nearest');fig.colorbar(im,ax=ax,shrink=.8,label='区域地形高程（米）')
for b in net['branches']:
 q=np.array(b['coords_projected_m']);ax.plot(q[:,0]/1000,(q[:,1]-origin)/1000,color='#64748b',lw=1,alpha=.65)
ax.plot(tr[:,0]/1000,(tr[:,1]-origin)/1000,color='#2563eb',label='原主干',lw=2);ax.plot(ox[:,0]/1000,(ox[:,1]-origin)/1000,color='#dc2626',lw=2,label='原连接 T→P');q=candidate['xy'];ax.plot(q[:,0]/1000,(q[:,1]-origin)/1000,color='#7c3aed',lw=2,label='地形候选 T→Q（未采用）')
sel=obs['valid'].astype(bool)&(obs['modern_fraction']>=.5)&(obs['error']>0)&(obs['observed']>2*obs['error'])&(obs['local_x']>-3100)&(obs['local_x']<400)&(obs['local_y']>3158300)&(obs['local_y']<3161000);ax.quiver(obs['local_x'][sel]/1000,(obs['local_y'][sel]-origin)/1000,obs['observed_direction_x'][sel]*.13,obs['observed_direction_y'][sel]*.13,angles='xy',scale_units='xy',scale=1,color='black',width=.005,zorder=7,label='可信度筛选后的现代流向')
for label,xy,off in [('T 支流末端',T,(-.45,.2)),('P 原接入点',P,(-.65,-.3)),('Q 地形候选',Q,(.07,-.22))]:loc=(xy[0]/1000,(xy[1]-origin)/1000);ax.scatter(*loc,color='black',s=28,zorder=8);ax.annotate(label,loc,(loc[0]+off[0],loc[1]+off[1]),fontsize=9,bbox={'fc':'white','ec':'none','alpha':.9})
ax.set(xlim=(-3.0,.45),ylim=(1.85,4.7),aspect='equal',xlabel='投影东向（km）',ylabel='相对北向（km）',title='较低障碍接入点 ≠ 已确认的冰流连接');ax.legend(fontsize=8,loc='upper left')
ax=axs[1];ax.plot(os/1000,oh-oh[0],color='#dc2626',label='原连接 T→P');ax.plot(candidate['station']/1000,candidate['height']-candidate['height'][0],color='#7c3aed',label='候选 T→Q');ax.axhline(0,color='#475569',ls='--');ax.set(title='原始 F19 推算床面：沿路径仍需越过高处',xlabel='从支流 T 出发的路径距离（km）',ylabel='相对 T 网格的床面高差（米）');ax.grid(alpha=.2);ax.legend();ax.text(.03,.95,'原路径最大高差约 322 米\n候选约 120 米，仍不是无障碍下游连接',transform=ax.transAxes,va='top',fontsize=10,bbox={'fc':'white','ec':'none','alpha':.9});fig.suptitle('0 号支流汇流核对：三套地形指向同一候选，但流向证据未通过',fontsize=14);fig.tight_layout();fig.savefig(O/'comparison.png',dpi=150)
report=f"""# 0 号支流汇流位置对照

本次只做已有地形与观测的空间分析，原输入保持不变，没有重建古冰厚或运行 BP。

区域 DEM、旧冰床、原始 F19 推算床面，在同一几何诊断规则下都选择主干节点 12 作为较低障碍候选 Q；原 P 是节点 4。Q 沿主干位于 P 下游约 {ds:.1f} 米。这表明 P 不是既有主干上唯一可考虑的接入位置。

规则先最小化从 T 到任一原主干下游节点所需经过的最高地形，再在该高度限制内求最短路径；所有路径在古冰川范围内，禁止对角穿越掩膜缺口。这是一项地形诊断，不等同于冰流模型或真实谷地出口判定。

| 地形依据 | 原路径最高点相对 T | 候选最低越障高度相对 T |
|---|---:|---:|
| 区域 DEM | 378.7 m | 180.9 m |
| 旧冰床 | 265.0 m | 0.0 m |
| 原始 F19 冰床 | 321.6 m | 119.5 m |

三个结果均指向 Q，但床面之间的障碍高度差异显著，说明仅看旧冰床会夸大候选路径的顺畅程度。床面逆坡允许存在于冰川下，越障高度不是错误判据。

现代流向辅助检查没有支持直接采用候选：原路径 120 m 邻域内正误差且速度大于两倍误差的 5 个观测格，方向夹角中位数 87.0 度；F19 候选邻域内 6 格约 82.3 度。两组不是相同格点，不是正式改善率；样本少，现代方向不能直接代表古冰流。方向向量长度已经核对为 1。

结论：原 P 的定位有明确可查疑点，但 Q 只是地形候选，暂不替换。应沿支流自身谷地查明实际出口、确认其与原主干是否真正连通，再决定连接目标。不能仅为了降低古冰厚，把 P 移到一个更低的位置。候选路径未约束其他支流，采用前还需处理真实交汇拓扑。

图中的黑箭头只使用现代有冰、误差大于零且速度大于两倍误差的观测，不画在已退冰区域。原始区域 DEM 在现代冰川上代表地表，F19 与旧冰床是推算床面，两者含义不同。

实现中初次处理第三套床面时发现对角通行的高度约束不一致；已统一两阶段规则，保留并复用先前完成的两套路径。最终三条路径均通过范围、八邻接、对角通行与有限值检查。这是静态路径分析的实现修正，不涉及冰川模拟重试。
"""
(O/'analysis.md').write_text(report,encoding='utf-8');completion=dict(status='completed-read-only-comparison',candidate_adopted=False,new_simulations=0,mainline_distance_P_to_Q_m=ds,checks=checks,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [O/'summary.json',O/'comparison.png',O/'analysis.md',*O.glob('*.npz')]});(O/'completion.json').write_text(json.dumps(completion,indent=2),encoding='utf-8');print(json.dumps(completion))
