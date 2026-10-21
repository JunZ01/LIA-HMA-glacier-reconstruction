from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import map_coordinates
import plotly.graph_objects as go
ROOT=Path.cwd(); out=ROOT/'polim_cpu/reports/ice-surface-3d-v1';out.mkdir(exist_ok=False)
src=ROOT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz'
g=np.load(src);x=g['x'];y=g['y'];S=g['surface_elevation_m'];H=g['thickness_m'];B=g['bed_elevation_m'];mask=g['valid_mask'].astype(bool)&np.isfinite(S)&(H>0)
assert hashlib.sha256(src.read_bytes()).hexdigest()=='59140635db4b67ef14099b273d5fdbc2dc6e091a4460c54720c68748223e4768'
assert np.allclose(S[mask],(B+H)[mask],atol=.002)
x0=float(x.min());y0=float(y.min()); X,Y=np.meshgrid((x-x0)/1000,(y-y0)/1000)
c=np.load(ROOT/'polim_cpu/reports/branch0-production-trace-v1/connector-0.npz');xy=c['xy']; cx=(xy[:,0]-x0)/1000;cy=(xy[:,1]-y0)/1000
rc=np.vstack(((xy[:,1]-y[0])/(y[1]-y[0]),(xy[:,0]-x[0])/(x[1]-x[0])));cs=map_coordinates(S,rc,order=1,mode='nearest')/1000
# Plot only observed grid vertices: stride-two display, no smoothing or altered geometry.
def mesh_arrays(stride=2):
 xx=X[::stride,::stride];yy=Y[::stride,::stride];ss=S[::stride,::stride]/1000;hh=H[::stride,::stride];mm=mask[::stride,::stride];nr,nc=ss.shape
 ids=np.arange(nr*nc).reshape(nr,nc);a=ids[:-1,:-1].ravel();b=ids[:-1,1:].ravel();cc=ids[1:,:-1].ravel();d=ids[1:,1:].ravel();tri=np.concatenate((np.stack((a,b,cc),1),np.stack((b,d,cc),1)));tri=tri[np.all(mm.ravel()[tri],axis=1)]
 return xx.ravel(),yy.ravel(),ss.ravel(),hh.ravel(),tri
xx,yy,zz,hh,tri=mesh_arrays();norm=Normalize(0,400)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig=plt.figure(figsize=(16,8),facecolor='#f5f7fa');axes=[fig.add_subplot(121,projection='3d'),fig.add_subplot(122,projection='3d')]
for i,ax in enumerate(axes):
 t=tri if i==0 else tri[np.all((xx[tri]>2.5)&(xx[tri]<5.5)&(yy[tri]>1.5)&(yy[tri]<4.5),axis=1)]
 surf=ax.plot_trisurf(xx,yy,np.nan_to_num(zz),triangles=t,cmap='viridis',norm=norm,linewidth=0,antialiased=False,shade=False);surf.set_array(np.mean(hh[t],axis=1));surf.set_clim(0,400)
 ax.plot(cx,cy,cs+.012,color='#f05252',lw=2.3,zorder=10)
 for j,label in [(0,'P 主干连接点'),(-1,'T 支流末端')]:
  ax.scatter(cx[j],cy[j],cs[j]+.025,color='#c72535',s=42,depthshade=False);ax.text(cx[j],cy[j],cs[j]+.13,label,fontsize=10,color='#891523')
 ax.set_xlabel('东西向距离 (km)',labelpad=10);ax.set_ylabel('南北向距离 (km)',labelpad=10);ax.set_zlabel('冰面海拔 (km)',labelpad=10);ax.view_init(elev=37,azim=-65);ax.set_facecolor('#f5f7fa');ax.tick_params(labelsize=9)
 if i==0:
  ax.set_title('整体冰面',fontsize=16,pad=14);ax.set_xlim(X.min(),X.max());ax.set_ylim(Y.min(),Y.max());ax.set_zlim(3.6,6.8);ax.set_box_aspect((np.ptp(X),np.ptp(Y),3.2))
 else:
  ax.set_title('原连接区域放大',fontsize=16,pad=14);ax.set_xlim(2.5,5.5);ax.set_ylim(1.5,4.5);ax.set_zlim(5.1,6.5);ax.set_box_aspect((3,3,1.4))
fig.suptitle('2387 古冰川 · 三维重建冰面',fontsize=22,x=.08,ha='left',y=.97)
fig.text(.08,.905,'原始重建版本，尚未应用多出口修正  |  颜色 = 冰厚  |  红线 = 原生产连接线',fontsize=12,color='#465269')
cax=fig.add_axes([.91,.23,.016,.49]);cb=fig.colorbar(ScalarMappable(norm=norm,cmap='viridis'),cax=cax);cb.set_label('冰厚 (m)',fontsize=12)
fig.text(.08,.055,'水平与垂直均为 km，垂直不夸大；展示采样间隔 31.5 m，未平滑。P、T 标签定位于存档冰面。',fontsize=10,color='#465269')
fig.subplots_adjust(left=.025,right=.89,bottom=.13,top=.84,wspace=.05);fig.savefig(out/'ice-surface.png',dpi=170,facecolor=fig.get_facecolor());plt.close(fig)
# Interactive native-resolution mesh, with original holes/margins retained.
xx1,yy1,zz1,hh1,tr1=mesh_arrays(1);keep=np.unique(tr1);remap=np.full(xx1.size,-1);remap[keep]=np.arange(keep.size);tr1=remap[tr1]
f=go.Figure(go.Mesh3d(x=xx1[keep],y=yy1[keep],z=zz1[keep],i=tr1[:,0],j=tr1[:,1],k=tr1[:,2],intensity=hh1[keep],colorscale='Viridis',cmin=0,cmax=400,colorbar=dict(title='冰厚 (m)'),name='重建冰面',lighting=dict(ambient=.7,diffuse=.6),hovertemplate='东西 %{x:.2f} km<br>南北 %{y:.2f} km<br>海拔 %{z:.3f} km<br>冰厚 %{intensity:.1f} m<extra></extra>'))
f.add_trace(go.Scatter3d(x=cx,y=cy,z=cs+.005,mode='lines',line=dict(color='#ed394f',width=6),name='原生产连接线'))
f.add_trace(go.Scatter3d(x=cx[[0,-1]],y=cy[[0,-1]],z=cs[[0,-1]]+.025,mode='markers+text',text=['P 主干连接点','T 支流末端'],textposition='top center',marker=dict(size=5,color='#ed394f'),name='讨论位置'))
f.add_trace(go.Mesh3d(x=xx1[keep],y=yy1[keep],z=B.ravel()[keep]/1000,i=tr1[:,0],j=tr1[:,1],k=tr1[:,2],color='#a49a86',opacity=.8,name='冰床（点击图例显示）',visible='legendonly',showscale=False))
f.update_layout(title=dict(text='2387 古冰川三维冰面<br><sup>原始重建，未应用多出口修正 · 拖动旋转，滚轮缩放 · 颜色表示冰厚</sup>',x=.035),scene=dict(aspectmode='data',xaxis_title='东西向距离 (km)',yaxis_title='南北向距离 (km)',zaxis_title='海拔 (km)',camera=dict(eye=dict(x=1.35,y=-1.6,z=1.25))),margin=dict(l=0,r=0,b=0,t=95),height=850,legend=dict(x=.01,y=.93),paper_bgcolor='#f5f7fa')
f.write_html(out/'index.html',include_plotlyjs=True,config=dict(displaylogo=False,scrollZoom=True))
(out/'provenance.json').write_text(json.dumps(dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),status='original-LIA-reconstruction-not-multi-outlet-repaired',new_simulations=0,native_grid_m=15.75,static_display_stride=2,interactive_display_stride=1,smoothing=False,vertical_exaggeration=1,ice_cells=int(mask.sum()),maximum_thickness_m=float(H[mask].max()),surface_elevation_range_m=[float(S[mask].min()),float(S[mask].max())],connector_surface='sampled archived raster S, not recovered subpixel constraint S'),ensure_ascii=False,indent=2),encoding='utf-8')
print(str(out));print('P,T',xy[[0,-1]].tolist());print('range',np.nanmin(S[mask]),np.nanmax(S[mask]));print('FILES',[(p.name,p.stat().st_size) for p in out.iterdir()])
