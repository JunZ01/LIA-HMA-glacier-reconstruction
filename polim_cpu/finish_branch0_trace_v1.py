from pathlib import Path
import json,hashlib
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
R=Path(__file__).resolve().parent;O=R/'reports/branch0-production-trace-v1'
g=np.load(R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz');a=np.load(R/'reports/geometry-source-audit-v1/audit-fields.npz');z=np.load(O/'connector-0.npz');net=json.loads((R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json').read_text(encoding='utf-8'));b0=next(b for b in net['branches'] if b['branch_index']==0);b7=next(b for b in net['branches'] if b['branch_index']==7);P=np.array(b0['flows_to_point_projected_m']);T=np.array(b0['coords_projected_m'][-1]);m=np.load(R/'lia-inputs-v1/system-2387/modern-raw-f19-bp-control-v1/geometry.npz');rc=np.array([(P[1]-g['y'][0])/(g['y'][1]-g['y'][0]),(P[0]-g['x'][0])/(g['x'][1]-g['x'][0])]);rawB=float(map_coordinates(m['bed'],rc[:,None],order=1,mode='constant',cval=np.nan,prefilter=False)[0]);rows=[]
for label,i in [('main_junction',0),('tributary_endpoint',-1)]: rows.append(dict(point=label,B=float(z['B'][i]),H=float(z['H'][i]),S_recovered=float(z['B'][i]+z['H'][i]),archived_grid_S=float(z['S'][i])))
bed_drop=rows[0]['B']-rows[1]['B'];Srise=rows[1]['S_recovered']-rows[0]['S_recovered'];Hrise=rows[1]['H']-rows[0]['H'];endfloor=rows[0]['S_recovered']-rows[1]['B'];support=int(np.load(O/'production-support.npz')['support'].sum());result=dict(status='completed-static-source-trace',new_simulations=0,source_constraint_count=support,original_saved_constraint_count=444,constraint_count_matches=support==444,connector_length_m=float(z['station'][-1]),node_values=rows,H_rise_m=Hrise,bed_drop_m=bed_drop,S_rise_m=Srise,H_identity_residual=Hrise-bed_drop-Srise,endpoint_H_required_by_inherited_parent_surface_m=endfloor,raw_f19_parent_B_m=rawB,raw_f19_parent_B_change_m=rawB-rows[0]['B'],explanation='High prescribed junction plus upstream-monotone plastic surface creates endpoint thick ice. This is a mechanism trace,not proof the inferred paleo bed/topology is false. Raw F19 parent bed shift is static only; no new parent H or surface has been solved.',sampling_note='S_recovered uses production subpixel B plus stored constraint H. Archived grid S uses raster-cell B; endpoint mismatch is not a time change.',display_connector_note='Later polim profile connector is a separately inferred shortest-ice path; this audit restores the original production adapter geometry.',unchanged_sources=[str(R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz')])
(O/'mechanism.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False});fig,axs=plt.subplots(1,2,figsize=(13,6));origin=g['y'].min();ax=axs[0];X,Y=np.meshgrid(g['x']/1000,(g['y']-origin)/1000);dem=np.where(g['lia_mask'],a['dem'],np.nan);im=ax.pcolormesh(X,Y,dem,shading='nearest',cmap='terrain',vmin=4500,vmax=6500);fig.colorbar(im,ax=ax,shrink=.75,label='区域地形高程（米）')
for b,label,color in [(b0,'原 0 号支流','#111827'),(b7,'原主干','#2563eb')]:q=np.array(b['coords_projected_m']);ax.plot(q[:,0]/1000,(q[:,1]-origin)/1000,color=color,lw=2,label=label)
q=z['xy'];ax.plot(q[:,0]/1000,(q[:,1]-origin)/1000,color='#dc2626',lw=2,label='原生产连接段')
for label,xy,shift in [('P 主干汇流点',P,(.08,-.28)),('T 支流末端',T,(-.6,.22))]:loc=(xy[0]/1000,(xy[1]-origin)/1000);ax.scatter(*loc,c='#111827',s=35,zorder=5);ax.annotate(label,loc,(loc[0]+shift[0],loc[1]+shift[1]),fontsize=9,bbox={'fc':'white','ec':'none','alpha':.9})
A=np.array([-2139.7096677023073,3159517.8405094133]);ax.scatter(A[0]/1000,(A[1]-origin)/1000,c='#facc15',edgecolors='black',marker='*',s=150,zorder=6,label='A 原高速点');ax.set(xlim=(-3.9,.5),ylim=(1.7,4.8),aspect='equal',xlabel='投影东向（km）',ylabel='相对北向（km）',title='支流被接到高位主干点');ax.legend(fontsize=8,loc='upper left')
ax=axs[1];s=z['station']/1000;ax.plot(s,z['B'],label='连接段床面',color='#475569');ax.plot(s,z['S'],label='存档二维冰面',color='#0284c7');ax.fill_between(s,z['B'],z['B']+z['H'],alpha=.15,color='#0284c7');ax.axhline(rows[0]['S_recovered'],ls='--',color='#b45309',label='继承的主干冰面约 5744 米');ax.set(xlabel='从主干 P 到支流 T 的距离（km）',ylabel='高程（米）',title='逆向求解时：谷底降低，冰面继续升高');ax.text(.05,.06,'P：冰厚约 13 米 → T：约 396 米\n床面降低约 257 米，冰面升高约 127 米',transform=ax.transAxes,fontsize=10,bbox={'fc':'white','ec':'none','alpha':.92});ax.legend(fontsize=9,loc='upper left');ax.grid(alpha=.2);fig.suptitle('2387 · 厚冰峰值已存在于原生产连接段端点约束',fontsize=14);fig.tight_layout();fig.savefig(O/'mechanism.png',dpi=150)
report="""# 0 号支流厚冰来源追踪

已恢复原生产连接段几何，候选约束栅格数 444，与原记录一致。未调用中心线冰面求解器、二维重建或 BP，原输入未修改。

原生产连接段长约 1.783 km，从主干汇流点 P 到支流末端 T。P 的床面约 5731.6 m，T 约 5475.1 m，向支流方向床面降低约 256.6 m。原有效剪应力 85.932791 kPa，形状因子 0.8。塑性方程在逆向求解时要求冰面增加，结合继承的 P 冰面约 5744.3 m，T 在冰面尚未进一步抬升前就需要约 269.2 m 冰厚。原解再增加约 126.7 m 冰面高程，最终 T 冰厚约 395.9 m。

全场最大冰厚就在 T 对应约束栅格。二维插值将这片厚冰扩展至 A 周围；不是 BP 演化或后续像元分区首次制造了厚冰峰值。

这确定了厚冰的计算来源，但不能单凭床面逆坡判定汇流关系错误。仍需用谷地地形与原中心线生成证据，检验 P 是否为合理汇流点、连接段是否跨越真实地形障碍。后续报告使用的显示连接线与原生产连接线不同，不能直接当作原重建约束。

更换原始 F19 空间冰厚后，P 的现代床面静态估计约 5783.1 m，比旧值高约 51.4 m，因此不能预先宣称换冰厚来源就会消除高位汇流条件。这不是重建后的新冰面或新流速结果。

节点高程表采用生产亚网格床面与存档约束厚度求和；图上的存档二维冰面使用栅格中心床面，两者在端点有数米采样差异。

下一修复入口：核对 0 号支流与主干的汇流定位与连接路径，确定几何依据后再修改；无需重复已经失败的接缝平滑试验。
"""
(O/'analysis.md').write_text(report,encoding='utf-8');print(json.dumps(result,ensure_ascii=False))
