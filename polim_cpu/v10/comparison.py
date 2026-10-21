"""Read-only paired before/after analysis; no old numerical run is repeated."""
import json
import numpy as np
from .registration import RUN,MESH
from ..v3.runner import write_json,digest


def compare(result):
    from .report import REPORT
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.font_manager import FontProperties
    old=RUN.parent/'v9-integrated-native-evolution-v1'
    with np.load(MESH,allow_pickle=False) as z:mesh={k:z[k] for k in z.files}
    states=[]
    for path in [old/'real/frame-000/state.npz',RUN/'real/frame-000/state.npz',old/'real/frame-006/state.npz',RUN/'real/frame-006/state.npz']:
        with np.load(path,allow_pickle=False) as z:states.append({k:z[k] for k in ['h','E','u']})
    for key in ['h','E']:
        if not np.array_equal(states[0][key],states[1][key]):raise ValueError('Initial comparison state differs')
    speeds=[np.linalg.norm(s['u'][:,-1],axis=1) for s in states]
    peak=int(np.argmax(speeds[0]));wet=states[0]['h']>=1
    data=dict(old_initial_max_speed_m_year=float(speeds[0].max()),new_initial_max_speed_m_year=float(speeds[1].max()),
        peak_reduction_percent=float(100*(1-speeds[1].max()/speeds[0].max())),
        old_peak_node=peak,new_peak_node=int(np.argmax(speeds[1])),new_speed_at_old_peak_m_year=float(speeds[1][peak]),
        old_wet_median_m_year=float(np.median(speeds[0][wet])),new_wet_median_m_year=float(np.median(speeds[1][wet])),
        old_max_hourly_thickness_change_m=float(np.max(abs(states[2]['h']-states[0]['h']))),new_max_hourly_thickness_change_m=float(np.max(abs(states[3]['h']-states[1]['h']))),
        initial_H_and_enthalpy_bitwise_equal=True,initial_mesh_identical=True,geometry_changed=False,velocity_clipping=False,
        old_result_sha256=digest(old/'result.json'),new_result_sha256=digest(RUN/'result.json'),
        attribution='Material-law replacement only on same initial geometry and thermal state; final states include subsequent coupled response. No physical validation claim.')
    write_json(REPORT/'comparison.json',data)
    rows,cols,nodes=mesh['pixel_rows'],mesh['pixel_cols'],mesh['pixel_center_nodes'];xy=mesh['xy'][nodes]
    shape=(int(rows.max()+1),int(cols.max()+1));dx=15.75
    x0=float((xy[:,0]-cols*dx).mean())-dx/2;y0=float((xy[:,1]+rows*dx).mean())+dx/2
    origin_y=float(xy[:,1].min())
    extent=[x0/1000,(x0+shape[1]*dx)/1000,(y0-shape[0]*dx-origin_y)/1000,(y0-origin_y)/1000]
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True)
    norm=LogNorm(vmin=.01,vmax=max(speeds[0].max(),speeds[1].max()))
    for ax,values,label in zip(axes,speeds[:2],['修复前','材料流变修复后']):
        raster=np.full(shape,np.nan);raster[rows,cols]=values[nodes]
        im=ax.imshow(raster,extent=extent,origin='upper',cmap='viridis',norm=norm,interpolation='nearest')
        ax.set(title=f'{label} · 最高 {values.max():,.0f} m/年',xlabel='投影东向（km）',ylabel='相对北向（km）')
        point=mesh['xy'][peak];ax.plot(point[0]/1000,(point[1]-origin_y)/1000,'r+',ms=11,mew=1.5)
    fig.colorbar(im,ax=axes,shrink=.75,label='表面水平速度（m/年，共同对数色标）')
    fig.suptitle('2387 冰川 · 同一初始冰厚与冰温 · 红十字为原最高速度位置',fontsize=15)
    fig.savefig(REPORT/'comparison.png',dpi=140);plt.close(fig)
    return data
