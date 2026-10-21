"""Scientific plots from original DEM and actual derived-bed fields."""
import json
import numpy as np
from scipy.ndimage import map_coordinates
from .geometry import AUDIT,ARCHIVE,PROFILES
from .registration import RUN
from ..v3.runner import write_json


def render_profiles(root):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    with np.load(ARCHIVE,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with np.load(AUDIT/'audit-fields.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    with np.load(RUN/'geometry/bed-change.npz',allow_pickle=False) as z:delta=z['delta_bed_m'];pins=z['pinned_mask']
    branches=[]
    for p in sorted(PROFILES.glob('branch-*.npz')):
        with np.load(p,allow_pickle=False) as z:branches.append({k:z[k] for k in z.files})
    xy=np.concatenate([np.column_stack([p['easting_m'],p['northing_m']]) for p in branches]);P=d['peak_xy'];J=np.array([-134.49697636368546,3158715.5030236365]);step=15.75
    p=branches[7];c=np.column_stack([p['easting_m'],p['northing_m']]);idx=np.argmin(np.linalg.norm(c-J,axis=1));direction=c[min(idx+1,len(c)-1)]-c[max(idx-1,0)];direction/=np.linalg.norm(direction)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,2,figsize=(14,11),constrained_layout=True);oy=float(a['y'].min());extent=[(a['x'][0]-step/2)/1000,(a['x'][-1]+step/2)/1000,(a['y'][-1]-step/2-oy)/1000,(a['y'][0]+step/2-oy)/1000]
    dem=np.where(a['lia_mask'],d['dem'],np.nan);im=axes[0,0].imshow(dem,extent=extent,origin='upper',cmap='terrain');fig.colorbar(im,ax=axes[0,0],shrink=.7,label='原始地形高程（m）')
    lim=max(1,float(np.max(abs(delta))));im=axes[0,1].imshow(np.where(a['valid_mask'],delta,np.nan),extent=extent,origin='upper',cmap='RdBu_r',vmin=-lim,vmax=lim);fig.colorbar(im,ax=axes[0,1],shrink=.7,label='冰床改变量（m）')
    for ax in axes[0]:
        ax.contour(a['x']/1000,(a['y']-oy)/1000,d['modern_mask'].astype(float),levels=[.5],colors='#333333',linewidths=.6)
        for b in branches:ax.plot(b['easting_m']/1000,(b['northing_m']-oy)/1000,color='white',lw=.7,alpha=.8)
        for name,point in [('P',P),('J',J)]:ax.plot(point[0]/1000,(point[1]-oy)/1000,'r+');ax.text(point[0]/1000+.06,(point[1]-oy)/1000,name,color='red',weight='bold')
        ax.set(xlabel='投影东向（km）',ylabel='相对北向（km）')
    axes[0,0].set_title('原始地形、现代冰缘（黑线）与支流（白线）');axes[0,1].set_title('仅修改拼接带，支流约束区保持原值')
    output=[]
    for ax,point,unit,title in zip(axes[1],[P,J],[d['downhill'],direction],['P：穿过原最高速度点','J：穿过主要汇流点']):
        station=np.arange(-630.,630.+step/2,step);samples=point+station[:,None]*unit
        rc=np.vstack([(samples[:,1]-a['y'][0])/-step,(samples[:,0]-a['x'][0])/step])
        def sample(v):return map_coordinates(v,rc,order=1,mode='constant',cval=np.nan,prefilter=False)
        valid=sample(a['valid_mask'].astype(float))>.999
        fields={name:np.where(valid,sample(v),np.nan) for name,v in [('dem',d['dem']),('old_bed',a['bed_elevation_m']),('new_bed',a['bed_elevation_m']+delta),('old_surface',a['bed_elevation_m']+a['thickness_m']),('new_surface',a['bed_elevation_m']+a['thickness_m']+delta)]}
        for key,color,style,label in [('dem','#888888',':','原始地形'),('old_bed','#8c564b','--','原冰床'),('new_bed','#8c564b','-','转接冰床'),('old_surface','#0072b2','--','原冰面'),('new_surface','#0072b2','-','转接冰面')]:ax.plot(station,fields[key],color=color,ls=style,label=label,lw=1.5)
        projection=(xy-point)@unit;normal=abs((xy-point)@np.array([-unit[1],unit[0]]));near=(normal<=75)&(abs(projection)<=630)
        r=np.rint((xy[near,1]-a['y'][0])/-step).astype(int);c0=np.rint((xy[near,0]-a['x'][0])/step).astype(int)
        ax.scatter(projection[near],(a['bed_elevation_m']+a['thickness_m'])[r,c0],s=18,facecolors='none',edgecolors='#111111',label='邻近支流采样（离剖面≤75m）')
        ax.axvline(0,color='red',lw=.7);ax.grid(alpha=.2);ax.set(title=title,xlabel='相对剖面中心距离（m）',ylabel='高程（m）');ax.legend(fontsize=8,loc='best')
        for mapax in axes[0]:mapax.plot(samples[:,0]/1000,(samples[:,1]-oy)/1000,color='red',lw=.7)
        output.append(dict(name=title,station_m=station.tolist(),xy=samples.tolist(),fields={k:[None if not np.isfinite(x) else float(x) for x in v] for k,v in fields.items()}))
    fig.suptitle('2387 冰川 · 源地形与冰床拼接诊断',fontsize=17);fig.savefig(root/'profiles.png',dpi=140);plt.close(fig);write_json(root/'profiles.json',output)
