"""Static V3 geometry diagnostics only. Never constructs/runs a glacier model."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,html,itertools,json,pickle,sys
sys.dont_write_bytecode=True
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from pyproj import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import shape,LineString,Point
from shapely.ops import unary_union
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
INV=ROOT/'analysis/lia_51_inventory_v1';PRIOR=ROOT/'analysis/lia_51_same_pixel_pairs_v1'
CONTEXT=ROOT/'analysis/lia_51_systematic_analysis_v1'
SOURCES={};PAIRS=[(2622,2827),(1012,1044),(2939,2928)];IDS=sorted({x for p in PAIRS for x in p})
COLORS=['#307a9b','#b86d38'];TANGENTS=[100,250,500]
plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def reg(p,expected=None):
    p=Path(p);p=p if p.is_absolute() else ROOT/p
    assert p.name!='inputs.npz' and not any(x in p.as_posix().lower() for x in ['20cr','jionglacuo','s7t4zc34sr','temperature reconstruction.xlsx'])
    h=sha(p)
    if expected:assert h==expected.upper(),str(p)
    SOURCES[p.relative_to(ROOT).as_posix()]=h;return p
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [clean(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(name,x):(OUT/name).write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def parts(geom):
    if geom.is_empty:return []
    if geom.geom_type=='LineString':return [geom]
    if hasattr(geom,'geoms'):return [v for g in geom.geoms for v in parts(g)]
    return []


def load_inputs():
    methods=read(reg(OUT/'methods.json'))
    for p,h in methods['authority'].items():reg(p,h)
    manifest=read(reg(PRIOR/'output_manifest.json'));prior={}
    for name in ['pairs.json','pair_selection.json','validation.json']:
        prior[name]=read(reg(PRIOR/name,manifest[name]))
    assert prior['validation.json']['status']=='pass'
    assert [s['pair_id'] for s in prior['pair_selection.json']['spotlights']]==['_'.join(map(str,p)) for p in PAIRS]
    # Saved-output lineage only; earlier authority snapshots are not current launch authority.
    for p,h in read(reg(PRIOR/'source_manifest.json')).items():
        if p not in methods['authority']:reg(p,h)
    im=read(reg(INV/'sources.json'));inventory=read(reg(INV/'inventory.json'))
    cm=read(reg(CONTEXT/'output_manifest.json'));hyps=read(reg(CONTEXT/'hypsometry.json',cm['hypsometry.json']))
    fm=read(reg(ROOT/'analysis/lia_51_stage_controls_v1/output_manifest.json'))
    features=read(reg(ROOT/'analysis/lia_51_stage_controls_v1/features_51.json',fm['features_51.json']))
    assert len(inventory)==len(hyps)==len(features)==51
    for r in inventory:
        if r['lia_unique_id'] in IDS:
            for k in ['lia_network_source','lia_source_grid']:reg(r[k],im[r[k]])
    return methods,{r['lia_unique_id']:r for r in inventory},{r['system']:r for r in hyps},{r['system']:r for r in features},prior['pairs.json']


def static_geometry(inv,old_hyp):
    sid=inv['lia_unique_id']
    with gzip.open(reg(inv['lia_network_source']),'rb') as f:n=pickle.load(f)
    assert n['main_branch_index']==inv['lia_main_branch']
    branch=next(b for b in n['branches'] if b['branch_index']==n['main_branch_index'])
    xy=np.asarray(branch['coords_projected_m'],float);line=LineString(xy);L=line.length
    assert np.isclose(L/1000,inv['lia_length_km'],atol=1e-8) and np.linalg.norm(xy[-1]-np.asarray(n['main_outlet_projected_m']))<1e-5
    with xr.open_dataset(reg(inv['lia_source_grid'])) as ds:
        assert int(ds.attrs['lia_unique_id'])==sid and CRS.from_user_input(n['crs'])==CRS.from_user_input(ds.spatial_ref.attrs['crs_wkt'])
        mi=list(ds.member_id.values.astype(str)).index('farinotti2019')
        z=ds.surface_elevation_m.isel(member=mi).values.astype(float);bed=ds.bed_elevation_m.isel(member=mi).values.astype(float);h=ds.thickness_m.isel(member=mi).values.astype(float)
        outline=ds.lia_mask.values>0
        valid=(ds.valid_mask.isel(member=mi).values>0)&outline&np.isfinite(z)&np.isfinite(bed)&np.isfinite(h)
        ice=valid&(h>1e-6);x=ds.x.values.copy();y=ds.y.values.copy()
        excluded=(ds.excluded_component_mask.values>0) if 'excluded_component_mask' in ds else np.zeros_like(ice)
    dx,dy=float(np.median(np.diff(x))),float(np.median(np.diff(y)))
    assert np.allclose(np.diff(x),dx) and np.allclose(np.diff(y),dy)
    cell=abs(dx*dy);zz=z[ice];hh=h[ice];area=len(zz)*cell/1e6;volume=float(hh.sum()*cell/1e9)
    assert zz.min()>=2500 and zz.max()<=8500
    assert np.isclose(area,old_hyp['total_area_km2'],rtol=1e-10) and np.isclose(volume,old_hyp['volume_km3'],rtol=2e-7)
    assert np.max(abs((z-bed-h)[valid]))<.01
    rel=(zz-zz.min())/(zz.max()-zz.min());bins={}
    for width in [100,200,300]:
        edges=np.arange(2500,8501,width);counts=np.histogram(zz,edges)[0]
        assert counts.sum()==len(zz)
        bins[str(width)]=dict(edges_m=edges,counts=counts,area_km2=counts*cell/1e6,area_fraction=counts/len(zz))
    close200=np.array(old_hyp['area_km2']);assert np.allclose(bins['200']['area_km2'],close200,atol=1e-9)
    # Raster polygons keep holes and disconnected components, not a convex hull.
    transform=Affine(dx,0,x[0]-dx/2,0,dy,y[0]-dy/2)
    polygons=unary_union([shape(g) for g,v in shapes(ice.astype(np.uint8),mask=ice,connectivity=4,transform=transform) if v==1])
    assert polygons.is_valid and np.isclose(polygons.area,cell*ice.sum(),rtol=2e-9)
    vy,vx=np.where(valid);tree=cKDTree(np.c_[x[vx],y[vy]])
    def sample(points):
        dist,idx=tree.query(np.atleast_2d(points));ok=dist<=1.5*np.hypot(dx,dy)
        return {k:np.where(ok,a[vy[idx],vx[idx]],np.nan) for k,a in [('surface_m',z),('bed_m',bed),('thickness_m',h)]}|{'distance_m':dist,'valid':ok,'row':vy[idx],'col':vx[idx]}
    dist=np.unique(np.r_[np.arange(0,L,100.),L]);points=np.array([line.interpolate(L-d).coords[0] for d in dist]);sampled=sample(points)
    assert np.isclose(sampled['surface_m'][0],inv['lia_terminus_surface_m'],atol=1e-6) and np.isclose(sampled['surface_m'][-1],inv['lia_head_surface_m'],atol=1e-6)
    widths={};crosslines={}
    for tangent in TANGENTS:
        values=[];censored=[];reasons=[];normals=[]
        for d,p in zip(dist,points):
            st=L-d;up=np.array(line.interpolate(max(0,st-tangent)).coords[0]);down=np.array(line.interpolate(min(L,st+tangent)).coords[0])
            vector=down-up
            if np.linalg.norm(vector)<1e-10:
                values.append(np.nan);censored.append(False);reasons.append('degenerate_tangent');normals.append([np.nan,np.nan]);continue
            normal=np.array([-vector[1],vector[0]])/np.linalg.norm(vector);normals.append(normal)
            cut=LineString([p-10000*normal,p+10000*normal])
            pieces=[g for g in parts(cut.intersection(polygons)) if g.distance(Point(p))<=1e-7]
            if not pieces:values.append(np.nan);censored.append(False);reasons.append('centre_outside_positive_ice');continue
            assert len(pieces)==1,(sid,d,'ambiguous_contiguous_section')
            g=pieces[0];end=np.array(g.coords)[[0,-1]];is_censored=np.max(np.linalg.norm(end-p,axis=1))>=9999.999
            values.append(g.length);censored.append(is_censored);reasons.append('normal_limit' if is_censored else 'valid')
            if tangent==250 and d in [500,1000,2000]:crosslines[str(int(d))]=np.array(g.coords)
        widths[str(tangent)]=dict(width_m=np.array(values),censored=censored,reasons=reasons,normal_unit_vectors=normals)
    zones=[]
    for a,b in [(0,1000),(1000,3000),(0,.2*L)]:
        pts=np.array([line.interpolate(L-v).coords[0] for v in [a,b]]);s=sample(pts)
        slope=float(np.degrees(np.arctan2(s['surface_m'][1]-s['surface_m'][0],b-a))) if np.all(s['valid']) else np.nan
        zones.append(dict(start_m=a,end_m=b,surface_slope_deg=slope,endpoint_sample_distance_m=s['distance_m']))
    ws=[]
    for tangent in TANGENTS:
        for a,b in [(200,1000),(1000,3000)]:
            region=(dist>=a)&(dist<=b);w=np.asarray(widths[str(tangent)]['width_m']);ok=region&np.isfinite(w)&~np.array(widths[str(tangent)]['censored'])
            ws.append(dict(tangent_half_window_m=tangent,start_m=a,end_m=b,n=int(ok.sum()),eligible_n=int(region.sum()),coverage=float(ok.sum()/region.sum()),median_width_m=float(np.median(w[ok])) if ok.any() else np.nan))
    quantiles=np.linspace(0,1,1001)
    stat=dict(system=sid,mainline_length_km=L/1000,area_km2=area,volume_km3=volume,outline_area_km2=float(outline.sum()*cell/1e6),valid_area_km2=float(valid.sum()*cell/1e6),positive_coverage=float(ice.sum()/outline.sum()),excluded_area_km2=float(excluded.sum()*cell/1e6),
        cell_dx_m=dx,cell_dy_m=dy,cell_area_m2=cell,positive_cells=int(ice.sum()),grid_shape=list(ice.shape),branch_count=len(n['branches']),
        zmin_m=float(zz.min()),zmax_m=float(zz.max()),zmedian_m=float(np.median(zz)),zmean_m=float(zz.mean()),terminus_surface_m=float(sampled['surface_m'][0]),
        mean_thickness_m=float(hh.mean()),normalized_mean_height=float(rel.mean()),relative_area_fractions={str(v):float(np.mean(rel<v)) for v in [.25,.5,.75]},
        below_terminus_plus_fractions={str(v):float(np.mean(zz<sampled['surface_m'][0]+v)) for v in [100,300,500]},
        quantile_probabilities=quantiles,elevation_quantiles_m=np.quantile(zz,quantiles),normalized_elevation_quantiles=np.quantile(rel,quantiles),hypsometry=bins,
        surface_slope_zones=zones,width_zones=ws,profile_valid_count=int(sampled['valid'].sum()),profile_count=len(dist),
        profile_max_sample_distance_m=float(sampled['distance_m'].max()),sample_limit_m=float(1.5*np.hypot(dx,dy)),source_grid=inv['lia_source_grid'],source_network=inv['lia_network_source'])
    profile=dict(system=sid,distance_from_terminus_m=dist,xy_projected_m=points,**sampled,widths=widths,display_crosssections=crosslines,mainline_xy=xy)
    return stat,profile,dict(x=x,y=y,z=z,ice=ice,outline=outline,xy=xy,branches=n['branches'],dx=dx,dy=dy)


def compare(stats,profiles,prior):
    results=[]
    for a,b in PAIRS:
        sa,sb=stats[a],stats[b];p=next(p for p in prior if p['pair_id']==f'{a}_{b}')
        results.append(dict(pair_id=f'{a}_{b}',a=a,b=b,absolute_median_elevation_gap_m=sb['zmedian_m']-sa['zmedian_m'],
            normalized_quantile_distance=float(np.mean(abs(np.array(sa['normalized_elevation_quantiles'])-np.array(sb['normalized_elevation_quantiles'])))),
            absolute_histogram_overlap={str(w):float(np.minimum(sa['hypsometry'][str(w)]['area_fraction'],sb['hypsometry'][str(w)]['area_fraction']).sum()) for w in [100,200,300]},
            lower_half_area_fraction_a=sa['relative_area_fractions']['0.5'],lower_half_area_fraction_b=sb['relative_area_fractions']['0.5'],
            prior_length_late_contrast_pp=p['metrics']['length_km']['late_contrast_pp'],prior_volume_late_contrast_pp=p['metrics']['volume_km3']['late_contrast_pp'],
            confounding=dict(base_temp_bias_gap_c=p['differences']['base_temp_bias_c'],q_gap_c=p['differences']['q_mean_1500_2000_c'],melt_ratio=p['melt_ratio'],glen_ratio=p['glen_ratio'],
                length_operator_a=p['length_operator_a'],length_operator_b=p['length_operator_b'])))
    return results


def figsave(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def draw_figures(stats,profiles,maps,features):
    fig,axes=plt.subplots(3,2,figsize=(12,14))
    for i,pair in enumerate(PAIRS):
        for j,sid in enumerate(pair):
            m=maps[sid];ax=axes[i,j];end=m['xy'][-1];extent=[(m['x'][0]-m['dx']/2-end[0])/1000,(m['x'][-1]+m['dx']/2-end[0])/1000,(m['y'][-1]+m['dy']/2-end[1])/1000,(m['y'][0]-m['dy']/2-end[1])/1000]
            im=ax.imshow(np.where(m['ice'],m['z'],np.nan),extent=extent,origin='upper',cmap='terrain',vmin=3000,vmax=7200)
            for branch in m['branches']:
                xy=np.array(branch['coords_projected_m']);ax.plot((xy[:,0]-end[0])/1000,(xy[:,1]-end[1])/1000,color='#6b7780',lw=.5,alpha=.6)
            xy=m['xy'];ax.plot((xy[:,0]-end[0])/1000,(xy[:,1]-end[1])/1000,color=COLORS[j],lw=1.5)
            for coords in profiles[sid]['display_crosssections'].values():
                coords=np.array(coords);ax.plot((coords[:,0]-end[0])/1000,(coords[:,1]-end[1])/1000,color='#ce3059',lw=1.3)
            ax.scatter([0],[0],color='#222222',s=28,zorder=4)
            ax.set_title(f"{sid} | {stats[sid]['area_km2']:.1f} km² | {stats[sid]['zmedian_m']:.0f}m 中位高程")
            ax.set_xlabel('相对末端投影东向距离 / km');ax.set_ylabel('相对末端投影北向距离 / km');ax.set_aspect('equal')
    fig.subplots_adjust(left=.09,right=.84,top=.94,bottom=.07,hspace=.3,wspace=.3)
    cax=fig.add_axes([.88,.3,.018,.4]);fig.colorbar(im,cax=cax,label='配对V3表面高程 / m')
    fig.suptitle('静态冰体形状与主流线：三组代表配对',fontsize=17)
    fig.text(.5,.012,'同一高程色标；黑点为地理主流线末端。红线：距末端0.5、1、2km的局部横断面。不是动态模型网格。',ha='center',fontsize=9)
    figsave(fig,'01_static_maps')
    for a,b in PAIRS:
        fig,axes=plt.subplots(2,3,figsize=(16,8.4))
        for j,sid in enumerate([a,b]):
            s,p=stats[sid],profiles[sid];color=COLORS[j];d=np.array(p['distance_from_terminus_m'])/1000
            z=np.asarray(p['surface_m']);bed=np.asarray(p['bed_m']);H=np.asarray(p['thickness_m'])
            axes[0,0].plot(d,z,color=color,label=f'{sid} 表面');axes[0,0].plot(d,bed,color=color,ls='--',lw=1,label=f'{sid} 重建床面')
            near=d<=3+1e-9
            axes[0,1].plot(d[near],(z-s['terminus_surface_m'])[near],color=color,label=str(sid));axes[0,1].set_xlim(0,3)
            w=np.asarray(p['widths']['250']['width_m'],float);ws=np.array([p['widths'][str(t)]['width_m'] for t in TANGENTS],float)
            okay=np.all(np.isfinite(ws),axis=0);low=np.full(len(d),np.nan);high=low.copy();low[okay]=ws[:,okay].min(axis=0);high[okay]=ws[:,okay].max(axis=0)
            axes[0,2].plot(d[near],w[near],color=color,label=str(sid));axes[0,2].fill_between(d[near],low[near],high[near],color=color,alpha=.15);axes[0,2].set_xlim(0,3)
            h=s['hypsometry']['100'];edges=np.array(h['edges_m']);axes[1,0].stairs(np.array(h['area_fraction'])*100,edges,color=color,label=str(sid),lw=1.6)
            axes[1,1].plot(s['normalized_elevation_quantiles'],np.array(s['quantile_probabilities'])*100,color=color,label=str(sid))
            axes[1,2].plot(d,H,color=color,label=str(sid))
        labels=[('整条主线：配对栅格表面与床面','距末端上游距离 / km','高程 / m'),('末端上游3km：相对末端抬升','距末端上游距离 / km','表面高程 − 末端高程 / m'),('末端上游3km：局部静态冰体横宽','距末端上游距离 / km','横宽 / m'),('绝对高程—面积分布','表面高程 / m','每100m带占总面积 / %'),('自身高差归一化后的面积累积','(Z−Zmin) / (Zmax−Zmin)','累积面积 / %'),('整条主线：静态重建冰厚','距末端上游距离 / km','冰厚 / m')]
        for ax,(title,xlabel,ylabel) in zip(axes.flat,labels):
            ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.grid(alpha=.16);ax.legend(fontsize=8)
        axes[1,0].set_xlim(min(stats[a]['zmin_m'],stats[b]['zmin_m'])-100,max(stats[a]['zmax_m'],stats[b]['zmax_m'])+100)
        fig.suptitle(f'{a} — {b} | 静态几何解释线索，不是独立机制验证',fontsize=17)
        fig.text(.5,.012,'横宽阴影：法线方向估计窗口100/250/500m敏感性；缺测不补齐。高程—面积使用整个系统的正冰厚栅格。床面和冰厚为重建值。',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,.95]);figsave(fig,f'pair_{a}_{b}')
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    for ax,key,label in zip(axes,['normalized_mean_height','lower_half_area_fraction','mean_thickness_m'],['归一化平均高程','自身高差下半部面积比例','静态平均冰厚 / m']):
        vals=[r[key] for r in features.values()];ax.scatter(np.sort(vals),np.linspace(-.06,.06,len(vals)),c='#bfcbd2',s=22,label='全部51条背景')
        for i,pair in enumerate(PAIRS):
            for j,sid in enumerate(pair):
                value=stats[sid]['relative_area_fractions']['0.5'] if key=='lower_half_area_fraction' else stats[sid][key]
                ax.scatter(value,i+.5,c=COLORS[j],s=45);ax.annotate(str(sid),(value,i+.5),xytext=(2,7 if j==0 else -14),textcoords='offset points',fontsize=8)
        ax.set_yticks([0,.5,1.5,2.5],['51条背景','2622—2827','1012—1044','2939—2928']);ax.set_xlabel(label);ax.grid(axis='x',alpha=.15)
    fig.suptitle('六条代表冰川在51条静态几何背景中的位置',fontsize=16);fig.tight_layout();figsave(fig,'02_context_51')


def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def format_number(x,n=2):return '缺测' if x is None or not np.isfinite(x) else f'{x:.{n}f}'
def create_report(stats,comparisons,summary,features):
    text=['<h1>三组代表冰川：沿程几何与高程—面积分布</h1><p class="lead">51条作为背景 · 六条固定代表 · 已有静态几何与历史结果对照</p>',
          '<div class="note">本轮没有运行冰川模型。V3/F19是静态重建几何，不等于模拟每一年的状态；沿程地理距离不能与历史模型长度直接互换。下面是可供后续检验的形状差异，不是地形因果效应。</div>',
          '<p><a href="../lia_51_same_pixel_pairs_v1/index.html">上一轮：完整历史曲线与配对诊断</a></p>']
    if summary.get('interpretations'):text+=['<h2>主要发现</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in summary['interpretations'])+'</ul>']
    text+=['<h2>1. 平面形状与采样位置</h2><img src="01_static_maps.png"><p>从原主流线坐标重新采样配对V3栅格；未使用旧网络内可能错位的表面高程序列。横断面只取包含主线位置的连续冰体区段，不跨越无冰间隔拼接。</p>']
    text.append(table(['冰川','面积km²','中位高程m','自身高差下半部面积%','末端+500m以下面积%','0—1km坡度°','1—3km坡度°','0.2—1km横宽m','宽度有效点','正冰厚覆盖率%'],[[sid,format_number(s['area_km2']),format_number(s['zmedian_m'],0),format_number(s['relative_area_fractions']['0.5']*100,1),format_number(s['below_terminus_plus_fractions']['500']*100,1),format_number(s['surface_slope_zones'][0]['surface_slope_deg']),format_number(s['surface_slope_zones'][1]['surface_slope_deg']),format_number(next(w['median_width_m'] for w in s['width_zones'] if w['tangent_half_window_m']==250 and w['start_m']==200),0),str(next(w['n'] for w in s['width_zones'] if w['tangent_half_window_m']==250 and w['start_m']==200))+'/9',format_number(s['positive_coverage']*100,1)] for sid,s in stats.items()]))
    for p in comparisons:
        a,b=p['a'],p['b'];text += [f'<h2>{a} — {b}：几何与历史对照</h2><img src="pair_{a}_{b}.png">']
        text.append(f'<p>100m高程带面积分布重叠比例为{p["absolute_histogram_overlap"]["100"]:.1%}；中位高程B−A为{p["absolute_median_elevation_gap_m"]:.0f}m。自身高差归一化后的分布距离为{p["normalized_quantile_distance"]:.3f}（越小越接近，不是显著性检验）。</p>')
        if summary.get('pair_interpretations',{}).get(p['pair_id']):text.append('<div class="note">'+html.escape(summary['pair_interpretations'][p['pair_id']])+'</div>')
        text.append(f'<p>对应上一轮19世纪相对17世纪的配对差：长度{p["prior_length_late_contrast_pp"]:+.2f}个百分点，体积{p["prior_volume_late_contrast_pp"]:+.2f}个百分点。该差值是各自历史最大值归一化后的比较，不是绝对末端退缩距离。</p>')
        c=p['confounding'];text.append(f'<p>仍未控制的参数差异：基础温偏B−A {c["base_temp_bias_gap_c"]:+.2f}℃，附加q {c["q_gap_c"]:+.2f}℃，融化因子比{c["melt_ratio"]:.2f}，冰流参数比{c["glen_ratio"]:.2f}。静态形状一致或不同都不能排除这些作用。</p>')
    text+=['<h2>在51条样本中的位置</h2><img src="02_context_51.png"><p>六条是按上一轮配对规则选择的代表，不在这六条上重新拟合回归或推广至整个喜马拉雅。</p><h2>数据范围与限制</h2><ul><li>正冰厚掩膜约覆盖原LIA轮廓的一部分，未填补缺失及排除区域；逐条覆盖率已报告。</li><li>局部横宽是静态冰体横断面宽度，不是谷地宽度、模型有效宽度或汇流面积等效宽度。</li><li>高程—面积使用整个系统，沿程剖面仅代表登记主线；二者空间范围不同。</li><li>归一化高度不等于平衡线；末端以上500m面积也不是积累区面积比。</li><li>静态重建几何不是独立观测验证，不能用其体积与某历史年份体积相减来推算质量损失。</li><li>本轮不能证明响应时间、气候作用或真实分化年代，需要实际模型几何和独立约束才能进一步归因。</li></ul>']
    text.append('<details><summary>宽度方向敏感性及质量检查</summary>'+table(['冰川','切向半窗口m','上游区间m','有效/应有点','中位横宽m'],[[sid,w['tangent_half_window_m'],f"{w['start_m']}—{w['end_m']}",f"{w['n']}/{w['eligible_n']}",format_number(w['median_width_m'],0)] for sid,s in stats.items() for w in s['width_zones']])+'</details>')
    files=['methods.json','geometry_6.json','profiles_6.json','pair_comparisons.json','summary.json','source_manifest.json','validation.json','output_manifest.json']
    text.append('<p>'+' · '.join(f'<a href="{f}">{f}</a>' for f in files)+'</p>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三组冰川的沿程几何</title><style>body{font:16px/1.75 "Microsoft YaHei",sans-serif;margin:0;background:#f3f6f8;color:#263947}main{max-width:1350px;margin:auto;padding:30px}h1{font-size:31px}h2{margin-top:42px}.lead{font-size:19px;color:#44748a}.note{background:#e5edf2;border-left:5px solid #44809b;padding:18px}img{max-width:100%;height:auto;background:white;border:1px solid #dce3e7;margin:14px 0}table{border-collapse:collapse;font-size:13px;white-space:nowrap}td,th{border:1px solid #d6e0e6;padding:8px;text-align:left}th{background:#e3edf1}.scroll{overflow:auto}a{color:#226c8c}details{background:white;padding:14px}</style><main>'+''.join(text)+'</main></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')


def main():
    if (OUT/'validation.json').exists():assert read(OUT/'validation.json')['status']!='pass','Finalized output; use new version.'
    methods,inventory,hyps,features,prior=load_inputs();stats={};profiles={};maps={}
    for sid in IDS:
        stats[sid],profiles[sid],maps[sid]=static_geometry(inventory[sid],hyps[sid]);print(f'Geometry verified/extracted: {sid}',flush=True)
    comparisons=compare(stats,profiles,prior)
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),systems=IDS,pairs=PAIRS,context_systems=51,geometry_source='paired static V3/F19',simulations_launched=0,raw_climate_access=False,formal_acceptance_unchanged=True)
    if (OUT/'interpretations.json').exists():summary.update(read(reg(OUT/'interpretations.json')))
    save('geometry_6.json',list(stats.values()));save('profiles_6.json',list(profiles.values()));save('pair_comparisons.json',comparisons);save('summary.json',summary)
    draw_figures(stats,profiles,maps,features);create_report(stats,comparisons,summary,features)
    for p,h in SOURCES.items():assert sha(ROOT/p)==h,p
    save('source_manifest.json',SOURCES);save('validation.json',dict(status='built-independent-verification-pending',authority_hashes=methods['authority'],source_count=len(SOURCES),simulations_launched=0,raw_climate_access=False))
    save('output_manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='output_manifest.json' and '__pycache__' not in p.parts})
    print(json.dumps(comparisons,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
