"""Georeferenced 2387 profiles and an explicit, conservative spatial partition.

Static geometry product only. Original routes, inferred gap connectors,
centreline samples and area/volume-equivalent fields remain distinguishable.
"""
import base64
import heapq
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
from .network_data_reader import read_network
from .v3.runner import PROJECT, digest, write_json
from .thermal_attribution_v1 import validate, ref

BASE=PROJECT/'validation_data/results/paleo_surface_validation_v0'
SOURCE=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1'
NETWORK=BASE/'lia_gdir_anchor_reroute_core191_v3/2387/component_0/lia_centerline_network.pkl.gz'
OUT=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired'
REPORT=PROJECT/'polim_cpu/reports/spatial-profiles-v3-paired'
EVIDENCE=BASE/'polim_lia_system2387_spatial_profiles_v3_paired'


def plain(v):
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,np.generic): return v.item()
    if isinstance(v,dict): return {k:plain(a) for k,a in v.items()}
    if isinstance(v,(list,tuple)): return [plain(a) for a in v]
    return v


def arclength(q): return np.r_[0.,np.cumsum(np.sqrt(np.sum(np.diff(q,axis=0)**2,axis=1)))]
def sample(q,s):
    d=arclength(q); return np.column_stack([np.interp(s,d,q[:,j]) for j in (0,1)])
def project(q,p):
    delta=np.diff(q,axis=0); t=np.clip(np.sum((p-q[:-1])*delta,axis=1)/np.sum(delta*delta,axis=1),0,1)
    locations=q[:-1]+t[:,None]*delta; j=int(np.argmin(np.sum((locations-p)**2,axis=1)))
    return float(arclength(q)[j]+t[j]*np.sqrt(np.sum(delta[j]**2))),locations[j],float(np.sqrt(np.sum((locations[j]-p)**2)))


class IceGraph:
    def __init__(self,a):
        self.mask=a['valid_mask']&(a['thickness_m']>0); self.rows,self.cols=np.where(self.mask)
        self.xy=np.column_stack([a['x'][self.cols],a['y'][self.rows]]); self.tree=cKDTree(self.xy)
        self.ids=np.full(self.mask.shape,-1,dtype=np.int32); self.ids[self.mask]=np.arange(len(self.rows))
        dx=abs(float(np.median(np.diff(a['x'])))); dy=abs(float(np.median(np.diff(a['y']))))
        self.area=dx*dy; self.steps=[(dr,dc,float(np.hypot(dc*dx,dr*dy))) for dr in (-1,0,1) for dc in (-1,0,1) if dr or dc]
        self.nr,self.nc=self.mask.shape
    def neighbours(self,i,allow_corner_contact=False):
        r,c=int(self.rows[i]),int(self.cols[i])
        for dr,dc,cost in self.steps:
            rr,cc=r+dr,c+dc
            if not (0<=rr<self.nr and 0<=cc<self.nc): continue
            j=int(self.ids[rr,cc])
            if j<0: continue
            # Do not jump across the corner between two non-ice pixels.
            if not allow_corner_contact and dr and dc and (not self.mask[r+dr,c] or not self.mask[r,c+dc]): continue
            yield j,cost
    def route(self,start,end):
        _,ends=self.tree.query(np.asarray([start,end])); first,last=map(int,ends)
        distance=np.full(len(self.xy),np.inf); parent=np.full(len(self.xy),-1,dtype=np.int32); distance[first]=0.; queue=[(0.,first)]
        while queue:
            d,i=heapq.heappop(queue)
            if d!=distance[i]: continue
            if i==last: break
            for j,cost in self.neighbours(i):
                new=d+cost
                if new<distance[j]: distance[j]=new; parent[j]=i; heapq.heappush(queue,(new,j))
        if not np.isfinite(distance[last]): raise ValueError('No within-ice connector')
        path=[last]
        while path[-1]!=first: path.append(int(parent[path[-1]]))
        q=np.vstack([start,self.xy[path[::-1]],end]); keep=np.r_[True,np.sqrt(np.sum(np.diff(q,axis=0)**2,axis=1))>1e-8]
        return q[keep]
    def partition(self,stations):
        # Multi-source shortest grid-distance partition, not inferred ice flow.
        offset,seeds=self.tree.query(stations); distance=np.full(len(self.xy),np.inf); owner=np.full(len(self.xy),-1,dtype=np.int32); queue=[]
        for k,(i,d) in enumerate(zip(seeds,offset)):
            if d<distance[i]: distance[i]=d; owner[i]=k; heapq.heappush(queue,(float(d),k,int(i)))
        while queue:
            d,k,i=heapq.heappop(queue)
            if d!=distance[i] or k!=owner[i]: continue
            for j,cost in self.neighbours(i,allow_corner_contact=True):
                new=d+cost
                if new<distance[j]-1e-10 or (abs(new-distance[j])<=1e-10 and k<owner[j]):
                    distance[j]=new; owner[j]=k; heapq.heappush(queue,(new,k,j))
        if np.any(owner<0): raise ValueError('Some ice pixels have no connected profile')
        return owner,distance


def build():
    started=time.perf_counter(); control=validate(); facts=digest(PROJECT/'LIA_PROJECT_FACTS.md')
    package=json.loads((SOURCE/'input-package.json').read_text()); net=read_network(NETWORK)
    if digest(SOURCE/'geometry.npz')!=package['array_archive']['sha256'].lower(): raise ValueError('Spatial source changed')
    if net['crs']!=package['metadata']['crs_wkt']: raise ValueError('Network/source CRS mismatch')
    with np.load(SOURCE/'geometry.npz',allow_pickle=False) as f: a={k:f[k].copy() for k in f.files}
    for p in (OUT,REPORT,EVIDENCE): p.mkdir(parents=True,exist_ok=False)
    plan={'schema':'polim-spatial-profile-plan-v1','system':2387,'user_authorization':'ok 开始; 不同支流冰厚不对', 'source_pair_rule':'LIA_PROJECT_FACTS Sections168/176: anchor-reroute V3/F19 plus anchor-reroute V3 network; >1e-6 ice mask',
          'facts_sha256':facts,'source_package':ref(SOURCE/'input-package.json'),'network':ref(NETWORK),
          'code':ref(Path(__file__)),'data_reader':ref(Path(__file__).with_name('network_data_reader.py')),
          'scope':'Static geographic profiles and explicit pixel partition; no dynamics or climate decoding',
          'spacing_m':50.,'original_routes':'Preserve original coordinates and registered parent indices',
          'gaps':'Shortest 8-neighbour route within positive source ice, no corner cutting, join exact registered parent point; inferred connectors distinctly labeled',
          'field_sampling':'Nearest positive source-ice cell, sampling offset saved, no modifications to source raster',
          'partition':'8-neighbour within-ice shortest-distance Voronoi including source corner contacts; deterministic ties; not a hydrological catchment or velocity-derived flow field',
          'technical_revision':'v3 corrects source/network pair and authoritative positive-ice mask; prior v2 profiles withdrawn as model inputs',
          'geometry_fields':'Centreline bed/thickness separate from area/volume-equivalent thickness and width; no implicit selection for momentum',
          'terminal_policy':'All source ice pixels assigned once, area/volume closed, parent graph acyclic, junction coordinates shared; no model promotion'}
    write_json(EVIDENCE/'plan.json',plan); write_json(OUT/'original-network.json',plain(net))
    graph=IceGraph(a); branches={int(b['branch_index']):b for b in net['branches']}; main=int(net['main_branch_index'])
    if len(branches)!=8 or branches[main]['parent_index'] is not None: raise ValueError('Unexpected source topology')
    for i in branches:
        visited=set(); k=i
        while k is not None:
            if k in visited or k not in branches: raise ValueError('Invalid/cyclic topology')
            visited.add(k); k=branches[k]['parent_index']
        if main not in visited: raise ValueError('Branch not connected to declared main root')
    joined={}; junctions={}; connectors={}; source_lengths={}
    for i,b in branches.items():
        q=np.asarray(b['coords_projected_m'],dtype=float); source_lengths[i]=float(arclength(q)[-1])
        if np.any(np.diff(arclength(q))<=0) or not np.isfinite(q).all(): raise ValueError('Invalid original route')
        parent=b['parent_index']
        if parent is not None:
            station,target,gap=project(np.asarray(branches[parent]['coords_projected_m']),np.asarray(b['flows_to_point_projected_m']))
            if gap>1e-6: raise ValueError('Registered junction not on parent route')
            connector=graph.route(q[-1],target); joined[i]=np.vstack([q,connector[1:]]); connectors[i]=connector
            junctions[i]={'parent':int(parent),'parent_station_m':station,'coordinate':target.tolist(),
                          'original_endpoint_gap_m':float(np.linalg.norm(q[-1]-target)),'inferred_connector_length_m':float(arclength(connector)[-1])}
        else: joined[i]=q; connectors[i]=np.empty((0,2))
    offsets={}; points=[]; station_arrays={}; ds_arrays={}; global_ids=[]; local_ids=[]; records=[]; base=0
    for i in sorted(branches):
        length=float(arclength(joined[i])[-1]); stops=[0.,length,source_lengths[i]]
        stops.extend(j['parent_station_m'] for j in junctions.values() if j['parent']==i)
        s=np.sort(np.r_[np.arange(0,length,50.),stops]); s=s[np.r_[True,np.diff(s)>1e-7]]
        p=sample(joined[i],s); faces=np.r_[0.,(s[:-1]+s[1:])/2,length]; ds=np.diff(faces)
        if np.any(ds<=0): raise ValueError('Invalid along-profile cell length')
        offsets[i]=(base,len(s)); station_arrays[i]=s; ds_arrays[i]=ds; points.extend(p); global_ids.extend([i]*len(s)); local_ids.extend(range(len(s))); base+=len(s)
    points=np.asarray(points); owner,distance=graph.partition(points); counts=np.bincount(owner,minlength=len(points)); area=counts*graph.area
    source_h=a['thickness_m'][graph.mask]; source_b=a['bed_elevation_m'][graph.mask]
    volume=np.bincount(owner,weights=source_h*graph.area,minlength=len(points))
    bedsum=np.bincount(owner,weights=source_b*graph.area,minlength=len(points))
    has=area>0; equivalent_h=np.divide(volume,area,out=np.full_like(area,np.nan),where=has)
    equivalent_b=np.divide(bedsum,area,out=np.full_like(area,np.nan),where=has)
    offset,nearest=graph.tree.query(points); rr=graph.rows[nearest]; cc=graph.cols[nearest]
    ds=np.concatenate([ds_arrays[i] for i in sorted(branches)])
    profiles={'branch_index':np.asarray(global_ids),'local_index':np.asarray(local_ids),'easting_m':points[:,0],'northing_m':points[:,1],
              'station_m':np.concatenate([station_arrays[i] for i in sorted(branches)]),'cell_length_m':ds,
              'source_row':rr,'source_col':cc,'sampling_offset_m':offset,
              'centreline_bed_m':a['bed_elevation_m'][rr,cc],'centreline_thickness_m':a['thickness_m'][rr,cc],
              'centreline_surface_m':a['surface_elevation_m'][rr,cc],
              'allocated_pixel_count':counts,'allocated_area_m2':area,'allocated_volume_m3':volume,
              'equivalent_width_m':area/ds,'equivalent_thickness_m':equivalent_h,'equivalent_mean_bed_m':equivalent_b,
              'has_allocated_ice':has}
    for i in sorted(branches):
        start,n=offsets[i]; take=slice(start,start+n); junction=junctions.get(i)
        if junction:
            pstart,pn=offsets[junction['parent']]; si=station_arrays[junction['parent']]
            parent_node=int(np.argmin(abs(si-junction['parent_station_m']))); gi=pstart+parent_node
            mismatch=float(np.linalg.norm(points[start+n-1]-points[gi]))
            if mismatch>1e-6: raise ValueError('Shared junction coordinate mismatch')
            junction.update(parent_local_node=parent_node,parent_global_node=gi,child_global_node=start+n-1,coordinate_mismatch_m=mismatch)
        with (OUT/f'branch-{i:02d}.npz').open('xb') as f: np.savez_compressed(f,**{k:v[take] for k,v in profiles.items()})
        records.append({'branch':i,'parent':branches[i]['parent_index'],'global_node_start':start,'node_count':n,
                        'original_length_m':source_lengths[i],'total_length_m':float(station_arrays[i][-1]),
                        'allocated_area_m2':float(area[take].sum()),'allocated_volume_m3':float(volume[take].sum()),
                        'zero_area_nodes':int((~has[take]).sum()),'junction':junction,'file':ref(OUT/f'branch-{i:02d}.npz')})
    owner_raster=np.full(graph.mask.shape,-1,dtype=np.int32); owner_raster[graph.mask]=owner
    branch_raster=np.full(graph.mask.shape,-1,dtype=np.int32); branch_raster[graph.mask]=profiles['branch_index'][owner]
    with (OUT/'pixel-allocation.npz').open('xb') as f: np.savez_compressed(f,node_id=owner_raster,branch_id=branch_raster,source_ice_mask=graph.mask,geodesic_distance_m=distance,source_ice_rows=graph.rows,source_ice_cols=graph.cols)
    features=[]
    for i in sorted(branches):
        for kind,q in [('original',np.asarray(branches[i]['coords_projected_m'])),('inferred_connector',connectors[i])]:
            if len(q): features.append({'type':'Feature','properties':{'branch':i,'parent':branches[i]['parent_index'],'kind':kind},'geometry':{'type':'LineString','coordinates':q.tolist()}})
    # Projected JSON is explicit about CRS and is not mislabeled RFC7946 GeoJSON.
    write_json(OUT/'projected-network.json',{'schema':'projected-line-features-v1','crs_wkt':net['crs'],'features':features})
    source_volume=float(np.sum(source_h,dtype=np.float64)*graph.area); source_area=float(len(source_h)*graph.area)
    closure={'source_ice_pixels':len(source_h),'allocated_pixel_total':int(counts.sum()),'unassigned_pixels':int((owner<0).sum()),
             'area_difference_m2':float(area.sum()-source_area),'volume_difference_m3':float(volume.sum()-source_volume),
             'volume_relative_error':float(abs(volume.sum()-source_volume)/source_volume),
             'source_npz_unchanged':digest(SOURCE/'geometry.npz')==package['array_archive']['sha256'].lower(),
             'network_source_unchanged':digest(NETWORK)==plan['network']['sha256'].lower(),
             'positive_ice_threshold_m':1e-6,'maximum_source_sampling_offset_m':float(offset.max()),'zero_area_nodes':int((~has).sum()),
             'maximum_junction_coordinate_mismatch_m':max(j['coordinate_mismatch_m'] for j in junctions.values()),'acyclic_graph':True}
    if counts.sum()!=len(source_h) or closure['volume_relative_error']>1e-12 or not closure['source_npz_unchanged']: raise ValueError('Source allocation closure failed')
    exported={'schema':'polim-spatial-profile-package-v1','system':2387,'status':'paired-source-diagnostics-complete-not-validated-dynamic-geometry',
              'main_branch':main,'source':ref(SOURCE/'input-package.json'),'plan':ref(EVIDENCE/'plan.json'),'branches':records,
              'partition':ref(OUT/'pixel-allocation.npz'),'network':ref(OUT/'projected-network.json'),'closure':closure,
              'scientific_acceptance':False,'launch_authorized':False,
              'limitations':['Connectors are inferred within-mask shortest paths, not observed flow paths',
                'Pixel allocation is geometric Voronoi, not a drainage-basin validation',
                'Centreline fields and mass-equivalent fields are distinct; v3 single-flowline solver has no tributary junction coupling',
                'Zero-area nodes remain explicit geometry-only nodes; not given artificial width or mass',
                'No 200 km dry tail generated for this new network; downstream geometry remains to be specified before dynamics']}
    write_json(OUT/'input-package.json',exported)
    render(a,net,joined,connectors,profiles,records,branch_raster,closure)
    result={'schema':'polim-spatial-profiles-terminal-v1','status':'geometry-completed','controller_at_start':control,'facts_sha256':facts,
            'package':ref(OUT/'input-package.json'),'report':ref(REPORT/'index.html'),'figure':ref(REPORT/'network.png'),
            'new_simulations':0,'accepted_count_effect':0,'scientific_acceptance':False,'closure':closure,
            'wall_seconds':time.perf_counter()-started,'branch_count':len(records),'profile_nodes':len(points)}
    write_json(EVIDENCE/'terminal.json',result)
    print(json.dumps(result,indent=2))


def render(a,net,joined,connectors,p,records,allocation,c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x0=float(a['x'].min()); y0=float(a['y'].min()); fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    colors=plt.get_cmap('tab10'); mask=allocation>=0; xx=(a['x']-x0)/1000; yy=(a['y']-y0)/1000
    im=axes[0,0].pcolormesh(xx,yy,np.ma.masked_where(~mask,a['thickness_m']),cmap='Greys',shading='nearest'); fig.colorbar(im,ax=axes[0,0],label='Thickness (m)')
    for r in records:
        i=r['branch']; original=np.asarray(net['branches'][i]['coords_projected_m']); q=connectors[i]
        axes[0,0].plot((original[:,0]-x0)/1000,(original[:,1]-y0)/1000,color=colors(i),lw=2,label=f'B{i}')
        if len(q): axes[0,0].plot((q[:,0]-x0)/1000,(q[:,1]-y0)/1000,color=colors(i),ls='--',lw=1.5)
        axes[0,0].text((original[0,0]-x0)/1000,(original[0,1]-y0)/1000,str(i),fontsize=10)
    axes[0,0].set(title='Source routes (solid), inferred gaps (dashed)',xlabel='Local easting (km)',ylabel='Local northing (km)',aspect='equal')
    axes[0,1].pcolormesh(xx,yy,np.ma.masked_where(~mask,allocation),cmap='tab10',vmin=0,vmax=9,shading='nearest')
    axes[0,1].set(title='Unique within-ice spatial allocation',xlabel='Local easting (km)',ylabel='Local northing (km)',aspect='equal')
    for r in records:
        i=r['branch']; take=p['branch_index']==i; s=p['station_m'][take]/1000
        axes[1,0].plot(s,p['centreline_surface_m'][take],color=colors(i),label=f'B{i}')
        axes[1,1].plot(s,p['centreline_thickness_m'][take],color=colors(i),label=f'B{i}')
    axes[1,0].set(title='Source-sampled surface on geographic routes',xlabel='Actual along-route distance (km)',ylabel='Elevation (m)'); axes[1,0].legend(ncol=4)
    axes[1,1].set(title='Source-sampled centreline thickness',xlabel='Actual along-route distance (km)',ylabel='Thickness (m)')
    fig.suptitle('System 2387 | 1 main route + 7 tributaries | static geometry only',fontsize=16)
    fig.savefig(REPORT/'network.png',dpi=150); plt.close(fig)
    rows=''.join(f'<tr><td>{r["branch"]}</td><td>{"出口" if r["parent"] is None else r["parent"]}</td><td>{r["original_length_m"]/1000:.2f}</td><td>{(r["junction"] or {}).get("inferred_connector_length_m",0)/1000:.2f}</td><td>{r["allocated_area_m2"]/1e6:.3f}</td></tr>' for r in records)
    png=base64.b64encode((REPORT/'network.png').read_bytes()).decode()
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 空间剖面与支流</title><style>body{{max-width:1160px;margin:36px auto;padding:0 22px;background:#f3f6f9;color:#183047;font:16px/1.75 system-ui}}section{{background:white;padding:24px;margin:18px 0;border-radius:14px}}img{{width:100%}}a{{color:#206b9b}}td,th{{padding:9px 18px;border-bottom:1px solid #d9e3ed;text-align:left}}table{{border-collapse:collapse;width:100%}}</style><h1>配套 V3 冰厚的空间采样已修正</h1><p>系统 2387 · 1 条主流线、7 条支流 · {len(p['station_m'])} 个剖面节点 · 未运行新的动力学模拟</p>
<section><h2>本次完成</h2><p>沿现有地理路径按约 50 m 间距提取冰厚、底高和表面高程，并保留原始汇流拓扑。主流线长度保持 4.677 km，其他支流各自使用真实路径长度，未将全系统高程排序压缩成一条线。源冰体的 {c['source_ice_pixels']:,} 个像元全部唯一分配，总面积与体积保持闭合。</p><p>部分原支流终点与登记汇流点之间存在缺口；虚线为在正冰厚掩膜内求得的最短连接路径，属于重建假设。彩色分区采用八邻接掩膜内几何距离分配，按正式大于 10⁻⁶ m 的正冰厚掩膜分配；尚不能当作观测或物理流场确定的汇流区。</p></section>
<img src="data:image/png;base64,{png}" alt="原始主支流、推断连接段、唯一像元分区和各支流沿程剖面">
<section><h2>支流明细</h2><table><tr><th>支流</th><th>流入</th><th>原路径 km</th><th>推断连接 km</th><th>分配面积 km²</th></tr>{rows}</table></section>
<section><h2>接入求解器前需要保留的区别</h2><p>每个节点同时保存中心线采样值与面积／体积等效字段，两者没有静默替换。采样用最近的有效有冰像元，最大偏移为 {c['maximum_source_sampling_offset_m']:.2f} m，具体像元索引和距离逐点保存。{c['zero_area_nodes']} 个未分配面积的节点保留为几何节点，没有给它们添加虚构宽度或冰量。</p><p>本版仅修正空间采样与分区诊断，尚不是通过静态科学门槛的动力学几何。第三版仍是单流线求解器，支流汇入的质量与焓通量、汇流处动量连接，以及新的下游无冰域尚需接入；本页不代表多支流热力模拟已完成或科学验收通过。</p><p>合并核对：无遗漏像元；面积差 {c['area_difference_m2']:.3e} m²；体积差 {c['volume_difference_m3']:.3e} m³；汇流接点坐标最大差 {c['maximum_junction_coordinate_mismatch_m']:.3e} m。源几何与原网络文件保持不变。</p>
<p><a href="../../lia-inputs-v1/system-2387/spatial-profiles-v3-paired/input-package.json">剖面输入说明</a> · <a href="../../../validation_data/results/paleo_surface_validation_v0/polim_lia_system2387_spatial_profiles_v3_paired/terminal.json">完整证据</a> · <a href="../spatial-source-audit-v1/index.html">二维来源报告</a></p></section></html>'''
    (REPORT/'index.html').write_text(text,encoding='utf-8')


if __name__=='__main__': build()
