"""Read-only live page and final 50/100m century comparison."""
import os,json,time,copy,re,hashlib,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from .century_v1 import PROJECT,R,E,RUN,REPORT,SOURCE,NAME,hashes
from .connection_control_v1 import arrays,read
from .v3.runner import write_json,digest
from .thermal_attribution_v1 import validate,ref
from .v9.storage import restore

STYLE='body{font:17px Microsoft YaHei,sans-serif;background:#eef3f7;color:#213b4d;margin:0}main{max-width:1400px;margin:auto;padding:28px}p,li{line-height:1.85}table{border-collapse:collapse;width:100%;background:white}td,th{padding:13px;border-bottom:1px solid #dce5eb;text-align:left}img{width:100%;margin:16px 0;border-radius:10px;background:white}.note{background:white;border-left:5px solid #009999;padding:18px}a{color:#087f9e}'

def snapshot():
    REPORT.mkdir(parents=True,exist_ok=True);rows=[]
    for dx in [50,100]:
        p=RUN/f'grid-{dx}m/status.json'
        try:s=read(p)
        except (OSError,ValueError):s={}
        year=s.get('year',s.get('end_year',1));wall=s.get('elapsed_wall_seconds',s.get('wall_seconds',0));stage=s.get('stage',s.get('status','启动中'));steps=s.get('accepted_steps',0)
        # Display last sealed time separately from a transported pending stage.
        f=RUN/f'grid-{dx}m/frame-{steps:03d}/metadata.json'
        try:accepted=read(f)['time']
        except (OSError,ValueError):accepted=1
        rows.append(f'<tr><td>{dx} 米</td><td>{accepted:.2f} / 100 年</td><td>{stage}</td><td>{steps}</td><td>{wall/60:.1f} 分钟</td></tr>')
    page=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>50 / 100 米百年对照 · 运行中</title><style>{STYLE}</style><main><h1>50 / 100 米：推进到第 100 年</h1><p class="note">从已保存的第 1 年接续。30 天步长上限，CFL 自动缩短，每个接受步更新 BP。固定床面和材料，零表面质量收支。下表年份来自已封存检查点。</p><table><tr><th>网格</th><th>已完成模拟年份</th><th>当前阶段</th><th>新增接受步</th><th>实际耗时</th></tr>{"".join(rows)}</table><p>每年保存结果，逐步保留恢复状态；两种网格独立运行。完成后在此展示体积、冰厚和流速差异。</p><p><a href="../grid-triplet-v1/index.html">此前一年网格对照</a> · <a href="../monthly-working-settings-v1/index.html">当前 30 天设置</a></p></main></html>'
    tmp=REPORT/f'.index-{os.getpid()}.tmp';tmp.write_text(page,encoding='utf-8');os.replace(tmp,REPORT/'index.html')

def state_at(dx,year):
    src=SOURCE/f'grid-{dx}m'
    if year==0:folder=src/'frame-000'
    elif year==1:folder=src/'monthly365/frame-013'
    else:
        r=read(RUN/f'grid-{dx}m/years/year-{year:03d}.json');folder=(PROJECT/r['checkpoint']['path']).parent
    meta=read(folder/'metadata.json');st,u,_=restore(folder,meta['contract']);return st,u,folder

def common_map(m50,m100,a100,x100,y100,mask100,points50):
    # Exact P1 interpolation inside each retained100m pixel fan, no extrapolation.
    rc=np.column_stack([np.floor((points50[:,1]-(y100[0]+50))/-100).astype(int),np.floor((points50[:,0]-(x100[0]-50))/100).astype(int)])
    lookup=np.full(mask100.shape,-1,int);lookup[m100['pixel_rows'],m100['pixel_cols']]=np.arange(len(m100['pixel_rows']))
    valid=(rc[:,0]>=0)&(rc[:,0]<lookup.shape[0])&(rc[:,1]>=0)&(rc[:,1]<lookup.shape[1]);ids=np.full(len(rc),-1,int);ids[valid]=lookup[rc[valid,0],rc[valid,1]];valid &= ids>=0
    pix=ids[valid];center=m100['pixel_center_nodes'][pix];cc=m100['xy'][center];d=(points50[valid]-cc)/100;dy=-d[:,1];dx=d[:,0];edge=np.where(abs(dx)>=abs(dy),np.where(dx>=0,1,3),np.where(dy>=0,2,0));corners=m100['pixel_corner_nodes'][pix];tri=np.column_stack([center,corners[np.arange(len(pix)),edge],corners[np.arange(len(pix)),(edge+1)%4]])
    xy=m100['xy'][tri];mat=np.stack([xy[:,1]-xy[:,0],xy[:,2]-xy[:,0]],axis=2);w=np.linalg.solve(mat,points50[valid]-xy[:,0]);weights=np.column_stack([1-w.sum(axis=1),w]);assert weights.min()>-1e-9
    out=np.full((len(rc),)+a100.shape[1:],np.nan);out[valid]=np.einsum('ij,ij...->i...',weights,a100[tri]);return out,valid

def final_report():
    validate();result=read(RUN/'result.json');assert result['implementation']==hashes();REPORT.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':11})
    series={};summary=[];verified=0
    from .v10.model import topology
    for dx in [50,100]:
        src=SOURCE/f'grid-{dx}m';root=RUN/f'grid-{dx}m';mesh=arrays(src/'mesh.npz');geo=topology(mesh);A=arrays(src/'mapping.npz')['A'];vals=[]
        for year in [0,1]:
            st,u,_=state_at(dx,year);vals.append(dict(year=year,volume_m3=float(st['h']@geo['dual_area']),max_H_m=float(st['h'].max()),max_speed_m_year=float(np.linalg.norm(u[:,-1],axis=1).max()),export_m3=0 if year==0 else read(src/'monthly365/result.json')['export_m3'],wall_seconds=0))
        for p in sorted((root/'years').glob('year-*.json')):
            a=read(p);m=a['metrics'];vals.append(dict(year=a['year'],volume_m3=m['volume_m3'],max_H_m=m['max_thickness_m'],max_speed_m_year=m['max_speed_m_year'],export_m3=m['full_export_m3'],wall_seconds=m['elapsed_wall_seconds']))
        # Recount accepted states independently of mutable progress files.
        for folder in root.glob('frame-*'):
            meta=read(folder/'metadata.json');st,u,_=restore(folder,meta['contract']);assert np.isfinite(st['h']).all() and np.min(st['h'])>=0 and np.isfinite(u).all() and np.array_equal(st['A'],A);verified+=1
        series[str(dx)]=vals;last=vals[-1];status=read(root/'result.json') if (root/'result.json').exists() else read(root/'status.json');summary.append(dict(grid_m=dx,status=status['status'],last_year=last['year'],volume_year0_km3=vals[0]['volume_m3']/1e9,volume_final_km3=last['volume_m3']/1e9,volume_change_percent=(last['volume_m3']/vals[0]['volume_m3']-1)*100,max_speed_final_m_year=last['max_speed_m_year'],max_H_final_m=last['max_H_m'],wall_seconds=status['wall_seconds'],mass_relative=status.get('mass_relative'),accepted_steps=status.get('accepted_steps')))
    plt.figure(figsize=(14,4.4));fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for dx,color in [(50,'#078e9b'),(100,'#d78048')]:
        v=series[str(dx)];years=[x['year'] for x in v]
        for ax,key,scale in zip(axs,['volume_m3','max_speed_m_year','max_H_m'],[1e9,1,1]):ax.plot(years,[x[key]/scale for x in v],label=f'{dx} 米',color=color,lw=2);ax.set_xlabel('模拟年份');ax.legend()
    for ax,title,unit in zip(axs,['总冰体积','最大表面速度','最大冰厚'],['立方千米','米/年','米']):ax.set(title=title,ylabel=unit);ax.grid(alpha=.2)
    fig.savefig(REPORT/'evolution.png',dpi=150);plt.close('all')
    maps=[];m50=arrays(SOURCE/'grid-50m/mesh.npz');m100=arrays(SOURCE/'grid-100m/mesh.npz');mp100=arrays(SOURCE/'grid-100m/mapping.npz');ci=m50['pixel_center_nodes'];points=m50['xy'][ci]
    end=min(summary[0]['last_year'],summary[1]['last_year']);selected=[y for y in [1,10,25,50,100] if y<=end]
    for year in selected:
        a,u,_=state_at(50,year);b,v,_=state_at(100,year);mapped,ok=common_map(m50,m100,b['h'],mp100['x'],mp100['y'],mp100['mask'],points);mv,_=common_map(m50,m100,v[:,-1],mp100['x'],mp100['y'],mp100['mask'],points);dh=mapped[ok]-a['h'][ci][ok];du=mv[ok]-u[ci,-1][ok]
        maps.append(dict(year=year,common_area_km2=float(ok.sum()*2500/1e6),H_difference_RMS_m=float(np.sqrt(np.mean(dh**2))),H_difference_max_m=float(abs(dh).max()),velocity_difference_relative_L2=float(np.sqrt(np.sum(du**2)/max(np.sum(u[ci,-1][ok]**2),1e-100)))))
    st50,u50,_=state_at(50,end);st100,u100,_=state_at(100,end);fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained');x0=m50['xy'][:,0].min();y0=m50['xy'][:,1].min();hmax=max(st50['h'].max(),st100['h'].max());umax=max(np.linalg.norm(u50[:,-1],axis=1).max(),np.linalg.norm(u100[:,-1],axis=1).max())
    for j,(dx,mesh,st,u) in enumerate([(50,m50,st50,u50),(100,m100,st100,u100)]):
        for i,values in enumerate([st['h'],np.linalg.norm(u[:,-1],axis=1)]):
            ax=axs[i,j];im=ax.tripcolor((mesh['xy'][:,0]-x0)/1000,(mesh['xy'][:,1]-y0)/1000,mesh['triangles'],values,shading='gouraud',cmap=['Blues','viridis'][i],vmin=0,vmax=[hmax,umax][i],rasterized=True);ax.set(aspect='equal',title=f'{dx} 米 · 第 {end} 年'+['冰厚','表面速度'][i],xlabel='东西距离（km）',ylabel='南北距离（km）');fig.colorbar(im,ax=ax,shrink=.7,label=['米','米/年'][i])
    fig.savefig(REPORT/'final-maps.png',dpi=130);plt.close(fig)
    out=dict(status=result['status'],grids=summary,series=series,spatial_comparison=maps,verified_states=verified,source=ref(RUN/'result.json'),interpretation='Year0/1 retained from prior run; years2-100 continued. Fixed material,zeroSMB,no thermalfeedback,fixed outer footprint. Velocity/geometry differences include initial grid representation. 100m fields interpolated with native P1 triangles onto50m cellcenters within common coverage only;no extrapolation. Volume uses exactFEMdualareas.')
    write_json(REPORT/'summary.json',out);table=''
    for v in summary:table+=f'<tr><td>{v["grid_m"]} 米</td><td>{v["last_year"]}</td><td>{v["volume_final_km3"]:.6f}</td><td>{v["volume_change_percent"]:+.3f}%</td><td>{v["max_speed_final_m_year"]:.3f}</td><td>{v["max_H_final_m"]:.2f}</td><td>{v["wall_seconds"]/60:.1f} 分钟</td></tr>'
    page=f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>50 / 100 米百年对照</title><style>{STYLE}</style><main><h1>50 / 100 米：百年动力学对照</h1><p class="note">从各自已保存的第 1 年继续，30 天步长上限，CFL 自动缩短，BP 每步更新。第 0、1 年结果直接沿用，没有重跑。</p><table><tr><th>网格</th><th>完成年份</th><th>末期体积（km³）</th><th>相对第0年</th><th>最大速度（m/年）</th><th>最大冰厚（m）</th><th>续算耗时</th></tr>{table}</table><img src="evolution.png" alt="冰体积、流速和冰厚的百年变化"><img src="final-maps.png" alt="两种网格末期冰厚与速度"><p>固定床面和材料、零表面质量收支，当前计算范围固定；冰可向外流出。曲线展示这一设置下的动力调整，不含气候和热力驱动。总量预算计入边界流出。</p><p><a href="summary.json">完整年度数值与共同区域空间差异</a> · <a href="../grid-triplet-v1/index.html">此前一年对照</a></p></main></html>'
    (REPORT/'index.html').write_text(page,encoding='utf-8');write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},verified_states=verified));print(json.dumps(dict(grids=summary,spatial_comparison=maps,verified_states=verified)))

def finish():
    validate();v=read(REPORT/'summary.json');result=read(RUN/'result.json')
    for p,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/p)==h
    home=R/'index.html';old=home.read_text(encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
    (REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
    link='<a class="current" href="reports/century-50-100-v1/index.html"><b>新增：50 / 100 米百年对照</b><p>查看实际年度体积、冰厚、流速及两种网格的耗时。</p></a>'
    home.write_text(old.replace(anchor,anchor+link,1),encoding='utf-8')
    write_json(REPORT/'navigation-delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},home_sha256=digest(home)))
    cp=E/'completion.json';write_json(cp,dict(status=v['status'],summary=ref(REPORT/'summary.json'),delivery=ref(REPORT/'navigation-delivery.json'),result=ref(RUN/'result.json'),accepted_count_effect=0))
    sp=PROJECT/'LIA_CURRENT_STATE.json';sb=sp.read_bytes();s=json.loads(sb);fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();assert digest(fp)==s['scientific_contract']['sha256'].lower();num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8-sig'),re.M)))+1
    add=f'\n\n## {num}. System2387 50/100m century continuation\n\n- User authorized100years and100m comparison. Continued retainedv20year1states with30daycap,CFL0.35,unchangedBP andfixedA/bed/zeroSMB/fixedouterdomain. No initial orfirstyear replay. TwoCPUworkers8threads each;exactcheckpoint/statusfailure preflightpassed.\n'
    for a in v['grids']:add+=f'- {a["grid_m"]}m: status{a["status"]},lastyear{a["last_year"]},finalvolume{a["volume_final_km3"]:.12g}km3,changefromyear0{a["volume_change_percent"]:.9g}percent;maxspeed{a["max_speed_final_m_year"]:.9g}m/year,maxH{a["max_H_final_m"]:.9g}m;wall{a["wall_seconds"]:.6f}s,steps{a["accepted_steps"]},massrelative{a["mass_relative"]}.\n'
    add+=f'- {v["verified_states"]}acceptedstates restored/hashverified,finite/nonnegative,fixedA. Spatialcomparison uses100mP1fields at50mcenters oncommoncoverage;initialprojectiondifferencesretained. No climate/thermal transient,observationalacceptance,productionpromotion orcountchange. Evidence `{ref(cp)["path"]}`.\n'
    fh=hashlib.sha256(fb+add.encode('utf-8')).hexdigest().upper();b={**s['research_branches'][NAME],'status':v['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],'evidence':ref(cp),'next_legal_action':'Read century comparison; no automatic longer continuation'}
    route=read(PROJECT/s['authoritative_route']['path']);route['version']+='+polim-v22-century-completed';route[NAME]=b
    if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
    rp=E/'route-completed.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=b;n['scientific_contract']['sha256']=fh;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+='+polim-v22-century-completed';n['updated_at']=datetime.now(timezone.utc).isoformat()
    if sp.read_bytes()!=sb or fp.read_bytes()!=fb:raise RuntimeError('Concurrent controller change')
    tmp=sp.with_name(f'.LIA_CURRENT_STATE.centuryclose.{os.getpid()}.tmp');write_json(tmp,n)
    with fp.open('ab') as f:f.write(add.encode('utf-8'));f.flush();os.fsync(f.fileno())
    os.replace(tmp,sp)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {n["updated_at"]} - Century comparison completed\n\n50/100m savedyear1 continuations finalized; annual outputs and gridcomparison preserved. Facts{num};`{ref(cp)["path"]}`.\n')
    write_json(E/'completed-validation.json',validate());print(json.dumps(dict(status=v['status'],facts_section=num)))

if __name__=='__main__':
    action=sys.argv[1]
    if action=='snapshot':snapshot()
    elif action=='report':final_report()
    elif action=='finish':finish()
    elif action=='watch':
        while not (RUN/'result.json').exists():snapshot();time.sleep(30)
        final_report()
