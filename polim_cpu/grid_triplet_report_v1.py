"""Combined read-only comparison, plots, and terminal transaction for v20."""
from .grid_triplet_v1 import PROJECT,R,E,I,SEED,RUN,REPORT,NAME,arrays,read,codehash,transition
from .v3.runner import write_json,digest
from .thermal_attribution_v1 import validate,ref
from .v9.storage import restore
from .v10.model import topology
import numpy as np
import json,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def report():
    validate();result=read(RUN/'result.json');term=read(RUN/'terminal-evidence.json');assert term['implementation']==codehash()
    for p,h in term['files'].items(): assert digest(RUN/p)==h,p
    REPORT.mkdir(parents=True,exist_ok=False);rows=[];verified=0;loaded=[]
    for c in result['cases']:
        if c['status']!='completed': continue
        g=c['grid'];dx=g['cell_m'];root=RUN/f'grid-{dx}m';mesh=arrays(root/'mesh.npz');mapping=arrays(root/'mapping.npz');geo=topology(mesh)
        for folder in root.rglob('frame-*'):
            if not folder.is_dir(): continue
            state,u,meta=restore(folder,result['contract']);assert np.array_equal(state['A'],mapping['A']) and np.isfinite(state['h']).all() and np.min(state['h'])>=0 and np.isfinite(u).all();verified+=1
        initial,ui,_=restore(root/'frame-000',result['contract']);final,uf,_=restore(root/'monthly365'/f'frame-{c["monthly"]["BP_solves"]:03d}',result['contract'])
        mask=mapping['mask'];coverage=mapping['coverage'];native_area=float(coverage.sum()*dx**2);outside=float(((1-coverage)[mask]).sum()*dx**2);missing=float(coverage[~mask].sum()*dx**2)
        row=dict(grid_m=dx,nodes=g['nodes'],area_change_percent=g['initial_area_change_relative']*100,footprint_symmetric_difference_percent=(outside+missing)/native_area*100,fringe_volume_transfer_percent=g['fringe_volume_fraction']*100,fringe_max_distance_m=g['fringe_transfer_max_center_distance_m'],volume_remap_relative=g['volume_remap_relative'],components=g['components'],spatial_velocity_error_percent=g['spatial_velocity_error']['velocity_relative_L2']*100,spatial_peak_error_percent=g['peak_relative_error']*100,spatial_pass=g['spatial_screen_pass'],monthly_daily_H_max_m=c['month_vs_day']['H_max_m'],monthly_daily_H_RMS_m=c['month_vs_day']['H_RMS_m'],monthly_daily_velocity_percent=c['month_vs_day']['velocity_relative_L2']*100,temporal_pass=c['temporal_screen_pass'],annual_wall_seconds=c['monthly']['wall_seconds'],daily30_wall_seconds=c['daily']['wall_seconds'],initial_BP_seconds=g['initial_BP']['wall_seconds'],annual_BP_solves=c['monthly']['BP_solves'],annual_transport_substeps=c['monthly']['substeps'],annual_dt_min_days=min(v['dt_days'] for v in c['monthly']['timeline']),annual_dt_max_days=max(v['dt_days'] for v in c['monthly']['timeline']),conditional_1000yr_hours=c['millennium_hours_conditional'],initial_peak_m_year=g['peak_m_year'],final_peak_m_year=float(np.linalg.norm(uf[:,-1],axis=1).max()),annual_H_change_max_m=float(abs(final['h']-initial['h']).max()),annual_export_m3=c['monthly']['export_m3'],annual_mass_relative=c['monthly']['checks']['mass_relative'])
        rows.append(row);loaded.append((mesh,initial,ui,final,uf,c,mapping))
    # All fields below describe computed cases, never a fixed choice before outcomes.
    passing=[r['grid_m'] for r in rows if r['spatial_pass'] and r['temporal_pass']]
    summary=dict(status=result['status'],rows=rows,screen_passed_grid_m=passing,sealed_states_verified=verified,total_wall_seconds=result['wall_seconds'],failed_cases=[c for c in result['cases'] if c['status']!='completed'],interpretation='365day fixed-bed/material zero-SMB BP evolution with fixed footprint. Temporal error is evaluated at30days against daily stepping, not a one-year daily reference. Spatial velocity error compares initial coarse solutions with conservatively projected native velocity. Costs include evolution/checkpoints but exclude initial remapping/BP and future climate/thermal/domain changes. Exact outline not preserved; report signed area and symmetric difference separately.',source=ref(RUN/'result.json'))
    halfroot=R/'runs/v21-grid50-halfmonth-v1';half=read(halfroot/'result.json')
    for p,h in read(halfroot/'terminal-evidence.json')['files'].items(): assert digest(halfroot/p)==h,p
    summary['halfmonth50']=dict(screen_pass=half['screen_pass'],error=half['error'],wall_seconds=half['evolution']['wall_seconds'],days_executed=30,step_cap_days=15,evidence=ref(halfroot/'result.json'))
    summary['working_configuration']=dict(grid_m=50,step_cap_days=15,scope='15day cap tested for30days;365day trajectory uses30day cap') if half['screen_pass'] and any(v['grid_m']==50 and v['spatial_pass'] for v in rows) else None
    write_json(REPORT/'summary.json',summary)
    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),'axes.unicode_minus':False,'font.size':10})
    if rows:
        fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');labels=[f'{v["grid_m"]} m' for v in rows]
        axs[0].bar(labels,[v['annual_wall_seconds'] for v in rows],color='#058e9e');axs[0].set(title='实际推进 365 天',ylabel='计算耗时（秒）')
        axs[1].bar(labels,[v['spatial_velocity_error_percent'] for v in rows],color='#467eac');axs[1].axhline(10,ls='--',color='#c64448');axs[1].set(title='初始速度场与细网格投影之差',ylabel='面积加权相对差异（%）')
        axs[2].bar(labels,[v['monthly_daily_H_max_m'] for v in rows],color='#de9354');axs[2].axhline(.1,ls='--',color='#c64448');axs[2].set(title='第 30 天：月更新与日更新之差',ylabel='最大冰厚差（米）')
        fig.savefig(REPORT/'comparison.png',dpi=150);plt.close(fig)
        source=arrays(I/'mesh.npz');x0=source['xy'][:,0].min();y0=source['xy'][:,1].min();limits=((source['xy'][:,0].min()-x0)/1000,(source['xy'][:,0].max()-x0)/1000,(source['xy'][:,1].min()-y0)/1000,(source['xy'][:,1].max()-y0)/1000)
        fig,axs=plt.subplots(3,len(rows),figsize=(5.2*len(rows),13),squeeze=False,layout='constrained')
        vmax=max(v['initial_peak_m_year'] for v in rows);hmax=max(float(z[1]['h'].max()) for z in loaded);dmax=max(v['annual_H_change_max_m'] for v in rows)
        boundary=source['xy'][source['boundary_edges']]
        from matplotlib.collections import LineCollection
        for j,(mesh,si,ui,sf,uf,c,mp) in enumerate(loaded):
            dx=c['grid']['cell_m'];xy=mesh['xy'];fields=[si['h'],np.linalg.norm(ui[:,-1],axis=1),sf['h']-si['h']]
            for k in range(3):
                ax=axs[k,j];im=ax.tripcolor((xy[:,0]-x0)/1000,(xy[:,1]-y0)/1000,mesh['triangles'],fields[k],shading='gouraud',cmap=['Blues','viridis','RdBu_r'][k],vmin=[0,0,-dmax][k],vmax=[hmax,vmax,dmax][k],rasterized=True)
                lines=(boundary-np.array([x0,y0]))/1000;ax.add_collection(LineCollection(lines,colors='#424c54',linewidths=.25,alpha=.5));ax.set(aspect='equal',xlim=limits[:2],ylim=limits[2:],title=f'{dx} 米 · '+['初始冰厚','初始表面速度','365 天冰厚变化'][k],xlabel='东西距离（km）',ylabel='南北距离（km）');fig.colorbar(im,ax=ax,shrink=.7,label=['米','米/年','米'][k])
        fig.savefig(REPORT/'maps.png',dpi=130);plt.close(fig)
    table=''
    for v in rows:
        table+=f'<tr><td>{v["grid_m"]} m</td><td>{v["nodes"]:,}</td><td>{v["area_change_percent"]:+.2f}%</td><td>{v["spatial_velocity_error_percent"]:.2f}%</td><td>{v["monthly_daily_H_max_m"]:.3f} m</td><td>{v["monthly_daily_velocity_percent"]:.3f}%</td><td>{v["annual_wall_seconds"]:.1f} 秒</td><td>{v["conditional_1000yr_hours"]:.1f} 小时</td><td>空间：{"通过" if v["spatial_pass"] else "未通过"}<br>月步长：{"通过" if v["temporal_pass"] else "未通过"}</td></tr>'
    detail=''.join(f'<li>{v["grid_m"]} 米：边界对称差 {v["footprint_symmetric_difference_percent"]:.2f}%；边缘转移冰量 {v["fringe_volume_transfer_percent"]:.3f}%；一年 BP 更新 {v["annual_BP_solves"]} 次；最大冰厚变化 {v["annual_H_change_max_m"]:.2f} 米；初始/末期最大速度 {v["initial_peak_m_year"]:.2f}/{v["final_peak_m_year"]:.2f} 米/年。</li>' for v in rows)
    verdict=('通过本次空间与时间步长两项标准的网格：'+ '、'.join(f'{x} 米' for x in passing)+'。') if passing else '三种网格已完成对照，尚无网格同时通过本次空间与时间步长两项标准。'
    if summary['working_configuration']:
        verdict=f'本次工作组合：50 米网格＋15 天步长上限。半月对照新增两次 BP 求解，前 30 天相对每日更新的最大冰厚差 {half["error"]["H_max_m"]*100:.2f} 厘米，速度差 {half["error"]["velocity_relative_L2"]*100:.3f}%，通过预设标准。以下表格保留原 30 天步长的一年对照。'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>50 / 100 / 200 米 · BP 一年试算</title><style>body{{font:16px Microsoft YaHei,sans-serif;background:#eff4f7;color:#203746;margin:0}}main{{max-width:1500px;margin:auto;padding:26px}}p,li{{line-height:1.9}}img{{width:100%;background:white;border-radius:12px;margin:16px 0}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}td,th{{padding:13px;text-align:left;border-bottom:1px solid #dbe4eb}}.scroll{{overflow:auto}}.note{{background:white;border-left:5px solid #058e9e;padding:18px}}a{{color:#087f9e}}</style><main><h1>50 / 100 / 200 米：BP 模型已推进一年</h1><p class="note">{verdict}三种网格共用初始源几何和材料；30 天为步长上限，按 CFL 自动缩短，每步更新 BP 速度。前 30 天另有每日更新对照。</p><div class="scroll"><table><tr><th>网格</th><th>节点</th><th>面积变化</th><th>初始速度差</th><th>月/日冰厚差</th><th>月/日速度差</th><th>一年实际耗时</th><th>千年条件外推</th><th>预设标准</th></tr>{table}</table></div><img src="comparison.png" alt="网格精度与一年计算耗时"><img src="maps.png" alt="三种网格的初始冰厚、速度与一年冰厚变化"><p>地图灰线为原始细网格边界，各列采用相同范围与色标。重采样采用像元精确重叠；目标格冰川覆盖比例达到一半则保留。舍弃边缘格中的冰量转移到最近保留格，总体积守恒。面积净变化小不代表边界完全相同，因此另列边界对称差和转移冰量。</p><ul>{detail}</ul><p>这是一年零质量收支、固定材料和固定计算范围下的动力学试算。月/日误差对应第 30 天；一年末态没有每日步长参考；15 天步长另实算 30 天。千年成本从实测一年线性外推，包含逐步存储，不包含初始化、气候、热力及长期范围变化成本。</p><p>预设标准：面积净变化≤5%，连通分量不变，初始速度加权差≤10%，峰值差≤20%；第30天月/日冰厚最大差≤0.1米，速度加权差≤1%。</p><p><a href="summary.json">完整数值</a> · <a href="../millennium-cost-v1/index.html">上一轮 63/126 米对照</a></p></main></html>'''
    (REPORT/'index.html').write_text(page,encoding='utf-8');write_json(REPORT/'delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},states_verified=verified));print(json.dumps(summary,ensure_ascii=True))

def finish():
    validate();s=read(REPORT/'summary.json');result=read(RUN/'result.json')
    for p,h in read(REPORT/'delivery.json')['files'].items(): assert digest(REPORT/p)==h
    home=R/'index.html';old=home.read_text(encoding='utf-8');(REPORT/'previous-project-index.html').write_text(old,encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
    link='<a class="current" href="reports/grid-triplet-v1/index.html"><b>新增：50 / 100 / 200 米 · 一年试算</b><p>三种网格、月步长与日更新对照的实际耗时和冰厚变化。</p></a>';home.write_text(old.replace(anchor,anchor+link,1),encoding='utf-8')
    write_json(REPORT/'navigation-delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},home_sha256=digest(home)))
    completion=dict(status=result['status'],summary=ref(REPORT/'summary.json'),result=ref(RUN/'result.json'),delivery=ref(REPORT/'navigation-delivery.json'),accepted_count_effect=0,production_replacement=False,screen_passed_grid_m=s['screen_passed_grid_m']);write_json(E/'completion.json',completion)
    facts=(PROJECT/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');num=max(map(int,re.findall(r'^## (\d+)\.',facts,re.M)))+1
    addition=f'\n\n## {num}. System2387 exact50/100/200m BP year comparison\n\n- User authorized three grids and30day maximum adaptive timestep. Same sealedv16 independent-outlet source, unchanged three-sigma BP/CFL0.35 conservative transport, fixed material/bed/domain andzeroSMB/thermalfeedback. Eachgrid oneinitial BP,30days daily control and365day adaptive30day path, BPupdated everyacceptedstep;day30endpoint preserved without replay. Mapping: exactrectangleoverlap,majoritycoverage mask,nearestretainedcell transfer of fringe extensivevolume/area/bed/A/velocitymoments,existingcorner averaging. This preserves totalvolume,not exact outline orlocalgeometry. No globalHrescaling orparameterfit.\n'
    for v in s['rows']:
        addition+=f'- {v["grid_m"]}m: {v["nodes"]}nodes;area net{v["area_change_percent"]:.6f}percent,symmetricdifference{v["footprint_symmetric_difference_percent"]:.6f}percent;fringevolume transferred{v["fringe_volume_transfer_percent"]:.6f}percent;initialvelocityL2difference{v["spatial_velocity_error_percent"]:.6f}percent,spatialpass={v["spatial_pass"]}. Day30monthlyvsdailyHmax{v["monthly_daily_H_max_m"]:.9g}m,velocityL2{v["monthly_daily_velocity_percent"]:.6f}percent,temporalpass={v["temporal_pass"]}.365dayswall{v["annual_wall_seconds"]:.6f}s,{v["annual_BP_solves"]}BPupdates,steps{v["annual_dt_min_days"]:.6g}-{v["annual_dt_max_days"]:.6g}days;conditional1000year{v["conditional_1000yr_hours"]:.6f}hours. Peakvelocityinitial/final{v["initial_peak_m_year"]:.6f}/{v["final_peak_m_year"]:.6f}m/year;maximumannualHchange{v["annual_H_change_max_m"]:.6f}m;relativebudget{v["annual_mass_relative"]:.6g}.\n'
    h=s['halfmonth50'];addition+=f'- Separatelyregisteredv21 reuses50m initial/dailyreference, twoBPsolves with15daycap to30days. Hmaxdifference{h["error"]["H_max_m"]:.9g}m,velocityL2{h["error"]["velocity_relative_L2"]:.9g},screenpass={h["screen_pass"]};wall{h["wall_seconds"]:.6f}s. Workingconfiguration{s["working_configuration"]}. Original30daycapfailuresretained,thresholdsnotwaived. Threehalfmonthstatesvalidated.\n'
    addition+=f'- Status{s["status"]};{s["sealed_states_verified"]}sealedstates hashrestored andfinite/nonnegative/fixedA. Bothscreens passedgrids{s["screen_passed_grid_m"]}. Timeaccuracycomparison covers30days,notwholeyeardailyreference;thousandyearcost excludesinitialization/climate/thermal/domain/runtimechanges. Accuracy failuresnotpromoted,thresholdsunchanged. No historicalsimulation,formalacceptance,mainlinechange orcountchange. Evidence `{ref(E/"completion.json")["path"]}`.\n'
    transition('completed',E/'completion.json',addition);print(json.dumps(dict(status='completed',facts_section=num,passed=s['screen_passed_grid_m'])))

if __name__=='__main__':
    import sys
    {'report':report,'finish':finish}[sys.argv[1]]()
