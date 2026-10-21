"""Saved-state figures; no simulation is started by rendering."""
from pathlib import Path
import json
import numpy as np
from . import gpu_long_storage_v1 as st

def render(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve();project=Path(__file__).resolve().parents[1];cfg=st.read(root/'config.json');out=st.read(project/cfg['evidence_root']/'completion.json')
    dest=project/'polim_cpu/reports'/cfg['case_id'];dest.mkdir(exist_ok=True)
    rows=out['rows'];z=st.arrays(root/'input.npz');last=rows[-1] if rows else None
    if rows:
        fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
        years=[cfg['start_year']]+[r['year'] for r in rows]
        axes[0].plot(years,[out['initial_volume_m3']/1e9]+[r['volume_m3']/1e9 for r in rows]);axes[0].set_ylabel('Ice volume (km3)')
        axes[1].plot([r['year'] for r in rows],[r['max_speed_m_year'] for r in rows]);axes[1].set_ylabel('Maximum surface speed (m/year)')
        axes[2].plot([r['year'] for r in rows],np.array([r['smb_volume_m3'] for r in rows])/1e6,label='Surface mass balance');axes[2].plot([r['year'] for r in rows],-np.array([r['exported_m3'] for r in rows])/1e6,label='Boundary export');axes[2].set_ylabel('Cumulative volume change (million m3)');axes[2].legend(fontsize=8)
        for ax in axes:ax.set_xlabel('Calendar year');ax.grid(alpha=.2)
        fig.savefig(dest/'evolution.png',dpi=160);plt.close(fig)
        state,_=st.restore(root/f'frame-{last["step"]:06d}',st.read(root/'contract.json')['contract'])
        fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained');xy=z['xy']/1000
        for ax,values,title in zip(axes,[state['h'],state['h']-z['h']],['Final ice thickness (m)','Thickness change (m)']):
            plot=ax.tripcolor(xy[:,0],xy[:,1],z['tri'],values,shading='gouraud',cmap='viridis' if title.startswith('Final') else 'coolwarm');fig.colorbar(plot,ax=ax);ax.set_title(title);ax.set_aspect('equal');ax.set_xlabel('Easting (km)');ax.set_ylabel('Northing (km)')
        fig.savefig(dest/'thickness.png',dpi=160);plt.close(fig)
    summary={k:v for k,v in out.items() if k!='rows'};summary['config']=cfg
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    ending=f"已推进到{last['year']:.3f}年；体积{last['volume_m3']/1e9:.4f} km³；最大表面速度{last['max_speed_m_year']:.2f}米/年。" if last else '尚无接受状态。'
    status='完成' if out['status']=='completed' else '已停止并保留断点'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 历史气候演化</title><style>body{{background:#eef3f5;color:#243c47;font:17px/1.8 system-ui}}main{{max-width:1120px;margin:30px auto;padding:20px}}img{{max-width:100%;background:white}}pre{{background:white;padding:18px;overflow:auto}}a{{color:#167b8c}}</style><main><h1>1470—1570年逐年气候驱动：{status}</h1><p>{ending}</p><p>50米网格，30天步长上限，GPU BP。计算累计{out['wall_seconds']/60:.2f}分钟；核对了{len(out['annual_CPU_checks'])}个整数年CPU动量残差。</p><img src="evolution.png" alt="冰量、最大流速和质量收支随年份变化"><img src="thickness.png" alt="最终冰厚和冰厚变化"><p>温度采用项目1470—1569年逐年月数据；降水异常关闭，重复月气候平均值。每年按当前冰面高度更新全年质量平衡，积累和受可用冰量限制的消融参与每个时间步。固定冰床和材料参数，尚未启用热力反馈。</p><p>初始状态来自此前50米零质量平衡动力调整的第100年断点，将它指定为本次试验1470年的起点；这不是1470年观测几何，也不是已通过气候平衡或历史重建验收的结果。计算范围保持原网格，允许冰流出，未扩展域外冰缘。</p><h2>运行与查看</h2><pre>python -m polim_cpu.gpu_model status "{root}"
python -m polim_cpu.gpu_model report "{root}"</pre><p>新任务使用 run 配置文件；每个任务独立目录。中断后 resume 原目录读取检查点，已完成任务不会重算。关闭的失败任务需先登记原因和恢复版本。</p><p><a href="summary.json">完整摘要</a> · <a href="../gpu-working-v31r3/index.html">求解器精度交接</a> · <a href="../../GPU_MODEL_USAGE.md">使用说明</a></p><p>{out.get('error') or ''}</p></main></html>'''
    (dest/'index.html').write_text(page,encoding='utf-8');return dest
