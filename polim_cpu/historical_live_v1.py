"""Read-only live report; never starts or restarts a simulation."""
from pathlib import Path
import json,time,os,sys,html
from datetime import datetime

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def render(root):
    root=Path(root).resolve();project=Path(__file__).resolve().parents[1];cfg=read(root/'config.json');E=project/cfg['evidence_root'];report=project/'polim_cpu/reports'/cfg['case_id'];report.mkdir(exist_ok=True)
    if (E/'completion.json').exists():return False
    status=read(root/'status.json') if (root/'status.json').exists() else {};year=status.get('year',cfg['start_year']);records=[]
    for f in sorted(root.glob('year-*.json')):records.append(read(f))
    # Include annual states in the adopted prefix without recomputing dynamics.
    if cfg.get('adopt_root'):
        for f in sorted((project/cfg['adopt_root']).glob('year-*.json')):
            row=read(f)
            if row['year'] not in [x['year'] for x in records]:records.append(row)
    records.sort(key=lambda x:x['year'])
    tables=''.join(f"<tr><td>{int(r['year'])}</td><td>{r['volume_m3']/1e9:.4f}</td><td>{r['max_speed_m_year']:.2f}</td><td>{r['BP_residual']:.2e}</td></tr>" for r in records[-15:])
    progress=max(0,min(100,(year-cfg['start_year'])/(cfg['end_year']-cfg['start_year'])*100))
    state='运行中' if status.get('stage')=='running' else status.get('status','准备中')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="30"><meta name="viewport" content="width=device-width,initial-scale=1"><title>2387 历史气候运行进度</title><style>body{{background:#edf3f5;color:#243c47;font:17px/1.8 system-ui}}main{{max-width:1050px;margin:30px auto;padding:24px}}section{{background:white;padding:22px;margin:20px 0;border-radius:12px}}progress{{width:100%;height:24px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #dde5e8}}a{{color:#147d90}}</style><main><h1>1470—1570年历史气候模拟</h1><section><b>{state} · 当前{year:.3f}年</b><progress max="100" value="{progress}"></progress><p>已推进{year-cfg['start_year']:.2f}/100年；累计接受{status.get('step',0)}步。页面每30秒刷新。任务完成后自动显示完整报告。</p><p>这是正在计算的进度，尚未完成100年。</p></section><section><h2>已保存年度结果</h2><table><tr><th>年份</th><th>冰体积 km³</th><th>最大表面速度 m/年</th><th>BP残差</th></tr>{tables}</table></section><section><h2>本次设置</h2><p>系统2387，50米网格，30天步长上限；逐年历史温度，项目默认月降水气候平均值；每年按当前冰面更新质量平衡。固定冰床、固定材料参数，无热力反馈。起点为此前零质量平衡动力调整的保存状态，指定为试验1470年，并非观测1470年几何或已验收历史重建。</p><p>每步保存断点；自动释放可重建的辅助矩阵临时文件。质量与动量收敛要求保持不变。按已测早期速度，任务可能需要数小时，随冰川状态变化。</p></section><section><h2>清理与使用</h2><p>已删除275个缓存或无引用临时文件、598个过期辅助矩阵文件，并去重11份完全相同的数据副本，总计释放约21.40 GiB。科学输入、有效分段、恢复断点和失败诊断证据保留。</p><p><a href="../../GPU_MODEL_USAGE.md">运行、查看和断点接续说明</a> · <a href="../gpu-working-v31r3/index.html">GPU求解器交接结果</a></p></section><p>页面更新时间：{datetime.now().isoformat(timespec='seconds')}。{html.escape(str(status.get('error','')))}</p></main></html>'''
    temporary=report/'index.live.tmp';temporary.write_text(body,encoding='utf-8')
    if (E/'completion.json').exists():temporary.unlink();return False
    os.replace(temporary,report/'index.html');return True

if __name__=='__main__':
    root=Path(sys.argv[1]);deadline=time.monotonic()+22000
    while time.monotonic()<deadline:
        if not render(root):break
        if '--follow' not in sys.argv:break
        time.sleep(30)
