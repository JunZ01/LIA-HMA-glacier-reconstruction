"""Read completed BP benchmark evidence, publish figures and close the branch."""
import copy,json,os,re,sys,shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from .v3.runner import PROJECT,digest,write_json
from .thermal_attribution_v1 import validate,ref
from .v8.registration import NAME,EVIDENCE,implementation

ROOT=PROJECT/'polim_cpu';RUN=ROOT/'runs/v8-bp-boundary-benchmarks-v1';REPORT=ROOT/'reports/bp-boundary-benchmarks-v1'
REFERENCE=ROOT/'reference/ismip-hom-2008'


def render():
    validate();d=json.loads((RUN/'terminal-evidence.json').read_text(encoding='utf-8'))
    REPORT.mkdir(parents=True,exist_ok=False)
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,ax=plt.subplots(2,2,figsize=(13.8,9),constrained_layout=True);colors={3:'#e69f00',9:'#0072b2'}
    for target,name,title,refname in [(ax[0,0],'hom-a020','起伏基岩：ISMIP-HOM A，20 km','rhi2a020.txt'),(ax[0,1],'hom-e000','Arolla 剖面：ISMIP-HOM E1，无滑动','fpa1e000.txt')]:
        r=np.loadtxt(REFERENCE/refname);target.plot(r[:,0]/1000,r[:,1],color='#222222',lw=2,label='公开 BP 参考结果')
        for nz in (3,9):
            p=np.loadtxt(RUN/f'{name}-nz{nz}/surface-profile.txt');target.plot(p[:,0]/1000,p[:,1],color=colors[nz],lw=1.8,ls='--' if nz==3 else '-',label=f'{nz} 个垂向节点')
        target.set(title=title,xlabel='沿程距离（km）',ylabel='表面水平速度（m/年）');target.legend(fontsize=9);target.grid(alpha=.2)
    target=ax[1,0];exact=next(r['analytic_surface_m_year'] for r in d['results'] if r['case']=='slab')
    z=np.linspace(0,1,201);target.plot(exact*(1-(1-z)**4),z,color='#222222',lw=2,label='BP 解析解')
    for nz in (3,9):
        with np.load(RUN/f'slab-nz{nz}/fields.npz',allow_pickle=False) as f:u=f['velocity'][0,:,0]
        target.plot(u,np.linspace(0,1,nz),'o-',color=colors[nz],label=f'{nz} 个垂向节点',ms=4)
    target.set(title='规则冰层：垂向速度剖面',xlabel='水平速度（m/年）',ylabel='相对冰厚高度 σ');target.legend(fontsize=9);target.grid(alpha=.2)
    target=ax[1,1];names=['slab','hom-a020','hom-e000'];labels=['规则冰层\n表面误差','起伏基岩\n曲线误差','Arolla\n曲线误差'];x=np.arange(3)
    for nz,offset in ((3,-.19),(9,.19)):
        values=[100*next(r.get('surface_relative_error',r.get('profile_relative_L2_error')) for r in d['results'] if r['case']==n and r['vertical_nodes']==nz) for n in names]
        bars=target.bar(x+offset,values,.36,color=colors[nz],label=f'{nz} 个垂向节点');target.bar_label(bars,fmt='%.2f%%',fontsize=9,padding=3)
    target.set(xticks=x,xticklabels=labels,ylabel='相对误差（%）',title='本轮比较：9 节点误差均降低',ylim=(0,21));target.legend(fontsize=9);target.grid(axis='y',alpha=.2)
    fig.suptitle('BP 边界与垂向分辨率：一轮合并检查',fontsize=18,fontweight='bold')
    fig.savefig(REPORT/'benchmark-comparison.png',dpi=160);plt.close(fig)
    rows=[]
    for name,label in [('slab','规则冰层（解析解）'),('hom-a020','起伏基岩（RHI2）'),('hom-e000','Arolla（FPA1）')]:
        pair=[next(r for r in d['results'] if r['case']==name and r['vertical_nodes']==nz) for nz in (3,9)]
        error=[r.get('surface_relative_error',r.get('profile_relative_L2_error')) for r in pair]
        rows.append(f'| {label} | {100*error[0]:.2f}% | {100*error[1]:.2f}% |')
    md='''# BP 边界已实现，垂向加密明显降低算例误差

已完成一轮合并检查：6 个固定几何 BP 求解全部收敛，5 组边界检查通过，独立进程从已封存断点恢复后与原结果逐位相同。整轮计算约 26.35 秒，不含数据下载和报告整理。

| 对照算例 | 3 个垂向节点 | 9 个垂向节点 |
|---|---:|---:|
'''+ '\n'.join(rows)+'''

![BP 标准算例对比](benchmark-comparison.png)

## 本次解决了什么

边界现在必须逐段明确标记。给定速度、侧壁法向不穿透使用严格约束；侧壁可附加线性切向摩擦。末端可施加明确的冰压减水压载荷，并在水位处分段积分。周期边界共享对应节点的速度。底部保持无滑动，零冰厚端点固定为零速度。

五组检查覆盖干末端、部分浸水末端、给定载荷、入口速度与侧壁摩擦功、周期连接；与独立积分或规定约束的误差均低于 9×10⁻¹⁶。实际运行的标准算例主要检验周期连接、底部与零厚度端点；末端压力和侧壁摩擦目前通过载荷/约束检查，尚未完成独立的非线性受载前缘算例。

## 结果如何解读

规则冰层表面速度解析值为 23.6272 m/年。3 节点误差 12.50%，9 节点误差 0.78%。起伏基岩与 Arolla 的 9 节点曲线相对 L2 误差分别为 1.03% 和 7.64%，峰值差分别为 0.91% 和 9.62%。所有非线性相对残差均小于预定的 10⁻⁷。

本轮事先冻结的工程对照门槛为：9 节点规则冰层误差低于 2%，两个公开 BP 对照的曲线误差低于 10%、峰值差低于 15%。全部通过。这些是本项目的初步工程门槛，不是文献规定的验收标准，也不等于真实冰川验证。只有两个垂向分辨率，尚不足以证明网格收敛；Arolla 仍有 7.64% 的差异，横向网格、源几何插值和参考离散方法的影响尚未分离。

## 与真实冰川的关系

本轮没有重新计算 2387 的速度。此前约 137244 m/年的异常高速仍未解决。算例中加密后速度总体提高，因此不能把“增加垂向节点”当作压低异常高速的修复办法。

下一步需要依据真实冰川边缘、分水岭、截断位置和外部介质，给实际边界分区；汇流处继续在同一网格内共享速度并传递应力，不设置人为支流壁。还需改为分块装配：当前 v8 的一次装配上限为 100 万棱柱，真实原生网格加至 9 个垂向节点约有 282 万棱柱，不能直接套用当前小算例装配。真实几何与参数的物理适用性仍需处理。

## 对照来源与定义

参数采用最终论文：A=10⁻¹⁶ Pa⁻³·年⁻¹，ρ=910 kg/m³，g=9.81 m/s²，n=3。A 算例平均冰厚 1000 m、振幅 500 m、长度 20 km、坡度 0.5°，水平 40×40 网格；E1 使用公开 100 m Arolla 剖面，线性插值至 50 m，横向周期延拓宽度 100 m 仅用于表示平面应变。[Pattyn 等，2008，表 1、表 3 及算例定义](https://tc.copernicus.org/articles/2/95/2008/tc-2-95-2008.pdf)

选取论文表 3 中的 BP 类 LMLa 结果：RHI2（谱方法）与 FPA1（有限差分）。A 的参考水平速度分量插值到 y/L=1/4 后取模；E 使用表面沿程水平速度。附录的 E000 对应无滑动 E1，E001 对应零摩擦区实验。原始数据与转换后的曲线均保存哈希。[官方数据镜像](https://frank.pattyn.web.ulb.be/ismip/welcome.html)、[文件格式说明，第 25–27 页](https://frank.pattyn.web.ulb.be/ismip/ismiphom.pdf)

旧网页的 A 平均冰厚与最终论文不同；本轮按最终论文 1000 m 冻结。旧说明中的 Arolla 床面/表面列文字次序也与实际文件相反，本轮使用实际文件的 x、床面、表面、标志，确保非负冰厚。数据文件 E 的距离列已经是米，未再乘长度。

末端采用有效 BP 载荷 p_ice−p_water，水位处单独积分；侧壁不穿透与切向摩擦分开表达。[icepack 方法论文，2.3 节与 4.5.2 节](https://gmd.copernicus.org/articles/14/4593/2021/)

模型范围保持 BP。未执行外部模型代码，未推进冰川时间，未读取气候，主线验收仍为 58/191。
'''
    (REPORT/'REPORT.md').write_text(md,encoding='utf-8')
    import markdown
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BP 边界与算例结果</title><style>body{margin:0;background:#f3f6f8;color:#20313e;font-family:"Microsoft YaHei",sans-serif;line-height:1.8}main{max-width:1120px;margin:32px auto;padding:36px 44px;background:white;border-radius:16px}h1{font-size:30px;line-height:1.5}h2{margin-top:38px;font-size:21px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{text-align:left;padding:12px 18px;border-bottom:1px solid #dce4ea}th{background:#eaf2f7}a{color:#0072b2}p{margin:18px 0}@media(max-width:700px){main{margin:0;padding:20px}h1{font-size:24px}}</style><main>'''+markdown.markdown(md,extensions=['tables'])+'</main></html>'
    (REPORT/'index.html').write_text(html,encoding='utf-8')
    # Root is a link page so report-relative scientific figures remain valid.
    (ROOT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=reports/bp-boundary-benchmarks-v1/index.html"><title>BP 算例结果</title><a href="reports/bp-boundary-benchmarks-v1/index.html">查看 BP 边界与算例结果</a></html>',encoding='utf-8')
    write_json(REPORT/'report-evidence.json',dict(source=ref(RUN/'terminal-evidence.json'),report=ref(REPORT/'REPORT.md'),figure=ref(REPORT/'benchmark-comparison.png'),html=ref(REPORT/'index.html'),driver=ref(Path(__file__)),status=d['status']))
    print(str(REPORT/'index.html'))


def close():
    validate();statepath=PROJECT/'LIA_CURRENT_STATE.json';before=statepath.read_bytes();state=json.loads(before)
    facts=PROJECT/'LIA_PROJECT_FACTS.md';oldfacts=facts.read_bytes()
    if digest(facts)!=state['scientific_contract']['sha256'].lower():raise ValueError('Facts changed')
    d=json.loads((RUN/'terminal-evidence.json').read_text(encoding='utf-8'))
    if d['implementation']!=implementation():raise ValueError('Frozen code changed')
    for name,sha in d['output_hashes'].items():
        if digest(RUN/name)!=sha:raise ValueError(f'Output changed: {name}')
    branch=state['research_branches'][NAME]
    if not branch['launch_authorized']:raise ValueError('Branch already closed')
    now=datetime.now(timezone.utc).isoformat()
    evidence=dict(status=d['status'],numerical_evidence=ref(RUN/'terminal-evidence.json'),report_evidence=ref(REPORT/'report-evidence.json'),
        plan=branch['plan'],all_recorded_output_hashes_verified=True,accepted_count_effect=0,maximum_model='Blatter-Pattyn',created_at=now)
    write_json(EVIDENCE/'completion.json',evidence)
    text=oldfacts.decode('utf-8');number=max(map(int,re.findall(r'^## (\d+)\.',text,re.M)))+1
    addition=f'''\n\n## {number}. PoLIM BP boundary and published benchmark development completed\n\n- User-authorized BP-only combined suite completed six fixed-geometry solves (slab,ISMIP-HOM A20km,E1 no-slip;3 and9 vertical nodes),five analytic boundary constraint/load checks and an independent-process sealed checkpoint continuation. All six nonlinear residuals below1e-7; restart bitwise identical. Standard-geometry computation26.35135seconds. No real2387 solve,climate read,time evolution,Full-Stokes run or acceptance change.\n- Slab surface error12.5000percent at3 nodes and0.78125percent at9. Published LMLa BP references: RHI2 A profile L2 error9.7681 to1.0307percent; FPA1 E1 profile error17.2776 to7.6382percent. Nine-node peak differences0.9103percent and9.6228percent. Frozen engineering thresholds(2percent slab;10percent profile and15percent peak for external references)passed; not a formal real-glacier scientific gate or proof of mesh convergence. Six physical and numerical parameters remained frozen; no fitting.\n- Explicit boundary API supports velocity,impermeable walls,optional linear tangential friction,ice-minus-water effective front traction with waterline split,and periodic equality; basal no-slip retained. Front/wall features have load/constraint checks; independent nonlinear loaded-front validation remains outstanding. Actual2387 boundaries remain unclassified and prior137244m/year anomaly unresolved. Native nine-node solve needs block assembly beyond current1million-prism budget; not authorized by this completed suite.\n- Reference identity: final2008 paper A meanH1000m,not legacy webpage1600m; official E000 file is no-slip E1. Original Arolla numeric columns are x,bed,surface,flag despite reversed labels in old protocol prose. Reference archive/raw/conversion hashes recorded. No real source geometry altered. Evidence: `{ref(EVIDENCE/'completion.json')['path']}`.\n'''
    newfacts=oldfacts+addition.encode('utf-8');factsha=__import__('hashlib').sha256(newfacts).hexdigest().upper()
    record=dict(stage='terminal',status=d['status'],plan=branch['plan'],terminal_evidence=ref(EVIDENCE/'completion.json'),source_route=state['authoritative_route'],created_at=now,accepted_count_effect=0)
    write_json(EVIDENCE/'terminal-registration.json',record)
    newbranch={**branch,'status':'terminal-engineering-benchmarks-passed','launch_authorized':False,'allowed_stages':[],
        'evidence':ref(EVIDENCE/'terminal-registration.json'),'next_legal_action':'Classify actual glacier boundaries and design block assembly under a new scoped plan; no new numerical launch'}
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-v8-bp-boundary-terminal';route[NAME]=newbranch
    if 'scientific_contract' in route:route['scientific_contract']={**state['scientific_contract'],'sha256':factsha}
    target=EVIDENCE/'route-terminal.json';write_json(target,route)
    updated=copy.deepcopy(state);updated['research_branches'][NAME]=newbranch;updated['scientific_contract']['sha256']=factsha
    updated['authoritative_route']={**ref(target),'version':route['version'],'supersedes':state['authoritative_route']['version']}
    updated['state_id']+='+polim-v8-bp-boundary-terminal';updated['updated_at']=now
    for key in ('counts','population','current_focus','next_action'):
        if updated[key]!=state[key]:raise ValueError('Mainline changed')
    if statepath.read_bytes()!=before or facts.read_bytes()!=oldfacts:raise ValueError('Concurrent state/facts change')
    temp=statepath.with_name(f'.LIA_CURRENT_STATE.bpboundaryclose.{os.getpid()}.tmp');write_json(temp,updated);os.replace(temp,statepath)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — BP v8 boundary suite terminal\n\nSix solves,five boundary checks and exact independent restart complete. Frozen BP engineering reference thresholds passed; no real-glacier acceptance or high-speed resolution claimed. Facts section{number}. Evidence `{ref(EVIDENCE/"completion.json")["path"]}`. Preserve all concurrent branches and mainline58/191.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(EVIDENCE/'terminal-controller-validation.json',validate())
    print(json.dumps(dict(status=d['status'],facts_section=number,accepted=58)))


if __name__=='__main__':
    if sys.argv[1]=='render':render()
    elif sys.argv[1]=='close':close()
