"""Read-only source adapter for the frozen LIA geometry population.

Exports geometry only. Does not import a dynamics/MB backend, decode climate
arrays, truncate ice-free domain, prescribe thermal data, or launch a model.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

PROJECT=Path(__file__).resolve().parent.parent
SOURCE=PROJECT/'validation_data/results/paleo_surface_validation_v0/lia_191_gpu_four_group_implementation_v2'
PLAN=PROJECT/'validation_data/results/paleo_surface_validation_v0/lia_gpu_production_integration_v1/plan.frozen.json'
FIELDS=('system_bed_h','system_widths_m','system_thick','system_valid_mask','system_nx','system_dx_meter')
VALIDATOR=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py'


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): value.update(block)
    return value.hexdigest()


def write(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)


def export(out):
    # Project facts must have been read by the operator before this command.
    # The controller check and source fingerprints are the single entry check.
    validation=subprocess.run([sys.executable,str(VALIDATOR)],cwd=PROJECT,capture_output=True,text=True)
    if validation.returncode: raise RuntimeError(validation.stdout+validation.stderr)
    validation_record=json.loads(validation.stdout)
    if validation_record['status']!='pass': raise RuntimeError('Controller validation failed')
    controls={name:sha(PROJECT/name) for name in ('LIA_PROJECT_FACTS.md','LIA_EXECUTION_CONTRACT.md','LIA_CURRENT_STATE.json')}
    state=json.loads((PROJECT/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'))
    if controls['LIA_PROJECT_FACTS.md'].lower()!=state['scientific_contract']['sha256'].lower():
        raise RuntimeError('Scientific contract hash differs from current state')
    plan=json.loads(PLAN.read_text(encoding='utf-8'))
    manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    inputs=plan['inputs']
    for name,path in (('manifest',SOURCE/'manifest.json'),('arrays',SOURCE/'inputs.npz')):
        if (PROJECT/inputs[name]['path']).resolve()!=path.resolve() or sha(path).lower()!=inputs[name]['sha256'].lower():
            raise RuntimeError(f'Frozen source mismatch: {name}')
    if manifest['inputs_sha256'].lower()!=inputs['arrays']['sha256'].lower(): raise RuntimeError('Source manifests disagree')
    systems=manifest['systems']; ids=[int(s['system']) for s in systems]
    if len(ids)!=191 or len(set(ids))!=191: raise RuntimeError('Population must contain 191 unique systems')
    with np.load(SOURCE/'inputs.npz',allow_pickle=False) as archive:
        arrays={key:archive[key] for key in FIELDS}  # Deliberate whitelist: no climate decoding.
    rows=[]
    for i,system in enumerate(systems):
        n=int(arrays['system_nx'][i]); dx=float(arrays['system_dx_meter'][i])
        if n!=int(system['nx']) or dx<=0 or not np.isfinite(dx): raise ValueError('Invalid grid metadata')
        mask=arrays['system_valid_mask'][i]
        if not np.all(mask[:n]) or np.any(mask[n:]): raise ValueError('Unexpected padded-domain mask')
        bed,h,w=[arrays[key][i,:n] for key in ('system_bed_h','system_thick','system_widths_m')]
        if not all(np.isfinite(v).all() for v in (bed,h,w)) or np.any(h<0) or np.any(w<=0):
            raise ValueError(f'Invalid source geometry for {ids[i]}')
        wet=h>0; groups=int(np.count_nonzero(wet & np.r_[True,~wet[:-1]]))
        volume=float(np.sum(h*w)*dx)
        # V2 currently interprets first/last samples as domain endpoints and
        # gives them half cell width. OGGM input samples are cell centres.
        endpoint_difference=float((h[0]*w[0]+h[-1]*w[-1])*dx/2)
        wet_count=int(wet.sum())
        rows.append({'system':ids[i],'source_index':i,'nx':n,'dx_m':dx,
            'ice_nodes':wet_count,'ice_free_nodes':int((~wet).sum()),'ice_segments':groups,
            'source_volume_m3':volume,'v2_endpoint_volume_deficit_m3':endpoint_difference,
            'v2_endpoint_volume_deficit_fraction':endpoint_difference/volume if volume else 0.,
            'v2_full_domain_dense_estimate_gb':(n*9)**2*8*5/1e9,
            'ice_only_dense_estimate_gb':(wet_count*9)**2*8*5/1e9,
            'v2_direct_geometry_compatible':bool(wet.all()),'launch_authorized':False})
    selected=min(rows,key=lambda r:(r['ice_nodes'],r['system']))
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    write(out/'controller-validation.json',validation_record)
    with (out/'geometry-only.npz').open('xb') as stream: np.savez_compressed(stream,**arrays)
    with (out/'inventory.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(out/'inventory.json',rows)
    sample=out/f"system-{selected['system']}"; sample.mkdir()
    i=selected['source_index']; n=selected['nx']; dx=selected['dx_m']
    geometry=np.column_stack((np.arange(n)*dx,arrays['system_bed_h'][i,:n],arrays['system_thick'][i,:n],arrays['system_widths_m'][i,:n]))
    with (sample/'geometry.csv').open('x',encoding='utf-8',newline='') as stream:
        np.savetxt(stream,geometry,delimiter=',',header='x_m,bed_m,thickness_m,width_m',comments='',fmt='%.17g')
    specification={'schema':'polim-lia-input-package-v1','status':'geometry-exported-model-integration-required',
        'system':selected['system'],'selection':'fewest initially ice-covered nodes; system ID tie-break; technical sizing only, not a scientific representative',
        'geometry_file':'geometry.csv','source_row':i,'full_domain_retained':True,'ice_free_domain_retained':True,
        'coordinate_convention':'x_m = bin index * dx; original samples are cell centres, first centre at relative x=0',
        'source_cell_width_m':dx,'source_cell_faces':'x_m +/- dx/2',
        'geometry_type':'frozen single rectangular pseudoflowline; not original multi-tributary topology',
        'thermal_initial_state':None,'surface_temperature_boundary':None,'surface_mass_balance':None,
        'launch_authorized':False,'scientific_acceptance':False,
        'requires':['ice-free mask and moving margin with no thickness floor','source-compatible cell-centre volume and fluxes',
                    'CUDA solve restricted to ice-supported DOFs without deleting the physical domain',
                    'production spatial SMB adapter and an explicit thermal/initialization contract',
                    'new research-route transaction before any LIA dynamics launch']}
    write(sample/'input-package.json',specification)
    # One final data round trip, source-preservation check and package seal.
    with np.load(out/'geometry-only.npz',allow_pickle=False) as restored:
        roundtrip={key:restored[key].dtype==arrays[key].dtype and restored[key].shape==arrays[key].shape and restored[key].tobytes()==arrays[key].tobytes() for key in FIELDS}
    csv_restored=np.genfromtxt(sample/'geometry.csv',delimiter=',',skip_header=1)
    csv_equal=csv_restored.tobytes()==geometry.tobytes()
    source_unchanged=sha(SOURCE/'inputs.npz').lower()==inputs['arrays']['sha256'].lower()
    controls_unchanged=all(sha(PROJECT/name)==value for name,value in controls.items())
    if not all(roundtrip.values()) or not csv_equal or not source_unchanged or not controls_unchanged:
        raise RuntimeError('Export preservation check failed')
    summary={'schema':'polim-lia-integration-readiness-v1','status':'input-adapter-complete; dynamics-not-run',
        'population':191,'direct_v2_compatible_systems':sum(r['v2_direct_geometry_compatible'] for r in rows),
        'systems_with_ice_free_domain':sum(r['ice_free_nodes']>0 for r in rows),
        'full_nodes_range':[min(r['nx'] for r in rows),max(r['nx'] for r in rows)],
        'ice_nodes_range':[min(r['ice_nodes'] for r in rows),max(r['ice_nodes'] for r in rows)],
        'full_dense_gb_range':[min(r['v2_full_domain_dense_estimate_gb'] for r in rows),max(r['v2_full_domain_dense_estimate_gb'] for r in rows)],
        'selected_technical_input':selected,'source_plan':{'path':str(PLAN),'sha256':sha(PLAN)},
        'source':inputs,'decoded_npz_fields':list(FIELDS),'climate_arrays_decoded':False,
        'roundtrip_bitwise':roundtrip,'csv_roundtrip_bitwise':csv_equal,'source_unchanged':source_unchanged,
        'control_files_unchanged':controls_unchanged,'control_hashes':controls,'launch_authorized':False,
        'accepted_count_effect':0,'dynamics_performed':False,'next_implementation':specification['requires']}
    write(out/'readiness.json',summary)
    document=f'''# 接入本项目的下一步

已完成 191 条冻结矩形拟流线的几何输入转换，原始无冰区、宽度、冰厚、有效域和网格间距全部保留。全部数组及示例 CSV 往返逐位一致。未解码气候数组，未进行动力模拟。

当前第二版不能直接用于这些输入：191 条都含无冰区，而第二版要求处处有冰；完整网格为 {summary['full_nodes_range'][0]}–{summary['full_nodes_range'][1]} 个水平点。按当前 9 层稠密 GPU 解法估算需要 {summary['full_dense_gb_range'][0]:.1f}–{summary['full_dense_gb_range'][1]:.1f} GB，不能直接套用演示方法。

下一版先完成冰缘推进/退缩与活动计算域：完整物理域保持不变，只把有冰位置的未知量送入动量求解；质量输运覆盖全部原始网格。同步统一单元中心的体积定义，再连接随高度变化的表面质量平衡和明确的热力输入。

已准备系统 {selected['system']} 的完整 CSV，供技术接入使用（有冰 {selected['ice_nodes']} 格、无冰 {selected['ice_free_nodes']} 格）。它是按计算规模选择的输入，不是已经通过科学代表性选择的对象。没有把演示用的恒定温度或零 SMB 填入 LIA 运行配置。

`geometry-only.npz` 保存六个原始几何数组；`inventory.csv` 列出全部系统；`system-{selected['system']}/input-package.json` 记录坐标约定和仍需实现的接口。该文件不是可启动仿真的配置。

现有 `LIA_CURRENT_STATE.json` 的新运行授权为 false；启动 LIA 数值试验需要按 `LIA_EXECUTION_CONTRACT.md` 第 4、7 节登记新的模型分支与版本化事务。本轮只完成数据接入准备，主线和已有结果保持不变。
'''
    (out/'NEXT.md').write_text(document,encoding='utf-8')
    render(out,summary,rows)
    write(out/'commit.json',{'files':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},'adapter_sha256':sha(__file__)})
    return {k:summary[k] for k in ('status','population','direct_v2_compatible_systems','selected_technical_input','dynamics_performed')}


def render(out,summary,rows):
    table=''.join(f"<tr><td>{r['system']}</td><td>{r['nx']}</td><td>{r['ice_nodes']}</td><td>{r['ice_free_nodes']}</td><td>{r['v2_full_domain_dense_estimate_gb']:.1f}</td></tr>" for r in sorted(rows,key=lambda r:r['system']))
    selected=summary['selected_technical_input']['system']
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LIA 项目接入</title>
<style>body{{max-width:1040px;margin:40px auto;padding:0 24px;font:16px/1.7 system-ui;color:#17233b;background:#f4f7fb}}article{{background:white;padding:24px;border-radius:14px;margin:20px 0}}h1{{margin-bottom:8px}}a{{color:#166fa3}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #dae1e9;padding:7px}}details{{margin-top:20px}}.status{{color:#087866;font-weight:700}}</style>
<h1>从示例走向 LIA 项目</h1><p class="status">191 条冻结剖面已导出 · 几何逐位保留 · 未启动动力模拟</p>
<article><h2>下一步：冰缘和真实网格</h2><p>所有剖面都含无冰区。第二版还需要让冰缘能够推进、退缩，并只在有冰位置求解动量，同时保留完整的物理域和输运网格。</p><p>全域直接套用当前 GPU 稠密解法约需 {summary['full_dense_gb_range'][0]:.1f}–{summary['full_dense_gb_range'][1]:.1f} GB。应先完成活动计算域，再连接项目的表面质量平衡和热力输入。</p></article>
<article><h2>准备好的输入</h2><p><a href="system-{selected}/geometry.csv">系统 {selected} 完整剖面 CSV</a> · <a href="inventory.csv">191 条剖面清单</a> · <a href="readiness.json">数据与校验记录</a> · <a href="NEXT.md">具体接入说明</a></p><p>示例按有冰网格数量选择，只用于技术接入。原始气候数组未解码，未填入假定的 LIA 温度或零质量平衡。</p></article>
<details><summary>查看全部剖面规模</summary><table><thead><tr><th>系统</th><th>全域格数</th><th>有冰格数</th><th>无冰格数</th><th>9 层全域稠密估算 GB</th></tr></thead><tbody>{table}</tbody></table></details>
<p>本页是输入适配结果。当前执行状态未授权新的 LIA 模拟；现有主线、事实和验收数量均保持不变。</p></html>'''
    (out/'index.html').write_text(content,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(export(args.out),ensure_ascii=False,indent=2))
