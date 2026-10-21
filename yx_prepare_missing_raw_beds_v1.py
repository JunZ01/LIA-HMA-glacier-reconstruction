from pathlib import Path
import sys,json,argparse,subprocess
ROOT=Path('H:/ToYangxin/result_yx/surface_reconstruction_20261009_v5');INPUT=ROOT.parent/'surface_reconstruction_20261009_v4'
sys.path.insert(0,str(ROOT))
import raw_product_static_worker as base
import pandas as pd,geopandas as gpd
def prepare(uid):
 project=base.PROJECT
 (project/'LIA_PROJECT_FACTS.md').read_text(encoding='utf-8');(project/'LIA_EXECUTION_CONTRACT.md').read_text(encoding='utf-8')
 state=json.loads((project/'LIA_CURRENT_STATE.json').read_text(encoding='utf-8'));b=state['research_branches']['yx_multioutlet_modern_static_v06']
 check=subprocess.run([sys.executable,str(base.REPO/'ice_surface_reconstruction/drivers/validate_lia_execution_state_v1.py')],capture_output=True)
 assert check.returncode==0 and b['launch_authorized'] and uid in b['authorized_uid'] and state['current_focus']['state']!='decision-required'
 row=pd.read_csv(INPUT/'system_manifest.csv');row=row.loc[row.lia_unique_id.eq(uid)].iloc[0];work=ROOT/'members'/row.record_id;work.mkdir(parents=True,exist_ok=True)
 fingerprint=base.digest(ROOT/'reconstruction_config.json')+':'+base.digest(INPUT/'system_manifest.csv')+':'+str(uid)
 cpfile=work/'checkpoint.json';cp=json.loads(cpfile.read_text(encoding='utf-8')) if cpfile.exists() else {'fingerprint':fingerprint,'completed_stages':[]}
 assert cp['fingerprint']==fingerprint
 if 'raw-product-bed' in cp['completed_stages']:return
 assert not (work/'result.json').exists()
 selected=gpd.read_file(INPUT/'modern_rgi7_members.gpkg');selected=selected.loc[selected.lia_unique_id.eq(uid)].copy();config=json.loads((ROOT/'reconstruction_config.json').read_text(encoding='utf-8'))
 if 'source-products' not in cp['completed_stages']:
  base.product_targets(selected,work);cp['completed_stages'].append('source-products');base.atom(cpfile,cp)
 targets=pd.read_csv(work/'product_targets.csv')
 sc=json.loads((base.REPO/'config/stage0a_bed_full_farinotti_v0.json').read_text(encoding='utf-8'));sc.update(oggm_workdir_path=str(ROOT/'modern_bed_workdir'),use_multiprocessing=False,mp_processes=1,surface_dem_path=config['surface_dem_path'])
 base.configure(sc,work)
 existing=all((ROOT/'modern_bed_workdir/per_glacier'/rid[:-6]/rid[:-3]/rid/'glacier_grid.json').exists() for rid in selected.rgi_id)
 gdirs=base.workflow.init_glacier_directories(selected.rgi_id.tolist() if existing else selected,reset=False,force=False)
 if 'modern-grid-geometry' not in cp['completed_stages']:
  for task,kwargs in [(base.tasks.define_glacier_region,{'source':'USER'}),(base.tasks.glacier_masks,{})]:
   for g in gdirs:task(g,continue_on_error=False,**kwargs)
  cp['completed_stages'].append('modern-grid-geometry');base.atom(cpfile,cp)
 base.raw_beds(selected,gdirs,targets,work);cp['completed_stages'].append('raw-product-bed');base.atom(cpfile,cp)
 base.atom(work/'source_preparation_only_v06.json',{'uid':uid,'no_old_endpoint_evaluation':True,'completed_stages':cp['completed_stages']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--uid',type=int,required=True);prepare(p.parse_args().uid)
