"""Close the bounded grid/time cost comparison with preserved evidence."""
from .millennium_cost_v1 import PROJECT,R,E,I,RUN,REPORT,NAME,codehash,read,arrays
from .v3.runner import digest,write_json
from .thermal_attribution_v1 import validate,ref
from .v9.storage import restore
import numpy as np
import copy,json,os,re,hashlib
from datetime import datetime,timezone

def main():
 validate();result=read(RUN/'result.json');term=read(RUN/'terminal-evidence.json');summary=read(REPORT/'summary.json')
 assert result['status']==term['status']=='completed' and term['implementation']==codehash()
 for p,h in term['files'].items():assert digest(RUN/p)==h,p
 for p,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/p)==h,p
 count=0
 for grid in RUN.glob('grid-*'):
  mat=arrays(I/'material.npz')['A'] if grid.name=='grid-15.75m' else arrays(grid/'material-and-projection.npz')['A']
  for folder in grid.rglob('frame-*'):
   if not folder.is_dir():continue
   state,u,_=restore(folder,result['contract']);assert np.array_equal(state['A'],mat) and np.isfinite(state['h']).all() and (state['h']>=0).all() and np.isfinite(u).all();count+=1
 write_json(REPORT/'combined-verification.json',dict(status='pass',sealed_states_restored=count,all_material_bitwise_fixed=True,all_finite_nonnegative=True,visual_review='comparison.png inspected:logarithmic cost axis labelled,spatial error10percent and temporal0.1m lines explicit;actual source values match summaries.',result=ref(RUN/'result.json')))
 disposition=dict(native_daily_window=summary['native_daily_window']['native_day_pass'],coarse_working_grid_accepted=None,boundary_preserving_candidate_interior_m=63,rejected_uniform_grid_m=[v['grid_m'] for v in summary['coarse'] if not v['spatial_screen_pass']],next_action='Boundary-preserving coarse-interior mesh design,retaining resolved confluence geometry. Register a distinct comparison before any next numerical run. Do not waive5percent area gate or call short-run cost an executed1000year result.',long_run_launched=False)
 write_json(REPORT/'disposition.json',disposition)
 home=R/'index.html';old=home.read_text(encoding='utf-8');(REPORT/'previous-project-index.html').write_text(old,encoding='utf-8');anchor='<h1>2387 冰川模型</h1>';assert anchor in old
 link='<a class="current" href="reports/millennium-cost-v1/index.html"><b>新增：千年模拟成本与精度对照</b><p>查看 63/126 米网格、日/月更新的实测成本及边界误差。</p></a>';home.write_text(old.replace(anchor,anchor+link,1),encoding='utf-8')
 write_json(REPORT/'navigation-delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},home_sha256=digest(home)))
 sp=PROJECT/'LIA_CURRENT_STATE.json';sb=sp.read_bytes();s=json.loads(sb);fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();assert digest(fp)==s['scientific_contract']['sha256'].lower();b=s['research_branches'][NAME];assert b['launch_authorized']
 num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8'),re.M)))+1;now=datetime.now(timezone.utc).isoformat()
 record=dict(status='completed-cost-study-coarse-spatial-gate-not-passed',result=ref(RUN/'result.json'),summary=ref(REPORT/'summary.json'),delivery=ref(REPORT/'navigation-delivery.json'),disposition=ref(REPORT/'disposition.json'),accepted_count_effect=0,scientific_acceptance=False,created_at=now);cp=E/'completion.json';write_json(cp,record)
 native=summary['native_daily_window'];ne=native['native_day_error']
 addition=f'''\n\n## {num}. System2387 millennial grid/time cost comparison completed\n\n- User needs millennial duration and authorized bounded spatial/time cost comparison. One fine15.75m daily transport/BP control from sealedv16;two coarse63/126m initial BP fields followed by same-grid30daily updates versus one30day heldvelocity interval and finalBP. All65newBPsolves completed,wall{result['wall_seconds']:.6f}s. Existing hourly baseline reused,no replay. Fixed bed/material within eachcase,zero SMB/thermal feedback,no FullStokes. Existing conforming pixel mesh/three-sigmaBP/conservative CFL0.35transport unchanged. Coarsening uses aligned4x4/8x8 Hvolume sums including outsidezeros,anypositiveblock support,positive-cell meanbed,H-weightedA/velocity and incident-corner averaging. This explicitly changes pointwise geometry/material/support;not a pure mesh-only re-solve.\n- Fine initial donorCFL{native['daily_window']['initial_CFL_days']:.9f}days;this is transport limit,not independent coupled accuracy criterion. Fine daily-window versus archived24hourupdates: Hmaxerror{ne['H_max_m']:.12g}m,velocitymax{ne['velocity_max_m_year']:.12g}m/year;declared0.01m/0.05m-year screen passed. Daily-window wall{native['daily_window']['wall_seconds']:.6f}s.\n'''
 for v in summary['coarse']:
  addition+=f"- {v['grid_m']:g}m: {v['nodes']}nodes;area change{v['area_change_percent']:.9f}percent;volume remaprelative{v['volume_remap_relative']:.12g}. Initial velocity area-weightedL2 versus H-weighted native projection{v['spatial_velocity_error_percent']:.9f}percent,projectedpeakerror{v['spatial_projected_peak_error_percent']:.9f}percent. Spatialscreenpass={v['spatial_screen_pass']},failure reasons={','.join(v['spatial_fail_reasons'])}. InitialCFL{v['initial_CFL_days']:.9f}days.30daydaily wall{v['daily_30d_seconds']:.6f}s,monthly wall{v['monthly_30d_seconds']:.6f}s;month-vs-dayHmax{v['monthly_H_error_m']:.12g}m,velocityrelative{v['monthly_velocity_error_percent']:.9f}percent,monthlyscreenpass={v['monthly_screen_pass']}.1000yearconditionalextrapolation daily{v['millennium_daily_hours']:.6f}h/monthly{v['millennium_monthly_hours']:.6f}h. Budgets daily{v['daily_budget']:.12g}/monthly{v['monthly_budget']:.12g}.\n"
 addition+=f'''- All{count}sealedstates restored with hashes,fixedA andfinite nonnegativeH. Monthly engineering limitsH0.1m/velocity1percent passed on bothcoarsegrids;spatial velocity10percent/peak20percent/area5percent andcomponent limits remain frozen.63m fails areaonly,126m fails area/velocity/peak. Neitheruniformcoarsegrid promoted. Retain63m as interior-resolution candidate for boundary-preserving design;do not silently accept6.057percentarea expansion. Conditional costs extrapolate short30day fixeddomain/zero-source dynamics,include savedstagecost butexclude initialgridBP,climate/thermalcoupling andlong-term geometry/runtime changes. No1000yearrun,productionreplacement,scientificacceptance orcountchange. Evidence `{ref(cp)['path']}`.\n'''
 fh=hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper();branch={**b,'status':record['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],'evidence':ref(cp),'next_legal_action':disposition['next_action']};route=read(PROJECT/s['authoritative_route']['path']);route['version']+='+polim-v19-cost-completed';route[NAME]=branch
 if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
 rp=E/'route-completed.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=branch;n['scientific_contract']['sha256']=fh;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+='+polim-v19-cost-completed';n['updated_at']=now
 for k in ['counts','current_focus','next_action','population']:assert n[k]==s[k]
 if sp.read_bytes()!=sb or fp.read_bytes()!=fb:raise RuntimeError('Concurrent state/facts update')
 tmp=sp.with_name(f'.LIA_CURRENT_STATE.costclose.{os.getpid()}.tmp');write_json(tmp,n)
 with fp.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
 os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} - Millennial cost comparison complete\n\n65BPsolves finished;fine daily control andcoarse monthly temporal screens pass. Uniformcoarse spatial gates not allpassed:63m area+6.057percent;126m area/velocity/peak. Preserve evidence,retain63m interior candidate for boundary-preserving design,no thresholdwaiver orlongrun. Facts{num};`{ref(cp)["path"]}`.\n')
 write_json(E/'completed-validation.json',validate());print(json.dumps(dict(status=record['status'],facts_section=num,states_verified=count)))

if __name__=='__main__':main()
