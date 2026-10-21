"""Close the registered cadence comparison and retain its working setting."""
from .bp_cadence_6h_v1 import PROJECT,R,E,RUN,REPORT,NAME,codehash,read
from .v3.runner import digest,write_json
from .thermal_attribution_v1 import validate,ref
import copy,json,os,re,hashlib
from datetime import datetime,timezone

def main():
 validate();result=read(RUN/'result.json');t=read(RUN/'terminal-evidence.json');summary=read(REPORT/'summary.json');c=result['comparison']
 assert result['status']==t['status']=='completed' and t['implementation']==codehash()
 for p,h in t['output_hashes'].items():assert digest(RUN/p)==h,p
 for p,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/p)==h,p
 write_json(REPORT/'visual-review.json',dict(status='reviewed',images=['comparison.png'],observations='Legible Chinese labels;H error in millimetres;velocity error sawtooth and6hour refresh ticks agree with all-hours table;wall-time bars match result.'))
 settings=dict(system=2387,bp_update_interval_hours=6 if c['status']=='pass' else 1,transport_maximum_output_interval_hours=1,CFL_fraction=0.35,material='frozen-v16',mass_balance='zero',thermal_update=False,bed='unchanged-v16',adaptive_BP_update=False,verified_duration_hours=24,evidence=ref(RUN/'comparison.json'),next_run_requires_versioned_registration=True)
 write_json(REPORT/'next-run-settings.json',settings)
 # Add a navigation link to the project landing page, preserving its previous form.
 home=R/'index.html';old=home.read_text(encoding='utf-8');(REPORT/'previous-project-index.html').write_text(old,encoding='utf-8')
 link='<a class="current" href="reports/bp-cadence-6h-v1/index.html"><b>新增：BP 更新频率优化</b><p>每 6 小时更新一次流速；查看实际加速与误差对照。</p></a>'
 assert '<h1>2387 冰川模型</h1>' in old;home.write_text(old.replace('<h1>2387 冰川模型</h1>','<h1>2387 冰川模型</h1>'+link,1),encoding='utf-8')
 write_json(REPORT/'navigation-delivery.json',dict(files={p.name:digest(p) for p in REPORT.iterdir() if p.is_file()},home_sha256=digest(home)))
 sp=PROJECT/'LIA_CURRENT_STATE.json';sb=sp.read_bytes();s=json.loads(sb);fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();assert digest(fp)==s['scientific_contract']['sha256'].lower()
 b=s['research_branches'][NAME];assert b['launch_authorized'];num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8'),re.M)))+1;now=datetime.now(timezone.utc).isoformat()
 record=dict(status='completed-six-hour-cadence-'+c['status'],result=ref(RUN/'result.json'),comparison=ref(REPORT/'summary.json'),delivery=ref(REPORT/'navigation-delivery.json'),working_settings=ref(REPORT/'next-run-settings.json'),new_BP_solves=4,new_initial_BP_solves=0,baseline_recomputed=False,scientific_acceptance=False,accepted_count_effect=0,created_at=now)
 cp=E/'completion.json';write_json(cp,record)
 bpwall=sum(v['momentum'].get('wall_seconds',0) for v in result['timeline'][1:]);write_json(REPORT/'timing-breakdown.json',dict(BP_solver_seconds=bpwall,numerical_seconds=result['numerical_wall_seconds'],non_BP_including_transport_checkpoints_and_verification_seconds=result['numerical_wall_seconds']-bpwall,note='Non-BP time not separated into transport,IO and verification;do not attribute it all to disk.'))
 addition=f'''\n\n## {num}. System2387 six-hour BP update accuracy and cost comparison\n\n- User approves reduced velocity-update frequency against saved hourly baseline. One registered24hour experiment from exact v16sealed h,A,u;BP updated at6/12/18/24hours only,held velocity explicitly marked between updates. Hourly transport endpoints and CFL0.35 remain unchanged,isolating momentum cadence. Existing v17baseline25states reused read-only,no baseline solve replay. Fixed bed/material,zero SMB/thermal update,no Full-Stokes,no geometry reconstruction.25sealed outputs verified with nonnegative finiteH,fixedA,exact endtime andglobal/local mass budgets.\n- Four BPsolves versus24;{result['total_substeps']}transport substeps. Numerical wall{result['numerical_wall_seconds']:.6f}s versus archived baseline{c['baseline_wall_seconds']:.6f}s,observed ratio{summary['observed_speedup']:.6f};full new wallincluding extra paired comparison{result['wall_seconds']:.6f}s. BPsolver sum{bpwall:.6f}s,other time includes transport/checkpoints/verification and isnot all attributed toIO. Same25hourly outputs;archived baseline includes its final verification while new numerical timing excludes extra paired comparison and source rehash;different launch times,not controlledrepeat benchmark. Linear one-year extrapolation{summary['linear_year_hours']:.6f}hours,not measured annual run.\n- All25hours compared,including held-velocity times. MaximumHdifference{c['maximum_H_error_m']:.12g}m;maximum surface velocity vector difference{c['maximum_surface_velocity_vector_error_m_year']:.12g}m/year;maximum area-weighted surface velocity relativeL2{c['maximum_velocity_area_relative_L2']:.12g}. Prespecified engineering limits0.01m,0.05m/year,0.001relative: {c['status']}. Final velocity vector maxerror{summary['final_velocity_vector_error_m_year']:.12g}m/year. Globalrelativebudget{result['checks']['relative_mass_budget']:.12g},junction{result['checks']['relative_junction_budget']:.12g}. These are declared time-discretization tolerance choices,not observational confidence intervals.\n- Retain BPupdateinterval{settings['bp_update_interval_hours']}hours as next working setting for this candidate. This is fixed cadence,not adaptive update;single-day agreement doesnotvalidate long-term error or authorize unregistered long runs. No scientificpromotion,productionreplacement oraccepted-countchange. Evidence `{ref(cp)['path']}`.\n'''
 fh=hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper();branch={**b,'status':record['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],'evidence':ref(cp),'next_legal_action':'Use retained cadence setting for a separately registered longer continuation.'}
 route=read(PROJECT/s['authoritative_route']['path']);route['version']+='+polim-v18-cadence6h-completed';route[NAME]=branch
 if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
 rp=E/'route-completed.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=branch;n['scientific_contract']['sha256']=fh;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+='+polim-v18-cadence6h-completed';n['updated_at']=now
 for k in ['counts','current_focus','next_action','population']:assert n[k]==s[k]
 if sp.read_bytes()!=sb or fp.read_bytes()!=fb:raise RuntimeError('Concurrent state/facts update')
 tmp=sp.with_name(f'.LIA_CURRENT_STATE.cadenceclose.{os.getpid()}.tmp');write_json(tmp,n)
 with fp.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
 os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} - Six-hour BP cadence comparison complete\n\nCadence comparison{c["status"]};fourBPsolves,observed{summary["observed_speedup"]:.3f}x speedup,maximumHerror{c["maximum_H_error_m"]:.9f}m andvelocityerror{c["maximum_surface_velocity_vector_error_m_year"]:.9f}m/year. Retain declared working setting;no baseline replay,acceptance or automaticlongrun. Facts{num};`{ref(cp)["path"]}`.\n')
 write_json(E/'completed-validation.json',validate());print(json.dumps(dict(status=record['status'],facts_section=num,BP_seconds=bpwall)))

if __name__=='__main__':main()
