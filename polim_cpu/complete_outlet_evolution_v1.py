"""Record the completed day-one experiment without changing other branches."""
from .outlet_evolution_v1 import PROJECT,E,RUN,REPORT,NAME,codehash,read
from .v3.runner import digest,write_json
from .thermal_attribution_v1 import validate,ref
import copy,json,os,re,hashlib
from datetime import datetime,timezone

def main():
 validate();result=read(RUN/'result.json');t=read(RUN/'terminal-evidence.json');summary=read(REPORT/'summary.json')
 assert result['status']==t['status']=='completed' and t['implementation']==codehash()
 for p,h in t['output_hashes'].items():assert digest(RUN/p)==h,p
 for p,h in read(REPORT/'delivery.json')['files'].items():assert digest(REPORT/p)==h,p
 assert (REPORT/'visual-review.json').exists()
 sp=PROJECT/'LIA_CURRENT_STATE.json';sb=sp.read_bytes();s=json.loads(sb);fp=PROJECT/'LIA_PROJECT_FACTS.md';fb=fp.read_bytes();assert digest(fp)==s['scientific_contract']['sha256'].lower()
 b=s['research_branches'][NAME];assert b['launch_authorized'];num=max(map(int,re.findall(r'^## (\d+)\.',fb.decode('utf-8'),re.M)))+1;now=datetime.now(timezone.utc).isoformat()
 record=dict(status='completed-one-day-fixed-material-continuation',result=ref(RUN/'result.json'),comparison=ref(REPORT/'summary.json'),delivery=ref(REPORT/'delivery.json'),visual_review=ref(REPORT/'visual-review.json'),new_BP_solves=24,new_initial_BP_solves=0,scientific_acceptance=False,accepted_count_effect=0,created_at=now)
 cp=E/'completion.json';write_json(cp,record);tl=result['timeline']
 addition=f'''\n\n## {num}. System2387 independent-outlet one-day dynamics completed\n\n- User continued current2387 task. Exact sealed v16 h,A,u reused without initial momentum solve;unchanged mesh,bed,frozen material,three-sigma BP,no-slip base. Existing v12 conservative median-dual upwind transport and v13r1 momentum used for24 hourly updates over1/365year. Fixed footprint with exterior outflow,no inflow,zero SMB,zero thermal update;no reconstruction or parameter adjustment. Sealed v12 restart and v13r1 assembly fixtures reused,one zero-step source residual preflight;25 hourly states and Newton/pending transport checkpoints retained.\n- Completed24hours,{result['total_substeps']}transport substeps,wall{result['wall_seconds']:.3f}seconds. Maximum speed{tl[0]['max_speed_m_year']:.10f}to{tl[-1]['max_speed_m_year']:.10f}m/year;old abnormal pointA{tl[0]['old_anomaly_A_speed_m_year']:.10f}to{tl[-1]['old_anomaly_A_speed_m_year']:.10f}m/year. Maximum nodal H change{tl[-1]['maximum_H_change_m']:.10f}m. Initial/final volumes{tl[0]['volume_m3']:.10f}/{tl[-1]['volume_m3']:.10f}m3;export{result['exported_volume_m3']:.10f}m3;global mass error{result['mass_error_m3']:.12g}m3,relative{result['checks']['relative_mass_budget']:.12g};junctionrelative{result['checks']['relative_junction_budget']:.12g}. All25 sealed states restored,finite/nonnegative H and bitwise fixedA checked;all mass sources zero and final time exact.\n- Retain this stable single-day dynamics result as working candidate. It establishes this short fixed-material continuation,not long-term equilibrium,climate/thermal coupling,independent paleo-geometry validation or formal scientific acceptance. No original baseline replay,no production replacement,no accepted-count effect,no automaticextension. Hourly maps use native pixel-center fields;reported extrema include all nodes. Evidence `{ref(cp)['path']}`.\n'''
 fh=hashlib.sha256(fb+addition.encode('utf-8')).hexdigest().upper();branch={**b,'status':record['status'],'launch_authorized':False,'real_momentum_authorized':False,'real_glacier_evolution_authorized':False,'allowed_stages':[],'evidence':ref(cp),'next_legal_action':'Use completed day-one candidate;register any longer continuation separately.'}
 route=read(PROJECT/s['authoritative_route']['path']);route['version']+='+polim-v17-day1-completed';route[NAME]=branch
 if 'scientific_contract' in route:route['scientific_contract']={**s['scientific_contract'],'sha256':fh}
 rp=E/'route-completed.json';write_json(rp,route);n=copy.deepcopy(s);n['research_branches'][NAME]=branch;n['scientific_contract']['sha256']=fh;n['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':s['authoritative_route']['version']};n['state_id']+='+polim-v17-day1-completed';n['updated_at']=now
 for k in ['counts','current_focus','next_action','population']:assert n[k]==s[k]
 if sp.read_bytes()!=sb or fp.read_bytes()!=fb:raise RuntimeError('Concurrent state/facts update')
 tmp=sp.with_name(f'.LIA_CURRENT_STATE.day1close.{os.getpid()}.tmp');write_json(tmp,n)
 with fp.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
 os.replace(tmp,sp)
 with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} - System2387 one-day continuation completed\n\n24hour fixed-material BP/transport finished,25sealed states verified. Peak{tl[-1]["max_speed_m_year"]:.6f}m/year,oldA{tl[-1]["old_anomaly_A_speed_m_year"]:.6f}m/year,maxHchange{tl[-1]["maximum_H_change_m"]:.6f}m. Global/local conservation passed. Facts{num};no count or production changes. `{ref(cp)["path"]}`.\n')
 write_json(E/'completed-validation.json',validate());print(json.dumps(dict(status=record['status'],facts_section=num)))

if __name__=='__main__':main()
