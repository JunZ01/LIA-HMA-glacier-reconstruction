"""Register completed observational audit without simulation authorization."""
import copy,json,os,re,hashlib
from datetime import datetime,timezone
from .v3.runner import PROJECT,digest,write_json
from .thermal_attribution_v1 import validate,ref
from .observed_velocity_v1 import OUT

def main():
    validate();path=PROJECT/'LIA_CURRENT_STATE.json';before=path.read_bytes();state=json.loads(before);facts=PROJECT/'LIA_PROJECT_FACTS.md';original=facts.read_bytes()
    if digest(facts)!=state['scientific_contract']['sha256'].lower():raise ValueError('Facts changed')
    name='polim_lia_observed_velocity_comparison_v1'
    if name in state['research_branches']:raise ValueError('Already registered')
    s=json.loads((OUT/'summary-v2.json').read_text(encoding='utf-8'))
    for p,sha in s['source_hashes'].items():
        if digest(PROJECT/p)!=sha:raise ValueError('Input changed: '+p)
    if json.loads((OUT/'page-check.json').read_text())['status']!='pass':raise ValueError('Page check failed')
    # Original delivery manifest remains immutable; v2 includes export/quality clarification.
    write_json(OUT/'delivery-v2.json',dict(status='completed',summary=ref(OUT/'summary-v2.json'),files={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file()},new_simulations=0))
    ev=PROJECT/'validation_data/results/paleo_surface_validation_v0'/name;ev.mkdir(exist_ok=False)
    now=datetime.now(timezone.utc).isoformat();number=max(map(int,re.findall(r'^## (\d+)\.',original.decode('utf-8'),re.M)))+1;m=s['subsets']['modern_overlap'];j=s['junction'];pk=s['peak']
    completion=dict(status='completed-read-only-comparison',summary=ref(OUT/'summary-v2.json'),delivery=ref(OUT/'delivery-v2.json'),source_hashes_verified=True,new_simulations=0,accepted_count_effect=0,scientific_acceptance=False,created_at=now,user_authorization='User explicitly starts proposed target-glacier satellite velocity comparison using existing simulations.')
    cp=ev/'completion.json';write_json(cp,completion)
    addition=f'''\n\n## {number}. Target2387 modern satellite velocity comparison completed\n\n- User starts the proposed read-only target-glacier comparison. Reused local NASA ITS_LIVE v2.1 RGI14A120m climatology2014-2024,time intercept2018-01-01; all eight target-clip SHA256 values match original download registry. This is not a single-year velocity or an LIA observation. Reused original paired geometry and v12 saved initial/day1 results with no numerical run,calibration,source modification or accepted-count effect.\n- Model15.75m source pixel-centre values aggregated into actual120m observation cells. Common mask requires >=50percent sampled modern source-mask area and >=50percent LIA area;{m['native_cells']}unique cells,all have finite nonnegative v,finite vx/vy,positive observation count. Observed median/P95/max{m['observed']['median']:.9f}/{m['observed']['p95']:.9f}/{m['observed']['max']:.9f}m/year;day1 model{m['model_day1']['median']:.9f}/{m['model_day1']['p95']:.9f}/{m['model_day1']['max']:.9f}m/year on the same cells. The model has very slow areas as well as localized high speeds;this is a spatial reference,not proof of physical failure or a modern-geometry residual.\n- Major junction240m centre-radius selection has{j['observed']['n']}observed cells,all in common mask: observed median{j['observed']['median']:.9f},model median{j['model_day1']['median']:.9f}m/year. Radius differs from v12 conservation126m and serves observational120m sampling only. Highest original model node{pk['node']}remains outside modern source mask;its120m cell modern fraction{pk['modern_fraction_of_observation_cell']:.9f}. Day1 nodal H{pk['H_m']:.9f}m,nearby native model surface slope{pk['model_surface_slope_degrees']:.9f}degrees versus modern DEM{pk['modern_dem_slope_degrees']:.9f}degrees. Derivative is resolution-sensitive,not measured bed/ice-surface truth. Boundary mixed-cell observed speed is not a direct truth value for the LIA peak.\n- Integer uint16 v_error at1m/year precision has{m['zero_stored_error_cells']}stored zero errors in common mask;zero does not mean no uncertainty. Original auxiliary resolved flags including error0 are explicitly withdrawn. Current summary-v2 and comparison-fields-v2 retain raw speeds but only positive error permits SNR>2 diagnostic;{m['positive_error_snr_gt2_cells']}common cells qualify. This is not a scientific gate. Original profile JSON export failed on NaN after arrays/statistics were saved;null-safe exporter reused those arrays,not a new model run. Earlier outputs retained and final delivery-v2 identifies all corrections.\n- Delivered four-panel map,modern/model slope comparison,all-eight-branch profiles and individual selectable branch views. System modern mask follows original bed-source gridded mask;member masks follow RGI polygons,so member counts are not additive. The source masks are inventory-era rather than a newly mapped2024 margin. Coordinate roundtrip and source hashes verified;all eight page selections and local assets passed one combined check. No Full-Stokes,automatic new simulation or mainline replacement. Evidence `{ref(cp)['path']}`.\n'''
    sha=hashlib.sha256(original+addition.encode('utf-8')).hexdigest().upper()
    branch=dict(status=completion['status'],launch_authorized=False,real_momentum_authorized=False,real_glacier_evolution_authorized=False,full_stokes_authorized=False,allowed_stages=[],evidence=ref(cp),accepted_count_effect=0,production_replacement_authorized=False,next_legal_action='Use observed comparison to scope next geometry/physics implementation; no numerical launch in this read-only audit.')
    route=json.loads((PROJECT/state['authoritative_route']['path']).read_text(encoding='utf-8'));route['version']+='+polim-observed-velocity-v1';route[name]=branch
    if 'scientific_contract' in route:route['scientific_contract']={**state['scientific_contract'],'sha256':sha}
    rp=ev/'route-reviewed.json';write_json(rp,route);new=copy.deepcopy(state);new['research_branches'][name]=branch;new['scientific_contract']['sha256']=sha;new['authoritative_route']={**ref(rp),'version':route['version'],'supersedes':state['authoritative_route']['version']};new['updated_at']=now;new['state_id']+='+polim-observed-velocity-v1'
    for k in ['counts','population','current_focus','next_action']:
        if new[k]!=state[k]:raise ValueError('Mainline changed')
    if path.read_bytes()!=before or facts.read_bytes()!=original:raise ValueError('Concurrent state update')
    tmp=path.with_name(f'.LIA_CURRENT_STATE.obsvelocity.{os.getpid()}.tmp');write_json(tmp,new);os.replace(tmp,path)
    with (PROJECT/'LIA_DECISION_LOG.md').open('a',encoding='utf-8') as f:f.write(f'\n\n## {now} — Target2387 observed velocity comparison delivered\n\nRead-only modern satellite versus LIA-geometry BP comparison;no simulation or parameter tuning. Raw source hashes verified,spatial differences and epoch/mixed-pixel/integer-error limitations explicit. Facts section{number};completion `{ref(cp)["path"]}`. Counts and unrelated tasks preserved.\n')
    with facts.open('ab') as f:f.write(addition.encode('utf-8'));f.flush();os.fsync(f.fileno())
    write_json(ev/'controller-validation.json',validate());print(json.dumps(dict(status=completion['status'],facts_section=number,new_simulations=0)))

if __name__=='__main__':main()
