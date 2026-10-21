"""Bind original geometry, permitted 1441-1470 climate, and baseline parameters."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT=Path(__file__).resolve().parents[2]
SOURCE=PROJECT/'validation_data/results/paleo_surface_validation_v0/lia_191_gpu_four_group_implementation_v2'
DESIGN=PROJECT/'validation_data/results/paleo_surface_validation_v0/lia_191_gpu_four_group_design_v10_262_v1'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(out,system=2387):
    from .runner import write_json
    source=json.loads((SOURCE/'manifest.json').read_text())
    expected='6f7876068803086ea7bf00d376109c36046f86aabfc2e39552ebba33d6a44099'
    if sha(SOURCE/'inputs.npz')!=expected or source['inputs_sha256'].lower()!=expected: raise ValueError('Frozen input bundle changed')
    matches=[i for i,s in enumerate(source['systems']) if int(s['system'])==system]
    if len(matches)!=1: raise ValueError('System identity not unique')
    i=matches[0]
    fields=('system_bed_h','system_thick','system_widths_m','system_nx','system_dx_meter',
            'climate_years','system_climate_temp','system_climate_prcp','system_climate_gradient','system_climate_ref_hgt')
    with np.load(SOURCE/'inputs.npz',allow_pickle=False) as data:
        years=data['climate_years']
        if not np.array_equal(years,np.arange(1441,1471)): raise ValueError('Only the frozen allowed 1441-1470 climate is permitted')
        n=int(data['system_nx'][i]); dx=float(data['system_dx_meter'][i])
        arrays={'x':np.arange(n,dtype=float)*dx,'cell_width':np.full(n,dx),
                'bed':data['system_bed_h'][i,:n],'h':data['system_thick'][i,:n],'width':data['system_widths_m'][i,:n],
                'years':years,'temp':data['system_climate_temp'][i], 'prcp':data['system_climate_prcp'][i],
                'gradient':data['system_climate_gradient'][i],'ref_hgt':data['system_climate_ref_hgt'][i]}
    design=json.loads((DESIGN/'four_group_parameter_matrix.json').read_text())
    baseline=design['baseline_source']; path=PROJECT/baseline['path']
    if sha(path)!=baseline['sha256'].lower(): raise ValueError('Baseline source changed')
    with path.open(encoding='utf-8-sig',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if int(r['lia_unique_id'])==system]
    fields_base=baseline['per_system_fields']
    unique={json.dumps({k:r[k] for k in fields_base},sort_keys=True) for r in rows}
    if len(unique)!=1: raise ValueError('Baseline parameters are not unique')
    original=json.loads(unique.pop())
    parameters={k:float(original[k]) for k in ('base_temp_bias_degC','melt_f','prcp_fac','glen_a_pa_3_s_1','fs','temp_melt_degC')}
    if parameters['fs']!=0: raise ValueError('This pilot supports the inherited no-sliding baseline only')
    parameters.update(layers=9,basal='zero',flowband=True,glen_n=3,relative_tolerance=1e-6,max_iterations=100,
        strain_regularization=1e-30,momentum_transition_m=1.,geothermal_w_m2=.05,
        temperate_diffusivity_ratio=.001,water_cap=.03,thermal_feedback=True)
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    with (out/'inputs.npz').open('xb') as stream: np.savez_compressed(stream,**arrays)
    config={'case_id':f'lia-{system}-v3-technical','system':system,'input_npz':'inputs.npz',
        'input_sha256':sha(out/'inputs.npz'),'start_year':1441,'duration_years':.1,'max_dt_years':.005,
        'parameters':parameters,'initialization':'climate-derived surface temperature plus conductive geothermal profile capped at pressure melting',
        'thermal_hypothesis_authorized':True,'scientific_acceptance':False,'historical_reconstruction':False,
        'research_branch':'polim_lia_v3_system2387_pilot_v1'}
    write_json(out/'case.json',config)
    write_json(out/'provenance.json',{'source':str(SOURCE),'source_sha256':expected,'source_index':i,'system':system,
        'source_contract':source['contract'],'baseline_source':baseline,'baseline':original,
        'climate_years':[1441,1470],'forbidden_or_forward_climate_decoded':False,'npz_fields_decoded':list(fields),
        'geometry_unchanged':True,'dry_domain_retained':True,'coordinate_convention':'cell centres, every original bin has full dx',
        'density_kg_m3':900,'seconds_per_year':31536000,'glen_scale':'original calibrated value at pressure-corrected -10 C; Arrhenius and water multiplier elsewhere',
        'thermal_assumptions':{'geothermal_w_m2':.05,'observed_ice_temperature_available':False,'surface':'annual-mean climate at local surface, at most 0 C'},
        'approval':'User: 开始; user selected 采用这些假设，完成热力耦合试算',
        'purpose':'single-system short technical coupled pilot; no equilibrium or historical score, no population promotion'})
    return str(out/'case.json')
