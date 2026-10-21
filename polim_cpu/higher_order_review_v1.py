"""Read-only source geometry audit accompanying the higher-order literature review.
No momentum solve, no modified geometry, no climate decoding.
"""
import json
from pathlib import Path
import numpy as np
import xarray as xr
from .v3.runner import PROJECT,write_json,digest
from .spatial_pair_guard import require_paired_source
from .thermal_attribution_v1 import validate,ref

OUT=PROJECT/'polim_cpu/reports/higher-order-model-review-v1'


def main():
    controller=validate();OUT.mkdir(parents=True,exist_ok=False)
    package=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json'
    doc=json.loads(package.read_text(encoding='utf-8'));require_paired_source(package,PROJECT/doc['network']['path'])
    archive=PROJECT/doc['array_archive']['path'];source=PROJECT/doc['source']['path']
    meshpath=PROJECT/'polim_cpu/runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
    fields=PROJECT/'polim_cpu/runs/v7-native-global-bp-r1/fields.npz'
    with np.load(archive,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    with xr.open_dataset(source) as d:
        i=list(map(str,d.member_id.values)).index('farinotti2019')
        identical={k:bool(np.array_equal(a[k],d[k].isel(member=i).values,equal_nan=True)) for k in ('thickness_m','bed_elevation_m','surface_elevation_m')}
    with np.load(meshpath,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    with np.load(fields,allow_pickle=False) as z:u=z['velocity_m_year']
    vel=np.linalg.norm(u[:,-1],axis=1);cent=m['pixel_center_nodes'];j=int(np.argmax(vel[cent]))
    r=int(m['pixel_rows'][j]);c=int(m['pixel_cols'][j]);S=a['surface_elevation_m'];B=a['bed_elevation_m'];H=a['thickness_m']
    dx=float(a['x'][1]-a['x'][0]);dy=float(a['y'][1]-a['y'][0])
    def gradient(f,k):return np.array([(f[r,c+k]-f[r,c-k])/(2*k*dx),(f[r+k,c]-f[r-k,c])/(2*k*dy)])
    scales=[]
    for k in (1,2,4,8):
        gs=gradient(S,k);gb=gradient(B,k);gh=gradient(H,k);norm=np.linalg.norm(gs);direction=gs/norm
        scales.append(dict(span_m=2*k*abs(dx),surface_slope=float(norm),bed_slope=float(np.linalg.norm(gb)),
            thickness_slope=float(np.linalg.norm(gh)),bed_contribution_along_surface_gradient=float(gb@direction),
            thickness_contribution_along_surface_gradient=float(gh@direction)))
    result=dict(status='read-only-audit-completed',system=2387,source_archive_exact=identical,
        surface_closure_max_m=float(np.max(abs((B+H-S)[a['valid_mask']]))),
        native_center_bed_exact=bool(np.array_equal(m['bed'][cent],B[a['valid_mask']])) ,
        native_center_H_exact=bool(np.array_equal(m['thickness'][cent],H[a['valid_mask']])) ,
        fastest_source_pixel=dict(row=r,col=c,x_m=float(a['x'][c]),y_m=float(a['y'][r]),H_m=float(H[r,c]),B_m=float(B[r,c]),S_m=float(S[r,c]),velocity_m_year=float(vel[cent[j]])),
        gradient_scales=scales,gradient_method='centered difference over raw source, span is each axial end-to-end distance; no smoothing or model solve',
        sources=[ref(p) for p in (package,archive,source,meshpath,fields,Path(__file__))],
        A_SI=7.538801981725612e-23,A_Pa_minus3_year_minus1=7.538801981725612e-23*31536000,
        existing_boundary='basal no-slip; all lateral perimeter natural zero effective BP traction; no physical segment classification',
        findings=['No field swap or change between source archive and source netCDF',
            'Source center H and bed unchanged in native mesh; subpixel corner interpolation still present',
            'Steep surface already in source; bed and thickness gradients both contribute',
            'Source identity and b+H closure do not establish reconstruction accuracy',
            'No evidence here isolates cause of excessive speeds; boundary, rheology and BP approximation remain candidates'],
        geometry_modified=False,new_momentum_solves=0,new_time_steps=0,climate_read=False,scientific_acceptance=False,controller=controller)
    write_json(OUT/'source-audit.json',result)
    print(json.dumps({k:result[k] for k in ('status','source_archive_exact','surface_closure_max_m','fastest_source_pixel','gradient_scales')}))


if __name__=='__main__':main()
