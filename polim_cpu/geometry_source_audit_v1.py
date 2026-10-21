"""Read-only geometry provenance, no evolution or geometry modification."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates,distance_transform_edt
from scipy.spatial import cKDTree
from .v3.runner import PROJECT,write_json,digest
from .thermal_attribution_v1 import validate
from .spatial_pair_guard import require_paired_source
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers import lia_system_grid as grid
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.paths import resolve

OUT=PROJECT/'polim_cpu/reports/geometry-source-audit-v1'

def main():
    validate();OUT.mkdir(exist_ok=False,parents=True)
    source=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json';package=json.loads(source.read_text(encoding='utf-8'));require_paired_source(source,PROJECT/package['network']['path'])
    with np.load(PROJECT/package['array_archive']['path'],allow_pickle=False) as z:a={k:z[k] for k in z.files}
    cfgpath=PROJECT/'Himalaya_greatGlacier/With_Rec_climate_organized/ice_surface_reconstruction/config/lia_surface_production_v0.json';cfg=json.loads(cfgpath.read_text(encoding='utf-8'))
    target=grid.SystemGrid(package['metadata']['crs_wkt'],float(a['x'][0]),float(a['y'][0]),len(a['x']),len(a['y']),float(a['x'][1]-a['x'][0]),float(a['y'][1]-a['y'][0]),OUT/'unused.json')
    dem,deminfo=grid._load_system_topography(Path(cfg['surface_dem_path']),target)
    ids=['RGI60-15.04813','RGI60-15.04811','RGI60-15.04814'];dirs=[grid._gdir_path(resolve(cfg['stage0_gdir_root']),s) for s in ids]
    bed,_,modern,info=grid.mosaic_system_bed(dirs,target,'farinotti2019',system_topography_m=dem)
    mpath=PROJECT/'polim_cpu/runs/v6-global-bp-r1/system2387-native-mesh/mesh.npz'
    with np.load(mpath,allow_pickle=False) as z:m={k:z[k] for k in z.files}
    with np.load(PROJECT/'polim_cpu/runs/v10-material-rheology-v1/real/frame-000/state.npz',allow_pickle=False) as z:u=z['u']
    speed=np.linalg.norm(u[:,-1],axis=1);i=int(np.argmax(speed));xy=m['xy'][i];r=int(round((xy[1]-target.y0)/target.dy));c=int(round((xy[0]-target.x0)/target.dx))
    profiles=[]
    for p in sorted((PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired').glob('branch-*.npz')):
        with np.load(p,allow_pickle=False) as z:profiles.append({k:z[k] for k in z.files})
    coords=np.concatenate([np.column_stack([p['easting_m'],p['northing_m']]) for p in profiles]);dist,j=cKDTree(coords).query(xy)
    def grad(field,k=1):return np.array([(field[r,c+k]-field[r,c-k])/(2*k*target.dx),(field[r+k,c]-field[r-k,c])/(2*k*target.dy)])
    S=a['surface_elevation_m'];H=a['thickness_m'];B=a['bed_elevation_m'];g=grad(S);direction=-g/np.linalg.norm(g)
    scales=[]
    for k in [1,2,4,8]:scales.append(dict(span_m=2*k*target.dx,dem_slope=float(np.linalg.norm(grad(dem,k))),bed_slope=float(np.linalg.norm(grad(B,k))),surface_slope=float(np.linalg.norm(grad(S,k))),H_slope=float(np.linalg.norm(grad(H,k)))))
    np.savez_compressed(OUT/'audit-fields.npz',dem=dem,bed_reconstructed=bed,modern_mask=modern,peak_xy=xy,downhill=direction,speed=speed,profile_xy=coords)
    result=dict(status='read-only-audit',peak_node=i,peak_xy=xy.tolist(),nearest_source_row=r,nearest_source_col=c,peak_is_modern_ice=bool(modern[r,c]),nearest_centerline_distance_m=float(dist),nearest_centerline_xy=coords[j].tolist(),dem_m=float(dem[r,c]),bed_m=float(B[r,c]),H_m=float(H[r,c]),surface_m=float(S[r,c]),bed_mosaic_exact_float32=bool(np.array_equal(bed.astype(np.float32),B.astype(np.float32),equal_nan=True)),bed_mosaic_max_difference_m=float(np.nanmax(abs(bed-B))),gradient_scales=scales,modern_boundary_distance_m=float(distance_transform_edt(modern if modern[r,c] else ~modern)[r,c]*target.dx),sources=[dict(path=str(p),sha256=digest(p)) for p in [source,cfgpath,mpath,*[d/'gridded_data.nc' for d in dirs]]],mosaic=info,dem_source=deminfo,new_simulations=0)
    write_json(OUT/'audit.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
