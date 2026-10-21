"""A disclosed derived-bed transition, preserving H and all centreline supports.

This is a source-resolution regularization assumption, not observed bed recovery.
The raw regional DEM and modern F19 bed remain immutable. No velocity is used.
"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt,binary_dilation
from ..v3.runner import PROJECT,digest,write_json

AUDIT=PROJECT/'polim_cpu/reports/geometry-source-audit-v1'
ARCHIVE=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz'
PROFILES=PROJECT/'polim_cpu/lia-inputs-v1/system-2387/spatial-profiles-v3-paired'
WIDTH=63.0  # coarsest contributing modern-bed source pixel; no speed fitting


def smoothstep(x):
    t=np.clip(x,0.,1.);return t*t*(3-2*t)


def build(mesh,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    with np.load(AUDIT/'audit-fields.npz',allow_pickle=False) as z:dem=z['dem'];modern=z['modern_mask']
    with np.load(ARCHIVE,allow_pickle=False) as z:a={k:z[k] for k in z.files}
    old=a['bed_elevation_m'];mask=a['valid_mask'];dx=float(a['x'][1]-a['x'][0]);dy=float(a['y'][1]-a['y'][0]);pins=np.zeros(mask.shape,bool)
    # Freeze bilinear support plus one source-cell ring: protects both raster
    # centreline sampling and centre/corner triangles around the same points.
    for path in sorted(PROFILES.glob('branch-*.npz')):
        with np.load(path,allow_pickle=False) as z:
            x=(z['easting_m']-a['x'][0])/dx;y=(z['northing_m']-a['y'][0])/dy
            for rr in [np.floor(y).astype(int),np.ceil(y).astype(int)]:
                for cc in [np.floor(x).astype(int),np.ceil(x).astype(int)]:pins[rr,cc]=True
            pins[z['source_row'],z['source_col']]=True
    pins=binary_dilation(pins,structure=np.ones((3,3),bool))
    margin_distance=np.maximum(distance_transform_edt(modern)*dx-dx/2,0.)
    pin_distance=distance_transform_edt(~pins)*dx
    weight=(1-smoothstep(margin_distance/WIDTH))*smoothstep(pin_distance/WIDTH)
    delta=np.where(modern & mask & ~pins,weight*(dem-old),0.)
    result={k:v.copy() for k,v in mesh.items()};nodes=mesh['pixel_center_nodes'];corners=mesh['pixel_corner_nodes'] if 'pixel_corner_nodes' in mesh else mesh['corner_nodes']
    values=delta[mesh['pixel_rows'],mesh['pixel_cols']];nodal=np.zeros(len(mesh['xy']));nodal[nodes]=values
    sums=np.bincount(corners.ravel(),weights=np.repeat(values,4),minlength=len(nodal));count=np.bincount(corners.ravel(),minlength=len(nodal));selected=count>0;nodal[selected]=sums[selected]/count[selected]
    result['bed']=mesh['bed']+nodal
    if not np.array_equal(result['thickness'],mesh['thickness']):raise ValueError('H changed')
    if np.any(delta[pins]!=0):raise ValueError('Branch constraints changed')
    profile_node_change=0.
    for path in sorted(PROFILES.glob('branch-*.npz')):
        with np.load(path,allow_pickle=False) as z:
            rr=z['source_row'];cc=z['source_col'];lookup={(int(r),int(c)):j for j,(r,c) in enumerate(zip(mesh['pixel_rows'],mesh['pixel_cols']))}
            for r,c in zip(rr,cc):
                j=lookup.get((int(r),int(c)))
                if j is not None:profile_node_change=max(profile_node_change,float(np.max(abs(nodal[np.r_[nodes[j],corners[j]]]))))
    if profile_node_change!=0:raise ValueError('Mesh support at branch constraints changed')
    with (root/'mesh.npz').open('xb') as f:np.savez_compressed(f,**result)
    with (root/'bed-change.npz').open('xb') as f:np.savez_compressed(f,delta_bed_m=delta,pinned_mask=pins,transition_weight=weight,nodal_delta_bed_m=nodal)
    summary=dict(status='derived-bed-built',width_m=WIDTH,width_basis='coarsest contributing modern F19 grid cell63m; fixed before dynamic result',
        original_H_exact=True,original_mesh_connectivity_exact=True,all_centerline_support_changes_m=profile_node_change,
        changed_source_pixels=int(np.count_nonzero(delta)),maximum_bed_change_m=float(np.max(abs(delta))),
        peak_node_bed_change_m=float(nodal[31985]),source_volume_unchanged=True,
        pin_conflict_policy='Freeze original support; do not override a branch constraint to remove a margin seam',
        scientific_claim='Constrained source-resolution bed transition; not observed basal topography or validated zero ice at every modern margin',
        source_geometry_modified=False,derived_mesh_sha256=digest(root/'mesh.npz'))
    write_json(root/'summary.json',summary);return result,summary
