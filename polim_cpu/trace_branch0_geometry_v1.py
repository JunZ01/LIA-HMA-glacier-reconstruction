"""Restore geometry-only production adapter; never call a surface or BP solver."""
from pathlib import Path
import json,numpy as np
from scipy.ndimage import map_coordinates
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.gridded_bed import GriddedBed
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.oggm_network_adapter import FlowlineInput,build_network,EXPLICIT_CONNECTOR
from Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.junction_connector import ConnectorCostWeights
R=Path(__file__).resolve().parent
O=R/'reports/branch0-production-trace-v1';O.mkdir(exist_ok=False)
g=np.load(R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz',allow_pickle=False);a=np.load(R/'reports/geometry-source-audit-v1/audit-fields.npz',allow_pickle=False);net=json.loads((R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json').read_text(encoding='utf-8'));pkg=json.loads((R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/input-package.json').read_text(encoding='utf-8'))
assert net['crs']==pkg['metadata']['crs_wkt']
x,y=g['x'],g['y'];dx=x[1]-x[0];dy=y[1]-y[0];B=a['bed_reconstructed'];mask=g['lia_mask'].astype(bool);sy,sx=np.gradient(B,dx,dx);bed=GriddedBed(B,mask,np.arctan(np.hypot(sx,sy)),float(dx),'farinotti2019');branches=sorted(net['branches'],key=lambda t:t['branch_index']);coords={int(b['branch_index']):np.column_stack([(np.array(b['coords_projected_m'])[:,0]-x[0])/dx,(np.array(b['coords_projected_m'])[:,1]-y[0])/dy]) for b in branches};inputs=[]
for b in branches:
 i=int(b['branch_index']);q=coords[i];parent=b['parent_index'];pi=None
 if parent is not None:
  jp=b['flows_to_point_projected_m'];j=np.array([(jp[0]-x[0])/dx,(jp[1]-y[0])/dy]);dist=np.linalg.norm(coords[parent]-j,axis=1);pi=int(np.argmin(dist));contact=np.linalg.norm(coords[parent]-q[-1],axis=1);ci=int(np.argmin(contact))
  if contact[ci]<=1:pi=ci
 sampled=map_coordinates(B,[q[:,1],q[:,0]],order=1,mode='nearest')
 inputs.append(FlowlineInput(i,parent,pi,sampled,np.array(b['reference_surface_m']),float(b['dx_meter']),float(dx),q))
build=build_network(inputs,junction_transfer_policy=EXPLICIT_CONNECTOR,gridded_bed=bed,connector_weights=ConnectorCostWeights(alpha=0.,beta=0.),avoid_existing_network_cells=True)
records=[]
for co in build.connector_audit:
 br=next(b for b in build.branches if b.branch_id==co.connector_branch_id);rc=np.array(co.pixels_rc);H=g['thickness_m'][rc[:,0],rc[:,1]].astype(float);S=g['surface_elevation_m'][rc[:,0],rc[:,1]].astype(float);bb=br.bed_elevation_m; station=br.distance_m;xy=np.column_stack([x[rc[:,1]],y[rc[:,0]]]);np.savez_compressed(O/f'connector-{co.tributary_flowline_index}.npz',rc=rc,xy=xy,station=station,B=bb,H=H,S=S,DEM=a['dem'][rc[:,0],rc[:,1]],modern=a['modern_mask'][rc[:,0],rc[:,1]])
 records.append(dict(branch=co.tributary_flowline_index,parent=co.parent_flowline_index,length_m=co.total_length_m,points=len(rc),max_H=float(H.max()),max_H_index=int(H.argmax()),parent_B=float(bb[0]),end_B=float(bb[-1]),parent_H=float(H[0]),end_H=float(H[-1]),parent_S=float(S[0]),end_S=float(S[-1]),min_B=float(bb.min()),max_B=float(bb.max()),upstream_S_falls=int(np.sum(np.diff(S)<-.01)),endpoint_rc=rc[-1].tolist()))
# All original-plus-production-connector raster support before head exclusion.
support=np.zeros(mask.shape,bool)
for q in coords.values():
 rc=np.rint(q[:,::-1]).astype(int);support[rc[:,0],rc[:,1]]=True
for co in build.connector_audit:
 rc=np.array(co.pixels_rc);support[rc[:,0],rc[:,1]]=True
rr,cc=np.unravel_index(np.nanargmax(g['thickness_m']),g['thickness_m'].shape)
result=dict(status='geometry-only-production-path-trace',new_surface_solves=0,new_BP_solves=0,weights=dict(alpha=0,beta=0),connector_count=len(records),connectors=records,global_max=dict(row=int(rr),col=int(cc),H=float(g['thickness_m'][rr,cc]),on_candidate_constraint=bool(support[rr,cc])))
np.savez_compressed(O/'production-support.npz',support=support)
(O/'trace.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
