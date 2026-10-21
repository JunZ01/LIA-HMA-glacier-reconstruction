"""Terrain drainage diagnostic using a temporary filled surface, never change glacier inputs."""
from pathlib import Path
import heapq,json
import numpy as np
from scipy.spatial import cKDTree
R=Path(__file__).resolve().parent;O=R/'reports/branch0-outlet-trace-v1';O.mkdir(exist_ok=False)
g=np.load(R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz');a=np.load(R/'reports/geometry-source-audit-v1/audit-fields.npz');m=np.load(R/'lia-inputs-v1/system-2387/modern-raw-f19-bp-control-v1/geometry.npz');net=json.loads((R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json').read_text(encoding='utf-8'));obs=np.load(R/'reports/observed-velocity-v1/comparison-fields-v2.npz');x,y=g['x'],g['y'];dx=float(x[1]-x[0]);dy=float(y[1]-y[0]);ny,nx=a['dem'].shape;branches={b['branch_index']:b for b in net['branches']};T=np.array(branches[0]['coords_projected_m'][-1]);tr=np.array(branches[7]['coords_projected_m']);P=np.array(branches[0]['flows_to_point_projected_m']);start=tuple(np.rint([(T[1]-y[0])/dy,(T[0]-x[0])/dx]).astype(int));moves=[(dr,dc,float(np.hypot(dr,dc))) for dr in [-1,0,1] for dc in [-1,0,1] if dr or dc]
def flood(F):
 if not np.isfinite(F).all():raise RuntimeError('Full DEM context must be finite')
 visited=np.zeros(F.shape,bool);filled=F.copy();parent=np.full(F.shape,-1,np.int64);queue=[];rank=np.full(F.shape,-1,np.int64)
 for r in range(ny):
  for c in ([0,nx-1] if r not in [0,ny-1] else range(nx)):
   if not visited[r,c]:visited[r,c]=True;heapq.heappush(queue,(float(F[r,c]),r,c))
 k=0
 while queue:
  v,r,c=heapq.heappop(queue);rank[r,c]=k;k+=1
  for dr,dc,_ in moves:
   rr,cc=r+dr,c+dc
   if 0<=rr<ny and 0<=cc<nx and not visited[rr,cc]:
    visited[rr,cc]=True;parent[rr,cc]=r*nx+c;h=max(float(F[rr,cc]),v);filled[rr,cc]=h;heapq.heappush(queue,(h,rr,cc))
 return filled,parent,rank

def route(F,filled,parent,start):
 route=[start];seen={start}
 for _ in range(F.size):
  r,c=route[-1]
  if parent[r,c]<0:break
  options=[]
  for dr,dc,length in moves:
   rr,cc=r+dr,c+dc
   if 0<=rr<ny and 0<=cc<nx and filled[rr,cc]<filled[r,c]-1e-10:options.append(((filled[r,c]-filled[rr,cc])/length,rr,cc))
  if options:_,rr,cc=max(options)
  else:rr,cc=divmod(int(parent[r,c]),nx)
  if (rr,cc) in seen:raise RuntimeError('Drainage cycle')
  seen.add((rr,cc));route.append((rr,cc))
 return np.array(route)

def stats(rc,F,filled):
 rr,cc=rc.T;xy=np.column_stack([x[cc],y[rr]]);station=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];outside=np.flatnonzero(~g['lia_mask'][rr,cc].astype(bool));first=int(outside[0]) if outside.size else len(rc)-1;near,j=cKDTree(tr).query(xy[:first+1]);tangent=np.diff(xy[:first+1],axis=0);mid=(xy[:first]+xy[1:first+1])/2;tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-30);ox,oy=obs['local_x'],obs['local_y'];dist,idx=cKDTree(mid).query(np.column_stack([ox.ravel(),oy.ravel()]));dist=dist.reshape(ox.shape);idx=idx.reshape(ox.shape);dot=tangent[idx,0]*obs['observed_direction_x']+tangent[idx,1]*obs['observed_direction_y'];angle=np.degrees(np.arccos(np.clip(dot,-1,1)));ok=obs['valid'].astype(bool)&(obs['modern_fraction']>=.5)&(obs['error']>0)&(obs['observed']>2*obs['error'])&(dist<=120)&np.isfinite(angle)
 return dict(first_LIA_exit_xy=xy[first].tolist(),first_LIA_exit_height=float(F[tuple(rc[first])]),inside_length_m=float(station[first]),frame_exit_xy=xy[-1].tolist(),frame_exit_height=float(F[tuple(rc[-1])]),max_fill_inside_m=float(np.max(filled[rr[:first+1],cc[:first+1]]-F[rr[:first+1],cc[:first+1]])),nearest_original_mainline_m=float(near.min()),nearest_main_index=int(j[near.argmin()]),direction_samples=int(ok.sum()),direction_median_deg=float(np.median(angle[ok])) if ok.any() else None,first_exit_index=first),xy,station
results={}
fields={'regional_DEM':a['dem'],'old_bed':a['bed_reconstructed'],'raw_F19_bed':np.where(m['mask'],m['bed'],a['dem'])}
for name,F in fields.items():
 filled,parent,rank=flood(F);rc=route(F,filled,parent,start);s,xy,station=stats(rc,F,filled);np.savez_compressed(O/(name+'.npz'),rc=rc,xy=xy,station=station,height=F[rc[:,0],rc[:,1]],filled=filled[rc[:,0],rc[:,1]])
 mainstart=tuple(np.rint([(P[1]-y[0])/dy,(P[0]-x[0])/dx]).astype(int));mainrc=route(F,filled,parent,mainstart);ms,mxy,mstation=stats(mainrc,F,filled);np.savez_compressed(O/(name+'-main.npz'),rc=mainrc,xy=mxy,station=mstation,height=F[mainrc[:,0],mainrc[:,1]],filled=filled[mainrc[:,0],mainrc[:,1]])
 inter=set(map(tuple,rc))&set(map(tuple,mainrc));results[name]=dict(tributary=s,main=ms,drainage_paths_intersect=bool(inter),frame_exit_separation_m=float(np.linalg.norm(xy[-1]-mxy[-1])),first_LIA_exit_separation_m=float(np.linalg.norm(np.array(s['first_LIA_exit_xy'])-np.array(ms['first_LIA_exit_xy']))))
summary=dict(status='completed-terrain-outlet-audit',new_simulations=0,source_modified=False,fields=results,method='Priority-Flood depression filling on full rectangular DEM context; steepest descent on filled elevations, parent-chain fallback on flats. Filling is used only for terrain routing, not as glacier bed or ice surface. No prescribed main-line target.',reference='https://doi.org/10.1016/j.cageo.2013.04.024',limitations=['Terrain drainage is not proof of historical ice divides.','Modern surface and inferred bed routes are complementary proxies,not independent ice-thickness observations.','Domain-edge exits are truncated terrain routes; first LIA exits are also reported.'])
(O/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
