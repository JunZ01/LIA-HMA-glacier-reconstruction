"""Static terrain candidates, no glacier reconstruction, dynamics or source mutation."""
from pathlib import Path
import json,heapq,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
R=Path(__file__).resolve().parent;O=R/'reports/junction-terrain-check-v1';O.mkdir(exist_ok=True)
if (O/'summary.json').exists(): raise RuntimeError('Do not repeat completed analysis')
g=np.load(R/'lia-inputs-v1/system-2387/spatial-source-paired-v1/geometry.npz',allow_pickle=False);a=np.load(R/'reports/geometry-source-audit-v1/audit-fields.npz',allow_pickle=False);m=np.load(R/'lia-inputs-v1/system-2387/modern-raw-f19-bp-control-v1/geometry.npz',allow_pickle=False);net=json.loads((R/'lia-inputs-v1/system-2387/spatial-profiles-v3-paired/original-network.json').read_text(encoding='utf-8'));old=np.load(R/'reports/branch0-production-trace-v1/connector-0.npz',allow_pickle=False);obs=np.load(R/'reports/observed-velocity-v1/comparison-fields-v2.npz',allow_pickle=False)
x,y=g['x'],g['y'];dx=float(x[1]-x[0]);dy=float(y[1]-y[0]);X,Y=np.meshgrid(x,y);mask=g['lia_mask'].astype(bool)&np.isfinite(a['dem']);ny,nx=mask.shape;b0=next(b for b in net['branches'] if b['branch_index']==0);b7=next(b for b in net['branches'] if b['branch_index']==7);T=np.array(b0['coords_projected_m'][-1]);P=np.array(b0['flows_to_point_projected_m']);tr=np.array(b7['coords_projected_m']);pi=int(np.argmin(np.linalg.norm(tr-P,axis=1)));start=tuple(np.rint([(T[1]-y[0])/dy,(T[0]-x[0])/dx]).astype(int));targets={}
for j,q in enumerate(tr):
 rc=tuple(np.rint([(q[1]-y[0])/dy,(q[0]-x[0])/dx]).astype(int))
 if j>=pi and 0<=rc[0]<ny and 0<=rc[1]<nx and mask[rc]:targets[rc]=j
moves=[(dr,dc,float(np.hypot(dr,dc)*dx)) for dr in [-1,0,1] for dc in [-1,0,1] if dr or dc]
def neighbours(r,c,allowed):
 for dr,dc,step in moves:
  nr,nc=r+dr,c+dc
  if 0<=nr<ny and 0<=nc<nx and allowed[nr,nc] and (not(dr and dc) or (allowed[r+dr,c] and allowed[r,c+dc])):yield nr,nc,step

def terrain_route(F):
 allowed=mask&np.isfinite(F);cost=np.full(F.shape,np.inf);cost[start]=F[start];queue=[(float(F[start]),*start)];threshold=None
 while queue:
  v,r,c=heapq.heappop(queue)
  if v!=cost[r,c]:continue
  if (r,c) in targets:threshold=v;break
  for nr,nc,_ in neighbours(r,c,allowed):
   nv=max(v,float(F[nr,nc]),float(F[r,nc]),float(F[nr,c])) if nr!=r and nc!=c else max(v,float(F[nr,nc]))
   if nv<cost[nr,nc]:cost[nr,nc]=nv;heapq.heappush(queue,(nv,nr,nc))
 if threshold is None:raise RuntimeError('No terrain connection')
 allowed &= F<=threshold+1e-8;dist=np.full(F.shape,np.inf);dist[start]=0;parent=np.full(F.shape,-1,dtype=np.int64);queue=[(0.,*start)];end=None
 while queue:
  v,r,c=heapq.heappop(queue)
  if v!=dist[r,c]:continue
  if (r,c) in targets:end=(r,c);break
  for nr,nc,step in neighbours(r,c,allowed):
   nv=v+step
   if nv<dist[nr,nc]:dist[nr,nc]=nv;parent[nr,nc]=r*nx+c;heapq.heappush(queue,(nv,nr,nc))
 if end is None:raise RuntimeError('No shortest threshold route')
 route=[end]
 while route[-1]!=start:
  k=int(parent[route[-1]]);route.append(divmod(k,nx))
 return np.array(route[::-1]),targets[end],threshold

def direction_stats(xy):
 # Polygon/mask overlap and SNR are taken from saved independent observation products.
 tangent=np.diff(xy,axis=0);mid=(xy[:-1]+xy[1:])/2;norm=np.linalg.norm(tangent,axis=1);tangent=tangent/np.maximum(norm[:,None],1e-30);ox,oy=obs['local_x'],obs['local_y'];d,j=cKDTree(mid).query(np.column_stack([ox.ravel(),oy.ravel()]));d=d.reshape(ox.shape);j=j.reshape(ox.shape);v=np.stack([obs['observed_direction_x'],obs['observed_direction_y']],axis=-1);dot=np.sum(tangent[j]*v,axis=-1);angle=np.degrees(np.arccos(np.clip(dot,-1,1)));base=(d<=120)&obs['valid'].astype(bool)&(obs['modern_fraction']>=.5)&np.isfinite(angle)&(obs['observed']>0);strict=base&(obs['error']>0)&(obs['observed']>2*obs['error'])
 def st(t):return dict(n=int(t.sum()),median_angle_deg=float(np.median(angle[t])) if t.any() else None,median_speed_myr=float(np.median(obs['observed'][t])) if t.any() else None)
 return dict(all_valid=st(base),positive_error_snr2=st(strict))
newbed=np.where(m['mask'],m['bed'],a['dem']);fields={'regional_DEM':a['dem'],'old_bed':a['bed_reconstructed'],'raw_F19_bed':newbed};results={}
oldrc=old['rc'][::-1];oldxy=old['xy'][::-1]
for label,F in fields.items():
 saved=O/(label+'.npz')
 if saved.exists():
  stored=np.load(saved,allow_pickle=False);rc=stored['rc'];xy=stored['xy'];h=stored['height'];station=stored['station'];j=targets[tuple(rc[-1])];threshold=float(h.max())
 else:
  rc,j,threshold=terrain_route(F);xy=np.column_stack([x[rc[:,1]],y[rc[:,0]]]);h=F[rc[:,0],rc[:,1]];station=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))];np.savez_compressed(saved,rc=rc,xy=xy,height=h,station=station)
 oh=F[oldrc[:,0],oldrc[:,1]]
 results[label]=dict(main_native_index=int(j),main_xy=tr[j].tolist(),candidate_length_m=float(station[-1]),minimum_barrier_above_start_m=float(max(0,threshold-F[start])),old_route_barrier_above_start_m=float(max(0,oh.max()-F[start])),old_route_end_minus_start_m=float(oh[-1]-F[start]),candidate_total_ascent_m=float(np.clip(np.diff(h),0,None).sum()),candidate_end_minus_start_m=float(h[-1]-h[0]),candidate_direction=direction_stats(xy))
result=dict(status='completed-static-terrain-candidates',new_simulations=0,source_modified=False,old_main_index=pi,old_main_xy=P.tolist(),tributary_endpoint=T.tolist(),old_direction=direction_stats(oldxy),candidates=results,method='Minimise maximum terrain elevation required to reach any saved downstream main-line node; then shortest eight-neighbour path under that height threshold, within LIA mask, no corner cutting. No speed fitting. Candidates are geographic diagnostics, not accepted flowlines; contemporary surface and inferred beds are different terrain proxies.',limitations=['Bed-uphill glacier motion is possible; terrain minimax is not a physical ice-flow solution.','Original other branches/connectors are not obstacles here: candidate intersections require topology reconciliation before adoption.','Modern directions only constrain current ice and are not LIA observations.'])
(O/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(result))
