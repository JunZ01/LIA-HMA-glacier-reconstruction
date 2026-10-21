"""Conforming four-triangle pixel mesh, no connection across a corner alone.

Pixel centers retain source B/H. Shared corners average incident pixel B/H.
Equal-area triangles preserve total area and volume exactly, but interpolate
within pixels. Original pixel quantities remain the scientific source ledger.
"""
import numpy as np
from scipy import ndimage


def build(mask,x,y,bed,thickness):
    mask=np.asarray(mask,dtype=bool);rows,cols=np.nonzero(mask);n=len(rows)
    if not n or bed.shape!=mask.shape or thickness.shape!=mask.shape:raise ValueError('Empty or invalid source')
    if len(x)<2 or len(y)<2:raise ValueError('Insufficient coordinate grid')
    dx=float(x[1]-x[0]);dy=float(y[1]-y[0])
    if dx==0 or dy==0 or not np.allclose(np.diff(x),dx) or not np.allclose(np.diff(y),dy):raise ValueError('Regular grid required')
    if not np.isfinite(bed[mask]).all() or not np.isfinite(thickness[mask]).all() or np.any(thickness[mask]<=0):raise ValueError('Positive finite source required')
    # Corner order: (row,col),(row,col+1),(row+1,col+1),(row+1,col).
    pixel_ids=np.full(mask.shape,-1,dtype=np.int64);pixel_ids[mask]=np.arange(n)
    parent=np.arange(n*4,dtype=np.int64)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for i,(r,c) in enumerate(zip(rows,cols)):
        if c+1<mask.shape[1] and mask[r,c+1]:
            j=int(pixel_ids[r,c+1]);union(i*4+1,j*4);union(i*4+2,j*4+3)
        if r+1<mask.shape[0] and mask[r+1,c]:
            j=int(pixel_ids[r+1,c]);union(i*4+3,j*4);union(i*4+2,j*4+1)
    roots=np.array([find(i) for i in range(n*4)])
    _,inverse=np.unique(roots,return_inverse=True);corners=inverse.reshape(n,4);nc=int(inverse.max()+1)
    corner_rc=np.stack([np.column_stack([rows,cols]),np.column_stack([rows,cols+1]),
                        np.column_stack([rows+1,cols+1]),np.column_stack([rows+1,cols])],axis=1).reshape(-1,2)
    counts=np.bincount(inverse,minlength=nc)
    corner_x=np.bincount(inverse,weights=x[0]+(corner_rc[:,1]-.5)*dx,minlength=nc)/counts
    corner_y=np.bincount(inverse,weights=y[0]+(corner_rc[:,0]-.5)*dy,minlength=nc)/counts
    cb=np.bincount(inverse,weights=np.repeat(bed[mask],4),minlength=nc)/counts
    ch=np.bincount(inverse,weights=np.repeat(thickness[mask],4),minlength=nc)/counts
    xy=np.column_stack([np.r_[corner_x,x[cols]],np.r_[corner_y,y[rows]]])
    B=np.r_[cb,bed[mask]];H=np.r_[ch,thickness[mask]];center=nc+np.arange(n)
    triangles=np.stack([np.column_stack([center,corners[:,k],corners[:,(k+1)%4]]) for k in range(4)],axis=1).reshape(-1,3)
    points=xy[triangles];signed=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
    wrong=signed<0;triangles[wrong]=triangles[wrong][:,[0,2,1]]
    area=np.abs(signed)*.5
    all_edges=np.sort(np.concatenate([triangles[:,[0,1]],triangles[:,[1,2]],triangles[:,[2,0]]]),axis=1)
    edges,edge_counts=np.unique(all_edges,axis=0,return_counts=True)
    if np.any(edge_counts>2):raise ValueError('Nonmanifold mesh')
    boundary=edges[edge_counts==1]
    labels,components=ndimage.label(mask,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    source_volume=thickness[mask]*abs(dx*dy)
    mesh_pixel_volume=(area*np.mean(H[triangles],axis=1)).reshape(n,4).sum(axis=1)
    metrics=dict(source_pixels=n,nodes=len(xy),triangles=len(triangles),components=int(components),
        source_area_m2=float(n*abs(dx*dy)),mesh_area_m2=float(area.sum()),
        source_volume_m3=float(source_volume.sum()),mesh_volume_m3=float(mesh_pixel_volume.sum()),
        min_thickness_m=float(H.min()),max_thickness_m=float(H.max()),
        max_within_pixel_volume_change_m3=float(abs(mesh_pixel_volume-source_volume).max()),
        relative_l1_pixel_volume_change=float(abs(mesh_pixel_volume-source_volume).sum()/source_volume.sum()),
        source_centers_exact=True,diagonal_only_corner_connections=False,
        interpolated_geometry=True,scientific_acceptance=False)
    if not np.isclose(metrics['source_volume_m3'],metrics['mesh_volume_m3'],rtol=1e-12):raise ValueError('Volume transfer failure')
    return dict(xy=xy,triangles=triangles,bed=B,thickness=H,boundary_edges=boundary,
                pixel_rows=rows,pixel_cols=cols,pixel_center_nodes=center,pixel_corner_nodes=corners,
                source_pixel_volume_m3=source_volume,mesh_pixel_volume_m3=mesh_pixel_volume,
                pixel_component=labels[mask]),metrics


def synthetic():
    x=np.arange(-600.,601.,100.);y=np.arange(0.,1501.,100.);xx,yy=np.meshgrid(x,y)
    # Two arms converge at y=650 m; a common trunk continues downstream.
    center=np.maximum(650-yy,0)*.65
    mask=((abs(xx-center)<=150)|(abs(xx+center)<=150))&(yy<=650)
    mask|=(abs(xx)<=200)&(yy>650)
    H=100.+np.zeros_like(xx);B=3500-.045*yy
    return build(mask,x,y,B,H)
