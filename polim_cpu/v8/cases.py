"""Frozen standard geometries, no real-glacier geometry inference."""
from pathlib import Path
import numpy as np
from .boundary import edge_geometry

REFERENCE=Path(__file__).resolve().parents[1]/'reference/ismip-hom-2008'


def rectangle(x,y,bed,thickness):
    xx,yy=np.meshgrid(x,y,indexing='ij');xy=np.column_stack([xx.ravel(),yy.ravel()])
    nx=len(x)-1;ny=len(y)-1;tri=[]
    for i in range(nx):
        for j in range(ny):
            a=i*(ny+1)+j;b=(i+1)*(ny+1)+j;c=b+1;d=a+1
            tri.extend([[a,b,c],[a,c,d]])
    return dict(xy=xy,triangles=np.array(tri,dtype=np.int32),bed=np.broadcast_to(bed(xy),(len(xy),)).copy(),
                thickness=np.broadcast_to(thickness(xy),(len(xy),)).copy())


def periodic(mesh,x=True,y=True):
    xy=mesh['xy'];edges,_,_=edge_geometry(mesh);tags=[];pairs=[]
    for axis,enabled in enumerate((x,y)):
        if not enabled:continue
        lo=np.flatnonzero(xy[:,axis]==xy[:,axis].min());hi=np.flatnonzero(xy[:,axis]==xy[:,axis].max())
        lo=lo[np.argsort(xy[lo,1-axis])];hi=hi[np.argsort(xy[hi,1-axis])]
        if not np.array_equal(xy[lo,1-axis],xy[hi,1-axis]):raise ValueError('Unmatched periodic edges')
        if not np.allclose(mesh['thickness'][lo],mesh['thickness'][hi],rtol=1e-12,atol=1e-9):raise ValueError('Periodic H mismatch')
        pairs.extend(zip(lo.tolist(),hi.tolist()))
    for edge in edges:
        isperiodic=any(enabled and np.ptp(xy[edge,axis])==0 for axis,enabled in enumerate((x,y)))
        tags.append(dict(kind='periodic' if isperiodic else 'natural'))
    return dict(basal='no-slip',edges=tags,periodic_pairs=pairs)


def case(name):
    if name=='slab':
        slope=-np.tan(np.deg2rad(.5));H=1000.
        mesh=rectangle(np.linspace(0,5000,5),np.linspace(0,5000,5),lambda p:slope*p[:,0]-H,lambda p:H)
        return mesh,periodic(mesh)
    if name=='hom-a020':
        L=20000.;slope=-np.tan(np.deg2rad(.5))
        H=lambda p:1000.-500.*np.sin(2*np.pi*p[:,0]/L)*np.sin(2*np.pi*p[:,1]/L)
        mesh=rectangle(np.linspace(0,L,41),np.linspace(0,L,41),lambda p:slope*p[:,0]-H(p),H)
        return mesh,periodic(mesh)
    if name=='hom-e000':
        d=np.loadtxt(REFERENCE/'arolla100.dat');x=np.linspace(0,5000,101)
        # Width is a computational periodic extrusion, not a physical side wall.
        mesh=rectangle(x,[0.,100.],lambda p:np.interp(p[:,0],d[:,0],d[:,1]),
                       lambda p:np.interp(p[:,0],d[:,0],d[:,2]-d[:,1]))
        return mesh,periodic(mesh,x=False,y=True)
    raise ValueError(name)


def surface_profile(name,mesh,u):
    xy=mesh['xy'];y=5000. if name=='hom-a020' else 0.
    select=np.flatnonzero(xy[:,1]==y);select=select[np.argsort(xy[select,0])]
    return np.column_stack([xy[select,0],np.linalg.norm(u[select,-1,:],axis=1)])


def slab_surface_exact(A=1e-16,rho=910.,g=9.81):
    s=-np.tan(np.deg2rad(.5));H=1000.
    return -.5*A*(rho*g*s)**3*H**4/(1+4*s*s)**2
