"""Float64 PoLIM classic momentum subset, x-z, metres and years.

Grid/interpolation, momentum stencils, viscosity, and vertical velocity derive
from PoLIM cc5d08541ab67987f7652adcdc7a734530580e82. See LICENSE/NOTICE.
The bounded relaxed Picard controller and conservative evolution are new.
"""
from dataclasses import dataclass
from typing import Protocol
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from . import _stencils


@dataclass(frozen=True)
class Parameters:
    rho: float = 910.0
    gravity: float = 9.81
    glen_n: float = 3.0
    glen_a: float = 1e-16  # Pa^-n year^-1
    strain_regularization: float = 1e-30  # year^-2 (added to squared strain)
    basal: str = 'zero'
    beta2: float = 1000.0  # Pa year m^-1
    flowband: bool = False  # constant width only
    layers: int = 21
    relaxation: float = 0.5
    max_iterations: int = 150
    relative_tolerance: float = 1e-6


class LinearBackend(Protocol):
    name: str
    def solve(self, matrix, rhs): ...


class ScipyBackend:
    name = 'scipy-cpu-float64'
    def solve(self, matrix, rhs):
        return spsolve(matrix.tocsc(), rhs)


def to_stagger(a):
    a = np.asarray(a, dtype=np.float64)
    return np.concatenate((a[..., :1], (a[..., :-1]+a[..., 1:])/2, a[..., -1:]), axis=-1)


def to_main(a):
    return np.concatenate((a[..., :1], (a[..., 1:-2]+a[..., 2:-1])/2, a[..., -1:]), axis=-1)


def dx_stagger(a, dx):
    d = np.empty_like(a)
    d[..., 2:-2] = (to_main(a)[..., 2:-1]-to_main(a)[..., 1:-2])/dx
    d[..., 0] = (-8*a[..., 0]+9*a[..., 1]-a[..., 2])/(3*dx)
    d[..., 1] = (-4*a[..., 0]+3*a[..., 1]+a[..., 2])/(3*dx)
    d[..., -2] = (-a[..., -3]-3*a[..., -2]+4*a[..., -1])/(3*dx)
    d[..., -1] = (8*a[..., -1]-9*a[..., -2]+a[..., -3])/(3*dx)
    return d


def dz(a, spacing):
    return np.gradient(a, spacing, axis=0, edge_order=2)


@dataclass
class Grid:
    x: np.ndarray
    bed: np.ndarray
    thickness: np.ndarray
    width: np.ndarray
    layers: int

    def __post_init__(self):
        for key in ('x', 'bed', 'thickness', 'width'):
            setattr(self, key, np.asarray(getattr(self, key), dtype=np.float64).copy())
        if self.x.ndim != 1 or self.x.size < 5:
            raise ValueError('At least five equally spaced main-grid points are required')
        if any(a.shape != self.x.shape or not np.isfinite(a).all() for a in (self.x,self.bed,self.thickness,self.width)):
            raise ValueError('Geometry arrays must be finite, one-dimensional, and equal in size')
        self.dx = float(self.x[1]-self.x[0])
        if self.dx <= 0 or not np.allclose(np.diff(self.x),self.dx,rtol=1e-12,atol=1e-12):
            raise ValueError('x must increase uniformly in metres')
        if np.any(self.thickness <= 0) or np.any(self.width <= 0):
            raise ValueError('v0 requires strictly positive thickness and width; no artificial ice floor')
        if not np.all(self.width == self.width[0]):
            raise NotImplementedError('v0 supports constant width only')
        if self.layers < 3:
            raise ValueError('At least three vertical levels required')
        self.zeta = np.linspace(0,1,self.layers)
        self.dzeta = 1/(self.layers-1)
        self.hs = to_stagger(self.thickness)
        self.bs = to_stagger(self.bed)
        self.xs = to_stagger(self.x)
        self.slope = np.diff(self.bed+self.thickness)/self.dx
        self.slope = np.r_[self.slope[0]/2,self.slope,self.slope[-1]/2]
        b = np.diff(self.bed)/self.dx
        self.bed_slope = np.r_[b[0]/2,b,b[-1]/2]
        self.metric = -(dx_stagger(self.bs,self.dx)[None,:]+self.zeta[:,None]*dx_stagger(self.hs,self.dx)[None,:])/self.hs
        # Preserve the upstream full-domain loop's uncomputed last column.
        self.metric[:,-1] = 0
        self.cell_width = np.full(self.x.size,self.dx)
        self.cell_width[[0,-1]] *= .5

    @property
    def volume(self):
        return float(np.sum(self.thickness*self.width*self.cell_width))


def _pad(a):
    return np.pad(a, [(1,0)]*np.ndim(a))


def assemble(grid, viscosity, p):
    """Sparse A u = b; u flattens in MATLAB column-major order."""
    N, Ms = viscosity.shape
    if viscosity.shape != grid.metric.shape or not np.isfinite(viscosity).all() or np.any(viscosity <= 0):
        raise ValueError('Viscosity must be positive finite [layers, staggered points]')
    if p.basal not in ('zero','linear'):
        raise NotImplementedError('Supported basal laws: zero, linear (upstream LFlaw)')
    if p.beta2 <= 0 or not np.isfinite(p.beta2):
        raise ValueError('beta2 must be finite and positive')
    eta, d, H, dx, dzeta = viscosity, grid.metric, grid.hs, grid.dx, grid.dzeta
    ex, ez = dx_stagger(eta,dx), dz(eta,dzeta)
    ddx, ddz = dx_stagger(d,dx), dz(d,dzeta)
    L1 = _pad(4*eta)
    L2 = _pad(eta*(4*d*d+1/H**2))
    L3 = _pad(8*d*eta)
    L4 = _pad(4*eta*ddx+4*eta*d*ddz+4*d*ex+ez*(4*d*d+1/H**2))
    L5 = _pad(4*ex+4*d*ez)
    L6 = _pad(-eta/grid.width[0]**2 if p.flowband else np.zeros_like(eta))
    A = sparse.lil_matrix((N*Ms+1,N*Ms+1), dtype=np.float64)
    rhs = np.zeros(N*Ms+1)
    for i in range(2,Ms):
        for j in range(2,N):
            k = (i-1)*N+j
            rhs[k] = p.rho*p.gravity*grid.slope[i-1]
            _stencils.interior(A,L1,L2,L3,L4,L5,L6,i,j,k,N,Ms,dx,dzeta)
    ls = _pad(4*H*grid.slope/(1-4*H*d[-1]*grid.slope)*dzeta/dx)
    zeros = np.zeros(Ms+1)
    _stencils.surface(A,L1,L2,L3,L4,L5,ls,zeros,zeros,N,Ms,dx,dzeta)
    for i in range(2,Ms):
        rhs[i*N] = p.rho*p.gravity*grid.slope[i-1]
        if p.basal == 'zero':
            A[(i-1)*N+1,(i-1)*N+1] = 1
        else:
            rhs[(i-1)*N+1] = p.rho*p.gravity*grid.slope[i-1]
    if p.basal == 'linear':
        denom = 1-4*H*grid.bed_slope*d[0]
        lb1 = _pad(4*dzeta*H*grid.bed_slope/denom/dx)
        lb2 = _pad(2*dzeta*H*p.beta2/eta[0]/denom)
        _stencils.basal_linear(A,L1,L2,L3,L4,L5,L6,lb1,lb2,N,Ms,dx,dzeta)
    for k in list(range(1,N+1))+list(range((Ms-1)*N+1,Ms*N+1)):
        A[k,k] = 1
        rhs[k] = 0
    return A.tocsr()[1:,1:], rhs[1:]


def rheology(grid, u, p):
    ux = dx_stagger(u,grid.dx)+grid.metric*dz(u,grid.dzeta)
    uz = dz(u,grid.dzeta)/grid.hs
    strain2 = ux**2+uz**2/4
    if p.flowband:
        strain2 += (u/grid.width[0])**2/4
    eta = .5*p.glen_a**(-1/p.glen_n)*(strain2+p.strain_regularization)**((1-p.glen_n)/(2*p.glen_n))
    return eta, to_main(4*eta*strain2)


def vertical_velocity(grid,u_s):
    u = to_main(u_s)
    derivative = np.gradient(u,grid.dzeta,axis=0,edge_order=1)
    integral = np.diff(u_s,axis=1)/grid.cell_width + to_main(grid.metric)*derivative
    w_vs = u[0]*to_main(grid.bed_slope)-grid.thickness*grid.dzeta*(np.cumsum(integral,axis=0)-integral[:1]/2)
    w = np.vstack((u[0]*to_main(grid.bed_slope),(w_vs[:-1]+w_vs[1:])/2))
    return w


def solve_velocity(grid,p,u_initial=None,backend=None):
    backend = backend or ScipyBackend()
    if not (p.glen_a>0 and p.glen_n>0 and p.strain_regularization>0 and 0<p.relaxation<=1 and p.max_iterations>=1 and p.relative_tolerance>0):
        raise ValueError('Invalid rheology or nonlinear-solver parameters')
    u = np.zeros_like(grid.metric) if u_initial is None else np.array(u_initial,copy=True)
    if u.shape != grid.metric.shape or not np.isfinite(u).all():
        raise ValueError('Invalid initial velocity')
    eta = np.full_like(u,1e13/31556926) if u_initial is None else rheology(grid,u,p)[0]
    history=[]
    for iteration in range(1,p.max_iterations+1):
        A,b = assemble(grid,eta,p)
        candidate = backend.solve(A,b).reshape(u.shape,order='F')
        if not np.isfinite(candidate).all():
            raise FloatingPointError('Non-finite momentum solution')
        change = float(np.linalg.norm(candidate-u)/max(np.linalg.norm(candidate),1e-30))
        u += p.relaxation*(candidate-u)
        eta,heat = rheology(grid,u,p)
        history.append({'iteration':iteration,'relative_change':change})
        if change <= p.relative_tolerance:
            break
    A,b = assemble(grid,eta,p)
    residual = float(np.linalg.norm(A@u.ravel(order='F')-b)/max(np.linalg.norm(b),1e-30))
    return {'u':u,'w':vertical_velocity(grid,u),'viscosity':eta,'strain_heat':heat,
            'converged':change<=p.relative_tolerance,'relative_residual':residual,
            'iterations':len(history),'iteration_history':history,'backend':backend.name}


def evolve(grid,u,dt_years,smb=0.0):
    """New conservative explicit finite-volume step, closed outer faces.

    Interior staggered-face fluxes are depth-integrated higher-order velocity.
    This is not a port of PoLIM's thickness diffusion scheme.
    """
    flux = np.trapz(u,grid.zeta,axis=0)*grid.hs*grid.width[0]
    flux[[0,-1]] = 0
    rate = np.broadcast_to(np.asarray(smb,dtype=float),grid.thickness.shape)-np.diff(flux)/(grid.width*grid.cell_width)
    if not np.isfinite(rate).all() or not np.isfinite(dt_years) or dt_years<=0:
        raise ValueError('Invalid timestep or surface mass balance')
    speed=float(np.max(np.abs(u)))
    dt=min(dt_years,.25*grid.dx/max(speed,1e-30))
    if np.any(rate<0):
        dt=min(dt,float(np.min(.8*grid.thickness[rate<0]/-rate[rate<0])))
    h=grid.thickness+dt*rate
    if np.any(h<=0):
        raise FloatingPointError('Thickness became nonpositive; no clipping performed')
    updated=Grid(grid.x,grid.bed,h,grid.width,grid.layers)
    source=float(np.sum(np.broadcast_to(smb,h.shape)*grid.width*grid.cell_width)*dt)
    return updated,dt,updated.volume-grid.volume-source
