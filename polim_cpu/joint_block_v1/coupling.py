"""True joint BP/CN block Newton with bounded unaccepted new-wet recovery."""
import hashlib
import numpy as np
from scipy import sparse
from .client import Backend
from .algebra import flux_matrix,auxiliary,fgmres
from polim_cpu.cn_flux_time_v1.mass import Mass as BaseMass,accepted
from polim_cpu.diffusion_predictor_v2 import predict
from polim_cpu.geometry_probe_v2.response import wet,freeze_masks,branch_events
from polim_cpu.geometry_coupled_v1.coupling import feasible
from polim_cpu.geometry_coupled_v1.backend import BudgetStop,LinearFailure
from polim_cpu.geometry_window_v1.ledger import preview

def identity(h,u,wet,alpha,beta):
    d=hashlib.sha256(b'v52-block-1')
    for a in (h,u,wet,alpha,beta):d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()

def matrices(z,r):
    n=len(z['bed']);a=sparse.csr_matrix((r['A_values'],z['csr_indices'],z['csr_indptr']),shape=(4*n,4*n))
    p=r['wet'].astype(float);a=(sparse.diags(p)@a@sparse.diags(p)+sparse.diags(1-p)).tocsr()
    ids=np.broadcast_to(z['ids'][:,:,None],r['B_local'].shape).ravel()
    cols=np.broadcast_to(z['tri'][:,None,:],r['B_local'].shape).ravel()
    b=sparse.coo_matrix((r['B_local'].ravel(),(ids,cols)),shape=(4*n,n)).tocsr()
    return a,b

class Mass(BaseMass):
    def __init__(self,*a,inherited=None,**kw):super().__init__(*a,**kw);self.inherited=inherited
    def choose(self,h,u):
        """Explicit branch operation, NEVER called by J or a pure residual."""
        before=self.eq.revision
        with self.w.span('fixed_speed_and_branch_selection'):fixed,_=self.eq.limited(u,h)
        return fixed,self.eq.revision!=before
    def pure(self,h,u,fixed):
        e=self.eq.evaluate(h,u);R=e['R'];source=float(e['source'].sum());export=float(e['oi'].sum())
        d=dict(local=e['local'],mass=e['mass'],physical_local=e['local'],physical_mass=e['mass'],
            fixed_point=float(self.area@abs(h-fixed)/max(self.eq.V0,1.)),physical_path=e['path']['physical'],
            active_equal=bool(np.array_equal(h==0,fixed==0)),source=source,export=export,unmet=float(e['U'].sum()),
            potential=self.eq.dt*float(self.area@self.eq.smb),H_min=float(h.min()),active_count=int((h==0).sum()),
            all_finite=bool(all(np.isfinite(x).all() for x in (h,u,R,e['source'],e['U']))),scale=e['scale'],
            residual_L1_m3=float(abs(R).sum()),residual_sum_m3=float(R.sum()),limiter_revision=self.eq.revision,
            BE_edges=int((self.eq.alpha==0).sum()),BE_outlets=int((self.eq.beta==0).sum()))
        d.update(preview(self.ledger,float(self.area@h),source,export))
        if self.inherited is not None:
            other=preview(self.inherited,float(self.area@h),source,export)
            d.update(annual_cumulative_mass_relative=d['cumulative_mass_relative'],inherited_cumulative_mass_relative=other['cumulative_mass_relative'])
            d['cumulative_mass_relative']=max(d['cumulative_mass_relative'],other['cumulative_mass_relative'])
        return d,e

def scales(s,mass):
    return dict(force=max(float(np.linalg.norm(s['force'])),1.),volume=max(mass.eq.V0,1.),
        local=s['e']['scale'],mass=max(mass.eq.V0,abs(s['d']['source']),abs(s['d']['export']),1.))

def merit(s,scale):
    # Same five v51 components, with parent denominators frozen during descent.
    return max(float(np.linalg.norm(s['F']))/(1e-6*scale['force']),float(abs(s['e']['R']).sum())/(1e-8*scale['local']),
        abs(float(s['e']['R'].sum()))/(1e-10*scale['mass']),s['d']['cumulative_mass_relative']/1e-10,s['d']['fixed_point']/1e-8)

class Process:
    def __init__(self,backend,adapter,h0,u0,smb,dt,ledger,inherited):
        self.b=backend;self.w=backend.w;self.ad=adapter;self.bed=backend.mesh['bed'];self.mass=Mass(adapter,h0,u0,smb,self.w,dt,ledger,inherited=inherited)
        self.serial=0;self.history=[];self.z=backend.z
    def state(self,h,u,tag,fixed=None):
        if fixed is None:fixed,changed=self.mass.choose(h,u)
        else:changed=False
        with self.w.span('joint_residual'):
            r=self.b.pure(h,u);d,e=self.mass.pure(h,u,fixed)
        s=dict(H=h.copy(),u=u.copy(),F=r['F'],force=r['force'],wet=r['wet'],d=d,e=e,revision=self.mass.eq.revision)
        s['bp']=float(np.linalg.norm(s['F'])/max(np.linalg.norm(s['force']),1.));s['point']=identity(h,u,s['wet'],self.mass.eq.alpha,self.mass.eq.beta)
        self.serial+=1;name=f'state-{self.serial:03d}-{tag}'
        self.w.save(name,H=h,u=u,F=s['F'],force=s['force'],wet=s['wet'],R=e['R'],alpha=self.mass.eq.alpha,beta=self.mass.eq.beta,
            actual_nodes=e['source'],unmet=e['U'],internal_volume=e['qi'],boundary_volume=e['oi'])
        self.w.event('joint-state',name=name,point=s['point'],bp=s['bp'],branch_selection_changed=changed,**d)
        return s,changed
    def direction(self,s,recovery=False):
        w=self.w;w.count('joint_directions',8);number=w.counts['joint_directions'];w.location=f'joint direction {number}'
        if recovery:w.count('recovery_directions')
        if s['revision']!=self.mass.eq.revision:raise LinearFailure('Stale limiter branch before direction')
        with w.span('J_build'):
            r=self.b.blocks(s['H'],s['u']);A,B=matrices(self.z,r)
            # The bind yields one counted original F evaluation; attach the same
            # pure mass evaluation, never a fixed-speed solve, for its audit.
            d,e=self.mass.pure(s['H'],s['u'],s['H'])
            mismatch=np.linalg.norm(r['F']-s['F'])/max(np.linalg.norm(s['force']),1.)
            w.event('build-residual-consistency',force_scaled=float(mismatch),gate=1e-10)
            if mismatch>1e-10:raise LinearFailure('Pure F differs from original F')
            with w.span('C_local_assembly'):C=flux_matrix(self.ad,s['H'],s['u'],self.mass.eq.weights(s['e']),r['wet'][::4])
            with w.span('D_assembly'):D=self.mass.eq.matrix(s['u'],s['e']).tocsr()
            P,lu,info=auxiliary(A,B,C,D,w)
        sc=scales(s,self.mass);n=len(s['H']);nu=4*n;point=r['point'];token=s['point']
        sr=np.r_[np.full(nu,1/(sc['force']*1e-6)),np.full(n,np.sqrt(n)/(sc['volume']*1e-8))]
        sx=np.r_[np.full(nu,max(float(abs(s['u']).max()),1.)),np.full(n,max(float(s['H'].max()),1.))]
        G=np.r_[s['F'],s['e']['R']];calls=[0]
        def check():
            if token!=identity(s['H'],s['u'],s['wet'],self.mass.eq.alpha,self.mass.eq.beta):raise LinearFailure('J changed during linear solve')
        def action(v):
            check()
            with w.span('J_action'):
                q=sx*v;return sr*np.r_[A@q[:nu]+B@q[nu:],C@q[:nu]+D@q[nu:]]
        def pre(v):
            check();calls[0]+=1
            if calls[0]>16:raise BudgetStop('16 block preconditions per direction')
            q=v/sr
            with w.span('block_precondition_action'):
                zu=self.b.V(q[:nu],point)
                with w.span('C_action'):right=q[nu:]-C@zu
                with w.span('P_LU_apply'):yh=lu.solve(right)
                with w.span('B_action'):by=B@yh
                yu=zu-self.b.V(by,point)
            return np.r_[yu,yh]/sx
        w.save(f'direction-{number:02d}-input',H=s['H'],u=s['u'],F=s['F'],R=s['e']['R'],residual_scaling=sr,variable_scaling=sx,wet=s['wet'],alpha=self.mass.eq.alpha,beta=self.mass.eq.beta)
        with w.span('joint_FGMRES'):x,result=fgmres(action,pre,-sr*G,w,f'direction-{number:02d}-linear')
        dx=sx*x;w.save(f'direction-{number:02d}',du=dx[:nu],dH=dx[nu:])
        w.event('joint-direction',number=number,recovery=recovery,point=token,GPU_point=point,precondition_actions=calls[0],V_cycles=2*calls[0],P=info,**result)
        return dx[:nu],dx[nu:]
    def trial(self,s,du,dh,k,tag):
        alpha=2.**(-k);h,hit=feasible(s['H'],dh,alpha);u=s['u']+alpha*du;u[~np.repeat(wet(h,self.bed),4)]=0.
        self.w.count('paired_backtrack_candidates');t,changed=self.state(h,u,tag)
        new=t['wet'][::4]&~s['wet'][::4]
        self.w.save(f'paired-displacement-{self.serial:03d}',alpha=np.array(alpha),deltaH=h-s['H'],delta_u=u-s['u'],projected=hit,new_wet=new)
        self.w.event('paired-trial',alpha=alpha,new_wet_nodes=np.flatnonzero(new).tolist(),projected_nodes=np.flatnonzero(hit).tolist(),
            velocity_increment_norm=float(np.linalg.norm(u-s['u'])),branch=branch_events(self.ad,h,u,self.bed,freeze_masks(self.ad,s['H'],s['u'],self.bed)),
            limiter_changed=changed,projection_booked_as_SMB=False)
        return t,changed,bool(new.any()),alpha
    def recover(self,child,parent,outer_alpha,scale):
        self.w.count('new_wet_recovery_started');base=child
        with self.w.span('new_wet_recovery'):
            for j in range(2):
                du,dh=self.direction(base,True);adopted=None
                for k in range(8):
                    t,changed,new,alpha=self.trial(base,du,dh,k,'recovery')
                    if changed:
                        # Discard this direction, never continue its backtracking
                        # after a new discrete equation has been selected.
                        p,_=self.state(parent['H'],parent['u'],'parent-new-limiter')
                        return None,p
                    ok=t['d']['physical_path'] and t['d']['all_finite'] and t['d']['H_min']>=0
                    parent_pass=ok and (merit(t,scale)<=(1-1e-4*outer_alpha)*merit(parent,scale) or accepted(t['d'],t['bp']))
                    self.w.event('recovery-comparison',internal=j+1,alpha=alpha,parent_merit=merit(parent,scale),trial_merit=merit(t,scale),adopt_to_parent=parent_pass)
                    if parent_pass:self.w.count('new_wet_recovery_success');return t,None
                    if ok and merit(t,scale)<=(1-1e-4*alpha)*merit(base,scale):adopted=t;break
                if adopted is None:break
                base=adopted
        self.w.count('new_wet_recovery_failed');return None,None
    def solve(self,h,u):
        s,_=self.state(h,u,'initial');parent_iteration=0
        while True:
            self.w.location='parent '+str(parent_iteration);row=dict(joint_updates=self.w.counts['joint_directions'],bp=s['bp'],**s['d']);self.history.append(row)
            if accepted(s['d'],s['bp']):
                final,changed=self.state(s['H'],s['u'],'final')
                if not changed and accepted(final['d'],final['bp']):
                    self.mass.save('accepted-state',final['H'],final['u'])
                    self.w.save('accepted-momentum',H=final['H'],u=final['u'],F=final['F'],force=final['force'],wet=final['wet'])
                    return dict(status='accepted',accepted=True,history=self.history,final=final['d'])
                s=final;continue
            du,dh=self.direction(s);sc=scales(s,self.mass);taken=None;restart=None;recovery_used=False
            for k in range(8):
                t,changed,new,alpha=self.trial(s,du,dh,k,'ordinary')
                if changed:
                    restart,_=self.state(s['H'],s['u'],'parent-new-limiter')
                    self.w.event('old-linearization-invalidated',reason='alpha/beta changed; discard direction and relinearize parent',old=s['revision'],new=restart['revision'])
                    break
                physical=t['d']['physical_path'] and t['d']['all_finite'] and t['d']['H_min']>=0
                good=physical and (merit(t,sc)<=(1-1e-4*alpha)*merit(s,sc) or accepted(t['d'],t['bp']))
                self.w.event('parent-descent',alpha=alpha,base_merit=merit(s,sc),trial_merit=merit(t,sc),adopted=good)
                if good:taken=t;break
                if physical and new and not recovery_used:
                    recovery_used=True;taken,restart=self.recover(t,s,alpha,sc)
                    if taken is not None or restart is not None:break
            if restart is not None:s=restart;continue
            if taken is None:raise LinearFailure('Paired backtracking and bounded new-wet recovery exhausted')
            s=taken;parent_iteration+=1;self.w.count('internal_iterate_adopted')

def attempt(backend,adapter,h_old,u_old,smb,*,end_year,ledger,step_dt,inherited=None):
    w=backend.w
    with w.span('predictor'):h,diag,parts=predict(h_old,backend.mesh['bed'],smb,step_dt,adapter.rates(u_old))
    u=u_old.copy();u[~np.repeat(wet(h,backend.mesh['bed']),4)]=0.
    w.save('predictor',H=h,u=u,**parts);w.event('predictor',diagnostics=diag)
    return Process(backend,adapter,h_old,u_old,smb,step_dt,ledger,inherited).solve(h,u)
