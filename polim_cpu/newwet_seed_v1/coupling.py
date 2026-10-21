"""True joint BP/CN block Newton with bounded unaccepted new-wet recovery."""
import hashlib
import numpy as np
from scipy import sparse
from .client import Backend
from .seed import after_adopt
from polim_cpu.cn_flux_time_v1.transport import TransportFailure
from .structure import Cache,c_values
from .algebra import flux_matrix,auxiliary,fgmres,MAX_ARNOLDI,require_block_V
from polim_cpu.cn_flux_time_v1.mass import Mass as BaseMass,accepted
from polim_cpu.diffusion_predictor_v2 import predict
from polim_cpu.geometry_probe_v2.response import wet,freeze_masks,branch_events
from polim_cpu.geometry_coupled_v1.coupling import feasible
from polim_cpu.geometry_coupled_v1.backend import BudgetStop,LinearFailure
from polim_cpu.geometry_window_v1.ledger import preview
from .diagnostics import apply_parent,partition,branch_difference,remaining,PATH_KEYS

def identity(h,u,wet,alpha,beta):
    d=hashlib.sha256(b'v52-block-1')
    for a in (h,u,wet,alpha,beta):d.update(np.ascontiguousarray(a).tobytes())
    return d.hexdigest()

def matrices(z,r):
    n=len(z['bed']);a=sparse.csr_matrix((r['A_values'],z['csr_indices'],z['csr_indptr']),shape=(4*n,4*n))
    p=r['wet'].astype(float);a=(sparse.diags(p)@a@sparse.diags(p)+sparse.diags(1-p)).tocsr()
    return a

class Mass(BaseMass):
    def __init__(self,*a,inherited=None,**kw):
        super().__init__(*a,**kw);self.inherited=inherited;self.selection=False
        original=self.eq.evaluate
        def measured(h,u):
            key='fixed_speed_mass_evaluation' if self.selection else 'formal_mass_only_evaluation'
            with self.w.span(key):return original(h,u)
        self.eq.evaluate=measured
    def choose(self,h,u):
        """Explicit branch operation, NEVER called by J or a pure residual."""
        before=self.eq.revision
        self.selection=True
        try:
            with self.w.span('fixed_speed_and_branch_selection'):fixed,_=self.eq.limited(u,h)
        finally:self.selection=False
        return fixed,self.eq.revision!=before
    def pure(self,h,u,fixed):
        e=self.eq.evaluate(h,u);R=e['R'];source=float(e['source'].sum());export=float(e['oi'].sum())
        d=dict(local=e['local'],mass=e['mass'],physical_local=e['local'],physical_mass=e['mass'],
            fixed_point=None if fixed is None else float(self.area@abs(h-fixed)/max(self.eq.V0,1.)),physical_path=e['path']['physical'],
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
        self.serial=0;self.history=[];self.z=backend.z;self.parent_operators={};self.structure_cache=Cache();self.seed_enabled=getattr(backend,"seed_enabled",False);self.seed_seen=False
    def state(self,h,u,tag,fixed=None,locked=False,material_nodes=None):
        self.w.tick()
        if self.w.counts['pure_joint_residual']>=128:raise BudgetStop('128 complete joint evaluations exhausted before candidate work')
        lock_info=None
        if locked:
            self.mass.selection=True
            try:
                with self.w.span('seed_fixed_branch_transport'):fixed,aux=self.mass.eq.fixed_newton(u,h)
                lock_info=dict(ok=bool(aux['path']['physical']),reason='fixed-alpha/beta path physical' if aux['path']['physical'] else 'would require limiter update')
            except TransportFailure as exc:fixed=None;lock_info=dict(ok=False,reason=str(exc))
            finally:self.mass.selection=False
            changed=False
        elif fixed is None:fixed,changed=self.mass.choose(h,u)
        else:changed=False
        with self.w.span('joint_residual'):
            r=self.b.pure(h,u,material_nodes);d,e=self.mass.pure(h,u,fixed)
        s=dict(H=h.copy(),u=u.copy(),F=r['F'],force=r['force'],wet=r['wet'],d=d,e=e,revision=self.mass.eq.revision)
        s['bp']=float(np.linalg.norm(s['F'])/max(np.linalg.norm(s['force']),1.));s['point']=identity(h,u,s['wet'],self.mass.eq.alpha,self.mass.eq.beta)
        self.serial+=1;name=f'state-{self.serial:03d}-{tag}'
        self.w.save(name,H=h,u=u,F=s['F'],force=s['force'],wet=s['wet'],R=e['R'],alpha=self.mass.eq.alpha,beta=self.mass.eq.beta,
            actual_nodes=e['source'],unmet=e['U'],internal_volume=e['qi'],boundary_volume=e['oi'])
        s['name']=name;s['locked']=lock_info
        if r.get('material'):
            with self.w.span('seed_material_output'):self.w.save(name+'-material',**r['material'])
        with self.w.span('diagnostic_recording'):
            self.w.save(name+'-path',**{key:e['path'][key] for key in PATH_KEYS},Q1=e['q1'],O1=e['o1'])
            self.w.event('joint-state-region',name=name,**partition(s['F'],s['force']),budget_remaining=remaining(self.w))
        self.w.event('joint-state',name=name,point=s['point'],bp=s['bp'],branch_selection_changed=changed,**d)
        return s,changed
    def direction(self,s,recovery=False):
        w=self.w;w.count('joint_directions',16);number=w.counts['joint_directions'];w.location=f'joint direction {number}'
        if recovery:w.count('recovery_directions')
        if s['revision']!=self.mass.eq.revision:raise LinearFailure('Stale limiter branch before direction')
        with w.span('J_build'):
            r=self.b.blocks(s['H'],s['u'])
            with w.span('A_existing_projection'):A=matrices(self.z,r)
            with w.span('diagnostic_recording'):
                w.save(f'direction-{number:02d}-newwet-material',**r['material'])
            # The bind yields one counted original F evaluation; attach the same
            # pure mass evaluation, never a fixed-speed solve, for its audit.
            d,e=self.mass.pure(s['H'],s['u'],s['H'])
            mismatch=np.linalg.norm(r['F']-s['F'])/max(np.linalg.norm(s['force']),1.)
            w.event('build-residual-consistency',force_scaled=float(mismatch),gate=1e-10)
            if mismatch>1e-10:raise LinearFailure('Pure F differs from original F')
            with w.span('C_current_coefficients'):cv,_=c_values(self.ad,s['H'],s['u'],self.mass.eq.weights(s['e']),r['wet'][::4])
            with w.span('BC_structure_update'):B,C,cache_info=self.structure_cache.update(self.z,self.ad,r['B_local'],cv)
            w.event('BC-structure-cache',direction=number,**cache_info,B_nnz=B.nnz,C_nnz=C.nnz)
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
            if calls[0]>MAX_ARNOLDI:raise BudgetStop('24 block preconditions per direction')
            require_block_V(w)
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
        self.parent_operators[s['point']]=dict(blocks=(A,B,C,D),G=G.copy(),sr=sr.copy(),sx=sx.copy(),
            number=number,revision=s['revision'],alpha=self.mass.eq.alpha.copy(),beta=self.mass.eq.beta.copy())
        w.event('direction-budget',number=number,parent_state=s['name'],remaining=remaining(w),work=w.record())
        return dx[:nu],dx[nu:]
    def trial(self,s,du,dh,k,tag):
        alpha=2.**(-k);h,hit=feasible(s['H'],dh,alpha);u=s['u']+alpha*du;u[~np.repeat(wet(h,self.bed),4)]=0.
        parent=self.parent_operators[s['point']]
        if s['revision']!=self.mass.eq.revision or not np.array_equal(parent['alpha'],self.mass.eq.alpha) or not np.array_equal(parent['beta'],self.mass.eq.beta):
            raise LinearFailure('Diagnostic parent J identity does not match trial parent')
        displacement=np.r_[u-s['u'],h-s['H']]
        # Exactly one extra immutable parent J action; before branch selection.
        with self.w.span('diagnostic_J_action'):Js=apply_parent(parent['blocks'],displacement)
        self.w.count('paired_backtrack_candidates');self.w.count('shared_direction_candidates')
        material_nodes=np.flatnonzero(wet(h,self.bed)&~s['wet'][::4])
        t,changed=self.state(h,u,tag,material_nodes=material_nodes)
        with self.w.span('diagnostic_recording'):
            Gtrial=np.r_[t['F'],t['e']['R']];elin=parent['G']+Js;emodel=Gtrial-parent['G']-Js
            self.w.save(f'actual-model-{self.serial:03d}',s=displacement,Js=Js,G_parent=parent['G'],G_trial=Gtrial,
                e_linear=elin,e_model=emodel,residual_scaling=parent['sr'],variable_scaling=parent['sx'],
                parent_point=np.array(s['point']),child_point=np.array(t['point']),parent_alpha=parent['alpha'],
                parent_beta=parent['beta'],child_alpha=self.mass.eq.alpha,child_beta=self.mass.eq.beta)
            nu=len(s['u'])
            blocks={label:dict(parent=float(np.linalg.norm(parent['G'][sl])),actual=float(np.linalg.norm(Gtrial[sl])),
                linear=float(np.linalg.norm(elin[sl])),model=float(np.linalg.norm(emodel[sl])),
                closure=float(np.linalg.norm(Gtrial[sl]-elin[sl]-emodel[sl]))) for label,sl in [('F',slice(0,nu)),('R',slice(nu,None))]}
            self.w.event('actual-model',direction=parent['number'],parent=s['name'],child=t['name'],alpha=alpha,
                different_equation=changed,branch=branch_difference(s,t),blocks=blocks,budget_remaining=remaining(self.w))
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
            du,dh=self.direction(s);sc=scales(s,self.mass);taken=None;restart=None;recovery_used=False;ordinary_adoption=False
            for k in range(8):
                t,changed,new,alpha=self.trial(s,du,dh,k,'ordinary')
                if changed:
                    restart,_=self.state(s['H'],s['u'],'parent-new-limiter')
                    self.w.count('invalidated_directions')
                    self.w.event('old-linearization-invalidated',reason='alpha/beta changed; discard direction and relinearize parent',old=s['revision'],new=restart['revision'])
                    break
                physical=t['d']['physical_path'] and t['d']['all_finite'] and t['d']['H_min']>=0
                good=physical and (merit(t,sc)<=(1-1e-4*alpha)*merit(s,sc) or accepted(t['d'],t['bp']))
                self.w.event('parent-descent',alpha=alpha,base_merit=merit(s,sc),trial_merit=merit(t,sc),adopted=good)
                if good:taken=t;ordinary_adoption=True;break
                if physical and new and not recovery_used:
                    recovery_used=True;taken,restart=self.recover(t,s,alpha,sc)
                    if taken is not None or restart is not None:break
            if restart is not None:s=restart;continue
            if taken is None:raise LinearFailure('Paired backtracking and bounded new-wet recovery exhausted')
            previous=s;s=taken;parent_iteration+=1;self.w.count('internal_iterate_adopted')
            if ordinary_adoption:s=after_adopt(self,previous,s,k+1)

def attempt(backend,adapter,h_old,u_old,smb,*,end_year,ledger,step_dt,inherited=None):
    w=backend.w
    with w.span('predictor'):h,diag,parts=predict(h_old,backend.mesh['bed'],smb,step_dt,adapter.rates(u_old))
    u=u_old.copy();u[~np.repeat(wet(h,backend.mesh['bed']),4)]=0.
    w.save('predictor',H=h,u=u,**parts);w.event('predictor',diagnostics=diag)
    return Process(backend,adapter,h_old,u_old,smb,step_dt,ledger,inherited).solve(h,u)
