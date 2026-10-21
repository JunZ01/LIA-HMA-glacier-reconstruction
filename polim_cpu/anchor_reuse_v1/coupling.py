"""Current true residual, explicitly lagged anchor response, bounded refresh."""
import numpy as np
from .baseline import (Process as BaseProcess,Backend,matrices,scales,merit,accepted,predict,wet,
                       LinearFailure,BudgetStop,feasible,freeze_masks,branch_events,identity)
from .algebra import auxiliary,fgmres,MAX_ARNOLDI,require_block_V
from .structure import c_values
from .policy import refresh_reasons,model_quality,may_retry
from .diagnostics import apply_parent,branch_difference,remaining

class Process(BaseProcess):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw);self.anchor=None;self.failed_refreshes=set();self.force_refresh=None;self.last_direction_info={}

    def build_anchor(self,s,reasons):
        w=self.w;w.count('anchor_builds',12);number=w.counts['anchor_builds'];old=self.anchor
        if old is not None:
            if old['point']==s['point']:raise LinearFailure('Duplicate anchor rebuild at same scientific state')
            w.event('anchor-retired',anchor=old['number'],point=old['point'],uses=old['uses'],reasons=reasons)
        with w.span('J_build'):
            r=self.b.blocks(s['H'],s['u'])
            with w.span('A_existing_projection'):A=matrices(self.z,r)
            with w.span('diagnostic_recording'):w.save(f'anchor-{number:02d}-newwet-material',**r['material'])
            d,e=self.mass.pure(s['H'],s['u'],s['H'])
            mismatch=np.linalg.norm(r['F']-s['F'])/max(np.linalg.norm(s['force']),1.)
            w.event('build-residual-consistency',force_scaled=float(mismatch),gate=1e-10)
            if mismatch>1e-10:raise LinearFailure('Pure F differs from original F')
            with w.span('C_current_coefficients'):cv,_=c_values(self.ad,s['H'],s['u'],self.mass.eq.weights(s['e']),r['wet'][::4])
            with w.span('BC_structure_update'):B,C,cache_info=self.structure_cache.update(self.z,self.ad,r['B_local'],cv)
            w.event('BC-structure-cache',anchor=number,**cache_info,B_nnz=B.nnz,C_nnz=C.nnz)
            with w.span('D_assembly'):D=self.mass.eq.matrix(s['u'],s['e']).tocsr()
            P,lu,info=auxiliary(A,B,C,D,w)
        sc=scales(s,self.mass);n=len(s['H']);nu=4*n
        sr=np.r_[np.full(nu,1/(sc['force']*1e-6)),np.full(n,np.sqrt(n)/(sc['volume']*1e-8))]
        sx=np.r_[np.full(nu,max(float(abs(s['u']).max()),1.)),np.full(n,max(float(s['H'].max()),1.))]
        # One GPU anchor; CPU parent references survive bounded recovery until
        # the outer direction's remaining candidate diagnostics are finished.
        self.anchor=dict(number=number,point=s['point'],gpu_point=r['point'],branch=s['branch'],
            blocks=(A,B,C,D),lu=lu,P_info=info,sr=sr,sx=sx,uses=0)
        w.save(f'anchor-{number:02d}-input',H=s['H'],u=s['u'],F=s['F'],R=s['e']['R'],residual_scaling=sr,variable_scaling=sx,**s['branch'])
        w.event('anchor-built',anchor=number,state=s['name'],point=s['point'],GPU_point=r['point'],reasons=reasons,P=info,work=w.record())
        return self.anchor

    def correct_with_anchor(self,s,recovery=False,forced=None):
        w=self.w;w.count('joint_directions',32);number=w.counts['joint_directions'];w.location=f'anchor direction {number}'
        if recovery:w.count('recovery_directions')
        if s['revision']!=self.mass.eq.revision:raise LinearFailure('Stale formal discrete state before direction')
        a=self.anchor;effect=s.get('anchor_model') if a is not None and a['point']!=s['point'] else None
        reasons=refresh_reasons(a['branch'] if a else None,s['branch'],effect)
        if forced:reasons.append(forced)
        if reasons:a=self.build_anchor(s,reasons)
        else:w.count('anchor_reuse_directions')
        a['uses']+=1;A,B,C,D=a['blocks'];sr=a['sr'];sx=a['sx'];nu=len(s['u']);G=np.r_[s['F'],s['e']['R']];calls=[0]
        reused=a['point']!=s['point']
        self.last_direction_info=dict(number=number,anchor=a['number'],anchor_point=a['point'],parent=s['point'],parent_name=s['name'],reused=reused)
        w.event('anchor-direction-start',**self.last_direction_info,current_BP=s['bp'],current_local=s['d']['local'],refresh_reasons=reasons)
        def check():
            if self.anchor is not a or self.b.current!=a['gpu_point']:raise LinearFailure('Anchor arrays/GPU binding changed within linear solve')
        def action(v):
            check()
            with w.span('anchor_J_action'):
                q=sx*v;return sr*np.r_[A@q[:nu]+B@q[nu:],C@q[:nu]+D@q[nu:]]
        def pre(v):
            check();calls[0]+=1
            if calls[0]>24:raise BudgetStop('24 anchor precondition actions')
            require_block_V(w);q=v/sr
            with w.span('block_precondition_action'):
                zu=self.b.V(q[:nu],a['gpu_point'])
                with w.span('C_action'):right=q[nu:]-C@zu
                with w.span('P_LU_apply'):yh=a['lu'].solve(right)
                with w.span('B_action'):by=B@yh
                yu=zu-self.b.V(by,a['gpu_point'])
            return np.r_[yu,yh]/sx
        w.save(f'direction-{number:02d}-input',H=s['H'],u=s['u'],F=s['F'],R=s['e']['R'],residual_scaling=sr,variable_scaling=sx,wet=s['wet'],alpha=self.mass.eq.alpha,beta=self.mass.eq.beta,anchor=np.array(a['number']),anchor_point=np.array(a['point']))
        with w.span('joint_FGMRES'):x,result=fgmres(action,pre,-sr*G,w,f'direction-{number:02d}-anchor-linear')
        dx=sx*x;w.save(f'direction-{number:02d}',du=dx[:nu],dH=dx[nu:])
        w.event('joint-direction',number=number,recovery=recovery,point=s['point'],anchor=a['number'],anchor_point=a['point'],reused=reused,
            anchor_linear_residual=result['true_relative'],precondition_actions=calls[0],V_cycles=2*calls[0],P=a['P_info'],**result)
        self.parent_operators[s['point']]=dict(blocks=a['blocks'],G=G.copy(),sr=sr.copy(),sx=sx.copy(),number=number,anchor=a['number'],anchor_point=a['point'],
            revision=s['revision'],alpha=self.mass.eq.alpha.copy(),beta=self.mass.eq.beta.copy())
        w.event('direction-budget',number=number,parent_state=s['name'],remaining=remaining(w),work=w.record())
        return dx[:nu],dx[nu:]

    def direction(self,s,recovery=False):
        forced=self.force_refresh;self.force_refresh=None
        try:return self.correct_with_anchor(s,recovery,forced)
        except LinearFailure as exc:
            # Identity, true-operator and shared-state failures are not eligible.
            if not str(exc).startswith('Joint FGMRES '):raise
            a=self.anchor
            if a is None or not may_retry(a['point'],s['point'],s['point'] in self.failed_refreshes):raise
            self.failed_refreshes.add(s['point']);self.w.count('stale_failure_refresh');self.w.event('stale-linear-failure',parent=s['point'],anchor=a['number'],error=str(exc))
            return self.correct_with_anchor(s,recovery,'stale-linear-failure')

    def trial(self,s,du,dh,k,tag):
        alpha=2.**(-k);h,hit=feasible(s['H'],dh,alpha);u=s['u']+alpha*du;u[~np.repeat(wet(h,self.bed),4)]=0.
        parent=self.parent_operators[s['point']]
        if s['revision']!=self.mass.eq.revision or not np.array_equal(parent['alpha'],self.mass.eq.alpha) or not np.array_equal(parent['beta'],self.mass.eq.beta):
            raise LinearFailure('Diagnostic parent J identity does not match trial parent')
        displacement=np.r_[u-s['u'],h-s['H']]
        # One immutable anchor action on actual parent->trial displacement before branch selection; not current-J Taylor error.
        with self.w.span('anchor_diagnostic_action'):Js=apply_parent(parent['blocks'],displacement)
        self.w.count('paired_backtrack_candidates');t,changed=self.state(h,u,tag)
        with self.w.span('diagnostic_recording'):
            Gtrial=np.r_[t['F'],t['e']['R']]
            quality,elin,emodel=model_quality(parent['G'],Gtrial,Js,parent['sr'],not changed)
            t['anchor_model']=quality;t['direction_anchor']=dict(number=parent['anchor'],point=parent['anchor_point']);t['model_parent_point']=s['point']
            self.w.save(f'anchor-model-{self.serial:03d}',s=displacement,Js=Js,G_parent=parent['G'],G_trial=Gtrial,
                G_pred=elin,model_defect=emodel,anchor=np.array(parent['anchor']),anchor_point=np.array(parent['anchor_point']),residual_scaling=parent['sr'],variable_scaling=parent['sx'],
                parent_point=np.array(s['point']),child_point=np.array(t['point']),parent_alpha=parent['alpha'],
                parent_beta=parent['beta'],child_alpha=self.mass.eq.alpha,child_beta=self.mass.eq.beta)
            nu=len(s['u'])
            blocks={label:dict(parent=float(np.linalg.norm(parent['G'][sl])),actual=float(np.linalg.norm(Gtrial[sl])),
                linear=float(np.linalg.norm(elin[sl])),model=float(np.linalg.norm(emodel[sl])),
                closure=float(np.linalg.norm(Gtrial[sl]-elin[sl]-emodel[sl]))) for label,sl in [('F',slice(0,nu)),('R',slice(nu,None))]}
            self.w.event('anchor-model',anchor=parent['anchor'],anchor_point=parent['anchor_point'],quality=quality,direction=parent['number'],parent=s['name'],child=t['name'],alpha=alpha,
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
                self.adopted(adopted,'recovery-internal');base=adopted
        self.w.count('new_wet_recovery_failed');return None,None
    def adopted(self,t,kind):
        info=t.get('anchor_model');a=t.get('direction_anchor',{})
        reused=bool(a and a['point']!=t['model_parent_point'])
        if reused:self.w.count('reused_internal_adoptions')
        if reused and kind=='parent':self.w.count('reused_parent_adoptions')
        self.w.event('anchor-internal-adopted',state=t['name'],kind=kind,anchor=a.get('number'),reused=reused,model=info,BP=t['bp'],local=t['d']['local'])

    def solve(self,h,u):
        s,_=self.state(h,u,'initial');parent_iteration=0
        while True:
            self.w.location='parent '+str(parent_iteration);self.history.append(dict(joint_updates=self.w.counts['joint_directions'],bp=s['bp'],**s['d']))
            if accepted(s['d'],s['bp']):
                final,changed=self.state(s['H'],s['u'],'final')
                if not changed and accepted(final['d'],final['bp']):
                    self.mass.save('accepted-state',final['H'],final['u']);self.w.save('accepted-momentum',H=final['H'],u=final['u'],F=final['F'],force=final['force'],wet=final['wet'])
                    return dict(status='accepted',accepted=True,history=self.history,final=final['d'])
                s=final;continue
            self.parent_operators.clear()
            du,dh=self.direction(s);direction_anchor=self.last_direction_info['anchor_point'];sc=scales(s,self.mass);taken=None;restart=None;recovery_used=False
            for k in range(8):
                t,changed,new,alpha=self.trial(s,du,dh,k,'ordinary')
                if changed:
                    restart,_=self.state(s['H'],s['u'],'parent-new-limiter');self.w.count('invalidated_directions')
                    self.w.event('old-linearization-invalidated',reason='alpha/beta changed; re-evaluate parent before anchor',old=s['revision'],new=restart['revision']);break
                physical=t['d']['physical_path'] and t['d']['all_finite'] and t['d']['H_min']>=0
                good=physical and (merit(t,sc)<=(1-1e-4*alpha)*merit(s,sc) or accepted(t['d'],t['bp']))
                self.w.event('parent-descent',alpha=alpha,base_merit=merit(s,sc),trial_merit=merit(t,sc),adopted=good)
                if good:taken=t;break
                if physical and new and not recovery_used:
                    recovery_used=True;taken,restart=self.recover(t,s,alpha,sc)
                    if taken is not None or restart is not None:break
            if restart is not None:s=restart;continue
            if taken is None:
                if may_retry(direction_anchor,s['point'],s['point'] in self.failed_refreshes):
                    self.failed_refreshes.add(s['point']);self.force_refresh='stale-backtracking-failure';self.w.count('stale_failure_refresh')
                    self.w.event('stale-backtracking-failure',parent=s['point'],anchor_point=direction_anchor);continue
                raise LinearFailure('Current/refreshed anchor paired backtracking and recovery exhausted')
            self.adopted(taken,'parent');s=taken;parent_iteration+=1;self.w.count('internal_iterate_adopted')

def attempt(backend,adapter,h_old,u_old,smb,*,end_year,ledger,step_dt,inherited=None):
    if not getattr(backend,'anchor_enabled',True):
        from .baseline import attempt as original
        return original(backend,adapter,h_old,u_old,smb,end_year=end_year,ledger=ledger,step_dt=step_dt,inherited=inherited)
    w=backend.w
    with w.span('predictor'):h,diag,parts=predict(h_old,backend.mesh['bed'],smb,step_dt,adapter.rates(u_old))
    u=u_old.copy();u[~np.repeat(wet(h,backend.mesh['bed']),4)]=0.
    w.save('predictor',H=h,u=u,**parts);w.event('predictor',diagnostics=diag)
    return Process(backend,adapter,h_old,u_old,smb,step_dt,ledger,inherited).solve(h,u)
