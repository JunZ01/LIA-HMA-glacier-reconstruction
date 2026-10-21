"""Frozen scalar economics policy; no model calls and no online timing inputs."""
import math,hashlib
import numpy as np

def branch_id(b):
    h=hashlib.sha256(b'v58-exact-v57-branch')
    for key in ('wet','alpha','beta','donor','outlet','reflection'):
        a=np.ascontiguousarray(b[key]);h.update(key.encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    h.update(str(int(b['revision'])).encode());return h.hexdigest()

def nonmomentum_pass(s):
    d=s['d']
    return bool(d['local']<=1e-8 and d['fixed_point']<=1e-8 and d['mass']<=1e-10
        and d['annual_cumulative_mass_relative']<=1e-10 and d['inherited_cumulative_mass_relative']<=1e-10
        and d['cumulative_mass_relative']<=1e-10 and d['physical_path'] and d['all_finite'] and d['H_min']>=0
        and np.all(s['u'][~s['wet']]==0))

def positive_log_drop(a,b):
    if not all(math.isfinite(x) and x>0 for x in (a,b)):return None
    value=math.log(a)-math.log(b)
    return value if math.isfinite(value) and value>0 else None

class History:
    """Only adopted parent->child intervals; raw V endpoints retain failed work."""
    def __init__(self,anchor,point,branch,f,V):
        self.anchor=anchor;self.point=point;self.branch=branch;self.f=f;self.V=V
        self.last_point=point;self.last_V=V;self.first_seen=False;self.first=None;self.recent=[]
        self.last_reason='no-first-correction';self.records=[]
    def adopt(self,parent,child,parent_branch,child_branch,fa,fb,V,reused,kind):
        dv=V-self.last_V;drop=positive_log_drop(fa,fb)
        same=self.branch==parent_branch==child_branch;contiguous=parent==self.last_point
        valid=bool(same and contiguous and drop is not None and dv>0)
        r=dict(anchor=self.anchor,parent=parent,child=child,kind=kind,reused=bool(reused),f_parent=fa if math.isfinite(fa) else None,f_child=fb if math.isfinite(fb) else None,
            invalid_norm_repr=None if math.isfinite(fa) and math.isfinite(fb) else repr((fa,fb)),
            V_start=self.last_V,V_end=V,delta_V=dv,same_branch=same,contiguous=contiguous,valid=valid,log_drop=drop)
        if not same:why='branch-changed'
        elif not contiguous:why='noncontiguous-adoption'
        elif drop is None:why='nonpositive-or-invalid-norm-decrease'
        elif dv<=0:why='nonpositive-V-interval'
        else:why='valid'
        if not self.first_seen:
            self.first_seen=True
            if valid and parent==self.point and not reused:self.first={**r,'g_f':drop/dv};why='first-valid'
            else:why='first-correction-unavailable:'+why
        elif valid and reused and self.first is not None:
            self.recent.append(r);self.recent=self.recent[-2:]
        else:self.recent=[]
        if not same or not contiguous:self.first=None;self.recent=[]
        self.last_point=child;self.last_V=V;self.last_reason=why;r['reason']=why;self.records.append(r)
        return r
    def decide(self,*,point,branch,f,force,bp,other_pass,coeff,original=(),enabled=True):
        out=dict(available=False,trigger=False,reason=None,anchor=self.anchor,parent=point,branch=branch,
            g_f=None,g_s=None,L=None,W_stale=None,W_fresh=None,c_V=coeff['c_V'],c_J=coeff['c_J'],
            predicted_saved_seconds=None,required_saved_seconds=1.25*coeff['c_J'],degraded=False,economical=False,
            original_reasons=list(original),history_points=len(self.recent),first=self.first,
            history_last_reason=self.last_reason,recent=self.recent)
        def stop(why):out['reason']=why;return out
        if not enabled:return stop('switch-disabled-v57-scheduling')
        if original:return stop('original-refresh-priority')
        if not math.isfinite(bp):return stop('invalid-current-BP')
        if bp<=1e-6:return stop('BP-already-passed-momentum-prediction-not-applicable')
        if not other_pass:return stop('nonmomentum-gates-not-passed')
        if branch!=self.branch:return stop('branch-mismatch')
        if point!=self.last_point:return stop('current-state-not-last-adopted')
        if self.first is None:return stop('first-correction-unavailable')
        if len(self.recent)!=2:return stop('need-two-consecutive-reused-adoptions')
        a,b=self.recent
        if a['child']!=b['parent'] or b['child']!=point or not a['reused'] or not b['reused']:return stop('noncontiguous-reuse-window')
        drop=positive_log_drop(a['f_parent'],b['f_child']);dv=a['delta_V']+b['delta_V'];gf=self.first['g_f']
        if drop is None or dv<=0:return stop('invalid-window-decrease-or-V')
        gs=drop/dv
        if not all(math.isfinite(x) and x>0 for x in (gf,gs,f)) or not math.isfinite(force) or force<0:return stop('invalid-efficiency-or-current-norm')
        target=1e-6*max(force,1.);L=math.log(f)-math.log(target)
        if not math.isfinite(L) or L<=0:return stop('nonpositive-target-distance')
        a1=L/(2*gs);a2=L/(2*gf)
        if not math.isfinite(a1) or not math.isfinite(a2):return stop('nonfinite-work-estimate')
        ws=2*math.ceil(a1);wf=2*math.ceil(a2);saved=coeff['c_V']*(ws-wf)
        if not math.isfinite(saved):return stop('nonfinite-cost-estimate')
        degraded=gs<=.5*gf;economical=saved>=1.25*coeff['c_J']
        out.update(available=True,trigger=bool(degraded and economical),reason='economic-efficiency-degradation' if degraded and economical else 'retain-anchor',
            g_f=gf,g_s=gs,L=L,f_current=f,f_target=target,current_force_scale=max(force,1.),W_stale=ws,W_fresh=wf,
            predicted_saved_seconds=saved,degraded=degraded,economical=economical)
        return out
