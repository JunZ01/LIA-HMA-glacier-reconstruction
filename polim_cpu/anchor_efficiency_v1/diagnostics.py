"""Read-only candidate diagnostics; immutable CPU parent blocks, no model calls."""
import numpy as np

TRACK = np.array([540,9954,9955,9956,9999])
PATH_KEYS = ('a','b','c','minimum','theta_star','ell','H_path','dry','dry_lo','dry_hi',
             'lambda_min','lambda_max','exhaustion','rewetting')

def apply_parent(blocks, displacement):
    A,B,C,D=blocks;nu=A.shape[0];v=displacement
    return np.r_[A@v[:nu]+B@v[nu:],C@v[:nu]+D@v[nu:]]

def partition(F,force):
    mask=np.zeros(len(F),bool);mask[(4*TRACK[:,None]+np.arange(4)).ravel()]=True
    den=max(float(np.linalg.norm(force)),1.)
    return dict(F_norm=float(np.linalg.norm(F)),force_scale=den,
                newwet_F_norm=float(np.linalg.norm(F[mask])),rest_F_norm=float(np.linalg.norm(F[~mask])),
                node_F_norm=np.linalg.norm(F.reshape(-1,4)[TRACK],axis=1).tolist())

def branch_difference(parent,child):
    a,b=parent['e']['path'],child['e']['path']
    def label(p):return np.where(p['ell']>0,np.where(p['theta_star']==0,1,np.where(p['theta_star']==1,2,3)),0)
    return dict(wet_changed=int(np.count_nonzero(parent['wet']!=child['wet'])//4),
        reflection_minimizer_branch_changed=int(np.count_nonzero(label(a)!=label(b))),
        dry_membership_changed=int(np.count_nonzero(a['dry']!=b['dry'])),
        theta_star_value_changed=int(np.count_nonzero(a['theta_star']!=b['theta_star'])),
        dry_endpoint_value_changed=int(np.count_nonzero((a['dry_lo']!=b['dry_lo'])|(a['dry_hi']!=b['dry_hi']))))

def remaining(w):
    return {k:limit-w.counts[k] for k,limit in [('joint_directions',32),('anchor_builds',12),('V_requests',256),('pure_joint_residual',128)]}
