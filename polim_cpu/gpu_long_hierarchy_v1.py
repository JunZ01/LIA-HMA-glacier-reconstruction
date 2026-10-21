"""CPU setup service: exact saved current tangent to a bounded AMG hierarchy."""
from .solver_acceleration_v1 import *
from scipy.sparse import csr_matrix,diags

def build(folder):
    start=time.perf_counter();z=arrays(folder/'matrix.npz')
    k=csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape']))
    if 'wet' in z:
        w=diags(z['wet'].astype(float));k=(w@k@w+diags((~z['wet']).astype(float))).tocsr()
    d=1/np.sqrt(k.diagonal());scaled=(diags(d)@k@diags(d)).tocsr()
    modes=np.zeros((len(d),4))
    for j in range(4):modes[j::4,j]=1/d[j::4]
    modes/=np.linalg.norm(modes,axis=0);np.random.seed(2387)
    ml=original.pyamg.smoothed_aggregation_solver(scaled.tobsr(blocksize=(4,4)),B=modes,symmetry='symmetric',max_levels=12,max_coarse=160)
    out=dict(scaling=d,levels=len(ml.levels))
    def put(name,m):
        m=m.tocsr();out.update({name+'_data':m.data,name+'_indices':m.indices,name+'_indptr':m.indptr,name+'_shape':np.array(m.shape)})
    for i,l in enumerate(ml.levels):
        k=l.A.tocsr();put(f'A{i}',k);dg=k.diagonal();out[f'invdiag{i}']=1/dg;out[f'omega{i}']=.8/np.max(np.asarray(abs(k).sum(axis=1)).ravel()/dg)
        if i<len(ml.levels)-1:put(f'P{i}',l.P);put(f'R{i}',l.R)
    with (folder/'hierarchy.npz').open('xb') as f:np.savez(f,**out)
    write_json(folder/'result.json',dict(status='completed',seconds=time.perf_counter()-start,sha256=digest(folder/'hierarchy.npz'),matrix_sha256=digest(folder/'matrix.npz')))

if __name__=='__main__':
    # A persistent independent CPU interpreter amortizes import cost over years.
    for line in sys.stdin:
        try:build(Path(json.loads(line)['folder']));print('{"ok":true}',flush=True)
        except Exception as exc:print(json.dumps(dict(ok=False,error=repr(exc),traceback=traceback.format_exc())),flush=True)
