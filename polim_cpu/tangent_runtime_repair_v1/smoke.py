"""Actual GPU import/I/O/comparison path, tiny arrays only; no model instantiated."""
import sys,os,traceback,time
from pathlib import Path
from .paired import compare_pair,save,read,write,ah,np,torch,tensor

def run(root,mode):
    root=Path(root);start=time.perf_counter()
    try:
        assert not os.environ.get('KMP_DUPLICATE_LIB_OK')
        x=np.asarray([3.,4.,0.,0.]);y=x.copy();y[2]=1e-7
        save(root/'input.npz',x=x,y=y)
        with np.load(root/'input.npz') as a:xx=a['x'];yy=a['y']
        assert ah(xx)==ah(x) and ah(yy)==ah(y)
        out=[compare_pair(xx,yy,0),compare_pair(np.zeros(4),np.full(4,1e-12),1)]
        # Exercise Torch CUDA reductions/SVD and transfer before comparison;
        # no BP, AMG or real/synthetic tangent solve is invoked.
        u,s,v=torch.linalg.svd(torch.stack([tensor(x),tensor(y)],1),full_matrices=False)
        save(root/'result-arrays.npz',singular=s.cpu().numpy())
        write(root/'comparison.json',out)
        assert all(r['passed'] for r in out)
        assert abs(out[0]['denominator']-5.)<1e-14 and abs(out[0]['absolute']-1e-7)<1e-19
        assert abs(out[1]['absolute']-2e-12)<1e-24
        torch.cuda.synchronize()
        write(root/'completed-work.json',dict(seconds=time.perf_counter()-start,comparisons=2,full_BP=0,real_tangent=0,time_steps=0))
        if mode=='fail':raise RuntimeError('Injected post-comparison exception')
        write(root/'result.json',dict(status='pass',torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name(),python=sys.version,mode=mode))
    except BaseException as exc:
        write(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':run(sys.argv[1],sys.argv[2])
