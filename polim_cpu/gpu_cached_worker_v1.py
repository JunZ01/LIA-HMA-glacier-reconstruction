"""Bounded cached-tangent and CUDA Graph comparisons."""
from . import gpu_energy_worker_v1 as old
from .gpu_energy_worker_v1 import *
from .gpu_cached_v1 import BP,Multigrid,PCG,solve
old.solve=solve

def main():
    root=Path(sys.argv[1]);torch.cuda.set_per_process_memory_fraction(.10)
    write(root/'gpu-launch.json',dict(pid=os.getpid(),device=torch.cuda.get_device_name(0),torch=torch.__version__,start=time.time(),command=sys.argv,checkpoint='same v29 pending/accepted and5Newton',estimate_minutes=[1,5],stop='20Newton,200CG,120seconds/BP,900seconds/case'))
    try:
        start=time.perf_counter();z=load(root/'input.npz');bp=BP(z);tr=Transport(z);mg=Multigrid(z);net=Network(z);torch.cuda.synchronize();setup=time.perf_counter()-start
        preflight(root,z,bp,tr,net);del net
        r,_=bp.evaluate(tensor(z['u']),True);eager=PCG(bp,mg);graph=PCG(bp,mg,True)
        graph.capture(r);eager.reset(r);graph.reset(r)
        for _ in range(8):eager.step()
        for _ in range(2):graph.graph.replay()
        error=float((eager.y-graph.y).norm()/eager.y.norm().clamp_min(1.));assert error<=1e-12
        mem=torch.cuda.max_memory_allocated();assert mem<.08*torch.cuda.get_device_properties(0).total_memory
        write(root/'new-kernels.json',dict(status='pass',eager_graph_8_iteration_relative_error=error,capture_seconds=graph.capture_seconds,peak_bytes=mem))
        # Reuse both captured graph and its buffers, charged as initial setup.
        results={}
        for kind,pcg in [('cached',eager),('graph',graph)]:
            bp.pcg=pcg;results[kind]=case(root,z,bp,tr,mg,kind)
        write(root/'gpu-result.json',dict(status='completed' if all(v['status']=='completed' for v in results.values()) else 'failed',setup_seconds=setup,graph_capture_seconds=graph.capture_seconds,cases=results,peak_memory_bytes=torch.cuda.max_memory_allocated()))
    except Exception as exc:
        write(root/'gpu-result.json',dict(status='technical-failure',error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
