"""Read-only failed pending-state linearization audit; no time integration."""
from .gpu_allocator_recovery_worker_v1 import Hierarchy
from .gpu_energy_worker_v1 import torch,tensor,Path,np,sys,json,time
from .gpu_allocator_safe_v1 import BP,PCG
from . import gpu_long_storage_v1 as st

def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir();z=st.arrays(root/'input.npz');pending=st.arrays(root/'pending-000001/state.npz');torch.cuda.set_per_process_memory_fraction(.10)
    bp=BP(z);bp.prepare(tensor(pending['h']));u=tensor(pending['u']);bp.evaluate(u,True);old=bp.values.clone();u[~bp.wet]=0;r,_=bp.evaluate(u,True)
    difference=float((bp.values-old).norm()/bp.values.norm());hier=Hierarchy(out,z)
    try:
        mg=hier.build(bp,1);pcg=PCG(bp,mg,True);start=time.perf_counter()
        try:
            d,k,err=pcg.solve(r);result=dict(status='pass',iterations=k,linear_residual=err,seconds=time.perf_counter()-start)
        except Exception as e:result=dict(status='failed',error=repr(e),seconds=time.perf_counter()-start)
        result.update(tangent_relative_change_from_clearing_dry_velocity=difference,dry_nodes=int((~bp.wet).sum()/4),pending_state_sha256=st.digest(root/'pending-000001/state.npz'),dynamics_performed=False)
        st.write(out/'result.json',result);print(json.dumps(result))
    finally:hier.close()

if __name__=='__main__':main()
