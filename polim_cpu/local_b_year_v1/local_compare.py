"""Registered saved-state pair: one D bind, old/new local B once, no solve."""
from polim_cpu.local_b_batch_v1.device import *
from pathlib import Path
import json,sys,time,hashlib,traceback
def main():
    plan=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'));root=Path(sys.argv[2]);e=None;rows=[];start=time.perf_counter();result={}
    def save(name,data):
        with (root/name).open('x',encoding='utf8') as f:json.dump(data,f,indent=2,allow_nan=False)
    def sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
    try:
        with np.load(plan['common_input_path'],allow_pickle=False) as f:z={k:f[k] for k in f.files}
        e=Engine(z,root,True);e.begin(time.time()+180)
        previous=None
        assert len(plan['paired_points'])==1
        for number,point in enumerate(plan['paired_points'],1):
            with np.load(point['path'],allow_pickle=False) as f:h=f['H'].copy();u=f['u'].copy();oldwet=f['wet'].copy() if 'wet' in f.files else None
            tick=time.perf_counter();e.bind(h,u,True);torch.cuda.synchronize();common=time.perf_counter()-tick
            if oldwet is not None:assert np.array_equal(oldwet,e.bp.wet.cpu().numpy())
            oldwet=e.bp.wet.cpu().numpy().copy()
            common_id=e.state_id;out={};times={};materials={}
            for label,fn in ([('old',original_B),('new',local_geometry_blocks)] if number%2 else [('new',local_geometry_blocks),('old',original_B)]):
                torch.cuda.synchronize();t=time.perf_counter();out[label]=fn(e.bp,e.bound_u);torch.cuda.synchronize();times[label]=time.perf_counter()-t
                materials[label]={k:v.copy() for k,v in e.bp.diagnostic_material.items()}
            x=out['old'];y=out['new'];diff=float((x-y).abs().max());scale=max(float(x.abs().max()),1.)
            direction=torch.sin(torch.arange(x.shape[2],device='cuda',dtype=T)+.37);xo=x@direction;yn=y@direction
            act=float((xo-yn).norm());ad=max(float(xo.norm()),1.)
            # Actual saved thickness direction, when available, gathered locally.
            dp=Path(point['direction']);actual={}
            with np.load(dp,allow_pickle=False) as f:
                if 'dH' in f.files:
                    dh=tensor(f['dH']);xx=torch.einsum('eij,ej->ei',x,dh[e.bp.tri]);yy=torch.einsum('eij,ej->ei',y,dh[e.bp.tri]);actual=dict(error=float((xx-yy).norm()),scale=max(float(xx.norm()),1.))
            parent_unchanged=True if previous is None else bool(torch.equal(previous[0],previous[1]))
            row=dict(number=number,input=point,operator_identity=common_id,H_sha256=sha(h),u_sha256=sha(u),common_F_Fu_seconds=common,seconds=times,matrix_error=diff,matrix_scale=scale,matrix_tolerance=1e-13*scale,action_error=act,action_scale=ad,action_tolerance=1e-12*ad,actual_direction=actual,projection_zero=bool((y[~e.bp.wet[e.bp.ids]]==0).all()),finite=bool(torch.isfinite(y).all()),parent_unchanged=parent_unchanged,material_equal=all(np.array_equal(materials['old'][k],materials['new'][k]) for k in materials['old']),peak_memory=torch.cuda.max_memory_allocated(),memory_quota=int(torch.cuda.get_device_properties(0).total_memory*.10))
            row['passed']=diff<=row['matrix_tolerance'] and act<=row['action_tolerance'] and (not actual or actual['error']<=1e-12*actual['scale']) and all(row[k] for k in ('projection_zero','finite','parent_unchanged','material_equal')) and row['peak_memory']<=row['memory_quota']
            with (root/f'pair-{number:02d}.npz').open('xb') as f:np.savez_compressed(f,old=x.cpu().numpy(),new=y.cpu().numpy(),H=h,u=u,wet=oldwet)
            rows.append(row);save(f'pair-{number:02d}.json',row)
            previous=(y,y.clone())
            if not row['passed']:raise RuntimeError('Frozen local B equality gate failed at point '+str(number))
        old=sum(r['seconds']['old'] for r in rows);new=sum(r['seconds']['new'] for r in rows)
        result=dict(status='passed',rows=rows,old_seconds=old,new_seconds=new,net_saved_seconds=old-new,reduction=(old-new)/old)
    except BaseException as ex:result=dict(status='failed',rows=rows,error=repr(ex),traceback=traceback.format_exc())
    finally:
        if e:result['meter']=e.m.record();e.close()
        result.update(command_work_seconds=time.perf_counter()-start,full_BP=0,exact_Fu_solves=0,joint_linear_solves=0,time_steps=0);save('result.json',result)
    return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
