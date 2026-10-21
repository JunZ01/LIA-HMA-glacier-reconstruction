"""Synthetic tensors only: no glacier geometry, model residual or solve."""
from .device import *
from types import SimpleNamespace
from contextlib import nullcontext
import json,sys,traceback
from pathlib import Path
def main():
    target=Path(sys.argv[1]);result={'real_model_evaluations':0,'real_solves':0};torch.manual_seed(6042)
    try:
        n=7;nn=541;r=lambda *s:torch.randn(*s,device='cuda',dtype=T)
        tri=torch.tensor([[540,1,2]]*n,device='cuda');ids=torch.arange(12,device='cuda')[None].repeat(n,1)
        bp=SimpleNamespace(tri=tri,ids=ids,wet=torch.ones(nn*4,device='cuda',dtype=torch.bool),h=r(nn).abs()+.1,bed=r(nn),grad=r(n,3,2),lam=r(12,3).abs()+.2,sign=r(12,6),sigma=r(12),B=r(n,12,5,12),lin_bu=r(n,12,12),lin_eta=r(n,12).abs()+.1,lin_e2=r(n,12).abs()+.1,weight=r(n,12).abs(),gs=r(n,2),phi=r(12,6),m=SimpleNamespace(span=lambda k:nullcontext()))
        rows=[]
        for name in ('wet','dry_projection','thin'):
            if name=='dry_projection':bp.wet[4:8]=False
            if name=='thin':bp.h[:]=1e-9
            u=r(nn*4);old=original_B(bp,u);keep=old.clone();new=local_geometry_blocks(bp,u)
            diff=float((new-old).abs().max());den=max(float(old.abs().max()),1.);tol=1e-13*den
            direction=r(3);x=old@direction;y=new@direction;act=float((x-y).norm());actden=max(float(x.norm()),1.)
            row=dict(name=name,absolute=diff,denominator=den,tolerance=tol,action_error=act,action_tolerance=1e-12*actden,finite=bool(torch.isfinite(new).all()),projection_zero=bool((new[~bp.wet[bp.ids]]==0).all()),parent_unchanged=bool(torch.equal(old,keep)))
            row['passed']=diff<=tol and act<=row['action_tolerance'] and row['finite'] and row['projection_zero'] and row['parent_unchanged'];rows.append(row)
        result.update(checks=rows,passed=all(x['passed'] for x in rows),device=torch.cuda.get_device_name(),torch=str(torch.__version__))
    except BaseException as ex:result.update(passed=False,error=repr(ex),traceback=traceback.format_exc())
    with target.open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
