"""CLI; CUDA is isolated before importing any numerical libraries."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'


def main():
    parser=argparse.ArgumentParser(description='Experimental coupled BP flowline model v2')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('demo','run','resume','batch'):
        command=sub.add_parser(name)
        if name in ('run','batch'): command.add_argument('config',nargs='+' if name=='batch' else None)
        command.add_argument('--out',required=True)
        command.add_argument('--device',choices=('cpu','cuda'))
        if name!='batch': command.add_argument('--steps',type=int)
        command.add_argument('--no-report',action='store_true')
    command=sub.add_parser('report'); command.add_argument('out')
    command=sub.add_parser('arolla'); command.add_argument('--out',required=True)
    command=sub.add_parser('verify'); command.add_argument('--out',required=True); command.add_argument('--gpu',action='store_true')
    args=parser.parse_args()
    device=getattr(args,'device',None) or 'cpu'
    if args.command=='resume':
        manifest=json.loads((Path(args.out)/'run.json').read_text())
        recorded=manifest['payload']['runtime']
        device=args.device or recorded['backend']
        if recorded.get('torch_site'): os.environ.setdefault('POLIM_TORCH_SITE',recorded['torch_site'])
        if recorded.get('python_executable'): os.environ.setdefault('POLIM_CUDA_PYTHON',recorded['python_executable'])
    if device=='cuda':
        from . import gpu
        if os.environ.get('POLIM_V2_CUDA_ISOLATED')!='1':
            code=gpu.launch(sys.argv[1:])
            if code==0 and not args.no_report:
                from .report import render
                if args.command=='batch':
                    for root in Path(args.out).glob('case-*'):
                        if list(root.glob('step-*')): render(root)
                else: render(Path(args.out))
            return code
        gpu.initialize()
    if args.command=='report':
        from .report import render
        print(render(Path(args.out))); return 0
    if args.command=='arolla':
        from .inputs import prepare_arolla
        print(prepare_arolla(Path(args.out))); return 0
    if args.command=='verify':
        from .verify import verify
        return verify(Path(args.out),args.gpu)
    from .runner import run,batch,demo_config
    if args.command=='batch':
        result=batch(args.config,args.out,device)
        print(json.dumps(result,indent=2)); return int(any(v['status']=='failed' for v in result))
    config=None; base=None
    if args.command=='demo': config=demo_config()
    elif args.command=='run':
        path=Path(args.config); config=json.loads(path.read_text(encoding='utf-8')); base=path.parent
    result=run(config,args.out,device=device,resume=args.command=='resume',steps=args.steps,base=base,make_report=not args.no_report)
    print(json.dumps({k:v for k,v in result.items() if k!='history'},indent=2))
    return 0


if __name__=='__main__':
    try: sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        sys.exit(1)
