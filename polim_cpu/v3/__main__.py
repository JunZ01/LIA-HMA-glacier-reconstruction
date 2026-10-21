import argparse
import json
import os
from pathlib import Path
import sys
import warnings

for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'


def main():
    parser=argparse.ArgumentParser(description='Moving ice support and climate-coupled flowline v3')
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('run','resume','demo'):
        p=sub.add_parser(name)
        if name=='run': p.add_argument('config')
        p.add_argument('--out',required=True); p.add_argument('--device',choices=('cpu','cuda')); p.add_argument('--stop-step',type=int)
        p.add_argument('--no-report',action='store_true')
    p=sub.add_parser('prepare'); p.add_argument('--out',required=True)
    p=sub.add_parser('report'); p.add_argument('out')
    p=sub.add_parser('verify'); p.add_argument('--out',required=True); p.add_argument('--gpu',action='store_true')
    args=parser.parse_args(); device=getattr(args,'device',None) or 'cpu'
    if args.command=='resume':
        record=json.loads((Path(args.out)/'run.json').read_text())['payload']['runtime']
        device=args.device or record['device']
        if record.get('torch_site'): os.environ.setdefault('POLIM_TORCH_SITE',record['torch_site'])
        if record.get('python_executable'): os.environ.setdefault('POLIM_CUDA_PYTHON',record['python_executable'])
    if device=='cuda':
        from . import gpu
        if os.environ.get('POLIM_V3_CUDA_ISOLATED')!='1':
            result=gpu.launch(sys.argv[1:])
            if result==0 and not args.no_report:
                try:
                    from .report import render
                    render(Path(args.out))
                except Exception as exc: warnings.warn('State saved; optional report failed: '+str(exc))
            return result
        gpu.initialize()
    if args.command=='prepare':
        from .inputs import prepare
        print(prepare(Path(args.out))); return 0
    if args.command=='report':
        from .report import render
        print(render(Path(args.out))); return 0
    if args.command=='verify':
        from .verify import verify
        return verify(Path(args.out),args.gpu)
    from .runner import run
    config=None; base=None
    if args.command=='run':
        path=Path(args.config).resolve(); config=json.loads(path.read_text(encoding='utf-8')); base=path.parent
    elif args.command=='demo':
        from .verify import fixture_input
        path=fixture_input(Path(str(args.out)+'-input').resolve())
        config=json.loads(path.read_text()); base=path.parent
    result=run(config,args.out,device=device,resume=args.command=='resume',stop_step=args.stop_step,base=base,make_report=not args.no_report)
    print(json.dumps(result,indent=2)); return 0


if __name__=='__main__':
    try: sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc(); sys.exit(1)
