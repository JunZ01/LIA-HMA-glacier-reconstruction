"""python -m polim_cpu demo|run|resume|batch|report"""
import argparse
import json
import os
from pathlib import Path

def main():
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='1'
    parser=argparse.ArgumentParser(description='PoLIM CPU prototype; isolated output, checkpoint every step')
    subs=parser.add_subparsers(dest='command',required=True)
    demo=subs.add_parser('demo'); demo.add_argument('--out',required=True); demo.add_argument('--steps',type=int)
    run=subs.add_parser('run'); run.add_argument('config'); run.add_argument('--out',required=True); run.add_argument('--steps',type=int)
    resume=subs.add_parser('resume'); resume.add_argument('out'); resume.add_argument('--steps',type=int,required=True)
    batch=subs.add_parser('batch'); batch.add_argument('configs',nargs='+'); batch.add_argument('--out',required=True)
    report=subs.add_parser('report'); report.add_argument('out')
    args=parser.parse_args()
    from . import runner
    if args.command=='demo': result=runner.run(runner.demo_config(),args.out,steps=args.steps)
    elif args.command=='run':
        path=Path(args.config)
        result=runner.run(json.loads(path.read_text(encoding='utf-8')),args.out,steps=args.steps,base=path.parent)
    elif args.command=='resume': result=runner.run(None,args.out,resume=True,steps=args.steps)
    elif args.command=='batch': result=runner.batch(args.configs,args.out)
    else:
        from .report import render
        result={'report':str(render(args.out))}
    if isinstance(result,dict): result={k:v for k,v in result.items() if k!='history'}
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
