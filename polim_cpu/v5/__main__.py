import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from .registration import register

parser=argparse.ArgumentParser(description='Synthetic branch coupling only; no real glacier run command')
sub=parser.add_subparsers(dest='command',required=True)
r=sub.add_parser('register');r.add_argument('stage',choices=['development','terminal']);r.add_argument('--result')
v=sub.add_parser('verify');v.add_argument('--out',required=True)
c=sub.add_parser('resume-fixture');c.add_argument('root')
args=parser.parse_args()
if args.command=='register':register(args.stage,args.result)
elif args.command=='verify':
    from .verify import verify
    verify(args.out)
else:
    from .verify import resume_fixture
    resume_fixture(args.root)
