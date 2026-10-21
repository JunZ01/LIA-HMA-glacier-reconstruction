import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse
parser=argparse.ArgumentParser(description='Global BP synthetic mechanics and static real geometry')
sub=parser.add_subparsers(dest='command',required=True)
r=sub.add_parser('register');r.add_argument('stage',choices=['development','terminal']);r.add_argument('--result')
v=sub.add_parser('verify');v.add_argument('--out',required=True)
r=sub.add_parser('resume-fixture');r.add_argument('root')
a=parser.parse_args()
if a.command=='register':
    from .registration import register
    register(a.stage,a.result)
elif a.command=='verify':
    from .verify import verify
    verify(a.out)
else:
    from .verify import resume
    resume(a.root)
