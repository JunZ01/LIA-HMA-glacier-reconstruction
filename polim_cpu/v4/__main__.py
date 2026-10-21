import argparse
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'


def main():
    parser=argparse.ArgumentParser(description='Experimental graph volume/enthalpy transport')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('register');p.add_argument('stage',choices=('development','terminal'));p.add_argument('--result')
    p=sub.add_parser('verify');p.add_argument('--out',required=True)
    p=sub.add_parser('resume-fixture');p.add_argument('out')
    args=parser.parse_args()
    if args.command=='register':
        from .registration import register
        register(args.stage,args.result)
    elif args.command=='verify':
        from .verify import verify
        verify(args.out)
    else:
        from .verify import resume_fixture
        resume_fixture(args.out)


if __name__=='__main__':main()
