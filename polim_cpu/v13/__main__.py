import sys
cmd=sys.argv[1] if len(sys.argv)>1 else 'open'
if cmd=='prepare':
    from .registration import register
    register()
elif cmd=='run':
    from .run import run
    run()
else:raise SystemExit('prepare or run; report opens separately')
