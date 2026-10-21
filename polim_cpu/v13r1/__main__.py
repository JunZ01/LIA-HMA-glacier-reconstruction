import sys
if sys.argv[1]=='prepare':
    from .registration import register
    register()
elif sys.argv[1]=='run':
    from .run import run
    run()
