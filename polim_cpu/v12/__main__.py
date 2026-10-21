import sys

def main():
    cmd=sys.argv[1] if len(sys.argv)>1 else 'open'
    if cmd=='prepare':
        from .registration import register
        register()
    elif cmd=='run':
        from .run import run
        run()
    elif cmd=='resume-fixture':
        from .run import fixture
        from .registration import active
        fixture(sys.argv[2],active(),resume=True)
    elif cmd in ['render','open']:
        from .report import render
        p=render();print(p)
        if cmd=='open':
            import webbrowser
            webbrowser.open(p.as_uri())
    elif cmd=='finish':
        from .complete import complete
        complete()
    else:raise SystemExit('prepare,run,render,open,finish')

if __name__=='__main__':main()
