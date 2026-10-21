import sys
from pathlib import Path


def main():
    command=sys.argv[1] if len(sys.argv)>1 else 'open'
    if command=='prepare':
        from .registration import register
        register('development')
    elif command=='run':
        from .run import run
        run()
    elif command=='resume-fixture':
        from .run import resume_fixture
        resume_fixture(sys.argv[2])
    elif command in ('render','open'):
        from .registration import RUN
        if command=='open' and not (RUN/'result.json').exists():
            if RUN.exists():raise SystemExit('Existing run is unfinished or failed; preserve checkpoints and reconcile its registered state before continuation.')
            from .run import run
            run()
        from .report import render
        path=render();print(path)
        if command=='open':
            import webbrowser
            webbrowser.open(path.as_uri())
    elif command=='finish':
        from .registration import register,RUN
        register('terminal',RUN/'terminal-evidence.json')
    else:raise SystemExit('Commands: prepare,run,render,open,finish')


if __name__=='__main__':main()
