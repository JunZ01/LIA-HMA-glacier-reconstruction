"""Isolate incompatible optional Triton 2 from eager CPU-only torch training."""
import sys,runpy
sys.modules['triton']=None
script=sys.argv.pop(1)
runpy.run_path(script,run_name='__main__')
