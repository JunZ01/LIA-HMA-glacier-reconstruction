"""Project-local CUDA launch without modifying installed DLLs or global settings."""
import glob
import os
from pathlib import Path
import subprocess
import sys


def launch(arguments):
    if os.environ.get('KMP_DUPLICATE_LIB_OK','').strip():
        raise RuntimeError('KMP_DUPLICATE_LIB_OK is forbidden')
    project=Path(__file__).resolve().parents[2]
    site=Path(os.environ.get('POLIM_TORCH_SITE',str(project/'.runtime/torch-cu128-v1/site-packages'))).resolve()
    python=Path(os.environ.get('POLIM_CUDA_PYTHON',r'D:\Anaconda\envs\SS4\python.exe'))
    if not python.is_file() or not (site/'torch/__init__.py').is_file():
        raise RuntimeError('Set POLIM_CUDA_PYTHON and POLIM_TORCH_SITE to a compatible CUDA Python/runtime')
    environment=os.environ.copy(); environment.pop('PYTHONHOME',None)
    environment.pop('KMP_DUPLICATE_LIB_OK',None)
    windows=Path(os.environ.get('SystemRoot',r'C:\Windows'))
    environment['PATH']=os.pathsep.join(map(str,(site/'torch/lib',windows/'System32',windows)))
    environment['PYTHONPATH']=os.pathsep.join((str(site),str(project)))
    environment['POLIM_V2_CUDA_ISOLATED']='1'; environment['POLIM_TORCH_SITE']=str(site)
    environment['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    return subprocess.call([str(python),'-m','polim_cpu.v2',*arguments],env=environment)


def initialize():
    if os.environ.get('KMP_DUPLICATE_LIB_OK','').strip(): raise RuntimeError('Forbidden OpenMP override')
    site=Path(os.environ['POLIM_TORCH_SITE']); target=os.path.normcase(str(site/'torch/lib/*.dll'))
    original=glob.glob
    def filtered(pattern,*args,**kwargs):
        values=original(pattern,*args,**kwargs)
        if os.path.normcase(str(pattern))==target:
            return [v for v in values if Path(v).name.lower()!='libiompstubs5md.dll']
        return values
    glob.glob=filtered
    try:
        import torch
    finally:
        glob.glob=original
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    return torch
