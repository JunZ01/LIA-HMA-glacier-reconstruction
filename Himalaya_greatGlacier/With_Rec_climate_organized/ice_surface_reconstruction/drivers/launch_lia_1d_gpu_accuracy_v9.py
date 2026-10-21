from __future__ import annotations
import os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]; PY=Path("D:/Anaconda/envs/SS4/python.exe"); SITE=ROOT/".runtime/torch-cu128-v1/site-packages"; LIB=SITE/"torch/lib"; WIN=Path(os.environ.get("SystemRoot","C:/Windows")); RUN=Path(__file__).with_name("run_lia_1d_gpu_accuracy_v9.py")
NAMES=("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")
def main():
    env=os.environ.copy(); env.pop("KMP_DUPLICATE_LIB_OK",None); env.pop("PYTHONHOME",None); env["PATH"]=os.pathsep.join(map(str,(LIB,WIN/"System32",WIN))); env["PYTHONPATH"]=str(SITE); env["LIA_1D_GPU_ACCURACY_V9_CHILD"]="1"
    for n in NAMES: env[n]="1"
    return subprocess.run([str(PY),str(RUN)],env=env,check=False).returncode
if __name__=="__main__": raise SystemExit(main())
