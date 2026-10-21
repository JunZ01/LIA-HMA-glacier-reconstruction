@echo off
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set NUMEXPR_NUM_THREADS=1
set NUMBA_NUM_THREADS=8
cd /d "%~dp0.."
"d:\ProgramData\Anaconda3\python.exe" -m polim_cpu.v10 open
if errorlevel 1 pause
