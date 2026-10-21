"""Bounded NVTX observations; no model computations, GPU sync, or array copies."""
import os,time,json,ctypes,threading,functools,atexit
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[2]
NVTX=ctypes.WinDLL('C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll')
NVTX.nvtxRangePushA.argtypes=[ctypes.c_char_p];NVTX.nvtxRangePushA.restype=ctypes.c_int;NVTX.nvtxRangePop.restype=ctypes.c_int
ROLE='unset';OUT=None;EVENTS=[];LIMIT=200000;DROPPED=0;TLS=threading.local();PATCHED=set();RPC_SERIAL=0
@contextmanager
def span(name):
    global DROPPED
    rid=getattr(TLS,'rid','');label=ROLE+'|'+str(name)+(('|rid='+rid) if rid else '')
    t=time.perf_counter_ns();cpu=time.thread_time_ns();NVTX.nvtxRangePushA(label.encode('utf8'))
    try:yield
    finally:
        NVTX.nvtxRangePop();end=time.perf_counter_ns();elapsed_cpu=time.thread_time_ns()-cpu
        if len(EVENTS)<LIMIT:EVENTS.append((label,os.getpid(),threading.get_native_id(),t,end,elapsed_cpu))
        else:DROPPED+=1
def init(role,out):
    global ROLE,OUT
    ROLE=role;OUT=Path(out);OUT.mkdir(parents=True,exist_ok=True);atexit.register(drain)
def drain():
    global OUT
    if OUT is None:return
    path=OUT/f'observer-{ROLE}-{os.getpid()}.json';t=time.perf_counter()
    # Flush once, at exit. Raw NVTX is the authoritative cross-process time axis.
    with path.open('x',encoding='utf8') as f:json.dump(dict(role=ROLE,pid=os.getpid(),clock='perf_counter_ns; use Nsight NVTX mapped timestamps for cross-process analysis',columns=['label','pid','tid','begin','end','thread_cpu_ns'],events=EVENTS,dropped=DROPPED,limit=LIMIT),f)
    with (OUT/f'observer-drain-{ROLE}-{os.getpid()}.json').open('x') as f:json.dump(dict(seconds=time.perf_counter()-t,bytes=path.stat().st_size),f)
    OUT=None
def wrap(obj,key,label=None):
    token=(id(obj),key)
    if token in PATCHED:return
    PATCHED.add(token);original=getattr(obj,key)
    @functools.wraps(original)
    def measured(*a,**kw):
        with span(label or key):return original(*a,**kw)
    setattr(obj,key,measured)
def wrap_span(cls,prefix):
    original=cls.span
    @contextmanager
    def measured(self,key,*a,**kw):
        with span(prefix+str(key)):
            with original(self,key,*a,**kw):yield
    cls.span=measured
def io_hooks():
    import zipfile
    # Preserve original file objects: replacing them could change NumPy's tofile path.
    for obj,key in [(os,'fsync'),(os,'replace'),(json,'dumps'),(json,'dump'),(zipfile._ZipWriteFile,'write')]:wrap(obj,key,'IO/'+key)
def connection_hooks():
    import multiprocessing.connection as mc
    orig_send,orig_recv=mc._ConnectionBase.send,mc._ConnectionBase.recv
    def send(self,obj):
        global RPC_SERIAL
        rid=getattr(TLS,'rid','')
        with span('RPC/send-pack-and-write'):
            if isinstance(obj,dict) and rid:obj={**obj,'__observation_request_id__':rid}
            result=orig_send(self,obj)
        if ROLE=='GPU' and getattr(TLS,'service_range',None):
            cm=TLS.service_range;TLS.service_range=None;cm.__exit__(None,None,None);TLS.rid=''
        return result
    def recv(self):
        with span('RPC/receive-read-and-unpack'):obj=orig_recv(self)
        if isinstance(obj,dict) and '__observation_request_id__' in obj:
            rid=obj.pop('__observation_request_id__')
            if ROLE=='GPU':
                TLS.rid=rid;cm=span('RPC/service/'+str(obj.get('kind')));cm.__enter__();TLS.service_range=cm
            elif rid!=getattr(TLS,'rid',''):raise RuntimeError('Observation request identity mismatch')
        return obj
    mc._ConnectionBase.send=send;mc._ConnectionBase.recv=recv
    for name in ('_send_bytes','_recv_bytes','poll'):
        cls=mc._ConnectionBase if name=='poll' else mc.Connection
        wrap(cls,name,'RPC/'+name)
    # Serialization stays byte-for-byte on the original protocol and arrays.
    original=mc._ForkingPickler.dumps
    @classmethod
    def dumps(cls,obj,protocol=None):
        with span('RPC/pickle-dumps'):return original(obj,protocol)
    mc._ForkingPickler.dumps=dumps
    original_loads=mc._ForkingPickler.loads
    def loads(*a,**kw):
        with span('RPC/pickle-loads'):return original_loads(*a,**kw)
    mc._ForkingPickler.loads=loads
def cpu_hooks(control):
    import numpy as np
    from polim_cpu.geometry_coupled_v1.backend import Work
    from polim_cpu.anchor_efficiency_v1 import client,baseline,algebra
    from polim_cpu.geometry_window_v1 import ledger
    from polim_cpu.geometry_probe_v2 import common
    wrap_span(Work,'phase/')
    for key in ('progress','save','event'):wrap(Work,key,'record/'+key)
    for key in ('digest','hashes','matches','arrays'):wrap(common,key,'files/'+key)
    # gate() references original common.matches; its digest lookup is still measured.
    for key in ('gate','write'):wrap(control,key,'management/'+key)
    for key in ('restore','checkpoint'):wrap(ledger,key,'checkpoint/'+key)
    for key in ('savez','savez_compressed'):wrap(np,key,'array-serialization/'+key)
    for key in ('cholesky','solve','lstsq'):wrap(np.linalg,key,'CPU-linalg/'+key)
    for key in ('direction','state','trial','recover'): 
        if hasattr(baseline.Process,key):wrap(baseline.Process,key,'joint/'+key)
    original=client.Backend.rpc
    @functools.wraps(original)
    def rpc(self,kind,**kw):
        global RPC_SERIAL
        RPC_SERIAL+=1;old=getattr(TLS,'rid','');TLS.rid=f'{os.getpid()}-{RPC_SERIAL:05d}-{kind}'
        try:
            with span('RPC/client/'+kind):return original(self,kind,**kw)
        finally:TLS.rid=old
    client.Backend.rpc=rpc
    # Change only the observation wrapper used to enter the same GPU service.
    import subprocess
    old_popen=subprocess.Popen
    def popen(args,*a,**kw):
        if isinstance(args,list) and 'polim_cpu.anchor_efficiency_v1.service' in args:
            args=[('polim_cpu.path_profile_restart_v1.service_entry' if v=='polim_cpu.anchor_efficiency_v1.service' else v) for v in args]
        return old_popen(args,*a,**kw)
    subprocess.Popen=popen
    io_hooks();connection_hooks()
def gpu_hooks():
    from polim_cpu.cn_cuda_v1.device import Meter
    from polim_cpu.anchor_efficiency_v1.device import Engine
    wrap_span(Meter,'phase/')
    for key in ('__init__','bind','ensure_preconditioner','blocks','pure','V','close'):wrap(Engine,key,'GPU/'+key)
    io_hooks();connection_hooks()
