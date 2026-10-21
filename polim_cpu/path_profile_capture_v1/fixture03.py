"""Three-process NVTX/CUDA-graph fixture. No real geometry or model imports."""
import os,sys,time,json,traceback,subprocess,ctypes,threading
from pathlib import Path
from contextlib import contextmanager
from multiprocessing.connection import Listener,Client
from . import observe
ROOT=Path(__file__).resolve().parents[2]
N=ctypes.WinDLL('C:/Program Files/NVIDIA Corporation/NvToolsExt/bin/x64/nvToolsExt64_1.dll')
N.nvtxRangePushA.argtypes=[ctypes.c_char_p];N.nvtxRangePushA.restype=ctypes.c_int
N.nvtxRangePop.restype=ctypes.c_int
@contextmanager
def mark(name):
    N.nvtxRangePushA(name.encode('utf8'))
    try:yield
    finally:N.nvtxRangePop()
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def device(p):
    cfg=json.loads(Path(p).read_text());root=Path(cfg['root']);observe.init('GPU',root/'trace');observe.io_hooks();observe.connection_hooks();c=Client(tuple(cfg['address']),authkey=bytes.fromhex(cfg['auth']));info=dict(pid=os.getpid(),model_work=0)
    try:
        with mark('fixture:GPU:import-and-init'):
            import torch
            torch.cuda.init();info.update(torch=str(torch.__version__),cuda=torch.version.cuda,device=torch.cuda.get_device_name())
        with mark('fixture:GPU:graph-build'):
            x=torch.ones(512,device='cuda',dtype=torch.float64);y=torch.empty_like(x);s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):torch.mul(x,2,out=y)
            torch.cuda.current_stream().wait_stream(s);torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):torch.mul(x,2,out=y)
        c.send(dict(ok=True,info=info))
        while True:
            with mark('fixture:GPU:receive'):q=c.recv()
            if q['kind']=='close':break
            rid=q['request_id']
            try:
                with mark('fixture:GPU:request:'+rid):
                    if q['kind']=='error':raise ValueError('intentional fixture-only error')
                    g.replay();torch.cuda.synchronize();result=y.cpu().tolist()
                reply=dict(ok=True,request_id=rid,total=sum(result))
            except ValueError as ex:
                reply=dict(ok=False,request_id=rid,error=repr(ex));save(root/('expected-error-'+rid+'.json'),reply)
            with mark('fixture:GPU:reply:'+rid):c.send(reply)
        save(root/'device-result.json',dict(status='pass',**info,graph_builds=1,graph_replays=1,CUDA_initializations=1))
    except BaseException as ex:
        save(root/'device-failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
    finally:c.close()
def worker(root):
    from polim_cpu.path_profile_v1.probe import env_gpu
    root=Path(root);root.mkdir();observe.init('CPU',root/'trace');observe.io_hooks();observe.connection_hooks();key=os.urandom(24);l=Listener(('127.0.0.1',0),authkey=key);c=None;p=None
    cfg=dict(root=str(root),address=l.address,auth=key.hex());save(root/'service.json',cfg)
    cmd=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.path_profile_capture_v1.fixture03','device',str(root/'service.json')]
    save(root/'launch.json',dict(command=cmd,model_work=0))
    try:
        with (root/'device.log').open('x') as f:
            with mark('fixture:CPU:GPU-start'):
                p=subprocess.Popen(cmd,cwd=ROOT,env=env_gpu(),stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
                l._listener._socket.settimeout(60);c=l.accept();assert c.poll(60);hello=c.recv();assert hello['ok']
            replies=[]
            for rid,kind in [('r001','graph'),('r002','error')]:
                observe.TLS.rid=rid
                with mark('fixture:CPU:request:'+rid):
                    with mark('fixture:CPU:send:'+rid):c.send(dict(kind=kind,request_id=rid))
                    with mark('fixture:CPU:receive:'+rid):assert c.poll(30);r=c.recv()
                    assert r['request_id']==rid;replies.append(r)
            observe.TLS.rid=''
            assert replies[0]['total']==1024 and not replies[1]['ok']
            c.send(dict(kind='close'));p.wait(30);assert p.returncode==0
        save(root/'worker-result.json',dict(status='pass',pid=os.getpid(),gpu_pid=p.pid,replies=replies,model_work=0))
    except BaseException as ex:
        save(root/'worker-failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise
    finally:
        if c:c.close()
        l.close()
        if p and p.poll() is None:p.terminate();p.wait(10)
def supervisor(root):
    root=Path(root);root.mkdir();observe.init('SUP',root/'trace');observe.io_hooks();(root/'supervisor').mkdir();cmd=[sys.executable,'-m','polim_cpu.path_profile_capture_v1.fixture03','worker',str(root/'native')]
    save(root/'supervisor/launch.json',dict(command=cmd,model_work=0));t=time.perf_counter()
    with (root/'supervisor/worker.log').open('x') as f:
        with mark('fixture:supervisor:worker'):
            p=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,creationflags=0x08000000)
            try:p.wait(150)
            except subprocess.TimeoutExpired:subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();raise
    save(root/'supervisor/result.json',dict(exit_code=p.returncode,pid=os.getpid(),worker_pid=p.pid,seconds=time.perf_counter()-t,model_work=0));assert p.returncode==0
if __name__=='__main__':
    mode,arg=sys.argv[1:3]
    if mode=='device':device(arg)
    elif mode=='worker':worker(arg)
    elif mode=='supervisor':supervisor(arg)
    else:raise ValueError(mode)
