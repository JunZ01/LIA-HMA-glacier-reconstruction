"""No-model three-process profiler capability check. No project model imports."""
import os,sys,json,time,traceback,subprocess,threading
from pathlib import Path
from multiprocessing.connection import Listener,Client
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'polim_cpu/runs/v59-path-profile-v1/preparation'
SITE=ROOT/'.runtime/torch-cu128-v1/site-packages'
events=[]
def mark(name,begin,end,**kw):
    if len(events)>=10000:raise RuntimeError('Bounded probe trace exceeded')
    events.append(dict(name=name,pid=os.getpid(),tid=threading.get_native_id(),start_unix_ns=begin,end_unix_ns=end,**kw))
def write(p,obj):
    with Path(p).open('x',encoding='utf8') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
def env_gpu():
    e=os.environ.copy()
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):e[k]='1'
    e.pop('KMP_DUPLICATE_LIB_OK',None);e.pop('PYTHONHOME',None)
    e.update(POLIM_TORCH_SITE=str(SITE),PYTHONPATH=str(SITE)+os.pathsep+str(ROOT),PATH=str(SITE/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return e
def device(config):
    cfg=json.loads(Path(config).read_text());out=Path(cfg['root']);conn=None;prof=None;info={};start=time.time_ns()
    try:
        conn=Client(tuple(cfg['address']),authkey=bytes.fromhex(cfg['auth']))
        conn.send(dict(ok=True,stage='boot',pid=os.getpid()))
        write(out/'before-torch-import.json',dict(pid=os.getpid(),wall=time.time(),executable=sys.executable,sys_path=sys.path))
        import torch
        write(out/'after-torch-import.json',dict(pid=os.getpid(),wall=time.time(),version=str(torch.__version__)))
        info.update(python=sys.version,executable=sys.executable,torch=str(torch.__version__),cuda=str(torch.version.cuda),supported_activities=[str(x) for x in torch.profiler.supported_activities()],kineto_available=torch.autograd.kineto_available())
        write(out/'profiler-availability.json',info)
        # CUPTI/Kineto maps GPU times to its CPU clock. Export remains raw; no manual CUDA-Event stitching.
        prof=torch.autograd.profiler.profile(use_device='cuda',use_kineto=True,use_cpu=True,record_shapes=False,profile_memory=False,with_stack=False)
        prof.__enter__()
        with torch.profiler.record_function('probe:init-and-graph'):
            torch.cuda.init();info['device']=torch.cuda.get_device_name();info['CUDA_initializations']=1
            stream=torch.cuda.Stream();x=torch.ones(512,device='cuda',dtype=torch.float64);y=torch.empty_like(x)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):torch.mul(x,2,out=y)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):torch.mul(x,2,out=y)
        conn.send(dict(ok=True,info=info))
        while True:
            a=time.time_ns();req=conn.recv();mark('receive',a,time.time_ns(),request_id=req['request_id']);rid=req['request_id'];kind=req['kind']
            if kind=='close':break
            a=time.time_ns()
            try:
                with torch.profiler.record_function('request:'+rid+':'+kind):
                    if kind=='intentional-error':raise ValueError('intentional no-model exception')
                    graph.replay();torch.cuda.synchronize();result=y.cpu().tolist()
                reply=dict(ok=True,request_id=rid,sum=sum(result),count=len(result))
            except Exception as exc:
                reply=dict(ok=False,request_id=rid,error=repr(exc));write(out/('device-error-'+rid+'.json'),dict(reply,traceback=traceback.format_exc()))
            mark('service-handling',a,time.time_ns(),request_id=rid)
            a=time.time_ns();conn.send(reply);mark('send',a,time.time_ns(),request_id=rid)
        torch.cuda.synchronize();prof.__exit__(None,None,None);prof.export_chrome_trace(str(out/'kineto-raw.json'));prof=None
        info.update(graph_builds=1,graph_replays=1,model_BP=0,model_residual=0,model_assembly=0,model_linear=0,time_steps=0)
        write(out/'device-result.json',dict(status='completed',info=info))
    except BaseException as exc:
        write(out/'device-fatal.json',dict(error=repr(exc),traceback=traceback.format_exc(),info=info))
        if conn:
            try:conn.send(dict(ok=False,error=repr(exc)))
            except Exception:pass
        raise
    finally:
        if prof:
            try:prof.__exit__(None,None,None);prof.export_chrome_trace(str(out/'kineto-partial.json'))
            except Exception:pass
        if conn:conn.close()
        mark('device-lifetime',start,time.time_ns());write(out/'device-events.json',events)
def worker(out):
    out=Path(out);key=os.urandom(24);listener=Listener(('127.0.0.1',0),authkey=key);proc=None;conn=None;t=time.time_ns()
    cfg=dict(root=str(out),address=listener.address,auth=key.hex());write(out/'service.json',cfg)
    try:
        with (out/'device.log').open('x') as log:
            proc=subprocess.Popen(['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.path_profile_v1.probe','device',str(out/'service.json')],cwd=ROOT,env=env_gpu(),stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
            listener._listener._socket.settimeout(35);conn=listener.accept();boot=conn.recv();assert boot['ok'],boot
            assert conn.poll(120),'GPU capability initialization exceeded no-model allowance';hello=conn.recv();assert hello['ok'],hello
            replies=[]
            for i,kind in enumerate(['graph','intentional-error']):
                rid=f'r{i:03d}';a=time.time_ns();conn.send(dict(request_id=rid,kind=kind));mark('request-send',a,time.time_ns(),request_id=rid)
                a=time.time_ns();assert conn.poll(20);reply=conn.recv();mark('request-await-unpack',a,time.time_ns(),request_id=rid)
                assert reply['request_id']==rid;replies.append(reply)
            assert replies[0]['sum']==1024 and replies[0]['count']==512 and not replies[1]['ok']
            conn.send(dict(request_id='close',kind='close'));conn.close();conn=None;proc.wait(30);assert proc.returncode==0
        write(out/'worker-result.json',dict(status='pass',pid=os.getpid(),device_pid=proc.pid,hello=hello,replies=replies,exception_recorded=(out/'device-error-r001.json').exists()))
    except BaseException as exc:
        write(out/'worker-failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),device_pid=proc.pid if proc else None));raise
    finally:
        if conn:conn.close()
        listener.close()
        if proc and proc.poll() is None:proc.terminate();proc.wait(10)
        mark('worker-lifetime',t,time.time_ns());write(out/'worker-events.json',events)
def supervisor(name='kineto-probe'):
    out=RUN/name;out.mkdir();(out/'supervisor').mkdir();(out/'native').mkdir();native=out/'native';start=time.perf_counter();t=time.time_ns();cmd=[sys.executable,'-m','polim_cpu.path_profile_v1.probe','worker',str(native)]
    write(out/'supervisor/launch.json',dict(command=cmd,model_work_zero=True))
    with (out/'supervisor/worker.log').open('x') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        try:p.wait(180)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(p.pid),'/T','/F'],capture_output=True);p.wait();raise
    mark('supervisor-lifetime',t,time.time_ns());write(out/'supervisor/events.json',events)
    trace=native/'kineto-raw.json';caps=dict(process_tree_fixture=p.returncode==0,raw_trace_exists=trace.exists(),cuda_kernel_events=0,cuda_graph_runtime_events=0,correlated_GPU_events=0)
    if trace.exists():
        tr=json.loads(trace.read_text());ev=tr.get('traceEvents',[])
        caps.update(trace_keys=list(tr),event_categories=sorted({str(e.get('cat')) for e in ev}),baseTimeNanoseconds=tr.get('baseTimeNanoseconds'))
        caps['cuda_kernel_events']=sum(e.get('cat')=='kernel' for e in ev)
        caps['cuda_graph_runtime_events']=sum('cudaGraphLaunch' in str(e.get('name')) or 'cuGraphLaunch' in str(e.get('name')) for e in ev)
        caps['correlated_GPU_events']=sum(e.get('cat') in ['kernel','gpu_memcpy'] and ('correlation' in e.get('args',{}) or 'External id' in e.get('args',{})) for e in ev)
    passed=caps['process_tree_fixture'] and caps['cuda_kernel_events']>0 and caps['cuda_graph_runtime_events']>0 and caps['correlated_GPU_events']>0 and caps.get('baseTimeNanoseconds') is not None
    write(out/'capability.json',dict(status='pass' if passed else 'blocked-insufficient-trace',exit_code=p.returncode,seconds=time.perf_counter()-start,capabilities=caps))
    print(json.dumps(dict(status='pass' if passed else 'blocked-insufficient-trace',seconds=time.perf_counter()-start,capabilities=caps)))
if __name__=='__main__':
    if len(sys.argv)==1:supervisor()
    elif sys.argv[1]=='worker':worker(sys.argv[2])
    elif sys.argv[1]=='device':device(sys.argv[2])
    elif sys.argv[1]=='check':supervisor(sys.argv[2])
