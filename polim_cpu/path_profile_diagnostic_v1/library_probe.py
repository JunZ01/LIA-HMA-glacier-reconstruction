"""No-model DLL identity and CUPTI subscription check; standalone isolated child."""
import ctypes as C,json,os,sys,time,traceback,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'polim_cpu/runs/v59r1-profile-diagnostic-v1'
def save(name,data):
    with (OUT/name).open('x',encoding='utf8') as f:json.dump(data,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
def modules():
    k=C.WinDLL('kernel32',use_last_error=True);k.GetCurrentProcess.restype=C.c_void_p;p=k.GetCurrentProcess()
    enum=k.K32EnumProcessModules;enum.argtypes=[C.c_void_p,C.POINTER(C.c_void_p),C.c_uint32,C.POINTER(C.c_uint32)];enum.restype=C.c_int
    name=k.K32GetModuleFileNameExW;name.argtypes=[C.c_void_p,C.c_void_p,C.c_wchar_p,C.c_uint32];name.restype=C.c_uint32
    buf=(C.c_void_p*4096)();needed=C.c_uint32();assert enum(p,buf,C.sizeof(buf),C.byref(needed))
    result=[]
    for h in buf[:needed.value//C.sizeof(C.c_void_p)]:
        s=C.create_unicode_buffer(32768);assert name(p,h,s,len(s));v=s.value
        if any(x in Path(v).name.lower() for x in ('cupti','cuda','cublas','torch_cuda','kineto')):result.append(v)
    return result
def child():
    start=time.perf_counter();report=dict(model_work=dict(BP=0,residual=0,assembly=0,linear=0,time_steps=0),pid=os.getpid(),executable=sys.executable)
    try:
        import torch
        report.update(torch=str(torch.__version__),cuda=torch.version.cuda,loaded_after_import=modules())
        torch.cuda.init()
        report.update(device=torch.cuda.get_device_name(0),device_count=torch.cuda.device_count(),current_device=torch.cuda.current_device(),capability=list(torch.cuda.get_device_capability(0)),CUDA_initializations=1,loaded_after_cuda_init=modules())
        libs=[p for p in report['loaded_after_cuda_init'] if 'cupti' in Path(p).name.lower()]
        assert len(libs)==1,libs
        p=Path(libs[0]);report['actual_cupti']=dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        save('library-identity.json',report)
        lib=C.WinDLL(str(p));sub=lib.cuptiSubscribe
        callbacktype=C.WINFUNCTYPE(None,C.c_void_p,C.c_uint32,C.c_uint32,C.c_void_p)
        callback=callbacktype(lambda userdata,domain,cbid,data:None);handle=C.c_void_p()
        sub.argtypes=[C.POINTER(C.c_void_p),callbacktype,C.c_void_p];sub.restype=C.c_int
        code=int(sub(C.byref(handle),callback,None));report['subscription_result']=code
        report['subscription_after_explicit_CUDA_initialization']=True
        save('subscription.json',dict(code=code,code_2_meaning='CUPTI_ERROR_INVALID_DEVICE',loaded_library=report['actual_cupti'],CUDA_initialized_before_call=True))
        if code!=0:
            report.update(status='blocked-at-direct-cupti-subscription',further_cupti_calls=False)
            return
        unsub=lib.cuptiUnsubscribe;unsub.argtypes=[C.c_void_p];unsub.restype=C.c_int
        assert unsub(handle)==0
        # Only reached if raw subscription succeeds. A no-model graph checks the collector.
        with torch.autograd.profiler.profile(use_device='cuda',use_kineto=True,use_cpu=True) as prof:
            x=torch.ones(512,device='cuda',dtype=torch.float64);y=torch.empty_like(x)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):torch.mul(x,2,out=y)
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):torch.mul(x,2,out=y)
            graph.replay();torch.cuda.synchronize();v=float(y.sum().item())
        prof.export_chrome_trace(str(OUT/'graph-trace.json'))
        tr=json.loads((OUT/'graph-trace.json').read_text());ev=tr['traceEvents']
        report.update(sum=v,kernels=sum(e.get('cat')=='kernel' for e in ev),graph_calls=sum('GraphLaunch' in e.get('name','') for e in ev),status='trace-generated')
    except BaseException as ex:
        report.update(status='technical-error',error=repr(ex),traceback=traceback.format_exc());raise
    finally:
        report['seconds']=time.perf_counter()-start;save('child-result.json',report)
def parent():
    from polim_cpu.path_profile_v1.probe import env_gpu
    cmd=['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.path_profile_diagnostic_v1.library_probe','child']
    save('launch.json',dict(command=cmd,scope='no-model CUDA initialization and direct CUPTI subscription; graph only if subscription succeeds',timeout_seconds=90,environment_source='unchanged path_profile_v1.probe.env_gpu'))
    start=time.perf_counter()
    with (OUT/'child.log').open('x') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env_gpu(),stdout=log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        try:p.wait(90)
        except subprocess.TimeoutExpired:p.terminate();p.wait(10)
    save('parent-result.json',dict(returncode=p.returncode,seconds=time.perf_counter()-start,pid=p.pid,model_work=0))
    print((OUT/'child-result.json').read_text() if (OUT/'child-result.json').exists() else 'Child report unavailable; see parent result/log')
if __name__=='__main__':
    if len(sys.argv)>1:child()
    else:parent()
