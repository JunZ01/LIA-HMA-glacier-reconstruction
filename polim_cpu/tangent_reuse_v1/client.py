"""CPU outer Newton/transport with a stateful true CUDA dynamics service."""
import os,sys,time,json,subprocess,hashlib
from pathlib import Path
from multiprocessing.connection import Listener
from types import SimpleNamespace
from contextlib import contextmanager
import numpy as np
from polim_cpu.geometry_probe_v2.common import ROOT,INPUT,write
from polim_cpu.geometry_coupled_v1.backend import Work as BaseWork

class Work(BaseWork):
    def __init__(self,*a,status_block=False,**kw):super().__init__(*a,**kw);self.status_block=status_block;self.fallbacks=0
    def progress(self):
        if self.status_block:
            # Actual destination is made a directory, so Windows cannot replace it.
            path=self.root/'work.json'
            if not path.exists():path.mkdir()
            super().progress();self.fallbacks+=1
        else:super().progress()

class Backend:
    def __init__(self,z,w,root,prepare_only=False,reuse=True):
        self.w=w;self.mesh=dict(bed=z['bed'],xy=z['xy'],triangles=z['tri']);self.cursor=dict(counts={},seconds={},events=[]);self.current=None
        self.root=Path(root);self.root.mkdir();self.listener=Listener(('127.0.0.1',0),authkey=(key:=os.urandom(24)))
        cfg=dict(root=str(self.root),address=self.listener.address,auth=key.hex(),input=str(INPUT),prepare_only=prepare_only,reuse=reuse)
        path=self.root/'service.json';write(path,cfg);env=os.environ.copy();site=ROOT/'.runtime/torch-cu128-v1/site-packages'
        env.pop('KMP_DUPLICATE_LIB_OK',None);env.pop('PYTHONHOME',None)
        env.update(POLIM_TORCH_SITE=str(site),PYTHONPATH=str(site)+os.pathsep+str(ROOT),PATH=str(site/'torch/lib')+os.pathsep+'C:/Windows/System32;C:/Windows',CUBLAS_WORKSPACE_CONFIG=':4096:8')
        self.log=(self.root/'service.log').open('x');t=time.perf_counter()
        self.process=subprocess.Popen(['D:/Anaconda/envs/SS4/python.exe','-m','polim_cpu.tangent_reuse_v1.service',str(path)],cwd=ROOT,env=env,stdout=self.log,stderr=subprocess.STDOUT,creationflags=0x08000000)
        self.listener._listener._socket.settimeout(40)
        self.conn=self.listener.accept();hello=self.conn.recv()
        if not hello.get('ok'):raise RuntimeError(hello)
        self.environment={k:v for k,v in hello.items() if k!='meter'};self.import_meter(hello['meter']);self.w.times['GPU_service_startup']=time.perf_counter()-t
        write(self.root/'environment.json',self.environment)
    def import_meter(self,m):
        for k,v in m['counts'].items():self.w.counts[k]+=v-self.cursor['counts'].get(k,0)
        for k,v in m['seconds'].items():self.w.times[k]+=v-self.cursor['seconds'].get(k,0)
        for e in m['events'][len(self.cursor['events']):]:self.w.event('GPU-'+e['event'],**{k:v for k,v in e.items() if k!='event'})
        self.cursor=m
    def rpc(self,kind,**kw):
        self.w.tick();t=time.perf_counter();self.w.counts['IPC_calls']+=1
        self.conn.send(dict(kind=kind,**kw))
        if not self.conn.poll(max(.001,self.w.deadline-time.time()) if self.w.deadline else 120):raise TimeoutError('GPU IPC deadline')
        msg=self.conn.recv();self.w.times['IPC_roundtrip']+=time.perf_counter()-t
        if 'meter' in msg:self.import_meter(msg['meter'])
        if not msg.get('ok'):raise RuntimeError(msg)
        return msg['result']
    def begin(self,w):self.w=w;self.rpc('begin',deadline=w.deadline)
    def evaluate(self,h,u,tangent=False):
        r=self.rpc('bind',H=h,u=u,linear=tangent);self.current=r['point'];self.current_arrays=(h.copy(),u.copy())
        return r['F'],None,r['force'],dict(wet=r['wet'])
    def full(self,h,u,kind='quarter',tol=1e-6):
        assert tol==1e-6
        with self.w.span('quarter_BP',8):r=self.rpc('full',H=h,u=u)
        self.current=r['point'];self.current_arrays=(h.copy(),r['u'].copy());self.w.event('BP-complete',info=r['info']);self.w.save(f'BP-{self.w.counts["full_BP"]:02d}-terminal',H=h,u=r['u'])
        return r['u'],r['info']
    def linearize(self,h,u):
        F,_,f,a=self.evaluate(h,u,True);return Linearization(self,h.copy(),u.copy(),F,f,a,self.current)
    def ru(self,h,u,du,weights):
        if not (np.array_equal(h,self.current_arrays[0]) and np.array_equal(u,self.current_arrays[1])):raise RuntimeError('Ru must use current linearization state')
        return self.rpc('ru',point=self.current,du=du,weights=weights)['value']
    @contextmanager
    def measured(self):yield
    def close(self):
        t=time.perf_counter()
        try:
            try:self.conn.send(dict(kind='close'))
            except (EOFError,BrokenPipeError,OSError):pass
            self.conn.close()
            try:self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill','/PID',str(self.process.pid),'/T','/F'],capture_output=True);self.process.wait();self.w.event('GPU-shutdown-timeout')
        finally:self.listener.close();self.log.close();self.w.times['GPU_service_shutdown']+=time.perf_counter()-t

class Linearization:
    def __init__(self,b,h,u,F,f,a,point):self.__dict__.update(locals());del self.self
    def ensure(self):
        if self.b.current!=self.point:
            self.b.evaluate(self.h,self.u,True)
            if self.b.current!=self.point:raise RuntimeError('Linearization cache identity mismatch')
    def fh(self,v):self.ensure();return self.b.rpc('fh',point=self.point,dH=v)['value']
    def solve(self,rhs,label):
        self.ensure();before=self.b.w.counts['tangent_Krylov'];r=self.b.rpc('solve',point=self.point,rhs=rhs)
        self.b.w.event('tangent',label=label,iterations=self.b.w.counts['tangent_Krylov']-before,status=0,true_relative=None);return r['value']
    def feedback(self,v,weights):
        self.ensure();before=self.b.w.counts['tangent_Krylov'];r=self.b.rpc('feedback',point=self.point,dH=v,weights=weights)
        self.b.w.event('tangent',label='fused FH/tangent/Ru',iterations=self.b.w.counts['tangent_Krylov']-before,status=0,true_relative=None)
        return r['du'],r['Ru']
