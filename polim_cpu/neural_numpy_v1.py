"""Dependency-free float64 MLP/Adam and the registered BP continuation."""
from . import neural_predictor_v1 as n
import shutil
OLD_E=n.E.with_name('polim_v26r2_neural_predictor_v1');OLD_RUN=n.RUN.with_name('v26r2-neural-predictor-v1')
NAME='polim_v26numpy_neural_predictor_v1';E=OLD_E.with_name(NAME);RUN=OLD_RUN.with_name('v26numpy-neural-predictor-v1')
def configure():n.NAME=NAME;n.E=E;n.RUN=RUN

def register():
    n.NAME='polim_v26r2_neural_predictor_v1';n.E=OLD_E;n.RUN=OLD_RUN;n.validate()
    failure=dict(status='technical-failure-before-training',cause='Existing torch/SS4 loads duplicate OpenMP runtime at CPU training initialization. No trained model or BP advancement. Unsafe override never enabled in torch child.',
        action='Stop torch dependency repair. New disclosed NumPy float64 implementation of same MLP/Adam; seed2387 now NumPy PCG64 so initialization differs. One directional finite-difference backprop check.',
        user_authorization='User requested testing and explicitly allows continuing after errors; implementation version changes disclosed before first training/physics run.')
    n.write_json(OLD_E/'technical-failure.json',failure);n.transition('technical-failure',OLD_E/'technical-failure.json')
    configure();E.mkdir(exist_ok=False);p=n.read(OLD_E/'plan.frozen.json')
    p.update(schema=NAME,backend_change=failure)
    p['implementation'][(n.R/'neural_numpy_v1.py').relative_to(n.PROJECT).as_posix()]=n.digest(__file__)
    p['resources']['training_backend']='NumPy float64,BLAS1,no torch or GPU'
    p['method']+=' Backend now pure NumPy; Kaiming uniform +/-1/sqrt(fan_in),PCG64 seed2387,Adam betas0.9/0.999 eps1e-8,fullbatch MSE.'
    n.write_json(E/'plan.frozen.json',p);n.transition('authorized',E/'plan.frozen.json')

def train():
    np=n.np;start=n.time.perf_counter();z=n.arrays(RUN/'training.npz');nt=int(z['ntrain']);x=z['x'];y=z['y'];rng=np.random.default_rng(2387)
    dims=[x.shape[1],64,64,y.shape[1]];params=[]
    for a,b in zip(dims[:-1],dims[1:]):
        params.extend([rng.uniform(-1/np.sqrt(a),1/np.sqrt(a),(b,a)),rng.uniform(-1/np.sqrt(a),1/np.sqrt(a),b)])
    def forward(xx,pp):
        acts=[xx]
        for i in range(3):
            q=acts[-1]@pp[2*i].T+pp[2*i+1];acts.append(np.tanh(q) if i<2 else q)
        return acts
    def lossgrad(xx,yy,pp):
        a=forward(xx,pp);d=2*(a[-1]-yy)/yy.size;grads=[None]*6
        for i in range(2,-1,-1):
            grads[2*i]=d.T@a[i];grads[2*i+1]=d.sum(0)
            if i:d=(d@pp[2*i])*(1-a[i]**2)
        return float(np.mean((a[-1]-yy)**2)),grads
    loss,g=lossgrad(x[:nt],y[:nt],params);direction=[rng.normal(size=p.shape) for p in params];norm=np.sqrt(sum(np.sum(d*d) for d in direction));direction=[d/norm for d in direction]
    analytical=sum(float(np.sum(a*b)) for a,b in zip(g,direction));eps=1e-5
    plus=lossgrad(x[:nt],y[:nt],[p+eps*d for p,d in zip(params,direction)])[0]
    minus=lossgrad(x[:nt],y[:nt],[p-eps*d for p,d in zip(params,direction)])[0]
    fd=(plus-minus)/(2*eps);error=abs(fd-analytical)/max(abs(fd),abs(analytical),1e-10);assert error<1e-5,error
    m=[np.zeros_like(p) for p in params];v=[p.copy() for p in m];best=float('inf');history=[]
    for epoch in range(1,1201):
        loss,g=lossgrad(x[:nt],y[:nt],params)
        for i in range(6):
            m[i]=.9*m[i]+.1*g[i];v[i]=.999*v[i]+.001*g[i]**2
            params[i]-=.002*(m[i]/(1-.9**epoch))/(np.sqrt(v[i]/(1-.999**epoch))+1e-8)
        if epoch%25==0:
            val=float(np.mean((forward(x[nt:],params)[-1]-y[nt:])**2));history.append(dict(epoch=epoch,train=loss,validation=val))
            if val<best:best=val;best_epoch=epoch;selected=[p.copy() for p in params]
    weights={}
    for i in range(3):weights[f'w{i}']=selected[2*i];weights[f'b{i}']=selected[2*i+1]
    with (RUN/'network.npz').open('xb') as f:np.savez(f,**weights)
    reloaded=n.arrays(RUN/'network.npz');assert all(np.array_equal(reloaded[k],v) for k,v in weights.items())
    result=dict(backend='NumPy float64 MLP Adam',device='CPU',threads=1,seed=2387,epochs=1200,selected_epoch=best_epoch,
        validation_mse=best,training_seconds=n.time.perf_counter()-start,parameters=sum(p.size for p in params),
        directional_gradient_relative_error=error,exact_weight_reload=True,history=history)
    n.write_json(RUN/'training-result.json',result);n.write_json(RUN/'training-process.json',dict(seconds=n.time.perf_counter()-start,returncode=0,backend='NumPy in-process'))
    n.write_json(RUN/'model-seal.json',dict(files={p:n.digest(RUN/p) for p in ['basis.npz','network.npz','training.npz','training-result.json']}))
    print(n.json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True)

def run():
    configure();n.validate();p=n.read(E/'plan.frozen.json')
    assert n.read(n.PROJECT/'LIA_CURRENT_STATE.json')['research_branches'][NAME]['launch_authorized']
    for f,h in {**p['input_hashes'],**p['implementation']}.items():assert n.digest(n.PROJECT/f)==h
    RUN.mkdir(exist_ok=False);n.write_json(E/f'fingerprint-{n.seal(p)}.json',dict(output=str(RUN)))
    n.write_json(RUN/'launch.json',dict(pid=n.os.getpid(),start=n.datetime.now(n.timezone.utc).isoformat(),command='python -m polim_cpu.neural_numpy_v1 run',resources=p['resources'],checkpoint='every pending/accepted/Newton state',stop='900s finite test'))
    for f in ['basis.npz','training.npz','dataset-result.json']:shutil.copyfile(OLD_RUN/f,RUN/f)
    start=n.time.perf_counter();train();result=n.case()
    out=dict(status='completed',continuation=result,dataset=n.read(RUN/'dataset-result.json'),training=n.read(RUN/'training-result.json'),training_process=n.read(RUN/'training-process.json'),total_elapsed_seconds=n.time.perf_counter()-start)
    out['amortized_test_seconds']=out['dataset']['seconds']+out['training_process']['seconds']+result['wall_seconds']
    n.write_json(RUN/'result.json',out)

if __name__=='__main__':
    if n.sys.argv[1]=='register':register()
    elif n.sys.argv[1]=='run':run()
