"""Small supervised reduced-order neural predictor; isolated torch runtime."""
import os, sys, time, json
from pathlib import Path
assert not os.environ.get('KMP_DUPLICATE_LIB_OK')
import numpy as np
import torch


def main(root):
    start = time.perf_counter(); torch.set_num_threads(1); torch.manual_seed(2387)
    torch.use_deterministic_algorithms(True)
    z = np.load(root/'training.npz'); n = int(z['ntrain'])
    x = torch.from_numpy(z['x']); y = torch.from_numpy(z['y'])
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.Tanh(),
             torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,y.shape[1])).double()
    opt = torch.optim.Adam(model.parameters(),lr=.002)
    best = float('inf'); history = []; best_epoch = 0
    for epoch in range(1,1201):
        opt.zero_grad(); loss = ((model(x[:n])-y[:n])**2).mean(); loss.backward(); opt.step()
        if epoch % 25 == 0:
            with torch.no_grad(): val = float(((model(x[n:])-y[n:])**2).mean())
            history.append(dict(epoch=epoch,train=float(loss.detach()),validation=val))
            if val < best:
                best=val; best_epoch=epoch; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    weights = {}
    for i,j in enumerate([0,2,4]):
        weights[f'w{i}']=model[j].weight.detach().numpy();weights[f'b{i}']=model[j].bias.detach().numpy()
    with (root/'network.npz').open('xb') as f: np.savez(f,**weights)
    q=z['x'].copy()
    for i in range(3):
        q=q@weights[f'w{i}'].T+weights[f'b{i}']
        if i<2:q=np.tanh(q)
    with torch.no_grad(): delta=float(np.max(np.abs(q-model(x).numpy())))
    assert delta<1e-10
    report=dict(torch=torch.__version__,device='CPU',threads=1,seed=2387,epochs=1200,
                selected_epoch=best_epoch,validation_mse=best,training_seconds=time.perf_counter()-start,
                parameters=sum(p.numel() for p in model.parameters()),numpy_export_max_difference=delta,history=history)
    (root/'training-result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)


if __name__=='__main__':main(Path(sys.argv[1]))
