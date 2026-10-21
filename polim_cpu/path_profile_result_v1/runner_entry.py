"""Independent observer entry, original current-J scientific runner."""
import sys,json
from pathlib import Path
from . import observe
cfgpath=Path(sys.argv[1]);cfg=json.loads(cfgpath.read_text());observe.init('CPU',Path(cfg['root'])/'trace')
with observe.span('import-and-bind-original-CPU-runner'):
    from . import control
    observe.cpu_hooks(control)
    from polim_cpu.anchor_efficiency_v1 import runner
    for key in ('ROOT','RUN','PLAN','INPUT','read','write','arrays','digest','ahash','gate'):setattr(runner,key,getattr(control,key))
with observe.span('scientific-CPU-command'):runner.run(cfgpath)
