"""Observe the original service; no scientific equations or policies changed."""
import sys,json
from pathlib import Path
from . import observe
cfg=json.loads(Path(sys.argv[1]).read_text());observe.init('GPU',Path(cfg['root'])/'trace')
with observe.span('import-original-GPU-service'):
    from polim_cpu.anchor_efficiency_v1 import service
observe.gpu_hooks()
with observe.span('GPU-service-command'):service.run(sys.argv[1])
