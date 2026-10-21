"""Same CPU runner and management for both cold variants; only GPU B flag differs."""
import sys,json,os
from pathlib import Path
from . import runner,control
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
assert cfg['variant'] in ('baseline','candidate')
os.environ['POLIM_BATCH_LOCAL_B']='1' if cfg['variant']=='candidate' else '0'
runner.RUN=control.RUN/cfg['variant']
runner.run(sys.argv[1])
