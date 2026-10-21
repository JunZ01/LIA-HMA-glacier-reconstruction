"""Render contact sheets for scientific-figure layout QA, not a new analysis."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parent
files=sorted((OUT/'glaciers').glob('*.png'),key=lambda p:int(p.stem))
qa=OUT/'visual_qa';qa.mkdir(exist_ok=True)
for offset in range(0,len(files),9):
    fig,axes=plt.subplots(3,3,figsize=(18,16.2))
    for ax in axes.ravel():ax.axis('off')
    for ax,p in zip(axes.ravel(),files[offset:offset+9]):
        ax.imshow(plt.imread(p));ax.set_title(p.stem,fontsize=12,pad=0)
    fig.subplots_adjust(left=.005,right=.995,bottom=.005,top=.98,hspace=.04,wspace=.03)
    fig.savefig(qa/f'sheet_{offset//9+1:02}.png',dpi=110)
    plt.close(fig)
print(f'{len(files)} individual figures in {(len(files)+8)//9} review sheets')
manifest=OUT/'output_manifest.json'
if manifest.exists():
    hashes=json.loads(manifest.read_text(encoding='utf-8'))
    for p in list(qa.glob('*.png'))+[Path(__file__)]:
        hashes[p.relative_to(OUT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest().upper()
    manifest.write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding='utf-8')
