"""Read only the two selected published LMLa outputs; execute no archive code."""
import io,json,hashlib,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

with zipfile.ZipFile(ROOT/'supplement.zip') as outer:
    with zipfile.ZipFile(io.BytesIO(outer.read('tc-2007-0019-sp2.zip'))) as archive:
        for name in ('rhi2a020','fpa1e000'):
            member=f'ismip_all/{name[:4]}/{name}.txt'
            with (ROOT/f'{name}.raw.txt').open('xb') as f:f.write(archive.read(member))
a=np.loadtxt(ROOT/'rhi2a020.raw.txt');xs=np.unique(a[:,0]);ys=np.unique(a[:,1])
assert a.shape==(len(xs)*len(ys),8) and xs.min()>=0 and xs.max()<=1
order=np.lexsort((a[:,0],a[:,1]));a=a[order].reshape(len(ys),len(xs),8)
assert np.all(a[:,:,0]==xs) and np.all(a[:,:,1]==ys[:,None])
j=np.searchsorted(ys,.25);w=(.25-ys[j-1])/(ys[j]-ys[j-1])
uv=(1-w)*a[j-1,:,2:4]+w*a[j,:,2:4]
with (ROOT/'rhi2a020.txt').open('x') as f:np.savetxt(f,np.column_stack([xs*20000,np.linalg.norm(uv,axis=1)]),header='x_m horizontal_surface_speed_m_year; published RHI2 BP; interpolate velocity components to y/L=0.25')
e=np.loadtxt(ROOT/'fpa1e000.raw.txt');assert e.shape[1]==5 and e[:,0].min()==0 and e[:,0].max()==5000
with (ROOT/'fpa1e000.txt').open('x') as f:np.savetxt(f,np.column_stack([e[:,0],abs(e[:,1])]),header='x_m horizontal_surface_speed_m_year; published FPA1 BP; E000=no slip (paper E1)')
manifest=dict(publication='Pattyn et al. 2008, The Cryosphere 2,95-108, doi:10.5194/tc-2-95-2008',
    paper_url='https://tc.copernicus.org/articles/2/95/2008/tc-2-95-2008.pdf',
    archive_url='https://frank.pattyn.web.ulb.be/ismip/tc-2-95-2008-supplement.zip',
    geometry_url='https://frank.pattyn.web.ulb.be/ismip/arolla100.dat',
    protocol_url='https://frank.pattyn.web.ulb.be/ismip/ismiphom.pdf',
    license='Article CC BY 3.0; supplemental benchmark data attributed to their submitting authors',
    model_classification='Final paper Table3: rhi2=LMLa spectral and fpa1=LMLa finite difference. LMLa is classic Blatter/Pattyn. No external model code executed.',
    conversion=dict(A='Raw columns1-2 normalized x,y; columns3-4 horizontal surface velocity m/year. Linear component interpolation between bracketing y rows to y/L=.25, then Euclidean norm. x multiplied by20000m; no velocity scaling.',
        E='Raw column1 already actual x in0..5000m (despite generic protocol hat notation); column2 surface horizontal velocity m/year, absolute value. Filename000 is no slip;001 is zero-traction patch and excluded.'),
    geometry_notes='A uses final paper mean H1000m. Arolla data columns are x,bed,surface,flag as established by positive thickness; legacy protocol prose reverses bed/surface column labels.',
    source_hashes={p.name:sha(p) for p in [ROOT/'supplement.zip',ROOT/'arolla100.dat',ROOT/'rhi2a020.raw.txt',ROOT/'fpa1e000.raw.txt']},
    output_hashes={p.name:sha(p) for p in [ROOT/'rhi2a020.txt',ROOT/'fpa1e000.txt']},
    preparation_sha256=sha(Path(__file__)))
with (ROOT/'reference-manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
print(json.dumps(dict(A_shape=list(a.shape),A_profile_rows=len(xs),E_profile_rows=len(e),hashes=manifest['output_hashes'])))
