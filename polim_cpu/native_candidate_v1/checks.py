"""Small synthetic data and mock processes; zero glacier model work."""
from .control import ROOT,E,write,read
from .state import seal,load_bundle,ledger
from polim_cpu import packed_storage_v1 as st
from polim_cpu.c_control_repair_v1.process import launch
import tempfile,time,sys,json,ast,subprocess
from pathlib import Path
import numpy as np

def run():
    t=time.perf_counter();tests={};records=[]
    with tempfile.TemporaryDirectory(prefix='v62-prep-') as name:
        root=Path(name)
        for mode in ('outer','outer-fail'):
            p=root/mode;(p/'supervisor').mkdir(parents=True);(p/'native').mkdir()
            r=launch([sys.executable,'-m','polim_cpu.c_control_repair_v1.process',str(p),mode],p/'supervisor',p,time.time()+20)
            tests[mode]=(r['exit_code']==0)==(mode=='outer');tests[mode+'-layout']=all((p/k/f).exists() for k in ('native','supervisor') for f in ('worker.log','launch.json'));records.append(r)
        p=root/'collision';p.mkdir();(p/'launch.json').write_text('{}')
        r=launch([sys.executable,'-c','raise Exception("must not run")'],p,p,time.time()+10);tests['no-overwrite']=r['exit_code']==-2 and 'pid' not in r
        h=np.array([1.,2.]);u=np.zeros(8);meta=dict(year=1553.,step=34,last_solve=1553.,V0=4.,smb_volume=1.,exported=0.,row=dict(BP_residual=1e-7,mass_relative=0.),window_rows=[dict(year=1553.)]);st.save(root,'frame-000034',h,u,meta,'synthetic')
        ident=dict(grid='synthetic',physics='synthetic');l=ledger(4.,5.,1.,0.)
        b=seal(root/'accepted.json',root/'frame-000034','synthetic',ident,l,dict(year=1552,V0=4))
        a,m,bb=load_bundle(root/'accepted.json',ident)
        tests['full-native-recovery']=np.array_equal(a['h'],h) and np.array_equal(a['u'],u) and m==b['metadata'] and bb['window_ledger']==l
        tests['candidate-rejected']=False
        d=dict(b,kind='unaccepted-candidate');st.write(root/'candidate.json',d)
        try:load_bundle(root/'candidate.json',ident)
        except ValueError:tests['candidate-rejected']=True
        tests['identity-rejected']=False
        try:load_bundle(root/'accepted.json',dict(grid='wrong'))
        except ValueError:tests['identity-rejected']=True
        tests['year-ledger-continuation']=ledger(l['V0'],6.,l['source']+1.,0.)['mass_relative']==0
        for status in ('completed','numerical-stop','environment-or-technical-error','input-error'):
            p=root/(status+'.json');st.write(p,dict(status=status,accepted=status=='completed'));tests[status+'-serialization']=st.read(p)['status']==status
        from polim_cpu.historical_climate_v1 import annual
        from types import SimpleNamespace
        climate=dict(years=np.array([1552,1553]),temp=np.zeros((2,12)),prcp=np.ones((2,12)),gradient=np.full((2,12),-.0065),ref_hgt=0.)
        params=dict(base_temp_bias_degC=0.,melt_f=10.,prcp_fac=3.,temp_melt_degC=-1.)
        backend=SimpleNamespace(xp=np,array=np.asarray)
        tests['annual-current-surface']=not np.array_equal(annual(h,climate,params,1553,backend),annual(h+20,climate,params,1553,backend))
        tests['missing-year-no-repeat']=False
        try:annual(h,climate,params,1554,backend)
        except ValueError:tests['missing-year-no-repeat']=True
        from .entry import check_config
        for case in ('A','R'):tests['config-'+case]=check_config(Path(__file__).parent/'configs'/f'{case}.json')['passed']
        bad=read(Path(__file__).parent/'configs/A.json');bad['end_year']=1555;st.write(root/'bad-config.json',bad)
        try:check_config(root/'bad-config.json');tests['year-parameter-rejected']=False
        except ValueError:tests['year-parameter-rejected']=True
        src=(Path(__file__).parent/'worker.py').read_text();orig=(ROOT/'polim_cpu/cn_validation_v1/native_v35.py').read_text()
        # All original solve, transport, stepping and acceptance statements are identical.
        lo=src.index('        # Candidate adapter:');hi=src.index('        actual_nodes=torch.zeros_like(h)',lo)
        restored=src[:lo]+"        smb=tensor(st.arrays(Path(cfg['shared_forcing']))['smb'])\n"+src[hi:]
        tests['only-climate-and-entry-audit-adapter']=restored==orig
        tests['original-native-acceptance-and-solver']=all(k in src for k in ['from polim_cpu.adaptive_bp_solve_v1 import solve','assert final_bp<=1e-6','assert mass<=1e-10'])
        r=subprocess.run([sys.executable,'-m','polim_cpu.native_candidate_v1.entry','check',str(Path(__file__).parent/'configs/A.json')],cwd=ROOT,capture_output=True,text=True)
        tests['actual-CLI-check']=r.returncode==0 and json.loads(r.stdout)['passed'];records.append(dict(command=r.args,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
        tests['syntax']=True
    result=dict(passed=all(tests.values()),tests=tests,processes=records,seconds=time.perf_counter()-t,real_BP=0,real_transport=0,real_residual=0,real_matrix=0,real_steps=0,synthetic_climate_only=True)
    write(E/'preparation.json',result)
    assert result['passed'],tests
if __name__=='__main__':run()
