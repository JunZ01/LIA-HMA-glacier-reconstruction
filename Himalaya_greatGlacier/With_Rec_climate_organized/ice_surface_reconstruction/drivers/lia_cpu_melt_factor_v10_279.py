"""CPU20 v10.279: authorized melt0.7/1.3, full900 spinup then fixed forward.
Original dynamics/MB and six curve gates are reused. No GPU or automatic retry.
"""
from __future__ import annotations
import os
THREADS={k:"1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","NUMEXPR_MAX_THREADS","VECLIB_MAXIMUM_THREADS","BLIS_NUM_THREADS")}
for k,v in THREADS.items(): os.environ[k]=v
import argparse, copy, gzip, hashlib, json, math, pickle, shutil, subprocess, sys, time, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
DRIVERS=Path(__file__).resolve().parent
RES=ROOT/"validation_data/results/paleo_surface_validation_v0"
TX=RES/"lia_cpu_melt_factor_v10_279_transaction_v1"
OUT=ROOT/"w279"
PYTHON="D:/Anaconda/envs/oggm_env2/python.exe"
MODULE="Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_cpu_melt_factor_v10_279"
PARAMS=("delta_t_eq_degC","base_temp_bias_degC","melt_f","prcp_fac","glen_a_pa_3_s_1","fs","temp_melt_degC")
def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest().upper()
def ref(p):
    p=Path(p)
    return {"path":p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p),"sha256":sha(p)}
def check(r):
    if sha(ROOT/r["path"])!=r["sha256"].upper(): raise ValueError("hash-mismatch:"+r["path"])
def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if hasattr(v,"item"): return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v): return None
    return v
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("x",encoding="utf-8",newline="\n") as f:
        json.dump(clean(obj),f,ensure_ascii=False,allow_nan=False,indent=2);f.write("\n");f.flush();os.fsync(f.fileno())
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False).encode()).hexdigest().upper()
def rename(src,dst):
    for i in range(8):
        try: os.rename(src,dst);return
        except OSError:
            if i==7: raise
            time.sleep(min(.05*2**i,1))
def status(p,obj,replace=os.replace):
    errors=[]
    for i in range(4):
        temp=p.with_name(".t"+str(os.getpid())+"."+str(time.time_ns()))
        try: write(temp,obj);replace(temp,p);return p
        except OSError as e: errors.append(repr(e));time.sleep(.05*(i+1))
    try:
        q=p.parent/"fallback"/(p.stem+"."+str(time.time_ns())+".json")
        write(q,{"primary":str(p),"errors":errors,"payload":obj});return q
    except OSError as e:
        print("STATUS_UNAVAILABLE "+repr(e),file=sys.stderr,flush=True);return None
def event(obj):
    try:
        with (OUT/"r/events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(clean(obj),ensure_ascii=False)+"\n");f.flush()
    except OSError as e: print("EVENT_WARNING "+repr(e),file=sys.stderr,flush=True)
def contract():
    m=read(TX/"transaction_manifest.json")
    for r in m["code"]+[m["matrix"],m["inputs"],m["rules"],m["technical_tests"]]:check(r)
    return m
def row_for(stage,key):
    p=TX/"rows"/(key+".json") if stage=="spinup" else OUT/"q"/(key+".json")
    row=read(p)
    if row["key"]!=key or row["stage"]!=stage:raise ValueError("row-stage-key")
    check(row["inputs"])
    if stage=="spinup":
        if digest(row["fingerprint_payload"])!=row["fingerprint"]:raise ValueError("spin-fingerprint")
        for n in PARAMS:
            if row[n]!=row["fingerprint_payload"]["parameters"][n]:raise ValueError("row-parameter")
    else:
        if digest(row["fingerprint_payload"])!=row["fingerprint"]:raise ValueError("forward-fingerprint")
        for n in PARAMS:
            if row[n]!=row["fingerprint_payload"]["parameters"][n]:raise ValueError("forward-row-parameter")
        check(row["parent_result"]);check(row["source_checkpoint"])
        if not read(ROOT/row["parent_result"]["path"])["stable"]:raise ValueError("nonstable-parent")
    return row
def mb_for(row,stage):
    from oggm import cfg,utils
    from oggm.core.massbalance import MonthlyTIModel
    item=read(ROOT/row["inputs"]["path"])
    for r in item["files"]:check(r)
    cfg.PARAMS["temp_melt"]=row["temp_melt_degC"]
    gdir=utils.GlacierDirectory(row["rgi_id"],base_dir=str(OUT/"g"))
    start,end=(1441,1470) if stage=="spinup" else (1470,2000)
    suffix=item["spinup_suffix"] if stage=="spinup" else item["forward_suffix"]
    return MonthlyTIModel(gdir,filename="gcm_data",input_filesuffix=suffix,
        melt_f=row["melt_f"],prcp_fac=row["prcp_fac"],
        temp_bias=row["base_temp_bias_degC"]+(row["delta_t_eq_degC"] if stage=="spinup" else 0),
        check_calib_params=False,ys=start,ye=end,repeat=stage=="spinup")
def save_cp(root,model,year,row,history,counters):
    from . import lia_all191_adaptive_spinup_v10_260 as base
    cp=root/"cp";cp.mkdir(parents=True,exist_ok=True);final=cp/str(year)
    if final.exists():raise FileExistsError("immutable-checkpoint-collision")
    tmp=cp/(".t"+str(year)+"."+str(os.getpid())+"."+str(time.time_ns()));tmp.mkdir()
    with (tmp/"s.gz").open("xb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0) as f:pickle.dump(model.fls,f,pickle.HIGHEST_PROTOCOL)
        raw.flush();os.fsync(raw.fileno())
    write(tmp/"h.json",history)
    c={"schema":"lia-joint-state-history-checkpoint-v10.279","year":year,"stage":row["stage"],
       "fingerprint":row["fingerprint"],"lia_unique_id":row["lia_unique_id"],"rgi_id":row["rgi_id"],
       "parameters":{p:row[p] for p in PARAMS},"initial_state_hash":row["initial_state_hash"],
       "state_sha256":sha(tmp/"s.gz"),"history_sha256":sha(tmp/"h.json"),"history_rows":len(history),
       "topology":base.topology_signature(model.fls),"state":history[-1],"counters":counters}
    write(tmp/"c.json",c);rename(tmp,final);return final
def load_cp(path,row):
    from . import lia_all191_adaptive_spinup_v10_260 as base
    c=read(path/"c.json")
    if any(c[n]!=row[n] for n in ("fingerprint","stage","lia_unique_id","rgi_id","initial_state_hash")):raise ValueError("checkpoint-identity")
    if c["parameters"]!={p:row[p] for p in PARAMS}:raise ValueError("checkpoint-parameters")
    if sha(path/"s.gz")!=c["state_sha256"] or sha(path/"h.json")!=c["history_sha256"]:raise ValueError("checkpoint-integrity")
    h=read(path/"h.json");start=0 if row["stage"]=="spinup" else 1470
    if [r["year"] for r in h]!=list(range(start,c["year"]+1)) or len(h)!=c["history_rows"]:raise ValueError("history-state-not-synchronous")
    with gzip.open(path/"s.gz","rb") as f:fls=pickle.load(f)
    if base.topology_signature(fls)!=c["topology"]:raise ValueError("checkpoint-topology")
    return c,fls,h
def worker(stage,key):
    import numpy as np
    from . import lia_all191_adaptive_spinup_v10_260 as base
    start_clock=time.perf_counter();m=contract();row=row_for(stage,key)
    parent=read(OUT/"r/parent.json")
    if parent["transaction"]["sha256"]!=sha(TX/"transaction_manifest.json"):raise ValueError("parent-contract")
    root=OUT/("s" if stage=="spinup" else "f")/key
    root.mkdir(parents=True,exist_ok=False)
    try:
        base._configure_oggm(root/"w",root/"w")
        item=read(ROOT/row["inputs"]["path"]);check(item["geometry"])
        if stage=="spinup":fls=[base._read_flowline(ROOT/item["geometry"]["path"])];start,end=0,900
        else:
            spinrow=row_for("spinup",key)
            cp,fls,_=load_cp(ROOT/row["source_checkpoint"]["path"].replace("/c.json",""),spinrow)
            if cp["year"]!=900 or cp["state_sha256"]!=row["initial_state_hash"]:raise ValueError("forward-handoff")
            start,end=1470,2000
        mb=mb_for(row,stage);model=base.make_model(fls,mb,row["glen_a_pa_3_s_1"],start)
        for a,b in zip(fls,model.fls):np.testing.assert_array_equal(a.section,b.section)
        history=[base.snapshot(model,start)]
        counters={"accepted_steps":0,"rejected_steps":0,"minimum_dt_seconds":None}
        original_step=model.step
        def step(dt):
            actual=float(original_step(dt));counters["accepted_steps"]+=1
            counters["minimum_dt_seconds"]=actual if counters["minimum_dt_seconds"] is None else min(actual,counters["minimum_dt_seconds"])
            return actual
        model.step=step
        timing={"initialization_seconds":time.perf_counter()-start_clock,"integration_seconds":0.,"checkpoint_seconds":0.}
        flags=[]
        for year in range(start,end+1):
            if year>start:
                t=time.perf_counter();model.run_until(float(year));timing["integration_seconds"]+=time.perf_counter()-t
                history.append(base.snapshot(model,year))
            h=history[-1]
            for fl in model.fls:
                if not all(np.isfinite(np.asarray(getattr(fl,n))).all() for n in ("section","thick","surface_h","bed_h","widths_m")):raise ValueError("nonfinite-full-state")
            if not all(math.isfinite(h[p]) for p in ("volume_m3","area_m2","length_m","minimum_thickness_m","maximum_thickness_m")):raise ValueError("nonfinite-state")
            if h["minimum_thickness_m"] < -1e-9:raise ValueError("negative-thickness-state")
            if h["boundary_margin_m"]<5000 or min(h[p] for p in ("volume_m3","area_m2","length_m"))<=0:
                flags.append({"year":year,"margin":h["boundary_margin_m"],"volume":h["volume_m3"],"role":"warning-continue"})
            if (year-start)%30==0 or year==end:
                t=time.perf_counter();saved=save_cp(root,model,year,row,history,counters)
                cp,restored,stored=load_cp(saved,row)
                for a,b in zip(model.fls,restored):
                    np.testing.assert_allclose(a.thick,b.thick,rtol=1e-7,atol=0)
                    np.testing.assert_allclose(a.section,b.section,rtol=1e-7,atol=0)
                timing["checkpoint_seconds"]+=time.perf_counter()-t
                status(root/"progress.json",{"year":year,"stage":stage,"checkpoint":ref(saved/"c.json"),"counters":counters,"timing":timing})
        convergence=base.convergence_metrics({h["year"]:h for h in history},900) if stage=="spinup" else None
        result={"schema":"lia-cpu-melt-factor-member-v10.279","key":key,"stage":stage,"terminal":True,
            "status":"completed","technical_failure":False,"fingerprint":row["fingerprint"],
            "lia_unique_id":row["lia_unique_id"],"rgi_id":row["rgi_id"],"parameters":{p:row[p] for p in PARAMS},
            "melt_f_multiplier":row["melt_f_multiplier"],"year":end,"stable":convergence["passed"] if convergence else None,
            "convergence":convergence,"checkpoint_contract_pass":True,"final_checkpoint":ref(saved/"c.json"),
            "annual_history":ref(saved/"h.json"),"final_state":ref(saved/"s.gz"),"domain_warnings":flags,
            "runtime_seconds":time.perf_counter()-start_clock,"timing":timing,"counters":counters,
            "q":0 if stage=="forward" else None,"accepted_ledger_effect":0}
        write(root/"result.json",result);print(json.dumps(clean(result)),flush=True)
    except Exception as e:
        failure={"stage":stage,"key":key,"lia_unique_id":row["lia_unique_id"],"fingerprint":row["fingerprint"],
                 "terminal":True,"status":"technical-failure","technical_failure":True,"error":repr(e),
                 "traceback":traceback.format_exc(),"runtime_seconds":time.perf_counter()-start_clock}
        print(failure["traceback"],file=sys.stderr,flush=True)
        try:write(root/"result.json",failure)
        except OSError as x:print("RESULT_WRITE_FAILED "+repr(x),file=sys.stderr,flush=True)
        raise SystemExit(2)
def forward_row(row,result):
    cp=read(ROOT/result["final_checkpoint"]["path"])
    payload={"parent_spinup_fingerprint":row["fingerprint"],"parent_state_hash":cp["state_sha256"],
             "parameters":{p:row[p] for p in PARAMS},"stage":"forward","years":[1470,2000],"q":0,
             "inputs":row["inputs"],"contract":sha(TX/"transaction_manifest.json")}
    return {**row,"stage":"forward","fingerprint_payload":payload,"fingerprint":digest(payload),
            "source_checkpoint":result["final_checkpoint"],"parent_result":ref(OUT/"s"/row["key"]/"result.json"),
            "initial_state_hash":cp["state_sha256"]}
def score_record(row,result,rp):
    import numpy as np,pandas as pd
    from .score_lia_all191_forward_essential_standard_v3 import score_one
    check(result["annual_history"]);h=read(ROOT/result["annual_history"]["path"]);a=pd.DataFrame(h)
    if list(a.year)!=list(range(1470,2001)):raise ValueError("score-year-coverage")
    tl,tr=row["target_lia_length_m"],row["target_rgi_length_m"];L=a.length_m.to_numpy();sep=abs(tl-tr)
    out={"key":row["key"],"lia_unique_id":row["lia_unique_id"],"rgi_id":row["rgi_id"],
         "delta_t_eq_degC":row["delta_t_eq_degC"],"melt_f":row["melt_f"],"melt_f_multiplier":row["melt_f_multiplier"],
         "fingerprint":row["fingerprint"],"policy":"CPU-direction-v2-retrospective",
         "lia_gate":bool(abs(L.max()-tl)/tl<=.1),"rgi_gate":bool(abs(L[-1]-tr)/tr<=.1),
         "amplitude_gate":bool(np.ptp(L)>=.5*sep),
         "old_net_direction_diagnostic":bool(sep<300 or np.sign(L[-1]-L[0])==np.sign(tr-tl))}
    if (a[["length_m","area_m2","volume_m3"]].to_numpy()>0).all() and sep:
        old=score_one(pd.Series({**row,"production_member_key":row["key"]}),
             pd.Series({"lia_registered_length_m":tl,"rgi_registered_length_m":tr}),a,
             result["annual_history"]["sha256"],{**result,"forward_fingerprint":row["fingerprint"]},rp)
        for k in ("lag_gate_pass","bidirectional_multistage_gate_pass","artifact_gate_pass"):
            out[k]=old[k]
        out["shape_metrics"]={k:v for k,v in old.items() if k.startswith(("length_","persistent_","sustained_","terminal_","high_frequency_","flag_"))}
    else:out.update(lag_gate_pass=False,bidirectional_multistage_gate_pass=False,artifact_gate_pass=False,shape_evaluable=False)
    out["provisional_curve_pass"]=all(out[k] for k in ("lia_gate","rgi_gate","amplitude_gate","lag_gate_pass","bidirectional_multistage_gate_pass","artifact_gate_pass"))
    out["accepted_ledger_effect"]=0
    return clean(out)
def score_all():
    contract();matrix=read(OUT/"r/forward_matrix.json");rows=[];failures=[]
    for row in matrix:
        rp=OUT/"f"/row["key"]/"result.json"
        try:
            r=read(rp)
            if r["technical_failure"]:continue
            score=score_record(row,r,rp);write(OUT/"scores"/(row["key"]+".json"),score);rows.append(score)
        except Exception as e:
            fail={"key":row["key"],"error":repr(e),"traceback":traceback.format_exc()}
            write(OUT/"scores"/(row["key"]+".failure.json"),fail);failures.append(fail)
    write(OUT/"r/scoring_summary.json",{"completed":len(rows),"scoring_failures":failures,
        "provisional_cells":sum(r["provisional_curve_pass"] for r in rows),
        "provisional_systems":sorted({r["lia_unique_id"] for r in rows if r["provisional_curve_pass"]}),
        "accepted_ledger_effect":0,"rows":rows})
def controller():
    import psutil
    m=contract();a=read(TX/"authorization.json");check(a["transaction"])
    v=subprocess.run([PYTHON,str(DRIVERS/"validate_lia_execution_state_v1.py")],cwd=ROOT,capture_output=True,text=True)
    state=read(ROOT/"LIA_CURRENT_STATE.json")
    if v.returncode or not json.loads(v.stdout)["launch_authorized"] or state["next_action"].get("transaction_id")!="lia-cpu-melt-v10.279":raise ValueError("canonical-launch-authority")
    if state["current_focus"]["evidence"]["sha256"]!=sha(TX/"authorization.json"):raise ValueError("current-authority-evidence")
    (OUT/"r").mkdir(exist_ok=False);(OUT/"l").mkdir(exist_ok=False)
    parent={"pid":os.getpid(),"started_at":now(),"command":sys.argv,"transaction":ref(TX/"transaction_manifest.json"),
            "workers":20,"threads":THREADS,"output_root":str(OUT),"checkpoint_years":30,"authorized_stages":["spinup","forward","offline-score"]}
    write(OUT/"r/parent.json",parent)
    env=os.environ.copy();env.update(THREADS);env["PYTHONPATH"]=str(ROOT)
    matrix=read(TX/"spinup_matrix.json");peak=0.;completed_stages={};forward=[]
    for stage,rows in (("spinup",matrix),("forward",forward)):
        index=0;running={};completed=[];pause=None
        if stage=="forward":write(OUT/"r/forward_matrix.json",forward)
        while index<len(rows) or running:
            mem=psutil.virtual_memory();peak=max(peak,mem.percent)
            if mem.percent>80:pause="memory-above80-user-decision-required"
            decision=OUT/"r/user_decision.json"
            if pause and decision.exists():
                choice=read(decision)
                if choice.get("action")=="resume":
                    pause=None;rename(decision,decision.with_name("decision."+str(time.time_ns())+".json"))
            for slot,item in list(running.items()):
                proc,row,ol,el=item;rc=proc.poll()
                if rc is None:continue
                ol.close();el.close();rp=OUT/("s" if stage=="spinup" else "f")/row["key"]/"result.json"
                try:
                    r=read(rp)
                    if r["fingerprint"]!=row["fingerprint"] or r["key"]!=row["key"]:raise ValueError("terminal-identity")
                    bad=bool(r.get("technical_failure") or rc!=0)
                    if not bad:
                        for name in ("final_checkpoint","annual_history","final_state"):check(r[name])
                except Exception as e:r={"status":"missing-or-invalid-result","technical_failure":True,"error":repr(e)};bad=True
                record={"stage":stage,"key":row["key"],"lia_unique_id":row["lia_unique_id"],"pid":proc.pid,
                        "returncode":rc,"technical_failure":bad,"stable":r.get("stable"),"result":ref(rp) if rp.exists() else None}
                try:write(OUT/"r/done"/(stage+"."+row["key"]+".json"),record)
                except OSError as e:pause="controller-record-storage-needs-user-decision";print(repr(e),file=sys.stderr,flush=True)
                completed.append(record);event({"event":"finished",**record});del running[slot]
                if stage=="spinup" and not bad and r.get("stable"):
                    try:
                        frow=forward_row(row,r);write(OUT/"q"/(row["key"]+".json"),frow);forward.append(frow)
                    except Exception as e:
                        event({"event":"forward-materialization-failure","key":row["key"],"error":repr(e)})
                        write(OUT/"r/handoff_failures"/(row["key"]+".json"),{"key":row["key"],"technical_failure":True,"error":repr(e),"parent_result":ref(rp),"status":"needs-user-decision-no-retry"})
            while pause is None and index<len(rows) and len(running)<20:
                row=rows[index];slot=next(i for i in range(20) if i not in running)
                ol=(OUT/"l"/(stage+"."+row["key"]+".out.log")).open("x",encoding="utf-8")
                el=(OUT/"l"/(stage+"."+row["key"]+".err.log")).open("x",encoding="utf-8")
                cmd=[PYTHON,"-m",MODULE,"worker","--stage",stage,"--key",row["key"]]
                try:proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=ol,stderr=el)
                except OSError as e:
                    ol.close();el.close();index+=1
                    record={"stage":stage,"key":row["key"],"lia_unique_id":row["lia_unique_id"],"technical_failure":True,"stable":None,"error":repr(e),"status":"launch-failed-no-retry"}
                    write(OUT/"r/done"/(stage+"."+row["key"]+".json"),record);completed.append(record);event(record);continue
                running[slot]=(proc,row,ol,el);index+=1
                event({"event":"started","at":now(),"stage":stage,"slot":slot,"pid":proc.pid,"key":row["key"],
                       "lia_unique_id":row["lia_unique_id"],"melt_f_multiplier":row["melt_f_multiplier"],
                       "delta_t_eq_degC":row["delta_t_eq_degC"],"command":cmd})
                time.sleep(.5)
            invalid=sum(c["technical_failure"] for c in completed)
            snap={"phase":stage,"parent_pid":os.getpid(),"updated_at":now(),"completed":len(completed),
                "pending":len(rows)-index,"technical_failures":invalid,
                "technical_invalid_advisory":len(completed)>=120 and invalid/len(completed)>.15,
                "stable_new_states":len(forward),"pause_reason":pause,"cpu_percent":psutil.cpu_percent(),
                "memory_percent":mem.percent,"available_memory_bytes":mem.available,"peak_memory_percent":peak,
                "slots":[{"slot":i,"state":"running" if i in running else "idle",
                          "pid":running[i][0].pid if i in running else None,
                          "key":running[i][1]["key"] if i in running else None,
                          "lia_unique_id":running[i][1]["lia_unique_id"] if i in running else None} for i in range(20)]}
            if status(OUT/"r/status.json",snap) is None:pause="all-status-destinations-unavailable-user-decision"
            time.sleep(2)
        completed_stages[stage]=completed
        write(OUT/"r"/(stage+"_summary.json"),{"stage":stage,"completed":len(completed),"technical_failures":sum(c["technical_failure"] for c in completed),"records":completed})
    status(OUT/"r/status.json",{"phase":"offline-score","parent_pid":os.getpid(),"updated_at":now(),"stable_new_states":len(forward),"slots":[]})
    with (OUT/"l/score.out.log").open("x") as ol,(OUT/"l/score.err.log").open("x") as el:
        proc=subprocess.Popen([PYTHON,"-m",MODULE,"score"],cwd=ROOT,env=env,stdout=ol,stderr=el)
        while proc.poll() is None:time.sleep(2)
    write(OUT/"r/summary.json",{"status":"completed" if proc.returncode==0 else "completed-dynamics-scoring-technical-failure",
        "started_at":parent["started_at"],"finished_at":now(),"parent_pid":os.getpid(),"spinup_cells":len(matrix),
        "forward_cells":len(forward),"stage_summaries":completed_stages,"scoring_returncode":proc.returncode,
        "peak_memory_percent":peak,"accepted_ledger_effect":0,"automatic_retry":False})
    status(OUT/"r/status.json",{"phase":"terminal","parent_pid":os.getpid(),"updated_at":now(),"spinup_completed":len(matrix),
        "forward_completed":len(forward),"slots":[],"summary":ref(OUT/"r/summary.json")})
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["controller","worker","score"])
    p.add_argument("--stage",choices=["spinup","forward"]);p.add_argument("--key");a=p.parse_args()
    if a.mode=="controller":controller()
    elif a.mode=="worker":worker(a.stage,a.key)
    else:score_all()
if __name__=="__main__":main()
