"""v10.280 exact-contract recovery of 6492 unfinished v10.279 members.
The original numerical loop, MB, stability and scorer are retained.
Old result/state roots are read-only. Every scientific fingerprint is preserved.
"""
from __future__ import annotations
import os,sys,json,time,math,copy,subprocess,traceback,signal
from pathlib import Path
from . import lia_cpu_melt_factor_v10_279 as original
from .lia_cpu_melt_factor_v10_279 import THREADS,ROOT,DRIVERS,RES,PYTHON,PARAMS,now,read,sha,ref,check,clean,write,digest,rename,status,save_cp,load_cp,mb_for,score_record
TX=RES/"lia_cpu_melt_factor_resume_v10_280_transaction_v1"
OUT=ROOT/"w280"
MODULE="Himalaya_greatGlacier.With_Rec_climate_organized.ice_surface_reconstruction.drivers.lia_cpu_melt_resume_v10_280"
def contract():
    original.contract()
    m=read(TX/"transaction_manifest.json")
    for x in m["code"]+[m["matrix"],m["queue_index"],m["source_terminal"],m["technical_tests"]]:check(x)
    return m
def row_for(stage,key):
    if stage=="spinup":
        binding=read(TX/"queue_index.json")[key]
        row=original.row_for("spinup",key)
        if row["fingerprint"]!=binding["fingerprint"]:raise ValueError("recovery-fingerprint")
        if (original.OUT/"s"/key/"result.json").exists():raise ValueError("completed-member-cannot-rerun")
        return {**row,"resume_checkpoint":binding.get("checkpoint")}
    row=read(OUT/"q"/(key+".json"))
    if row["key"]!=key or row["stage"]!=stage or digest(row["fingerprint_payload"])!=row["fingerprint"]:raise ValueError("forward-row-identity")
    for n in PARAMS:
        if row[n]!=row["fingerprint_payload"]["parameters"][n]:raise ValueError("forward-row-parameters")
    check(row["inputs"]);check(row["parent_result"]);check(row["source_checkpoint"])
    if not read(ROOT/row["parent_result"]["path"])["stable"]:raise ValueError("nonstable-parent")
    return row
def event(obj):
    try:
        with (OUT/"r/events.jsonl").open("a",encoding="utf-8") as f:
            f.write(json.dumps(clean(obj),ensure_ascii=False)+"\n");f.flush()
    except OSError as e: print("EVENT_WARNING "+repr(e),file=sys.stderr,flush=True)
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
        source_cp=None;source_history=None
        if stage=="spinup":
            if row.get("resume_checkpoint"):
                check(row["resume_checkpoint"])
                source_cp,fls,source_history=load_cp((ROOT/row["resume_checkpoint"]["path"]).parent,row)
                start,end=source_cp["year"],900
            else:fls=[base._read_flowline(ROOT/item["geometry"]["path"])];start,end=0,900
        else:
            spinrow=row_for("spinup",key)
            cp,fls,_=load_cp(ROOT/row["source_checkpoint"]["path"].replace("/c.json",""),spinrow)
            if cp["year"]!=900 or cp["state_sha256"]!=row["initial_state_hash"]:raise ValueError("forward-handoff")
            start,end=1470,2000
        mb=mb_for(row,stage);model=base.make_model(fls,mb,row["glen_a_pa_3_s_1"],start)
        for a,b in zip(fls,model.fls):np.testing.assert_array_equal(a.section,b.section)
        history=source_history if source_history is not None else [base.snapshot(model,start)]
        if source_cp is not None:
            restored_state=base.snapshot(model,start)
            for name in ("length_m","area_m2","volume_m3","minimum_thickness_m","maximum_thickness_m"):
                np.testing.assert_allclose(restored_state[name],source_history[-1][name],rtol=1e-7,atol=0)
        counters=copy.deepcopy(source_cp["counters"]) if source_cp is not None else {"accepted_steps":0,"rejected_steps":0,"minimum_dt_seconds":None}
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
        result={"schema":"lia-cpu-melt-factor-member-v10.280","execution_attempt":"v10.280","resumed_from":row.get("resume_checkpoint") if stage=="spinup" else None,"execution_start_year":start,"key":key,"stage":stage,"terminal":True,
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
    if v.returncode or not json.loads(v.stdout)["launch_authorized"] or state["next_action"].get("transaction_id")!="lia-cpu-melt-resume-v10.280":raise ValueError("canonical-launch-authority")
    if state["current_focus"]["evidence"]["sha256"]!=sha(TX/"authorization.json"):raise ValueError("current-authority-evidence")
    (OUT/"r").mkdir(exist_ok=False);(OUT/"l").mkdir(exist_ok=False)
    parent={"pid":os.getpid(),"started_at":now(),"command":sys.argv,"transaction":ref(TX/"transaction_manifest.json"),
            "workers":20,"threads":THREADS,"output_root":str(OUT),"checkpoint_years":30,"authorized_stages":["spinup","forward","offline-score"]}
    write(OUT/"r/parent.json",parent)
    env=os.environ.copy();env.update(THREADS);env["PYTHONPATH"]=str(ROOT)
    matrix=[row_for("spinup",x["key"]) for x in read(TX/"recovery_queue.json")];peak=0.;completed_stages={};forward=[]
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
                "pending":len(rows)-index,"technical_failures":invalid,"reused_completed_spinups":1152,"total_completed_spinups":1152+len(completed) if stage=="spinup" else 7644,
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
        "forward_cells":len(forward),"reused_completed_spinups":1152,"total_spinup_population":7644,"stage_summaries":completed_stages,"scoring_returncode":proc.returncode,
        "peak_memory_percent":peak,"accepted_ledger_effect":0,"automatic_retry":False})
    status(OUT/"r/status.json",{"phase":"terminal","parent_pid":os.getpid(),"updated_at":now(),"spinup_completed":len(matrix),
        "forward_completed":len(forward),"slots":[],"summary":ref(OUT/"r/summary.json")})
def supervise():
    import msvcrt,psutil
    m=contract();auth=read(TX/"authorization.json");check(auth["transaction"])
    host=OUT/"host";host.mkdir(exist_ok=False)
    with (host/"writer.lock").open("x+b") as lock:
        lock.write(b"1");lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        env=os.environ.copy();env.update(THREADS);env["PYTHONPATH"]=str(ROOT)
        command=[PYTHON,"-u","-m",MODULE,"controller"]
        with (host/"parent.out.log").open("x",encoding="utf-8") as ol,(host/"parent.err.log").open("x",encoding="utf-8") as el:
            proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=ol,stderr=el,creationflags=subprocess.CREATE_NO_WINDOW)
            write(host/"launch.json",{"supervisor_pid":os.getpid(),"parent_pid":proc.pid,"started_at":now(),"command":command,"transaction":ref(TX/"transaction_manifest.json"),"ownership":"Windows-WMI-created-hidden-supervisor","automatic_restart":False})
            while proc.poll() is None:
                status(host/"status.json",{"supervisor_pid":os.getpid(),"parent_pid":proc.pid,"parent_alive":True,"at":now()});time.sleep(5)
            rc=proc.wait()
        records=[];invalid=[]
        for stage in ("spinup","forward"):
            for p in (OUT/("s" if stage=="spinup" else "f")).glob("*/result.json"):
                try:
                    result=read(p)
                    if not result["technical_failure"]:
                        for name in ("final_checkpoint","annual_history","final_state"):check(result[name])
                    records.append({"stage":stage,"key":result["key"],"technical_failure":result["technical_failure"],"stable":result.get("stable"),"result":ref(p)})
                except Exception as exc:invalid.append({"path":str(p),"error":repr(exc)})
        live=[]
        for p in psutil.process_iter(["cmdline"]):
            if p.pid!=os.getpid() and MODULE in " ".join(p.info.get("cmdline") or []):live.append(p.pid)
        summary=read(OUT/"r/summary.json") if (OUT/"r/summary.json").exists() else None
        write(host/"terminal.json",{"status":"completed" if rc==0 and summary and not invalid else "controller-technical-interruption","returncode":rc,"finished_at":now(),"parent_pid":proc.pid,"supervisor_pid":os.getpid(),"reused_completed_spinups":1152,"records":records,"invalid_results":invalid,"remaining_worker_pids":live,"controller_summary":summary,"stderr":ref(host/"parent.err.log"),"automatic_restart":False,"accepted_ledger_effect":0})
        status(host/"status.json",{"parent_alive":False,"returncode":rc,"at":now(),"terminal":ref(host/"terminal.json")})
def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["controller","worker","score","supervise"]);p.add_argument("--stage",choices=["spinup","forward"]);p.add_argument("--key");a=p.parse_args()
    def caught_signal(signum,frame):
        event({"event":"signal-received","signal":signum,"pid":os.getpid(),"at":now()})
        raise SystemExit(128+signum)
    if a.mode=="controller":
        for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,caught_signal)
        try:controller()
        except BaseException as exc:
            event({"event":"controller-exit","at":now(),"pid":os.getpid(),"exception":repr(exc),"traceback":traceback.format_exc()})
            raise
    elif a.mode=="supervise":
        sys.stdout=(OUT/"supervisor.out.log").open("x",encoding="utf-8",buffering=1)
        sys.stderr=(OUT/"supervisor.err.log").open("x",encoding="utf-8",buffering=1)
        try:supervise()
        except BaseException as exc:
            traceback.print_exc()
            write(OUT/"supervisor-exception.json",{"at":now(),"pid":os.getpid(),"error":repr(exc),"automatic_restart":False})
            raise
    elif a.mode=="worker":worker(a.stage,a.key)
    else:score_all()
if __name__=="__main__":main()
