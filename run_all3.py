import subprocess, os, sys, time
tags = os.environ.get("TAGS", "emb,none").split(",")
for tg in tags:
    t0 = time.time(); print(f"=== run TAG={tg} start", flush=True)
    env = dict(os.environ, TAG=tg)
    r = subprocess.run([sys.executable, "-u", "train_eval_v3.py"], env=env, stdout=open(f"run_{tg}.log", "w"), stderr=subprocess.STDOUT)
    print(f"=== run TAG={tg} exit={r.returncode} {time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
