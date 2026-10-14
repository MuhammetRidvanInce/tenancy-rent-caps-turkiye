"""Runs analysis stages 01-11 and English figures (13) using locally authorised TurkStat microdata."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import sys

CODES = Path(__file__).resolve().parent
VALID = CODES.parent / "outputs" / "validation"
VALID.mkdir(parents=True, exist_ok=True)
env = dict(os.environ, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
           OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
record = {"started": dt.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0], "stages": []}
for script in sorted(CODES.glob("[01][0-9]_*.py")):
    if script.name == "00_run_all.py" or script.name.startswith("12_"):
        continue
    t0 = dt.datetime.now()
    with (VALID / f"{script.stem}.txt").open("w", encoding="utf-8") as log:
        rc = subprocess.run([sys.executable, "-X", "utf8", str(script)], cwd=CODES, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    record["stages"].append({"script": script.name, "returncode": rc, "seconds": (dt.datetime.now() - t0).seconds})
    print(script.name, "rc", rc, flush=True)
    if rc != 0:
        break
record["finished"] = dt.datetime.now().isoformat(timespec="seconds")
record["status"] = "ok" if record["stages"] and all(s["returncode"] == 0 for s in record["stages"]) else "failed"
(VALID / "last_run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(record["status"])
sys.exit(0 if record["status"] == "ok" else 1)
