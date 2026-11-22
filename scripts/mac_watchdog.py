"""Keeps the long local training run alive: resumes it from its last checkpoint if the process dies or hangs.

It never touches a healthy run. A run counts as hung only after its log has been silent for --stall-min minutes
of time the Mac was actually awake (sleep gaps are ignored). It stops by itself once the run has finished, holds
a sleep assertion (caffeinate) while it watches, and backs off if the run keeps crashing right after a restart.

    .venv/bin/python scripts/mac_watchdog.py          # normally started by launchd (scripts/install_launchd.sh)
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_ARGS = ["-m", "autolearn.llm.train", "--stage", "pretrain", "--preset", "micro",
              "--data", "data/llm/pretrain_big,data/llm/pretrain_big2", "--out", "runs/long_micro",
              "--tokenizer", "data/llm/pretrain/tokenizer.json", "--batch", "8", "--accum", "4", "--lr", "8e-4",
              "--warmup", "2000", "--max-steps", "180000", "--eval-every", "1000", "--eval-iters", "40", "--resume"]


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def find_run(out_dir: str) -> int | None:
    r = subprocess.run(["pgrep", "-f", f"autolearn.llm.train.*--out {out_dir}"], capture_output=True, text=True)
    pids = [int(p) for p in r.stdout.split() if int(p) != os.getpid()]
    return pids[0] if pids else None


def finished(run_dir: Path, max_steps: int) -> bool:
    prog = run_dir / "progress.json"
    if prog.exists():
        try:
            if json.loads(prog.read_text()).get("done"):
                return True
        except json.JSONDecodeError:
            pass
    mf = run_dir / "metrics.jsonl"
    if mf.exists():  # older runs don't write progress.json: look at the last logged step
        with open(mf, "rb") as f:
            f.seek(max(0, mf.stat().st_size - 4096))
            for line in reversed(f.read().decode(errors="ignore").splitlines()):
                try:
                    return json.loads(line).get("step", 0) >= max_steps
                except json.JSONDecodeError:
                    continue
    return False


def stop(pid: int):
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(60):
            time.sleep(1)
            os.kill(pid, 0)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=str(ROOT / "logs/long_run.log"))
    ap.add_argument("--stall-min", type=float, default=60)
    ap.add_argument("--interval", type=float, default=120)
    a = ap.parse_args()
    global TRAIN_ARGS
    if os.environ.get("AUTOLEARN_WATCHDOG_ARGS"):  # tests watch a tiny run instead
        TRAIN_ARGS = json.loads(os.environ["AUTOLEARN_WATCHDOG_ARGS"])
    out_dir = TRAIN_ARGS[TRAIN_ARGS.index("--out") + 1]
    max_steps = int(TRAIN_ARGS[TRAIN_ARGS.index("--max-steps") + 1])
    run_log = Path(a.log)

    (ROOT / out_dir).mkdir(parents=True, exist_ok=True)
    lock = open(ROOT / out_dir / ".watchdog.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another watchdog is running")
        while True:  # stay alive (so launchd doesn't respawn us in a loop) and take over if it exits
            time.sleep(300)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                pass
    subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])  # keep the Mac awake while watching

    last_tick = time.time()
    last_progress, last_mtime = time.time(), None
    restarts: list[float] = []
    log(f"watching {out_dir} (stall limit {a.stall_min:.0f} min awake)")
    while True:
        now = time.time()
        if now - last_tick > 3 * a.interval:  # the Mac was asleep: don't count that as a silent run
            last_progress = now
        last_tick = now
        pid = find_run(out_dir)
        if pid:
            m = run_log.stat().st_mtime if run_log.exists() else None
            if m != last_mtime:
                last_mtime, last_progress = m, now
            elif now - last_progress > a.stall_min * 60:
                log(f"run {pid} silent for {(now - last_progress) / 60:.0f} min while awake: restarting it")
                stop(pid)
                pid = None
        if not pid:
            if finished(ROOT / out_dir, max_steps):
                log("training finished; watchdog exiting")
                return
            restarts = [t for t in restarts if now - t < 3600]
            if len(restarts) >= 4:
                log("run keeps dying right after restart; waiting 30 min before trying again")
                time.sleep(1800)
                restarts = []
                continue
            with open(run_log, "a") as lf:
                lf.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} watchdog: resuming from last checkpoint\n")
                lf.flush()
                p = subprocess.Popen([sys.executable, *TRAIN_ARGS], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            restarts.append(now)
            last_progress, last_mtime = time.time(), None
            log(f"started training run (pid {p.pid})")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
