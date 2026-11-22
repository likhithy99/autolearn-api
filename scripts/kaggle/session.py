"""One Kaggle GPU session of the AutoLearn LLM campaign (runs inside a Kaggle kernel).

Inputs (Kaggle datasets mounted under /kaggle/input):
  autolearn-code : tokenizer.json, campaign.json   (the Python package is embedded in this script)
  autolearn-ckpt : state.json, last.pt (model+optimizer), best.pt, metrics.jsonl  (absent on session 0)
Outputs (/kaggle/working, downloaded by the local campaign driver after the session):
  state.json, last.pt, best.pt, metrics.jsonl, [sft_best.pt], outcome.json

Built to never waste a session:
  * every training run is watched; a run that stops making progress (hung GPU collective, deadlock) is killed
  * a failed run is retried down a ladder (no torch.compile -> smaller micro-batch -> single GPU), each attempt
    resuming from the last checkpoint and keeping the same tokens per optimizer step
  * network steps (pip, dataset downloads) are retried with backoff
  * whatever happens, any progress made is written out as the new state (outcome.json records what went wrong)
"""

import base64
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

KROOT = os.environ.get("AUTOLEARN_KAGGLE_ROOT", "/kaggle")  # overridden only by the local tests
LOCAL_TEST = KROOT != "/kaggle"
WORK = f"{KROOT}/working"
T0 = time.time()
OUTCOME = {"attempts": [], "errors": []}


def log(msg):
    print(msg, flush=True)


def find(name):
    hits = sorted(glob.glob(f"{KROOT}/input/**/{name}", recursive=True))
    return hits[0] if hits else None


class Stalled(Exception):
    pass


def run(cmd, env=None, done_file=None, heartbeat=None, stall_min=None, grace_min=25):
    """Run a command in its own process group.

    done_file: treat its appearance as completion (child finished its work but hangs on exit).
    heartbeat/stall_min: kill the whole group if `heartbeat` (a file the child keeps updating) hasn't changed for
    stall_min minutes, after an initial grace period for compilation / data loading."""
    log(f"$ {cmd}")
    p = subprocess.Popen(cmd, shell=True, env=env, start_new_session=True)
    started, seen = time.time(), None

    def kill():
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()

    while p.poll() is None:
        time.sleep(10)
        if done_file and os.path.exists(done_file):
            seen = seen or time.time()
            if time.time() - seen > 60:
                log(f"{done_file} written but process still alive; killing it")
                kill()
                return
        if heartbeat and stall_min and time.time() - started > grace_min * 60:
            last = os.path.getmtime(heartbeat) if os.path.exists(heartbeat) else started
            if time.time() - max(last, started + grace_min * 60) > stall_min * 60:
                log(f"no progress for {stall_min} min ({heartbeat}); killing the run")
                kill()
                raise Stalled(cmd)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


def retry(fn, what, tries=4, delay=30):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 (network hiccups, HF rate limits, ...)
            OUTCOME["errors"].append(f"{what}: {e}")
            if i == tries - 1:
                raise
            log(f"{what} failed ({e}); retrying in {delay * 2 ** i}s")
            time.sleep(delay * 2 ** i)


campaign_path = find("campaign.json")
assert campaign_path, "autolearn-code dataset not attached"
CODE = os.path.dirname(campaign_path)
cfg = json.load(open(campaign_path))
SESSION_MINUTES = None  # __SESSION_MINUTES__
if SESSION_MINUTES:  # the driver sizes each session to the GPU quota left this week
    cfg["session_minutes"] = SESSION_MINUTES
SAFE_MODE = False  # __SAFE_MODE__
state_path = find("state.json")
state = json.load(open(state_path)) if state_path else {"session": 0, "stage": "pretrain"}
START = dict(state)
CK = os.path.dirname(state_path) if state_path else None
log(f"campaign {cfg}\nstate {state}\nsafe mode: {SAFE_MODE}")
EXPECT_SESSION = None  # __EXPECT_SESSION__
if EXPECT_SESSION is not None and state["session"] != EXPECT_SESSION:
    # Kaggle mounted an older checkpoint version (new upload still processing): stop before spending GPU time
    log("STALE_CHECKPOINT " + json.dumps({"mounted": state["session"], "expected": EXPECT_SESSION}))
    raise SystemExit(0)

# package: source files are embedded below by campaign.py (Kaggle mangles uploaded archives/folders)
EMBEDDED = {}  # __EMBEDDED_PACKAGE__
pkg_root = f"{KROOT}/temp/code"
assert EMBEDDED, "package not embedded; launch via scripts/kaggle/campaign.py"
for rel, b64 in EMBEDDED.items():
    dst = os.path.join(pkg_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(base64.b64decode(b64))
env = dict(os.environ, PYTHONPATH=pkg_root, HF_HUB_DISABLE_TELEMETRY="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
           TORCH_NCCL_ASYNC_ERROR_HANDLING="1")  # a failed collective aborts instead of hanging
if not LOCAL_TEST:
    retry(lambda: run("pip install -q 'tokenizers==0.23.3' datasets certifi", env), "pip install")

import torch  # noqa: E402

ngpu = max(1, torch.cuda.device_count())
log(f"GPUs: {ngpu} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

# carry over checkpoints from the previous session
for f in ("last.pt", "best.pt", "metrics.jsonl", "sft_best.pt"):
    if CK and os.path.exists(os.path.join(CK, f)):
        shutil.copy(os.path.join(CK, f), os.path.join(WORK, f))
if state["session"] > 0 and state["stage"] == "pretrain" and not os.path.exists(os.path.join(WORK, "last.pt")):
    log("MISSING_CHECKPOINT expected last.pt from the previous session but none was mounted")
    raise SystemExit(0)

tok = os.path.join(CODE, "tokenizer.json")
remaining_min = lambda: cfg["session_minutes"] - (time.time() - T0) / 60  # noqa: E731
OUTPUT_MARGIN = 20  # minutes kept free at the end for writing outputs


def ckpt_step(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 (truncated file from a killed run)
        return None


def ladder():
    """Training configurations to try in order; all keep tokens per optimizer step constant."""
    b, acc = cfg["batch"], cfg["accum"]
    rungs = []
    if not SAFE_MODE and cfg.get("compile", True):
        rungs.append(dict(gpus=ngpu, compile=True, batch=b, accum=acc))
    rungs.append(dict(gpus=ngpu, compile=False, batch=b, accum=acc))
    if b > 1:
        rungs.append(dict(gpus=ngpu, compile=False, batch=b // 2, accum=acc * 2))
    if ngpu > 1:
        rungs.append(dict(gpus=1, compile=False, batch=max(1, b // 2), accum=acc * 2 * ngpu))
    return rungs


def launcher(gpus):
    return f"torchrun --standalone --nproc_per_node={gpus} -m" if gpus > 1 else "python -m"


def train_with_ladder(make_cmd, out_dir, min_minutes=15):
    """Run training, falling down the ladder on failures, until it finishes or time runs out.
    make_cmd(rung, minutes) -> command. Returns True if a run exited cleanly."""
    rungs, i, fails = ladder(), 0, {}
    while remaining_min() - OUTPUT_MARGIN > min_minutes and i < len(rungs):  # all rungs failed -> give up
        rung = rungs[i]
        minutes = remaining_min() - OUTPUT_MARGIN
        t = time.time()
        try:
            cmd = make_cmd(rung, minutes)
            if LOCAL_TEST:
                cmd += " --device cpu"  # never compete with a training run on the local GPU
                if len(OUTCOME["attempts"]) < int(os.environ.get("AUTOLEARN_TEST_FAIL_ATTEMPTS", "0")):
                    cmd = "exit 1"  # fault injection
            run(cmd, env, heartbeat=os.path.join(out_dir, "metrics.jsonl"), stall_min=20)
            OUTCOME["attempts"].append({**rung, "ok": True, "minutes": round((time.time() - t) / 60, 1)})
            return True
        except (subprocess.CalledProcessError, Stalled) as e:
            OUTCOME["attempts"].append({**rung, "ok": False, "minutes": round((time.time() - t) / 60, 1),
                                        "error": type(e).__name__})
            log(f"training attempt failed with {rung} ({type(e).__name__}); "
                f"{remaining_min():.0f} min left, retrying from the last checkpoint")
            # a quick failure points at the configuration -> next rung; a late one (after real progress) may be
            # a transient fault -> retry the same rung once before moving down
            fails[i] = fails.get(i, 0) + 1
            if time.time() - t < 20 * 60 or fails[i] >= 2:
                i += 1
            time.sleep(15)
    return False


def pretrain():
    k, per = state["session"], cfg["files_per_session"]
    files = [cfg["files"][(k * per + i) % len(cfg["files"])] for i in range(per)]
    data_dir = f"{KROOT}/temp/data"
    for marker in glob.glob(f"{KROOT}/input/**/prep_session.json", recursive=True):  # prepared by prep.py
        info = json.load(open(marker))
        d = os.path.dirname(marker)
        if (info.get("session") == k and info.get("files") == files
                and info.get("tokens_per_session") == cfg["tokens_per_session"]
                and os.path.exists(os.path.join(d, "manifest.json"))):
            data_dir = d
            log(f"using data prepared by the CPU prep kernel: {d}")
            break
    if data_dir == f"{KROOT}/temp/data":
        def prep():
            shutil.rmtree(data_dir, ignore_errors=True)
            run(f"python -m autolearn.llm.data pretrain --out {data_dir} --tokens {cfg['tokens_per_session']} "
                f"--val-tokens 3M --tokenizer {tok} --dataset {cfg['dataset']} --data-files {','.join(files)}",
                env, done_file=f"{data_dir}/manifest.json")
        retry(prep, "pretrain data prep", tries=3)

    def cmd(r, minutes):
        resume = "--resume" if os.path.exists(os.path.join(WORK, "last.pt")) else ""
        return (f"{launcher(r['gpus'])} autolearn.llm.train --stage pretrain --preset {cfg['preset']} "
                f"--data {data_dir} --out {WORK} --tokenizer {tok} {resume} {'--compile' if r['compile'] else ''} "
                f"--max-steps {cfg['max_steps']} --stop-after-min {minutes:.0f} --batch {r['batch']} "
                f"--accum {r['accum']} --block {cfg['block']} --lr {cfg['lr']} --warmup {cfg['warmup']} "
                f"--eval-every {cfg['eval_every']} --eval-iters 40")
    train_with_ladder(cmd, WORK)


def sft():
    def prep():
        if os.path.exists(f"{KROOT}/temp/sft/manifest.json"):  # already prepared (earlier attempt / local test)
            return
        shutil.rmtree(f"{KROOT}/temp/sft", ignore_errors=True)
        run(f"python -m autolearn.llm.data sft --out {KROOT}/temp/sft --tokenizer {tok} "
            f"--max-conversations {cfg['sft_conversations']} --max-len {cfg['block']}", env,
            done_file=f"{KROOT}/temp/sft/manifest.json")
    retry(prep, "sft data prep", tries=3)
    out = f"{KROOT}/temp/sft_run"

    def cmd(r, minutes):
        # SFT is short: a retry starts over from the pretrained model with whatever time is left. No torch.compile:
        # compile time would land in the first steps the time budget is measured on and shrink the schedule.
        shutil.rmtree(out, ignore_errors=True)
        return (f"{launcher(r['gpus'])} autolearn.llm.train --stage sft --init {WORK}/best.pt "
                f"--data {KROOT}/temp/sft --out {out} --tokenizer {tok} --time-budget-min {minutes:.0f} "
                f"--max-steps {cfg.get('sft_max_steps', 20000)} "
                f"--batch {r['batch']} --accum {r['accum']} --lr 1e-4 "
                f"--eval-every 200 --eval-iters 40")
    train_with_ladder(cmd, out, min_minutes=30)
    prog = json.load(open(f"{out}/progress.json")) if os.path.exists(f"{out}/progress.json") else {}
    if os.path.exists(f"{out}/best.pt") and prog.get("max_steps"):
        frac = prog["step"] / prog["max_steps"]
        log(f"SFT reached step {prog['step']}/{prog['max_steps']} ({frac:.0%})")
        shutil.copy(f"{out}/best.pt", os.path.join(WORK, "sft_best.pt"))
        if frac >= 0.6:  # good enough to ship; otherwise the next session redoes SFT
            return True
    return False


def finalize():
    """Write the new state from whatever progress exists (called no matter how the session went)."""
    new = dict(START)
    if START["stage"] == "pretrain":
        ck = ckpt_step(os.path.join(WORK, "last.pt"))
        if ck and ck["step"] > START.get("step", 0):
            new.update(session=START["session"] + 1, step=ck["step"], val_loss=ck["val_loss"])
            if ck["step"] >= cfg["max_steps"]:
                new["stage"] = "sft"
    elif START["stage"] == "sft" and OUTCOME.get("sft_ok"):
        if os.path.exists(os.path.join(WORK, "last.pt")):
            os.remove(os.path.join(WORK, "last.pt"))  # optimizer state no longer needed
        new.update(stage="done", session=START["session"] + 1)
    OUTCOME.update(minutes=round((time.time() - T0) / 60, 1), start=START, end=new,
                   progressed=new["session"] > START["session"])
    json.dump(OUTCOME, open(os.path.join(WORK, "outcome.json"), "w"))
    if new["session"] > START["session"]:
        json.dump(new, open(os.path.join(WORK, "state.json"), "w"))
        log("SESSION DONE " + json.dumps(new))
    else:
        log("SESSION NO PROGRESS " + json.dumps(OUTCOME)[:2000])


try:
    if state["stage"] == "pretrain":
        pretrain()
    elif state["stage"] == "sft":
        OUTCOME["sft_ok"] = sft()
except BaseException as e:  # noqa: BLE001 (never lose progress to an unexpected error)
    OUTCOME["errors"].append(f"fatal: {type(e).__name__}: {e}")
    log(traceback.format_exc())
finally:
    finalize()
sys.exit(0)
