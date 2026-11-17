"""One Kaggle GPU session of the AutoLearn LLM campaign (runs inside a Kaggle kernel).

Inputs (Kaggle datasets mounted under /kaggle/input):
  autolearn-code : tokenizer.json, campaign.json   (the Python package is embedded in this script)
  autolearn-ckpt : state.json, last.pt (model+optimizer), best.pt, metrics.jsonl  (absent on session 0)
Outputs (/kaggle/working, downloaded by the local campaign driver after the session):
  state.json, last.pt, best.pt, metrics.jsonl, [sft_best.pt]
"""

import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import time

WORK = "/kaggle/working"
T0 = time.time()


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits[0] if hits else None


def run(cmd, env=None, done_file=None):
    """Run a command; if done_file is given, also treat its appearance as completion (guards against a child
    that finished its work but hangs on exit), killing the child after a short grace period."""
    print(f"$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, env=env)
    seen = None
    while p.poll() is None:
        time.sleep(10)
        if done_file and os.path.exists(done_file):
            seen = seen or time.time()
            if time.time() - seen > 60:
                print(f"{done_file} written but process still alive; killing it", flush=True)
                p.kill()
                p.wait()
                return
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, cmd)


campaign_path = find("campaign.json")
assert campaign_path, "autolearn-code dataset not attached"
CODE = os.path.dirname(campaign_path)
cfg = json.load(open(campaign_path))
state_path = find("state.json")
state = json.load(open(state_path)) if state_path else {"session": 0, "stage": "pretrain"}
CK = os.path.dirname(state_path) if state_path else None
print("campaign", cfg, "\nstate", state, flush=True)
EXPECT_SESSION = None  # __EXPECT_SESSION__
if EXPECT_SESSION is not None and state["session"] != EXPECT_SESSION:
    # Kaggle mounted an older checkpoint version (new upload still processing): stop before spending GPU time
    print("STALE_CHECKPOINT", json.dumps({"mounted": state["session"], "expected": EXPECT_SESSION}), flush=True)
    raise SystemExit(0)

# package: source files are embedded below by campaign.py (Kaggle mangles uploaded archives/folders)
EMBEDDED = {}  # __EMBEDDED_PACKAGE__
pkg_root = "/kaggle/temp/code"
assert EMBEDDED, "package not embedded; launch via scripts/kaggle/campaign.py"
for rel, b64 in EMBEDDED.items():
    dst = os.path.join(pkg_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(base64.b64decode(b64))
env = dict(os.environ, PYTHONPATH=pkg_root, HF_HUB_DISABLE_TELEMETRY="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
run("pip install -q 'tokenizers==0.23.3' datasets certifi", env)

import torch  # noqa: E402

ngpu = max(1, torch.cuda.device_count())
print(f"GPUs: {ngpu} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)
launcher = f"torchrun --standalone --nproc_per_node={ngpu} -m" if ngpu > 1 else "python -m"

# carry over checkpoints from the previous session
for f in ("last.pt", "best.pt", "metrics.jsonl", "sft_best.pt"):
    if CK and os.path.exists(os.path.join(CK, f)):
        shutil.copy(os.path.join(CK, f), os.path.join(WORK, f))
if state["session"] > 0 and state["stage"] == "pretrain" and not os.path.exists(os.path.join(WORK, "last.pt")):
    raise SystemExit("expected a checkpoint from the previous session but none was mounted")

tok = os.path.join(CODE, "tokenizer.json")
remaining_min = lambda: cfg["session_minutes"] - (time.time() - T0) / 60

if state["stage"] == "pretrain":
    k, per = state["session"], cfg["files_per_session"]
    files = [cfg["files"][(k * per + i) % len(cfg["files"])] for i in range(per)]
    data_dir = "/kaggle/temp/data"
    for marker in glob.glob("/kaggle/input/**/prep_session.json", recursive=True):  # prepared by prep.py
        info = json.load(open(marker))
        d = os.path.dirname(marker)
        if (info.get("session") == k and info.get("files") == files
                and info.get("tokens_per_session") == cfg["tokens_per_session"]
                and os.path.exists(os.path.join(d, "manifest.json"))):
            data_dir = d
            print(f"using data prepared by the CPU prep kernel: {d}", flush=True)
            break
    if data_dir == "/kaggle/temp/data":
        run(f"python -m autolearn.llm.data pretrain --out /kaggle/temp/data --tokens {cfg['tokens_per_session']} "
            f"--val-tokens 3M --tokenizer {tok} --dataset {cfg['dataset']} --data-files {','.join(files)}", env,
            done_file="/kaggle/temp/data/manifest.json")
    train_min = max(10, remaining_min() - 25)  # leave time to write outputs
    run(f"{launcher} autolearn.llm.train --stage pretrain --preset {cfg['preset']} --data {data_dir} "
        f"--out {WORK} --tokenizer {tok} {'--resume' if state['session'] > 0 else ''} "
        f"--max-steps {cfg['max_steps']} --stop-after-min {train_min:.0f} --batch {cfg['batch']} "
        f"--accum {cfg['accum']} --block {cfg['block']} --lr {cfg['lr']} --warmup {cfg['warmup']} "
        f"--eval-every {cfg['eval_every']} --eval-iters 40", env)
    ck = torch.load(os.path.join(WORK, "last.pt"), map_location="cpu", weights_only=False)
    state.update(session=k + 1, step=ck["step"], val_loss=ck["val_loss"])
    if ck["step"] >= cfg["max_steps"]:
        state["stage"] = "sft"
elif state["stage"] == "sft":
    run(f"python -m autolearn.llm.data sft --out /kaggle/temp/sft --tokenizer {tok} "
        f"--max-conversations {cfg['sft_conversations']} --max-len {cfg['block']}", env,
        done_file="/kaggle/temp/sft/manifest.json")
    train_min = max(10, remaining_min() - 25)
    run(f"{launcher} autolearn.llm.train --stage sft --init {WORK}/best.pt --data /kaggle/temp/sft "
        f"--out /kaggle/temp/sft_run --tokenizer {tok} --time-budget-min {train_min:.0f} --batch {cfg['batch']} "
        f"--accum {cfg['accum']} --lr 1e-4 --eval-every 200 --eval-iters 40", env)
    shutil.copy("/kaggle/temp/sft_run/best.pt", os.path.join(WORK, "sft_best.pt"))
    os.remove(os.path.join(WORK, "last.pt"))  # optimizer state no longer needed
    state.update(stage="done", session=state["session"] + 1)

json.dump(state, open(os.path.join(WORK, "state.json"), "w"))
print("SESSION DONE", json.dumps(state), flush=True)
