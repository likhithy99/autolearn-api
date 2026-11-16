"""Kaggle CPU kernel: prepares (downloads + tokenizes) the training shards for one upcoming GPU session.

CPU sessions don't count against the weekly GPU quota, so running data prep here while the previous GPU
session trains gives each GPU session ~1 more hour of actual training. campaign.py embeds the package and the
target session number below before pushing. The GPU session finds the result via prep_session.json and falls
back to preparing data itself if it is missing or for a different session.
"""

import base64
import glob
import json
import os
import subprocess
import sys

PREP_SESSION = None  # __PREP_SESSION__
EMBEDDED = {}  # __EMBEDDED_PACKAGE__

campaign_path = sorted(glob.glob("/kaggle/input/**/campaign.json", recursive=True))
assert campaign_path, "autolearn-code dataset not attached"
CODE = os.path.dirname(campaign_path[0])
cfg = json.load(open(campaign_path[0]))
assert PREP_SESSION is not None and EMBEDDED, "launch via scripts/kaggle/campaign.py"

pkg_root = "/kaggle/temp/code"
for rel, b64 in EMBEDDED.items():
    dst = os.path.join(pkg_root, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "wb") as fh:
        fh.write(base64.b64decode(b64))
env = dict(os.environ, PYTHONPATH=pkg_root, HF_HUB_DISABLE_TELEMETRY="1")
subprocess.run("pip install -q 'tokenizers==0.23.3' datasets certifi", shell=True, check=True, env=env)

k, per = PREP_SESSION, cfg["files_per_session"]
files = [cfg["files"][(k * per + i) % len(cfg["files"])] for i in range(per)]
out = "/kaggle/working/data"
cmd = (f"{sys.executable} -m autolearn.llm.data pretrain --out {out} --tokens {cfg['tokens_per_session']} "
       f"--val-tokens 3M --tokenizer {os.path.join(CODE, 'tokenizer.json')} --dataset {cfg['dataset']} "
       f"--data-files {','.join(files)}")
print("$", cmd, flush=True)
subprocess.run(cmd, shell=True, check=True, env=env)
assert os.path.exists(os.path.join(out, "manifest.json")), "data prep produced no manifest"
json.dump({"session": k, "files": files, "tokens_per_session": cfg["tokens_per_session"]},
          open(os.path.join(out, "prep_session.json"), "w"))
print("PREP DONE", k, files, flush=True)
