"""Multi-week LLM training on Kaggle's free GPUs, fully automated from your machine.

Each Kaggle GPU session lasts up to 12 h, and the free quota is ~30 GPU-hours per week. This driver:
  1. uploads the code + tokenizer as a private Kaggle dataset,
  2. pushes a private GPU kernel (scripts/kaggle/session.py) that streams fresh FineWeb-Edu shards,
     resumes from the last checkpoint and trains until the session ends,
  3. downloads the checkpoint, re-uploads it as a private dataset for the next session, and repeats,
  4. waits and retries when the weekly GPU quota runs out,
  5. after pretraining, runs chat fine-tuning (SFT) and publishes the model to the local LLM registry.

Prereq: a phone-verified Kaggle account and an API token (https://www.kaggle.com/settings):
either ~/.kaggle/access_token (+ --user <username>) or the legacy ~/.kaggle/kaggle.json.

    nohup .venv/bin/python scripts/kaggle/campaign.py > logs/kaggle_campaign.log 2>&1 &
"""

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")


def sh(args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {out}")
    return out


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def fineweb_files(subset):
    os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())
    from huggingface_hub import list_repo_files
    return sorted(f for f in list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
                  if f.startswith(f"sample/{subset}/") and f.endswith(".parquet"))


def dataset_exists(ref):
    # Kaggle answers 403 (not 404) for datasets that don't exist, so list our own datasets instead.
    slug = ref.split("/", 1)[1]
    out = sh(["datasets", "list", "--mine", "--search", slug, "--csv"], check=False)
    return any(line.split(",")[0] == ref for line in out.splitlines())


def upload_dataset(path: Path, ref: str, title: str, message: str):
    meta = {"title": title, "id": ref, "licenses": [{"name": "other"}]}
    (path / "dataset-metadata.json").write_text(json.dumps(meta))
    if dataset_exists(ref):
        sh(["datasets", "version", "-p", str(path), "-m", message, "-q"])
    else:
        sh(["datasets", "create", "-p", str(path), "-q"])  # private by default
    for _ in range(120):  # wait until processed so the next kernel mounts the new version
        st = sh(["datasets", "status", ref], check=False).lower()
        if "ready" in st:
            return
        if "error" in st:
            raise RuntimeError(f"dataset {ref} failed: {st}")
        time.sleep(15)
    raise RuntimeError(f"dataset {ref} not ready in time")


def kernel_status(ref):
    s = sh(["kernels", "status", ref], check=False).lower()
    for k in ("complete", "error", "cancel", "running", "queued"):
        if k in s:
            return k, s
    return "unknown", s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", default=None, help="Kaggle username (needed with access_token auth)")
    ap.add_argument("--preset", default="small")
    ap.add_argument("--target-tokens", default="4B", help="total pretraining tokens")
    ap.add_argument("--tokens-per-session", default="1400M")
    ap.add_argument("--files-per-session", type=int, default=2)
    ap.add_argument("--subset", default="100BT")
    ap.add_argument("--batch", type=int, default=4)  # 8 x 1024 OOMs on a 15GB T4
    ap.add_argument("--accum", type=int, default=16)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--gpus-assumed", type=int, default=2, help="for converting target tokens to steps")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--session-minutes", type=int, default=690, help="Kaggle hard limit is 720")
    ap.add_argument("--sft-conversations", type=int, default=300_000)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/llm/pretrain/tokenizer.json"))
    ap.add_argument("--llm-dir", default=str(ROOT / "models/llm"))
    ap.add_argument("--work", default=str(ROOT / "kaggle_campaign"))
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--quota-wait", type=int, default=3 * 3600)
    a = ap.parse_args()

    from autolearn.llm.data import parse_count
    cred = Path.home() / ".kaggle" / "kaggle.json"
    token = Path.home() / ".kaggle" / "access_token"
    if cred.exists():
        user = a.user or json.loads(cred.read_text())["username"]
    elif token.exists() or os.environ.get("KAGGLE_API_TOKEN"):
        user = a.user or os.environ.get("KAGGLE_USERNAME")
        if not user:
            sys.exit("using an access token: pass --user <your kaggle username>")
    else:
        sys.exit("no Kaggle credentials: create a token at https://www.kaggle.com/settings")
    work = Path(a.work)
    code, ckpt, kdir = work / "code", work / "ckpt", work / "kernel"
    for d in (code, ckpt, kdir):
        d.mkdir(parents=True, exist_ok=True)
    code_ref, ckpt_ref, kernel_ref = f"{user}/autolearn-code", f"{user}/autolearn-ckpt", f"{user}/autolearn-train"

    tokens_per_step = a.batch * a.block * a.accum * a.gpus_assumed
    max_steps = math.ceil(parse_count(a.target_tokens) / tokens_per_step)
    campaign = {"preset": a.preset, "dataset": "HuggingFaceFW/fineweb-edu", "files": fineweb_files(a.subset),
                "files_per_session": a.files_per_session, "tokens_per_session": a.tokens_per_session,
                "max_steps": max_steps, "batch": a.batch, "accum": a.accum, "block": a.block, "lr": a.lr,
                "warmup": min(1000, max_steps // 30), "eval_every": 500, "session_minutes": a.session_minutes,
                "sft_conversations": a.sft_conversations}
    log(f"campaign: {a.preset}, {a.target_tokens} tokens = {max_steps} steps x {tokens_per_step} tokens")

    # 1. code dataset
    (code / "autolearn.tar.gz").unlink(missing_ok=True)
    shutil.copy(a.tokenizer, code / "tokenizer.json")
    (code / "campaign.json").write_text(json.dumps(campaign))
    upload_dataset(code, code_ref, "autolearn-code", "code update")
    log("code dataset uploaded")

    # 2. checkpoint dataset (state only on first run)
    if not (ckpt / "state.json").exists():
        (ckpt / "state.json").write_text(json.dumps({"session": 0, "stage": "pretrain"}))
        upload_dataset(ckpt, ckpt_ref, "autolearn-ckpt", "init")

    def build_kernel():
        """Re-embed the current package before every push so code fixes reach the next session."""
        embedded = {str(f.relative_to(ROOT)): base64.b64encode(f.read_bytes()).decode()
                    for f in sorted((ROOT / "autolearn").rglob("*.py")) if "__pycache__" not in f.parts}
        src = (ROOT / "scripts/kaggle/session.py").read_text()
        marker = "EMBEDDED = {}  # __EMBEDDED_PACKAGE__"
        assert marker in src
        (kdir / "session.py").write_text(src.replace(marker, f"EMBEDDED = {embedded!r}"))
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel_ref, "title": "autolearn-train", "code_file": "session.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [code_ref, ckpt_ref],
            "competition_sources": [], "kernel_sources": []}))

    while True:
        state = json.loads((ckpt / "state.json").read_text())
        if state["stage"] == "done":
            break
        st, _ = kernel_status(kernel_ref)
        if st in ("running", "queued"):  # restarted driver: attach to the session already in flight
            log(f"attaching to running session {state['session']}")
        else:
            log(f"pushing session {state['session']} (stage={state['stage']}, "
                f"step={state.get('step', 0)}/{max_steps})")
            build_kernel()
            out = sh(["kernels", "push", "-p", str(kdir)], check=False)
            if "error" in out.lower() or "quota" in out.lower():
                log("push failed (likely weekly GPU quota):", out[:300], f"- retrying in {a.quota_wait // 3600}h")
                time.sleep(a.quota_wait)
                continue
            time.sleep(60)
        while True:
            st, raw = kernel_status(kernel_ref)
            if st in ("complete", "error", "cancel"):
                break
            time.sleep(a.poll)
        tmp = work / "download"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        sh(["kernels", "output", kernel_ref, "-p", str(tmp)], check=False)
        logs = list(tmp.glob("*.log"))
        if logs:
            shutil.copy(logs[0], work / f"session_{state['session']}.log")
        new_state = json.loads((tmp / "state.json").read_text()) if (tmp / "state.json").exists() else None
        if st != "complete" or not new_state or new_state["session"] <= state["session"]:
            tail = logs[0].read_text()[-2000:] if logs else raw
            log(f"session {state['session']} did not finish cleanly ({st}). Tail:\n{tail}")
            if "quota" in tail.lower() or "gpu" in raw:
                time.sleep(a.quota_wait)
                continue
            sys.exit(1)
        for f in tmp.iterdir():
            if f.suffix in (".pt", ".json", ".jsonl"):
                shutil.move(str(f), ckpt / f.name)
        log(f"session done: {json.dumps(new_state)}")
        upload_dataset(ckpt, ckpt_ref, "autolearn-ckpt", f"after session {new_state['session']}")

    # 5. publish
    sys.path.insert(0, str(ROOT))
    from autolearn.llm.engine import LLMRegistry
    v = LLMRegistry(a.llm_dir).publish(ckpt / "sft_best.pt", a.tokenizer,
                                       {"stage": "sft", "preset": a.preset, "trained_on": "kaggle",
                                        "pretrain_tokens": a.target_tokens})
    log(f"campaign complete: published {v} to {a.llm_dir}")


if __name__ == "__main__":
    main()
