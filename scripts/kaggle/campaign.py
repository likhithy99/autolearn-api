"""Multi-week LLM training on Kaggle's free GPUs, fully automated from your machine.

Each Kaggle GPU session lasts up to 12 h, and the free quota is 30 GPU-hours per week. This driver:
  1. uploads the code + tokenizer as a private Kaggle dataset,
  2. pushes a private GPU kernel (scripts/kaggle/session.py) that streams fresh FineWeb-Edu shards,
     resumes from the last checkpoint and trains until the session ends,
  3. downloads the checkpoint, re-uploads it as a private dataset for the next session, and repeats,
  4. tracks GPU hours used this week and sizes each session to fit, waiting for the weekly reset when needed,
  5. after pretraining, runs chat fine-tuning (SFT) and publishes the model to the local LLM registry.

It is built to keep going without supervision: every Kaggle call is retried, any session that made progress is
kept (even if it ended in an error), failed sessions are retried with growing waits and then in a safe mode, and
an unexpected exception in the driver itself is logged and retried instead of exiting. Run it under launchd
(scripts/kaggle/install_launchd.sh) so it also survives crashes and reboots. Only one driver runs at a time.

Prereq: a phone-verified Kaggle account and an API token (https://www.kaggle.com/settings):
either ~/.kaggle/access_token (+ --user <username>) or the legacy ~/.kaggle/kaggle.json.

    .venv/bin/python scripts/kaggle/campaign.py --user <you> >> logs/kaggle_campaign.log 2>&1
"""

import argparse
import base64
import datetime as dt
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
KAGGLE = str(ROOT / ".venv" / "bin" / "kaggle")
QUOTA_MIN = 30 * 60          # Kaggle free GPU quota per week
QUOTA_MARGIN_MIN = 20        # never plan a session closer than this to the quota
POLL_LIMIT_H = 13            # a kernel "running" longer than this is treated as lost (Kaggle's cap is 12 h)


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def sh(args, check=True, tries=4):
    """Run the Kaggle CLI, retrying transient failures (network, 5xx, rate limits)."""
    out = ""
    for i in range(tries):
        try:
            r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=1800)
            out = (r.stdout + r.stderr).strip()
            transient = r.returncode != 0 and any(k in out.lower() for k in (
                "timed out", "timeout", "connection", "temporarily", "502", "503", "504", "500", "429",
                "too many requests", "remote end closed", "ssl"))
            if not transient:
                if check and r.returncode != 0:
                    raise RuntimeError(f"kaggle {' '.join(args)} failed: {out}")
                return out
        except subprocess.TimeoutExpired:
            out = "timeout"
        if i < tries - 1:
            time.sleep(30 * 2 ** i)
    if check:
        raise RuntimeError(f"kaggle {' '.join(args)} failed after {tries} tries: {out}")
    return out


def fineweb_files(subset):
    os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())
    from huggingface_hub import list_repo_files
    for i in range(5):
        try:
            return sorted(f for f in list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
                          if f.startswith(f"sample/{subset}/") and f.endswith(".parquet"))
        except Exception as e:  # noqa: BLE001
            log(f"listing FineWeb files failed ({e}); retrying")
            time.sleep(60 * 2 ** i)
    raise RuntimeError("could not list FineWeb-Edu files")


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
    time.sleep(60)  # status can still report the previous version as ready right after a push
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


# ------------------------------------------------------------------ quota --
def week_start(now: float) -> float:
    """Kaggle's GPU quota week starts Saturday 00:00 UTC."""
    d = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    d = d.replace(hour=0, minute=0, second=0, microsecond=0) - dt.timedelta(days=(d.weekday() - 5) % 7)
    return d.timestamp()


class Ledger:
    """GPU minutes this driver has used, so sessions are sized to the quota that is left (a session Kaggle stops
    for lack of quota loses all its outputs)."""

    def __init__(self, path: Path):
        self.path = path
        self.d = json.loads(path.read_text()) if path.exists() else {"sessions": [], "blocked_until": 0}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(self.path)

    def used_min(self, now=None):
        now = now or time.time()
        ws = week_start(now)
        return sum(max(0, min(s["end"], now) - max(s["start"], ws)) / 60 + 5 for s in self.d["sessions"]
                   if s["end"] > ws)  # +5 min per session for startup/teardown we can't observe

    def remaining_min(self):
        return QUOTA_MIN - self.used_min()

    def open_session(self, start):
        self.d["sessions"].append({"start": start, "end": start})
        self.d["sessions"] = self.d["sessions"][-200:]
        self.save()

    def touch(self, end):
        if self.d["sessions"]:
            self.d["sessions"][-1]["end"] = end
            self.save()

    def block(self, until, why):
        self.d["blocked_until"] = until
        self.save()
        log(f"GPU quota: {why}; next attempt {time.strftime('%a %Y-%m-%d %H:%M', time.localtime(until))}")

    def blocked(self):
        return time.time() < self.d.get("blocked_until", 0)


def sleep_until(t, chunk=3600):
    while time.time() < t:
        time.sleep(max(1, min(chunk, t - time.time())))


# ------------------------------------------------------------------- main --
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
    ap.add_argument("--session-minutes", type=int, default=255, help="pretraining session length (Kaggle cap 720)")
    ap.add_argument("--sft-session-minutes", type=int, default=150)
    ap.add_argument("--sft-conversations", type=int, default=200_000)
    ap.add_argument("--tokenizer", default=str(ROOT / "data/llm/pretrain/tokenizer.json"))
    ap.add_argument("--llm-dir", default=str(ROOT / "models/llm"))
    ap.add_argument("--work", default=str(ROOT / "kaggle_campaign"))
    ap.add_argument("--poll", type=int, default=300)
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
    code, ckpt, kdir, pdir, outcomes = (work / "code", work / "ckpt", work / "kernel", work / "prep",
                                        work / "outcomes")
    for d in (code, ckpt, kdir, pdir, outcomes):
        d.mkdir(parents=True, exist_ok=True)

    # one driver at a time (a second copy, e.g. from launchd, waits until the first exits)
    lock = open(work / "driver.lock", "w")
    while True:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(60)

    code_ref, ckpt_ref, kernel_ref = f"{user}/autolearn-code", f"{user}/autolearn-ckpt", f"{user}/autolearn-train"
    prep_ref = f"{user}/autolearn-prep"
    prep_record = work / "prep.json"  # which session the prep kernel's latest run is for
    driver_state = work / "driver.json"  # consecutive failures (survives driver restarts)
    ledger = Ledger(work / "quota.json")

    def dstate():
        return json.loads(driver_state.read_text()) if driver_state.exists() else {"failures": 0}

    def set_failures(n):
        driver_state.write_text(json.dumps({"failures": n}))

    tokens_per_step = a.batch * a.block * a.accum * a.gpus_assumed
    max_steps = math.ceil(parse_count(a.target_tokens) / tokens_per_step)
    campaign = {"preset": a.preset, "dataset": "HuggingFaceFW/fineweb-edu", "files": fineweb_files(a.subset),
                "files_per_session": a.files_per_session, "tokens_per_session": a.tokens_per_session,
                "max_steps": max_steps, "batch": a.batch, "accum": a.accum, "block": a.block, "lr": a.lr,
                "warmup": min(1000, max_steps // 30), "eval_every": 500, "session_minutes": a.session_minutes,
                "sft_conversations": a.sft_conversations}
    log(f"campaign: {a.preset}, {a.target_tokens} tokens = {max_steps} steps x {tokens_per_step} tokens; "
        f"GPU quota used this week (tracked): {ledger.used_min() / 60:.1f} h")

    def setup():
        (code / "autolearn.tar.gz").unlink(missing_ok=True)
        shutil.copy(a.tokenizer, code / "tokenizer.json")
        (code / "campaign.json").write_text(json.dumps(campaign))
        upload_dataset(code, code_ref, "autolearn-code", "code update")
        log("code dataset uploaded")
        if not (ckpt / "state.json").exists():
            (ckpt / "state.json").write_text(json.dumps({"session": 0, "stage": "pretrain"}))
            upload_dataset(ckpt, ckpt_ref, "autolearn-ckpt", "init")

    def embed(script: str, extra: dict | None = None) -> str:
        """Re-embed the current package before every push so code fixes reach the next session."""
        embedded = {str(f.relative_to(ROOT)): base64.b64encode(f.read_bytes()).decode()
                    for f in sorted((ROOT / "autolearn").rglob("*.py")) if "__pycache__" not in f.parts}
        src = (ROOT / "scripts/kaggle" / script).read_text()
        marker = "EMBEDDED = {}  # __EMBEDDED_PACKAGE__"
        assert marker in src
        src = src.replace(marker, f"EMBEDDED = {embedded!r}")
        for m, v in (extra or {}).items():
            assert m in src, m
            src = src.replace(m, v)
        return src

    def build_kernel(use_prep: bool, session: int, minutes: int, safe: bool):
        (kdir / "session.py").write_text(embed("session.py", {
            "EXPECT_SESSION = None  # __EXPECT_SESSION__": f"EXPECT_SESSION = {session}",
            "SESSION_MINUTES = None  # __SESSION_MINUTES__": f"SESSION_MINUTES = {minutes}",
            "SAFE_MODE = False  # __SAFE_MODE__": f"SAFE_MODE = {safe}"}))
        (kdir / "kernel-metadata.json").write_text(json.dumps({
            "id": kernel_ref, "title": "autolearn-train", "code_file": "session.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [code_ref, ckpt_ref],
            "competition_sources": [], "kernel_sources": [prep_ref] if use_prep else []}))

    def push_prep(session: int):
        """CPU kernel (no GPU quota) that prepares the data for `session` while the current one trains."""
        (pdir / "prep.py").write_text(embed("prep.py", {"PREP_SESSION = None  # __PREP_SESSION__":
                                                        f"PREP_SESSION = {session}"}))
        (pdir / "kernel-metadata.json").write_text(json.dumps({
            "id": prep_ref, "title": "autolearn-prep", "code_file": "prep.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_internet": True,
            "dataset_sources": [code_ref], "competition_sources": [], "kernel_sources": []}))
        out = sh(["kernels", "push", "-p", str(pdir)], check=False)
        if "error" in out.lower():
            log(f"prep push for session {session} failed (the GPU session will prepare its own data):", out[:200])
            return
        prep_record.write_text(json.dumps({"session": session, "pushed_at": time.time()}))
        log(f"prep kernel pushed for session {session}")

    def prep_for(session: int) -> str:
        """'ready' | 'running' | 'none' for the prep kernel's data for this session."""
        rec = json.loads(prep_record.read_text()) if prep_record.exists() else {}
        if rec.get("session") != session:
            return "none"
        st, _ = kernel_status(prep_ref)
        return "ready" if st == "complete" else "running" if st in ("running", "queued") else "none"

    prep_tried: set = set()

    def iteration() -> bool:
        """Run (or attach to) one session and process its result. Returns True when the campaign is finished."""
        state = json.loads((ckpt / "state.json").read_text())
        if state["stage"] == "done":
            return True
        st, _ = kernel_status(kernel_ref)
        if st in ("running", "queued"):  # restarted driver: attach to the session already in flight
            log(f"attaching to running session {state['session']}")
            ledger.open_session(time.time())
        else:
            if ledger.blocked():
                sleep_until(ledger.d["blocked_until"])
                return False
            sft = state["stage"] == "sft"
            want = a.sft_session_minutes if sft else a.session_minutes
            minutes = int(min(want, ledger.remaining_min() - QUOTA_MARGIN_MIN))
            if minutes < (75 if sft else 60):
                ledger.block(week_start(time.time()) + 7 * 86400 + 600,
                             f"{ledger.remaining_min() / 60:.1f} h left this week, too little for a session")
                return False
            use_prep = False
            if state["stage"] == "pretrain":
                waited = 0
                while prep_for(state["session"]) == "running" and waited < 2 * 3600:
                    time.sleep(120)
                    waited += 120
                use_prep = prep_for(state["session"]) == "ready"
            safe = dstate()["failures"] >= 2
            log(f"pushing session {state['session']} (stage={state['stage']}, step={state.get('step', 0)}/"
                f"{max_steps}, {minutes} min, prepared data: {'yes' if use_prep else 'no'}"
                f"{', SAFE MODE' if safe else ''})")
            build_kernel(use_prep, state["session"], minutes, safe)
            out = sh(["kernels", "push", "-p", str(kdir)], check=False)
            if "quota" in out.lower():
                # Kaggle disagrees with our ledger (usage we didn't see): check again in 3 h
                ledger.block(time.time() + 3 * 3600, f"push refused: {out[:160]}")
                return False
            if "error" in out.lower():
                raise RuntimeError(f"kernel push failed: {out[:300]}")
            ledger.open_session(time.time())
            time.sleep(60)

        # ---- wait for the session to end
        t_attach = time.time()
        while True:
            st, raw = kernel_status(kernel_ref)
            ledger.touch(time.time())
            if st in ("complete", "error", "cancel"):
                break
            if time.time() - t_attach > POLL_LIMIT_H * 3600:
                log(f"session {state['session']} still '{st}' after {POLL_LIMIT_H} h; treating it as lost")
                break
            if st == "running" and state["stage"] == "pretrain":
                nxt = state["session"] + 1
                # once this session has surely mounted its inputs, prepare the next session's data on a CPU kernel
                if (time.time() - t_attach > 900 and nxt not in prep_tried and state.get("step", 0) < max_steps
                        and prep_for(nxt) == "none"):
                    prep_tried.add(nxt)  # one attempt per session; on failure the GPU session preps itself
                    push_prep(nxt)
            time.sleep(a.poll)

        # ---- collect outputs (kept even if the kernel ended in an error)
        tmp = work / "download"
        for i in range(5):
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir()
            sh(["kernels", "output", kernel_ref, "-p", str(tmp)], check=False)
            if any(tmp.iterdir()):
                break
            time.sleep(60 * (i + 1))
        logs = list(tmp.glob("*.log"))
        tail = logs[0].read_text()[-4000:] if logs else raw
        if logs:
            shutil.copy(logs[0], work / f"session_{state['session']}.log")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if (tmp / "outcome.json").exists():
            shutil.copy(tmp / "outcome.json", outcomes / f"session_{state['session']}_{stamp}.json")
        try:
            new_state = json.loads((tmp / "state.json").read_text()) if (tmp / "state.json").exists() else None
        except json.JSONDecodeError:
            new_state = None
        progressed = (new_state is not None and new_state.get("session", -1) > state["session"]
                      and (new_state.get("step", 0) >= state.get("step", 0) or new_state["stage"] != state["stage"]))
        if progressed:
            needed = ["last.pt"] if new_state["stage"] == "pretrain" else []
            needed += ["sft_best.pt"] if new_state["stage"] == "done" else []
            if all((tmp / f).exists() for f in needed):
                for f in tmp.iterdir():
                    if f.suffix in (".pt", ".json", ".jsonl") and f.name != "outcome.json":
                        shutil.move(str(f), ckpt / f.name)
                if new_state["stage"] == "done" and (ckpt / "last.pt").exists():
                    (ckpt / "last.pt").unlink()
                log(f"session done ({st}): {json.dumps(new_state)}")
                set_failures(0)
                if new_state["stage"] != "done":
                    upload_dataset(ckpt, ckpt_ref, "autolearn-ckpt", f"after session {new_state['session']}")
                return False
            log(f"session {state['session']} reported progress but outputs are missing: {needed}")

        # ---- no usable progress: work out why and react (never give up)
        if "STALE_CHECKPOINT" in tail:
            log(f"session {state['session']} mounted a stale checkpoint version; re-pushing in 10 min")
            time.sleep(600)
            return False
        if "MISSING_CHECKPOINT" in tail:
            log("the checkpoint dataset was missing files; re-uploading it")
            upload_dataset(ckpt, ckpt_ref, "autolearn-ckpt", f"re-upload for session {state['session']}")
            return False
        if "quota" in tail.lower() and "maximum weekly" in tail.lower():
            ledger.block(week_start(time.time()) + 7 * 86400 + 600, "session stopped by the weekly quota")
            return False
        n = dstate()["failures"] + 1
        set_failures(n)
        wait = min(4 * 3600, 600 * 2 ** (n - 1))
        log(f"session {state['session']} made no progress ({st}, failure {n} in a row"
            f"{'; next session runs in safe mode' if n >= 2 else ''}); retrying in {wait // 60} min. Tail:\n"
            f"{tail[-1500:]}")
        time.sleep(wait)
        return False

    # ---- main loop: nothing short of finishing the campaign ends it
    errors = 0
    while True:
        try:
            setup()
            break
        except Exception:  # noqa: BLE001
            errors += 1
            log(f"setup failed (attempt {errors}):\n{traceback.format_exc()}")
            time.sleep(min(3600, 120 * 2 ** min(errors, 5)))
    errors = 0
    while True:
        try:
            if iteration():
                break
            errors = 0
        except Exception:  # noqa: BLE001
            errors += 1
            wait = min(3600, 120 * 2 ** min(errors, 5))
            log(f"driver error (#{errors} in a row), retrying in {wait // 60} min:\n{traceback.format_exc()}")
            time.sleep(wait)

    # ---- publish
    for i in range(10):
        try:
            from autolearn.llm.engine import LLMRegistry
            v = LLMRegistry(a.llm_dir).publish(ckpt / "sft_best.pt", a.tokenizer,
                                               {"stage": "sft", "preset": a.preset, "trained_on": "kaggle",
                                                "pretrain_tokens": a.target_tokens})
            log(f"campaign complete: published {v} to {a.llm_dir}")
            return
        except Exception:  # noqa: BLE001
            log(f"publish failed:\n{traceback.format_exc()}")
            time.sleep(300)
    sys.exit(1)


if __name__ == "__main__":
    main()
