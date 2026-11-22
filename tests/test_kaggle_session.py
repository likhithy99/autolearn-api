"""scripts/kaggle/session.py run for real on this machine (CPU, tiny model): a pretraining session whose first
training attempt crashes, then the chat fine-tuning session. Needs the locally prepared data (skipped in CI)."""

import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRE, SFT = ROOT / "data/llm/pretrain", ROOT / "data/llm/sft"
pytestmark = pytest.mark.skipif(not (PRE / "manifest.json").exists() or not (SFT / "manifest.json").exists(),
                                reason="needs locally prepared pretrain + sft data")


def embedded_session(dst: Path):
    pkg = {str(f.relative_to(ROOT)): base64.b64encode(f.read_bytes()).decode()
           for f in sorted((ROOT / "autolearn").rglob("*.py")) if "__pycache__" not in f.parts}
    src = (ROOT / "scripts/kaggle/session.py").read_text()
    src = src.replace("EMBEDDED = {}  # __EMBEDDED_PACKAGE__", f"EMBEDDED = {pkg!r}")
    dst.write_text(src)
    return dst


def run_session(k: Path, extra_env=None):
    env = dict(os.environ, AUTOLEARN_KAGGLE_ROOT=str(k), PATH=f"{Path(sys.executable).parent}:{os.environ['PATH']}",
               **(extra_env or {}))
    script = embedded_session(k / "session.py")
    r = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


def test_session_recovers_and_completes(tmp_path):
    k = tmp_path / "k"
    code, prep = k / "input/code", k / "input/prep/data"
    for d in (code, prep, k / "working", k / "temp"):
        d.mkdir(parents=True)
    shutil.copy(PRE / "tokenizer.json", code / "tokenizer.json")
    cfg = {"preset": "nano", "dataset": "x", "files": ["a", "b"], "files_per_session": 2,
           "tokens_per_session": "1M", "max_steps": 20, "batch": 2, "accum": 1, "block": 128, "lr": 1e-3,
           "warmup": 2, "eval_every": 10, "session_minutes": 60, "sft_conversations": 10, "sft_max_steps": 12}
    (code / "campaign.json").write_text(json.dumps(cfg))
    for f in PRE.iterdir():
        if f.name.startswith(("train_0000", "val_")) or f.name == "manifest.json":
            (prep / f.name).symlink_to(f)
    (prep / "prep_session.json").write_text(json.dumps({"session": 0, "files": ["a", "b"],
                                                        "tokens_per_session": "1M"}))

    # session 0: the first training attempt crashes; the ladder retries and finishes pretraining
    out = run_session(k, {"AUTOLEARN_TEST_FAIL_ATTEMPTS": "1"})
    assert "training attempt failed" in out and "SESSION DONE" in out
    state = json.loads((k / "working/state.json").read_text())
    assert state == {**state, "session": 1, "stage": "sft", "step": 20}
    outcome = json.loads((k / "working/outcome.json").read_text())
    assert [x["ok"] for x in outcome["attempts"]] == [False, True]

    # session 1: chat fine-tuning from the pretrained checkpoint
    ck = k / "input/ckpt"
    ck.mkdir()
    for f in ("state.json", "last.pt", "best.pt"):
        shutil.move(str(k / "working" / f), ck / f)
    shutil.rmtree(k / "working")
    (k / "working").mkdir()
    (k / "temp/sft").symlink_to(SFT)
    out = run_session(k)
    assert "SESSION DONE" in out, out[-2000:]
    state = json.loads((k / "working/state.json").read_text())
    assert state["stage"] == "done" and state["session"] == 2
    assert (k / "working/sft_best.pt").exists() and not (k / "working/last.pt").exists()


def test_session_without_progress_writes_no_state(tmp_path):
    """Every attempt fails: no state.json (the driver retries the session), but outcome.json explains why."""
    k = tmp_path / "k"
    code, prep = k / "input/code", k / "input/prep/data"
    for d in (code, prep, k / "working", k / "temp"):
        d.mkdir(parents=True)
    shutil.copy(PRE / "tokenizer.json", code / "tokenizer.json")
    cfg = {"preset": "nano", "dataset": "x", "files": ["a", "b"], "files_per_session": 2,
           "tokens_per_session": "1M", "max_steps": 20, "batch": 2, "accum": 1, "block": 128, "lr": 1e-3,
           "warmup": 2, "eval_every": 10, "session_minutes": 37, "sft_conversations": 10}
    (code / "campaign.json").write_text(json.dumps(cfg))
    for f in PRE.iterdir():
        if f.name.startswith(("train_0000", "val_")) or f.name == "manifest.json":
            (prep / f.name).symlink_to(f)
    (prep / "prep_session.json").write_text(json.dumps({"session": 0, "files": ["a", "b"],
                                                        "tokens_per_session": "1M"}))
    out = run_session(k, {"AUTOLEARN_TEST_FAIL_ATTEMPTS": "99"})
    assert "SESSION NO PROGRESS" in out
    assert not (k / "working/state.json").exists()
    outcome = json.loads((k / "working/outcome.json").read_text())
    assert outcome["attempts"] and not any(x["ok"] for x in outcome["attempts"])
