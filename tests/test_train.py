"""Training loop: pretrain -> resume -> stop-after (multi-session) -> SFT, on a tiny CPU model."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from autolearn.llm.data import ShardWriter
from autolearn.llm.tokenizer import train_tokenizer

TEXT = ["Rivers flow to the sea and the sea feeds the clouds that rain on hills."] * 300


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    tok = train_tokenizer(TEXT, 300, str(d / "tokenizer.json"))
    ids = tok.encode(" ".join(TEXT))
    for split in ("train", "val"):
        np.asarray(ids, dtype=np.uint16).tofile(d / f"{split}_0000.bin")
    (d / "manifest.json").write_text(json.dumps({"tokenizer": str(d / "tokenizer.json")}))
    sft = d / "sft"
    sft.mkdir()
    for split in ("train", "val"):
        w = ShardWriter(sft, split, with_mask=True)
        for _ in range(60):
            w.write(*tok.render_chat([{"role": "user", "content": "Where do rivers go?"},
                                      {"role": "assistant", "content": "Rivers flow to the sea."}],
                                     add_generation_prompt=False))
        w.close()
    (sft / "manifest.json").write_text(json.dumps({"tokenizer": str(d / "tokenizer.json")}))
    return d


def train(*args, block="64"):
    cmd = [sys.executable, "-m", "autolearn.llm.train", "--device", "cpu", "--preset", "nano", "--block", block,
           "--batch", "4", "--accum", "1", "--eval-iters", "2", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


def test_pretrain_learns(corpus, tmp_path):
    out = train("--data", str(corpus), "--out", str(tmp_path), "--max-steps", "60", "--eval-every", "30",
                "--lr", "3e-3", "--warmup", "5")
    vals = [r["val_loss"] for r in out if "val_loss" in r]
    assert vals[-1] < vals[0] and out[-1]["done"]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()


def test_multi_session_resume(corpus, tmp_path):
    # session 1 stops immediately after its first step (stop-after), keeping the full schedule
    out = train("--data", str(corpus), "--out", str(tmp_path), "--max-steps", "40", "--eval-every", "20",
                "--stop-after-min", "0.0001")
    stop = [r for r in out if r.get("session_stop")]
    assert stop and stop[0]["of"] == 40
    step1 = torch.load(tmp_path / "last.pt", weights_only=False)["step"]
    # session 2 resumes model + optimizer and finishes the schedule
    out = train("--data", str(corpus), "--out", str(tmp_path), "--max-steps", "40", "--eval-every", "20",
                "--resume")
    assert out[-1]["done"] and out[-1]["steps"] == 40
    assert torch.load(tmp_path / "last.pt", weights_only=False)["step"] == 40 > step1


def test_sft_from_pretrained(corpus, tmp_path):
    train("--data", str(corpus), "--out", str(tmp_path / "pre"), "--max-steps", "20", "--eval-every", "20",
          block="128")
    out = train("--stage", "sft", "--init", str(tmp_path / "pre" / "best.pt"), "--data", str(corpus / "sft"),
                "--out", str(tmp_path / "sft"), "--max-steps", "20", "--eval-every", "10", "--lr", "1e-3",
                block="128")
    vals = [r["val_loss"] for r in out if "val_loss" in r]
    assert vals[-1] < vals[0]
    ck = torch.load(tmp_path / "sft" / "best.pt", weights_only=False)
    assert ck["meta"]["stage"] == "sft"
