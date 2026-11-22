"""scripts/mac_watchdog.py resumes a training run that dies, and exits once the run has finished."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/llm/pretrain"
pytestmark = pytest.mark.skipif(not (DATA / "manifest.json").exists(), reason="needs locally prepared data")


def test_watchdog_resumes_killed_run(tmp_path):
    out = tmp_path / "run"
    args = ["-m", "autolearn.llm.train", "--stage", "pretrain", "--preset", "nano", "--data", str(DATA),
            "--out", str(out), "--block", "64", "--batch", "2", "--accum", "1", "--max-steps", "150",
            "--eval-every", "25", "--eval-iters", "2", "--device", "cpu", "--resume"]
    log = tmp_path / "run.log"
    env = dict(os.environ, AUTOLEARN_WATCHDOG_ARGS=json.dumps(args))
    wd = subprocess.Popen([sys.executable, str(ROOT / "scripts/mac_watchdog.py"), "--log", str(log),
                           "--interval", "1"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        # wait for the first checkpoint, then kill the run the way a crash would
        t0 = time.time()
        while not (out / "last.pt").exists():
            assert time.time() - t0 < 300, "run never checkpointed"
            time.sleep(1)
        pid = int(subprocess.run(["pgrep", "-f", f"autolearn.llm.train.*--out {out}"], capture_output=True,
                                 text=True).stdout.split()[0])
        os.kill(pid, signal.SIGKILL)
        stdout, _ = wd.communicate(timeout=600)
    finally:
        if wd.poll() is None:
            wd.kill()
    assert wd.returncode == 0, stdout
    assert stdout.count("started training run") == 2, stdout
    assert "training finished" in stdout
    assert json.loads((out / "progress.json").read_text())["done"] is True
    assert "resuming from last checkpoint" in log.read_text()
