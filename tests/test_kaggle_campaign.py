"""The Kaggle campaign driver against a simulated Kaggle (fake CLI + fake clock): weeks of sessions run in seconds.

Checks that the driver never stops on failures, keeps progress from sessions that ended in an error, never plans a
session that the weekly GPU quota can't cover, and finishes the campaign (pretrain -> SFT -> publish)."""

import datetime as dt
import json
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "kaggle"))


class FakeClock:
    def __init__(self, start):
        self.t = start

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, 0.001)

    def strftime(self, fmt, t=None):
        return dt.datetime.fromtimestamp(self.t if t is None else t).strftime(fmt)

    def localtime(self, t=None):
        return t


class FakeKaggle:
    """Just enough of the Kaggle CLI: datasets (versioned snapshots), the training kernel, the prep kernel."""

    def __init__(self, clock, scenario, tmp: Path):
        self.clock, self.scenario, self.tmp = clock, list(scenario), tmp
        self.datasets: dict[str, Path] = {}
        self.kernel = None  # {"status", "until", "outputs": Path, "error": bool}
        self.pushes, self.gpu_minutes_by_week, self.flaky = [], {}, 0
        self.n = 0

    def week(self, t):
        d = dt.datetime.fromtimestamp(t, dt.timezone.utc)
        return (d - dt.timedelta(days=(d.weekday() - 5) % 7)).date().isoformat()

    def __call__(self, args, check=True, tries=4):
        if self.flaky:  # transient network failure surfacing through the driver
            self.flaky -= 1
            raise RuntimeError("kaggle: connection reset")
        cmd = args[:2]
        if cmd == ["datasets", "list"]:
            return "\n".join(f"{r},x" for r in self.datasets)
        if cmd[0] == "datasets" and cmd[1] in ("version", "create"):
            path = Path(args[args.index("-p") + 1])
            ref = json.loads((path / "dataset-metadata.json").read_text())["id"]
            snap = self.tmp / f"ds{self.n}"
            self.n += 1
            shutil.copytree(path, snap)
            self.datasets[ref] = snap
            return "ok"
        if cmd == ["datasets", "status"]:
            return "ready"
        if cmd == ["kernels", "status"]:
            if "prep" in args[2]:
                return 'status "KernelWorkerStatus.COMPLETE"'
            k = self.kernel
            if k is None:
                return 'status "KernelWorkerStatus.COMPLETE"'
            if self.clock.time() >= k["until"]:
                k["status"] = "error" if k["error"] else "complete"
            return f'status "KernelWorkerStatus.{k["status"].upper()}"'
        if cmd == ["kernels", "push"]:
            d = Path(args[args.index("-p") + 1])
            if (d / "prep.py").exists():
                return "pushed"
            return self.push_train(d)
        if cmd == ["kernels", "output"]:
            out = Path(args[args.index("-p") + 1])
            k = self.kernel
            if k.get("flaky_output"):
                k["flaky_output"] = False
                return "network error"
            for f in k["outputs"].iterdir():
                shutil.copy(f, out / f.name)
            return "ok"
        raise AssertionError(f"unexpected kaggle call {args}")

    def push_train(self, d: Path):
        src = (d / "session.py").read_text()
        expect = int(re.search(r"EXPECT_SESSION = (\d+)", src).group(1))
        minutes = int(re.search(r"SESSION_MINUTES = (\d+)", src).group(1))
        safe = "SAFE_MODE = True" in src
        now = self.clock.time()
        wk = self.week(now)
        used = self.gpu_minutes_by_week.get(wk, 0)
        if used >= 30 * 60:
            return "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."
        kind = self.scenario.pop(0) if self.scenario else "ok"
        if kind == "refuse_quota":
            return "Kernel push error: Maximum weekly GPU quota of 30.00 hours reached."
        ck = self.datasets["t/autolearn-ckpt"]
        state = json.loads((ck / "state.json").read_text())
        assert state["session"] == expect, "kernel would train on a different checkpoint than intended"
        # the quota check Kaggle applies while the session runs: a session that runs past it is killed
        run_min = minutes if kind not in ("stale", "fast_fail") else 3
        assert used + run_min <= 30 * 60, f"driver planned {minutes} min with only {30 * 60 - used} min of quota left"
        self.gpu_minutes_by_week[wk] = used + run_min
        self.pushes.append({"kind": kind, "session": expect, "minutes": minutes, "safe": safe, "t": now})
        out = self.tmp / f"out{self.n}"
        self.n += 1
        out.mkdir()
        log = []
        new = dict(state)
        if kind in ("ok", "error_with_progress"):
            if state["stage"] == "pretrain":
                new.update(session=state["session"] + 1, step=min(state.get("step", 0) + 1000, 3000), val_loss=3.0)
                if new["step"] >= 3000:
                    new["stage"] = "sft"
                (out / "last.pt").write_bytes(b"ckpt")
                (out / "best.pt").write_bytes(b"best")
            else:
                new.update(session=state["session"] + 1, stage="done")
                (out / "sft_best.pt").write_bytes(b"sft")
            (out / "state.json").write_text(json.dumps(new))
            log.append("SESSION DONE " + json.dumps(new))
        elif kind == "stale":
            log.append("STALE_CHECKPOINT {}")
        elif kind == "missing":
            log.append("MISSING_CHECKPOINT expected last.pt")
        else:  # no_progress / fast_fail
            log.append("Traceback ... RuntimeError: CUDA error")
        (out / "autolearn-train.log").write_text(json.dumps([{"data": x} for x in log]))
        self.kernel = {"status": "running", "until": now + run_min * 60, "outputs": out,
                       "error": kind in ("error_with_progress", "no_progress", "fast_fail"),
                       "flaky_output": kind == "error_with_progress"}
        return "Kernel version 1 successfully pushed"


def run_driver(tmp_path, monkeypatch, scenario, flaky=0, start="2026-11-23T12:00:00+00:00"):
    import campaign
    clock = FakeClock(dt.datetime.fromisoformat(start).timestamp())
    fake = FakeKaggle(clock, scenario, tmp_path / "kaggle")
    (tmp_path / "kaggle").mkdir()
    fake.flaky = flaky
    ftime = types.SimpleNamespace(time=clock.time, sleep=clock.sleep, strftime=clock.strftime,
                                  localtime=clock.localtime)
    monkeypatch.setattr(campaign, "time", ftime)
    monkeypatch.setattr(campaign, "sh", fake)
    monkeypatch.setattr(campaign, "fineweb_files", lambda subset: [f"f{i}.parquet" for i in range(8)])
    published = []
    import autolearn.llm.engine as engine
    monkeypatch.setattr(engine.LLMRegistry, "publish", lambda self, ck, tok, meta: published.append(ck) or "v9")
    tok = tmp_path / "tokenizer.json"
    tok.write_text("{}")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    # 3000 steps (3 pretrain sessions of 1000 fake steps) -> target = 3000 * tokens_per_step
    monkeypatch.setattr(sys, "argv", ["campaign", "--user", "t", "--work", str(tmp_path / "work"),
                                      "--tokenizer", str(tok), "--llm-dir", str(tmp_path / "llm"),
                                      "--target-tokens", str(3000 * 4 * 1024 * 16 * 2)])
    campaign.main()
    return fake, published, json.loads((tmp_path / "work" / "ckpt" / "state.json").read_text())


def test_campaign_happy_path(tmp_path, monkeypatch):
    fake, published, state = run_driver(tmp_path, monkeypatch, [])
    assert state["stage"] == "done" and published
    assert [p["session"] for p in fake.pushes] == [0, 1, 2, 3]
    assert fake.pushes[-1]["minutes"] == 150  # SFT session length


def test_campaign_survives_every_failure_kind(tmp_path, monkeypatch):
    scenario = ["ok", "no_progress", "fast_fail", "stale", "missing", "error_with_progress", "refuse_quota", "ok"]
    fake, published, state = run_driver(tmp_path, monkeypatch, scenario, flaky=3)
    assert state["stage"] == "done" and published
    kinds = [p["kind"] for p in fake.pushes]
    assert kinds[:6] == ["ok", "no_progress", "fast_fail", "stale", "missing", "error_with_progress"]
    # two failures in a row -> the next session runs in safe mode; progress resets the counter
    assert [p["safe"] for p in fake.pushes][:4] == [False, False, False, True]
    assert fake.pushes[-1]["safe"] is False
    # the session that ended in an error but made progress was kept (no session number repeated after it)
    i = kinds.index("error_with_progress")
    assert fake.pushes[i + 1]["session"] == fake.pushes[i]["session"] + 1


def test_campaign_respects_weekly_quota(tmp_path, monkeypatch):
    # 12 pretrain sessions of 255 min can't fit in one week: the driver must stop short of the quota and resume
    # after the Saturday reset, never pushing a session that would be cut off
    import campaign
    monkeypatch.setattr(campaign, "QUOTA_MARGIN_MIN", 20)
    scenario = ["no_progress"] * 9 + ["ok"] * 5
    fake, published, state = run_driver(tmp_path, monkeypatch, scenario)
    assert state["stage"] == "done"
    weeks = {fake.week(p["t"]) for p in fake.pushes}
    assert len(weeks) >= 2, "expected the campaign to continue in the following quota week"
    assert all(v <= 30 * 60 for v in fake.gpu_minutes_by_week.values())
