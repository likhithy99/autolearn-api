"""Continuous batching must give exactly the same greedy output as sequential generation."""

import threading

from autolearn.llm.engine import Engine
from autolearn.llm.lora import get_adapter, lora_modules
from tests.test_open_models import qwen_dir  # noqa: F401  (fixture)

PROMPTS = ["the cat sat", "hello world, paris", "mat", "the cat sat on the mat. hello", "is in france"]


def run_all(engine, jobs):
    """Run (prompt, max_tokens, adapter, stop) jobs concurrently; return their texts in order."""
    out = [None] * len(jobs)

    def go(i, p, n, ad, stop):
        ids, _ = engine.tok.render_chat([{"role": "user", "content": p}])
        out[i] = "".join(ev.get("delta", "") for ev in engine.stream(ids, max_tokens=n, temperature=0,
                                                                     adapter=ad, stop=stop))
    ts = [threading.Thread(target=go, args=(i, *j)) for i, j in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    return out


def test_batched_equals_sequential(qwen_dir):  # noqa: F811
    seq = Engine.open_model(qwen_dir, "cpu")
    jobs = [(p, n, None, None) for p, n in zip(PROMPTS, [12, 5, 20, 1, 9])]
    want = run_all(seq, [(p, n, a, s) for p, n, a, s in jobs[:1]]) + run_all(seq, jobs[1:2]) + \
        run_all(seq, jobs[2:3]) + run_all(seq, jobs[3:4]) + run_all(seq, jobs[4:5])  # one at a time
    bat = Engine.open_model(qwen_dir, "cpu").enable_batching(max_batch=3, max_len=256)
    peak, step = [0], bat.batcher._decode_step

    def spy():
        peak[0] = max(peak[0], len(bat.batcher.active))
        step()
    bat.batcher._decode_step = spy
    got = run_all(bat, jobs)  # 5 concurrent requests through 3 slots: slots are reused as requests finish
    assert got == want
    assert peak[0] >= 2  # sequences really were decoded together


def test_batched_adapters_are_isolated(qwen_dir):  # noqa: F811
    import torch
    e = Engine.open_model(qwen_dir, "cpu")
    g = torch.Generator().manual_seed(1)
    for m in lora_modules(e.model):  # a non-trivial adapter
        m.B.data = torch.randn(m.B.shape, generator=g) * 0.5
    custom = get_adapter(e.model)
    jobs = [("the cat sat", 10, None, None), ("the cat sat", 10, custom, None),
            ("hello world", 8, custom, None), ("hello world", 8, None, None)]
    want = [run_all(e, [j])[0] for j in jobs]
    assert want[0] != want[1]  # the adapter really changes the output
    b = Engine.open_model(qwen_dir, "cpu").enable_batching(max_batch=4, max_len=256)
    assert run_all(b, jobs) == want


def test_batched_stop_and_cancel(qwen_dir):  # noqa: F811
    e = Engine.open_model(qwen_dir, "cpu")
    full = run_all(e, [("the cat sat", 30, None, None)])[0]
    stop = full[5:8]
    want = run_all(e, [("the cat sat", 30, None, [stop])])[0]
    b = Engine.open_model(qwen_dir, "cpu").enable_batching(max_batch=2, max_len=256)
    assert run_all(b, [("the cat sat", 30, None, [stop]), ("mat", 30, None, None)])[0] == want
    # after a stop-string cancel the slot is freed: further requests still complete
    assert run_all(b, [("hello", 4, None, None)] * 3)[0] is not None
    assert not b.batcher.active
