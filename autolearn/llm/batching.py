"""Continuous batching: many concurrent requests share each forward pass.

Decoding a small model is memory-bandwidth bound (every step reads all the weights), so advancing B
sequences in one pass costs about as much as advancing one. A scheduler thread owns the model:
  * new requests are prefilled one at a time (their prompt KV is copied into a free batch slot),
  * then one decode step advances every active sequence together, each at its own position,
  * finished sequences free their slot immediately; waiting requests join at the next step.

KV caches live in preallocated per-layer buffers [max_batch, kv_heads, max_len, head_dim]; attention is masked
per row to that sequence's length. Requests are batched only with others using the same custom-model adapter.
With greedy decoding the output is identical to sequential generation (tested).
"""

from __future__ import annotations

import logging
import queue
import threading
import time

import torch
import torch.nn.functional as F

from .model import apply_rope

log = logging.getLogger("autolearn.llm.batching")
_END = object()


class Request:
    def __init__(self, prompt, max_new, sampler, adapter, constraint):
        self.prompt, self.max_new, self.sampler = prompt, max_new, sampler
        self.adapter, self.constraint = adapter, constraint
        self.out: queue.Queue = queue.Queue()
        self.slot = None
        self.pos = 0          # number of tokens in this slot's KV cache
        self.generated = 0
        self.cancelled = False


class BatchScheduler:
    def __init__(self, engine, max_batch: int = 8, max_len: int = 2048):
        self.engine, self.model = engine, engine.model
        c = engine.config
        self.max_batch, self.max_len = max_batch, min(max_len, c.block_size)
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        dtype = self.model.tok_emb.weight.dtype
        dev = engine.device
        self.k = [torch.zeros(max_batch, self.nkv, self.max_len, self.hd, dtype=dtype, device=dev)
                  for _ in range(c.n_layer)]
        self.v = [torch.zeros_like(t) for t in self.k]
        self.waiting: queue.Queue = queue.Queue()
        self.active: dict[int, Request] = {}
        self.adapter = 0  # adapter shared by the active batch (0 = none chosen yet)
        self._held: list[Request] = []  # waiting requests for a different adapter
        self.thread = threading.Thread(target=self._loop, name="llm-batcher", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------- public --
    def fits(self, prompt_len: int, max_new: int) -> bool:
        return prompt_len + max_new <= self.max_len

    def submit(self, prompt, max_new, sampler, adapter=None, constraint=None) -> Request:
        r = Request(prompt, max_new, sampler, adapter, constraint)
        self.waiting.put(r)
        return r

    @staticmethod
    def tokens(r: Request):
        """Iterate the token ids generated for a request (blocks until each is ready)."""
        try:
            while True:
                t = r.out.get()
                if t is _END:
                    return
                if isinstance(t, BaseException):
                    raise t
                yield t
        finally:
            r.cancelled = True  # consumer went away (e.g. client disconnected): free the slot

    # ---------------------------------------------------------- scheduler --
    def _loop(self):
        while True:
            try:
                self._admit(block=not self.active)
                if self.active:
                    with torch.no_grad(), self.engine.lock:
                        self._ensure_adapter()
                        self._decode_step()
            except Exception as e:  # never let the scheduler thread die
                log.exception("batch step failed")
                for r in list(self.active.values()):
                    r.out.put(e)
                    self._release(r)

    def _admit(self, block: bool):
        incoming = list(self._held)
        self._held = []
        try:
            if block and not incoming:
                incoming.append(self.waiting.get())
            while True:
                incoming.append(self.waiting.get_nowait())
        except queue.Empty:
            pass
        for r in incoming:
            if r.cancelled:
                continue
            free = [s for s in range(self.max_batch) if s not in self.active]
            if not self.active:
                self.adapter = r.adapter
            if not free or r.adapter is not self.adapter:
                self._held.append(r)
                continue
            with torch.no_grad(), self.engine.lock:
                self._ensure_adapter()
                self._prefill(r, free[0])
        if self._held and not self.active:  # batch drained: next adapter group goes on the next pass
            time.sleep(0)

    def _ensure_adapter(self):
        e = self.engine
        if e.supports_adapters and getattr(e, "_loaded_adapter", 0) is not self.adapter:
            from .lora import set_adapter
            set_adapter(self.model, self.adapter)
            e._loaded_adapter = self.adapter

    def _prefill(self, r: Request, slot: int):
        caches = [{} for _ in range(self.engine.config.n_layer)]
        idx = torch.tensor([r.prompt], device=self.engine.device)
        logits, _ = self.model(idx, caches=caches, start_pos=0)
        L = len(r.prompt)
        for i, c in enumerate(caches):
            self.k[i][slot, :, :L] = c["k"][0]
            self.v[i][slot, :, :L] = c["v"][0]
        r.slot, r.pos = slot, L
        self.active[slot] = r
        self._emit(r, logits[0, -1])

    def _decode_step(self):
        """One token for every active sequence. KV buffers are read in place (idle slots below the highest active
        one compute throwaway values); grouped-query heads are folded into the query dimension so K/V are never
        expanded or copied."""
        # slots fill lowest-first, so rows [0, highest active slot] cover the batch; read KV buffers as views
        B, dev, m = max(self.active) + 1, self.engine.device, self.model
        tok_l, pos_l = [0] * B, [0] * B  # built on the host, sent in one transfer each
        for s, r in self.active.items():
            tok_l[s], pos_l[s] = r.last, r.pos
        T = max(pos_l) + 1
        tok = torch.tensor(tok_l, device=dev)[:, None]
        pos = torch.tensor(pos_l, device=dev)
        rows = torch.arange(B, device=dev)
        cos, sin = m.rope_cos[pos][:, None, None, :], m.rope_sin[pos][:, None, None, :]  # (B,1,1,hd/2)
        mask = (torch.arange(T, device=dev)[None, :] <= pos[:, None])[:, None, None, :]  # (B,1,1,T)
        rep = self.nh // self.nkv
        x = m.tok_emb(tok)
        for i, blk in enumerate(m.blocks):
            a = blk.attn
            h = blk.ln1(x)
            q = a.wq(h).view(B, 1, self.nh, self.hd).transpose(1, 2)
            k, v = a.wkv(h).view(B, 1, 2, self.nkv, self.hd).unbind(2)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            self.k[i][rows, :, pos] = k[:, :, 0]
            self.v[i][rows, :, pos] = v[:, :, 0]
            qg = q.reshape(B, self.nkv, rep, self.hd)  # head h = kv * rep + r uses kv head h // rep
            y = F.scaled_dot_product_attention(qg, self.k[i][:B, :, :T], self.v[i][:B, :, :T], attn_mask=mask)
            x = x + a.wo(y.reshape(B, 1, self.nh * self.hd))
            x = x + blk.mlp(blk.ln2(x))
        logits = m.lm_head(m.norm(x))[:, -1].float().cpu()  # one device->host copy for the whole batch
        for s, r in list(self.active.items()):
            r.pos += 1
            self._emit(r, logits[s])

    def _emit(self, r: Request, logits):
        """Pick the next token for r (or finish it)."""
        if r.cancelled:
            return self._release(r)
        nxt = r.sampler(logits, r.constraint)
        if nxt is None:
            return self._release(r)
        r.out.put(nxt)
        r.last = nxt
        r.generated += 1
        if r.generated >= r.max_new or r.pos + 1 >= self.max_len:
            self._release(r)

    def _release(self, r: Request):
        if r.slot is not None and self.active.get(r.slot) is r:
            del self.active[r.slot]
        r.out.put(_END)
