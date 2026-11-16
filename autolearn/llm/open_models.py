"""Run open-weight chat models (Qwen2/2.5 and Llama-family checkpoints) on AutoLearn's own transformer.

These architectures are the same as ours (RMSNorm, RoPE, grouped-query attention, SwiGLU) up to a few options,
so the published safetensors weights are mapped straight onto `autolearn.llm.model.GPT`. No `transformers`
dependency, and the same KV-cached engine, streaming, metering and moderation serve every model.

    # download + register a model (served as `autolearn-pro` by the API)
    python -m autolearn.llm.open_models add Qwen/Qwen2.5-1.5B-Instruct --name autolearn-pro
    python -m autolearn.llm.open_models list
    python -m autolearn.llm.open_models chat autolearn-pro "What is the capital of France?"

Only use models whose licence allows commercial use (e.g. Qwen2.5 0.5B/1.5B/7B/14B/32B: Apache-2.0;
SmolLM2: Apache-2.0). The licence file is kept next to the weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .model import GPT, GPTConfig, rope_tables

SUPPORTED = {"Qwen2ForCausalLM", "LlamaForCausalLM", "MistralForCausalLM"}
MANIFEST = "autolearn.json"


# ---------------------------------------------------------------- tokenizer --
class ChatTokenizer:
    """Wraps a Hugging Face tokenizer.json with the model's chat format (ChatML or Llama-3 headers).
    Exposes the same interface the engine uses for our own tokenizer (encode/decode/render_chat/bos/eos/end)."""

    def __init__(self, model_dir: Path, fmt: str, system_prompt: str | None):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.fmt, self.system_prompt = fmt, system_prompt
        tid = self.t.token_to_id
        if fmt == "chatml":
            self.start, self.end = tid("<|im_start|>"), tid("<|im_end|>")
            self.eos = tid("<|endoftext|>")
        elif fmt == "llama3":
            self.start, self.end = tid("<|start_header_id|>"), tid("<|eot_id|>")
            self.eos = tid("<|end_of_text|>")
        else:
            raise ValueError(f"unknown chat format {fmt}")
        self.bos = tid("<|begin_of_text|>") if fmt == "llama3" else None
        # strings that must never be produced from user text (prompt-injection of control tokens)
        self.controls = [s for s in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|start_header_id|>",
                                     "<|end_header_id|>", "<|eot_id|>", "<|begin_of_text|>", "<|end_of_text|>")
                         if tid(s) is not None]

    @property
    def vocab_size(self) -> int:
        return self.t.get_vocab_size()

    def _clean(self, text: str) -> str:
        for c in self.controls:
            text = text.replace(c, c[:1] + "​" + c[1:])
        return text

    def encode(self, text: str) -> list[int]:
        return self.t.encode(self._clean(text), add_special_tokens=False).ids

    def decode(self, ids: list[int]) -> str:
        return self.t.decode(ids, skip_special_tokens=True)

    def render_chat(self, messages: list[dict], add_generation_prompt: bool = True, system: str | None = None):
        system = system or self.system_prompt
        if system and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": system}] + list(messages)
        ids = [self.bos] if self.bos is not None else []
        mask = [0] * len(ids)
        for m in messages:
            content = m["content"]
            if isinstance(content, list):  # OpenAI content parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            head, body, tail = self._turn(m["role"], content or "")
            train = int(m["role"] == "assistant")  # loss only on assistant replies (+ their end token)
            ids += head + body + [self.end] + tail
            mask += [0] * len(head) + [train] * (len(body) + 1) + [0] * len(tail)
        if add_generation_prompt:
            h = self._header("assistant")
            ids += h
            mask += [0] * len(h)
        return ids, mask

    def _header(self, role: str) -> list[int]:
        if self.fmt == "chatml":
            return [self.start] + self.t.encode(f"{role}\n", add_special_tokens=False).ids
        return ([self.start] + self.t.encode(role, add_special_tokens=False).ids
                + [self.t.token_to_id("<|end_header_id|>")] + self.t.encode("\n\n", add_special_tokens=False).ids)

    def _turn(self, role: str, content: str) -> list[int]:
        body = self.encode(content.strip() if self.fmt == "llama3" else content)
        tail = self.t.encode("\n", add_special_tokens=False).ids if self.fmt == "chatml" else []
        return self._header(role), body, tail


# -------------------------------------------------------------------- model --
def hf_config_to_gpt(cfg: dict, context: int) -> GPTConfig:
    arch = (cfg.get("architectures") or ["?"])[0]
    if arch not in SUPPORTED:
        raise ValueError(f"unsupported architecture {arch}; supported: {sorted(SUPPORTED)}")
    if cfg.get("rope_scaling"):
        raise ValueError("models with rope_scaling (e.g. Llama 3.1+) are not supported yet")
    hd = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    if hd * cfg["num_attention_heads"] != cfg["hidden_size"]:
        raise ValueError("head_dim * heads != hidden_size is not supported")
    return GPTConfig(
        vocab_size=cfg["vocab_size"], block_size=min(context, cfg.get("max_position_embeddings", context)),
        n_layer=cfg["num_hidden_layers"], n_head=cfg["num_attention_heads"],
        n_kv_head=cfg.get("num_key_value_heads", cfg["num_attention_heads"]), n_embd=cfg["hidden_size"],
        rope_theta=float(cfg.get("rope_theta", 10000.0)), qkv_bias=arch == "Qwen2ForCausalLM",
        ffn_hidden=cfg["intermediate_size"], norm_eps=float(cfg.get("rms_norm_eps", 1e-6)),
        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)))


def convert_state_dict(hf: dict, c: GPTConfig) -> dict:
    """Hugging Face parameter names -> ours (k_proj and v_proj are fused into wkv)."""
    sd = {"tok_emb.weight": hf["model.embed_tokens.weight"], "norm.weight": hf["model.norm.weight"]}
    sd["lm_head.weight"] = hf["model.embed_tokens.weight"] if c.tie_embeddings else hf["lm_head.weight"]
    for i in range(c.n_layer):
        p, q = f"model.layers.{i}.", f"blocks.{i}."
        a = p + "self_attn."
        sd[q + "ln1.weight"] = hf[p + "input_layernorm.weight"]
        sd[q + "ln2.weight"] = hf[p + "post_attention_layernorm.weight"]
        sd[q + "attn.wq.weight"] = hf[a + "q_proj.weight"]
        sd[q + "attn.wkv.weight"] = torch.cat([hf[a + "k_proj.weight"], hf[a + "v_proj.weight"]], 0)
        sd[q + "attn.wo.weight"] = hf[a + "o_proj.weight"]
        if c.qkv_bias:
            sd[q + "attn.wq.bias"] = hf[a + "q_proj.bias"]
            sd[q + "attn.wkv.bias"] = torch.cat([hf[a + "k_proj.bias"], hf[a + "v_proj.bias"]], 0)
        sd[q + "mlp.w1.weight"] = hf[p + "mlp.gate_proj.weight"]
        sd[q + "mlp.w3.weight"] = hf[p + "mlp.up_proj.weight"]
        sd[q + "mlp.w2.weight"] = hf[p + "mlp.down_proj.weight"]
    return sd


def load_open_model(model_dir: str | Path, device: str, dtype: torch.dtype | None = None):
    from safetensors.torch import load_file
    model_dir = Path(model_dir)
    man = read_manifest(model_dir)
    hf_cfg = json.loads((model_dir / "config.json").read_text())
    c = hf_config_to_gpt(hf_cfg, man.get("context", 8192))
    if dtype is None:
        dtype = torch.float32 if device == "cpu" else (torch.bfloat16 if device != "cuda" or
                                                       torch.cuda.is_bf16_supported() else torch.float16)
    hf = {}
    for f in sorted(model_dir.glob("*.safetensors")):
        hf.update(load_file(str(f)))
    sd = {k: v.to(dtype) for k, v in convert_state_dict(hf, c).items()}
    del hf
    with torch.device("meta"):
        model = GPT(c)  # no allocation / random init for billions of parameters
    model.load_state_dict(sd, strict=True, assign=True)
    if c.tie_embeddings:
        model.lm_head.weight = model.tok_emb.weight
    cos, sin = rope_tables(c.n_embd // c.n_head, c.block_size, c.rope_theta)
    model.rope_cos, model.rope_sin = cos, sin
    model.to(device).eval()
    tok = ChatTokenizer(model_dir, man.get("chat_format", "chatml"), man.get("system_prompt"))
    return model, c, tok


# ----------------------------------------------------------------- catalog --
def read_manifest(model_dir: Path) -> dict:
    f = Path(model_dir) / MANIFEST
    return json.loads(f.read_text()) if f.exists() else {}


def catalog(root: str | Path | None) -> dict[str, dict]:
    """{model_id: manifest} for every registered open model under root."""
    out = {}
    if root and Path(root).exists():
        for d in sorted(Path(root).iterdir()):
            m = read_manifest(d)
            if m and m.get("type", "chat") == "chat" and (d / "config.json").exists() and any(d.glob("*.safetensors")):
                out[m["id"]] = {**m, "path": str(d)}
    return out


def detect_format(model_dir: Path) -> str:
    vocab = json.loads((model_dir / "tokenizer.json").read_text()).get("added_tokens", [])
    names = {t["content"] for t in vocab}
    if "<|im_start|>" in names:
        return "chatml"
    if "<|start_header_id|>" in names:
        return "llama3"
    raise ValueError("could not detect chat format (expected ChatML or Llama-3 special tokens)")


def add(a):
    os.environ.setdefault("SSL_CERT_FILE", __import__("certifi").where())
    from huggingface_hub import snapshot_download
    d = Path(a.root) / a.name
    snapshot_download(a.repo, local_dir=str(d),
                      allow_patterns=["*.json", "*.safetensors", "LICENSE*", "NOTICE*", "README.md"])
    cfg = json.loads((d / "config.json").read_text())
    hf_config_to_gpt(cfg, a.context)  # fail early if unsupported
    man = {"id": a.name, "repo": a.repo, "chat_format": detect_format(d), "context": a.context,
           "system_prompt": a.system_prompt, "description": a.description}
    (d / MANIFEST).write_text(json.dumps(man, indent=2))
    print(json.dumps(man, indent=2))


DEFAULT_SYSTEM = "You are a helpful, honest and harmless AI assistant."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="models/open")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("add")
    s.add_argument("repo")
    s.add_argument("--name", required=True, help="model id customers will use, e.g. autolearn-pro")
    s.add_argument("--context", type=int, default=8192)
    s.add_argument("--system-prompt", default=DEFAULT_SYSTEM)
    s.add_argument("--description", default="")
    sub.add_parser("list")
    c = sub.add_parser("chat")
    c.add_argument("name")
    c.add_argument("prompt")
    c.add_argument("--device", default="auto")
    c.add_argument("--max-tokens", type=int, default=200)
    a = ap.parse_args()
    if a.cmd == "add":
        add(a)
    elif a.cmd == "list":
        print(json.dumps(catalog(a.root), indent=2))
    else:
        from .engine import Engine
        from .train import pick_device
        e = Engine.open_model(catalog(a.root)[a.name]["path"], str(pick_device(a.device)))
        for ev in e.chat_stream([{"role": "user", "content": a.prompt}], max_tokens=a.max_tokens,
                                temperature=0.7, seed=0):
            print(ev.get("delta", ""), end="", flush=True)
        print()


if __name__ == "__main__":
    main()
