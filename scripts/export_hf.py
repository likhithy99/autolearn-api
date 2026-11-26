"""Export a published AutoLearn model to the standard Hugging Face Llama format (+ ONNX for the browser).

AutoLearn's architecture (RMSNorm, RoPE, grouped-query attention, SwiGLU, tied embeddings, no biases) is the Llama
architecture, so the conversion is lossless: the script checks that both models produce the same logits.

    .venv-export/bin/python scripts/export_hf.py --version v2 --out exports/autolearn-40m-chat [--onnx]

Output: config.json, model.safetensors, tokenizer.json (+ chat template), generation_config.json, and with --onnx
an onnx/ folder (fp32 + int8 quantized, with KV cache) in the layout transformers.js expects.
Runs in a separate venv (.venv-export) that has transformers/optimum; it does not import torch-heavy repo code
beyond autolearn.llm.model.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from autolearn.llm.model import GPT, GPTConfig  # noqa: E402
from autolearn.llm.tokenizer import DEFAULT_SYSTEM  # noqa: E402

# Same layout as Tok.render_chat: <|bos|> then <|role|> content <|end|> per message, default system prompt.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] != 'system' %}{{ '<|system|>" + DEFAULT_SYSTEM + "<|end|>' }}{% endif %}"
    "{% for m in messages %}{{ '<|' + m['role'] + '|>' + (m['content'] | trim) + '<|end|>' }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}"
)


def convert(src: Path, out: Path):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    ck = torch.load(src / "model.pt", map_location="cpu", weights_only=False)
    c = GPTConfig(**ck["config"])
    sd = ck["model"]
    hd = c.n_embd // c.n_head
    ffn = c.ffn_hidden or 64 * ((int(2 * 4 * c.n_embd / 3) + 63) // 64)
    tok_src = src / "tokenizer.json"
    special = {t["content"]: t["id"] for t in json.loads(tok_src.read_text())["added_tokens"]}
    cfg = LlamaConfig(vocab_size=c.vocab_size, hidden_size=c.n_embd, intermediate_size=ffn,
                      num_hidden_layers=c.n_layer, num_attention_heads=c.n_head, num_key_value_heads=c.n_kv_head,
                      max_position_embeddings=c.block_size, rms_norm_eps=c.norm_eps, rope_theta=c.rope_theta,
                      tie_word_embeddings=c.tie_embeddings, attention_bias=c.qkv_bias, mlp_bias=False,
                      hidden_act="silu", bos_token_id=special["<|bos|>"], eos_token_id=special["<|end|>"],
                      pad_token_id=special["<|pad|>"], torch_dtype="float32")
    m = {"model.embed_tokens.weight": sd["tok_emb.weight"], "model.norm.weight": sd["norm.weight"]}
    if not c.tie_embeddings:
        m["lm_head.weight"] = sd["lm_head.weight"]
    kv = c.n_kv_head * hd
    for i in range(c.n_layer):
        p, q = f"blocks.{i}.", f"model.layers.{i}."
        wkv = sd[p + "attn.wkv.weight"]
        m.update({q + "self_attn.q_proj.weight": sd[p + "attn.wq.weight"],
                  q + "self_attn.k_proj.weight": wkv[:kv], q + "self_attn.v_proj.weight": wkv[kv:],
                  q + "self_attn.o_proj.weight": sd[p + "attn.wo.weight"],
                  q + "mlp.gate_proj.weight": sd[p + "mlp.w1.weight"], q + "mlp.up_proj.weight": sd[p + "mlp.w3.weight"],
                  q + "mlp.down_proj.weight": sd[p + "mlp.w2.weight"],
                  q + "input_layernorm.weight": sd[p + "ln1.weight"],
                  q + "post_attention_layernorm.weight": sd[p + "ln2.weight"]})
        if c.qkv_bias:
            bkv = sd[p + "attn.wkv.bias"]
            m.update({q + "self_attn.q_proj.bias": sd[p + "attn.wq.bias"],
                      q + "self_attn.k_proj.bias": bkv[:kv], q + "self_attn.v_proj.bias": bkv[kv:]})
    hf = LlamaForCausalLM(cfg)
    missing, unexpected = hf.load_state_dict({k: v.contiguous() for k, v in m.items()}, strict=False)
    missing = [k for k in missing if not (k == "lm_head.weight" and c.tie_embeddings)]
    assert not missing and not unexpected, (missing, unexpected)
    hf.tie_weights()
    hf.eval()

    # the converted model must compute exactly what ours does
    ours = GPT(c)
    ours.load_state_dict(sd)
    ours.eval()
    ids = torch.randint(0, c.vocab_size, (2, 64))
    with torch.no_grad():
        x = ours.tok_emb(ids)
        for blk in ours.blocks:
            x = blk(x, ours.rope_cos[:64], ours.rope_sin[:64])
        a = ours.lm_head(ours.norm(x))
        b = hf(ids).logits
    err = (a - b).abs().max().item()
    assert err < 1e-3, f"converted model differs (max logit diff {err})"
    print(f"converted: max logit difference {err:.2e}")

    out.mkdir(parents=True, exist_ok=True)
    hf.generation_config.update(do_sample=True, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1,
                                max_new_tokens=256, eos_token_id=[special["<|end|>"], special["<|eos|>"]])
    hf.save_pretrained(out, safe_serialization=True)
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_src), bos_token="<|bos|>", eos_token="<|end|>",
                                  pad_token="<|pad|>", chat_template=CHAT_TEMPLATE, model_max_length=c.block_size,
                                  clean_up_tokenization_spaces=False)
    tok.save_pretrained(out)
    meta = json.loads((src / "meta.json").read_text()) if (src / "meta.json").exists() else {}
    (out / "autolearn_meta.json").write_text(json.dumps({**meta, "source_version": src.name}, indent=2, default=str))

    # chat template must reproduce the training-time prompt exactly
    sys.path.insert(0, str(ROOT))
    from autolearn.llm.tokenizer import Tok
    msgs = [{"role": "user", "content": "Hello there"}, {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "What is 2+2?"}]
    want, _ = Tok(str(tok_src)).render_chat(msgs)
    got = tok.apply_chat_template(msgs, add_generation_prompt=True)
    assert want == got, f"chat template mismatch:\n{want}\n{got}"
    print("chat template matches training format")
    return out


def export_onnx(out: Path):
    from optimum.exporters.onnx import main_export
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_dir = out / "onnx"
    tmp = out / "_onnx_tmp"
    main_export(str(out), output=str(tmp), task="text-generation-with-past", opset=17, device="cpu",
                no_post_process=False)
    onnx_dir.mkdir(exist_ok=True)
    shutil.move(str(tmp / "model.onnx"), onnx_dir / "model.onnx")
    if (tmp / "model.onnx_data").exists():
        shutil.move(str(tmp / "model.onnx_data"), onnx_dir / "model.onnx_data")
    shutil.rmtree(tmp)
    quantize_dynamic(str(onnx_dir / "model.onnx"), str(onnx_dir / "model_quantized.onnx"),
                     weight_type=QuantType.QInt8)
    for f in sorted(onnx_dir.iterdir()):
        print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="registry version, e.g. v2")
    ap.add_argument("--llm-dir", default=str(ROOT / "models/llm"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", action="store_true")
    a = ap.parse_args()
    out = convert(Path(a.llm_dir) / "versions" / a.version, Path(a.out))
    if a.onnx:
        export_onnx(out)


if __name__ == "__main__":
    main()
