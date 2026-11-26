---
license: apache-2.0
language:
- en
library_name: transformers
pipeline_tag: text-generation
tags:
- llama
- from-scratch
- small-language-model
- transformers.js
- onnx
datasets:
- HuggingFaceFW/fineweb-edu
- HuggingFaceTB/smoltalk
---

# AutoLearn 40M Chat

A 40-million-parameter chat model **trained from scratch** on a single MacBook (Apple M3, 16 GB): tokenizer,
pretraining and chat fine-tuning, with no borrowed weights. Part of [AutoLearn API](https://github.com/likhithy99/autolearn-api),
an OpenAI-compatible AI platform built end to end.

**Try it in your browser:** [AutoLearn Chat](https://huggingface.co/spaces/Likhith003/autolearn-chat) (runs on your device, no server).

## Model

| | |
|---|---|
| Architecture | Llama-style decoder: RMSNorm, rotary position embeddings, grouped-query attention, SwiGLU, tied embeddings |
| Size | 40.4M parameters: 8 layers, hidden 512, 8 attention heads / 4 key-value heads, FFN 1408 |
| Tokenizer | Byte-level BPE, 32,768 tokens, trained on the pretraining corpus |
| Context | 512 tokens |

## Training

| Stage | Data | Compute | Result |
|---|---|---|---|
| Pretraining | 2.9B tokens of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (180,000 steps × 16,384 tokens) | Apple M3 GPU (MPS), ~9 days | validation loss 3.55 |
| Chat fine-tuning | 100,000 conversations from [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), loss on assistant turns only | Apple M3, 2.8 hours | validation loss 1.48 |

Training code: [`autolearn/llm/train.py`](https://github.com/likhithy99/autolearn-api/blob/main/autolearn/llm/train.py)
(the model itself is [`autolearn/llm/model.py`](https://github.com/likhithy99/autolearn-api/blob/main/autolearn/llm/model.py)).
These weights were converted losslessly to the Hugging Face Llama format with
[`scripts/export_hf.py`](https://github.com/likhithy99/autolearn-api/blob/main/scripts/export_hf.py).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Likhith003/autolearn-40m-chat"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo)
ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France?"}],
                              add_generation_prompt=True, return_tensors="pt")
out = model.generate(ids, max_new_tokens=100)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))  # The capital of France is Paris.
```

In the browser with [transformers.js](https://huggingface.co/docs/transformers.js) (int8 ONNX, 58 MB):

```js
import { pipeline } from "@huggingface/transformers";
const chat = await pipeline("text-generation", "Likhith003/autolearn-40m-chat", { dtype: "q8" });
const out = await chat([{ role: "user", content: "Write a haiku about the sea." }], { max_new_tokens: 80 });
```

## Limitations

This is a small research model. It writes fluent English but often states things that are false, loses track of
longer conversations, and has had no safety tuning beyond what is in SmolTalk. Don't use it for anything that
matters.

## License

Weights: Apache 2.0. Training data: FineWeb-Edu (ODC-By 1.0) and SmolTalk (Apache 2.0).
