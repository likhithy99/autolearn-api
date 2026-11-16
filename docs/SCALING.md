# Scaling the LLM: from laptop to a commercial-grade model

The code in `autolearn/llm/` is the same at every scale. What changes is **data volume, model size and
compute**. This page is the honest map of what each step buys and costs.

## Why size matters

Model quality is driven by parameters and training tokens. A compute-optimal rule of thumb
(Chinchilla) is about **20 training tokens per parameter**. Training compute is about
`6 × params × tokens` FLOPs.

| Preset | Params | Tokens (≈20×) | Compute (FLOPs) | Typical hardware | Wall-clock | Rough cloud cost* | What it can do |
|---|---|---|---|---|---|---|---|
| `mini` | 22M | 0.4B | 5e16 | M-series Mac | ~1 day | $0 | Fluent-ish English fragments; proves the pipeline |
| `small` | 120M | 2.5B | 2e18 | 1× A100/H100 | ~3–6 h | $10–30 | GPT-2-small level text; weak chat |
| `medium` | 350M | 7B (better: 30B+) | 1.5e19+ | 8× H100 | ~0.5–1 day | $200–1,000 | Coherent paragraphs, simple Q&A |
| `large` | 1.3B | 26B (better: 100B+) | 2e20+ | 8–64× H100 | days | $2k–20k | Useful narrow assistants, after good SFT |
| `xl` | 7B | 140B (modern: 1–2T) | 6e21–1e23 | 64–512× H100 | weeks | $50k–1M+ | Llama-2-7B class general assistant |
| frontier | 100B+ | 10T+ | 1e25+ | 10k+ GPUs | months | $50M–$1B+ | GPT-4 / Claude class |

\*Order-of-magnitude, at roughly $2–4 per H100-hour. Real runs add 20–50% for failed or experimental runs.

Modern models train far past 20 tokens per parameter (for example 1–15T tokens for 7–8B models),
because a smaller, longer-trained model is cheaper to *serve*.

## Data at scale

* `--dataset HuggingFaceFW/fineweb-edu --subset sample-100BT` (or the full 1.3T-token `default`), and
  `HuggingFaceFW/fineweb` (15T tokens) for breadth.
* Add code (`bigcode/the-stack-v2`, check its license terms) and math (`HuggingFaceTB/finemath`) for
  reasoning ability.
* Disk: 2 bytes per token. 100B tokens = 200 GB. Use `--shard-tokens` and an object store or NVMe.
* Run several `data pretrain` jobs in parallel over dataset shards. Exact dedup is built in. At web
  scale, add MinHash near-dedup.
* **Licensing:** FineWeb/FineWeb-Edu are ODC-By (commercial use allowed with attribution).
  SmolTalk is Apache-2.0. Check every added dataset's license and your local law before selling a
  model trained on it.

## Compute

```bash
# 8 GPUs on one node
torchrun --standalone --nproc_per_node=8 -m autolearn.llm.train --stage pretrain --preset medium \
  --data data/llm/pretrain --out runs/pretrain --batch 32 --accum 4 --block 2048 --compile --lr 6e-4

# multiple nodes: torchrun --nnodes=N --node_rank=i --rdzv_endpoint=host:port ...
```

* bf16 autocast, TF32, fused AdamW and `torch.compile` turn on automatically on CUDA.
* Checkpoints are written every eval. `--resume` continues after preemption, so spot/preemptible GPUs work.
* Beyond ~7B parameters a model no longer fits in one GPU's memory: move to FSDP or tensor
  parallelism (swap the DDP wrapper in `train.py` for `torch.distributed.fsdp`).

## Serving at scale

* `Dockerfile.gpu` runs the API on a GPU. The built-in engine handles one request at a time per process.
* For high traffic, export the weights to a dedicated inference server with continuous batching and
  paged KV cache (vLLM, TensorRT-LLM, SGLang). Keep this API in front for auth, quotas, metering,
  feedback and continual learning.
* Run the continual-learning worker (`python -m autolearn.llm.continual`) on its own GPU.

## Before you sell an LLM API, also plan for

* **Safety:** a moderation layer (input/output classifiers), abuse monitoring, usage policies, and red-teaming.
* **Evaluation:** standard benchmarks (MMLU, HellaSwag, ARC, GSM8K, HumanEval) and your own task
  suites run on every version. Add them to the continual-learning promotion gate.
* **Alignment:** a larger SFT set, then preference optimization at scale. The DPO loss already
  exists in `continual.py`.
* **Legal:** terms of service, privacy policy, a data-processing agreement for enterprise customers,
  and data residency. Training on customer data is opt-in only, enforced in code.
