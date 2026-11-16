#!/usr/bin/env bash
# Full from-scratch LLM pipeline: data -> tokenizer -> pretrain -> chat fine-tune -> publish.
# Defaults fit a 16 GB Apple-silicon laptop. On GPUs raise TOKENS / PRESET / MINUTES, and set
# NPROC to the number of GPUs (uses torchrun + DDP).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
TOKENS=${TOKENS:-300M}          # pretraining tokens to stream (2 bytes each on disk)
PRESET=${PRESET:-mini}          # nano | mini | micro | small | medium | large | xl
MINUTES=${MINUTES:-90}          # pretraining wall-clock budget
SFT_MINUTES=${SFT_MINUTES:-30}
BATCH=${BATCH:-16}
ACCUM=${ACCUM:-2}
NPROC=${NPROC:-1}
LLM_DIR=${LLM_DIR:-models/llm}

run() { if [ "$NPROC" -gt 1 ]; then torchrun --standalone --nproc_per_node="$NPROC" -m "$@"; else $PY -m "$@"; fi; }

[ -f data/llm/pretrain/manifest.json ] || $PY -m autolearn.llm.data pretrain --out data/llm/pretrain --tokens "$TOKENS"
[ -f data/llm/sft/manifest.json ] || $PY -m autolearn.llm.data sft --out data/llm/sft \
    --tokenizer data/llm/pretrain/tokenizer.json --max-conversations 100000 --max-len 768

run autolearn.llm.train --stage pretrain --preset "$PRESET" --data data/llm/pretrain --out runs/pretrain \
    --batch "$BATCH" --accum "$ACCUM" --lr 1e-3 --time-budget-min "$MINUTES"
run autolearn.llm.train --stage sft --init runs/pretrain/best.pt --data data/llm/sft --out runs/sft \
    --batch "$BATCH" --accum "$ACCUM" --lr 3e-4 --time-budget-min "$SFT_MINUTES" --eval-every 100

$PY - <<PYEOF
from autolearn.llm.engine import LLMRegistry
v = LLMRegistry("$LLM_DIR").publish("runs/sft/best.pt", "data/llm/pretrain/tokenizer.json",
                                    {"stage": "sft", "preset": "$PRESET", "pretrain_tokens": "$TOKENS"})
print("published", v)
PYEOF
