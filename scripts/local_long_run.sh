#!/usr/bin/env bash
# Multi-day from-scratch pretraining on this Mac, restarting from the last checkpoint after any crash.
# Stop it with:  pkill -f local_long_run.sh; pkill -f "autolearn.llm.train --stage pretrain"
set -uo pipefail
cd "$(dirname "$0")/.."
PRESET=${PRESET:-micro}
DATA=${DATA:-data/llm/pretrain_big,data/llm/pretrain_big2}
OUT=${OUT:-runs/long_${PRESET}}
STEPS=${STEPS:-180000}
mkdir -p "$OUT" logs
while true; do
  RESUME=""; [ -f "$OUT/last.pt" ] && RESUME="--resume"
  .venv/bin/python -m autolearn.llm.train --stage pretrain --preset "$PRESET" --data "$DATA" --out "$OUT" \
      --tokenizer data/llm/pretrain/tokenizer.json --batch 8 --accum 4 --lr 8e-4 --warmup 2000 \
      --max-steps "$STEPS" --eval-every 1000 --eval-iters 40 $RESUME >> logs/long_run.log 2>&1
  if grep -q '"done": true' logs/long_run.log; then echo "long run finished"; break; fi
  echo "$(date) trainer exited; restarting from checkpoint in 60s" >> logs/long_run.log
  sleep 60
done
