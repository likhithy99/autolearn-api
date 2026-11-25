#!/usr/bin/env bash
# Runs after the long Mac pretraining run (runs/long_micro) finishes:
#   1. waits for it to finish (survives pauses; does nothing while it is still training)
#   2. chat fine-tunes it (SFT) on the local SmolTalk shards, resuming from its own checkpoint after any crash
#   3. publishes the chat model to models/llm, made current only if it beats the current model's chat val loss
#   4. writes sample answers to logs/after_long_run_samples.txt
# Safe to re-run: every step is skipped once its output exists.
#     nohup bash scripts/after_long_run.sh >> logs/after_long_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
PRE=runs/long_micro
SFT=runs/long_micro_sft
TOK=data/llm/pretrain/tokenizer.json
SFT_STEPS=${SFT_STEPS:-2100}   # ~1 epoch of the 34M-token SFT set at 32 x 512 tokens per step
log() { echo "$(date '+%F %T') after_long_run: $*"; }

PIDF="$PRE/.after_long_run.pid"   # one copy at a time (macOS has no flock(1))
if [ -f "$PIDF" ] && [ "$(cat "$PIDF")" != "$$" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  log "already running (pid $(cat "$PIDF"))"; exit 0
fi
echo $$ > "$PIDF"
caffeinate -ims -w $$ &

done_flag() { $PY -c "import json,sys;sys.exit(0 if json.load(open('$1/progress.json')).get('done') else 1)" 2>/dev/null; }

# 1. wait for pretraining
log "waiting for $PRE to finish"
while ! done_flag "$PRE" || pgrep -f "autolearn.llm.train.*--out $PRE( |$)" >/dev/null; do sleep 300; done
log "pretraining finished: $(cat $PRE/progress.json)"
# its restart loop and watchdog exit on their own once the run is done; make sure they cannot restart it
pkill -f scripts/local_long_run.sh 2>/dev/null; pkill -f scripts/mac_watchdog.py 2>/dev/null

# 2. chat fine-tune, retrying from its checkpoint (up to 6 attempts, backing off)
for attempt in 1 2 3 4 5 6; do
  done_flag "$SFT" && break
  RESUME=""; [ -f "$SFT/last.pt" ] && RESUME="--resume"
  log "SFT attempt $attempt $RESUME"
  $PY -m autolearn.llm.train --stage sft --init "$PRE/best.pt" --data data/llm/sft --out "$SFT" \
      --tokenizer "$TOK" --batch 8 --accum 4 --lr 2e-4 --warmup 100 --max-steps "$SFT_STEPS" \
      --eval-every 250 --eval-iters 40 $RESUME >> logs/after_long_run_sft.log 2>&1
  done_flag "$SFT" && break
  log "SFT exited before finishing; retrying in $((attempt * 60))s"
  sleep $((attempt * 60))
done
if ! done_flag "$SFT"; then log "SFT did not finish after 6 attempts; giving up (see logs/after_long_run_sft.log)"; exit 1; fi

# 3 + 4. publish and sample
$PY - <<'PYEOF'
import json, time
from pathlib import Path
import torch
from autolearn.llm.engine import LLMRegistry, Engine

reg = LLMRegistry("models/llm")
sft_best = Path("runs/long_micro_sft/best.pt")
new_val = torch.load(sft_best, map_location="cpu", weights_only=False).get("val_loss")
already = [v for v in reg.versions() if v.get("source") == str(sft_best)]
if already:
    version = already[-1]["version"]
    print("already published as", version)
else:
    cur = reg.current()
    cur_meta = next((v for v in reg.versions() if v["version"] == cur), {})
    cur_val = cur_meta.get("val_loss") or float("inf")
    better = new_val is not None and new_val < cur_val
    meta = {"stage": "sft", "preset": "micro", "pretrain_steps": 180000, "pretrain_tokens": "2.9B",
            "pretrain_data": "FineWeb-Edu", "sft_data": "SmolTalk (100k conversations)",
            "trained_on": "Apple M3 (MPS)", "source": str(sft_best)}
    version = reg.publish(sft_best, "data/llm/pretrain/tokenizer.json", meta, make_current=better)
    print(f"published {version} (chat val loss {new_val:.4f} vs current {cur} {cur_val:.4f}); "
          f"{'now current' if better else 'kept ' + str(cur) + ' as current'}")

eng = Engine(reg.path(version))
prompts = ["What is photosynthesis?", "Give me three tips for studying for an exam.",
           "Explain what a neural network is to a 10-year-old.", "Write a short poem about the ocean.",
           "What is the capital of France?"]
with open("logs/after_long_run_samples.txt", "w") as f:
    f.write(f"model {version}  ({time.strftime('%F %T')})\n\n")
    for p in prompts:
        r = eng.chat([{"role": "user", "content": p}], max_tokens=160, temperature=0.7)
        f.write(f"Q: {p}\nA: {r['text'].strip()}\n\n")
print("samples written to logs/after_long_run_samples.txt")
PYEOF
log "all done"
