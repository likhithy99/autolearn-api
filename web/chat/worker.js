// Runs the model off the main thread. Messages in:  {type:"load", model}, {type:"generate", messages, params},
// {type:"stop"}. Messages out: {type:"progress"|"ready"|"token"|"done"|"error", ...}.
import {
  pipeline, env, TextStreamer, InterruptableStoppingCriteria,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

env.allowLocalModels = false;
const LOCAL = new URL(self.location.href).searchParams.get("local");  // dev: serve models from this path
if (LOCAL) { env.allowLocalModels = true; env.allowRemoteModels = false; env.localModelPath = LOCAL; }

let gen = null, loadedId = null;
const stopper = new InterruptableStoppingCriteria();

async function load(model) {
  if (loadedId === model.id && gen) return gen;
  if (gen) { try { await gen.dispose(); } catch (_) {} gen = null; }
  const files = {};
  gen = await pipeline("text-generation", model.id, {
    dtype: "q8", device: "wasm",
    progress_callback: (p) => {
      if (p.status === "progress" && p.total) {
        files[p.file] = { loaded: p.loaded, total: p.total };
        const all = Object.values(files);
        self.postMessage({ type: "progress",
          loaded: all.reduce((s, f) => s + f.loaded, 0), total: all.reduce((s, f) => s + f.total, 0) });
      }
    },
  });
  loadedId = model.id;
  return gen;
}

// Keep the newest turns that fit the model's context window, leaving room for the reply.
function fit(tok, messages, ctx, maxNew) {
  let msgs = messages.slice();
  const len = (m) => tok.apply_chat_template(m, { add_generation_prompt: true, tokenize: true, return_tensor: false }).length;
  while (msgs.length > 1 && len(msgs) > ctx - maxNew) msgs = msgs.slice(msgs[0].role === "user" && msgs.length > 2 ? 2 : 1);
  return msgs;
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "stop") { stopper.interrupt(); return; }
    if (data.type === "load") { await load(data.model); self.postMessage({ type: "ready", model: data.model.id }); return; }
    if (data.type === "generate") {
      const g = await load(data.model);
      const p = data.params;
      const ctx = g.model.config.max_position_embeddings || 512;
      const maxNew = Math.min(p.max_new_tokens, Math.floor(ctx / 2));
      const msgs = fit(g.tokenizer, data.messages, ctx, maxNew);
      stopper.reset();
      const t0 = performance.now(); let n = 0;
      const streamer = new TextStreamer(g.tokenizer, {
        skip_prompt: true, skip_special_tokens: true,
        callback_function: (text) => { n++; self.postMessage({ type: "token", text }); },
      });
      await g(msgs, {
        max_new_tokens: maxNew, do_sample: p.temperature > 0, temperature: p.temperature || 1,
        top_p: 0.9, top_k: 50, repetition_penalty: 1.1, streamer, stopping_criteria: stopper,
      });
      self.postMessage({ type: "done", seconds: (performance.now() - t0) / 1000, chunks: n, trimmed: msgs.length < data.messages.length });
    }
  } catch (e) {
    self.postMessage({ type: "error", message: String(e && e.message || e) });
  }
};
