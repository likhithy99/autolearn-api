"""Byte-level BPE tokenizer trained from scratch on our corpus, plus the chat template."""

from __future__ import annotations

from typing import Iterable

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
ROLES = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}
DEFAULT_SYSTEM = "You are a helpful, honest and harmless AI assistant."


def train_tokenizer(texts: Iterable[str], vocab_size: int, out_path: str) -> "Tok":
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    tok.save(out_path)
    return Tok(out_path)


def _neutralise(text: str) -> str:
    """Stop user text from smuggling in control tokens such as '<|end|>' (prompt injection)."""
    return text.replace("<|", "<​|")


class Tok:
    def __init__(self, path: str):
        self.path = path
        self.t = Tokenizer.from_file(path)
        self.special = {s: self.t.token_to_id(s) for s in SPECIAL_TOKENS}
        self.bos, self.eos, self.end = self.special["<|bos|>"], self.special["<|eos|>"], self.special["<|end|>"]

    @property
    def vocab_size(self) -> int:
        return self.t.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.t.encode(_neutralise(text), add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.t.encode_batch([_neutralise(t) for t in texts], add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self.t.decode(ids, skip_special_tokens=True)

    def render_chat(self, messages: list[dict], add_generation_prompt: bool = True,
                    system: str | None = DEFAULT_SYSTEM) -> tuple[list[int], list[int]]:
        """Return (token ids, loss mask). Mask is 1 only on assistant reply tokens (incl. <|end|>)."""
        ids, mask = [self.bos], [0]
        if system and not any(m["role"] == "system" for m in messages):
            messages = [{"role": "system", "content": system}] + list(messages)
        for m in messages:
            role = m.get("role")
            if role not in ROLES:
                continue
            content = m.get("content") or ""
            if isinstance(content, list):  # OpenAI multi-part content: keep text parts
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            head = [self.special[ROLES[role]]]
            body = self.encode(content.strip()) + [self.end]
            ids += head + body
            mask += [0] * len(head) + ([1] * len(body) if role == "assistant" else [0] * len(body))
        if add_generation_prompt:
            ids.append(self.special["<|assistant|>"])
            mask.append(0)
        return ids, mask
