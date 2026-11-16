"""Constrained decoding: guarantees the model's output is syntactically valid JSON (OpenAI `response_format`).

At each step the sampler only considers tokens whose text keeps the output a valid JSON prefix, checked by a
small character-level pushdown automaton. Once the top-level object closes, generation stops.
"""

from __future__ import annotations

import copy
import re

WS = " \t\n\r"
MAX_WS_RUN = 32  # stop the model from padding with endless whitespace


class JsonPrefix:
    """Incremental validator: feed(ch) returns False as soon as the text can no longer be valid JSON."""

    def __init__(self, top_level_object: bool = True):
        self.stack: list[str] = []      # "{" / "["
        self.state = "value"            # value | obj_key_or_end | obj_key | colon | after_value | string | number | literal | done
        self.top_object = top_level_object
        self.string_is_key = False
        self.escape = 0                 # 0 none, 1 after "\", 2-5 inside \\uXXXX
        self.literal = ""
        self.number = ""
        self.ws = 0
        self.started = False

    def copy(self) -> "JsonPrefix":
        return _clone(self)  # the stack list must never be shared with the copy

    @property
    def done(self) -> bool:
        return self.state == "done"

    def feed_text(self, text: str) -> bool:
        return all(self.feed(c) for c in text)

    def feed(self, c: str) -> bool:  # noqa: C901 (a state machine is clearest as one function)
        st = self.state
        if st == "string":
            if self.escape == 1:
                if c == "u":
                    self.escape = 2
                elif c in '"\\/bfnrt':
                    self.escape = 0
                else:
                    return False
                return True
            if self.escape >= 2:
                if c not in "0123456789abcdefABCDEF":
                    return False
                self.escape = 0 if self.escape == 5 else self.escape + 1
                return True
            if c == "\\":
                self.escape = 1
                return True
            if c == '"':
                self.state = "colon" if self.string_is_key else "after_value"
                self.string_is_key = False
                self.ws = 0
                return True
            return c >= " " or c == "\t"  # raw control characters are invalid inside strings
        if st == "number":
            if c in "0123456789+-.eE" and self._number_ok(self.number + c):
                self.number += c
                return True
            if not self._number_complete(self.number):
                return False
            self.state = "after_value"
            return self.feed(c)
        if st == "literal":
            if self.literal and c == self.literal[0]:
                self.literal = self.literal[1:]
                if not self.literal:
                    self.state = "after_value"
                return True
            return False
        if c in WS:
            if st == "done" or (not self.started and self.top_object):
                return False  # the object starts immediately and nothing follows it
            return self._ws()
        self.ws = 0
        if st == "done":
            return False
        if st in ("value",):
            if not self.started and self.top_object and c != "{":
                return False
            self.started = True
            if c == "{":
                self.stack.append("{")
                self.state = "obj_key_or_end"
            elif c == "[":
                self.stack.append("[")
                self.state = "value_or_end"
            elif c == '"':
                self.state, self.string_is_key = "string", False
            elif c in "-0123456789":
                self.state, self.number = "number", c
            elif c in "tfn":
                self.state, self.literal = "literal", {"t": "rue", "f": "alse", "n": "ull"}[c]
            else:
                return False
            return True
        if st == "value_or_end":  # just after "["
            if c == "]":
                self.stack.pop()
                self._closed()
                return True
            self.state = "value"
            return self.feed(c)
        if st in ("obj_key_or_end", "obj_key"):
            if c == "}" and st == "obj_key_or_end":
                self.stack.pop()
                self._closed()
                return True
            if c == '"':
                self.state, self.string_is_key = "string", True
                return True
            return False
        if st == "colon":
            if c == ":":
                self.state = "value"
                return True
            return False
        if st == "after_value":
            top = self.stack[-1] if self.stack else None
            if c == "," and top == "{":
                self.state = "obj_key"
                return True
            if c == "," and top == "[":
                self.state = "value"
                return True
            if (c == "}" and top == "{") or (c == "]" and top == "["):
                self.stack.pop()
                self._closed()
                return True
            return False
        return False

    def _ws(self) -> bool:
        self.ws += 1
        return self.ws <= MAX_WS_RUN

    def _closed(self):
        self.state = "after_value" if self.stack else "done"

    @staticmethod
    def _number_ok(s: str) -> bool:
        """Is s a prefix of a valid JSON number?"""
        return bool(_NUM_PREFIX.fullmatch(s))

    @staticmethod
    def _number_complete(s: str) -> bool:
        return bool(_NUM_FULL.fullmatch(s))


_NUM_INT = r"-?(?:0|[1-9]\d*)"
_NUM_PREFIX = re.compile(rf"-?|{_NUM_INT}(?:\.\d*)?|{_NUM_INT}(?:\.\d+)?[eE][+-]?\d*")
_NUM_FULL = re.compile(rf"{_NUM_INT}(?:\.\d+)?(?:[eE][+-]?\d+)?")


def _clone(p: JsonPrefix) -> JsonPrefix:
    q = copy.copy(p)
    q.stack = list(p.stack)
    return q


class JsonConstraint:
    """Plugs JsonPrefix into the sampler: filters candidate tokens and reports when the object is complete."""

    def __init__(self, tok, top_level_object: bool = True):
        self.tok = tok
        self.state = JsonPrefix(top_level_object)
        self._text: dict[int, str] = {}
        self.special = {i for i in (getattr(tok, "end", None), getattr(tok, "eos", None), getattr(tok, "bos", None))
                        if i is not None}

    def text(self, tid: int) -> str:
        t = self._text.get(tid)
        if t is None:
            t = self._text[tid] = self.tok.decode([tid])
        return t

    def allows(self, tid: int) -> bool:
        if tid in self.special:
            return False
        t = self.text(tid)
        if not t:
            return False
        if "�" in t:  # partial UTF-8 byte of a multi-byte character: only valid inside a string
            return self.state.state == "string" and self.state.escape == 0
        return self.state.copy().feed_text(t)

    def advance(self, tid: int):
        t = self.text(tid)
        if "�" not in t:
            self.state.feed_text(t)

    @property
    def done(self) -> bool:
        return self.state.done
