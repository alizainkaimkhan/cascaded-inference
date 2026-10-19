"""Frontier model wrapper with a write-once SQLite cache.

Every call is keyed on (model, item_id, sha256(prompt)). A cached key is never re-sent, so
the full pass costs one call per item ever, escalation is simulated offline by lookup, and
results survive model deprecation. Changing the prompt text changes the hash, so stale
answers can never be silently reused for a new prompt.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

LETTER_EXACT = re.compile(r"^\s*(?:answer\s*[:\-]?\s*)?[\(\[]?([ABCD])[\)\]\.:]?\s*$", re.IGNORECASE)
LETTER_LOOSE = re.compile(r"(?:answer\s*(?:is)?\s*[:\-]?\s*)?[\(\[]?\b([ABCD])\b[\)\]\.:]?")

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    model TEXT NOT NULL,
    item_id TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    prompt_version TEXT,
    response_text TEXT,
    parsed_letter TEXT,
    parse_status TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    raw_json TEXT,
    created_at TEXT,
    PRIMARY KEY (model, item_id, prompt_sha256)
)
"""


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def parse_letter(text: str) -> tuple[str | None, str]:
    """('B', 'exact') for a bare letter answer, ('B', 'loose') if found inside other text,
    (None, 'fail') otherwise. Loose parses are counted and inspected in the pilot."""
    if text is None:
        return None, "fail"
    text = text.replace("**", "").replace("__", "")  # markdown emphasis, e.g. "**A**"
    m = LETTER_EXACT.match(text)
    if m:
        return m.group(1).upper(), "exact"
    m = LETTER_LOOSE.search(text)
    if m:
        return m.group(1).upper(), "loose"
    return None, "fail"


@dataclass
class Completion:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    raw: dict


class AnthropicBackend:
    def __init__(self, model: str, max_tokens: int, temperature: float | None):
        import anthropic

        self.client, self.model, self.max_tokens, self.temperature = anthropic.Anthropic(), model, max_tokens, temperature

    def complete(self, prompt: str) -> Completion:
        kw = {"model": self.model, "max_tokens": self.max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = self.client.messages.create(**kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
        return Completion(text, r.usage.input_tokens, r.usage.output_tokens, r.model_dump())


class OpenAIBackend:
    def __init__(self, model: str, max_tokens: int, temperature: float | None):
        import openai

        self.client, self.model, self.max_tokens, self.temperature = openai.OpenAI(), model, max_tokens, temperature

    def complete(self, prompt: str) -> Completion:
        kw = {"model": self.model, "max_completion_tokens": self.max_tokens,
              "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            kw["temperature"] = self.temperature
        r = self.client.chat.completions.create(**kw)
        return Completion(r.choices[0].message.content or "", r.usage.prompt_tokens,
                          r.usage.completion_tokens, r.model_dump())


BACKENDS = {"anthropic": AnthropicBackend, "openai": OpenAIBackend}


class FrontierCache:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def get(self, model: str, item_id: str, prompt: str) -> dict | None:
        cur = self.conn.execute(
            "SELECT * FROM responses WHERE model=? AND item_id=? AND prompt_sha256=?",
            (model, item_id, sha256(prompt)),
        )
        row = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], row)) if row else None

    def put(self, model: str, item_id: str, prompt: str, prompt_version: str, c: Completion) -> None:
        letter, status = parse_letter(c.text)
        self.conn.execute(
            "INSERT OR IGNORE INTO responses VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (model, item_id, sha256(prompt), prompt_version, c.text, letter, status, c.input_tokens,
             c.output_tokens, json.dumps(c.raw, default=str),
             dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")),
        )
        self.conn.commit()


def _with_retries(fn, prompt: str, tries: int = 6):
    for attempt in range(tries):
        try:
            return fn(prompt)
        except Exception as e:  # rate limits, 5xx, timeouts
            if attempt == tries - 1:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  retry {attempt + 1} after {type(e).__name__}: {e}; sleeping {wait}s", flush=True)
            time.sleep(wait)


def run_uncached(backend, cache: FrontierCache, model: str, prompt_version: str,
                 jobs: list[tuple[str, str]], workers: int = 4) -> int:
    """jobs = [(item_id, prompt)]. Sends only uncached ones; returns how many were sent."""
    todo = [(i, p) for i, p in jobs if cache.get(model, i, p) is None]
    print(f"{len(jobs) - len(todo)} cached, {len(todo)} to send", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_with_retries, backend.complete, p): (i, p) for i, p in todo}
        for f in as_completed(futs):
            i, p = futs[f]
            cache.put(model, i, p, prompt_version, f.result())  # writes stay on this thread
            done += 1
            if done % 50 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
    return len(todo)
