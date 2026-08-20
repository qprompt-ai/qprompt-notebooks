"""Minimal chat client for the teacher model used by generate_dataset.py.

Plain HTTP (`requests`) against Docker Model Runner's local
`/chat/completions` endpoint -- not the `openai` package, and no API key
of any kind: this only ever talks to a model running on the local
machine. There is deliberately no support for a hosted/external API
here -- a different teacher means pulling a different local Docker
Model Runner tag (`docker model pull ...`), not pointing this at a
cloud endpoint.
"""
from __future__ import annotations

import re
import time

import requests

_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\n(.*?)\n?```$", re.DOTALL)
_CODE_START_RE = re.compile(r"^\s*(///|/\*|//|import\s|describe\(|it\()", re.MULTILINE)


def strip_markdown_fence(text: str) -> str:
    match = _CODE_FENCE_RE.match(text.strip())
    return match.group(1) if match else text


def strip_leading_prose(text: str) -> str:
    match = _CODE_START_RE.search(text)
    if not match:
        return text
    result = text[match.start():]
    lines = result.rstrip().split("\n")
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines)


class Teacher:
    """Talks to a local Docker Model Runner tag over plain HTTP. Defaults
    to its direct host engine endpoint (`docker model status --json` ->
    `endpointHost`) and a bigger local tag than whatever's being trained --
    distilling a model from itself teaches it nothing new, the whole point
    is a stronger model generating examples for a weaker one. `--teacher-
    base-url`/`--teacher-model` can point at a different local port or a
    different pulled tag; there's no API key anywhere in this path, by
    design -- see the module docstring.
    """

    def __init__(self, *, base_url: str, model: str, no_think: bool = True):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.no_think = no_think

    def complete(self, *, system: str, user: str, max_tokens: int = 900, max_retries: int = 3) -> str:
        # "/no_think" is Qwen3's hybrid-thinking-mode toggle. Without it, a
        # low max_tokens budget can get spent entirely on <think> reasoning
        # with an empty final `content`. No-op on non-Qwen3 models.
        suffix = " /no_think" if self.no_think else ""
        # Confirmed against a real long run: Docker Model Runner's backend
        # can drop an in-flight connection mid-batch (RemoteDisconnected)
        # without the daemon itself going down -- `docker model status`
        # showed it running again immediately after. Transient and
        # recoverable, not a reason to abandon the whole batch; retry with
        # backoff before giving up on this one call.
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Content-Type": "application/json"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user + suffix},
                        ],
                        "max_tokens": max_tokens,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
                return strip_leading_prose(strip_markdown_fence(content)).strip()
            except requests.exceptions.RequestException as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s, ...
        raise last_error
