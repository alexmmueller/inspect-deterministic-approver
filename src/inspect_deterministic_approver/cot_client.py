"""
Client for querying a local open-weight reasoning model (via Ollama) and
capturing BOTH its raw chain-of-thought and a summarized version of the
same trace -- enabling the "does monitor-relevant signal survive
summarization" experiment described in cot_monitor.py's module docstring.

Ollama's /api/chat returns raw, unfiltered reasoning for models exposing
a `thinking` field (confirmed empirically against deepseek-r1:8b: the
response separates `message.content` -- the final answer -- from
`message.thinking` -- the full internal reasoning trace, no provider-side
condensing in between).
"""

from __future__ import annotations

import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
REASONING_MODEL = "deepseek-r1:1.5b"
SUMMARIZER_MODEL = "qwen2.5:14b-instruct"


def query_with_raw_cot(prompt: str, model: str = REASONING_MODEL, timeout: int = 900) -> tuple[str, str]:
    """Query a reasoning model and return (final_answer, raw_thinking)."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    message = result.get("message", {})
    return message.get("content", ""), message.get("thinking", "")


def summarize_cot(raw_thinking: str, model: str = SUMMARIZER_MODEL, timeout: int = 120) -> str:
    """Generate a condensed summary of a raw CoT trace, standing in for
    what a hosted provider's internal summarizer would produce -- this is
    the empirical stand-in for Anthropic's/OpenAI's summarization step,
    which we cannot observe directly since it happens inside their
    infrastructure. Deliberately uses a DIFFERENT, smaller-purpose model
    (a plain instruct model, not a reasoning model) to mirror Anthropic's
    documented approach of using "a smaller model to condense lengthy
    thought processes."
    """
    prompt = (
        "Summarize the following reasoning trace in 2-3 sentences, "
        "capturing the key decision points and any hesitations or "
        "alternative options considered:\n\n"
        f"{raw_thinking}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("message", {}).get("content", "")
