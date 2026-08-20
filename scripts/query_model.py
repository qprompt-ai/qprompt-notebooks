"""Sends a single prompt to a local model with the right system prompt
attached -- the fair way to manually test one of this library's models,
since `docker model run <tag> "<prompt>"` has no way to attach a system
prompt at all (confirmed: `docker model run --help` has no --system/-s
flag), and every model here is trained under, and meant to be served
under, a specific one (see specs/*.json's system_prompt field).

Usage:
    python3 scripts/query_model.py \\
        ghcr.io/qprompt-ai/qwen3-testplan:0.6B-Q4_K_M \\
        "Verify that a user can log in with valid credentials."

    # public/foreign model, or to force a specific spec/system prompt:
    python3 scripts/query_model.py ai/qwen3:0.6B-Q4_K_M "..." --system "You are a helpful assistant."
    python3 scripts/query_model.py <tag> "..." --spec specs/qwen3-testplan.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same two helpers as datagen/teacher.py (small intentional duplication,
# not a cross-directory import -- this script and datagen/ are each
# meant to be runnable standalone) mirroring what qprompt_a2a's real
# call_llm applies to every response before using it.
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


def _repo_name(model_tag: str) -> str:
    """"yassineouhadi/qwen3-testplan:0.6B-Q4_K_M" -> "qwen3-testplan"."""
    return model_tag.rsplit("/", 1)[-1].split(":", 1)[0]

def _autodetect_spec(model_tag: str) -> Path | None:
    repo_name = _repo_name(model_tag)
    matches = [p for p in sorted((REPO_ROOT / "specs").glob("*.json")) if json.loads(p.read_text()).get("docker_repo_name") == repo_name]
    if len(matches) > 1:
        names = ", ".join(str(p.relative_to(REPO_ROOT)) for p in matches)
        print(f"error: model tag {model_tag!r} matches multiple specs ({names}) -- pass --spec to disambiguate", file=sys.stderr)
        sys.exit(1)
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_tag", help="the model tag to query, e.g. ghcr.io/qprompt-ai/qwen3-testplan:0.6B-Q4_K_M")
    parser.add_argument("prompt", help="the user-turn prompt")
    parser.add_argument(
        "--spec",
        help="path to a spec JSON file (specs/*.json) -- supplies the system prompt. "
        "Auto-detected from the model tag's repo name if omitted.",
    )
    parser.add_argument(
        "--system",
        help="use this literal text as the system prompt instead of a spec's -- overrides --spec/auto-detection",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:12434/engines/llama.cpp/v1",
        help="local Docker Model Runner endpoint (default: its direct host engine endpoint)",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="don't append Qwen3's /no_think suffix (leave thinking mode on; irrelevant for non-Qwen3 models)",
    )
    parser.add_argument("--max-tokens", type=int, default=500)
    args = parser.parse_args()

    if args.system is not None:
        system_prompt = args.system
    else:
        spec_path = Path(args.spec) if args.spec else _autodetect_spec(args.model_tag)
        if spec_path is None:
            print(
                f"note: no spec in specs/*.json matches model tag {args.model_tag!r} "
                f"(repo name {_repo_name(args.model_tag)!r}) -- querying with no system prompt.",
                file=sys.stderr,
            )
            system_prompt = None
        else:
            if not args.spec:
                print(f"auto-detected spec: {spec_path.relative_to(REPO_ROOT)}", file=sys.stderr)
            system_prompt = json.loads(spec_path.read_text())["system_prompt"]

    user_content = args.prompt if args.think else f"{args.prompt} /no_think"
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": args.model_tag,
            "messages": messages,
            "max_tokens": args.max_tokens,
        },
        timeout=180,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"] or ""
    cleaned = strip_leading_prose(strip_markdown_fence(raw)).strip()

    print("=== raw ===")
    print(raw)
    if cleaned != raw.strip():
        print("\n=== after call_llm's post-processing (fence/leading-prose stripped) ===")
        print(cleaned)


if __name__ == "__main__":
    main()
