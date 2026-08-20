"""Generates additional training examples for a spec via a teacher LLM.

Pure collection, no validation: this repo trains models, it doesn't grade
generated code or plans -- that's the qprompt agent graph's job
(syntaxvalidator/cypressExec and friends run for real, at inference time,
in qprompt-langgraph/qprompt-a2a), not something to duplicate here.
Whatever the teacher produces is written straight to the seed dataset;
manual review is expected before training on it.

Runs locally, not on Colab -- it needs a running teacher model (Docker
Model Runner by default). Output is merged + deduped into the spec's
existing seed JSONL; regenerate the notebook afterward to pick up the
larger dataset.

Two `input_generation.mode` values in the spec JSON are supported, and
neither is specific to any one domain -- both are driven entirely by the
spec, same as everything else in this library:
  - "topics": the teacher invents the io_a value itself, grounded in a
    `topics` list the spec declares (pairs of (topic, a concern that
    value should cover), so generations lean on real domain practice
    rather than generic phrasing) via a `request_system_prompt` the spec
    also declares.
  - "from_spec": io_a values are drawn from another spec's already
    generated seed dataset (e.g. codegen's plans come from the test-plan
    spec's output) -- run that spec's generation first.

Usage:
    pip install -r datagen/requirements.txt
    python3 datagen/generate_dataset.py specs/qwen3-testplan.json --count 20
    python3 datagen/generate_dataset.py specs/qwen3-cypress-codegen.json --count 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from teacher import Teacher

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, examples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def dedupe(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        key = json.dumps(ex, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def generate_from_topics(spec: dict, teacher: Teacher, count: int, on_example) -> list[dict]:
    io_a, io_b = spec["io_keys"]
    request_system_prompt = spec["input_generation"]["request_system_prompt"]
    topics = list(spec["input_generation"]["topics"])
    random.shuffle(topics)
    examples: list[dict] = []
    for i in range(count):
        topic, note = topics[i % len(topics)]
        generated_input = teacher.complete(
            system=request_system_prompt,
            user=f"Topic: {topic}. Make sure it implies: {note}.",
            max_tokens=150,
        )
        generated_output = teacher.complete(system=spec["system_prompt"], user=generated_input, max_tokens=500)
        example = {io_a: generated_input, io_b: generated_output}
        examples.append(example)
        on_example(example)
        print(f"  [{len(examples)}/{count}] {topic!r}")
    return examples


def generate_from_other_spec(spec: dict, teacher: Teacher, count: int, source_inputs: list[str], on_example) -> list[dict]:
    io_a, io_b = spec["io_keys"]
    inputs = list(source_inputs)
    random.shuffle(inputs)
    examples: list[dict] = []
    for i, generated_input in enumerate(inputs[:count]):
        generated_output = teacher.complete(system=spec["system_prompt"], user=generated_input, max_tokens=900)
        example = {io_a: generated_input, io_b: generated_output}
        examples.append(example)
        on_example(example)
        print(f"  [{len(examples)}/{min(count, len(inputs))}] source #{i}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", help="path to a spec JSON file (specs/*.json)")
    parser.add_argument("--count", type=int, default=20, help="number of new examples to add")
    parser.add_argument(
        "--teacher-base-url",
        default="http://localhost:12434/engines/llama.cpp/v1",
        help="local Docker Model Runner endpoint (default: its direct host engine endpoint). "
        "No API keys/hosted endpoints -- this only ever talks to a model on the local machine.",
    )
    parser.add_argument(
        "--teacher-model",
        default="ai/qwen3:8B-Q4_K_M",
        help="a locally pulled teacher model tag -- deliberately bigger than the model being "
        "trained; distilling a model from itself teaches it nothing new",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="don't append Qwen3's /no_think suffix (leave thinking mode on; irrelevant for non-Qwen3 teachers)",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    seed_path = REPO_ROOT / spec["seed_path"]
    existing = load_jsonl(seed_path)

    teacher = Teacher(base_url=args.teacher_base_url, model=args.teacher_model, no_think=not args.think)

    # Confirmed against a real long run: a teacher request can fail
    # permanently (retries in Teacher.complete exhausted) partway through a
    # large batch. Writing only at the very end meant a crash at, say,
    # example 34/100 discarded all 34 -- write to disk after every single
    # example instead, so a mid-batch failure loses at most the one
    # in-flight example, not the whole batch.
    collected: list[dict] = []

    def on_example(example: dict) -> None:
        collected.append(example)
        write_jsonl(seed_path, dedupe(existing + collected))

    mode = spec.get("input_generation", {}).get("mode", "topics")
    try:
        if mode == "topics":
            generate_from_topics(spec, teacher, args.count, on_example)
        elif mode == "from_spec":
            source_spec_path = REPO_ROOT / "specs" / spec["input_generation"]["spec"]
            source_spec = json.loads(source_spec_path.read_text())
            source_examples = load_jsonl(REPO_ROOT / source_spec["seed_path"])
            source_field = spec["input_generation"]["field"]
            source_inputs = [ex[source_field] for ex in source_examples]
            if not source_inputs:
                print(f"error: {source_spec_path} has no examples yet -- generate it first", file=sys.stderr)
                sys.exit(1)
            generate_from_other_spec(spec, teacher, args.count, source_inputs, on_example)
        else:
            print(f"error: unknown input_generation.mode {mode!r}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        merged = dedupe(existing + collected)
        print(
            f"\nstopped early after {len(collected)}/{args.count} examples ({exc.__class__.__name__}: {exc}) -- "
            f"already-generated examples were saved as they were produced. "
            f"{seed_path.relative_to(REPO_ROOT)} now has {len(merged)} total. Re-run to continue.",
            file=sys.stderr,
        )
        sys.exit(1)

    merged = dedupe(existing + collected)
    print(
        f"{len(collected)} new examples added (unvalidated -- review before training) -- "
        f"{seed_path.relative_to(REPO_ROOT)} now has {len(merged)} total "
        f"(regenerate the notebook: python3 tools/generate_notebook.py {spec_path})"
    )


if __name__ == "__main__":
    main()
