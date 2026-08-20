# qprompt-notebooks

A standalone model library: a repeatable pipeline for training,
fine-tuning, and publishing small models for arbitrary tasks — not tied
to any one domain or any one downstream project. Each model is a
**spec** (`specs/*.json`) + **seed data** (`data/*.jsonl`) that a generator
tool (`tools/generate_notebook.py`) turns into a self-contained Colab
fine-tuning notebook. Training produces an artifact pushed to a container
registry as a plain OCI model artifact, referenceable from anywhere that
accepts one — a Compose `models:` block, a bare `docker model run`,
whatever — the same way any predefined catalog model like `ai/qwen3:...`
is referenced, just under a chosen namespace. GHCR (`ghcr.io/qprompt/...`)
is the recommended registry, not Docker Hub — GitHub orgs are free and
GHCR hosts public packages at no cost, whereas a real Docker Hub
Organization needs a paid plan.

Each notebook targets a specific task and is published as a reusable model artifact:

| Notebook | Trained for | Registry repo |
|---|---|---|
| [`notebooks/cypress_codegen_finetune.ipynb`](notebooks/cypress_codegen_finetune.ipynb) | Turning a validated test plan into Cypress/TypeScript code | `qwen3-cypress-codegen` |
| [`notebooks/test_plan_finetune.ipynb`](notebooks/test_plan_finetune.ipynb) | Turning a feature request into a structured, code-free test plan | `qwen3-testplan` |

Both fine-tune `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` (Unsloth's pre-quantized
build of the same weights behind Docker's `ai/qwen3:0.6B-Q4_K_M` catalog
entry) with a LoRA adapter, via [Unsloth](https://unsloth.ai) — ~70% less
VRAM and ~2x faster than plain transformers+PEFT, runs comfortably on
Colab's free T4 tier. Neither notebook is hand-written — both are generated
output; edit the spec/seed data and regenerate instead of editing a
notebook directly.

## Running

The seed datasets (`data/*.jsonl`, same content as embedded in each
notebook) started as a handful of hand-written illustrative examples and
now also include teacher-generated ones (see "Growing the seed data"
below) — unvalidated, subject to manual review — still small relative to a
real training set (1,000+ examples is typical for a task-specific
fine-tune). Growing and reviewing the dataset further is expected before
training something production-ready.

## Pipeline

Each notebook, in order:

1. Loads the base model in 4-bit via `unsloth.FastLanguageModel` and
   attaches a LoRA adapter.
2. Fine-tunes with `trl.SFTTrainer` on the seed dataset, formatted with the
   spec's exact system prompt via the tokenizer's chat template, and saves
   the LoRA adapter (for reproducibility/resuming).
3. Evaluates: prints the eval-loss trend from training (on the held-out
   split `SFTConfig(eval_strategy="epoch")` already computed each epoch),
   then generates on each held-out example and prints input/expected/model
   output side by side. Neither is an automated pass/fail score — the
   output requires manual review to judge whether it's good enough to
   move on.
4. Merges the adapter into the base weights and quantizes straight to GGUF
   (`Q4_K_M` by default, set per-spec via `quant`) in one call —
   `save_pretrained_gguf`/`push_to_hub_gguf`, no separate llama.cpp build
   step. Docker Model Runner serves one set of weights, not a base model +
   adapter pair, so this merge is required.
5. Downloads the `.gguf` file, or pushes it to a Hugging
   Face Hub repo if `HF_HUB_REPO` is set in the config cell.

## Adding a new model to the library

A different domain, a different task, a different base model — gets added
the same way:

1. **Write a spec**: `specs/<docker-repo-name>.json`. Fields:
   - `docker_repo_name` — also used as the GGUF filename stem.
   - `base_model_id` — a Hugging Face model id. An Unsloth pre-quantized
     variant (`unsloth/<model>-unsloth-bnb-4bit`) is preferred where one
     exists — same weights, less memory/time to load than quantizing a
     plain HF checkpoint on the fly.
   - `quant` — GGUF quantization (e.g. `Q4_K_M`).
   - `trained_for` — a short description of the task this model performs.
   - `io_keys` — a `[input_field, output_field]` pair naming the two keys in each seed example.
   - `seed_path` — path (repo-relative) to the seed JSONL.
   - `notebook_path` — path (repo-relative) to write the generated notebook.
   - `system_prompt` — the exact system prompt whatever serves this model should use at inference time, so training matches serving.
   - `input_generation` *(optional)* — how `datagen/generate_dataset.py`
     should produce more seed examples for this spec; see "Growing the
     seed data" below. Omitted if seed data is only ever written by hand.
     `{"mode": "topics", "request_system_prompt": "...", "topics": [[topic, concern], ...]}`
     or `{"mode": "from_spec", "spec": "<other-spec>.json", "field": "<output_field>"}`.
2. **Write seed data**: `data/<name>_seed.jsonl`, one JSON object per line
   with exactly the two keys named in `io_keys`.
3. **Generate**: `python3 tools/generate_notebook.py specs/<name>.json`.
4. Open the generated notebook in Colab, train, and follow the publishing
   steps below.

## Growing the seed data

A dozen hand-written examples is enough to wire up the pipeline, not
enough to train something accurate — task-specific fine-tunes typically
want 1,000+ examples. `datagen/generate_dataset.py` grows a spec's seed
dataset with teacher-generated examples.

This is pure collection, not validation: whatever the teacher produces is
written straight to the seed dataset, no automated pass/fail gate of any
kind. Grading generated plans or code is the qprompt agent graph's job
(`planValidator`, `syntaxvalidator`, `cypressExec` run for real, at
inference time, in `qprompt-langgraph`/`qprompt-a2a`).

Runs locally (it needs a running teacher model; Docker
Model Runner by default):

```bash
pip install -r datagen/requirements.txt
python3 datagen/generate_dataset.py specs/qwen3-testplan.json --count 100
python3 datagen/generate_dataset.py specs/qwen3-cypress-codegen.json --count 100
python3 tools/generate_notebook.py specs/*.json  # pick up the larger dataset
```

By default it talks to Docker Model Runner's direct host engine endpoint
(`docker model status --json` → `endpointHost`) with `ai/qwen3:8B-Q4_K_M`
as the teacher — deliberately a bigger local model than the one being
trained, since distilling a model from itself teaches it nothing new.
`--teacher-base-url`/`--teacher-model` override to a different local
Docker Model Runner port or a different pulled tag. No API keys anywhere
in this path, by design — there's no support for a hosted/external API
here; pulling a bigger local tag (`docker model pull ...`) is the
intended way to get a stronger teacher, not pointing at a cloud endpoint.

Two `input_generation.mode` values exist today, neither tied to any one
domain -- both driven entirely by the spec itself, same as everything
else in this library:
- `"topics"` (`qwen3-testplan`) — the teacher invents each example from a
  topic in the spec's own `input_generation.topics` list (pairs of
  `[topic, a concern that example should cover]`, e.g. "cover the
  empty-results case explicitly"), using the spec's own
  `input_generation.request_system_prompt` to generate the input side.
  Grounding generations in a curated topic list keeps them leaning on
  real domain practice rather than generic phrasing.
- `"from_spec"` (`qwen3-cypress-codegen`) — examples are generated from
  another spec's already-generated output (here, `qwen3-testplan`'s
  plans) as the teacher's input.

A spec's `input_generation.topics` list is a starting point, not
exhaustive — extending it with more topics over time yields more
training data for free on the next `generate_dataset.py` run.

## Publishing

Colab has no Docker daemon, so the final `docker model package` /
`docker model push` step runs on the local machine, after downloading (or
pulling from the HF Hub) the `.gguf` file the notebook produced:

```bash
./scripts/push_model.sh <path-to-model.gguf> ghcr.io/qprompt/<repo-name>:0.6B-Q4_K_M
```

`docker model package`/`push` are registry-agnostic — the same script
works with a Docker Hub tag too (`<dockerhub-username>/<repo-name>:tag`,
no registry host needed, same as it's always worked), GHCR is just the
recommended default now. See
[`scripts/push_model.sh`](scripts/push_model.sh) for the exact commands
it runs.

## CI (GitHub Actions)

Two workflows in `.github/workflows/`, and nothing about either one trains
a model — standard GitHub Actions runners have no GPU, so Unsloth
fine-tuning only ever happens manually in Colab, same as running a
notebook locally would.

- **`validate.yml`** — runs on every push/PR, no GPU needed: specs are
  valid JSON, seed data is valid JSONL, every notebook's code cells still
  parse, and (the one that actually catches drift) regenerating every
  notebook from its spec produces byte-identical output to what's
  committed. This is exactly the kind of check that would have caught
  earlier mismatches between a hand-edited notebook and its generator
  automatically, instead of a few turns of manual diffing.
- **`publish-model.yml`** — manual (`workflow_dispatch`) only, by design:
  there's no automated quality gate in this repo (see "Growing the seed
  data" above), so triggering this by hand after actually reading a
  Colab run's eval output *is* the approval step. Takes a Hugging Face
  Hub repo + GGUF filename and a target GHCR tag (e.g.
  `ghcr.io/qprompt/qwen3-testplan:0.6B-Q4_K_M`), downloads the GGUF, and
  runs the same `docker model package --push` step
  `scripts/push_model.sh` does locally.

No repo secrets to create for `publish-model.yml` — it pushes to GHCR
using the workflow's own automatically-provided `GITHUB_TOKEN` (Model
Runner supports GHCR the same way it supports Docker Hub), gated by the
`permissions: packages: write` already declared in the workflow file.

A GitHub-hosted runner can't reach a file on this machine or a browser
download, only fetch from somewhere on the network -- so the GGUF has to
already be on the Hub before triggering `publish-model.yml`.
`scripts/push_to_huggingface.sh` handles that hop (needs `hf auth login`
first):

```bash
./scripts/push_to_huggingface.sh <path-to-model.gguf> <hf-username>/<repo-name>
```

(Set `HF_HUB_REPO` in the notebook's config cell to push there directly
from Colab instead, if preferred — either way gets the file onto the Hub.)
If publishing straight from this machine instead of via GitHub Actions,
skip this step entirely and use `scripts/push_model.sh` directly.

## Using a published model

The published tag is a plain OCI model artifact — referenceable from
anywhere that accepts one, no integration with this repo required. For
example, in a `qprompt-dsl` `models:` block:

```
qwen3_lora:
  model: ghcr.io/qprompt/qwen3-cypress-codegen:0.6B-Q4_K_M
  context_size: 8192
```

(then regenerate — `qprompt-cli generate` -> `qprompt-langgraph render` /
`qprompt-a2a render` — and rebuild the affected Docker images), or in any
other Docker Compose project's own `models:` key -- all identical to
using a predefined catalog model.

`docker model run <tag> "<prompt>"` is the one exception: it sends only a
user message, with no way to attach a system prompt (`docker model run
--help` has no `--system`/`-s` flag) -- confirmed against a real pushed
model, output silently drifts out of the trained `Title:`/`Objective:`/
`Steps:` format without it. Every model in this library is trained under,
and expects to be served under, a specific system prompt (`specs/*.json`'s
`system_prompt` field), so `docker model run` alone isn't a fair way to
test one. Use `scripts/query_model.py` instead (below), or send the
system prompt explicitly to the OpenAI-compatible endpoint the way any
real caller (qprompt-dsl included) already does.

## Manually testing a published model

```bash
python3 scripts/query_model.py \
  ghcr.io/qprompt/qwen3-testplan:0.6B-Q4_K_M \
  "Verify that a user can log in with valid credentials."
```

The spec (and its `system_prompt`) is auto-detected from the model tag's
repo name. A tag that doesn't match anything in `specs/*.json` just queries with no system
prompt instead of erroring -- same tool, same command shape, whether it's
one of this library's models or not. `--spec`/`--system` override
auto-detection when needed. Appends `/no_think` by default (`--think` to
leave Qwen3's thinking mode on), and prints both the raw response and the
version after `call_llm`'s fence/leading-prose stripping when they
differ. Consistent with the rest of this repo's approach to quality: no
automated scoring, just a fair way to read the output.
