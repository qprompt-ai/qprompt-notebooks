# qprompt-notebooks

A model training library. Each model is a spec (`specs/*.json`) plus
seed data (`data/*.jsonl`), and `tools/generate_notebook.py` turns that
into a self-contained notebook. Trained models are pushed as OCI
artifacts to a container registry and referenced from anywhere that
accepts one (a Compose `models:` block, `docker model run`, etc.), same
as a predefined catalog model like `ai/qwen3:...`.

GHCR (`ghcr.io/qprompt-ai/...`) is the default registry. Docker Hub
Organizations require a paid plan; GitHub orgs and GHCR public packages
don't.

| Notebook | Trained for | Registry repo |
|---|---|---|
| [`notebooks/cypress_codegen_finetune.ipynb`](notebooks/cypress_codegen_finetune.ipynb) | Test plan → Cypress/TypeScript code | `qwen3-cypress-codegen` |
| [`notebooks/test_plan_finetune.ipynb`](notebooks/test_plan_finetune.ipynb) | Feature request → structured test plan | `qwen3-testplan` |

Both fine-tune `unsloth/Qwen3-0.6B-unsloth-bnb-4bit` with a LoRA adapter
via [Unsloth](https://unsloth.ai), runs fine on Colab's free T4. Neither
notebook is hand-written. Edit the spec/seed data and regenerate.

## Seed data

`data/*.jsonl` has a mix of hand-written and teacher-generated examples,
unvalidated. Review it before training. Task-specific fine-tunes
typically want 1,000+ examples; a dozen is enough to wire up the
pipeline, not enough for production.

## Pipeline

Each notebook, in order:

1. Loads the base model in 4-bit, attaches a LoRA adapter.
2. Fine-tunes with `trl.SFTTrainer`, saves the adapter.
3. Evaluates: prints eval loss per epoch and generates on held-out
   examples for manual review. No automated pass/fail.
4. Merges the adapter and quantizes to GGUF (`Q4_K_M` by default).
5. Pushes the `.gguf` to a Hugging Face Hub repo.

## Adding a new model

1. Write `specs/<name>.json`:
   - `docker_repo_name` — GGUF filename stem.
   - `base_model_id` — HF model id (prefer an Unsloth `-bnb-4bit` variant).
   - `quant` — GGUF quantization, e.g. `Q4_K_M`.
   - `trained_for` — one-line task description.
   - `io_keys` — `[input_field, output_field]` names.
   - `seed_path`, `notebook_path` — repo-relative paths.
   - `system_prompt` — must match what serves this model at inference time.
   - `input_generation` *(optional)* — see below.
2. Write `data/<name>_seed.jsonl`, one JSON object per line.
3. `python3 tools/generate_notebook.py specs/<name>.json`
4. Train, then publish (below).

## Growing the seed data

`datagen/generate_dataset.py` adds teacher-generated examples to a
spec's seed dataset.

```bash
pip install -r datagen/requirements.txt
python3 datagen/generate_dataset.py specs/qwen3-testplan.json --count 100
python3 datagen/generate_dataset.py specs/qwen3-cypress-codegen.json --count 100
python3 tools/generate_notebook.py specs/*.json
```

Runs locally against Docker Model Runner (`ai/qwen3:8B-Q4_K_M` teacher by
default, bigger than the model being trained). No API keys, no hosted
endpoints; pull a bigger local tag instead. Override with
`--teacher-base-url`/`--teacher-model`.

Two `input_generation.mode` values, both spec-driven:
- `"topics"` — teacher invents examples from `input_generation.topics`
  (`[topic, concern]` pairs) using `input_generation.request_system_prompt`.
- `"from_spec"` — examples drawn from another spec's already-generated
  output (`{"spec": "...", "field": "..."}`).

## Publishing

```bash
./scripts/push_model.sh <path-to-model.gguf> ghcr.io/qprompt-ai/<repo>:0.6B-Q4_K_M
```

Works with a Docker Hub tag too (`<user>/<repo>:tag`, no registry host).
See [`scripts/push_model.sh`](scripts/push_model.sh).

## CI

Two workflows, neither trains anything (no GPU on standard runners).

- **`validate.yml`** — every push/PR: specs valid, seed data valid,
  notebook cells parse, notebooks match what their specs regenerate.
- **`publish-model.yml`** — manual trigger only. Downloads a GGUF from a
  Hugging Face Hub repo and pushes it to GHCR using the workflow's own
  `GITHUB_TOKEN` (no secrets to configure).

The GGUF needs to be on the Hub first, since a GitHub runner can't reach
this machine:

```bash
./scripts/push_to_huggingface.sh <path-to-model.gguf> <hf-username>/<repo-name>
```

(Or set `HF_HUB_REPO` in the notebook to push there from Colab directly.)
Publishing straight from this machine skips this step: use
`scripts/push_model.sh`.

## Using a published model

```
qwen3_lora:
  model: ghcr.io/qprompt-ai/qwen3-cypress-codegen:0.6B-Q4_K_M
  context_size: 8192
```

Works in any Compose `models:` block, same as a predefined catalog tag.

`docker model run <tag> "<prompt>"` won't give a fair test: it has no
way to attach a system prompt, and every model here is trained to be
served under one (`specs/*.json`'s `system_prompt`). Use
`scripts/query_model.py` instead, or attach the system prompt yourself.

## Testing a model manually

```bash
python3 scripts/query_model.py \
  ghcr.io/qprompt-ai/qwen3-testplan:0.6B-Q4_K_M \
  "Verify that a user can log in with valid credentials."
```

Spec and system prompt are auto-detected from the tag's repo name.
Unmatched tags query with no system prompt instead of erroring.
`--spec`/`--system` override detection. Appends `/no_think` by default
(`--think` to disable). Prints both the raw response and the
fence/prose-stripped version when they differ.
