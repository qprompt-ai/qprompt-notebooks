"""Turns a model spec (JSON) + a seed dataset (JSONL) into a self-contained
Colab fine-tuning notebook.

This is the library's actual scaling mechanism: adding a new model --
a different domain, a different task, whatever the next small
fine-tuned model needs to be -- means writing a spec + seed data, not
hand-editing notebook JSON. The published result is a plain OCI model
tag on Docker Hub; nothing here is specific to any one consumer of that
tag. See specs/*.json for the models that exist today, and README.md for
the full workflow.

Usage:
    python3 tools/generate_notebook.py specs/qwen3-cypress-codegen.json
    python3 tools/generate_notebook.py specs/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def md(*lines: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)}


def code(*lines: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "\n".join(lines),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build(spec: dict, seed: list[dict]) -> dict:
    docker_repo_name = spec["docker_repo_name"]
    base_model_id = spec["base_model_id"]
    trained_for = spec["trained_for"]
    io_a, io_b = spec["io_keys"]
    system_prompt = spec["system_prompt"]
    quant = spec.get("quant", "Q4_K_M")

    cells = []

    quant_method = quant.lower()

    cells.append(md(
        f"# Fine-tune {base_model_id} for {trained_for}",
        "",
        f"Generated from `specs/{docker_repo_name}.json` via `tools/generate_notebook.py`, "
        "edit the spec and regenerate rather than hand-editing this file.",
    ))

    cells.append(code(
        "!pip install -q -U unsloth",
    ))

    cells.append(code(
        "import json",
        "import torch",
        "from unsloth import FastLanguageModel",
        "from datasets import Dataset",
        "from trl import SFTConfig, SFTTrainer",
        "",
        f'BASE_MODEL_ID = "{base_model_id}"',
        "MAX_SEQ_LENGTH = 2048",
        f'OUTPUT_DIR = "/content/{docker_repo_name}"',
        'ADAPTER_DIR = f"{OUTPUT_DIR}/adapter"',
        'GGUF_DIR = f"{OUTPUT_DIR}/gguf"',
        f'GGUF_QUANT = "{quant_method}"',
        "",
        "HF_HUB_REPO = None",
    ))

    cells.append(md("## Load the base model and attach a LoRA adapter"))

    cells.append(code(
        "model, tokenizer = FastLanguageModel.from_pretrained(",
        "    model_name=BASE_MODEL_ID,",
        "    max_seq_length=MAX_SEQ_LENGTH,",
        "    dtype=None,",
        "    load_in_4bit=True,",
        ")",
        "if tokenizer.pad_token is None:",
        "    tokenizer.pad_token = tokenizer.eos_token",
        "",
        "model = FastLanguageModel.get_peft_model(",
        "    model,",
        "    r=16,",
        "    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\", \"gate_proj\", \"up_proj\", \"down_proj\"],",
        "    lora_alpha=32,",
        "    lora_dropout=0.05,",
        "    bias=\"none\",",
        "    use_gradient_checkpointing=\"unsloth\",",
        "    random_state=3407,",
        ")",
    ))

    cells.append(md(
        "## Seed dataset",
        "",
        f"Starter ({io_a} -> {io_b}) pairs, kept inline so this notebook runs standalone "
        "from just this one file on Colab. The same examples also live in "
        f"`{spec['seed_path']}` for versioning outside the notebook -- edit that file and "
        "regenerate rather than editing the cell below by hand.",
    ))

    seed_lines = [f"SYSTEM_PROMPT = {system_prompt!r}", "", "SEED_EXAMPLES = ["]
    for ex in seed:
        seed_lines.append(f"    {json.dumps(ex)},")
    seed_lines.append("]")
    cells.append(code(*seed_lines))

    cells.append(code(
        "def to_chat_text(example):",
        "    messages = [",
        "        {\"role\": \"system\", \"content\": SYSTEM_PROMPT},",
        f'        {{"role": "user", "content": example[{io_a!r}]}},',
        f'        {{"role": "assistant", "content": example[{io_b!r}]}},',
        "    ]",
        "    return {\"text\": tokenizer.apply_chat_template(messages, tokenize=False)}",
        "",
        "dataset = Dataset.from_list(SEED_EXAMPLES).map(to_chat_text)",
        "dataset = dataset.train_test_split(test_size=min(2, len(dataset) // 6 or 1), seed=42)",
        "dataset",
    ))

    cells.append(md("## Train"))

    cells.append(code(
        "trainer = SFTTrainer(",
        "    model=model,",
        "    processing_class=tokenizer,",
        "    train_dataset=dataset[\"train\"],",
        "    eval_dataset=dataset[\"test\"],",
        "    args=SFTConfig(",
        "        output_dir=OUTPUT_DIR,",
        "        num_train_epochs=3,",
        "        per_device_train_batch_size=2,",
        "        gradient_accumulation_steps=4,",
        "        learning_rate=2e-4,",
        "        optim=\"adamw_8bit\",",
        "        weight_decay=0.01,",
        "        lr_scheduler_type=\"linear\",",
        "        logging_steps=1,",
        "        eval_strategy=\"epoch\",",
        "        save_strategy=\"no\",",
        "        dataset_text_field=\"text\",",
        "        max_length=MAX_SEQ_LENGTH,",
        "        seed=3407,",
        "        report_to=\"none\",",
        "        fp16=not torch.cuda.is_bf16_supported(),",
        "        bf16=torch.cuda.is_bf16_supported(),",
        "    ),",
        ")",
        "trainer.train()",
    ))

    cells.append(code(
        "model.save_pretrained(ADAPTER_DIR)",
        "tokenizer.save_pretrained(ADAPTER_DIR)",
    ))

    cells.append(md(
        "## Evaluate",
        "- the eval-loss trend training already computed each epoch "
        "(`SFTConfig(eval_strategy=\"epoch\")` above).",
        "- a side-by-side look at what the model actually produces on the "
        f"held-out ({io_a} -> {io_b}) examples it never trained on.",
    ))

    cells.append(code(
        "eval_history = [",
        "    (h[\"epoch\"], h[\"eval_loss\"])",
        "    for h in trainer.state.log_history",
        "    if \"eval_loss\" in h",
        "]",
        "for epoch, loss in eval_history:",
        "    print(f\"epoch {epoch:.1f}: eval_loss={loss:.4f}\")",
    ))

    cells.append(code(
        "FastLanguageModel.for_inference(model)",
        "",
        "for example in dataset[\"test\"]:",
        "    messages = [",
        "        {\"role\": \"system\", \"content\": SYSTEM_PROMPT},",
        f'        {{"role": "user", "content": example[{io_a!r}]}},',
        "    ]",
        "    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)",
        "    model_inputs = tokenizer(prompt_text, return_tensors=\"pt\").to(model.device)",
        "    output_ids = model.generate(**model_inputs, max_new_tokens=500)",
        "    generated = tokenizer.decode(output_ids[0][model_inputs[\"input_ids\"].shape[1]:], skip_special_tokens=True)",
        "    print(\"=\" * 80)",
        f'    print("INPUT:", example[{io_a!r}])',
        f'    print(\"\\nEXPECTED:\", example[{io_b!r}])',
        "    print(\"\\nMODEL OUTPUT:\", generated)",
        "",
        "FastLanguageModel.for_training(model)",
    ))

    cells.append(md(
        "## Export a merged, quantized GGUF for Docker Model Runner",
        "",
        "Unsloth merges the LoRA adapter into the base weights and quantizes to GGUF in "
        "one call -- no separate llama.cpp clone/build/quantize step needed.",
    ))

    cells.append(code(
        "if HF_HUB_REPO:",
        "    from huggingface_hub import login",
        "    login()",
        "    model.push_to_hub_gguf(HF_HUB_REPO, tokenizer, quantization_method=GGUF_QUANT)",
        "    print(f\"pushed to https://huggingface.co/{HF_HUB_REPO}\")",
        "else:",
        "    import glob",
        "    from google.colab import files",
        "    model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method=GGUF_QUANT)",
        "    gguf_paths = glob.glob(f\"{OUTPUT_DIR}/**/*.gguf\", recursive=True)",
        "    gguf_path = next((p for p in gguf_paths if GGUF_QUANT.lower() in p.lower()), gguf_paths[0])",
        "    files.download(gguf_path)",
    ))

    return notebook(cells)


def generate(spec_path: Path) -> Path:
    spec = json.loads(spec_path.read_text())
    seed = load_jsonl(REPO_ROOT / spec["seed_path"])
    nb = build(spec, seed)
    out_path = REPO_ROOT / spec["notebook_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(nb, indent=1))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="+", help="path(s) to spec JSON files (specs/*.json)")
    args = parser.parse_args()

    for raw_path in args.specs:
        spec_path = Path(raw_path)
        out_path = generate(spec_path)
        print(f"{spec_path} -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
