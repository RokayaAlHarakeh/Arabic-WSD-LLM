# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for the paper *"Zero-Shot and Fine-Tuned Evaluation of Generative LLMs for Arabic Word Sense Disambiguation"* (ArabicNLP 2025, `../2025.arabicnlp-main.24.pdf`). It benchmarks four LLMs (GPT-4o, LLaMA 3.1-8B, Qwen 2.5-7B, Gemma 2-9B) on two Arabic WSD datasets, in zero-shot and LoRA fine-tuned settings. These are **research scripts, not an application**: no build, no linter, no test suite. "Running" means executing a stage script by hand.

The task: given a sentence, a target word, and candidate **sense IDs + Arabic definitions**, predict the correct sense ID.

## Repository layout = a `Model / Mode / Dataset` matrix

`GPT-4o/`, `Gemma/`, `Llama/`, `Qwen/` each contain `Zero-shot(s)/` and/or `Fine-tuning/`, each split into `Dataset-A/` and `Dataset-B/`. Every leaf is a small set of scripts that follow the **same two templates** below — so once you understand one model's pipeline you understand all of them. (Naming is inconsistent across older dirs: `Llama/Fineuning`, `Qwen/.../finetunig.py`, zero-shot stage-2 files named `get_predictions.py.py`.)

- **Dataset A** = El-Razzaz (gloss-based, short sentences).
- **Dataset B** = SALMA (sense-annotated corpus, long sequences up to ~4096 tokens; B-specific helpers exist: `find_max_seq_len.py`, `filter_longer.py`).

## The shared data contract (`Datasets/<Dataset>/`)

Everything is keyed off three UTF-8 JSON shapes, in `train80` / `dev20` / `test` splits:
- `*_set.json` — `[{sentence_id, sentence, words:[{word_id, word, senses:[sense_id,...]}]}]`
- `*_truth.json` — same shape, but each word has `target_sense` (the gold sense_id)
- `*_dictionary.json` — `[{sense_id, definition}]` (the gloss lookup)

The `create_*.py` scripts under `Datasets/<Dataset>/` produce these from the raw sources. Pred-vs-truth evaluation aligns by `(sentence_id, word_id)` or by list order, so **do not reorder `*_set.json`**.

## Pipeline 1 — Zero-shot (2 stages, text-log based)

1. `query_model.py` runs the model over `test_set.json` and writes a **plain-text log** in a strict format the parser depends on:
   ```
   ===== Query for Word: <word> =====
   Sentence: "<sentence>"
   Response:
   <raw model text>
   ```
   Backends differ by model: open models (Gemma/Llama/Qwen) use a local **Ollama** server via `langchain_ollama`/`OllamaLLM`; GPT-4o uses `langchain_openai.ChatOpenAI`.
2. `get_predictions.py` parses that log with a large multi-template regex (`SENSE_REGEX`), extracts **exactly one** sense ID per block (returns `"none"` if 0 or ≥2 distinct IDs), aligns to gold words, writes predictions JSON, and optionally evaluates. It's a CLI:
   ```
   python get_predictions.py --log debug_info.txt --truth test_truth.json --pred predictions.json --eval eval_report.json
   ```

## Pipeline 2 — Fine-tuning (4 stages, Unsloth + LoRA)

1. `create_finetuning_dataset.py` → JSONL of **Alpaca-format** records `{instruction, input, output}` where `output` is the gold sense_id. The `instruction` is a fixed WSD prompt; `input` packs sentence + target word + candidate senses.
2. `finetuning.py` → `unsloth.FastLanguageModel` + `get_peft_model` (LoRA) + `trl.SFTTrainer`, 4-bit, Alpaca prompt + `eos`, saves `merged_16bit/` (optionally pushes to HF Hub).
3. `infer_model.py` → loads the tuned model, greedy generation, **regex-extracts** the sense ID (here it's simple — the model is trained to emit just the ID + `<eos>`), writes predictions JSON.
4. `eval.py` → sklearn `accuracy_score` + macro `precision_recall_fscore_support`; `"none"` is treated as just another label; writes a report JSON.

**The inference prompt must exactly match the training prompt.** Output extraction (free text → sense ID) is the main fragility in both pipelines.

## Actively developed path: `Gemma/Fine-tuning/Dataset-A/`

This is the maintained, Colab-ready slice (an internship reproducing the Gemma row with Gemma 2-2B, then Gemma 4-E4B). Unlike the rest of the repo, its `finetuning.py` / `infer_model.py` / `eval.py` are **parameterized via env vars** so the same code serves both models — change only the base model:

| Env var | Default | Purpose |
|---|---|---|
| `WSD_PROJECT_DIR` | `/content/drive/MyDrive/WSD_Project` | Drive root for `data/` + `outputs/<tag>/` |
| `WSD_BASE_MODEL` | `unsloth/gemma-2-2b` | base checkpoint to fine-tune |
| `WSD_MODEL_TAG` | `gemma2_2b` | names the output subfolder |
| `WSD_MAX_SEQ_LEN` | `1024` | lower to `768` if a T4 OOMs |

Driven by `Gemma2_2B_DatasetA.ipynb` / `Gemma4_E4B_DatasetA.ipynb` (clone fork → stage data → run the 3 scripts). See `Gemma/Fine-tuning/Dataset-A/RUNBOOK.md` for the full run order, hyperparameters, and results table. Hyperparameters: LoRA `r=32, alpha=32, dropout=0.05`, 4-bit, packing, 3 epochs, lr `2e-4`, `adamw_8bit`, seed 3407.

## Commands & environment

- Install: `pip install -r requirements.txt`. Note gaps: `trl`/`peft` are pulled transitively by `unsloth`; zero-shot open models additionally need `langchain_ollama` plus a running Ollama with the model pulled (e.g. `ollama pull gemma2:9b-instruct-q8_0`).
- Secrets via `.env` (loaded by `python-dotenv`): `HF_TOKEN` (gated HF models / Hub push), `OPENAI_API_KEY` (GPT-4o).
- Most scripts elsewhere in the repo have **empty-string path placeholders** (`test_path = ""`) or hardcoded absolute paths — fill these before running. The Gemma/Dataset-A scripts are the exception (env-driven).
- Fine-tuning targets a **Colab GPU (T4)**, not the local laptop; `venv/` and `unsloth_compiled_cache/` are local-only artifacts. `test_one_sample.py`, `test_gemma2b.py`, `download_gemma2b.py` are ad-hoc smoke scripts, not a test suite.

## Conventions / gotchas

- All data is **Arabic UTF-8**: always read/write with `encoding="utf-8"` and dump with `ensure_ascii=False`.
- The local Windows console is cp1252 — printing Arabic to stdout raises `UnicodeEncodeError`. Set `PYTHONIOENCODING=utf-8` or avoid printing raw Arabic when running scripts locally.
- No local PDF text extractor is installed; read the paper PDF in a viewer (or `pip install pypdf`).
