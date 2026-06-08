# Runbook — Gemma Fine-Tuning for Arabic WSD (Dataset A)

Reproduce the paper's Gemma WSD result with **Gemma 2-2B** (Phase 1), then test whether
**Gemma 4-E4B** does better (Phase 2). Training runs on **free Colab T4**; artifacts live in
Google Drive at `MyDrive/WSD_Project/`.

## Pipeline (4 scripts, all in this folder)

| Script | Role |
|---|---|
| `create_finetuning_dataset.py` | builds `fine_tuning_dataset_elrazzaz.jsonl` from the train80 split (already done — output is committed) |
| `finetuning.py` | Unsloth + LoRA SFT (4-bit) → saves `merged_16bit/` |
| `infer_model.py` | runs the tuned model over `test_set.json` → `predictions_<tag>.json` |
| `eval.py` | accuracy + macro-F1 → `report_<tag>.json` |

All three runnable scripts are **parameterized via env vars** so the same code serves both phases:

| Env var | Default | Purpose |
|---|---|---|
| `WSD_PROJECT_DIR` | `/content/drive/MyDrive/WSD_Project` | Drive root for data + outputs |
| `WSD_BASE_MODEL` | `unsloth/gemma-2-2b` | base checkpoint to fine-tune |
| `WSD_MODEL_TAG` | `gemma2_2b` | names the output subfolder |
| `WSD_MAX_SEQ_LEN` | `1024` | lower to `768` if a T4 OOMs on the 4B model |

Drive layout produced:
```
WSD_Project/
  data/           # Dataset-A json + the training jsonl (staged by the notebook)
  outputs/<tag>/  # checkpoints/, merged_16bit/, predictions_<tag>.json, report_<tag>.json, debug_<tag>.txt
```

## How to run

The fixed scripts + notebooks are committed on branch **`phase1-gemma2-2b-datasetA`**. `origin` is the
team lead's repo, so push to **your own fork** (the Colab notebook clones it):

1. Create a fork on GitHub (web UI): `https://github.com/<your-username>/Arabic-WSD-LLM`.
2. Add it as a remote and push the branch:
   ```
   git remote add fork https://github.com/<your-username>/Arabic-WSD-LLM.git
   git push -u fork phase1-gemma2-2b-datasetA
   ```
3. Open **`Gemma2_2B_DatasetA.ipynb`** in Colab → GPU runtime → set `REPO_URL` to your fork
   (`BRANCH` is already `phase1-gemma2-2b-datasetA`) → Run all.
4. Repeat with **`Gemma4_E4B_DatasetA.ipynb`** (confirm the exact Unsloth Gemma-4 id via the discovery
   cell; the only difference is the Config cell).

> Tip: if you merge the branch into your fork's `main`, set `BRANCH = 'main'` in the clone cell.

## Hyperparameters (kept from the paper's recipe)

LoRA `r=32, alpha=32, dropout=0.05`; targets all attn+MLP projections; 4-bit; `max_seq_len=1024`;
`packing=True`; 3 epochs; lr `2e-4`; `adamw_8bit`; batch 1 × grad-accum 8; seed 3407.
Keep these identical across both phases — change **only** `WSD_BASE_MODEL`.

## Phase 0 — target recorded ✅ (from paper Table 3)

**Validation target — Gemma 2-9B fine-tuned on Dataset A: Accuracy = 89.39%, Macro-F1 = 81.72.**

The paper's full Table 3 (Accuracy / Macro-F1, %):

| Model | A zero-shot | A fine-tuned | B zero-shot | B fine-tuned |
|---|---|---|---|---|
| **Gemma 2-9B** | 65.34 / 50.64 | **89.39 / 81.72** | 72.46 / 56.45 | 87.23 / 67.80 |
| LLaMA 3.1-8B | 48.59 / 38.28 | 90.42 / 83.20 | 54.78 / 39.98 | 88.51 / 69.41 |
| Qwen 2.5-7B | 67.40 / 53.02 | 90.77 / 83.98 | 55.99 / 47.97 | 82.22 / 63.07 |
| GPT-4o | 79.16 / 67.92 | — | 79.55 / 64.23 | — |

Confirmed: the paper's Dataset-A recipe matches our `finetuning.py` exactly (epochs 3, bs 1 × 8 accum,
lr 2e-4, max_len 1024, LoRA r=32/α=32/dropout 0.05, packing, AdamW_8bit, wd 0.01, linear sched,
warmup 50, seed 3407, 4-bit). One difference: the paper used an **NVIDIA L4**; we use a free **T4**
(fine for 2B; watch VRAM for the 4B in Phase 2).

## Results — fill in after each run

| Model | Params | Accuracy (%) | Macro-F1 | Source |
|---|---|---|---|---|
| Gemma 2-9B | 9B | 89.39 | 81.72 | paper Table 3, Dataset A (target) |
| Gemma 2-2B | 2B | _(report_gemma2_2b.json)_ | | mine, Phase 1 |
| Gemma 4-E4B | 4B | _(report_gemma4_e4b.json)_ | | mine, Phase 2 |

**Question to answer for the team:** does Gemma 4-E4B beat Gemma 2 on Arabic WSD? Note compute/time
and any OOM workaround (e.g. `WSD_MAX_SEQ_LEN=768`).

## Notes / gotchas

- Exact paper numbers won't match (4-bit + different hardware); aim for the same ballpark and the same ranking.
- A low macro-F1 with high accuracy is expected when many sense classes are rare — that's the paper's pattern too.
- `eval.py` aligns predictions and ground truth positionally, so don't reorder `test_set.json`.
- Leftover dirs from earlier 9B attempts (`outputs_gemma_elrazzaz/`, `gemma_9b_elrazzaz_merged_16bit/`) are not used by this pipeline and can be ignored/cleaned.
