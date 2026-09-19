# CPT Student Bundle

Everything needed to run continued pretraining of **Gemma 2-2B** on the Lebanese legal corpus, extracted from the `CPT_Ai_lawyer_preprocessing_local` repository.

Follow `Week2_CPT_Plan.md` (supplied separately). This file says what each script is and what has to change.

---

## What you will NOT find here, and why

The original repo has ~35 more scripts covering raw-source inspection, source-specific chunking, per-source quality checks, and dataset combination. **None are needed.** The corpus you receive is already inspected, chunked, quality-checked, combined, and header-cleaned. You start from clean text.

---

## `01_data_prep/`

| File | Status | What to do |
|---|---|---|
| `subsample_corpus.py` | ✅ already run by the supervisor | Reference only. Shows how the 30M-token subset was drawn — whole documents, per-source proportions preserved. Its printed table goes in Chapter 8. |
| `02_pack_dataset_4096.py` | ✅ **no code edit** | Already CLI-parameterised. Run with different flags. |

### Packing

```bash
python 01_data_prep/02_pack_dataset_4096.py \
  --model_name google/gemma-2-2b \
  --input_file  splits/train_doclevel.jsonl \
  --output_file packed_2048/train_packed_2048.jsonl \
  --report_file reports/train_packed_2048.txt \
  --max_seq_length 2048 \
  --append_eos --drop_remainder
```

Repeat for val and test.

> ⚠️ `--append_eos` and `--drop_remainder` are `store_true` flags. The original 4096 run used **both**. Omitting either silently changes the data.

> ⚠️ Check the report header says `MODEL: google/gemma-2-2b`. Packing with the wrong tokenizer produces nonsense loss from step 1 and looks like a training bug.

### You still have to write one script

`03b_split_by_document.py` — the document-level split. Use `04_reference/03_split_train_val_test_ORIGINAL.py` as the starting point, and make three changes:

1. Group by **`source_id`** (the document), not `source` (the corpus name)
2. Ratios **90/5/5**, not 97/2/1
3. Per-source floors: use `max(1, ...)`, **not** the original's `max(4, ...)` / `max(20, ...)`

Reason for (3): in this subset `associated_studies_ar` has only **7 documents**. The original floor of 4 for val + 2 for test would leave **1 document for training** and the source effectively disappears.

---

## `02_training/`

Run these **in order**.

| File | Status | Purpose |
|---|---|---|
| `01_verify_unsloth_env.py` | ✅ as-is | Unsloth imports, GPU visible |
| `02_validate_packed_dataset.py` | ✅ as-is | Packed JSONL readable, shapes correct |
| `03_smoke_test_cpt.py` | ⚙️ change model name | Short CPT run, checks loss is finite |
| `04_remove_remainder_metadata.py` | ✅ as-is | Utility if packing left extra columns |
| `05_full_training_ORIGINAL.py` | ⚙️ **4 edits — see below** | The production run |

### The four edits to `05_full_training_ORIGINAL.py`

Copy it to `05_full_training_gemma2_2b.py` first.

**1. Make `boto3` optional** — it is imported at module level and will fail on Colab.

```python
try:
    import boto3
except ImportError:
    boto3 = None
```

**2. Change the defaults**

```python
MODEL_NAME     = "google/gemma-2-2b"      # was google/gemma-4-12B
MAX_SEQ_LENGTH = 2048                      # was 4096
OUTPUT_DIR     = "/content/drive/MyDrive/CPT_Project/adapters/gemma2_2b_cpt_v1"
TRAIN_FILE     = "packed_2048/train_packed_2048.jsonl"
VAL_FILE       = "packed_2048/val_packed_2048.jsonl"
```

**3. Fix the loader return signature.** Gemma 4 12B is multimodal and returns a **processor**:

```python
model, processor = FastLanguageModel.from_pretrained(...)
```

Gemma 2-2B is text-only and returns a **tokenizer**. Rename the variable, then check that `processing_class=` on the `Trainer` still accepts it in your installed TRL/transformers version. If it raises a TypeError, use `tokenizer=` instead.

**This is the most likely place the script breaks. The smoke test will tell you.**

**4. Always pass `--skip_s3_upload`.** There is no S3 bucket; checkpoints go to Drive.

### Leave everything else alone

`packed_data_collator`, `validate_example`, `strip_non_training_columns`, the resume logic, `load_best_model_at_end`, and `run_summary.json` are validated and model-agnostic.

### Hyperparameters — use what worked

| Parameter | Value |
|---|---|
| `lora_r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.0 |
| `target_modules` | the 7 projections (unchanged) |
| `learning_rate` | **2e-4** |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.03 |
| `optim` | adamw_8bit |
| `batch_size` / `grad_accum` | 1 / 8 |
| `bf16` | **True** — requires L4 or A100, **not T4** |
| `num_epochs` | 1 |

> The 12B run used `learning_rate=2e-4` and took perplexity from 556.01 to 23.28 on this exact corpus. **Do not lower it.** Generic advice about continued pretraining needing 1e-5 to 5e-5 does not apply here.

### Running

```bash
# validate the whole path on a tiny slice FIRST
python 02_training/05_full_training_gemma2_2b.py \
    --dry_run_samples 64 --num_epochs 1 --skip_s3_upload

# the real run
python 02_training/05_full_training_gemma2_2b.py --skip_s3_upload
```

Resume is automatic — re-running the same command picks up the latest checkpoint. No flags needed.

---

## `03_evaluation/`

| File | Status | Purpose |
|---|---|---|
| `next_token_eval_gemma.py` | ✅ **new, written for this project** | Base vs CPT next-token evaluation |
| `continuous_next_word_build_dataset.py` | ✅ as-is | Builds the word-level rolling benchmark (model-agnostic) |

The original repo has **no Gemma-side next-token evaluation script** — only Sonnet/Bedrock ones. `next_token_eval_gemma.py` fills that gap and reproduces the exact metric set in `docs/evaluation.md`: top-1, top-5, top-10, average target probability, average NLL, perplexity, average rank.

### Three commands

```bash
# 1. BUILD ONCE. Never rebuild between models.
python 03_evaluation/next_token_eval_gemma.py build \
  --input splits/test_doclevel.jsonl \
  --out   eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --n 1000 --seed 42

# 2. score the BASE model
python 03_evaluation/next_token_eval_gemma.py score \
  --eval_set eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --out eval/results_base.json --use_unsloth

# 3. score the CPT model against the SAME eval set
python 03_evaluation/next_token_eval_gemma.py score \
  --eval_set eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --adapter .../final_adapter \
  --out eval/results_cpt.json --use_unsloth

# 4. comparison table, including per-source deltas
python 03_evaluation/next_token_eval_gemma.py compare \
  --results eval/results_base.json eval/results_cpt.json
```

> ⚠️ **Build the eval set once.** If you rebuild it between models the sampled positions change and base-vs-CPT is no longer a comparison. The script refuses to score if the tokenizer name does not match the one used at build time.

### Per-source breakdown

Both `score` and `compare` report per-source metrics automatically. `docs/evaluation.md` shows Claude Sonnet scoring highest on legislation (44%) and lowest on bibliographic rulings (24%). **If your CPT model reproduces that ordering, that is corroboration across two very different models** — a finding worth a paragraph. If it does not, the divergence is also worth a paragraph.

---

## `04_reference/`

Not run. Read when needed.

| File | Why it is here |
|---|---|
| `03_split_train_val_test_ORIGINAL.py` | Starting point for your `03b_split_by_document.py`. **Note the record-level split and the per-source floors — both need changing.** |
| `05_checking_modules.py` + `lora_modules.txt` | Lists every module LoRA can attach to. Useful if you try `modules_to_save=["embed_tokens"]` in the optional Run B. |
| `10_step_timing.txt` | Timing from the 12B run, for comparison |
| `next_word_inference_sonnet_bedrock_REFERENCE.py` | Bedrock-specific, so not directly usable — but its Arabic normalisation and word-matching logic is worth reading if you extend to word-level evaluation |
| `subsample_corpus.py` | (in `01_data_prep/`) how your corpus subset was drawn |

---

## `docs/`

| File | Why it matters |
|---|---|
| `evaluation.md` | **The evaluation protocol and the 12B reference numbers.** Base Gemma 4 12B: top-1 31.4%, perplexity 556.01. Final CPT checkpoint: top-1 62.0%, perplexity 23.28. Cite these as prior work — do not re-run them. |
| `ORIGINAL_REPO_README.md` | Full pipeline description, chunking parameters, data sources |
| `DATA_PREPROCESSING_PHASE.md` | Detailed preprocessing documentation |

---

## Order of operations

```
1. decompress corpus subset            ->  legal_corpus_subset.jsonl
2. write + run 03b_split_by_document.py ->  splits/{train,val,test}_doclevel.jsonl
3. 02_pack_dataset_4096.py  x3          ->  packed_2048/*.jsonl
4. 01_verify_unsloth_env.py             ->  GPU is L4 or A100, not T4
5. 02_validate_packed_dataset.py        ->  shapes correct
6. 05_full_training...  --dry_run_samples 64  ->  smoke test
7. next_token_eval_gemma.py build       ->  eval_set_1000.json
8. next_token_eval_gemma.py score (base) ->  results_base.json
9. 05_full_training...                  ->  final_adapter
10. next_token_eval_gemma.py score (cpt) ->  results_cpt.json
11. next_token_eval_gemma.py compare     ->  the results table
```

Steps 1–3 and 7 need **no GPU**. Do not start a GPU runtime for them.
