# Week 2 — CPT Implementation Plan

**Continued pretraining of Gemma 2-2B on the Lebanese legal corpus, QLoRA, RunPod.**

| | |
|---|---|
| Base model | `google/gemma-2-2b` (same as the SFT chapter — this is what makes the comparison controlled) |
| Method | Unsloth + 4-bit QLoRA, causal LM objective |
| Sequence length | 2048 |
| Corpus | 15–20M token subset of the Lebanese legal corpus |
| Reference pipeline | `CPT_Ai_lawyer_preprocessing_local` repo |
| Compute | RunPod, RTX 4090 Community Cloud (~$0.34/hr) — see `RunPod_Setup_Guide.md` |
| Duration | 3 days, 1–2 GPU sessions |

---

---

# THE CODE BUNDLE — what she receives and what she writes

She does **not** get repository access. Everything needed is in `cpt_student_bundle.zip` (19 files, 81 KB), organised by the order of use.

## `01_data_prep/`

| File | Status | Used in |
|---|---|---|
| `02_pack_dataset_4096.py` | ✅ **run as-is** — already CLI-parameterised, no code edit | Part 1.2 |
| `subsample_corpus.py` | 📖 reference — already run by the supervisor | Part 0 |

## `02_training/` — run in this order

| File | Status | Used in |
|---|---|---|
| `01_verify_unsloth_env.py` | ✅ as-is | Part 3.1 |
| `02_validate_packed_dataset.py` | ✅ as-is | Part 3.2 |
| `03_smoke_test_cpt.py` | ⚙️ change model name | Part 3.3 |
| `04_remove_remainder_metadata.py` | ✅ as-is (utility) | Part 1.2 if needed |
| `05_full_training_ORIGINAL.py` | ⚙️ **4 edits — see Part 2** | Part 3.5 |

## `03_evaluation/`

| File | Status | Used in |
|---|---|---|
| `next_token_eval_gemma.py` | ✅ **new — written for this project** | Part 4.2 |
| `continuous_next_word_build_dataset.py` | ✅ as-is (model-agnostic) | optional |

## `04_reference/` — read, do not run

| File | Why it is included |
|---|---|
| `03_split_train_val_test_ORIGINAL.py` | Starting point for `03b_split_by_document.py`. **Both its record-level split and its per-source floors need changing** — see Part 1.1. |
| `05_checking_modules.py` + `lora_modules.txt` | Every module LoRA can attach to. Needed if she tries `modules_to_save=["embed_tokens"]` in Part 5. |
| `10_step_timing.txt` | Timing from the 12B run |
| `next_word_inference_sonnet_bedrock_REFERENCE.py` | Bedrock-specific, so not runnable — but its Arabic normalisation and word-matching logic is worth reading |

## `docs/`

| File | Why it matters |
|---|---|
| `evaluation.md` | **The 12B reference numbers.** Base: top-1 31.4%, perplexity 556.01. Final CPT: top-1 62.0%, perplexity 23.28. Cite as prior work; do not re-run. |
| `ORIGINAL_REPO_README.md` | Full pipeline description, chunking parameters, data sources |
| `DATA_PREPROCESSING_PHASE.md` | Detailed preprocessing documentation |
| `Week2_CPT_Plan.md` | This document |

## What is deliberately NOT in the bundle

About 35 scripts from the original repo: raw-source inspection, source-specific chunking, per-source quality checks, dataset combination, header cleaning. **None apply.** The corpus she receives is already inspected, chunked, quality-checked, combined and header-cleaned. She starts from clean text.

## 🔨 The one script she must write herself

`03b_split_by_document.py` — the document-level split (Part 1.1). Three changes from the original:

1. Group by **`source_id`** (the document), not `source` (the corpus name)
2. Ratios **90/5/5**, not 97/2/1
3. Per-source floors **`max(1, ...)`**, not `max(4, ...)` / `max(20, ...)`

Everything else is run-as-is or a documented edit.

---

# PART 0 — Data handover (supervisor, before Day 1)
**This part is done by the supervisor, on the laptop that holds the corpus. It removes ~3 hours from the student's Day 1.**

## The three source files

| File | Location | Size | Send? |
|---|---|---|---|
| `clean/train_cleanheaders.jsonl` | laptop only (gitignored) | **701 MB** | after subsampling |
| `clean/val_cleanheaders.jsonl` | in the repo | 14 MB | after subsampling |
| `clean/test_cleanheaders.jsonl` | in the repo | 7.2 MB | after subsampling |
| `packed/train_..._packed_4096.jsonl` | laptop only | 1.03 GB | ❌ **never** |
| `packed/val_..._packed_4096.jsonl` | in the repo | 21 MB | ❌ **never** |
| `packed/test_..._packed_4096.jsonl` | in the repo | 11 MB | ❌ **never** |

> ⚠️ **Do not send any packed file.** All four were tokenized with the **Gemma 4** tokenizer. The student trains Gemma 2-2B — a different vocabulary — so those `input_ids` are meaningless to her model. She repacks from clean text in Part 1. Skipping these removes 1.03 GB from the transfer.

**Also not needed:** raw sources, `combined_data/`, `source_cpt_files/`, `split_4096_archive/`, checkpoints, adapters.

## 0.1 Verify the schema before anything else

```bash
wc -l train_cleanheaders.jsonl          # expect ~196,983
head -1 train_cleanheaders.jsonl | python3 -m json.tool | head -20
# must show: id, source_id, chunk_index, total_chunks, text, final_cpt_source
```

**If `source_id` is absent, stop.** The document-level re-split in Part 1 depends on it, and without it the leakage problem cannot be fixed.

## 0.2 Subsample by whole documents

The full corpus is ~131M tokens across the three clean files. The student's run targets 15–20M. Sending 701 MB so she can discard three quarters of it wastes time on both ends.

Run `subsample_corpus.py` on the laptop. It samples **entire documents** — never splitting a document's chunks — and preserves the per-source proportions of the full corpus.

```bash
python subsample_corpus.py \
  --inputs train_cleanheaders.jsonl val_cleanheaders.jsonl test_cleanheaders.jsonl \
  --output legal_corpus_subset.jsonl \
  --target_tokens 30000000
```

**Why 30M and not 18M:** it gives the student headroom to do the document-level split herself and still land at ~18M for training, plus margin for a second run on more data if the first config needs adjusting.

**Save the printed per-source table.** It goes into her Chapter 8 as the corpus description.

## 0.3 Compress

Measured compression on this corpus: **4.32×**.

| Stage | Size |
|---|---|
| All three clean files, raw | ~722 MB |
| Subsampled to 30M tokens | ~166 MB |
| **Gzipped** | **~38 MB** |

```bash
gzip -9 legal_corpus_subset.jsonl
```

## 0.4 Hand over the file

At ~38 MB gzipped, any channel works. Two options:

**Google Drive** — upload `legal_corpus_subset.jsonl.gz`, share the folder **to her account specifically**, not "anyone with the link". Most of this is public-record material, but the ADL rulings and bibliographic set should not be indexable.

**Direct upload to the pod** — drag and drop into the JupyterLab file browser at `/workspace`, then:

```bash
cd /workspace
gunzip legal_corpus_subset.jsonl.gz
```

Either way the file ends up at `/workspace/legal_corpus_subset.jsonl` on the network volume, where it survives pod termination.

### Acceptance — observed values from the actual run

```
documents        : 192,472  ->  43,284 kept
records          : 203,073  ->  45,636 kept
estimated tokens : 134,264,628  ->  30,019,618  (22.3%)
```

- `legal_corpus_subset.jsonl.gz` is ~35–40 MB
- All seven sources present in the kept set
- The student can decompress and read line 1 as valid JSON with a `source_id` field

**Keep the script's printed per-source table.** It is reproduced in Part 1.0 and goes into Chapter 8.

---

# PART 1 — Day 1 morning: CPU preparation

**No GPU. Do not deploy a pod for any of this.** Run it on a laptop or in free Colab — paying $0.34/hr to tokenize is pure waste.

Part 0 already pooled the three clean splits and subsampled them to ~30M tokens, so this part is now **~90 minutes, not three hours**. The input is one file:

```
/content/legal_corpus_subset.jsonl     # ~166 MB, ~30M tokens, whole documents
```

## 1.0 What the subset actually contains

The subsample has been run. These are the real figures — check against them after decompressing:

| | Full corpus | Subset received |
|---|---|---|
| Documents | 192,472 | **43,284** |
| Records (chunks) | 203,073 | **45,636** |
| Estimated tokens | 134,264,628 | **30,019,618** (22.3%) |

Per source:

| Source | Docs | Tokens | **Tokens / doc** |
|---|---|---|---|
| `associated_studies_ar` | 7 | 180,591 | **25,799** |
| `adl_rulings` | 727 | 3,063,813 | **4,214** |
| `gazette_legal_core` | 11,319 | 11,401,003 | 1,007 |
| `legislations` | 13,426 | 10,683,363 | 796 |
| `related_provisions` | 888 | 412,914 | 465 |
| `gazette_section2_sample` | 4,751 | 2,051,948 | 432 |
| `bibliographic_rulings` | 12,166 | 2,225,986 | **183** |

**This table goes in Chapter 8 as the corpus description.**

Two things follow from it, and both matter:

**(a) A 140× spread in document length.** At 2048 tokens per packed block, one block holds roughly **11 whole `bibliographic_rulings` documents** separated by EOS, while a single `associated_studies_ar` document spans about **12 blocks**. This is what EOS separators exist for, so it is not a problem — but state it in the methodology rather than letting an examiner find it.

**(b) It is a free error-analysis axis.** The project's existing `evaluation.md` reports Claude Sonnet scoring **lowest on bibliographic rulings (24%)** and **highest on legislation (44%)**. If the CPT model reproduces that ordering, it is independent corroboration across two very different models. **Break the next-token evaluation down by source** in Part 4 — it costs nothing and may produce a finding.

## 1.1 Re-split at document level — the fix

The original `03_split_train_val_test.py` groups by **source** (`official_gazette`, `adl`…) and shuffles **records**. Records are chunks, and chunking used `OVERLAP_CHARS = 1200`. Measured leakage in the existing split:

| Split | From multi-chunk documents | Siblings in another split |
|---|---|---|
| val | 166 of 4,061 (4.1%) | 164 |
| test | 93 of 2,029 (4.6%) | 91 |

Consecutive chunks share 1,200 characters of literal text, so ~4% of the held-out set overlaps training data.

**Write `03b_split_by_document.py`** — the only script in this whole plan she writes from scratch. Start from `04_reference/03_split_train_val_test_ORIGINAL.py` in the bundle: group by `source_id` (not `source`), shuffle the *document IDs*, then assign whole documents to splits. Use `SEED = 42` and stratify per source.

### Split ratios

The original used 97/2/1, appropriate for a 134M-token corpus. On a 30M-token subset that leaves a thin held-out set, so use **90/5/5**:

| Split | Share | Approx. tokens |
|---|---|---|
| train | 90% | ~27M |
| val | 5% | ~1.5M |
| test | 5% | ~1.5M |

A ~1.5M-token test split is ample for the next-token evaluation, which samples 1,000 examples.

If the training set should be closer to 18M than 27M, take a further **document-level** slice of the train split only — never re-slice val or test, and never slice at record level.

### ⚠️ Per-source floors — do NOT copy the original script's numbers

The original applies these floors:

```python
if source == "associated_studies_ar":
    val_n = max(4, val_n)      # WRONG at this corpus size
    test_n = max(2, test_n)
if source == "related_provisions":
    val_n = max(20, val_n)
    test_n = max(10, test_n)
```

Those were tuned for the **full** corpus. In this subset `associated_studies_ar` has only **7 documents**. Forcing 4 into val and 2 into test leaves **1 document for training** — the source effectively disappears from the training data.

Use `max(1, ...)` instead, giving 5 train / 1 val / 1 test. One document there is still ~26,000 tokens, which is more than enough for a small source.

`related_provisions` (888 docs) is fine at a plain 5%, but check it rather than assume.

### Acceptance

```python
train_docs = {r["source_id"] for r in train}
val_docs   = {r["source_id"] for r in val}
test_docs  = {r["source_id"] for r in test}

# 1. No document appears in two splits
assert not (train_docs & val_docs)
assert not (train_docs & test_docs)
assert not (val_docs  & test_docs)

# 2. Every source appears in all three splits
sources = {r["final_cpt_source"] for r in all_records}
for split_name, split in [("train", train), ("val", val), ("test", test)]:
    present = {r["final_cpt_source"] for r in split}
    missing = sources - present
    assert not missing, f"{split_name} is missing sources: {missing}"

# 3. No source lost its training data to a floor
for s in sources:
    n = sum(1 for r in train if r["final_cpt_source"] == s)
    assert n >= 3, f"source {s} has only {n} training records"
```

Print documents and records per split per source. **This goes in the thesis** — it is a measurable improvement over the original pipeline.

## 1.2 Repack with the Gemma 2 tokenizer

The packing script is already parameterised — no code edit, only flags:

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

> ⚠️ `--append_eos` and `--drop_remainder` are `store_true` flags. The original 4096 run used both. Omitting them silently changes the data.

### Acceptance

- Report header shows `MODEL: google/gemma-2-2b` — **check this first**; packing with the wrong tokenizer produces nonsense loss from step 1
- `Average tokens per packed block: 2048.0`
- Dropped tokens are a negligible fraction
- Train blocks ≈ 13,000 at 27M tokens (≈ 8,800 if sliced to 18M)
- `input_ids` and `attention_mask` both length 2048 on every row

## 1.3 Record the tokenizer fertility

She already measured Gemma fertility on Arabic in §2.3.3 of the report. Run the same measurement on this legal corpus and compare.

Legal Arabic should fragment worse than general Arabic. If so, that is a direct callback from Chapter 2 to Chapter 8 and it motivates the `modules_to_save` experiment in Part 5.

**Deliverable: one number and one sentence.**

---

# PART 2 — Day 1 afternoon: adapt the training script

Copy `02_training/05_full_training_ORIGINAL.py` → `05_full_training_gemma2_2b.py`. Four edits.

## 2.1 Make `boto3` optional

Currently imported at module top level; it will fail unless boto3 happens to be installed.

```python
try:
    import boto3
except ImportError:
    boto3 = None
```

## 2.2 Change the defaults

```python
MODEL_NAME     = "google/gemma-2-2b"          # was google/gemma-4-12B
MAX_SEQ_LENGTH = 2048                          # was 4096
OUTPUT_DIR     = "/workspace/adapters/gemma2_2b_cpt_v1"
TRAIN_FILE     = ".../packed_2048/train_packed_2048.jsonl"
VAL_FILE       = ".../packed_2048/val_packed_2048.jsonl"
```

## 2.3 Check the loader return signature

The 12B script unpacks a **processor** (Gemma 4 is multimodal):

```python
model, processor = FastLanguageModel.from_pretrained(...)
```

Gemma 2-2B is text-only and returns a **tokenizer**. Rename the variable, and check that `processing_class=` on the `Trainer` still accepts it in the installed TRL/transformers version. If it errors, use `tokenizer=`.

**This is the most likely place the script breaks. Find out in the smoke test, not in the real run.**

## 2.4 Always pass `--skip_s3_upload`

There is no S3 bucket. Without the flag the script raises at startup:

```
ValueError: --s3_bucket is required unless --skip_s3_upload is used.
```

Checkpoints go to the network volume via `OUTPUT_DIR`.

## 2.5 Everything else stays

Keep `packed_data_collator`, `validate_example`, `strip_non_training_columns`, resume logic, `load_best_model_at_end`, and `run_summary.json` exactly as they are. They are validated and model-agnostic.

## 2.6 Hyperparameters — use what worked

| Parameter | Value | Note |
|---|---|---|
| `lora_r` | 16 | as in the 12B run |
| `lora_alpha` | 32 | ratio 2.0 |
| `lora_dropout` | 0.0 | |
| `target_modules` | the 7 projections | unchanged |
| `learning_rate` | **2e-4** | **validated on this corpus** — do not lower it |
| `lr_scheduler_type` | cosine | |
| `warmup_ratio` | 0.03 | |
| `weight_decay` | 0.01 | |
| `max_grad_norm` | 1.0 | |
| `optim` | adamw_8bit | |
| `batch_size` / `grad_accum` | 1 / 8 | effective batch 8 |
| `bf16` | True | **bf16-capable GPU only** — see Part 7.3 |
| `num_epochs` | 1 | |
| `save_steps` | 100 | |
| `save_total_limit` | 3 | network volume space |

> **On the learning rate:** earlier guidance said CPT needs 1e-5 to 5e-5. The 12B run in the repo used **2e-4** and took perplexity from 556.01 to 23.28. That number is validated on this exact corpus. Use it.

---

# PART 3 — Day 1 evening: verify, then launch

Run in order. Do not skip.

## 3.1 Environment

```bash
python training_codes/01_verify_unsloth_env.py
nvidia-smi
```

**Confirm `torch.cuda.is_bf16_supported()` returns True.** If the pod was assigned a T4, V100 or P100, terminate it and redeploy — `bf16=True` will fail there. See Part 7.3.

## 3.2 Dataset validation

```bash
python training_codes/02_validate_packed_dataset.py
```

## 3.3 Smoke test

```bash
python 05_full_training_gemma2_2b.py --dry_run_samples 64 --num_epochs 1 --skip_s3_upload
```

### Acceptance — all four must hold

| Check | Expected |
|---|---|
| `print_trainable_parameters()` | ~0.5–1.5% trainable. If 0% or ~100%, LoRA did not attach. |
| Loss | Finite and decreasing. Any NaN → stop. |
| Peak VRAM | Well under the card's limit |
| Adapter saved | `adapter_config.json` + `adapter_model.safetensors` present |

**Record seconds per optimizer step.** Full run time = `sec_per_step × (blocks / 8)`.

## 3.4 Written hypothesis — before launching

Dated, one paragraph, three predictions:

- Held-out legal perplexity → **down**
- Next-token top-1 accuracy → **up**
- Dataset A accuracy (retention) → **roughly flat** (base is frozen under QLoRA)

Five minutes. It converts the writeup from description into "we predicted X, observed Y."

## 3.5 Launch

```bash
python 05_full_training_gemma2_2b.py --skip_s3_upload
```

Expected runtime, memory and cost are in **Part 7** below. Read it before launching — it also covers which GPUs are unusable and what to do if the pod gets one.

---

# PART 4 — Day 2: evaluate

## 4.1 Baseline first

Run the evaluation on **base Gemma 2-2B with no adapter**. This is the comparison point for everything.

## 4.2 Run the next-token evaluation

> **The original repo has no Gemma-side evaluation script** — only the Sonnet/Bedrock ones. `03_evaluation/next_token_eval_gemma.py` in the bundle was written for this project and reproduces the exact metric set in `docs/evaluation.md`: top-1, top-5, top-10, average probability of the correct token, average NLL, perplexity, average rank, plus a per-source breakdown.

Three commands, in this order:

```bash
# 1. BUILD ONCE — CPU only, no GPU needed
python 03_evaluation/next_token_eval_gemma.py build \
  --input splits/test_doclevel.jsonl \
  --out   eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --n 1000 --seed 42

# 2. score the BASE model (no adapter)
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

# 4. comparison table with per-source deltas
python 03_evaluation/next_token_eval_gemma.py compare \
  --results eval/results_base.json eval/results_cpt.json
```

> ⚠️ **Build the eval set exactly once.** Rebuilding between models changes the sampled positions and base-vs-CPT stops being a comparison. The script guards against the most common version of this mistake — it refuses to score if the tokenizer name does not match the one recorded at build time — but it cannot detect a rebuild with the same tokenizer and a different seed.

Step 1 needs no GPU. Do it during Part 1.

### Per-source breakdown

Both `score` and `compare` produce it automatically. `docs/evaluation.md` reports Claude Sonnet on the same corpus:

| Source | Sonnet accuracy |
|---|---|
| Legislation | **44%** |
| Official Gazette | 38% |
| ADL rulings | 35% |
| Related provisions | 34% |
| Associated study | 34% |
| Bibliographic rulings | **24%** |

If the CPT model reproduces this ordering, that is independent corroboration across two very different models — a free finding. If it does not, the divergence is itself worth a paragraph.

## 4.3 The results table

| Model | Top-1 | Top-5 | Top-10 | Avg NLL | Perplexity | Dataset A acc |
|---|---|---|---|---|---|---|
| Base Gemma 2-2B | | | | | | 90.42%* |
| CPT Gemma 2-2B | | | | | | |
| *(reference)* Base Gemma 4 12B | 31.4% | 50.7% | 61.0% | 6.321 | 556.01 | — |
| *(reference)* CPT Gemma 4 12B | 62.0% | 80.3% | 86.1% | 3.150 | 23.28 | — |

\* from the SFT adapter, not the base — label this carefully.

The 12B rows are cited from prior work in the same project, **not re-run**. They give a scale reference for free.

## 4.4 Retention check

Run the **CPT** model on the Dataset A test set. QLoRA freezes the base, so expect little forgetting. Whatever the result, it is evidence for the thesis.

## 4.5 Qualitative

10–15 legal prompts, base output vs CPT output side by side. Include 3–4 in the thesis, the rest in an appendix.

## 4.6 Gate

**Pass:** CPT beats base on ≥2 of 3 metrics.

**Fail:** check in this order —
1. Did the adapter actually load at inference?
2. Does the loss curve show learning, or is it flat?
3. Is the evaluation set truly held out (document-level)?
4. Only then consider the learning rate.

One retry, overnight. Then write up whatever happened.

---

# PART 5 — Optional Run B (only if Run A passed and time allows)

Add embedding adaptation:

```python
modules_to_save=["embed_tokens"]
```

**Why it might matter:** Gemma 2 has a 256k vocabulary and ties input/output embeddings. Her own fertility measurement (Part 1.5) shows how heavily Arabic legal terms fragment. Adapting the embeddings is the mechanism that would help.

**Why it might break:** `embed_tokens` is 590M parameters — larger than the rest of the model's trainable set by an order of magnitude. Wrapping it creates a full trainable copy, and tied embeddings are a known sharp edge in PEFT.

**Check before committing a session:** run `print_trainable_parameters()` after wrapping and do the 64-sample smoke test. If trainable jumps to ~600M and VRAM holds, proceed.

If both runs complete, **A vs B is a free mini-ablation** on whether embedding adaptation matters for domain vocabulary — a self-contained finding at the cost of one extra hour.

---

# PART 6 — Day 3: write

## 6.1 Chapter 8 — Methodology: CPT

- Corpus: sources, token counts, document counts
- **The document-level re-split and why** — include the leakage measurement
- Repacking at 2048 with the Gemma 2 tokenizer
- Configuration table, with a column comparing SFT vs CPT settings
- Why the configuration differs — cite her own §4.6

## 6.2 Chapter 9 — Results: CPT

- Base vs CPT on all metrics
- Loss curve
- Retention on Dataset A
- Qualitative samples
- The 12B reference rows, clearly marked as prior work

## 6.3 Chapter 10 — Discussion: the dissociation

**This is the thesis conclusion.**

> **SFT changed the task format.** The model learned to select a sense ID from candidates already present in the prompt. No knowledge was added — everything required was in the context window. This is why 2B matched 8B on Dataset A.
>
> **CPT changed the domain distribution.** The model learned legal vocabulary, collocations and register. It did not gain task ability.

Frame as **"what does each objective change,"** never "which is better." The two are not directly comparable — different data, different objective, different metric — and the dissociation holds regardless of effect size.

Cite Gururangan et al. (2020), *"Don't Stop Pretraining"* for the DAPT framing.

## 6.4 Limitations — CPT side

1. No matched-token-budget instruction-SFT control on the legal corpus
2. Single run, single seed, single corpus
3. Subset of the full corpus (~18M of ~131M tokens)
4. Perplexity and next-token accuracy measure distributional fit, not legal reasoning
5. The 12B comparison is cited, not reproduced under matched conditions

---

# PART 7 — Runtime, memory and cost (RunPod)

**Platform: RunPod, RTX 4090 on Community Cloud.** Colab Pro is not purchasable from Lebanon. Setup steps are in `RunPod_Setup_Guide.md`.

## 7.1 How the run is sized

```
tokens trained on           = T
packed blocks               = T / 2048
effective batch             = batch_size(1) x grad_accum(8) = 8 blocks
tokens per optimizer step   = 8 x 2048 = 16,384
optimizer steps             = T / 16,384
```

| Corpus for training | Packed blocks | Optimizer steps |
|---|---|---|
| 18M tokens | ~8,800 | ~1,100 |
| 27M tokens (90/5/5 of a 30M subset) | ~13,200 | ~1,650 |

## 7.2 Expected wall-clock

Gemma 2-2B, 4-bit QLoRA, r=16, seq 2048, batch 1 x grad_accum 8, gradient checkpointing on. Planning figures only — **the smoke test in Part 3.3 gives the true seconds-per-step**, and the script writes it to `run_summary.json`.

| GPU | bf16 | Community rate | sec / step | 18M tokens | 27M tokens |
|---|---|---|---|---|---|
| **RTX 4090 24 GB** | ✅ Ada | **~$0.34/hr** | ~4–7 | **1–2 h** | **1.5–3 h** |
| RTX A5000 24 GB | ✅ Ampere | ~$0.25/hr | ~6–10 | 1.5–3 h | 2.5–4.5 h |
| L4 24 GB | ✅ Ada | ~$0.40/hr | ~6–12 | 2–4 h | 3–6 h |
| A40 48 GB | ✅ Ampere | ~$0.40/hr | ~5–8 | 1.5–2.5 h | 2–4 h |

```
full run seconds = sec_per_step x optimizer_steps
```

**The 4090 is faster than an A100 40GB on this workload and a fraction of the price.** Take it whenever it is available.

## 7.3 GPUs that must NOT be used

| GPU | Architecture | Why not |
|---|---|---|
| T4 | Turing | **no bf16** |
| V100 | Volta | **no bf16** |
| P100 | Pascal | **no bf16** |

The training script sets `bf16=True`. Gemma 2 uses logit soft-capping (attention 50.0, final 30.0), and fp16 overflow on this family is a real failure mode. A multi-hour CPT run has far more exposure to it than a short SFT run.

**Check before doing anything else:**

```bash
python -c "import torch; print(torch.cuda.is_bf16_supported())"   # must print True
```

If it prints `False`, terminate the pod and redeploy. Do not "work around" it by switching to fp16.

## 7.4 Memory footprint

Peak VRAM for Gemma 2-2B, 4-bit base, r=16 adapters, batch 1:

| Component | seq 1024 | seq 2048 |
|---|---|---|
| 4-bit base weights | ~1.5 GB | ~1.5 GB |
| Embeddings (256k vocab, tied, not quantized) | ~1.2 GB | ~1.2 GB |
| LoRA params + grads + 8-bit Adam | ~0.3 GB | ~0.3 GB |
| Activations (gradient checkpointing on) | ~0.8 GB | ~1.6 GB |
| Logits + fused cross-entropy | ~1.0 GB | ~2.0 GB |
| **Peak** | **~5 GB** | **~7 GB** |

Comfortable on a 24 GB card. **Memory is not the constraint at 2B — time is.**

Run B (`modules_to_save=["embed_tokens"]`) adds roughly **+3.5 GB**: a full trainable fp32 copy of the 590M-parameter embedding matrix plus its gradient and optimizer state. Still fine on 24 GB.

## 7.5 Cost

| Item | Hours | Cost |
|---|---|---|
| Smoke tests, environment checks | 0.5 | $0.17 |
| Baseline evaluation (base model) | 0.5 | $0.17 |
| **Run A — CPT training, 27M tokens** | ~2.5 | $0.85 |
| Evaluation of Run A | 0.5 | $0.17 |
| Optional Run B | ~2.5 | $0.85 |
| Retry margin | ~2.5 | $0.85 |
| **GPU subtotal** | **~9 h** | **~$3.06** |
| Network volume, 30 GB, 2 weeks | | ~$2.10 |
| **Total** | | **~$5.20** |

**$10 of credit leaves roughly 90% headroom.**

### 🔴 The rule that protects the budget

> **Terminate the pod whenever she is not actively using the GPU.**

RunPod bills for every minute a pod runs, idle or not. A 4090 left running over a weekend is about **$16** — more than the entire budget. Redeploying takes under a minute, packages persist on the network volume, and nothing is lost.

**No GPU is needed for:** the document-level split (Part 1.1), packing (Part 1.2), fertility measurement (Part 1.3), or building the evaluation set (Part 4.2). Do those on a laptop or in free Colab.

## 7.6 Storage — everything under `/workspace`

The container disk is **destroyed when the pod is terminated**. The 30 GB network volume mounted at `/workspace` is not.

| What | Size | Location |
|---|---|---|
| Corpus subset, gzipped | ~38 MB | `/workspace` |
| Corpus subset, decompressed | ~166 MB | `/workspace` |
| Repacked 2048 JSONL, all splits | ~400–600 MB | `/workspace/packed_2048/` |
| Installed Python packages | ~8–12 GB | `/workspace` (`pip install` from `/workspace`) |
| One checkpoint | ~130–180 MB | `/workspace/adapters/...` |
| 3 checkpoints (`save_total_limit=3`) | ~500 MB | same |
| Final adapter | ~85 MB | same |

30 GB is comfortable. Set `OUTPUT_DIR = "/workspace/adapters/gemma2_2b_cpt_v1"`.

**No Google Drive, no FUSE mount.** `/workspace` is fast network storage, so the "copy to local disk first" advice that applies on Colab is unnecessary here.

## 7.7 Evaluation time

Forward-only, so fast:

| Task | Time |
|---|---|
| Next-token eval, 1,000 examples | 5–15 min |
| Held-out perplexity | 5–10 min |
| Dataset A retention, 3,110 items (generative) | 20–40 min |

Budget about an hour per model — and **the base model counts as a model**.

## 7.8 Sessions and interruptions

No session cap, no idle timeout, no compute-unit arithmetic to track.

The one risk is Community Cloud: hosts are independent operators, so a pod can be interrupted with no SLA. Mitigations already in place:

- The training script **resumes automatically** from the latest checkpoint in `OUTPUT_DIR`. Re-running the same command picks up where it stopped — no flags needed.
- Checkpoints are on the network volume, which survives pod termination.
- Keep `save_steps=100`. Drop to 50 if interruptions prove frequent.

If interruptions become a real problem, Secure Cloud runs the same 4090 at ~$0.74/hr — roughly double, still under $8 for the whole week.

---

# Failure modes, ranked

| # | Risk | Symptom | Response |
|---|---|---|---|
| 1 | Pod assigned a non-bf16 GPU (T4/V100/P100) | `is_bf16_supported()` returns False | Terminate and redeploy on 4090/A5000/L4/A40 — Part 7.3 |
| 2 | `processor` vs `tokenizer` signature | TypeError at `Trainer` init | Caught by the smoke test; rename |
| 3 | `boto3` import | ImportError on first line | Fixed in Part 2.1 |
| 4 | `source_id` missing from data | Re-split impossible | Check in Part 0 acceptance |
| 4b | Copied the original per-source floors | `associated_studies_ar` left with 1 training doc | Use `max(1, ...)`; assertion 3 in Part 1.1 catches it |
| 4c | A source vanishes from val or test | Per-source eval breakdown impossible | Assertion 2 in Part 1.1 |
| 5 | Forgot `--append_eos` / `--drop_remainder` | Blocks not exactly 2048 | Packing report catches it |
| 6 | Wrong tokenizer used for packing | Nonsense loss from step 1 | Verify `MODEL:` line in the packing report header |
| 6b | Supervisor sent a packed 4096 file and she used it | Nonsense loss from step 1 | Never use a `*_packed_4096.jsonl` — all are Gemma 4 |
| 7 | Network volume fills up | Save failure mid-run | `save_total_limit=3`; 30 GB is enough |
| 7b | Work written outside `/workspace` | Lost on pod termination | Everything under `/workspace` — Part 7.6 |
| 7c | Pod left running idle | ~$8/day burned | Terminate after every session — Part 7.5 |
| 8 | Eval set rebuilt between base and CPT | Base vs CPT numbers not comparable | Build once in Part 4.2; the script blocks a tokenizer mismatch but not a re-seed |
| 9 | `--skip_s3_upload` omitted | `ValueError: --s3_bucket is required` at startup | Part 2.4 |

---

# Timeline

| Day | Morning | Afternoon | Evening |
|---|---|---|---|
| **1** | Part 1 — re-split + repack (~90 min; Part 0 done by supervisor beforehand) | Part 2 — adapt script | Part 3 — verify, launch |
| **2** | Part 4 — baseline + CPT eval | Gate check; qualitative | Part 5 — optional Run B |
| **3** | Run B results if applicable | Part 6 — write Ch. 8–9 | Ch. 10 discussion |

**Minimum 2 days. Budget 3.** Runtime, memory, cost and storage figures are in Part 7.

### Across the whole week

| | Value |
|---|---|
| Total GPU hours | **~9 h** |
| GPU cost, RTX 4090 Community at $0.34/hr | **~$3.06** |
| Network volume, 30 GB, 2 weeks | ~$2.10 |
| **Total** | **~$5.20 of a $10 budget** |
| Fits the 3-day window? | ✅ comfortably |
| Risk of fp16 loss spike | none — bf16 on a 4090 |

---

# Order of operations

```
1.  decompress corpus subset                     ->  legal_corpus_subset.jsonl
2.  write + run 03b_split_by_document.py         ->  splits/{train,val,test}_doclevel.jsonl
3.  02_pack_dataset_4096.py  x3                  ->  packed_2048/*.jsonl
4.  next_token_eval_gemma.py build               ->  eval/eval_set_1000.json
5.  01_verify_unsloth_env.py                     ->  is_bf16_supported() == True
6.  02_validate_packed_dataset.py                ->  shapes correct
7.  05_full_training... --dry_run_samples 64     ->  smoke test
8.  next_token_eval_gemma.py score (base)        ->  eval/results_base.json
9.  05_full_training... --skip_s3_upload         ->  final_adapter
10. next_token_eval_gemma.py score (cpt)         ->  eval/results_cpt.json
11. next_token_eval_gemma.py compare             ->  the results table
```

Steps 1–4 need **no GPU**.

---

# The two rules

1. **Nothing in Part 0, Part 1 or the eval-set build needs a GPU.** Do those before deploying a pod. And terminate the pod after every session — RunPod bills idle time.
2. **The smoke test is not optional.** Every failure mode above except #1 is caught by 64 samples and five minutes.
