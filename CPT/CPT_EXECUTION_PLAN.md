# CPT Execution Plan — Gemma 2-2B on the Lebanese legal corpus

Companion to the supervisor's `Week2_CPT_Plan.md`. That document is the specification;
this one is the execution order, with the numbers measured from the delivered corpus,
the two places the spec collides with this repo's history, and the concept + writing work.

Branch: `week2-cpt`. Base model: `google/gemma-2-2b` — the same base as the frozen SFT
adapter, which is what makes the comparison controlled.

---

## 0. Status

| Stage | State |
|---|---|
| Part 0 — subsample to 30M tokens | ✅ done by supervisor |
| Part 1.1 — document-level split | ✅ **done** — `scripts/03b_split_by_document.py`, all checks pass |
| Part 1.2 — repack at 2048 | ✅ **done** — 13,262 / 696 / 719 blocks |
| Part 1.3 — tokenizer fertility | ✅ **done** — 2.337 tokens/word |
| Cloze probe (timeline Day 7) | ✅ **done** — 300 items, 253 documents |
| Eval set (1,000 next-token items) | ✅ **done** — all 7 sources |
| **Stage A complete** | ✅ 2026-09-19 Colab CPU; re-run identically on RunPod 2026-09-23 |
| Stage B — Unsloth fork | ✅ **resolved: plain transformers**, Unsloth never needed |
| Part 3 — smoke test | ✅ **passed** 2026-09-23 on RTX PRO 4000 Blackwell |
| Part 2 — adapt training script | ⬜ **read §2 first: the Unsloth fork** |
| Part 3 — verify, smoke, launch | ⬜ |
| Part 4 — evaluate | ⬜ |
| Part 6 — write Ch. 8–10 | ⬜ |

### Numbers already established

| | Value |
|---|---|
| Documents / records | 43,680 / 45,636 |
| Train | 39,310 docs, 41,095 records, **27.1M est. tokens** |
| Val / Test | 2,185 docs each, ~1.43M / ~1.45M est. tokens |
| Packed blocks at 2048 | **13,262** train / 696 val / 719 test |
| True train tokens (Gemma 2 tokenizer) | **27,162,160** (estimate was off by 0.1%) |
| Dropped in packing | 1,584 tokens — 0.006% |
| Optimizer steps at effective batch 8 | **1,658** |
| Leakage avoided vs record-level split | 5.9% of val, 5.6% of test |
| Fertility, legal Arabic | **2.337** tokens/word |
| Fertility, general Arabic (§2.3.3) | 2.079 — legal is **1.124×** worse |
| Fertility, English (§2.3.3) | 1.163 — legal is **2.009×** worse |

### Measured on the pod (RTX PRO 4000 Blackwell, 24 GB)

| | Value |
|---|---|
| Seconds per optimizer step | **8.01** |
| Projected full run, 1,658 steps | **~3.7 h ≈ $2.10** at $0.57/hr |
| Peak VRAM | **11.79 GB** of 23.42 |
| Trainable parameters | 20,766,720 of 2,635,108,608 — **0.7881%** |
| Initial loss (sanity probe) | **6.359** vs uniform-random 12.453 → ppl **577.8** |

That initial perplexity of 577.8 for base Gemma 2-2B sits beside the **556.01** reported
for base Gemma 4-12B on this same corpus in `docs/evaluation.md` — different model, nearly
the same starting point. Useful corroboration that the setup measures what the reference
measured, and worth a sentence in Chapter 9.

**Stack that works** (record it — this project has been burned by a silent version break):
torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, CUDA 12.8,
sm_120. Two fixes were needed: transformers 5.x dropped `warmup_ratio` (converted to
`warmup_steps`), and `eval_batch_size` had to drop from 4 to 1 because Gemma 2's 256k
vocabulary makes eval logits ~2 GB per sequence at 2048 tokens.

Stage A was reproduced exactly on Colab from the committed scripts: identical document
counts, token counts, leakage figures and cloze composition as the local run, same seed,
different OS. That is a reproducibility claim worth one sentence in Chapter 8.

The fertility result confirms the Chapter 2 prediction and is what justifies trying
`modules_to_save=["embed_tokens"]` in Run B: if legal terms fragment 12% harder than
general Arabic, a frozen embedding table is representing domain vocabulary poorly. At 2048
tokens a block holds ~876 legal Arabic words against ~985 of general Arabic or ~1,760 of
English — same compute, less content.

"Provisional" because `chars / 2.6` was calibrated on the **Gemma 4** tokenizer. The
packing report in Stage A gives the true count.

---

## Stage A — Finish data preparation (CPU only, ~90 min)

**No GPU runtime. Do not burn compute units on tokenization.**

### A.1 Regenerate the splits on Colab

The splits are 162 MB and gitignored, so don't upload them — regenerate. Takes about a
minute and keeps the seed reproducible.

```bash
git clone -b week2-cpt <your fork> /content/repo
cd /content/repo/CPT
cp /content/drive/MyDrive/CPT_Project/legal_corpus_subset.jsonl.gz /content/
gunzip /content/legal_corpus_subset.jsonl.gz

python scripts/03b_split_by_document.py \
  --input  /content/legal_corpus_subset.jsonl \
  --outdir /content/splits \
  --report reports/03b_split_by_document.txt
```

Expect `All acceptance checks passed.` and train ≈ 41,095 records. If the record count
differs, the corpus file differs — stop and find out why before continuing.

> Keep everything on `/content/`, not Drive. Drive is a FUSE mount and streaming a
> 162 MB JSONL from it is far slower than local disk.

### A.2 Repack at 2048 with the Gemma 2 tokenizer

Run as-is — no code edit, only flags. Three times:

```bash
for SPLIT in train val test; do
  python cpt_student_bundle/01_data_prep/02_pack_dataset_4096.py \
    --model_name google/gemma-2-2b \
    --input_file  /content/splits/${SPLIT}_doclevel.jsonl \
    --output_file /content/packed_2048/${SPLIT}_packed_2048.jsonl \
    --report_file reports/${SPLIT}_packed_2048.txt \
    --max_seq_length 2048 \
    --append_eos --drop_remainder
done
```

`google/gemma-2-2b` is gated — `huggingface-cli login` with your `HF_TOKEN` first.

**Acceptance, in this order:**

1. Report header says `MODEL: google/gemma-2-2b`. **Check this before anything else** — packing with the wrong tokenizer produces nonsense loss from step 1 and looks exactly like a training bug.
2. `Average tokens per packed block: 2048.0`
3. Every row has `input_ids` and `attention_mask` of length 2048
4. Dropped tokens are a negligible fraction
5. Train blocks ≈ 13,250

> ⚠️ `--append_eos` and `--drop_remainder` are `store_true`. Omitting either silently
> changes the data. The original 4096 run used both.

> ⚠️ Never use any `*_packed_4096.jsonl` file. All were tokenized with the Gemma 4
> vocabulary and are meaningless to your model.

### A.3 Record tokenizer fertility

Compare Gemma 2's fertility on this legal corpus against the general-Arabic figure in
§2.3.3 of your report. Legal Arabic should fragment worse.

Cheap version: `true_tokens / est_tokens` from the packing report versus `1.0`, plus a
direct `chars per token` measurement on a sample.

**Deliverable: one number and one sentence.** It is a direct callback from Chapter 2 to
Chapter 8, and it is the motivation for the optional `modules_to_save` run in §E.

### A.4 Build the evaluation set — once, and only once

```bash
python cpt_student_bundle/03_evaluation/next_token_eval_gemma.py build \
  --input /content/splits/test_doclevel.jsonl \
  --out   eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --n 1000 --seed 42
```

CPU only. **Copy `eval_set_1000.json` to Drive immediately** and never rebuild it.
Rebuilding between base and CPT changes the sampled positions and the comparison stops
being a comparison. The script refuses to score on a tokenizer mismatch, but it cannot
detect a rebuild with the same tokenizer and a different seed.

---

## Stage B — The Unsloth fork ⚠️

**Read this before touching the training script. It is the biggest risk in Week 2 and
the supervisor's plan does not know about it.**

The bundle is built on Unsloth. `05_full_training_ORIGINAL.py` imports
`FastLanguageModel` at module level (line 63) and uses it for loading, for
`get_peft_model`, and for `use_gradient_checkpointing="unsloth"`.

**But this repo already found Unsloth broken for Gemma 2.** Per
`broken-unsloth-gemma2-colab`: unsloth 2026.6.6 + transformers 5.5.0 produced a patched
forward that emitted only newlines, and it **corrupted the entire Phase-1 training run** —
loss started at ≈25, the adapter was unusable, and inference returned all-`none`. The fix
was to drop Unsloth entirely and rewrite `finetuning.py` on plain transformers QLoRA.

So the bundle's core assumption is one this project has already disproved once. Two
possible outcomes, and you decide at the smoke test, not before:

### Path 1 — Unsloth works on the current Colab stack

Possible: the breakage was version-specific and months have passed. Run
`02_training/01_verify_unsloth_env.py`, then the smoke test. If loss is finite and
decreasing and `print_trainable_parameters()` shows ~0.5–1.5%, proceed as the plan is
written. Pin the versions in `run_summary.json` the moment it works.

### Path 2 — Unsloth is still broken → swap in your own loader

You already have the replacement working: `Gemma/Fine-tuning/Dataset-A/finetuning.py`.
The edit is contained, because everything downstream of the loader in
`05_full_training_ORIGINAL.py` is **stock transformers** — `Trainer`,
`packed_data_collator`, `validate_example`, the resume logic, `run_summary.json` — and
none of it cares how the model was built.

Replace the `FastLanguageModel` block with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)   # copy from your finetuning.py
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb_config,
    attn_implementation="sdpa",       # correct for Gemma 2 softcapping, ~2x faster than eager
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
                                         target_modules=[...the 7 projections...],
                                         task_type="CAUSAL_LM"))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
```

Then `use_gradient_checkpointing="unsloth"` goes away, and `processing_class=processor`
becomes `processing_class=tokenizer`.

**Evaluation is unaffected either way.** `next_token_eval_gemma.py` already has a clean
non-Unsloth path — just omit `--use_unsloth` and it uses `AutoModelForCausalLM` +
`BitsAndBytesConfig` + `PeftModel`.

**Whichever path you take, write down which and why.** It is a methodology sentence in
Chapter 8 and a near-certain defense question.

---

## Stage C — Adapt, verify, launch

### C.1 The four edits (plan Part 2)

Copy `05_full_training_ORIGINAL.py` → `scripts/05_full_training_gemma2_2b.py`, then:

1. **`boto3` optional** — it is imported at module level and fails on Colab:
   ```python
   try:
       import boto3
   except ImportError:
       boto3 = None
   ```
2. **Defaults** — `MODEL_NAME = "google/gemma-2-2b"`, `MAX_SEQ_LENGTH = 2048`,
   `OUTPUT_DIR = "/content/drive/MyDrive/CPT_Project/adapters/gemma2_2b_cpt_v1"`,
   train/val files pointing at `/content/packed_2048/`.
3. **`processor` → `tokenizer`** — Gemma 4 is multimodal and returns a processor; Gemma 2
   is text-only and returns a tokenizer. If `processing_class=` errors on the installed
   TRL/transformers, use `tokenizer=`. *This is the most likely place the script breaks.*
4. **Always pass `--skip_s3_upload`** — there is no bucket; without it the script raises
   at startup.

Everything else stays. It is validated and model-agnostic.

### C.2 Hyperparameters

| Parameter | Value |
|---|---|
| `lora_r` / `lora_alpha` / `lora_dropout` | 16 / 32 / 0.0 |
| `target_modules` | the 7 projections |
| `learning_rate` | **2e-4** — see the contradiction in §G |
| scheduler / warmup / weight decay | cosine / 0.03 / 0.01 |
| `optim` | `adamw_8bit` |
| batch × grad accum | 1 × 8 (effective 8) |
| `bf16` | True — **L4/A100 only** |
| epochs | 1 |
| `save_steps` / `save_total_limit` | 100 / 3 |

### C.3 Pre-flight, in order

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
python cpt_student_bundle/02_training/01_verify_unsloth_env.py
python cpt_student_bundle/02_training/02_validate_packed_dataset.py
```

**If the GPU is a T4, restart the runtime until it isn't.** That is the first move, not a
config change — T4 is Turing and has no bf16, and Gemma 2's logit soft-capping (attention
50.0, final 30.0) makes fp16 overflow a real failure mode over a multi-hour run. The
plan's T4 fallback config is in its Part 7.4 if a T4 is genuinely unavoidable.

### C.4 Smoke test — not optional

```bash
python scripts/05_full_training_gemma2_2b.py --dry_run_samples 64 --num_epochs 1 --skip_s3_upload
```

All four must hold:

| Check | Expected |
|---|---|
| `print_trainable_parameters()` | ~0.5–1.5%. 0% or ~100% → LoRA did not attach |
| Loss | finite and decreasing. Any NaN → stop |
| Peak VRAM | well under the card |
| Adapter saved | `adapter_config.json` + `adapter_model.safetensors` |

**Record seconds per optimizer step**, then `full run = sec_per_step × 1,656`.

Every failure mode in the plan's table except GPU assignment is caught here, in five
minutes — including the Unsloth question in §B.

### C.5 Write the hypothesis — before launching

Dated, one paragraph, committed to git so the date is in the log:

- Held-out legal perplexity → **down**
- Next-token top-1 accuracy → **up**
- Dataset A accuracy (retention) → **roughly flat** (QLoRA freezes the base)

Five minutes, and it converts the chapter from description into *"we predicted X,
observed Y"* — which means a surprising result gets noticed instead of quietly
rationalised.

### C.6 Launch

```bash
python scripts/05_full_training_gemma2_2b.py --skip_s3_upload
```

Expect ~1.5–2 h on A100, 3–6 h on L4. The script resumes from the latest checkpoint in
`OUTPUT_DIR` automatically — a dropped session costs at most 100 steps, and re-running the
same command picks up where it left off with no flags.

**Copy the adapter and logs into the repo the same day.** The Week-1 mistake was leaving
everything Drive-only; the SFT run artifacts are *still* only in Drive.

---

## Timeline coverage

`FYP_Timeline.pdf` names three metrics and four artifacts for Week 2. All are covered:

| Timeline acceptance criterion | Where |
|---|---|
| split manifest with token counts | `reports/03b_split_by_document.txt` |
| dated hypothesis before first run | notebook §7 (commits it) |
| `legal_cloze.json` + generation script | `scripts/build_legal_cloze.py`, notebook §5.6 |
| baseline on all three metrics | notebook §8 (next-token) + §8.1 (cloze) |
| adapter + logs + **loss curve figure** | notebook §9 + §9.1 |
| results table, base vs CPT | notebook §10, §10.2, §11 |
| appendix with Arabic examples | notebook §11.5 (12 prompts) |

**The third metric is doubled deliberately.** The timeline specifies *cloze accuracy*; the
supervisor's newer `Week2_CPT_Plan.md` replaces it with *next-token accuracy* and ships the
script for it. Rather than pick one, run both — they measure different things. Next-token
samples positions at random, where function words and general Arabic dominate and the base
model already scores well; cloze targets legal terms only, concentrating the measurement
where CPT should help. Reporting both satisfies the timeline and gives the gate four
metrics instead of three.

**Report the cloze majority-class baseline.** `statute_term` has four possible answers and
المرسوم is 40% of them, so 40% is what always guessing scores. Only the margin above the
baseline is evidence. `score_legal_cloze.py` prints it per category.

---

## Stage D — Evaluate

### D.1 Baseline first

Score the **base model, no adapter**, on the eval set built in A.4. This is the comparison
point for the entire chapter. Do it while Run A trains if you have a second runtime.

```bash
python cpt_student_bundle/03_evaluation/next_token_eval_gemma.py score \
  --eval_set eval/eval_set_1000.json \
  --model_name google/gemma-2-2b \
  --out eval/results_base.json
```

Then the same with `--adapter .../final_adapter --out eval/results_cpt.json`, then:

```bash
python cpt_student_bundle/03_evaluation/next_token_eval_gemma.py compare \
  --results eval/results_base.json eval/results_cpt.json
```

Add `--use_unsloth` only if §B landed on Path 1.

### D.2 The results table

| Model | Top-1 | Top-5 | Top-10 | Avg NLL | Perplexity |
|---|---|---|---|---|---|
| Base Gemma 2-2B | | | | | |
| CPT Gemma 2-2B | | | | | |
| *(cited)* Base Gemma 4-12B | 31.4% | 50.7% | 61.0% | 6.321 | 556.01 |
| *(cited)* CPT Gemma 4-12B | 62.0% | 80.3% | 86.1% | 3.150 | 23.28 |

The 12B rows come from `cpt_student_bundle/docs/evaluation.md` and are **cited as prior
work in the same project, not re-run**. They give a scale reference for free. Label them
unmistakably — an examiner who thinks you claimed a 12B run will ask where it is.

### D.3 Per-source breakdown — a free finding

Both `score` and `compare` produce it automatically. `evaluation.md` reports Claude Sonnet
on this same corpus: Legislation 44% > Official Gazette 38% > ADL rulings 35% > Related
provisions 34% > Associated study 34% > **Bibliographic rulings 24%**.

If your CPT model reproduces that ordering, it is independent corroboration across two
very different models. If it does not, the divergence is itself worth a paragraph. Either
way it costs nothing.

The corpus makes the mechanism plausible: `bibliographic_rulings` averages ~113 tokens per
document while `associated_studies_ar` averages ~16,800 — a 140× spread. At 2048 tokens a
block holds ~11 whole bibliographic documents separated by EOS, while one study spans ~12
blocks. **State this in the methodology** rather than letting an examiner find it.

### D.4 Retention on Dataset A — read this carefully

The plan's Part 4.4 is ambiguous and the footnote in its results table makes it worse.
The clean comparison is:

| Run | Adapter | What it tells you |
|---|---|---|
| Base zero-shot on Dataset A | none | the retention baseline |
| **CPT zero-shot on Dataset A** | CPT adapter | did CPT damage general ability? |
| SFT (frozen, done) | SFT adapter | 90.42% — a *different* experiment |

**The 90.42% is not the retention baseline.** It comes from base + SFT adapter, a model
trained on the task. The retention question is whether base + *CPT* adapter is worse than
base alone at the same zero-shot task. Both numbers will be low — a 2B model zero-shot on
WSD is weak — but they are comparable to each other, which is the point.

This means **you must run base zero-shot on Dataset A yourself**; you do not have that
number yet. Budget an extra ~30 min.

Mechanically it is nearly free — `infer_model.py` already takes `WSD_ADAPTER_DIR`:

```bash
WSD_BASE_MODEL=google/gemma-2-2b \
WSD_ADAPTER_DIR=/content/drive/MyDrive/CPT_Project/adapters/gemma2_2b_cpt_v1/final_adapter \
python Gemma/Fine-tuning/Dataset-A/infer_model.py
```

Two snags to fix first:

- `infer_model.py` line 49 attaches an adapter unconditionally. For the **base** run you
  need an escape — add `if ADAPTER_DIR and ADAPTER_DIR.lower() != "none":` around it.
- Its `WSD_BASE_MODEL` default is `unsloth/gemma-2-2b` while the CPT side uses
  `google/gemma-2-2b`. These should be identical weights, but **standardise on one
  string** across packing, training and inference, and say which in the thesis.

Do not try to stack the SFT and CPT adapters. Multi-adapter composition in PEFT is a
different experiment with its own confounds — note it as future work.

### D.5 Qualitative

10–15 legal prompts, base output vs CPT output side by side. 3–4 in the thesis, the rest
in an appendix. This is the most persuasive evidence in the chapter for a non-specialist
reader, and it costs an hour.

### D.6 Gate

**Pass: CPT beats base on ≥2 of 3 metrics.**

If it fails, check in this order — the order matters, most failures are #1:

1. Did the adapter actually load at inference? (Print trainable params / compare outputs.)
2. Does the loss curve show learning, or is it flat?
3. Is the eval set truly held out at document level? (It is — asserted — but confirm the file you scored is the one you built.)
4. Only then consider the learning rate.

**One retry, overnight.** Then write up whatever happened. A negative result is still a
thesis result — see §F.

---

## Stage E — Optional Run B (only if Run A passed and time allows)

Add `modules_to_save=["embed_tokens"]`.

**Why it might matter:** Gemma 2 has a 256k vocabulary with tied embeddings, and your
fertility measurement in A.3 shows how heavily Arabic legal terms fragment. Adapting the
embeddings is the mechanism that would help.

**Why it might break:** `embed_tokens` is 590M parameters — an order of magnitude more
than everything else trainable. Wrapping it creates a full trainable copy (+~3.5 GB), and
tied embeddings are a known sharp edge in PEFT.

**Check before committing a session:** `print_trainable_parameters()` after wrapping, then
the 64-sample smoke test. If trainable jumps to ~600M and VRAM holds, proceed.

If both runs complete, A vs B is a **free mini-ablation** on whether embedding adaptation
matters for domain vocabulary — a self-contained finding for one extra hour.

---

## Stage F — Writing

### Chapter 8 — Methodology: CPT

- Corpus: sources, document counts, token counts (the per-source table from the split report)
- **The document-level re-split and why**, with the leakage measurement: 5.9% of val and 5.6% of test would have shared a document with train under a record-level split
- **The `source_id` collision** — 396 IDs shared between two sources, composite key, 43,680 documents not 43,284. This is your own correction to the supplied pipeline; say so plainly.
- Repacking at 2048 with the Gemma 2 tokenizer, and why the supplied 4096 packs were unusable
- The 140× document-length spread and what EOS separators do about it
- Configuration table with an SFT-vs-CPT column, and **why each row differs**
- Which Unsloth path you took (§B) and why

### Chapter 9 — Results: CPT

- Base vs CPT on all metrics, 12B rows clearly marked as cited
- Loss curve
- Per-source breakdown vs the Sonnet ordering
- Retention on Dataset A, with the §D.4 distinction spelled out
- Qualitative samples

### Chapter 10 — Discussion: the dissociation

**This is the thesis conclusion.**

> **SFT changed the task format.** The model learned to select a sense ID from candidates
> already present in the prompt. No knowledge was added — everything required was in the
> context window. This is why 2B matched 8B on Dataset A.
>
> **CPT changed the domain distribution.** The model learned legal vocabulary,
> collocations and register. It gained no task ability — a CPT-only model still cannot
> answer a structured WSD prompt.

Frame as **"what does each objective change"**, never "which is better." They are not
directly comparable — different data, different objective, different metric — and **the
dissociation holds regardless of effect size**. That is what makes the conclusion safe
even if the CPT gain is small.

Cite Gururangan et al. (2020), *Don't Stop Pretraining*, for the DAPT framing.

### Limitations — CPT side

1. No matched-token-budget instruction-SFT control on the legal corpus, so the SFT/CPT comparison is qualitative, not controlled
2. Single run, single seed, single corpus
3. ~27M of ~134M tokens — corpus scale bounds the achievable effect
4. Perplexity and next-token accuracy measure distributional fit, not legal reasoning
5. The 12B comparison is cited, not reproduced under matched conditions

---

## Stage G — Concepts to defend, and the contradictions to resolve

### The SFT vs CPT table — be able to explain every *why*, unprompted

| Parameter | SFT (Dataset A) | CPT (legal) | Why it changes |
|---|---|---|---|
| objective | instruction → response | plain causal LM | no supervision signal; you model the raw text distribution |
| loss | over the whole sequence (unmasked) | over the whole sequence | same here, but for different reasons — say which |
| `target_modules` | 7 attn+MLP projections | same 7 | *the supervisor kept these; the earlier plan wanted `all-linear`* |
| rank | 32 | 16 | *lower, not higher — see below* |
| `modules_to_save` | none | none (optional Run B) | legal Arabic tokenizes badly; letting embeddings move is how new terms become representable |
| learning rate | 2e-4 | **2e-4** | *see the contradiction below* |
| epochs | 3 | 1 | more passes over a domain corpus memorise rather than adapt |
| sequence length | 1024 | 2048 | legal documents are long; CPT wants context |
| packing | none (1 example/sequence) | **yes**, EOS-separated | CPT has no example boundaries to respect |

### ⚠️ Two live contradictions — resolve them before writing Chapter 8

**1. Learning rate.** `FYP_PLAN.md:403` says CPT needs 1e-5 to 5e-5 and calls reusing the
2e-4 SFT rate *"the most likely failure mode of this entire project."* The supervisor's
plan says use **2e-4**, because the 12B run on *this exact corpus* took perplexity from
556.01 to 23.28 at that rate.

**The supervisor's number wins** — it is measured on this data, and measurement beats
general guidance. But the two documents now contradict each other in your own repo, and an
examiner reading both will ask. Write the resolution explicitly: general CPT guidance
favours a lower rate to limit forgetting; this corpus has a validated rate; here is the
retention measurement showing forgetting did not occur.

**2. Rank.** The same earlier plan argued CPT needs rank **64–128** because absorbing a
domain needs more capacity than learning an output format. The supervisor specifies
**r=16** — *lower* than your SFT r=32. Same resolution: r=16 is what worked at 12B on this
corpus. But you should be ready to say why more capacity was not needed, and the honest
answer is that it is an open question your single run does not settle. Put it in
limitations.

### Questions you should be able to answer cold

1. Why is a document-level split necessary when the chunks were already shuffled?
2. How do you know CPT helped rather than memorised? *(Document-level held-out set; the eval set is drawn only from test documents.)*
3. Why does packing need EOS separators, and what breaks without them?
4. Why does CPT use a causal-LM loss when SFT used an instruction format?
5. What exactly does QLoRA freeze, and why does that bound how much forgetting is possible?
6. Why is perplexity alone a weak claim here, and what does next-token accuracy add?
7. What would a matched-token-budget control have looked like, and why didn't you run it?
8. Why did the supplied 4096 packs have to be discarded?
9. Your CPT model is worse at Dataset A than your SFT model — is that a failure? *(No: different objectives, and the CPT model was never trained on the task. That is the dissociation.)*

---

## Stage H — Risk register

| # | Risk | Symptom | Response |
|---|---|---|---|
| 1 | **Unsloth still broken for Gemma 2** | garbage generation, loss starting ≈25 | §B Path 2 — swap in your own loader. Caught by the smoke test. |
| 2 | Colab assigns a T4 | `bf16` error at load | Restart runtime until L4/A100 |
| 3 | `processor` vs `tokenizer` | TypeError at `Trainer` init | Caught by the smoke test; rename |
| 4 | `boto3` import | ImportError on line 1 | Edit 1 in §C.1 |
| 5 | Wrong tokenizer at packing | nonsense loss from step 1 | Verify `MODEL:` line in the packing report |
| 6 | A `*_packed_4096.jsonl` used by mistake | nonsense loss from step 1 | Never use one — all are Gemma 4 |
| 7 | Missing `--append_eos` / `--drop_remainder` | blocks not exactly 2048 | Packing report catches it |
| 8 | Eval set rebuilt between models | base vs CPT not comparable | Build once in A.4, copy to Drive |
| 9 | `--skip_s3_upload` omitted | `ValueError` at startup | Always pass it |
| 10 | Drive fills with checkpoints | save failure mid-run | `save_total_limit=3`; budget 2 GB |
| 11 | Artifacts left Drive-only | nothing to submit | Copy into the repo the same day |

---

## Suggested schedule

The supervisor's plan says 3 days; §B may add half a day if Unsloth has to be replaced.

| Day | Morning | Afternoon | Evening |
|---|---|---|---|
| **1** | Stage A (CPU: splits, packing, fertility, eval set) | §B decision + Stage C.1 edits | C.3–C.5 pre-flight, smoke, hypothesis → **launch** |
| **2** | D.1 baseline (while training, or after) | D.2–D.4 results + retention | D.5 qualitative; D.6 gate |
| **3** | Run B if A passed, else retry | Write Ch. 8–9 | Ch. 10 discussion + §G contradictions |

### The two rules

1. **Nothing in Stage A needs a GPU.** Do not burn compute units on tokenization.
2. **The smoke test is not optional.** Every risk above except GPU assignment is caught by
   64 samples and five minutes.

---

## Compute budget

~7.5 GPU hours total on Colab Pro ≈ **42 of the included 100 compute units** — smoke tests
0.5 h, baseline eval 0.5 h, Run A ~2 h, Run A eval 0.5 h, optional Run B ~2 h, retry
margin ~2 h. No top-ups needed. On the free tier the same work is 25–40 h and does not fit
the window, which is the practical argument for Pro.
