# CPT results — Gemma 2-2B on the Lebanese legal corpus

Run date **2026-09-23/24**. Hardware: RunPod, NVIDIA RTX PRO 4000 Blackwell (24 GB, sm_120).
Stack: torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2, CUDA 12.8.

All numbers below are measured, not estimated. Artifacts under `run_gemma2_2b/`.

---

## 1. The run

| | Value |
|---|---|
| Base model | `google/gemma-2-2b`, 4-bit NF4, double quant, bf16 compute |
| Method | QLoRA, r=16, α=32, dropout 0.0, 7 projections |
| Objective | causal LM over packed 2048-token blocks, EOS-separated |
| Corpus | 27,162,160 tokens / 13,262 blocks / 39,310 documents |
| Schedule | 1 epoch, lr 2e-4, cosine, 50 warmup steps, adamw_8bit |
| Batch | 1 × grad-accum 8 = 8 blocks = 16,384 tokens per step |
| **Steps** | **1,658** |
| **Wall clock** | **3 h 42 m 21 s** (8.05 s/step) |
| Peak VRAM | 11.79 GB of 23.42 |
| Cost | ~$2.10 at $0.57/hr |

### Loss

| | Value |
|---|---|
| Training loss | **5.8262** (step 1) → **1.3138** (step 1650) |
| Held-out eval loss | **1.7528** → **1.3110** |
| Held-out perplexity | **3.71** |

The eval loss fell monotonically across all 17 evaluation points with **no uptick**. Two
consequences worth stating:

- **No overfitting.** `load_best_model_at_end=True` selected the final checkpoint because it
  had the lowest eval loss, so the saved adapter is genuinely the best one.
- **One epoch is not convergence.** The model was still improving when training stopped. The
  gains below are therefore a **lower bound** on what this corpus can yield. One epoch was
  the supervisor's specification — chosen to adapt rather than memorise — and the curve shows
  that call was conservative, not excessive.

Figure: `run_gemma2_2b/cpt_loss_curve.png`.

---

## 2. Next-token evaluation

1,000 positions sampled from held-out **test documents**, seed 42. Built once; both models
scored against the identical file.

| Model | Top-1 | Top-5 | Top-10 | Avg P | Avg NLL | Perplexity | Avg rank |
|---|---|---|---|---|---|---|---|
| Base Gemma 2-2B | 21.3% | 32.9% | 40.0% | 0.185 | 6.693 | 806.47 | 1705.2 |
| **CPT Gemma 2-2B** | **69.8%** | **82.1%** | **87.7%** | **0.627** | **1.422** | **4.14** | **16.7** |
| **Δ** | **+48.5pp** | **+49.2pp** | **+47.7pp** | +0.443 | −5.271 | −802.3 | −1688.6 |

### Per source

| Source | n | Base | CPT | Δ |
|---|---|---|---|---|
| legislations | 340 | 27.4% | 80.9% | +53.5pp |
| gazette_legal_core | 249 | 25.7% | 78.3% | +52.6pp |
| gazette_section2_sample | 98 | 16.3% | 63.3% | +46.9pp |
| bibliographic_rulings | 264 | 12.1% | 53.4% | +41.3pp |
| adl_rulings | 22 | 22.7% | 54.5% | +31.8pp |
| related_provisions | 22 | 13.6% | 45.5% | +31.8pp |
| associated_studies_ar | 5 | 0.0% | 60.0% | +60.0pp |

**Every source improves.** No source regressed, which rules out the gain being concentrated
in one template-heavy corner of the corpus.

`associated_studies_ar` (n=5), `adl_rulings` and `related_provisions` (n=22) are too small to
interpret individually. The conclusion rests on the three sources with n ≥ 249.

---

## 3. Legal cloze probe

300 items over 253 held-out documents, three categories of 100. Targets are legal terms
only; arbitrary identifiers (decree numbers, dates, case numbers) are deliberately not
masked because nothing predicts them from context.

### 3.1 Constrained (closed-set) scoring — the headline

Candidates ranked by length-normalised log-probability, argmax taken.

| Category | Base | CPT | Δ | Majority baseline | Chance |
|---|---|---|---|---|---|
| collocation | 23.0% | **95.0%** | +72.0pp | 11.0% | 5.0% (20 cands) |
| defined_term | 65.0% | **92.0%** | +27.0pp | 11.0% | 4.2% (24 cands) |
| **statute_term** | **42.0%** | **66.0%** | **+24.0pp** | **40.0%** | 25.0% (4 cands) |
| **Overall** | **43.3%** | **84.3%** | **+41.0pp** | 14.7% | — |

**`statute_term` is the most informative row.** The base model scores 42% against a 40%
majority baseline — i.e. it is doing nothing but reflecting that المرسوم is the commonest
answer. CPT reaches 66%, clearing that baseline by 26 points, on items where the gold
instrument has been removed from the context and a *competing* instrument is often present.
That is direct evidence the model learned which legal instrument a citation names, rather
than only how legal text is shaped.

`collocation` (+72pp) is the largest gain and the least surprising: formulaic phrasing is
what a domain LM absorbs first. `defined_term` rises from an already-strong 65%, so the base
already knew much general Arabic legal vocabulary and CPT's contribution there is narrower.

### 3.2 Free-generation scoring — reported for completeness

| Category | Base | CPT | Majority baseline |
|---|---|---|---|
| collocation | 12.0% | 92.0% | 11.0% |
| defined_term | 32.0% | 82.0% | 11.0% |
| statute_term | **1.0%** | 57.0% | 40.0% |
| Overall | 15.0% | 77.0% | 14.7% |

**Why two scorings, and why the constrained one is primary.** The builder drops any item
whose answer already appears in its own context, to stop the model copying. For
`statute_term` that leaves only items where the gold instrument is *absent* while a competing
instrument usually appears earlier in the sentence — so free generation is primed toward a
wrong answer by construction and the base scored **below chance** (1.0% against a 40%
majority baseline; the model emitted المرسوم 37 times against 40 gold المرسوم answers yet was
correct once, which is only possible if the two are anti-correlated by design).

A base-to-CPT delta measured off that baseline would have been measuring the filter rather
than the model. Constrained scoring removes the artifact and supplies a defined chance level.
**This is a methodology point for Chapter 8, not merely a bug.**

---

## 4. Reference figures — cited, NOT reproduced

From `cpt_student_bundle/docs/evaluation.md`, Gemma 4-12B on the *full* corpus:

| Model | Top-1 | Top-5 | Top-10 | Avg NLL | Perplexity |
|---|---|---|---|---|---|
| Base Gemma 4-12B | 31.4% | 50.7% | 61.0% | 6.321 | 556.01 |
| CPT Gemma 4-12B | 62.0% | 80.3% | 86.1% | 3.150 | 23.28 |

> ⚠️ **These are not comparable to §2.** They come from a different evaluation set, built
> from a different split of the full 134M-token corpus. The numbers here are 1,000 positions
> from this project's 30M-token subset. **Do not claim to have outperformed the 12B run.**
> The defensible statement is that the direction and scale of improvement are consistent with
> the reference.

One legitimate comparison: base Gemma 2-2B enters at perplexity **577.8** (sanity probe)
against **556.01** for base Gemma 4-12B — near-identical starting points on this domain,
which corroborates that the setup measures what the reference measured.

---

## 5. Retention on Dataset A

*(pending — fill in from `report_base_zeroshot.json` and `report_cpt_retention.json`)*

| Model | Accuracy | Macro-F1 | n |
|---|---|---|---|
| Base Gemma 2-2B, zero-shot | | | 1,000 |
| CPT Gemma 2-2B, zero-shot | | | 1,000 |
| *(context)* SFT Gemma 2-2B | 90.42% | 0.8333 | 3,110 |

**The SFT row is context, not a baseline.** It comes from a model trained on this task and
answers a different question. Retention asks only whether base + CPT adapter is worse than
base alone, zero-shot.

### A measurement problem worth reporting

`infer_model.py`'s sense-ID extractor originally matched only a line containing nothing but
the ID — the format the SFT model was *trained* to emit. Run against any untrained model it
reports exactly **0.0%**, while the model is in fact answering, and often answering
correctly. Observed: base Gemma 2-2B produced *"The correct sense for the target word 'هرب'
is Sense ID: 14706, which means"* — the right answer — and was scored `none`.

The extractor now strips the echoed prompt and falls back to the first candidate ID appearing
in prose. Both SFT formats still extract identically (unit-tested).

Two things follow:

1. **A retention figure produced by the original extractor would have been false**, not
   merely pessimistic.
2. It sharpens the Chapter 10 argument. The untrained model can often *identify* the correct
   sense but frequently fails to *state* it usably — roughly 47% of base responses never
   commit to a candidate ID within the generation budget. Discrimination present, format
   absent. **That gap is exactly what SFT supplies and what CPT does not.**

---

## 6. Gate

Week-2 gate: CPT beats base on ≥ 2 of 3 metric families.

| # | Family | Result | Pass |
|---|---|---|---|
| 1 | Held-out legal perplexity | 806.47 → 4.14 | ✅ |
| 2 | Term prediction — next-token | 21.3% → 69.8% (+48.5pp) | ✅ |
| 2 | Term prediction — cloze | 43.3% → 84.3% (+41.0pp) | ✅ |
| 3 | Dataset A retention | pending | — |

**Passed on 2 of 3 families before retention was measured**, so retention is informative
rather than load-bearing: it says whether CPT cost anything, not whether the run succeeded.

---

## 7. Limitations

1. **Register, not reasoning.** Held-out perplexity of 4.14 is low because Lebanese gazette
   and legislation text is extremely formulaic — fixed openers, citation forms and closing
   formulae. Next-token accuracy here measures template regularity as much as legal
   knowledge. The cloze probe is the better evidence of domain vocabulary because it targets
   legal terms specifically.
2. **Not leakage.** The split is document-level with asserted zero overlap, and no held-out
   record's first 1,200 characters appear in training. The eval set is drawn only from test
   documents, and the split reproduced identically on three machines.
3. **One epoch, one seed, one corpus.** No variance estimate.
4. **27.2M of ~134M tokens.** Corpus scale bounds the achievable effect.
5. **No matched-token-budget instruction-SFT control** on the legal corpus, so the SFT/CPT
   comparison is qualitative rather than controlled.
6. **The 12B figures are cited, not reproduced** under matched conditions (§4).
7. **Retention n = 1,000 sentences**, not the full 3,110 of the frozen SFT run — different
   denominators.
8. Small-n sources (`associated_studies_ar` n=5, `adl_rulings` and `related_provisions`
   n=22) are not individually interpretable.

---

## 8. Reproducibility

The document-level split reproduced **byte-identically on three machines** — Windows local,
Colab Linux, RunPod Linux — from the committed scripts at seed 42: same 39,310 / 2,185 /
2,185 documents, same 27,162,160 training tokens, same 13,262 packed blocks, same 5.9% / 5.6%
leakage counterfactual, same 300-item cloze composition.

The `chars / 2.6` planning estimator predicted 27,136,502 training tokens against a true
Gemma 2 count of 27,162,160 — **0.1% error**.
