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

1,000 sentences from the El-Razzaz test set, **every item exactly two candidate senses**, so
chance is 50%. Both models scored in the same session on the same card, 24-token budget.

| Model | Accuracy | Macro-F1 | Precision | Recall | n |
|---|---|---|---|---|---|
| Base Gemma 2-2B, zero-shot | 24.9% | 0.1969 | 0.192 | 0.207 | 1,000 |
| **CPT Gemma 2-2B, zero-shot** | **36.5%** | **0.2685** | 0.2617 | 0.2825 | 1,000 |
| *(context)* SFT Gemma 2-2B | 90.42% | 0.8333 | 0.831 | 0.838 | 3,110 |

**The SFT row is context, not a baseline.** It comes from a model trained on this task, on a
different n. Retention asks only whether base + CPT adapter is worse than base alone.

### The decomposition — the key result

Overall accuracy conflates two abilities. Separating them:

| | Commits to an ID | Correct **given an answer** | Overall |
|---|---|---|---|
| Base | **52.5%** (525/1000) | **47.4%** (249/525) | 24.9% |
| CPT | **75.1%** (751/1000) | **48.6%** (365/751) | 36.5% |
| *(context)* SFT | ~100% | ~90% | 90.42% |

**The entire +11.6pp gain is format compliance.** Willingness to emit a parseable sense ID
rose 52.5% → 75.1%. Accuracy among answered items moved 47.4% → 48.6% — 1.2pp against a
pooled SE of ~2.8pp, i.e. nothing. Both 95% CIs contain 50%: **neither model discriminates
above chance.**

Note the base model scores *below* a random guesser (24.9% vs 50%) purely because it produces
no usable answer on 47.5% of items — it is at chance when it does answer, but silent half the
time.

Plausible mechanism: an epoch of EOS-terminated blocks dense with `رقم 1234` patterns made the
model far more likely to terminate and to emit ID-shaped tokens. It did not make it better at
choosing between two glosses.

### What this establishes

- **Retention: no forgetting.** QLoRA froze the base and general ability rose rather than
  fell. The question "did CPT damage the model" is answered emphatically.
- **The dissociation, measured on one axis.** CPT changed *form* without changing *task
  competence*; SFT changed both. This is stronger than inferring the dissociation across
  different metrics, because both objectives are measured here on the same task, the same
  test set and the same decomposition.

### A measurement problem worth reporting

`infer_model.py`'s sense-ID extractor originally matched only a line containing nothing but
the ID — the format the SFT model was *trained* to emit. Run against any untrained model it
reports exactly **0.0%**, while the model is in fact answering. Observed: base Gemma 2-2B
produced *"The correct sense for the target word 'هرب' is Sense ID: 14706, which means"* — the
right answer — and was scored `none`.

The extractor now strips the echoed prompt and falls back to the first candidate ID appearing
in prose. Both SFT formats still extract identically (unit-tested).

**A retention figure from the original extractor would have been false, not merely
pessimistic** — it would have reported 0.0% for both models and hidden the entire finding
above.

---

## 6. Qualitative comparison

12 legal prompts, base output vs CPT output. Both columns come from **one loaded model** —
`disable_adapter()` turns the CPT LoRA off — so the base column is provably the same base
weights rather than a separate load. Greedy decoding, 60 new tokens.
Full pairs: `run_gemma2_2b/qualitative.json`.

**Under greedy decoding the base model's characteristic failure is degenerate repetition,
while the CPT model produces legal structure.** 10 of 12 base continuations collapse into a
loop within 20–30 tokens, against 1 of 12 for CPT. **§6.1 shows that most of that gap is
an artifact of greedy decoding, not of training** — read the examples below for register and
citation form, not as evidence that CPT repaired degenerate generation.

### For the chapter — four clearest contrasts

**(a) وحيث أن الاجتهاد مستقر على** — "whereas settled jurisprudence holds that"

| | |
|---|---|
| Base | loops immediately: `ما يقال في هذا الباب` repeated five times. No content. |
| CPT | cites `المادة 10 من قانون 19/1/1951` and states an actual legal proposition in quotation marks — which court hears an appeal — in the form rulings use. |

**(b) على وزير الداخلية والبلديات** — the most verifiable item

| | |
|---|---|
| Base | lists ministers, then degenerates: `ووزير البيئة` five times. |
| CPT | `بناء على قانون الجمعيات الصادر في 3 آب 1909 ولا سيما المادة السادسة منه` |

The 1909 Ottoman-era Law of Associations **is** the governing Lebanese law for associations,
and `ولا سيما المادة … منه` ("and in particular Article … thereof") is the exact citation
formula Lebanese decrees use. Correct instrument, correct invocation.

**(c) الجريدة الرسمية اللبنانية** — format learned precisely

| | |
|---|---|
| Base | loops `اللجنة الوطنية للتنمية المستدامة`. |
| CPT | reproduces the gazette masthead — year, issue number, page range — then a decree with a plausible subject (transfer of appropriation from budget reserve to the Interior Ministry / Internal Security Forces). |

**(d) لجنة الخدمة المدنية** — correct institutional attachment

| | |
|---|---|
| Base | hallucinates a news item about 121 dismissals, repeating. |
| CPT | places it under `وزارة الداخلية والبلديات` with dated decree citations and a parenthetical giving each decree's purpose. |

### For the appendix — include these deliberately

**(e) يعاقب بالحبس من.** CPT produces the Lebanese penal formula exactly —
`شهر الى سنة وبالغرامة من خمسين الى مئة الف ليرة لبنانية`, imprisonment plus fine — and then
collapses into `كل من يبيع او يبيع او…`.

**(f) تسري أحكام هذا القانون على.** Correct property-law scope language, then loops on
`المؤسسات العامة أو البلديات`.

CPT reduced the looping but did not eliminate it. An appendix in which every example flatters
the model reads as curated; the four quantitative results carry the argument, so these belong
in it.

---

## 6.1 Is the repetition a decoding artifact? — measured

§6's examples use greedy decoding, which is known to induce loops in small models. Rather
than hedge about it, the same 12 prompts were re-run with nucleus sampling
(`top_p=0.9`, `temperature=0.8`, seed 3407), 3 samples per prompt per condition.

**Loop rate** = share of continuations whose most-repeated whitespace 4-gram occurs 3+ times.
The same metric is applied to the greedy outputs, so the rows are comparable.
Script: `scripts/sampled_repetition_check.py`. Data: `run_gemma2_2b/qualitative_sampled.json`.

| Decoding | n per condition | Base loop rate | CPT loop rate |
|---|---|---|---|
| Greedy (`do_sample=False`) | 12 | **83.3 %** (10/12) | 8.3 % (1/12) |
| Nucleus (`top_p=0.9`, `T=0.8`) | 36 | **8.3 %** (3/36) | 0.0 % (0/36) |

**Switching the base model's decoder — with no training whatsoever — drops its loop rate from
83.3 % to 8.3 %.** That single change accounts for almost the whole base-vs-CPT gap seen in
§6. Under matched sampled decoding the remaining difference, 3/36 vs 0/36, is not
statistically distinguishable (Fisher exact, two-tailed *p* = 0.24).

### What this licenses, and what it does not

- **Not supported:** “CPT fixed the base model's degenerate repetition.” The greedy comparison
  confounds training with decoder choice, and the confound explains most of the effect.
- **Supported:** under *identical* decoding, CPT changes *what* the model writes — register,
  citation formulae, statute structure. That is the claim §6's examples illustrate, and it is
  independently established by the quantitative results in §§2–5, which are teacher-forced or
  constrained and therefore involve no decoding at all.

This is the reason the headline metrics are perplexity, next-token accuracy and constrained
cloze rather than generated text: none of them depends on a sampling decision.

### Two caveats to state

1. **Greedy decoding — measured, see §6.1.** The loops are largely a decoding artifact. Do
   not claim CPT fixed the base model's repetition.
2. **Possible memorisation.** `المرسوم رقم 14953 تاريخ 19/7/2005` appears in *both* (c) and
   (d). That may be a genuinely frequent decree in the corpus rather than recall of one
   document, but a 12-prompt sample cannot distinguish the two. **Frame these as illustrative
   of register and citation form, not as evidence of factual recall.** The cloze
   `statute_term` result (§3.1) is the proper evidence for domain knowledge, because it is
   measured on held-out documents against a stated baseline.

---

## 7. Pre-registered predictions vs outcome

Three predictions were committed to git in `CPT_EXECUTION_PLAN.md` §C.5 at
**2026-09-19 11:46** (commit `24063ea`), four days before training began on 2026-09-23.
The commit timestamp is the evidence; nothing here was written after seeing results.

| Predicted | Observed | |
|---|---|---|
| Held-out legal perplexity → **down** | 806.47 → 4.14 | ✅ |
| Next-token top-1 → **up** | 21.3% → 69.8% | ✅ |
| Dataset A retention → **roughly flat** | 24.9% → **36.5%** (+11.6pp) | ❌ **wrong** |

**The failed prediction is the most valuable result of the run.** Retention was expected to be
flat because QLoRA freezes the base; instead general-task accuracy rose by 11.6 points. Having
committed the prediction beforehand is what made that a noticed surprise rather than something
rationalised in hindsight — and following it up produced the decomposition in §5, which is the
strongest evidence for the dissociation thesis.

The reasoning behind the wrong prediction was not wrong, only incomplete: QLoRA did freeze the
base, and discrimination was indeed unchanged (47.4% → 48.6%, n.s.). What went unanticipated
was that CPT would change the model's *output behaviour* — its willingness to terminate and to
emit ID-shaped tokens — enough to move overall accuracy without moving task competence at all.

---

## 8. Gate

Week-2 gate: CPT beats base on ≥ 2 of 3 metric families.

| # | Family | Result | Pass |
|---|---|---|---|
| 1 | Held-out legal perplexity | 806.47 → 4.14 | ✅ |
| 2 | Term prediction — next-token | 21.3% → 69.8% (+48.5pp) | ✅ |
| 2 | Term prediction — cloze | 43.3% → 84.3% (+41.0pp) | ✅ |
| 3 | Dataset A retention | 24.9% -> 36.5%, no forgetting | ✅ |

**Passed on 3 of 3 families.** The gate was already met on the first two before retention
was measured, so retention was informative rather than load-bearing -- and it came back
positive rather than flat.

---

## 9. Limitations

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

## 10. Reproducibility

The document-level split reproduced **byte-identically on three machines** — Windows local,
Colab Linux, RunPod Linux — from the committed scripts at seed 42: same 39,310 / 2,185 /
2,185 documents, same 27,162,160 training tokens, same 13,262 packed blocks, same 5.9% / 5.6%
leakage counterfactual, same 300-item cloze composition.

The `chars / 2.6` planning estimator predicted 27,136,502 training tokens against a true
Gemma 2 count of 27,162,160 — **0.1% error**.
