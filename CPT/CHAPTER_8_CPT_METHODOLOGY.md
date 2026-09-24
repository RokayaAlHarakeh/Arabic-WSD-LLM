# Chapter 8 — Methodology: Continued Pre-Training

## 8.1 Objective and rationale

Chapter 7 established that supervised fine-tuning (SFT) on Dataset A raised Gemma 2-2B to
90.42% accuracy, matching models four times its size. That result is about *task format*: the
model learned to select a sense identifier from candidates already supplied in its prompt.
Every piece of information needed for the decision was present in the context window, so the
adapter had only to learn the mapping from a structured prompt to a structured answer.

Continued pre-training (CPT) addresses the complementary question. Rather than teaching the
model a task, it continues the original pre-training objective — next-token prediction over
raw text — on a corpus drawn from a target domain. Gururangan et al. (2020) term this
domain-adaptive pre-training and show it improves downstream performance across eight tasks in
four domains, the gains scaling with the distance between the pre-training corpus and the
target domain. Lebanese legal Arabic is distant from Gemma 2's pre-training mixture on two
axes at once: language and register.

The two objectives are therefore not competing methods for one problem, and this chapter does
not present them as such. SFT and CPT consume different data, optimise different losses, and
are measured by different metrics. The experiment is designed to determine **what each
objective changes**, and Chapter 10 argues that the answer is a dissociation: SFT confers task
competence without adding knowledge; CPT shifts the domain distribution without conferring
task competence.

## 8.2 Corpus

The corpus is a subset of a Lebanese legal collection supplied by the project supervisor,
distributed as line-delimited JSON. Every record carries a `text` field, a `final_cpt_source`
identifying its collection, and a `source_id` identifying the document it came from. Parsing
the 45,636 records produced no malformed JSON and no empty documents.

The subset totals **27.1 million tokens across 43,680 documents** under the Gemma 2
tokenizer. It is drawn from seven sources of markedly different character:

| Source | Documents | Tokens | Share | Mean tokens/doc |
|---|---|---|---|---|
| `gazette_legal_core` | 10,187 | 10,341,416 | 38.1% | 1,015 |
| `legislations` | 12,084 | 9,671,129 | 35.6% | 800 |
| `adl_rulings` | 655 | 2,786,591 | 10.3% | 4,254 |
| `bibliographic_rulings` | 11,120 | 1,939,981 | 7.1% | 174 |
| `gazette_section2_sample` | 4,275 | 1,846,559 | 6.8% | 432 |
| `related_provisions` | 984 | 435,682 | 1.6% | 443 |
| `associated_studies_ar` | 5 | 115,144 | 0.4% | 23,029 |

(Figures are for the training split; proportions in validation and test match by construction.)

Two properties of this table shape later decisions. First, the corpus is **dominated by two
sources** — the official gazette and the legislation collection supply 73.7% of all tokens, so
any aggregate result is substantially a result about statutory and gazette prose. Section 8.6
therefore specifies a per-source breakdown rather than an aggregate figure alone. Second,
**mean document length varies by a factor of 132**, from 174 tokens for bibliographic ruling
records to 23,029 for the few long scholarly studies. Section 8.4 explains why this makes
document-boundary handling a design decision rather than an implementation detail.

### 8.2.1 Tokenizer fertility

Section 2.3.3 predicted that Arabic fragments more heavily than English under subword
tokenizers, and that specialised registers fragment worse still. This corpus permits a direct
measurement. Gemma 2's fertility — subword tokens per whitespace-delimited word — is:

| Text | Fertility | Relative to legal Arabic |
|---|---|---|
| English (§2.3.3) | 1.163 | 2.009× lower |
| General Arabic (§2.3.3) | 2.079 | 1.124× lower |
| **This legal corpus** | **2.337** | — |

The prediction holds: legal Arabic costs 2.337 tokens per word, 12.4% more than general Arabic
and slightly over twice English. The practical consequence is that a fixed token budget buys
less legal Arabic text than the token count suggests, and that many legal terms have no
single-token representation — they exist only as sequences of subword fragments. This is the
motivation for the optional embedding-training variant discussed in §8.8.

## 8.3 Document-level splitting

### 8.3.1 Why a record-level split is unsafe

The supplied pipeline split the corpus at the record level. Because long documents are stored
as multiple overlapping records — the chunker carries 1,200 characters of context across
boundaries — a record-level split can place one chunk of a document in training and an
overlapping chunk of the same document in the held-out set. The held-out text is then
partially visible during training, and held-out perplexity is optimistically biased by an
unknown amount.

Splitting at the **document** level removes this by construction: every record of a document
lands in the same split, so no held-out text can overlap training text.

The effect was measured rather than assumed. Running a counterfactual record-level split with
identical ratios and seed and counting held-out records sharing a document with training gives:

| Split | Records sharing a document with train | Share |
|---|---|---|
| Validation | 135 of 2,282 | 5.9% |
| Test | 127 of 2,282 | 5.6% |
| **Document-level (used)** | **0** | **0%** |

Roughly one held-out record in eighteen would have been contaminated. This is asserted by an
automated check in the split script, not by inspection.

### 8.3.2 A correction to the supplied pipeline: the identifier collision

The supplied specification groups records into documents by `source_id` alone. Auditing the
corpus before implementing it showed this to be unsafe: **396 `source_id` values occur in more
than one source collection**, and in no case do the colliding records belong to the same
document. Grouping by `source_id` alone would therefore merge unrelated documents from
different collections into single pseudo-documents, yielding 43,284 documents instead of the
correct 43,680.

The implementation uses the composite key `(final_cpt_source, source_id)`. The script fails
loudly on any record with a missing or empty `source_id` rather than silently assigning it to a
default group.

This is a deviation from the supplied plan, adopted after verifying the collision on the actual
data. It is recorded here because the split is the foundation of every held-out number in
Chapter 9: if it were wrong, all of them would be.

### 8.3.3 Configuration

Splitting is by document with a 90/5/5 ratio at seed 42, and the final split contains 39,310
training, 2,185 validation and 2,185 test documents — 90.05%, 4.96% and 4.99% by record count,
and 90.40%, 4.78% and 4.83% by token count. The script emits a manifest recording counts per
source per split, and asserts four acceptance conditions before writing: no document appears in
two splits, no split is empty, every input record is written exactly once, and the realised
ratios fall within tolerance of the requested ones. The split is deterministic; it was
regenerated on three machines (Windows, Colab, RunPod) and produced byte-identical output.

## 8.4 Packing

CPT optimises a causal language-modelling loss over fixed-length blocks. Documents are
concatenated and cut into blocks of **2,048 tokens**, with an end-of-sequence token appended to
each document before concatenation and the trailing partial block discarded.

**Why EOS separators matter.** Without them, the model is trained to predict the first token of
one document from the last tokens of an unrelated one. It learns a transition that does not
exist in the data-generating process. The EOS token gives it an explicit signal that context is
ending, which matters especially here: with a 132× spread in document length, a single
2,048-token block may contain one fragment of a long ruling or a dozen complete bibliographic
records.

**Why 2,048 rather than the supplied 4,096.** The supplied packs were discarded for a reason
unrelated to length: they had been tokenized with the **Gemma 4 vocabulary**, whose token
identifiers do not correspond to Gemma 2's. Training Gemma 2 on them would have produced
meaningless loss from the first step, in a way easily mistaken for a training bug. The corpus
was therefore repacked from the document-level splits with the Gemma 2 tokenizer, at 2,048
tokens to fit the available memory. As a safeguard, the packing report records the tokenizer
name in its header and this was verified before training.

Packing the training split yielded **13,262 blocks** from 27,162,160 tokens, discarding 1,584
remainder tokens — 0.006% of the corpus. Every block is exactly 2,048 tokens.

## 8.5 Model and training configuration

The base model is `google/gemma-2-2b`, loaded in 4-bit NF4 with double quantisation and
bfloat16 compute, adapted with QLoRA. The base weights are frozen and quantised; only the
low-rank adapter matrices are trained.

The intended framework, Unsloth, could not be used: the available build was incompatible with
the transformers version required by the Blackwell-generation GPU, and an earlier run on that
stack had produced a corrupted adapter. Training therefore uses `transformers`, `peft` and
`bitsandbytes` directly. This costs some throughput and changes no aspect of the method.

| Parameter | SFT (Chapter 7) | CPT (this chapter) | Why it differs |
|---|---|---|---|
| Objective | instruction → response | causal LM over raw text | No supervision signal exists; the target is the text distribution itself |
| Data unit | one example per sequence | packed 2,048-token blocks | CPT has no example boundaries to respect |
| Sequence length | 1,024 | 2,048 | Legal documents are long; the objective benefits from context |
| LoRA rank | 32 | **16** | Supervisor's specification, validated at 12B on this corpus — see §8.7 |
| LoRA alpha / dropout | 32 / 0.05 | 32 / 0.0 | No dropout: a single epoch over 27M tokens is not an overfitting regime |
| Target modules | 7 attention + MLP projections | same 7 | Unchanged, so the comparison isolates the objective |
| `modules_to_save` | none | none | Embeddings frozen; see §8.8 |
| Learning rate | 2e-4 | **2e-4** | Validated on this corpus — see §8.7 |
| Schedule | cosine | cosine, 3% warmup, weight decay 0.01 | Standard |
| Optimiser | `adamw_8bit` | `adamw_8bit` | Memory |
| Epochs | 3 | **1** | Repeated passes over a domain corpus memorise rather than adapt |
| Batch × accumulation | — | 1 × 8 (16,384 tokens/step) | Largest effective batch fitting 24 GB |

Training ran for **1,658 optimiser steps in 3 h 42 m** on an NVIDIA RTX PRO 4000 Blackwell
(24 GB), peaking at 11.79 GB of VRAM, at a compute cost of approximately $2.10. The software
stack was torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, bitsandbytes 0.50.2.

### 8.5.1 Verification before the full run

Three checks preceded the 3.7-hour run, each cheap enough to be worth its cost:

1. **Quantisation support.** A 4-bit load and single forward pass, run before any data
   preparation, confirmed that `bitsandbytes` supported the Blackwell architecture. Cost: $0.12
   against an hour of preparation that would otherwise have been wasted.
2. **Initial-loss probe.** At step zero the loss must fall below ln(vocab_size) — the loss of a
   uniform random predictor. A model fed corrupted or mis-tokenized data scores at or above
   this value. The training script aborts if the probe fails, which would have caught the
   Gemma 4 vocabulary mismatch described in §8.4.
3. **Smoke test.** A 20-step run at full configuration, verifying that loss decreases, memory
   fits, and checkpoints write correctly.

## 8.6 Evaluation design

All evaluation metrics, their baselines and their acceptance thresholds were fixed **before
training began** (§8.9). Four measures were specified, each with an explicit baseline, because
no single one is sufficient.

**1. Held-out perplexity** on the test split. The direct measure of the training objective. It
is reported first and trusted least: perplexity falls whenever the model's output distribution
moves toward the corpus, including through purely superficial adaptation to formatting and
orthography. A large drop is necessary for the experiment to have worked, but not sufficient
for any claim about knowledge.

**2. Next-token accuracy** on 1,000 positions sampled from test documents. Teacher-forced top-1
accuracy over positions sampled once at seed 42. The evaluation set was built once and reused
for both the base and CPT models; rebuilding it between conditions would resample the positions
and silently destroy the comparison. It adds to perplexity a concrete, interpretable quantity:
the share of positions the model gets exactly right.

**3. Legal cloze probe**, 300 items in three categories — statutory terminology, legal
collocations, and defined terms — generated automatically from **test documents only**, so no
probe item appears in training. Items are scored two ways. *Free generation* asks the model to
produce the masked span. *Constrained scoring* ranks a fixed candidate set by likelihood and
selects the highest, which yields a well-defined chance baseline and cannot be defeated by
formatting failures. Constrained scoring is the headline number, for reasons §8.6.1 explains.
Arbitrary identifiers such as decree numbers are excluded by design: recalling them would
measure memorisation, not domain competence.

**4. Retention on Dataset A.** The CPT model, with no task training, is evaluated on the SFT
task to test for catastrophic forgetting. Chapter 9 reports this as a decomposition rather than
a single accuracy, separating *whether the model produced a parseable answer* from *whether
that answer was correct* — the two can move independently, and conflating them is the most
likely way to misread this metric.

Majority-class and chance baselines are reported alongside every figure, since an accuracy
figure with no baseline is uninterpretable.

### 8.6.1 Metric choice and decoding

Metrics 1, 2 and 3-constrained are **teacher-forced or likelihood-ranked**: they involve no
sampling and no decoding strategy. This was deliberate. Free-text generation quality depends
heavily on the decoder, and Chapter 9 §6.1 demonstrates the size of that dependence — switching
the *base* model from greedy decoding to nucleus sampling, with no training whatsoever, reduces
its degenerate-repetition rate from 83.3% to 8.3%. A comparison based on generated text would
have confounded the training intervention with a decoder setting. The qualitative examples in
Chapter 9 are accordingly presented as illustrations of register and citation form, not as
evidence of a capability difference.

## 8.7 Two specification conflicts and their resolution

The supervisor's specification conflicts with general CPT guidance, including guidance recorded
earlier in this project's own planning documents, on two parameters. Both are recorded here
with their resolution.

**Learning rate.** The project's earlier planning document specifies 1e-5 to 5e-5 for CPT and
identifies reuse of the 2e-4 SFT rate as, in its own words, the most likely failure mode of the
project — the concern being catastrophic forgetting. The supervisor specifies 2e-4, on the
grounds that a 12B run on *this exact corpus* reduced perplexity from 556.01 to 23.28 at that
rate.

The supervisor's figure is a measurement on the target data, and measurement outranks general
guidance. The specification was followed, and because the disagreement was about forgetting,
the retention metric (§8.6, metric 4) was added specifically to test it. Chapter 9 reports the
outcome: Dataset A accuracy rose rather than fell, so no forgetting occurred at this rate. That
evidence is limited to the base-to-CPT direction actually measured; it does not establish the
rate is safe in a pipeline where CPT is followed by SFT, which was not run.

**LoRA rank.** The same earlier document argues that absorbing a domain requires more capacity
than learning an output format, and recommends rank 64–128. The supervisor specifies rank 16 —
*lower* than the rank 32 used for SFT. Again the specification was followed, on the same
grounds.

Unlike the learning rate, **this conflict is not resolved by the evidence collected.** A
natural argument would be that the training loss never plateaued, so capacity was not the
binding constraint. That argument does not survive inspection: the cosine schedule decays the
learning rate to approximately 1.5e-8 by step 1650, four orders of magnitude below peak, so the
flat tail of the loss curve is what the schedule produces by construction and carries no
information about capacity. Whether rank 16 limited the achievable adaptation is an open
question that a single run at a single rank cannot answer. It is recorded in the limitations.

## 8.8 Design decisions deliberately not taken

**Embeddings were not trained.** The fertility measurement (§8.2.1) shows legal terms fragment
into multiple subword units, and allowing the embedding and output layers to train
(`modules_to_save`) is the standard way to let new terminology acquire dedicated
representations. It was not done, for two reasons: it substantially increases trainable
parameters and memory, and it introduces a second change alongside the adapter, so any
difference in outcome could not be attributed to either. It is specified as a follow-up run.

**No matched-token-budget control.** A fully controlled comparison of SFT against CPT would
train an instruction-SFT model on the legal corpus at an equal token budget, isolating the
objective from the data. This was not run, and it is the principal limitation of the
SFT-versus-CPT comparison, which is therefore qualitative rather than controlled. Chapter 10's
dissociation argument is constructed so as not to depend on it: it claims the two objectives
change *different things*, not that one outperforms the other.

**Single run, single seed.** No variance estimate is available. Effect sizes should be read as
point estimates from one run.

## 8.9 Pre-registration

Because the evaluation offered several metrics and a single run, the analysis was vulnerable to
selecting whichever metric produced the most favourable result after seeing the outcomes. To
prevent this, three explicit predictions and a pass/fail gate were committed to the project's
version control **on 19 September 2026, four days before training began on 23 September**:

1. Held-out perplexity falls substantially.
2. Next-token accuracy rises.
3. Dataset A retention stays roughly flat, since the CPT model receives no task training.

The commit timestamp is verifiable in the repository history and precedes the first training
step. Chapter 9 reports all three against their outcomes, including the one that was wrong.

## 8.10 Reproducibility

The corpus split is deterministic at seed 42 and was verified byte-identical across three
platforms. The evaluation set and the cloze probe were each generated once and reused across
both models. Every script, configuration, evaluation output, the trained adapter and the full
trainer state are in the project repository. The complete run, including all evaluation, cost
approximately $5.40 of GPU time.
