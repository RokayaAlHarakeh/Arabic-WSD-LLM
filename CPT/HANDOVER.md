# Continued Pre-Training on the Lebanese Legal Corpus — Handover

**Rokaya Al Harakeh** · Gemma 2-2B · QLoRA · one epoch over 27.2M tokens

---

## 1. Summary

The base model was adapted to the supplied legal corpus with QLoRA under a causal
language-modelling objective, and benchmarked against the untrained base on four measures
fixed in writing before the run. **All four improved; the acceptance gate passed 3 of 3
metric families.** Total compute cost was approximately $5.40.

| Metric | Base | CPT | Change |
|---|---|---|---|
| Held-out perplexity | 806.47 | **4.14** | −802 |
| Next-token top-1 | 21.3% | **69.8%** | +48.5 pp |
| Legal cloze, constrained | 43.3% | **84.3%** | +41.0 pp |
| Dataset A retention (no forgetting check) | 24.9% | **36.5%** | +11.6 pp |

The cloze improvement is significant under a paired McNemar exact test, *p* < 10⁻¹²
(136 items gained, 13 lost, same 300 items scored for both models).

## 2. The three findings that matter

**Domain adaptation worked, and not only on boilerplate.** Next-token accuracy improved on
**every one of the seven corpus sources**, including `bibliographic_rulings`, the shortest and
least templated (+41.3 pp). The gain is not confined to formulaic gazette prose.

**The model learned instrument identity, not just register.** The cloze probe's
`statute_term` category is the strict test: the correct legal instrument is removed from the
context and a competing one is often present. The base model scores 42.0% against a 40.0%
majority baseline — i.e. it knows nothing beyond "المرسوم is commonest". CPT reaches 66.0%.
72% of the remaining errors are in this category, which is the expected residual: phrasing is
absorbed first, instrument identity is closer to factual knowledge and is bounded by corpus
scale.

**No catastrophic forgetting.** This was the stated risk of running at 2e-4 rather than the
1e-5–5e-5 that general CPT guidance recommends. General-task accuracy **rose** rather than
fell. The 2e-4 rate specified for this corpus is vindicated for the base→CPT direction; it has
not been tested in a CPT-then-SFT pipeline.

## 3. What is *not* established

- **The gain in general-task accuracy is format, not competence.** Decomposing the retention
  result: willingness to emit a parseable answer rose 52.5% → 75.1%, while accuracy among
  answered items moved 47.4% → 48.6% — nothing, against a pooled SE of 2.8 pp. CPT made the
  model *answer*, not *reason*.
- **One epoch over 27.2M of the ~134M available tokens.** Every figure is a lower bound.
- **One run, one seed, rank 16.** Whether a higher LoRA rank would absorb more of the domain
  is untested; the loss curve cannot settle it, because the cosine schedule drives the
  learning rate to ~1.5e-8 by the final steps and flattens the tail by construction.
- **Generation quality is not evidence here.** Side-by-side samples look dramatic, but 90% of
  the apparent difference in degenerate repetition is attributable to greedy decoding: under
  nucleus sampling the *base* model's loop rate falls from 83.3% to 8.3% with no training at
  all. The headline metrics above are teacher-forced or likelihood-ranked and never sample.

## 4. Using the adapter

`run_gemma2_2b/final_adapter/` is a standard PEFT LoRA adapter for `google/gemma-2-2b`
(r=16, α=32, the 7 attention and MLP projections, no embedding training).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b", quantization_config=bnb,
                                             device_map={"": 0}, attn_implementation="sdpa")
model = PeftModel.from_pretrained(model, "final_adapter").eval()
```

`model.disable_adapter()` gives the base model back from the same load, which is how every
base-vs-CPT comparison in this work was run — it removes the risk of comparing against a
differently configured load.

## 5. Reproducing

The document-level split is deterministic at seed 42 and reproduced byte-identically on three
machines. Run order and exact commands are in `RUNPOD_RUNBOOK.md`; the full method is
Chapter 8 of the thesis.

One correction to the supplied pipeline is worth noting for any future run on this corpus:
**grouping records into documents by `source_id` alone is unsafe.** 396 identifiers are shared
across source collections, and in no case do the colliding records belong to the same
document. The composite key `(final_cpt_source, source_id)` is required; without it, unrelated
documents merge and the held-out guarantee weakens. A record-level split would additionally
have leaked 5.9% of validation and 5.6% of test.

## 6. One decision needed from WhiteStork

The project repository is currently **public** on GitHub, and the committed evaluation
artifacts contain material derived from the supplied corpus:

| File | Exposure |
|---|---|
| `run_gemma2_2b/legal_cloze.json` | ~179,000 characters of corpus text, as 300 context windows from held-out documents |
| `run_gemma2_2b/eval_set_1000.json` | 1,000 tokenised context windows — token IDs, but decodable back to text |
| `run_gemma2_2b/final_adapter/` | 80 MB adapter trained on the corpus |

The underlying Lebanese gazette and legislation texts are published public law, so the content
is not confidential in itself. The curated and cleaned corpus is WhiteStork's work product,
which is a separate question. **Please confirm whether these artifacts may remain in a public
repository**, or whether the repository should be made private and the derived files removed
from its history. No corpus file, split or packed dataset is committed — only the evaluation
artifacts listed above.
