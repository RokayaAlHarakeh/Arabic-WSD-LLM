# -*- coding: utf-8 -*-
"""
Chapters 8-11: the continued pre-training half of the report.

Kept in its own module for the same reason Chapters 2-7 live in
build_chapters.py -- build_docx.py stays readable.

Every number here is measured. The sources are:
    CPT/RESULTS.md                        results
    CPT/reports/03b_split_by_document.txt split manifest
    CPT/reports/train_packed_2048.txt     packing
    CPT/run_gemma2_2b/trainer_state.json  loss curve
"""


# ======================================================================
#  CHAPTER 8 -- METHODOLOGY: CONTINUED PRE-TRAINING
# ======================================================================
def build_ch8(D):
    D.h1("Chapter 8 — Methodology: Continued Pre-Training")
    D.p("Chapters 6 and 7 adapted Gemma 2-2B to a task. This chapter adapts it to a **domain**. The "
        "objective, the data, the evaluation and most of the hyperparameters differ, and this chapter "
        "states each difference together with the reason for it. It closes with two points that matter "
        "more than any individual setting: the two places where the specification followed here "
        "contradicts general guidance, and the fact that every metric and threshold was fixed in writing "
        "before training began.")

    # ---- 8.1 -------------------------------------------------------
    D.h2("8.1  Objective and rationale")
    D.p("Chapter 7 established that supervised fine-tuning raised a 2-billion-parameter model to 90.42% "
        "on Dataset A, matching published models four times its size. That result concerns **task "
        "format**: the model learned to select a sense identifier from candidates already supplied in "
        "its prompt. Every piece of information required for the decision was present in the context "
        "window, so the adapter had only to learn a mapping from a structured prompt to a structured "
        "answer.")
    D.p("Continued pre-training (CPT) addresses the complementary question. Rather than teaching a task, "
        "it continues the original pre-training objective — next-token prediction over raw text — on a "
        "corpus drawn from a target domain. Gururangan et al. (2020) term this *domain-adaptive "
        "pre-training* and show it improves downstream performance across eight tasks in four domains, "
        "with gains scaling in the distance between the pre-training corpus and the target domain. "
        "Lebanese legal Arabic is distant from Gemma 2's pre-training mixture on two axes at once: "
        "language and register.")
    D.keypoint("The two objectives are not competing methods for one problem, and this report does not "
               "present them as such. They consume different data, optimise different losses and are "
               "measured by different metrics. The experiment asks **what each objective changes**. "
               "Chapter 10 argues the answer is a dissociation.")

    # ---- 8.2 -------------------------------------------------------
    D.h2("8.2  The corpus")
    D.p("The corpus is a subset of a Lebanese legal collection supplied by the industrial supervisor, "
        "distributed as line-delimited JSON. Each record carries a `text` field, a `final_cpt_source` "
        "naming its collection, and a `source_id` identifying the document it came from. Parsing the "
        "45,636 records produced no malformed JSON and no empty documents.")
    D.p("The subset totals **27.1 million tokens across 43,680 documents** under the Gemma 2 tokenizer, "
        "drawn from seven sources of markedly different character.")
    D.table([
        ["Source", "Documents", "Tokens", "Share", "Mean tokens/doc"],
        ["gazette_legal_core", "10,187", "10,341,416", "38.1%", "1,015"],
        ["legislations", "12,084", "9,671,129", "35.6%", "800"],
        ["adl_rulings", "655", "2,786,591", "10.3%", "4,254"],
        ["bibliographic_rulings", "11,120", "1,939,981", "7.1%", "174"],
        ["gazette_section2_sample", "4,275", "1,846,559", "6.8%", "432"],
        ["related_provisions", "984", "435,682", "1.6%", "443"],
        ["associated_studies_ar", "5", "115,144", "0.4%", "23,029"],
    ], widths=[5.0, 2.4, 2.8, 1.9, 2.7], align_right={1, 2, 3, 4})
    D.caption("Composition of the training split by source. Validation and test match by construction.")
    D.p("Two properties of this table drive later decisions. First, the corpus is **dominated by two "
        "sources**: the official gazette and the legislation collection together supply 73.7% of all "
        "tokens, so any aggregate figure is substantially a figure about statutory and gazette prose. "
        "Section 8.6 therefore specifies a per-source breakdown rather than an aggregate alone. Second, "
        "**mean document length varies by a factor of 132** — from 174 tokens for bibliographic ruling "
        "records to 23,029 for the handful of long scholarly studies — which makes document-boundary "
        "handling a design decision rather than an implementation detail (§8.4).")

    D.h3("8.2.1  Tokenizer fertility")
    D.p("Section 2.3.3 predicted that Arabic fragments more heavily than English under subword "
        "tokenization, and that specialised registers fragment worse still. This corpus permits a direct "
        "test. Fertility — subword tokens per whitespace-delimited word — was measured for Gemma 2 on a "
        "sample of the corpus and compared against the figures given in Chapter 2.")
    D.table([
        ["Text", "Fertility (tokens/word)", "Relative to legal Arabic"],
        ["English (§2.3.3)", "1.163", "2.009× lower"],
        ["General Arabic (§2.3.3)", "2.079", "1.124× lower"],
        ["**This legal corpus**", "**2.337**", "—"],
    ], widths=[5.6, 4.2, 4.2], align_right={1, 2})
    D.caption("Gemma 2 tokenizer fertility on Lebanese legal Arabic against the Chapter 2 reference figures.")
    D.p("The prediction holds. Legal Arabic costs 2.337 tokens per word — 12.4% more than general Arabic "
        "and slightly over twice English. Two consequences follow: a fixed token budget buys less legal "
        "text than a raw word count suggests, and many legal terms have no single-token representation, "
        "existing only as sequences of subword fragments. The second is the motivation for the embedding "
        "variant discussed in §8.8.")

    # ---- 8.3 -------------------------------------------------------
    D.h2("8.3  Document-level splitting")
    D.h3("8.3.1  Why a record-level split is unsafe")
    D.p("The supplied pipeline split the corpus at the record level. Because long documents are stored as "
        "several overlapping records — the chunker carries 1,200 characters of context across each "
        "boundary — a record-level split can place one chunk of a document in training and an "
        "overlapping chunk of the *same* document in the held-out set. Held-out text is then partially "
        "visible during training, and held-out perplexity is optimistically biased by an unknown amount.")
    D.p("Splitting at the **document** level removes this by construction: every record of a document "
        "lands in the same split, so no held-out text can overlap training text. The size of the effect "
        "was measured rather than assumed, by running a counterfactual record-level split at identical "
        "ratios and seed and counting held-out records that share a document with training.")
    D.table([
        ["Split", "Records sharing a document with train", "Share"],
        ["Validation", "135 of 2,282", "5.9%"],
        ["Test", "127 of 2,282", "5.6%"],
        ["**Document-level split (used)**", "**0**", "**0%**"],
    ], widths=[5.0, 6.0, 3.0], align_right={1, 2})
    D.caption("Contamination avoided by splitting at the document level, measured counterfactually.")
    D.p("Roughly one held-out record in eighteen would have been contaminated. This is asserted by an "
        "automated check inside the split script, not established by inspection.")

    D.h3("8.3.2  A correction to the supplied pipeline: the identifier collision")
    D.p("The supplied specification groups records into documents by `source_id` alone. Auditing the "
        "corpus before implementing it showed this to be unsafe: **396 `source_id` values occur in more "
        "than one source collection**, and in no case do the colliding records belong to the same "
        "document. Grouping by `source_id` alone would therefore merge unrelated documents from "
        "different collections into single pseudo-documents, yielding 43,284 documents instead of the "
        "correct 43,680.")
    D.p("The implementation uses the composite key `(final_cpt_source, source_id)`, and fails loudly on "
        "any record with a missing or empty identifier rather than silently assigning it to a default "
        "group.")
    D.keypoint("This is a deliberate deviation from the supplied plan, adopted after verifying the "
               "collision on the actual data. It is recorded because the split underpins every held-out "
               "number in Chapter 9: had it been wrong, all of them would be.")

    D.h3("8.3.3  Configuration and verification")
    D.p("Splitting is by document at a 90/5/5 ratio with seed 42, giving 39,310 training, 2,185 "
        "validation and 2,185 test documents — 90.05 / 4.96 / 4.99 percent by record count and "
        "90.40 / 4.78 / 4.83 percent by token count. The script emits a manifest recording counts per "
        "source per split and asserts four conditions before writing: no document appears in two splits, "
        "no split is empty, every input record is written exactly once, and the realised ratios fall "
        "within tolerance of those requested.")
    D.p("The split is deterministic. It was regenerated on three machines — Windows, Google Colab and the "
        "RunPod Linux instance — and produced byte-identical output each time.")

    # ---- 8.4 -------------------------------------------------------
    D.h2("8.4  Packing")
    D.p("CPT optimises a causal language-modelling loss over fixed-length blocks. Documents are "
        "concatenated and cut into blocks of **2,048 tokens**, with an end-of-sequence token appended to "
        "each document before concatenation and the trailing partial block discarded.")
    D.p("**Why the EOS separators matter.** Without them the model is trained to predict the first token "
        "of one document from the last tokens of an unrelated one — a transition that does not exist in "
        "the data-generating process. The EOS token supplies an explicit signal that context is ending. "
        "This matters more here than it would on a uniform corpus: with a 132-fold spread in document "
        "length, one 2,048-token block may hold a single fragment of a long ruling or a dozen complete "
        "bibliographic records.")
    D.p("**Why 2,048 rather than the supplied 4,096.** The supplied packs were discarded for a reason "
        "unrelated to length: they had been tokenized with the **Gemma 4 vocabulary**, whose token "
        "identifiers do not correspond to Gemma 2's. Training Gemma 2 on them would have produced "
        "meaningless loss from the first step, in a way easily mistaken for a bug in the training code. "
        "The corpus was repacked from the document-level splits with the Gemma 2 tokenizer, at 2,048 "
        "tokens to fit the available memory. As a safeguard the packing report records the tokenizer "
        "name in its header, and this was checked before training.")
    D.p("Packing the training split produced **13,262 blocks** from 27,162,160 tokens, discarding 1,584 "
        "remainder tokens — 0.006% of the corpus. Every block is exactly 2,048 tokens long.")

    # ---- 8.5 -------------------------------------------------------
    D.h2("8.5  Model and training configuration")
    D.p("The base model is `google/gemma-2-2b`, loaded in 4-bit NF4 with double quantization and bfloat16 "
        "compute, adapted with QLoRA exactly as described in Chapter 4. The base weights are frozen and "
        "quantized; only the low-rank adapter matrices are trained.")
    D.p("The intended framework, Unsloth, could not be used: the available build was incompatible with "
        "the transformers version required by the Blackwell-generation GPU, and an earlier run on that "
        "stack had produced a corrupted adapter. Training therefore drives `transformers`, `peft` and "
        "`bitsandbytes` directly. This costs some throughput and changes no aspect of the method.")
    D.table([
        ["Parameter", "SFT (Ch. 6)", "CPT (this chapter)", "Why it differs"],
        ["Objective", "instruction → response", "causal LM over raw text",
         "No supervision signal exists; the target is the text distribution itself"],
        ["Data unit", "one example per sequence", "packed 2,048-token blocks",
         "CPT has no example boundaries to respect"],
        ["Sequence length", "1,024", "2,048", "Legal documents are long; the objective benefits from context"],
        ["LoRA rank", "32", "**16**", "Supervisor's specification, validated at 12B on this corpus (§8.7)"],
        ["LoRA α / dropout", "32 / 0.05", "32 / 0.0",
         "One epoch over 27M tokens is not an overfitting regime"],
        ["Target modules", "7 attention + MLP projections", "the same 7",
         "Held constant so the comparison isolates the objective"],
        ["`modules_to_save`", "none", "none", "Embeddings frozen — see §8.8"],
        ["Learning rate", "2e-4", "**2e-4**", "Validated on this corpus (§8.7)"],
        ["Schedule", "cosine", "cosine, 3% warmup, decay 0.01", "Standard"],
        ["Optimiser", "`adamw_8bit`", "`adamw_8bit`", "Memory"],
        ["Epochs", "3", "**1**", "Repeated passes over a domain corpus memorise rather than adapt"],
        ["Batch × accumulation", "—", "1 × 8  (16,384 tokens/step)", "Largest effective batch fitting 24 GB"],
    ], widths=[3.0, 3.0, 3.6, 5.4])
    D.caption("Training configuration, against the supervised fine-tuning run of Chapter 6.")
    D.p("Training ran for **1,658 optimiser steps in 3 hours 42 minutes** on an NVIDIA RTX PRO 4000 "
        "Blackwell (24 GB), peaking at 11.79 GB of video memory, at a compute cost of roughly $2.10. The "
        "software stack was torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0 and bitsandbytes 0.50.2.")

    D.h3("8.5.1  Verification before the full run")
    D.p("Three checks preceded the 3.7-hour run, each cheap enough to justify itself:")
    D.numbered("**Quantization support.** A 4-bit load and a single forward pass, run before any data "
               "preparation, confirmed that `bitsandbytes` supported the Blackwell architecture — a $0.12 "
               "test against an hour of preparation that would otherwise have been wasted.")
    D.numbered("**Initial-loss probe.** At step zero the loss must fall below ln(vocab_size), the loss of "
               "a uniform random predictor. A model fed corrupted or mis-tokenized data scores at or "
               "above that value. The training script aborts if the probe fails, which is precisely the "
               "check that would have caught the Gemma 4 vocabulary mismatch of §8.4.")
    D.numbered("**Smoke test.** A 20-step run at full configuration, confirming that loss decreases, "
               "memory fits and checkpoints write correctly.")

    # ---- 8.6 -------------------------------------------------------
    D.h2("8.6  Evaluation design")
    D.p("All metrics, their baselines and the pass/fail threshold were fixed **before training began** "
        "(§8.9). Four measures were specified, each with an explicit baseline, because no single one is "
        "sufficient.")
    D.p("**1. Held-out perplexity** on the test split — the direct measure of the training objective. It "
        "is reported first and trusted least: perplexity falls whenever the output distribution moves "
        "toward the corpus, including through purely superficial adaptation to formatting and "
        "orthography. A large drop is necessary for the experiment to have worked and sufficient for no "
        "claim about knowledge.")
    D.p("**2. Next-token accuracy** on 1,000 positions sampled from test documents, teacher-forced, "
        "sampled once at seed 42. The evaluation file was built once and reused for both models; "
        "rebuilding it between conditions would resample the positions and silently destroy the "
        "comparison. It adds to perplexity an interpretable quantity: the share of positions the model "
        "gets exactly right.")
    D.p("**3. A legal cloze probe** of 300 items in three categories — statutory terminology, legal "
        "collocations and defined terms — generated automatically from **test documents only**, so no "
        "probe item appears in training. Items are scored two ways: *free generation*, where the model "
        "produces the masked span, and *constrained scoring*, where a fixed candidate set is ranked by "
        "length-normalised log-probability and the argmax taken. Constrained scoring is the headline "
        "number for the reasons given in §9.3.2. Arbitrary identifiers such as decree numbers are "
        "excluded by design: recalling them would measure memorisation, not domain competence.")
    D.p("**4. Retention on Dataset A.** The CPT model, which has received no task training, is evaluated "
        "on the Chapter 7 task to test for catastrophic forgetting. Chapter 9 reports it as a "
        "decomposition rather than a single accuracy, separating *whether the model produced a parseable "
        "answer* from *whether that answer was correct* — the two move independently, and conflating "
        "them is the easiest way to misread this metric.")
    D.p("Majority-class and chance baselines accompany every figure, since an accuracy without a floor is "
        "uninterpretable.")

    D.h3("8.6.1  Why the headline metrics do not involve decoding")
    D.p("Metrics 1, 2 and 3-constrained are **teacher-forced or likelihood-ranked**: none of them samples, "
        "and none depends on a decoding strategy. This was deliberate. Free-text generation quality "
        "depends heavily on the decoder, and §9.6.1 quantifies how heavily — switching the *base* model "
        "from greedy decoding to nucleus sampling, with no training at all, cuts its degenerate-repetition "
        "rate from 83.3% to 8.3%.")
    D.keypoint("A comparison built on generated text would have confounded the training intervention with "
               "a decoder setting. The qualitative examples in §9.6 are accordingly presented as "
               "illustrations of register and citation form, never as evidence of a capability difference.")

    # ---- 8.7 -------------------------------------------------------
    D.h2("8.7  Two specification conflicts, and their resolution")
    D.p("The configuration in §8.5 conflicts with general CPT guidance — including guidance recorded "
        "earlier in this project's own planning documents — on two parameters. Both are recorded here "
        "with their resolution, because a reader comparing the two documents will otherwise find the "
        "contradiction unaddressed.")
    D.h3("8.7.1  Learning rate")
    D.p("The project's earlier planning document specifies 1e-5 to 5e-5 for CPT and identifies reuse of "
        "the 2e-4 supervised rate as, in its own words, the most likely failure mode of the entire "
        "project — the concern being catastrophic forgetting. The supervisor specifies **2e-4**, on the "
        "grounds that a 12B run on *this exact corpus* reduced perplexity from 556.01 to 23.28 at that "
        "rate.")
    D.p("The supervisor's figure is a measurement on the target data, and a measurement outranks general "
        "guidance. The specification was followed — and because the disagreement was specifically about "
        "forgetting, the retention metric of §8.6 was added to test it directly. Section 9.5 reports the "
        "outcome: Dataset A accuracy **rose** rather than fell, so no forgetting occurred at this rate.")
    D.p("That evidence is bounded by what was measured. It covers the base-to-CPT direction only; it does "
        "not establish that the rate is safe in a pipeline where CPT is followed by supervised "
        "fine-tuning, which was not run.")
    D.h3("8.7.2  LoRA rank")
    D.p("The same earlier document argues that absorbing a domain requires more capacity than learning an "
        "output format, and recommends rank 64–128. The supervisor specifies rank **16** — *lower* than "
        "the rank 32 used for supervised fine-tuning. The specification was again followed, on the same "
        "grounds.")
    D.p("Unlike the learning rate, **this conflict is not resolved by the evidence collected.** The "
        "natural argument would be that the loss never plateaued, so capacity was not the binding "
        "constraint. That argument does not survive inspection: the cosine schedule decays the learning "
        "rate to approximately 1.5 × 10⁻⁸ by step 1650, four orders of magnitude below peak, so the flat "
        "tail visible in Figure 9.1 is what the schedule produces by construction and carries no "
        "information about capacity.")
    D.keypoint("Whether rank 16 limited the achievable adaptation is an open question that a single run "
               "at a single rank cannot answer. It is carried forward to the limitations (§11.4) and to "
               "future work.")

    # ---- 8.8 -------------------------------------------------------
    D.h2("8.8  Decisions deliberately not taken")
    D.p("**Embeddings were not trained.** The fertility measurement of §8.2.1 shows legal terms "
        "fragmenting into multiple subword units, and allowing the embedding and output layers to train "
        "(`modules_to_save`) is the standard way to let new terminology acquire dedicated "
        "representations. It was not done for two reasons: it substantially increases trainable "
        "parameters and memory, and it introduces a second change alongside the adapter, so any "
        "difference in outcome could not be attributed to either. It is specified as a follow-up run.")
    D.p("**No matched-token-budget control.** A fully controlled comparison of the two objectives would "
        "train an instruction-tuned model on the legal corpus at an equal token budget, isolating the "
        "objective from the data. This was not run, and it is the principal limitation of the comparison "
        "(§11.1). Chapter 10's argument is constructed so as not to depend on it: it claims the two "
        "objectives change *different things*, not that either outperforms the other.")
    D.p("**One run, one seed.** No variance estimate is available, and all effect sizes are point "
        "estimates from a single run.")

    # ---- 8.9 -------------------------------------------------------
    D.h2("8.9  Pre-registration")
    D.p("With several metrics available and only one run affordable, the analysis was exposed to a "
        "specific failure: selecting whichever metric produced the most favourable result after seeing "
        "the outcomes. To prevent it, three explicit predictions and a pass/fail gate were committed to "
        "the project's version control on **19 September 2026**, four days before training began on "
        "23 September.")
    D.numbered("Held-out legal perplexity falls substantially.")
    D.numbered("Next-token accuracy rises.")
    D.numbered("Dataset A retention stays roughly flat, because the CPT model receives no task training.")
    D.p("The commit timestamp is verifiable in the repository history and precedes the first training "
        "step. Section 9.7 reports all three against their outcomes — including the one that was wrong, "
        "which turned out to be the most productive result of the run.")

    # ---- 8.10 ------------------------------------------------------
    D.h2("8.10  Reproducibility")
    D.p("The corpus split is deterministic at seed 42 and was verified byte-identical across three "
        "platforms: the same 39,310 / 2,185 / 2,185 documents, the same 27,162,160 training tokens, the "
        "same 13,262 packed blocks and the same leakage counterfactual. The evaluation set and the cloze "
        "probe were each generated once and reused across both models. Every script, configuration file, "
        "evaluation output, the trained adapter and the complete trainer state are preserved in the "
        "project repository and reproduced in Appendix C. The whole experiment, including all "
        "evaluation, consumed approximately $5.40 of GPU time.")


# ======================================================================
#  CHAPTER 9 -- RESULTS: CONTINUED PRE-TRAINING
# ======================================================================
def build_ch9(D):
    D.h1("Chapter 9 — Results: Continued Pre-Training")
    D.p("This chapter reports the outcome of the continued pre-training run against the untrained base "
        "model. It proceeds from the training curve, through the three domain metrics, to the retention "
        "test on Dataset A, and closes by scoring the predictions registered in §8.9 against what "
        "actually happened.")
    D.p("Every comparison in this chapter is **base Gemma 2-2B against CPT Gemma 2-2B**, evaluated in the "
        "same session, on the same hardware, against identical evaluation files.")

    # ---- 9.1 -------------------------------------------------------
    D.h2("9.1  The training run")
    D.table([
        ["Quantity", "Value"],
        ["Optimiser steps", "1,658"],
        ["Wall clock", "3 h 42 m 21 s  (8.05 s/step)"],
        ["Peak video memory", "11.79 GB of 23.42 GB"],
        ["Training loss", "5.8262 (step 1) → **1.3138** (step 1,650)"],
        ["Held-out loss", "1.7528 (step 100) → **1.3110** (step 1,658)"],
        ["Held-out perplexity", "**3.71**"],
        ["Compute cost", "≈ $2.10"],
    ], widths=[6.0, 7.0])
    D.caption("The continued pre-training run at a glance.")
    D.figure("fig_cpt_loss.png",
             "Training and held-out loss over one epoch. The dashed grey line is the learning rate on "
             "the right-hand axis; note that the loss flattens exactly where the cosine schedule decays.",
             width_cm=14.5)
    D.p("Held-out loss fell monotonically across all seventeen evaluation points with **no uptick**. Two "
        "conclusions follow, and a third does not.")
    D.bullet("**No overfitting.** Selecting the best checkpoint by held-out loss returned the final one, "
             "so the saved adapter is genuinely the best of the run.")
    D.bullet("**The gains below are a lower bound**, because the run covered one epoch over 27.2M of the "
             "roughly 134M tokens available in the full corpus.")
    D.keypoint("**What the flat tail does *not* show.** Held-out loss moves by 0.0008 over the last 150 "
               "steps, which invites the reading that the model had converged. It does not support that "
               "reading: the cosine schedule has decayed the learning rate to about 1.5 × 10⁻⁸ by step "
               "1,650, so a model that is barely updating cannot display improvement. The flatness is "
               "what the schedule produces by construction, and the tail of this curve is uninformative "
               "in both directions — it shows neither that the corpus was exhausted nor that more "
               "remained to be gained.")

    # ---- 9.2 -------------------------------------------------------
    D.h2("9.2  Next-token evaluation")
    D.p("One thousand positions were sampled from held-out test documents at seed 42. The file was built "
        "once and both models were scored against it.")
    D.table([
        ["Model", "Top-1", "Top-5", "Top-10", "Avg NLL", "Perplexity", "Avg rank"],
        ["Base Gemma 2-2B", "21.3%", "32.9%", "40.0%", "6.693", "806.47", "1705.2"],
        ["**CPT Gemma 2-2B**", "**69.8%**", "**82.1%**", "**87.7%**", "**1.422**", "**4.14**", "**16.7**"],
        ["**Δ**", "**+48.5pp**", "**+49.2pp**", "**+47.7pp**", "−5.271", "−802.3", "−1688.6"],
    ], widths=[3.6, 1.9, 1.9, 2.0, 1.9, 2.1, 1.6], align_right={1, 2, 3, 4, 5, 6})
    D.caption("Next-token prediction on 1,000 held-out positions.")
    D.p("Perplexity on held-out legal text falls from **806.47 to 4.14**, and the average rank of the "
        "correct token falls from 1,705 to **16.7** — that is, the gold token moves from somewhere deep "
        "in a 256,000-token vocabulary to within the top twenty candidates on average.")

    D.h3("9.2.1  Per-source breakdown")
    D.p("Section 8.2 noted that two sources supply 73.7% of the corpus, so an aggregate gain could in "
        "principle be produced entirely by one template-heavy collection. Breaking the same 1,000 "
        "positions down by source rules that out.")
    D.figure("fig_cpt_nexttoken.png",
             "Next-token top-1 accuracy by source, base against CPT. Hatched bars mark sources with "
             "fewer than 100 sampled positions, which are not individually interpretable.",
             width_cm=13.5)
    D.keypoint("**Every source improves and none regresses.** The gain is not concentrated in the "
               "formulaic gazette material: `bibliographic_rulings`, the shortest and least templated "
               "source, gains 41.3 points. The conclusion rests on the three sources with n ≥ 249; the "
               "three with n ≤ 22 are reported for completeness and are too small to read individually.")

    # ---- 9.3 -------------------------------------------------------
    D.h2("9.3  The legal cloze probe")
    D.p("Perplexity and next-token accuracy both reward fluency, and legal prose is formulaic enough that "
        "a model could improve on both by learning document furniture alone. The cloze probe targets "
        "legal **terms** specifically: 300 items over 253 held-out documents, in three categories of 100.")

    D.h3("9.3.1  Constrained scoring — the headline result")
    D.table([
        ["Category", "Base", "CPT", "Δ", "Majority baseline", "Chance"],
        ["collocation", "23.0%", "**95.0%**", "+72.0pp", "11.0%", "5.0%"],
        ["defined_term", "65.0%", "**92.0%**", "+27.0pp", "11.0%", "4.2%"],
        ["**statute_term**", "42.0%", "**66.0%**", "+24.0pp", "**40.0%**", "25.0%"],
        ["**Overall**", "**43.3%**", "**84.3%**", "**+41.0pp**", "14.7%", "—"],
    ], widths=[3.4, 1.9, 1.9, 2.0, 3.0, 1.8], align_right={1, 2, 3, 4, 5})
    D.caption("Cloze accuracy under constrained scoring, with majority-class and chance baselines.")
    D.p("**The `statute_term` row is the most informative, and it is not the largest.** The base model "
        "scores 42.0% against a 40.0% majority baseline — that is, it is doing essentially nothing beyond "
        "reflecting that \\ar{المرسوم} is the commonest answer. CPT reaches 66.0%, clearing that baseline "
        "by 26 points, on items where the gold instrument has been removed from the context and a "
        "*competing* instrument is frequently present. That is direct evidence the model learned which "
        "legal instrument a given citation names, rather than only how legal text is shaped.")
    D.p("The `collocation` gain of 72 points is the largest and the least surprising: formulaic phrasing "
        "is what a domain language model absorbs first. `defined_term` rises from an already-strong 65%, "
        "indicating the base model already held much general Arabic legal vocabulary and that CPT's "
        "contribution in that category is correspondingly narrower.")

    D.h3("9.3.2  Free generation, and why it is not the headline")
    D.table([
        ["Category", "Base", "CPT", "Majority baseline"],
        ["collocation", "12.0%", "92.0%", "11.0%"],
        ["defined_term", "32.0%", "82.0%", "11.0%"],
        ["statute_term", "**1.0%**", "57.0%", "**40.0%**"],
        ["Overall", "15.0%", "77.0%", "14.7%"],
    ], widths=[3.6, 2.6, 2.6, 4.2], align_right={1, 2, 3})
    D.caption("Cloze accuracy under free generation, reported for completeness.")
    D.p("The `statute_term` figure of **1.0% against a 40.0% majority baseline** is far below chance, and "
        "the explanation is a property of the probe rather than of the model. The item builder drops any "
        "item whose answer already appears in its own context, to prevent the model from copying. For "
        "`statute_term` that filter leaves only items where the gold instrument is *absent* while a "
        "competing instrument usually appears earlier in the sentence — so free generation is primed "
        "toward a wrong answer by construction. The base model emitted \\ar{المرسوم} 37 times against 40 "
        "gold \\ar{المرسوم} answers and was correct once, which is possible only if the two have been made "
        "anti-correlated by the filter.")
    D.keypoint("A base-to-CPT delta measured from that floor would have been measuring the filter, not "
               "the model. Constrained scoring removes the artifact and supplies a defined chance level, "
               "which is why §8.6 designates it the primary measure. This is a methodological point, not "
               "merely a defect: an anti-copying filter that is not checked against a baseline can "
               "manufacture an arbitrarily large apparent improvement.")

    # ---- 9.4 -------------------------------------------------------
    D.h2("9.4  Reference figures from the 12B run")
    D.p("The supplied documentation reports a Gemma 4-12B run on the **full** corpus. Those figures are "
        "reproduced here for orientation only.")
    D.table([
        ["Model", "Top-1", "Top-5", "Top-10", "Avg NLL", "Perplexity"],
        ["Base Gemma 4-12B (cited)", "31.4%", "50.7%", "61.0%", "6.321", "556.01"],
        ["CPT Gemma 4-12B (cited)", "62.0%", "80.3%", "86.1%", "3.150", "23.28"],
    ], widths=[5.2, 1.9, 1.9, 2.0, 1.9, 2.1], align_right={1, 2, 3, 4, 5})
    D.caption("Published reference figures for Gemma 4-12B. NOT comparable with Table 9.2 — see below.")
    D.keypoint("**These numbers must not be read as a comparison.** They come from a different evaluation "
               "set, built from a different split of the full 134M-token corpus, while §9.2 reports 1,000 "
               "positions from this project's 27M-token subset. No claim is made here of having "
               "outperformed the 12B run. The defensible statement is that the *direction and scale* of "
               "improvement are consistent with the reference.")
    D.p("One comparison is legitimate. Base Gemma 2-2B enters at perplexity 577.8 on the sanity probe "
        "against 556.01 for base Gemma 4-12B — near-identical starting points on this domain, which "
        "corroborates that this setup measures what the reference measured.")

    # ---- 9.5 -------------------------------------------------------
    D.h2("9.5  Retention on Dataset A")
    D.p("The final metric asks whether domain adaptation damaged the model's general ability. The CPT "
        "model — which has never seen a WSD prompt — was evaluated zero-shot on 1,000 sentences from the "
        "Dataset A test set, every item offering exactly two candidate senses, so chance is 50%.")
    D.table([
        ["Model", "Accuracy", "Macro-F1", "Precision", "Recall", "n"],
        ["Base Gemma 2-2B, zero-shot", "24.9%", "0.1969", "0.192", "0.207", "1,000"],
        ["**CPT Gemma 2-2B, zero-shot**", "**36.5%**", "**0.2685**", "0.2617", "0.2825", "1,000"],
        ["*(context)* SFT Gemma 2-2B", "90.42%", "0.8333", "0.831", "0.838", "3,110"],
    ], widths=[5.0, 2.2, 2.0, 2.0, 1.8, 1.4], align_right={1, 2, 3, 4, 5})
    D.caption("Zero-shot retention on Dataset A. The supervised row is context, not a baseline.")
    D.p("The supervised row is included for orientation only: it comes from a model trained on this task, "
        "on a different n. The retention question is narrower — is base plus CPT adapter *worse* than "
        "base alone? It is not. Accuracy rose by 11.6 points.")

    D.h3("9.5.1  The decomposition — the key result of this chapter")
    D.p("An overall accuracy figure conflates two distinct abilities: whether the model produces a "
        "parseable answer at all, and whether that answer is right. Separating them changes the "
        "interpretation entirely.")
    D.table([
        ["Model", "Commits to an ID", "Correct given an answer", "Overall"],
        ["Base", "**52.5%**  (525/1000)", "47.4%  (249/525)", "24.9%"],
        ["**CPT**", "**75.1%**  (751/1000)", "48.6%  (365/751)", "36.5%"],
        ["*(context)* SFT", "≈ 100%", "≈ 90%", "90.42%"],
    ], widths=[3.0, 4.2, 4.2, 2.4], align_right={1, 2, 3})
    D.caption("Retention decomposed into format compliance and task discrimination.")
    D.keypoint("**The entire 11.6-point gain is format compliance.** Willingness to emit a parseable "
               "sense identifier rose from 52.5% to 75.1%. Accuracy *among items actually answered* moved "
               "from 47.4% to 48.6% — 1.2 points against a pooled standard error of about 2.8 points, "
               "which is nothing. Both 95% confidence intervals contain 50%: **neither model "
               "discriminates above chance.**")
    D.p("This also explains the otherwise puzzling fact that the base model scores *below* a random "
        "guesser. It is not worse than chance when it answers — it is at chance — but it produces no "
        "usable answer on 47.5% of items, and those count as errors.")
    D.p("A plausible mechanism is that an epoch of EOS-terminated blocks dense with citation patterns of "
        "the form \\ar{رقم ١٢٣٤} made the model markedly more likely to terminate cleanly and to emit "
        "identifier-shaped tokens. It did not make it better at choosing between two glosses.")
    D.p("Two things follow. **On retention:** QLoRA froze the base weights, and general ability rose "
        "rather than fell, so the question of whether CPT damaged the model is answered emphatically in "
        "the negative — which is also the empirical resolution of the learning-rate conflict in §8.7.1. "
        "**On the thesis:** CPT changed *form* without changing *task competence*, while supervised "
        "fine-tuning changed both. This is stronger evidence for the dissociation than comparing across "
        "different metrics would be, because both objectives are measured here on the same task, the "
        "same test set and the same decomposition.")

    D.h3("9.5.2  A measurement error worth reporting")
    D.p("The sense-identifier extractor inherited from Chapter 6 matched only a line containing nothing "
        "but the identifier — the format the supervised model was *trained* to emit. Run against any "
        "untrained model it reports exactly **0.0%**, while the model is in fact answering. The base "
        "model produced, for the target word \\ar{هرب}, the output *\"The correct sense for the target word "
        "is Sense ID: 14706, which means…\"* — the correct answer — and was scored as a non-response.")
    D.p("The extractor now strips the echoed prompt and falls back to the first candidate identifier "
        "appearing in prose. Both supervised output formats still extract identically, which is "
        "unit-tested.")
    D.keypoint("A retention figure produced by the original extractor would have been **false rather than "
               "merely noisy**, and it would have been false in the direction that made the headline "
               "story neater: an apparent catastrophic forgetting result of 0.0%. It is reported here "
               "because the decomposition in §9.5.1 depends on the extractor being correct.")

    # ---- 9.6 -------------------------------------------------------
    D.h2("9.6  Qualitative comparison")
    D.p("Twelve legal prompts were continued by both models. Both columns come from **one loaded model** "
        "— disabling the adapter turns CPT off — so the base column is provably the same base weights "
        "rather than a separate load. Greedy decoding, 60 new tokens.")
    D.p("Under greedy decoding the base model's characteristic failure is degenerate repetition, while "
        "the CPT model produces legal structure: 10 of 12 base continuations collapse into a loop within "
        "20–30 tokens, against 1 of 12 for CPT. The most verifiable contrast is the prompt "
        "\\ar{على وزير الداخلية والبلديات} (\"the Minister of Interior and Municipalities shall…\"), where "
        "the base model lists ministers and then repeats \\ar{ووزير البيئة} five times, while the CPT "
        "model continues:")
    D.listing(["بناء على قانون الجمعيات الصادر في 3 آب 1909 ولا سيما المادة السادسة منه"],
              caption="CPT continuation citing the 1909 Law of Associations.",
              arabic_lines=(0,))
    D.p("The 1909 Ottoman-era Law of Associations **is** the governing Lebanese law for associations, and "
        "\\ar{ولا سيما المادة … منه} (\"and in particular Article … thereof\") is the exact citation "
        "formula Lebanese decrees use. The model has produced the correct instrument in the correct "
        "invocation form. Further pairs are reproduced in Appendix D, deliberately including cases where "
        "CPT produces correct legal formulae and *then* degenerates.")

    D.h3("9.6.1  Is the repetition a decoding artifact? — measured")
    D.p("Greedy decoding is known to induce repetition loops in small models, so the comparison above "
        "risks attributing to training what belongs to the decoder. Rather than hedge, the same twelve "
        "prompts were re-run with nucleus sampling at three samples per prompt per condition. Loop rate "
        "is the share of continuations whose most-repeated 4-gram occurs three or more times; the same "
        "metric is applied to the greedy outputs so the rows are comparable.")
    D.table([
        ["Decoding", "n per condition", "Base loop rate", "CPT loop rate"],
        ["Greedy", "12", "**83.3%**  (10/12)", "8.3%  (1/12)"],
        ["Nucleus (top-p 0.9, T 0.8)", "36", "**8.3%**  (3/36)", "0.0%  (0/36)"],
    ], widths=[4.6, 2.8, 3.4, 3.0], align_right={1, 2, 3})
    D.caption("Degenerate-repetition rate under two decoding strategies.")
    D.keypoint("**Switching the base model's decoder — with no training whatsoever — drops its loop rate "
               "from 83.3% to 8.3%.** That single change accounts for almost the whole base-versus-CPT "
               "gap the greedy examples appear to show. Under matched sampled decoding the remaining "
               "difference, 3/36 against 0/36, is not statistically distinguishable (Fisher exact, "
               "two-tailed *p* = 0.24).")
    D.p("The claim that continued pre-training repaired the base model's degenerate generation is "
        "therefore **not supported**, and is not made. What the examples do support, under identical "
        "decoding, is that CPT changed *what* the model writes — its register, its citation formulae, "
        "its statutory structure. That claim is established independently, and without any decoding "
        "dependence at all, by the quantitative results in §§9.2–9.3.")

    # ---- 9.7 -------------------------------------------------------
    D.h2("9.7  Pre-registered predictions against outcome")
    D.p("The three predictions registered on 19 September, four days before training, score as follows.")
    D.table([
        ["Predicted", "Observed", "Outcome"],
        ["Held-out legal perplexity falls", "806.47 → 4.14", "Confirmed"],
        ["Next-token top-1 rises", "21.3% → 69.8%", "Confirmed"],
        ["Dataset A retention stays roughly flat", "24.9% → **36.5%** (+11.6pp)", "**Wrong**"],
    ], widths=[6.0, 4.6, 2.6])
    D.caption("Pre-registered predictions scored against measured outcomes.")
    D.keypoint("**The failed prediction is the most valuable result of the run.** Retention was expected "
               "to be flat because QLoRA freezes the base weights. Instead general-task accuracy rose by "
               "11.6 points. Having committed the prediction beforehand is what made that a *noticed "
               "surprise* rather than something rationalised after the fact — and pursuing it produced "
               "the decomposition of §9.5.1, which is the strongest single piece of evidence for the "
               "argument of Chapter 10.")
    D.p("The reasoning behind the wrong prediction was not mistaken so much as incomplete. QLoRA did "
        "freeze the base, and discrimination was indeed unchanged (47.4% → 48.6%, not significant). What "
        "went unanticipated was that CPT would alter the model's *output behaviour* — its willingness to "
        "terminate and to emit identifier-shaped tokens — enough to move overall accuracy without moving "
        "task competence at all.")

    # ---- 9.8 -------------------------------------------------------
    D.h2("9.8  Summary against the acceptance gate")
    D.p("The gate fixed in §8.9 required CPT to beat the base model on at least two of three metric "
        "families.")
    D.table([
        ["#", "Metric family", "Result", "Pass"],
        ["1", "Held-out legal perplexity", "806.47 → 4.14", "Yes"],
        ["2", "Term prediction — next-token", "21.3% → 69.8%  (+48.5pp)", "Yes"],
        ["2", "Term prediction — cloze (constrained)", "43.3% → 84.3%  (+41.0pp)", "Yes"],
        ["3", "Dataset A retention", "24.9% → 36.5%, no forgetting", "Yes"],
    ], widths=[1.0, 5.6, 4.6, 1.8], align_right={3})
    D.caption("Outcome against the pre-registered acceptance gate.")
    D.p("**Passed on three of three families.** The gate was already met on the first two before "
        "retention was measured, which is worth stating: retention was informative rather than "
        "load-bearing, and it returned a positive result rather than the predicted flat one.")


# ======================================================================
#  CHAPTER 10 -- DISCUSSION: THE DISSOCIATION
# ======================================================================
def build_ch10(D):
    D.h1("Chapter 10 — Discussion: What Each Objective Changes")
    D.p("Chapters 7 and 9 report two adaptations of the same 2-billion-parameter model. This chapter "
        "argues that read together they support a single conclusion, and that the conclusion is not the "
        "one a naive comparison would reach.")

    # ---- 10.1 ------------------------------------------------------
    D.h2("10.1  The question that should not be asked")
    D.p("The obvious question — *which objective is better?* — has no defensible answer here, and the "
        "temptation to answer it should be resisted explicitly. The two runs differ in every dimension "
        "that would have to be held constant:")
    D.table([
        ["Dimension", "Supervised fine-tuning", "Continued pre-training"],
        ["Data", "9,952 annotated WSD instances", "27.2M tokens of legal text"],
        ["Objective", "instruction → response", "next-token prediction"],
        ["Supervision", "gold sense identifiers", "none"],
        ["Evaluation", "accuracy on a labelled test set", "perplexity, cloze, next-token"],
    ], widths=[3.2, 5.0, 5.0])
    D.caption("The two experiments differ on every axis that a controlled comparison would fix.")
    D.p("A ranking across these two runs would be an artifact of the choice of metric. The productive "
        "question is different: **what did each objective change in the model?**")

    # ---- 10.2 ------------------------------------------------------
    D.h2("10.2  What supervised fine-tuning changed")
    D.keypoint("**Supervised fine-tuning changed the task format.** The model learned to select a sense "
               "identifier from candidates already present in its prompt. No knowledge was added, because "
               "none was required: everything needed for the decision was in the context window.")
    D.p("Three results from Chapter 7 support this reading rather than merely being consistent with it. "
        "The fine-tuned model produced **zero malformed outputs** across 3,110 test items, which is a "
        "statement about format compliance, not about legal or lexical knowledge. It exhibited **no "
        "positional bias**, despite the dataset's gold answer sitting in the second position 64.95% of "
        "the time — so it learned the task rather than the shortcut. And a 2B model **matched published "
        "8B models** on this dataset, which is the pattern expected when a task is discrimination "
        "between two visible strings rather than retrieval of absent knowledge: additional parameters "
        "buy knowledge, and knowledge is not what this task is short of.")

    # ---- 10.3 ------------------------------------------------------
    D.h2("10.3  What continued pre-training changed")
    D.keypoint("**Continued pre-training changed the domain distribution.** The model learned legal "
               "vocabulary, collocations and register. It gained no task ability: a CPT-only model still "
               "cannot answer a structured WSD question above chance.")
    D.p("Chapter 9 supports both halves of that statement separately. On the first, held-out legal "
        "perplexity fell from 806.47 to 4.14 and next-token accuracy rose by 48.5 points across **every** "
        "source in the corpus; the cloze probe's `statute_term` category rose from 42.0% — level with a "
        "40.0% majority baseline, i.e. no knowledge at all — to 66.0%, on items where the correct legal "
        "instrument had been removed from the context.")
    D.p("On the second, §9.5.1 is decisive, and it is decisive precisely because it was not the result "
        "that had been predicted. Dataset A accuracy rose by 11.6 points, which read naively looks like "
        "task improvement. Decomposed, the gain is **entirely** in the willingness to emit a parseable "
        "answer (52.5% → 75.1%), while accuracy among answered items moved from 47.4% to 48.6% — a change "
        "of 1.2 points against a pooled standard error of 2.8, with both confidence intervals containing "
        "50%.")

    # ---- 10.4 ------------------------------------------------------
    D.h2("10.4  The dissociation")
    D.p("Placing the two together gives a clean double dissociation on the two axes the experiments "
        "measure.")
    D.table([
        ["", "Task competence", "Domain distribution"],
        ["Supervised fine-tuning", "**Changed** (24.9% → 90.42%)", "Unchanged (no domain data seen)"],
        ["Continued pre-training", "**Unchanged** (47.4% → 48.6%, n.s.)", "**Changed** (PPL 806 → 4.14)"],
    ], widths=[4.4, 4.6, 4.6])
    D.caption("The dissociation: each objective moves one axis and leaves the other where it was.")
    D.p("The lower-left cell is the load-bearing one, and it is measured rather than assumed. It would "
        "have been possible to argue the dissociation weakly, by observing that CPT was evaluated on "
        "legal metrics and SFT on WSD metrics and that each improved on its own. That argument is "
        "circular. The decomposition in §9.5.1 avoids the circularity by measuring **both objectives on "
        "the same task, the same test set and the same split of behaviour into format and "
        "discrimination**.")
    D.keypoint("The conclusion holds **independently of effect size**. Even had the CPT perplexity gain "
               "been modest, the finding that it moved format compliance by 22.6 points while moving "
               "discrimination by 1.2 would carry the same meaning. The argument does not depend on CPT "
               "having worked well — only on *what* it changed.")

    # ---- 10.5 ------------------------------------------------------
    D.h2("10.5  Why this matters for practice")
    D.p("The dissociation has a direct practical consequence for anyone adapting a small model on a "
        "constrained budget: **the two objectives are not substitutes, and the choice between them "
        "follows from what is missing, not from which is more powerful.**")
    D.bullet("If the information needed for a task is already present in the prompt, the bottleneck is "
             "format, and supervised fine-tuning on a few thousand instances is sufficient — the "
             "Chapter 7 result shows a 2B model reaching published 8B performance this way for a few "
             "dollars of compute.")
    D.bullet("If the information is *absent* and must come from the weights, no amount of instruction "
             "tuning will supply it, and continued pre-training on domain text is the applicable tool.")
    D.bullet("A pipeline needing both — a legal-domain WSD system, for instance — needs both stages, in "
             "that order, and this project measured each in isolation rather than in combination.")

    # ---- 10.6 ------------------------------------------------------
    D.h2("10.6  Three methodological findings")
    D.p("Three results of this project concern how the measurements were made rather than what they "
        "showed. Each was found by checking a number that looked satisfactory, and each would have "
        "produced a wrong conclusion in the direction of a tidier story.")
    D.numbered("**The document-key collision (§8.3.2).** 396 identifiers were shared across source "
               "collections. Grouping by identifier alone, as specified, would have merged unrelated "
               "documents and silently weakened the leakage guarantee on which every held-out number "
               "depends.")
    D.numbered("**The extractor failure (§9.5.2).** An extractor written for the supervised output format "
               "reported 0.0% for any untrained model that was in fact answering correctly. Unchecked, it "
               "would have been reported as catastrophic forgetting — a dramatic and completely false "
               "result.")
    D.numbered("**The decoding confound (§9.6.1).** The base model's degenerate repetition, which the "
               "qualitative examples appear to show CPT repairing, is 90% attributable to greedy "
               "decoding. Measuring it prevented a claim the data does not support.")
    D.keypoint("The common thread is that each was a number that *looked* like a result. The safeguard "
               "that caught them was not skill but procedure: pre-registering predictions (§8.9) and "
               "requiring a baseline for every figure, so that a surprising number had to be explained "
               "rather than accepted.")


# ======================================================================
#  CHAPTER 11 -- LIMITATIONS
# ======================================================================
def build_ch11(D):
    D.h1("Chapter 11 — Limitations")
    D.p("This chapter states the limits of what the two experiments establish. They are given in "
        "descending order of how much they constrain the conclusions.")

    D.h2("11.1  No matched-token-budget control")
    D.p("The comparison between the two objectives is **qualitative, not controlled**. A controlled "
        "comparison would train an instruction-tuned model on the legal corpus at an equal token budget, "
        "isolating the objective from the data. Without it, the supervised and continued-pre-training "
        "runs differ in objective *and* in data *and* in evaluation simultaneously.")
    D.p("This is the most serious limitation in the report, and Chapter 10's argument is deliberately "
        "constructed to survive it: the dissociation claim rests on §9.5.1, where **both** objectives are "
        "measured on the same task and the same test set. That measurement does not require the control. "
        "The control would be required for any claim about relative effectiveness, and no such claim is "
        "made.")

    D.h2("11.2  Single corpus, single domain, single language variety")
    D.p("The continued pre-training result comes from one corpus of Lebanese legal Arabic. Whether it "
        "generalises to other Arabic varieties, to other specialised domains, or to Modern Standard "
        "Arabic of a non-legal register is untested. The corpus is also unusually formulaic, which "
        "§11.5 notes bounds the interpretation of the perplexity figure.")

    D.h2("11.3  Single model family and size")
    D.p("Both experiments use Gemma 2-2B. The finding that a 2B model matches published 8B results on "
        "Dataset A is a finding about this dataset and this model, and the mechanism proposed in §10.2 — "
        "that the task requires discrimination rather than knowledge — predicts the result would hold "
        "across families, but that prediction is not tested here.")

    D.h2("11.4  One run, one seed, one configuration")
    D.p("Neither experiment was repeated. No variance estimate exists, and all effect sizes are point "
        "estimates. Two specific consequences follow. First, differences of roughly one accuracy point on "
        "the Dataset A test set are not statistically distinguishable (§7.2.1), so the apparent ranking "
        "against published models should not be read as one. Second, the LoRA rank question raised in "
        "§8.7.2 remains open: rank 16 was specified and used, and **no evidence collected here shows "
        "whether it limited the achievable adaptation.** The flat tail of the loss curve cannot settle "
        "it, because the cosine schedule produces that flatness regardless.")

    D.h2("11.5  Corpus scale bounds the achievable effect")
    D.p("Training consumed 27.2M of the roughly 134M tokens in the full corpus, in a single epoch. The "
        "reported gains are therefore a lower bound on what this corpus can yield, and the reference 12B "
        "figures in §9.4 were obtained on the full corpus — one of several reasons they are not "
        "comparable.")

    D.h2("11.6  Perplexity measures register, not reasoning")
    D.p("Held-out perplexity of 4.14 is low in part because Lebanese gazette and legislation text is "
        "extremely formulaic: fixed openers, standard citation forms, standard closing formulae. "
        "Next-token accuracy on such text measures template regularity as much as legal knowledge. This "
        "is why the cloze probe was designed to target legal terms specifically, and why the "
        "`statute_term` category — where the majority baseline is high and the base model does not beat "
        "it — carries more evidential weight than the aggregate perplexity drop.")

    D.h2("11.7  Macro-F1 is reported despite being degenerate")
    D.p("Section 7.4 establishes that macro-averaged F1 on Dataset A reduces to accuracy rescaled by a "
        "denominator the model inflates through the variety of its own errors. It is nonetheless reported "
        "throughout, because the published results this work replicates report it and omitting it would "
        "make the comparison impossible. It should not be read as an independent measure of quality.")

    D.h2("11.8  No paired significance testing against published models")
    D.p("Establishing whether the difference between this work and the published fine-tuned results is "
        "real would require a paired test — McNemar's test on per-item agreement. The published per-item "
        "predictions are not released, so this is not possible. The defensible claim is therefore that "
        "the 2B model is **not worse** than the published 8B models on this dataset, not that it is "
        "better.")

    D.h2("11.9  Evaluation-set differences")
    D.p("Two smaller mismatches should be noted. Retention (§9.5) was measured on 1,000 Dataset A "
        "sentences rather than the full 3,110 used for the supervised evaluation, so the denominators "
        "differ. And three corpus sources in the per-source breakdown of §9.2.1 have fewer than 25 "
        "sampled positions and are not individually interpretable; the conclusion there rests on the "
        "three sources with n ≥ 249.")
