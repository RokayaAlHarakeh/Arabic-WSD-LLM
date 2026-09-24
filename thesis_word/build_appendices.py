"""General Conclusion (SFT half) plus Appendix A (source code) and Appendix B
(recovered run configuration).

Kept out of build_chapters.py because Appendix A reads the pipeline scripts off
disk at build time — the listings are never transcribed by hand, so they cannot
drift from the code that produced the results.
"""
import os

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "Gemma", "Fine-tuning", "Dataset-A")

# Local absolute paths are rewritten to a placeholder in the printed listings.
REDACTIONS = [
    (r"C:\Users\user\Desktop\FYP-Whitestork\WSD\Arabic-WSD-LLM\Datasets\Dataset-A",
     r"<REPO>\Datasets\Dataset-A"),
    (r"C:\Users\user\Desktop\FYP-Whitestork\WSD\Arabic-WSD-LLM",
     r"<REPO>"),
]


# ======================================================================
#  GENERAL CONCLUSION
# ======================================================================
def build_conclusion(D):
    D.h1("General Conclusion")
    D.p("This report set out to answer two questions. Whether a language model below three billion "
        "parameters can match published results on Arabic word sense disambiguation under a free-tier "
        "compute budget; and what supervised fine-tuning and continued pretraining each actually change "
        "about a model. The first question is now answered. The second is answered in part, and is "
        "completed by the continued-pretraining work that follows.")

    # ---- what was established --------------------------------------
    D.h2("What the supervised fine-tuning experiment established")
    D.p("Gemma 2-2B, adapted with QLoRA on a single free-tier GPU, reaches **90.42% accuracy** on the "
        "3,110-instance Dataset A test split. That matches the published LLaMA 3.1-8B result to the "
        "second decimal place and exceeds the same-family Gemma 2-9B result by 1.03 points, using "
        "roughly a quarter of the parameters. With a 95% confidence interval of [89.4, 91.4] the honest "
        "reading is not that the smaller model is *better*, but that it is **not worse** — which, given "
        "the difference in scale, is the more consequential claim.")
    D.p("Three further results go beyond replication, and each addresses one of the three gaps identified "
        "at the close of Chapter 5.")

    D.numbered("**The benchmark's true floor is 64.95%, not 50%.** Because every item offers exactly two "
               "candidates and the gold answer is the second of them about sixty-five percent of the "
               "time, a one-line rule that cannot read Arabic scores 64.95%. The figure is stable across "
               "all three splits, so it is a property of how the dataset was constructed rather than a "
               "sampling accident. The model's real contribution is therefore **+25.47 points over a "
               "trivial rule**, not +40 over a coin flip, and every published accuracy on this benchmark "
               "should be read against the same floor.")
    D.numbered("**Macro-F1 is degenerate on this dataset and should not be interpreted.** Every gold "
               "class has a support of exactly one, and the macro average is taken over a class list "
               "that the model itself enlarges whenever it predicts an identifier that is nobody's gold "
               "answer. The arithmetic resolves exactly: 2,812 correct predictions divided by a "
               "denominator of 3,356 classes, of which **246 exist only because the model was wrong in "
               "varied ways**. Two systems with identical accuracy can differ by eight points of "
               "macro-recall purely through the variety of their mistakes. The metric is reported here "
               "for comparability with the literature and for no other reason.")
    D.numbered("**The task is discrimination, not retrieval.** Every item supplies both candidate glosses "
               "inside the prompt, so the model is never required to recall what a word means. Three "
               "measurements support this reading: lexical overlap is worthless on this data, with "
               "Simplified Lesk adding 0.03 points over the positional rule and 64.6% of items tied; the "
               "model nonetheless resists misleading surface cues, answering correctly on 88.0% of the "
               "351 items where the *wrong* gloss shares more words with the sentence; and it shows no "
               "positional shortcut, being marginally *more* accurate on the rarer first position. What "
               "is left is a comparison of two short strings against a context — a low-dimensional "
               "operation, which is also why a rank-32 adapter over 1.6% of the parameters was enough to "
               "learn it.")

    D.keypoint("Together these explain the central puzzle of Chapter 7. Four models spanning two to nine "
               "billion parameters land within 1.4 accuracy points of one another because the benchmark "
               "does not reward what scale provides. Capacity buys stored knowledge; this task hands the "
               "knowledge to the model in the prompt.")

    D.p("A fourth result is narrower but worth recording, because it concerns deployability rather than "
        "accuracy. The fine-tuned model produced **zero malformed or out-of-set outputs across all 3,110 "
        "test items** — no refusals, no hallucinated identifiers, nothing the extraction step could not "
        "parse. Every one of the 298 errors was the other legitimate candidate. The replicated study "
        "reports 638 refusals from an eight-billion-parameter model on this same data before fine-tuning. "
        "Instruction fine-tuning eliminates the output-format problem completely, and does so at two "
        "billion parameters as reliably as at eight.")

    # ---- what it does not establish ---------------------------------
    D.h2("What this half of the project does not establish")
    D.p("The findings above concern one dataset, one language variety, one model family and one training "
        "run, and three qualifications follow from that.")
    D.bullet("The discrimination interpretation depends specifically on the **binary candidate structure "
             "of Dataset A**. It should not be carried over to benchmarks with larger sense inventories "
             "without being retested there.")
    D.bullet("**No paired significance test against the published systems was possible**, because their "
             "per-item predictions are not released. The comparison in Section 7.2 is therefore between "
             "an interval and three point estimates.")
    D.bullet("A configuration deviation — **sequence packing was enabled in the published recipe and "
             "disabled here** — means this run performed roughly 4.7 times more optimiser steps over the "
             "same data. Whether that helped, hurt, or did nothing was not ablated.")
    D.p("Section 7.7 sets out the full list of threats to validity. None of them undermines the three "
        "numbered findings, which are properties of the dataset and of the error distribution rather "
        "than of the training recipe.")

    # ---- lessons ----------------------------------------------------
    D.h2("Lessons learned")
    D.p("Two of the five lessons below cost a full training run to learn, and are recorded here in the "
        "hope that they save someone else the same week.")
    for t in [
        "**Unpinned dependencies are a correctness risk, not merely a reproducibility inconvenience.** An "
        "automatic upgrade installed a framework release whose forward pass for this model family was "
        "broken, and an entire training run was optimised against meaningless outputs before the fault "
        "was found. Pin versions, and record them alongside the results.",
        "**A training loss above the random baseline is a stopping condition, not a warm-up artefact.** "
        "The corrupt run logged an initial loss near 25, against the 12.45 expected from a uniform "
        "predictor over a 256,128-token vocabulary. That signal was visible in the first minutes and was "
        "misread for hours.",
        "**Make every stage independently restartable when the platform is unreliable.** Checkpointing "
        "every 100 steps with automatic resume, and a per-sentence checkpoint during inference, turned "
        "session disconnections from run-ending events into minor setbacks.",
        "**Layer-by-layer isolation is what converts a vague failure into a precise cause.** Testing the "
        "tokenizer, then the base model under two different libraries, then the adapter in turn located "
        "the fault definitively, where changing several things at once had not.",
        "**Audit a benchmark's metrics before trusting them.** The two most substantial contributions of "
        "this half of the project came not from training a better model, but from measuring what the "
        "dataset and its reported metrics actually do.",
    ]:
        D.bullet(t)

    # ---- future work ------------------------------------------------
    D.h2("Future work")
    D.p("Ordered by expected value relative to the effort required.")
    for t in [
        "**A matched-token-budget control.** Instruction fine-tuning on the same legal corpus used for "
        "continued pretraining, so that the comparison between the two objectives becomes controlled "
        "rather than qualitative. This is the single most valuable missing experiment.",
        "**Response-only loss masking.** Fifty-two percent of every training sequence is currently a "
        "constant prompt that receives gradient. Masking it would concentrate the entire signal on the "
        "answer at no additional cost.",
        "**A rank and quantization ablation.** Given that the task is low-dimensional, r = 8 would "
        "plausibly match r = 32 at a quarter of the adapter size; and an fp16 LoRA run would test the "
        "QLoRA quality claim directly on this task rather than on the authors' benchmarks.",
        "**Arabic-centric base models at comparable scale** — AraGPT2, Jais or ALLaM — which would "
        "separate the 1.79x tokenizer penalty measured in Section 2.3 from raw model capacity, and would "
        "address a limitation the replicated study itself acknowledges.",
        "**An encoder cross-encoder baseline on the same split**, for a like-for-like comparison with the "
        "ArabGlossBERT line of work under identical conditions instead of across reported figures.",
        "**Evaluation on Dataset B and on benchmarks with larger candidate sets**, to test whether the "
        "discrimination interpretation survives once the sense inventory grows beyond two.",
        "**Multiple random seeds**, to replace a single-run figure with a mean and a variance.",
    ]:
        D.bullet(t)

    D.todo("Extend this conclusion with the continued-pretraining findings and the SFT/CPT dissociation "
           "argument once Week 2 completes. The structure above is designed to take them without being "
           "rewritten: the CPT results become a second subsection under 'What was established', and the "
           "matched-budget control moves from Future Work into the results.")


# ======================================================================
#  APPENDIX A — pipeline source code
# ======================================================================
_SCRIPTS = [
    ("create_finetuning_dataset.py", "A.1", "Data staging",
     "Joins the three Dataset-A JSON files into the single Alpaca-format JSONL described in "
     "Section 6.2. This is the only stage that touches the raw dataset; everything downstream reads "
     "the JSONL it writes. The prompt text assembled at lines 51-61 is the one reproduced in "
     "Section 6.2.2, and the same text is reconstructed verbatim at inference time."),
    ("finetuning.py", "A.2", "QLoRA fine-tuning",
     "Loads the base model under 4-bit NF4 quantization, attaches the LoRA adapter described in "
     "Section 6.4, and trains for three epochs. Configuration is read from environment variables "
     "rather than edited between runs, so a single file serves both the 2B and the 4B experiments."),
    ("infer_model.py", "A.3", "Inference and answer extraction",
     "Reloads the quantized base, applies the saved adapter, generates greedily over the test split, "
     "extracts and validates the sense identifier, and writes a checkpoint after every sentence so "
     "that a disconnection costs at most one sentence of work."),
    ("eval.py", "A.4", "Evaluation",
     "Computes accuracy and the macro-averaged precision, recall and F1 discussed in Section 7.4. "
     "Note that \"none\" is treated as an ordinary label rather than being excluded, which matters "
     "for the macro average but not for this run, where no such output occurred."),
]


def _read_script(name, max_lines=None):
    """Return the file as a list of printable lines, or None if it is missing."""
    path = os.path.join(SRC_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n").replace("\t", "    ") for ln in f]
    out = []
    for ln in lines:
        for old, new in REDACTIONS:
            ln = ln.replace(old, new)
        out.append(ln)
    if max_lines and len(out) > max_lines:
        out = out[:max_lines] + ["", "... (%d further lines omitted)" % (len(out) - max_lines)]
    return out


def build_appendix_a(D):
    D.h1("Appendix A — Pipeline Source Code")
    D.p("The four scripts reproduced below constitute the complete supervised fine-tuning pipeline "
        "described in Chapter 6. They are read directly from the project repository when this document "
        "is generated rather than transcribed, so the code printed here cannot drift from the code that "
        "produced the results in Chapter 7. Line numbers are those of the source files.")
    D.p("Two edits are applied for presentation only. Local absolute paths are replaced by the "
        "placeholder `<REPO>`, and tabs are expanded to spaces. No logic is altered, and nothing is "
        "removed. Credentials are never present in these files: the Hugging Face token is read from an "
        "environment file that is excluded from version control.")
    D.p("The pipeline is parameterised entirely through environment variables — `WSD_PROJECT_DIR`, "
        "`WSD_BASE_MODEL`, `WSD_MODEL_TAG` and `WSD_MAX_SEQ_LEN` — so the four files are run unmodified "
        "for every experiment in this report. The exact values used are listed in Appendix B.")
    D.table([
        ["Stage", "Script", "Reads", "Writes"],
        ["A.1", "create_finetuning_dataset.py", "3 Dataset-A JSON files", "Alpaca JSONL"],
        ["A.2", "finetuning.py", "Alpaca JSONL", "LoRA adapter + trainer state"],
        ["A.3", "infer_model.py", "adapter, test_set.json", "predictions JSON"],
        ["A.4", "eval.py", "predictions, test_truth.json", "report JSON"],
    ], widths=[1.6, 5.2, 4.4, 4.3])
    D.caption("The four pipeline stages and their data dependencies.")

    for fname, num, title, blurb in _SCRIPTS:
        D.h2("%s  %s  (%s)" % (num, title, fname))
        D.p(blurb)
        lines = _read_script(fname)
        if lines is None:
            D.todo("%s was not found in the repository when this document was generated. Either paste "
                   "the file here by hand, or re-run build_docx.py from a checkout that contains it."
                   % fname)
            continue
        numbered = ["%3d  %s" % (i + 1, ln) for i, ln in enumerate(lines)]
        D.listing(numbered, caption="%s (%d lines) — %s." % (fname, len(lines), title), size=8.0)


# ======================================================================
#  APPENDIX B — recovered run configuration
# ======================================================================
def build_appendix_b(D):
    D.h1("Appendix B — Recovered Run Configuration")
    D.p("Every value in this appendix was recovered from artifacts that the training run itself wrote — "
        "the saved adapter configuration and the serialised training arguments — rather than read out of "
        "the source scripts. It therefore describes the run **as it actually executed**, including any "
        "environment variable that overrode a script default, and it is the authoritative record where "
        "the two disagree.")

    D.h2("B.1  Environment")
    D.table([
        ["Item", "Value"],
        ["Repository", "github.com/Yossranour1996/Arabic-WSD-LLM"],
        ["Branch", "phase1-gemma2-2b-datasetA"],
        ["Platform", "Google Colab, free tier"],
        ["Accelerator", "NVIDIA T4, 16 GB"],
        ["`WSD_BASE_MODEL`", "unsloth/gemma-2-2b"],
        ["`WSD_MODEL_TAG`", "gemma2_2b"],
        ["`WSD_MAX_SEQ_LEN`", "1024"],
        ["`WSD_PROJECT_DIR`", "/content/drive/MyDrive/WSD_Project"],
    ], widths=[5.5, 8.0])
    D.caption("Execution environment for the reported run.")
    D.todo("Confirm the accelerator model from the Colab session log before submission. The memory "
           "figures in Section 6.5 assume a 16 GB T4; if the session was allocated an L4 or an A100 "
           "instead, the headroom argument still holds but the numbers should be restated.")

    D.h2("B.2  Adapter configuration")
    D.p("Read from `adapter_config.json` in the saved adapter directory. Comments are added here for "
        "readability and are not present in the file.")
    D.listing([
        '{',
        '  "peft_type":               "LORA",',
        '  "peft_version":            "0.19.1",',
        '  "base_model_name_or_path": "unsloth/gemma-2-2b",',
        '  "task_type":               "CAUSAL_LM",',
        '',
        '  "r":                       32,',
        '  "lora_alpha":              32,        # scaling alpha/r = 1.0',
        '  "lora_dropout":            0.05,',
        '  "bias":                    "none",    # Gemma 2 carries no bias terms',
        '  "init_lora_weights":       true,      # B initialised to zero',
        '  "modules_to_save":         null,      # nothing trained in full precision',
        '  "use_rslora":              false,',
        '  "use_dora":                false,',
        '  "revision":                null,      # not recoverable - see Section 7.7',
        '',
        '  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",',
        '                     "gate_proj", "up_proj", "down_proj"]',
        '}',
    ], caption="Saved LoRA adapter configuration, as written by PEFT at the end of training.")

    D.h2("B.3  Training arguments")
    D.p("Deserialised from `training_args.bin`, saved alongside the final checkpoint.")
    D.listing([
        '# --- schedule -------------------------------------------------------',
        'num_train_epochs                = 3.0',
        'max_steps                       = -1        # epochs govern; 3,360 steps observed',
        'per_device_train_batch_size     = 1',
        'gradient_accumulation_steps     = 8         # effective batch size 8',
        'per_device_eval_batch_size      = 8',
        '',
        '# --- optimiser ------------------------------------------------------',
        'learning_rate                   = 0.0002    # 2e-4',
        'lr_scheduler_type               = linear',
        'warmup_steps                    = 50',
        'optim                           = adamw_8bit',
        'weight_decay                    = 0.01',
        'adam_beta1 / adam_beta2         = 0.9 / 0.999',
        'adam_epsilon                    = 1e-08',
        'max_grad_norm                   = 1.0',
        'seed                            = 3407',
        '',
        '# --- precision and memory -------------------------------------------',
        'bf16                            = True',
        'fp16                            = False',
        'gradient_checkpointing          = True',
        'gradient_checkpointing_kwargs   = {"use_reentrant": False}',
        '',
        '# --- logging, checkpointing, evaluation ------------------------------',
        'logging_steps                   = 20        # 168 loss points recorded',
        'save_strategy / save_steps      = steps / 100',
        'save_total_limit                = 2',
        'eval_strategy / eval_steps      = steps / 500   # 7 held-out measurements',
        'report_to                       = []',
    ], caption="Training arguments as executed, recovered from training_args.bin.")

    D.h2("B.4  Quantization and generation")
    D.p("The quantization configuration is applied to the frozen base at load time and is identical in "
        "training and inference; the attention implementation is matched across the two for the same "
        "reason. Generation is greedy, so the reported result is deterministic given the adapter.")
    D.listing([
        '# --- BitsAndBytes, applied to the frozen base -------------------------',
        'load_in_4bit                    = True',
        'bnb_4bit_quant_type             = "nf4"',
        'bnb_4bit_use_double_quant       = True',
        'bnb_4bit_compute_dtype          = bfloat16',
        'attn_implementation             = "sdpa"   # matched: train and inference',
        '',
        '# --- generation, at inference ----------------------------------------',
        'do_sample                       = False    # greedy, deterministic',
        'max_new_tokens                  = 128',
        'eos_token_id / pad_token_id     = <eos>',
        'use_cache                       = True',
    ], caption="Quantization and generation configuration.")

    D.h2("B.5  Derived quantities")
    D.p("None of the values below is stored anywhere; each is computed from the configuration above and "
        "from the recorded trainer state. They are collected here so that the arithmetic behind the "
        "figures quoted throughout Chapters 6 and 7 can be checked in one place.")
    D.table([
        ["Quantity", "Value", "Derivation"],
        ["Trainable parameters", "41,533,440", "7 modules × 26 layers at r = 32"],
        ["Share of the model", "1.6%", "41.5 M of 2.61 B"],
        ["MLP share of the adapter", "69%", "1,105,920 of 1,597,440 per layer"],
        ["Adapter on disk", "≈ 166 MB", "41.5 M parameters at fp32"],
        ["Training examples", "8,956", "9,952 staged, 996 held out"],
        ["Optimiser steps", "3,360", "⌈8,956 / 8⌉ × 3 epochs"],
        ["Steps had packing been on", "≈ 711", "4.74 examples per 1,024-token sequence"],
        ["Final training loss", "0.279", "trainer_state.json, step 3,360"],
        ["Final held-out loss", "0.615", "trainer_state.json, step 3,360"],
        ["Total training FLOPs", "7.196 × 10¹⁶", "trainer_state.json"],
    ], widths=[4.8, 3.0, 5.7])
    D.caption("Quantities derived from the recovered configuration and trainer state.")
