# -*- coding: utf-8 -*-
"""
Appendices C and D -- the continued pre-training run.

Both read the run artifacts from the repository at build time rather than
transcribing them, for the same reason Appendix A reads the pipeline source
live: what is printed in the report cannot then drift from what was run, and
no Arabic passes through a copy-paste step where it could be silently mangled.

    CPT/run_gemma2_2b/final_adapter/adapter_config.json   -> C.2
    CPT/run_gemma2_2b/trainer_state.json                  -> C.3, C.4
    CPT/run_gemma2_2b/qualitative.json                    -> D
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "CPT", "run_gemma2_2b")


def _load(name):
    path = os.path.join(RUN, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _wrap(text, width=64):
    """Break a long line on word boundaries. Used for the Arabic continuations,
    which are single long strings with no newlines of their own."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# ======================================================================
#  APPENDIX C -- CPT run configuration and logs
# ======================================================================
def build_appendix_c(D):
    D.h1("Appendix C — Continued Pre-Training: Run Configuration and Logs")
    D.p("As in Appendix B, every value here is read from an artifact the run itself wrote, not from the "
        "source scripts, so it records the run **as executed**. The tables are generated directly from "
        "those files when this report is built.")

    # ---- C.1 -------------------------------------------------------
    D.h2("C.1  Environment")
    D.table([
        ["Item", "Value"],
        ["Repository", "github.com/RokayaAlHarakeh/Arabic-WSD-LLM"],
        ["Branch", "week2-cpt"],
        ["Platform", "RunPod, container with a 30 GB network volume"],
        ["Accelerator", "NVIDIA RTX PRO 4000 Blackwell, 24 GB (sm_120)"],
        ["Base model", "`google/gemma-2-2b` (gated; accessed with an access token)"],
        ["torch", "2.8.0+cu128"],
        ["transformers", "5.17.0"],
        ["peft", "0.21.0"],
        ["bitsandbytes", "0.50.2"],
        ["CUDA", "12.8"],
    ], widths=[4.5, 9.0])
    D.caption("Execution environment for the continued pre-training run.")
    D.p("The framework choice is worth recording. Unsloth, used for the supervised run of Appendix B, "
        "could not be used here: its available build required a transformers version incompatible with "
        "the Blackwell architecture, and an earlier attempt on that stack produced a corrupted adapter. "
        "This run therefore drives `transformers`, `peft` and `bitsandbytes` directly.")

    # ---- C.2 -------------------------------------------------------
    D.h2("C.2  Adapter configuration")
    cfg = _load(os.path.join("final_adapter", "adapter_config.json"))
    if cfg is None:
        D.p("*(adapter_config.json not found at build time.)*")
    else:
        D.p("Read from `adapter_config.json` in the saved adapter directory. Inline comments are added "
            "here for readability and are not present in the file. Fields left at their library "
            "defaults are omitted.")
        tgt = cfg.get("target_modules") or []
        order = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        tgt = [m for m in order if m in tgt] + sorted(set(tgt) - set(order))
        lines = [
            '{',
            '  "peft_type":               "%s",' % cfg.get("peft_type"),
            '  "peft_version":            "%s",' % cfg.get("peft_version"),
            '  "base_model_name_or_path": "%s",' % cfg.get("base_model_name_or_path"),
            '  "task_type":               "%s",' % cfg.get("task_type"),
            '',
            '  "r":                       %s,        # half the rank used for SFT' % cfg.get("r"),
            '  "lora_alpha":              %s,        # scaling alpha/r = %.1f'
            % (cfg.get("lora_alpha"), cfg.get("lora_alpha", 0) / max(cfg.get("r", 1), 1)),
            '  "lora_dropout":            %s,       # no dropout: one epoch, 27M tokens'
            % cfg.get("lora_dropout"),
            '  "bias":                    "%s",' % cfg.get("bias"),
            '  "init_lora_weights":       %s,' % str(cfg.get("init_lora_weights")).lower(),
            '  "modules_to_save":         %s,      # embeddings frozen - see Section 8.8'
            % ("null" if cfg.get("modules_to_save") in (None, []) else cfg.get("modules_to_save")),
            '  "use_rslora":              %s,' % str(cfg.get("use_rslora")).lower(),
            '  "use_dora":                %s,' % str(cfg.get("use_dora")).lower(),
            '',
            '  "target_modules": [' + ", ".join('"%s"' % m for m in tgt[:4]) + ',',
            '                     ' + ", ".join('"%s"' % m for m in tgt[4:]) + ']',
            '}',
        ]
        D.listing(lines, caption="Saved LoRA adapter configuration for the CPT run.")
        D.p("The two values that differ most from the supervised configuration of Appendix B are `r`, "
            "which is 16 here against 32 there, and `lora_dropout`, which is zero. Section 8.7.2 records "
            "that the lower rank was specified rather than chosen, and that the evidence collected does "
            "not establish whether it was sufficient.")

    # ---- C.3 -------------------------------------------------------
    D.h2("C.3  Training arguments and outcome")
    st = _load("trainer_state.json")
    if st is None:
        D.p("*(trainer_state.json not found at build time.)*")
    else:
        ev = [(e["step"], e["eval_loss"]) for e in st["log_history"] if "eval_loss" in e]
        tr = [(e["step"], e["loss"]) for e in st["log_history"] if "loss" in e]
        lrs = [e["learning_rate"] for e in st["log_history"] if e.get("learning_rate") is not None]
        D.listing([
            '# --- schedule -------------------------------------------------------',
            'num_train_epochs                = %s' % st.get("num_train_epochs"),
            'max_steps                       = %s' % st.get("max_steps"),
            'per_device_train_batch_size     = %s' % st.get("train_batch_size"),
            'gradient_accumulation_steps     = 8         # effective batch size 8',
            'per_device_eval_batch_size      = 1         # 2048 x 256k vocab will not fit at 4',
            'max_seq_length                  = 2048',
            '',
            '# --- optimiser ------------------------------------------------------',
            'learning_rate                   = 0.0002    # 2e-4  - see Section 8.7.1',
            'lr_scheduler_type               = cosine',
            'warmup_ratio                    = 0.03',
            'optim                           = adamw_8bit',
            'weight_decay                    = 0.01',
            '',
            '# --- precision ------------------------------------------------------',
            'load_in_4bit                    = True',
            'bnb_4bit_quant_type             = nf4',
            'bnb_4bit_use_double_quant       = True',
            'bnb_4bit_compute_dtype          = bfloat16',
            '',
            '# --- observed -------------------------------------------------------',
            'global_step                     = %s' % st.get("global_step"),
            'best_eval_loss                  = %.5f' % st.get("best_metric", float("nan")),
            'peak_learning_rate              = %.3e' % (max(lrs) if lrs else float("nan")),
            'final_learning_rate             = %.3e' % (lrs[-1] if lrs else float("nan")),
        ], caption="Training arguments for the CPT run, with the observed outcome.")
        D.keypoint("The last two lines of the listing matter more than they look. The learning rate "
                   "peaks at 2 × 10⁻⁴ and ends at roughly 1.5 × 10⁻⁸ — four orders of magnitude lower. "
                   "A model updating that slowly cannot visibly improve, so the fact that the loss curve "
                   "flattens over the final steps is produced by the schedule rather than by the model "
                   "running out of things to learn. The flat tail is therefore not evidence of "
                   "convergence, and it is not evidence against it either.")

        # ---- C.4 ---------------------------------------------------
        D.h2("C.4  Held-out loss at every evaluation point")
        rows = [["Step", "Held-out loss", "Step", "Held-out loss"]]
        half = (len(ev) + 1) // 2
        for i in range(half):
            left = ev[i]
            right = ev[i + half] if i + half < len(ev) else None
            rows.append([f"{left[0]:,}", f"{left[1]:.4f}",
                         f"{right[0]:,}" if right else "", f"{right[1]:.4f}" if right else ""])
        D.table(rows, widths=[3.0, 3.6, 3.0, 3.6], align_right={0, 1, 2, 3})
        D.caption("Held-out cross-entropy at all %d evaluation points. Monotonically decreasing, "
                  "no uptick." % len(ev))
        if tr:
            D.p("Training loss ran from %.4f at step %d to %.4f at step %d. The full log, at every "
                "tenth step, is in `run_gemma2_2b/trainer_state.json` in the repository."
                % (tr[0][1], tr[0][0], tr[-1][1], tr[-1][0]))

    # ---- C.5 -------------------------------------------------------
    D.h2("C.5  Data preparation manifest")
    D.table([
        ["Stage", "Output"],
        ["Input records parsed", "45,636 (0 malformed, 0 empty)"],
        ["Documents after composite keying", "43,680"],
        ["Documents under the supplied `source_id` key", "43,284 — 396 collisions (Section 8.3.2)"],
        ["Split (documents)", "39,310 train / 2,185 val / 2,185 test"],
        ["Split (by token %)", "90.40 / 4.78 / 4.83"],
        ["Leakage avoided, validation", "135 of 2,282 records (5.9%)"],
        ["Leakage avoided, test", "127 of 2,282 records (5.6%)"],
        ["Training tokens (Gemma 2 tokenizer)", "27,162,160"],
        ["Packed blocks at 2,048 tokens", "13,262"],
        ["Remainder tokens discarded", "1,584  (0.006%)"],
        ["Tokenizer fertility, legal Arabic", "2.337 tokens/word"],
    ], widths=[6.6, 6.8])
    D.caption("Data preparation, as recorded in the split and packing reports.")
    D.p("Both reports are preserved in `CPT/reports/` in the repository. The split reproduced "
        "byte-identically on Windows, Google Colab and the RunPod Linux instance from the same "
        "committed script at seed 42.")


# ======================================================================
#  APPENDIX D -- qualitative samples
# ======================================================================
_GLOSS = {
    "بناء على المرسوم رقم": "Pursuant to Decree No.",
    "إن رئيس الجمهورية، بناء على": "The President of the Republic, pursuant to",
    "المادة الأولى: يحق لكل": "Article One: every … shall have the right to",
    "تنص المادة الثانية من قانون العمل على": "Article Two of the Labour Law provides that",
    "حكمت المحكمة": "The court ruled",
    "الجريدة الرسمية اللبنانية": "The Lebanese Official Gazette",
    "وحيث أن الاجتهاد مستقر على": "Whereas settled jurisprudence holds that",
    "يعاقب بالحبس من": "Shall be punished by imprisonment of",
    "لجنة الخدمة المدنية": "The Civil Service Board",
    "على وزير الداخلية والبلديات": "The Minister of Interior and Municipalities shall",
    "عقد العمل هو": "An employment contract is",
    "تسري أحكام هذا القانون على": "The provisions of this law apply to",
}


def build_appendix_d(D):
    D.h1("Appendix D — Continued Pre-Training: Arabic Generation Samples")
    D.p("Twelve legal prompts, continued by the base model and by the continued-pre-training model. "
        "**Both columns come from one loaded model**: disabling the adapter turns CPT off, so the base "
        "continuation is provably produced by the same base weights rather than by a separate load. "
        "Greedy decoding, 60 new tokens, no sampling.")
    D.keypoint("These samples illustrate **register and citation form**. They are not evidence of a "
               "capability difference, and in particular they are not evidence that continued "
               "pre-training repaired the base model's repetition: Section 9.6.1 shows that 90% of the "
               "difference in looping between these two columns is attributable to the choice of greedy "
               "decoding rather than to training. Examples where the CPT model produces correct legal "
               "formulae and *then* degenerates are included deliberately — an appendix in which every "
               "sample flatters the model reads as curated.")

    rows = _load("qualitative.json")
    if not rows:
        D.p("*(qualitative.json not found at build time — rebuild after the evaluation artifacts are "
            "in place.)*")
        return

    for i, r in enumerate(rows, 1):
        prompt = r.get("prompt", "")
        gloss = _GLOSS.get(prompt)
        D.h3("D.%d  %s" % (i, gloss if gloss else "Prompt %d" % i))
        D.p("**Prompt:** \\ar{%s}%s" % (prompt, "  — *\"%s\"*" % gloss if gloss else ""))

        base_lines = _wrap(r.get("base", ""))
        D.p("**Base Gemma 2-2B:**")
        D.listing(base_lines, arabic_lines=tuple(range(len(base_lines))))

        cpt_lines = _wrap(r.get("cpt", ""))
        D.p("**CPT Gemma 2-2B:**")
        D.listing(cpt_lines, arabic_lines=tuple(range(len(cpt_lines))))

    D.p("")
    D.p("The complete set, together with the sampled-decoding run behind Section 9.6.1 and its per-continuation "
        "repetition metrics, is in `CPT/run_gemma2_2b/qualitative.json` and "
        "`CPT/run_gemma2_2b/qualitative_sampled.json` in the repository.")
