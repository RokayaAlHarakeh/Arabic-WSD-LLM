# -*- coding: utf-8 -*-
"""
Generate the report's figures as PNGs into  figures/.

    python make_figures.py

Data-driven charts read the real run artifacts; conceptual diagrams are drawn.
No Arabic appears in any figure — matplotlib does not shape Arabic correctly
without extra dependencies, and every Arabic example already appears in the
body text where Word renders it properly.
"""
import os, json, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

DL = r"C:\Users\user\Downloads"
ERR = os.path.join(HERE, "..", "Gemma", "Fine-tuning", "Dataset-A", "error_analysis")

# ---- house style ------------------------------------------------------
NAVY, RED, GREY, LGREY, GREEN = "#1F4E79", "#C00000", "#595959", "#D9D9D9", "#2E7D32"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("  wrote", name)


def bar_labels(ax, bars, fmt="{:.2f}", dy=0.6, size=8.5):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=size)


# ======================================================================
#  DATA-DRIVEN FIGURES
# ======================================================================
def fig_fertility():
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    labels = ["Arabic\nsentences", "Arabic\nglosses", "Arabic\n(all)", "English\n(WikiText-2)"]
    vals = [2.272, 2.030, 2.079, 1.163]
    cols = [NAVY, NAVY, NAVY, GREY]
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    bars[2].set_color(RED)
    bar_labels(ax, bars, "{:.3f}", dy=0.03)
    ax.axhline(1.0, color=GREY, ls=":", lw=1)
    ax.text(3.42, 1.02, "1 token / word", fontsize=8, color=GREY, ha="right")
    ax.set_ylabel("Tokens per word (fertility)")
    ax.set_ylim(0, 2.65)
    ax.annotate("", xy=(3, 1.163), xytext=(2, 2.079),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.2))
    ax.text(2.5, 1.72, "1.79×", ha="center", color=RED, fontsize=10, fontweight="bold")
    ax.set_title("Gemma 2 tokenizer fertility: Arabic against English")
    save(fig, "fig_fertility.png")


def fig_memory():
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    labels = ["Full fine-tuning\n(fp32 Adam)", "Full fine-tuning\n(8-bit optimizer)",
              "LoRA\n(fp16 base)", "QLoRA\n(4-bit base)"]
    vals = [41.7, 26.0, 6.0, 2.5]
    cols = [RED, RED, NAVY, GREEN]
    bars = ax.bar(labels, vals, color=cols, width=0.6)
    bar_labels(ax, bars, "{:.1f} GB", dy=0.7)
    ax.axhline(16, color="black", ls="--", lw=1.3)
    ax.text(3.45, 17.0, "16 GB — available GPU", fontsize=8.5, ha="right")
    ax.set_ylabel("Peak training memory (GB)")
    ax.set_ylim(0, 48)
    ax.set_title("Memory required to adapt Gemma 2-2B (2.61B parameters)")
    save(fig, "fig_memory.png")


def fig_loss():
    src = os.path.join(DL, "trainer_state.json")
    if not os.path.exists(src):
        print("  [skip] fig_loss.png — trainer_state.json not found"); return
    st = json.load(open(src, encoding="utf-8"))
    tr = [(e["step"], e["loss"]) for e in st["log_history"] if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in st["log_history"] if "eval_loss" in e]
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.plot(*zip(*tr), color=NAVY, lw=1.1, label="Training loss (every 20 steps)")
    ax.plot(*zip(*ev), color=RED, lw=1.6, marker="o", ms=4.5,
            label="Held-out loss (every 500 steps)")
    for x in (1120, 2240):
        ax.axvline(x, color=GREY, ls=":", lw=0.9)
    for x, t in ((560, "epoch 1"), (1680, "epoch 2"), (2800, "epoch 3")):
        ax.text(x, 2.06, t, ha="center", fontsize=8, color=GREY)
    ax.annotate(f"{ev[-1][1]:.3f}", xy=ev[-1], xytext=(ev[-1][0] - 430, ev[-1][1] + 0.22),
                fontsize=8.5, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.annotate(f"{tr[-1][1]:.3f}", xy=tr[-1], xytext=(tr[-1][0] - 430, tr[-1][1] - 0.30),
                fontsize=8.5, color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY, lw=0.8))
    ax.set_xlabel("Optimiser step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_ylim(0, 2.25); ax.set_xlim(0, 3450)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Training and held-out loss over 3,360 optimiser steps")
    save(fig, "fig_loss.png")


def fig_baselines():
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    labels = ["Always\n1st candidate", "Random\n(2 candidates)", "Simplified Lesk\n(ties → 2nd)",
              "Always\n2nd candidate", "Fine-tuned\nGemma 2-2B"]
    vals = [35.05, 50.00, 64.98, 64.95, 90.42]
    cols = [GREY, GREY, GREY, RED, NAVY]
    bars = ax.bar(labels, vals, color=cols, width=0.62)
    bar_labels(ax, bars, "{:.2f}", dy=0.9)
    ax.axhline(64.95, color=RED, ls="--", lw=1.2)
    ax.text(0.02, 66.6, "true floor — 64.95%", fontsize=8.5, color=RED)
    ax.annotate("", xy=(4, 90.42), xytext=(4, 64.95),
                arrowprops=dict(arrowstyle="<->", color=NAVY, lw=1.2))
    ax.text(4.30, 77, "+25.47", rotation=90, va="center", color=NAVY,
            fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 104)
    ax.set_title("Model accuracy against the trivial and non-neural baselines")
    save(fig, "fig_baselines.png")


def fig_errors():
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8))
    (a, b), (c, e) = axes

    a.bar(["0.00", "0–0.10", "0.10–\n0.25", "0.25–\n0.50", "0.50+"],
          [10.20, 9.15, 8.81, 10.59, 15.62], color=[GREY]*4 + [RED], width=0.6)
    a.set_title("By gloss-pair similarity", fontsize=9.5)
    a.set_ylabel("Error rate (%)"); a.set_ylim(0, 19)
    a.text(4, 16.6, "n=32", fontsize=7.5, ha="center", color=RED)

    bb = b.bar(["word appears\nin gold gloss", "word absent"], [7.57, 14.61],
               color=[NAVY, RED], width=0.5)
    bar_labels(b, bb, "{:.2f}%", dy=0.35, size=8)
    b.set_title("By lemma anchoring", fontsize=9.5); b.set_ylim(0, 18)

    cc = c.bar(["1", "2–3", "4–9"], [11.05, 7.52, 6.38], color=NAVY, width=0.55)
    bar_labels(c, cc, "{:.2f}%", dy=0.25, size=8)
    c.set_title("By target-word frequency in test set", fontsize=9.5)
    c.set_ylabel("Error rate (%)"); c.set_xlabel("occurrences"); c.set_ylim(0, 14)

    ee = e.bar(["0–3", "4–6", "7–11", "12+"], [8.22, 9.56, 13.03, 16.67],
               color=[GREY, GREY, NAVY, NAVY], width=0.6)
    bar_labels(e, ee, "{:.2f}%", dy=0.3, size=8)
    e.set_title("By sentence length", fontsize=9.5); e.set_xlabel("words"); e.set_ylim(0, 20)

    fig.suptitle("What predicts error, across 3,110 test items", fontsize=11, y=1.00)
    fig.tight_layout()
    save(fig, "fig_errors.png")


def fig_position():
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(2); w = 0.36
    d = [35.05, 64.95]; m = [38.65, 61.35]
    b1 = ax.bar(x - w/2, d, w, label="Dataset (gold answer)", color=GREY)
    b2 = ax.bar(x + w/2, m, w, label="Model (its answers)", color=NAVY)
    bar_labels(ax, b1, "{:.2f}", dy=0.8, size=8)
    bar_labels(ax, b2, "{:.2f}", dy=0.8, size=8)
    ax.set_xticks(x); ax.set_xticklabels(["1st candidate", "2nd candidate"])
    ax.set_ylabel("Share of items (%)"); ax.set_ylim(0, 78)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("No positional bias: the model favours \"2nd\" less than the data does")
    save(fig, "fig_position.png")


def fig_tokenlen():
    src = os.path.join(HERE, "..", "Gemma", "Fine-tuning", "Dataset-A",
                       "fine_tuning_dataset_elrazzaz.jsonl")
    if not os.path.exists(src):
        print("  [skip] fig_tokenlen.png — training jsonl not found"); return
    try:
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    except Exception as ex:
        print("  [skip] fig_tokenlen.png —", ex); return
    TPL = ("Below is an instruction that describes a task, paired with an input that provides "
           "further context. Write a response that appropriately completes the request.\n\n"
           "### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}")
    lens = []
    for line in open(src, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        lens.append(len(tk.encode(TPL.format(r["instruction"], r["input"], r["output"]) + "<eos>",
                                  add_special_tokens=False)))
    lens = np.array(lens)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.hist(lens, bins=60, color=NAVY, alpha=0.85)
    ax.axvline(1024, color=RED, ls="--", lw=1.4)
    ax.text(1010, ax.get_ylim()[1]*0.82, "max_seq_len = 1024\nnothing truncated",
            color=RED, fontsize=8.5, ha="right")
    ax.axvline(np.median(lens), color="black", ls=":", lw=1.1)
    ax.text(np.median(lens)+14, ax.get_ylim()[1]*0.94,
            f"median {int(np.median(lens))}", fontsize=8.5)
    ax.set_xlabel("Tokens per formatted training example")
    ax.set_ylabel("Count"); ax.set_xlim(0, 1080)
    ax.set_title(f"Sequence length of all {len(lens):,} training examples (max {lens.max()})")
    save(fig, "fig_tokenlen.png")


def fig_adapter_split():
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    mods = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    vals = [139264, 106496, 106496, 139264, 368640, 368640, 368640]
    cols = [NAVY]*4 + [RED]*3
    bars = ax.bar(mods, [v/1000 for v in vals], color=cols, width=0.65)
    ax.set_ylabel("Parameters per layer (thousands)")
    ax.set_ylim(0, 440)
    ax.tick_params(axis="x", rotation=35, labelsize=8.5)
    ax.text(1.5, 405, "attention — 31%", color=NAVY, ha="center", fontsize=9, fontweight="bold")
    ax.text(5.0, 405, "MLP — 69%", color=RED, ha="center", fontsize=9, fontweight="bold")
    ax.set_title("Where the 41.5M adapter parameters sit (r = 32)")
    save(fig, "fig_adapter_split.png")


# ======================================================================
#  CONCEPTUAL DIAGRAMS
# ======================================================================
def _box(ax, x, y, w, h, text, fc="white", ec=NAVY, fs=9, bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight=("bold" if bold else "normal"), linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=lw))


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.2, 4.6, 9.6, 1.0,
         "Dataset A  —  three JSON files per split\nset  (question)      truth  (answer key)      dictionary  (glosses)",
         fc="#F2F2F2", ec=GREY, fs=8.5)
    stages = [
        (0.2, 3.0, "1. Data staging\ncreate_finetuning_dataset.py"),
        (2.6, 3.0, "2. Fine-tuning\nfinetuning.py"),
        (5.0, 3.0, "3. Inference\ninfer_model.py"),
        (7.4, 3.0, "4. Evaluation\neval.py"),
    ]
    for x, y, t in stages:
        _box(ax, x, y, 2.2, 1.1, t, fs=8)
    outs = [
        (0.2, 1.4, "9,952 Alpaca\nrecords (JSONL)"),
        (2.6, 1.4, "LoRA adapter\n41.5M params"),
        (5.0, 1.4, "predictions.json\n+ debug log"),
        (7.4, 1.4, "report.json\n90.42% accuracy"),
    ]
    for i, (x, y, t) in enumerate(outs):
        _box(ax, x, y, 2.2, 1.0, t, fc="#EEF3F8",
             ec=(GREEN if i == 3 else NAVY), fs=8,
             bold=(i == 3), lw=(1.6 if i == 3 else 1.0))
    _arrow(ax, 5.0, 4.6, 5.0, 4.15)
    for x, _, _ in stages:
        _arrow(ax, x + 1.1, 3.0, x + 1.1, 2.45)
    for x in (2.4, 4.8, 7.2):
        _arrow(ax, x, 1.9, x + 0.2, 1.9)
    for x in (2.4, 4.8, 7.2):
        _arrow(ax, x, 3.55, x + 0.2, 3.55)
    ax.set_title("The four-stage supervised fine-tuning pipeline", fontsize=11)
    save(fig, "fig_pipeline.png")


def fig_splits():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 3.0, 5.0, 4.0, 0.75, "Dataset A  —  15,549 senses", fc="#F2F2F2", ec=GREY, fs=9)
    ax.text(5.0, 4.72, "published 64 / 16 / 20 split", fontsize=7.5, color=GREY, ha="center")
    _box(ax, 0.15, 3.3, 3.0, 0.85, "train80\n9,952 instances", fs=8.5)
    _box(ax, 3.45, 3.3, 3.0, 0.85, "dev20\n2,487  (not used)", ec=GREY, fs=8.5)
    _box(ax, 6.75, 3.3, 3.1, 0.85, "test\n3,110 instances", ec=GREEN, lw=1.7, fs=8.5, bold=True)
    for x in (1.65, 4.95, 8.3):
        _arrow(ax, 5.0, 5.0, x, 4.2)
    ax.text(8.30, 2.95, "NEVER SEEN IN TRAINING", fontsize=7.5, color=GREEN,
            ha="center", fontweight="bold")
    ax.text(8.30, 2.62, "every reported accuracy", fontsize=7.5, color=GREEN, ha="center")
    _box(ax, 0.15, 1.35, 1.45, 0.8, "8,956\ngradient\nupdates", fs=7.5)
    _box(ax, 1.75, 1.35, 1.4, 0.8, "996\nmonitoring\nonly", ec=GREY, fs=7.5)
    _arrow(ax, 0.9, 3.3, 0.9, 2.2); _arrow(ax, 2.4, 3.3, 2.45, 2.2)
    ax.text(1.65, 2.32, "internal 90 / 10  (seed 42)", fontsize=7.5, color=GREY, ha="center")
    ax.text(1.65, 1.05, "not a test set — no reported\nnumber comes from here",
            fontsize=7, color=GREY, ha="center", style="italic")
    ax.set_title("How Dataset A is divided", fontsize=11)
    save(fig, "fig_splits.png")


def fig_lora():
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.text(0.35, 3.0, "x", fontsize=12, ha="center", va="center", style="italic")
    _arrow(ax, 0.7, 3.0, 1.5, 3.0)
    ax.plot([1.2, 1.2], [1.35, 4.5], color=GREY, lw=1.0)
    _arrow(ax, 1.2, 4.5, 1.9, 4.5); _arrow(ax, 1.2, 1.35, 1.9, 1.35)
    _box(ax, 1.9, 4.0, 2.5, 1.0, "W₀   (frozen)\n2304 × 2304", fc="#F2F2F2", ec=GREY, fs=9)
    ax.text(3.15, 3.72, "no gradient  ·  stored in NF4", fontsize=7.5, color=GREY, ha="center")
    _box(ax, 1.9, 0.85, 1.1, 1.0, "A\nr × d", fc="#EEF3F8", ec=RED, fs=8.5)
    _box(ax, 3.3, 0.85, 1.1, 1.0, "B\nd × r", fc="#EEF3F8", ec=RED, fs=8.5)
    _arrow(ax, 3.0, 1.35, 3.3, 1.35, color=RED)
    ax.text(3.15, 0.52, "trainable  ·  41.5M  ·  1.6%", fontsize=7.5, color=RED, ha="center")
    _arrow(ax, 4.4, 4.5, 6.5, 4.5); _arrow(ax, 4.4, 1.35, 5.6, 1.35, color=RED)
    _box(ax, 5.6, 0.95, 0.9, 0.8, "α / r", fc="white", ec=RED, fs=8.5)
    _arrow(ax, 6.5, 1.35, 6.9, 1.35, color=RED)
    ax.add_patch(plt.Circle((7.1, 3.0), 0.28, fc="white", ec=NAVY, lw=1.4))
    ax.text(7.1, 3.0, "+", fontsize=15, ha="center", va="center")
    ax.plot([6.9, 6.9], [1.35, 3.0], color=RED, lw=1.3)
    ax.plot([6.9, 6.82], [3.0, 3.0], color=RED, lw=1.3)
    ax.plot([6.5, 6.9], [4.5, 4.5], color=GREY, lw=1.3)
    ax.plot([6.9, 6.9], [4.5, 3.0], color=GREY, lw=1.3)
    _arrow(ax, 7.38, 3.0, 8.4, 3.0)
    ax.text(8.75, 3.0, "h", fontsize=12, ha="center", va="center", style="italic")
    ax.text(5.0, 5.55, "h  =  W₀ x  +  (α / r) · B A x", fontsize=11, ha="center")
    ax.text(5.0, 0.08, "B is initialised to zero, so ΔW = 0 at step 0 — training starts exactly at the "
                       "pretrained model", fontsize=7.5, ha="center", color=GREY, style="italic")
    ax.set_title("The LoRA decomposition", fontsize=11, y=1.02)
    save(fig, "fig_lora.png")


def fig_attention():
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    toks = ["he", "escaped", "the", "party", "by", "sleeping"]
    xs = np.linspace(0.45, 8.35, len(toks))
    for x, t in zip(xs, toks):
        fc = "#EEF3F8" if t == "escaped" else "white"
        ec = RED if t == "escaped" else GREY
        _box(ax, x, 0.45, 1.30, 0.62, t, fc=fc, ec=ec, fs=8.5,
             bold=(t == "escaped"), lw=(1.6 if t == "escaped" else 1.0))
    tgt_x, tgt_y = 2.90, 2.55
    weights = [0.05, 0.0, 0.06, 0.34, 0.10, 0.45]
    for x, w in zip(xs, weights):
        if w <= 0:
            continue
        # straight lines only: nothing routes through a neighbouring box
        ax.add_patch(FancyArrowPatch((x + 0.65, 1.07), (tgt_x, tgt_y), arrowstyle="-|>",
                                     mutation_scale=9, color=RED, lw=0.6 + 5.0 * w,
                                     alpha=0.30 + 0.60 * w))
    _box(ax, tgt_x - 1.55, tgt_y, 3.1, 0.85,
         "attention-weighted\nrepresentation of \"escaped\"",
         fc="#EEF3F8", ec=RED, fs=8)
    ax.text(6.9, 3.05,
            r"$\mathrm{softmax}\!\left(\dfrac{QK^{\top}}{\sqrt{d_k}}\right)V$",
            fontsize=12, ha="center", va="center")
    ax.text(6.9, 2.30, "arrow thickness = attention weight", fontsize=7.5,
            ha="center", color=GREY)
    ax.text(5.0, 4.85,
            "\"sleeping\" and \"party\" carry most of the weight -\n"
            "which is what selects the figurative sense of the verb",
            fontsize=8.5, ha="center", color=NAVY, linespacing=1.5)
    ax.set_title("Self-attention conditions a token on its context", fontsize=11)
    save(fig, "fig_attention.png")


def fig_prompt_flow():
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    _box(ax, 0.2, 4.5, 2.6, 1.15, "set.json\nsentence + word\n+ candidate IDs", fc="#F2F2F2",
         ec=GREY, fs=7.5)
    _box(ax, 3.1, 4.5, 2.6, 1.15, "truth.json\ngold sense ID", fc="#F2F2F2", ec=GREY, fs=7.5)
    _box(ax, 6.0, 4.5, 3.0, 1.15, "dictionary.json\nsense ID → gloss", fc="#F2F2F2", ec=GREY, fs=7.5)
    _box(ax, 2.1, 2.9, 5.8, 0.9, "join and flatten  →  one Alpaca record per target word", fs=8.5)
    for x in (1.5, 4.4, 7.5):
        _arrow(ax, x, 4.5, 5.0, 3.85)
    _arrow(ax, 5.0, 2.9, 5.0, 2.35)
    _box(ax, 1.5, 1.25, 7.0, 1.1,
         "instruction  (fixed, identical in all 9,952 records)\n"
         "input  (sentence + word + both glosses)      output  (\"14706\")",
         fc="#EEF3F8", ec=NAVY, fs=8)
    _arrow(ax, 5.0, 1.25, 5.0, 0.75)
    ax.text(5.0, 0.42, "rendered into the Alpaca template  →  tokenized  →  trained",
            fontsize=8.5, ha="center", color=NAVY)
    ax.text(5.0, 0.06, "the SAME construction is rebuilt at inference — verified byte-identical, "
                       "0 mismatches / 3,110",
            fontsize=7.2, ha="center", color=RED, style="italic")
    ax.set_title("Data staging: three source files to one training record", fontsize=11)
    save(fig, "fig_prompt_flow.png")


# ======================================================================
#  CPT figures (Chapter 9) -- read from the in-repo run artifacts
# ======================================================================
CPTRUN = os.path.join(HERE, "..", "CPT", "run_gemma2_2b")


def fig_cpt_loss():
    src = os.path.join(CPTRUN, "trainer_state.json")
    if not os.path.exists(src):
        print("  [skip] fig_cpt_loss.png -- trainer_state.json not found"); return
    st = json.load(open(src, encoding="utf-8"))
    tr = [(e["step"], e["loss"]) for e in st["log_history"] if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in st["log_history"] if "eval_loss" in e]
    lr = [(e["step"], e["learning_rate"]) for e in st["log_history"]
          if e.get("learning_rate") is not None]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(*zip(*tr), color=NAVY, lw=0.9, alpha=0.55,
            label="Training loss (every 10 steps)")
    ax.plot(*zip(*ev), color=RED, lw=1.6, marker="o", ms=4.0,
            label="Held-out loss (every 100 steps)")
    ax.annotate(f"{ev[-1][1]:.4f}", xy=ev[-1],
                xytext=(ev[-1][0] - 430, ev[-1][1] + 0.55), fontsize=8.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    ax.set_xlabel("Optimiser step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_xlim(0, 1700)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    # the learning rate, on a twin axis -- this is what explains the flat tail
    ax2 = ax.twinx()
    ax2.plot(*zip(*lr), color=GREY, ls="--", lw=0.9)
    ax2.set_ylabel("Learning rate", color=GREY, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=GREY, labelsize=8)
    ax2.grid(False)
    ax2.set_ylim(0, 2.2e-4)
    ax2.text(1230, 1.05e-4, "cosine decay", fontsize=7.8, color=GREY,
             rotation=-38, style="italic")

    ax.set_title("CPT loss over one epoch (1,658 steps), with the cosine schedule")
    save(fig, "fig_cpt_loss.png")


def fig_cpt_nexttoken():
    """Per-source next-token top-1, base vs CPT. Sources with n < 100 are hatched."""
    rows = [("legislations", 340, 27.4, 80.9), ("gazette_legal_core", 249, 25.7, 78.3),
            ("gazette_section2_sample", 98, 16.3, 63.3),
            ("bibliographic_rulings", 264, 12.1, 53.4),
            ("adl_rulings", 22, 22.7, 54.5), ("related_provisions", 22, 13.6, 45.5),
            ("associated_studies_ar", 5, 0.0, 60.0)]
    rows.sort(key=lambda r: r[3] - r[2])
    NL = chr(10)
    labels = [r[0].replace("_", NL, 1) + NL + "(n=" + str(r[1]) + ")" for r in rows]
    y = np.arange(len(rows)); h = 0.38
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    b1 = ax.barh(y + h / 2, [r[2] for r in rows], h, color=LGREY,
                 edgecolor=GREY, lw=0.6, label="Base Gemma 2-2B")
    b2 = ax.barh(y - h / 2, [r[3] for r in rows], h, color=NAVY, label="CPT Gemma 2-2B")
    for bars in (b1, b2):
        for bar, r in zip(bars, rows):
            if r[1] < 100:
                bar.set_hatch("///")
    for yy, r in zip(y, rows):
        ax.text(max(r[2], r[3]) + 2.0, yy, f"+{r[3] - r[2]:.1f}pp",
                va="center", fontsize=8.2, color=GREEN)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.2)
    ax.set_xlabel("Next-token top-1 accuracy (%)"); ax.set_xlim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Every source improves; hatched bars have n < 100", fontsize=10.5)
    save(fig, "fig_cpt_nexttoken.png")


if __name__ == "__main__":
    print("Generating figures into", FIG)
    for f in (fig_fertility, fig_memory, fig_loss, fig_baselines, fig_errors,
              fig_position, fig_tokenlen, fig_adapter_split,
              fig_pipeline, fig_splits, fig_lora, fig_attention, fig_prompt_flow,
              fig_cpt_loss, fig_cpt_nexttoken):
        try:
            f()
        except Exception as ex:
            print("  [FAIL]", f.__name__, "->", ex)
    print("done.")
