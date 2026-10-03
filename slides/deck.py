# -*- coding: utf-8 -*-
"""
Opening and Part 1 of the defense deck. Helpers live in build_slides.py.

Language is deliberately plain: this is spoken aloud, so short sentences and
ordinary words beat precise-but-dense ones. The numbers carry the talk.
"""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from build_slides import (
    blank, tb, run, para, rect, logo, title_of, takeaway, stat, bullets, table,
    figure, ar, ARFONT,
    NAVY, DEEP, INK, GREY, MUTE, TINT, LINE, WHITE, GREEN, RED, HEAD,
    W, H, M, TY, CY, DEFENSE_DATE, VARIANTS,
)

WS_LOGO = "whitestork_software_solutions_logo.jpg"
CODE_BG = RGBColor(0xF7, 0xF9, 0xFB)
MONO = "Consolas"

# the running example, used on slides 2 and 7 so the jury sees one item end to end
SENT = "وثمود الذين جابوا الصخر بالواد"
WORD = "جاب"
G_OK = "جاب الصخرة نقبها، خرقها"
G_NO = "جاب الخبر البلاد: عمها"


# ============================================================ TITLE =========
def s_title(prs, v):
    s = blank(prs)
    x = M
    for stem, h in (("logo_lu.png", 0.95), (v["second_logo"], 1.06), (WS_LOGO, 0.88)):
        y = 0.52 + (1.06 - h) / 2
        x += logo(s, stem, x, y, h) + 0.5

    tf = tb(s, W - M - 2.6, 0.64, 2.6, 0.3, align=PP_ALIGN.RIGHT)
    run(para(tf, True), DEFENSE_DATE, size=12, color=GREY, italic=True)

    tf = tb(s, M, 2.30, W - 2 * M, 1.7, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER; p.line_spacing = 1.08
    run(p, "PARAMETER-EFFICIENT ADAPTATION OF A\nSMALL LANGUAGE MODEL FOR ARABIC",
        size=34, color=INK, bold=True, font=HEAD)

    ln = rect(s, 4.15, 4.12, 5.03, 0.022, fill=LINE, shape=MSO_SHAPE.RECTANGLE)
    ln.shadow.inherit = False

    tf = tb(s, M, 4.34, W - 2 * M, 0.34, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER
    run(p, "Word Sense Disambiguation  ·  Lebanese Legal Continued Pre-Training",
        size=15, color=NAVY)

    tf = tb(s, M, 4.80, W - 2 * M, 0.32, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER
    run(p, v["banner"], size=12, color=GREY, bold=True)

    tf = tb(s, M, 5.80, 6.2, 1.2)
    p = para(tf, True); p.space_after = Pt(4)
    run(p, "Presented by  ", size=12.5, color=GREY)
    run(p, "Rokaya Al Harakeh", size=12.5, color=INK, bold=True)
    p = para(tf); p.space_after = Pt(3)
    run(p, "Academic supervisor:  Dr. [ Name ]", size=11.5, color=GREY)
    p = para(tf)
    run(p, "Company supervisor:  Dr. Daoud Baalbaki  —  WhiteStork", size=11.5, color=GREY)

    tf = tb(s, W - M - 5.3, 5.80, 5.3, 1.2, align=PP_ALIGN.RIGHT)
    for i, line in enumerate([t for t in v["institutions"] if t]):
        p = para(tf, i == 0); p.alignment = PP_ALIGN.RIGHT; p.space_after = Pt(2)
        run(p, line, size=11.5, color=GREY)
    return s


# ============================================================ 2. OBJECTIVES =
def s_objectives(prs):
    """Opens the talk: why this matters, what I set out to do, what it cost."""
    s = blank(prs)
    title_of(s, "What this project set out to do", kicker="Objectives")

    tf = tb(s, M, CY - 0.02, 6.1, 2.4)
    p = para(tf, True); p.space_after = Pt(11)
    run(p, "Large language models are very good at Arabic tasks — and very expensive "
           "to adapt. The published work on this benchmark all uses models of seven "
           "billion parameters and up.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(11)
    run(p, "This project asks whether a model four times smaller can keep up, and what "
           "it actually takes to adapt one.", size=15, color=INK, bold=True)
    p = para(tf)
    run(p, "Everything here runs on one small model, Gemma 2-2B, adapted in two "
           "completely different ways.", size=15, color=GREY)

    card = rect(s, 7.0, CY - 0.02, 5.73, 2.4, fill=WHITE, line=LINE)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.24); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(9)
    run(p, "Four things I set out to do", size=15, color=INK, bold=True)
    for n, (lead, rest) in enumerate([
            ("Match a 9B model with a 2B one, ", "on a free GPU."),
            ("Find the real floor — ", "nobody who published on this benchmark "
             "reported one."),
            ("Check the metrics ", "measure what everyone assumes."),
            ("Teach it Lebanese law, ", "and find out what that changes.")], 1):
        p = tfc.add_paragraph(); p.space_after = Pt(7)
        run(p, "%d.  " % n, size=14, color=NAVY, bold=True)
        run(p, lead, size=14, color=INK, bold=True)
        run(p, rest, size=14, color=GREY)

    tfh = tb(s, M, 4.18, W - 2 * M, 0.3)
    run(para(tfh, True), "Where it ended up", size=16, color=INK, bold=True)
    stat(s, M, 4.58, 2.87, "90.42%", "on Arabic word senses", "equal to 8B models",
         color=GREEN, h=1.5)
    stat(s, M + 3.073, 4.58, 2.87, "4.14", "perplexity on legal text",
         "down from 806", color=GREEN, h=1.5)
    stat(s, M + 6.146, 4.58, 2.87, "$0", "for the first half", "free Colab GPU", h=1.5)
    stat(s, M + 9.219, 4.58, 2.87, "$5.40", "for the second half", "rented GPU", h=1.5)
    return s


# ============================================================ PART 1 OPENER ==
def s_task(prs):
    """Opens Part 1: one real item from the dataset, before any numbers."""
    s = blank(prs)
    title_of(s, "The task, in one example", kicker="Part 1 · SFT")

    tf = tb(s, M, CY - 0.02, 5.7, 0.9)
    p = para(tf, True)
    run(p, "One Arabic word can mean different things. We give the model a sentence, "
           "one word in it, and two possible meanings. It picks one.",
        size=15, color=GREY)

    card = rect(s, M, 2.52, 5.7, 3.5, fill=WHITE, line=LINE)
    tf = card.text_frame
    tf.margin_left = tf.margin_right = Inches(0.24); tf.margin_top = Inches(0.2)

    p = tf.paragraphs[0]; p.space_after = Pt(4)
    run(p, "Sentence", size=11.5, color=MUTE, bold=True)
    p = tf.add_paragraph(); p.space_after = Pt(14)
    ar(p, SENT, size=19, color=INK)

    p = tf.add_paragraph(); p.space_after = Pt(4)
    run(p, "Target word", size=11.5, color=MUTE, bold=True)
    p = tf.add_paragraph(); p.space_after = Pt(14)
    ar(p, WORD, size=19, color=NAVY, bold=True)

    p = tf.add_paragraph(); p.space_after = Pt(6)
    run(p, "Two candidate meanings", size=11.5, color=MUTE, bold=True)
    p = tf.add_paragraph(); p.space_after = Pt(6)
    ar(p, G_OK + "   ← 4548", size=15, color=GREEN, bold=True)
    p = tf.add_paragraph()
    ar(p, G_NO + "   ← 4549", size=15, color=GREY)

    tf = tb(s, 6.68, 2.52, 6.05, 3.5)
    p = para(tf, True); p.space_after = Pt(12)
    run(p, "Here it means ", size=15, color=GREY)
    run(p, "to bore through rock", size=15, color=GREEN, bold=True)
    run(p, ", not ", size=15, color=GREY)
    run(p, "for news to spread", size=15, color=INK, bold=True)
    run(p, ". The answer is 4548.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(12)
    run(p, "Arabic makes this harder. ", size=15, color=INK, bold=True)
    run(p, "Words change shape a lot, small words attach to bigger ones, and the short "
           "vowels that would separate the meanings are simply not written down.",
        size=15, color=GREY)
    p = para(tf); p.space_after = Pt(12)
    run(p, "Notice what the model is handed. ", size=15, color=INK, bold=True)
    run(p, "Both meanings are already in front of it. It does not need to know "
           "anything — it needs to choose.", size=15, color=GREY)
    p = para(tf)
    run(p, "Remember that. It explains most of Part 1.", size=14.5, color=DEEP,
        italic=True)

    takeaway(s, "3,110 test items, always **exactly two** choices. We compare against a "
                "2025 paper that tested GPT-4o and three open models of 7 to 9 billion.")
    return s


# ============================================================ 3. WHAT IS FT =
def s_what_is_ft(prs):
    s = blank(prs)
    title_of(s, "What fine-tuning means", kicker="The method, plainly")

    tf = tb(s, M, CY - 0.04, 12.1, 0.85)
    p = para(tf, True)
    run(p, "A language model is first ", size=15.5, color=GREY)
    run(p, "pre-trained", size=15.5, color=INK, bold=True)
    run(p, " on an enormous amount of general text. That is where it learns language, "
           "and it costs millions, so nobody repeats it. Instead we take the finished "
           "model and ", size=15.5, color=GREY)
    run(p, "adapt it", size=15.5, color=INK, bold=True)
    run(p, " to what we need. That is fine-tuning.", size=15.5, color=GREY)

    cards = [
        ("1", "Pre-training", "Learn language in general. Done once, by Google, on "
         "trillions of words. We never touch this step.", GREY),
        ("2", "Fine-tuning", "Adapt the finished model using a small amount of our own "
         "data. Hours, not months. This project does it twice.", NAVY),
        ("3", "Two ways to adapt", "Show it worked examples of a task — or let it "
         "keep reading text from a new field. Those are SFT and CPT.", DEEP),
    ]
    x = M
    for num, head, body, col in cards:
        c = rect(s, x, 2.68, 3.88, 2.45, fill=WHITE, line=LINE)
        tf = c.text_frame
        tf.margin_left = tf.margin_right = Inches(0.24); tf.margin_top = Inches(0.2)
        p = tf.paragraphs[0]; p.space_after = Pt(5)
        run(p, num, size=26, color=col, bold=True, font=HEAD)
        p = tf.add_paragraph(); p.space_after = Pt(8)
        run(p, head, size=16, color=INK, bold=True)
        p = tf.add_paragraph()
        run(p, body, size=14, color=GREY)
        x += 4.105

    tf = tb(s, M, 5.42, 12.1, 0.75)
    p = para(tf, True)
    run(p, "The catch: ", size=15.5, color=INK, bold=True)
    run(p, "adapting even a finished model normally needs far more memory than a free "
           "GPU has. The next slide shows why — and what gets around it.",
        size=15.5, color=GREY)
    return s


# ============================================================ 4. MEMORY ====
def s_finetuning(prs):
    s = blank(prs)
    title_of(s, "Why fine-tuning needs a trick", kicker="The obstacle")
    tf = tb(s, M, CY, 5.9, 1.3)
    p = para(tf, True); p.space_after = Pt(9)
    run(p, "Training touches every weight. But the weights are not what fills the "
           "memory — the training bookkeeping is.", size=15, color=GREY)
    p = para(tf)
    run(p, "The optimiser keeps two extra numbers for every weight it trains.",
        size=14.5, color=GREY, italic=True)

    stat(s, M, 2.86, 2.8, "41.7 GB", "What training needs", "if we train everything",
         color=RED, h=1.55)
    stat(s, M + 3.0, 2.86, 2.8, "16 GB", "What we had", "a free Colab GPU",
         color=GREY, h=1.55)

    tfa = tb(s, M, 4.72, 5.8, 1.4)
    p = para(tfa, True)
    run(p, "Only 5.2 GB of that is the model. ", size=14.5, color=GREY)
    run(p, "The other 36.5 GB is bookkeeping — and it grows with how many weights "
           "you train, not with how big the model is.", size=14.5, color=INK, bold=True)

    figure(s, "fig_memory.png", 7.05, 1.78, w=5.7)
    takeaway(s, "So train fewer weights and the problem goes away. "
                "**That is the whole idea behind LoRA.**")
    return s


# ============================================================ 5. LoRA ======
def s_lora(prs):
    s = blank(prs)
    title_of(s, "LoRA and QLoRA", kicker="The trick")
    tf = tb(s, M, CY, 6.0, 2.4)
    p = para(tf, True); p.space_after = Pt(11)
    run(p, "LoRA.  ", size=15.5, color=INK, bold=True)
    run(p, "Freeze the model. Add two small matrices next to it and train only those. "
           "The change a model needs for one task turns out to be simple enough to fit "
           "in them.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(11)
    run(p, "QLoRA.  ", size=15.5, color=INK, bold=True)
    run(p, "On top of that, store the frozen model in 4 bits instead of 16. A quarter "
           "of the space, and the small trainable part stays at full precision.",
        size=15, color=GREY)
    p = para(tf)
    run(p, "Afterwards the small part folds back in, so nothing runs slower.",
        size=14.5, color=DEEP, italic=True)

    stat(s, M, 4.34, 2.8, "1.6%", "of the model is trained", "41.5M of 2.61B weights",
         h=1.5)
    stat(s, M + 3.0, 4.34, 2.8, "16×", "less memory", "now it fits the free GPU",
         color=GREEN, h=1.5)
    figure(s, "fig_lora.png", 7.1, 1.7, w=5.65)
    takeaway(s, "Both halves of this project use QLoRA on the same model. "
                "**The only thing that changes is what we train it on.**")
    return s


# ============================================================ 6. TWO + GOALS
def s_two(prs):
    s = blank(prs)
    title_of(s, "Two ways to fine-tune the same model",
             kicker="The central distinction")
    rows = [["", "Show it worked examples  (SFT)", "Let it read a new field  (CPT)"],
            ["Learns from", "question → answer pairs", "plain text, nothing labelled"],
            ["Our data", "9,952 Arabic WSD items", "27.2M words of Lebanese law"],
            ["Compared against", "a published paper", "the same model, untrained"],
            ["What it teaches", "how to answer", "what the field sounds like"]]
    table(s, M, CY, W - 2 * M, rows, [2.55, 4.78, 4.76], size=13.5, row_h=0.48,
          bold_rows=(4,))

    tf = tb(s, M, 4.46, W - 2 * M, 1.5)
    p = para(tf, True); p.space_after = Pt(10)
    run(p, "The bottom row is the point of the whole talk. ", size=15.5, color=INK,
        bold=True)
    run(p, "One of these teaches the model how to answer a question. The other teaches "
           "it what a field of writing sounds like. They are not interchangeable, and "
           "the rest of this talk measures both to show the difference is real.",
        size=15, color=GREY)
    p = para(tf)
    run(p, "Part 1 is the first column. Part 2 is the second.", size=15, color=DEEP,
        italic=True)

    takeaway(s, "The question is never which one is better. It is "
                "**what does each one change**.")
    return s


# ============================================================ PART 1 =======
def s_sft_setup(prs):
    """The real prompt, verbatim from create_finetuning_dataset.py."""
    s = blank(prs)
    title_of(s, "What the model actually sees", kicker="Part 1 · SFT")

    card = rect(s, M, CY - 0.04, 8.02, 3.92, fill=CODE_BG, line=LINE)
    tf = card.text_frame
    tf.margin_left = tf.margin_right = Inches(0.22); tf.margin_top = Inches(0.14)
    p = tf.paragraphs[0]; p.space_after = Pt(5)
    run(p, "ONE TRAINING EXAMPLE, EXACTLY AS BUILT", size=10.5, color=MUTE, bold=True)

    def line(text, col=GREY, bold=False, is_ar=False, size=11, after=2):
        q = tf.add_paragraph(); q.space_after = Pt(after)
        if is_ar:
            ar(q, text, size=size, color=col, bold=bold, align_right=False)
        else:
            run(q, text, size=size, color=col, bold=bold, font=MONO)

    line("### Instruction:", NAVY, True, after=3)
    line("You are tasked with performing Word Sense Disambiguation (WSD). Your job is "
         "to analyze the given sentence and identify the correct sense for the target "
         "word based on the context. For each sense, you are provided with a Sense ID "
         "and its definition. Using the context of the sentence, choose the most "
         "appropriate sense definition and provide the corresponding Sense ID.",
         GREY, after=8)
    line("### Input:", NAVY, True, after=3)
    line("Sentence: '" + SENT + "'", GREY, is_ar=True)
    line("Target Word: '" + WORD + "'", GREY, is_ar=True)
    line("Possible Senses:", GREY)
    line("[Sense ID: 4548, Definition: " + G_OK + "],", GREY, is_ar=True)
    line("[Sense ID: 4549, Definition: " + G_NO + "]", GREY, is_ar=True, after=8)
    line("### Response:", NAVY, True, after=3)
    line("4548", GREEN, True, size=12)

    table(s, 8.86, CY - 0.04, 3.85, [["The run", ""],
                                     ["Model", "Gemma 2-2B, 4-bit"],
                                     ["Examples", "9,952"],
                                     ["LoRA rank", "32"],
                                     ["Epochs", "3"],
                                     ["Hardware", "free Colab GPU"],
                                     ["Cost", "nothing"]],
          [1.75, 2.1], size=12.5, row_h=0.42)

    tf = tb(s, M, 5.42, W - 2 * M, 0.78)
    p = para(tf, True)
    run(p, "Both answers are inside the prompt. ", size=15.5, color=INK, bold=True)
    run(p, "Nothing has to be remembered. The model only has to tell two short pieces "
           "of text apart — which is why a small model can keep up.",
        size=15, color=GREY)
    takeaway(s, "The same prompt is rebuilt at test time, **character for character** "
                "— checked, 0 mismatches out of 3,110.")
    return s


def s_sft_headline(prs):
    s = blank(prs)
    title_of(s, "90.42% — as good as models four times bigger",
             kicker="Part 1 · Result")
    stat(s, M, CY, 2.72, "90.42%", "Correct", "2,812 of 3,110", h=1.5)
    stat(s, M + 2.92, CY, 2.72, "0.8333", "Macro-F1", "but see two slides on", h=1.5)
    stat(s, M + 5.84, CY, 2.72, "0", "Broken answers", "every output was usable",
         color=GREEN, h=1.5)

    table(s, M, 3.36, 8.56, [["Model", "Size", "Accuracy", "Macro-F1"],
                             ["Gemma 2-9B (paper)", "9B", "89.39", "81.72"],
                             ["LLaMA 3.1-8B (paper)", "8B", "90.42", "83.20"],
                             ["Qwen 2.5-7B (paper)", "7B", "90.77", "83.98"],
                             ["Gemma 2-2B (this work)", "2B", "90.42", "83.33"]],
          [3.46, 1.5, 1.8, 1.8], size=13, row_h=0.42, bold_rows=(4,))

    card = rect(s, 9.86, 3.36, 2.87, 2.52, fill=WHITE, line=LINE)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.18); tfc.margin_top = Inches(0.16)
    p = tfc.paragraphs[0]; p.space_after = Pt(7)
    run(p, "Is that gap real?", size=14.5, color=INK, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(7)
    run(p, "With 3,110 items the margin of error is ±0.53 points — so 89.4 to 91.4.",
        size=13, color=GREY)
    p = tfc.add_paragraph()
    run(p, "Every published score sits inside that. Gaps under one point mean nothing "
           "here.", size=12.5, color=RED)

    takeaway(s, "So I do not say 2B is better. I say it is "
                "**not worse** — at a quarter the size, and for free.")
    return s


def s_sft_baselines(prs):
    s = blank(prs)
    title_of(s, "What score do you get with no model at all?",
             kicker="Part 1 · Finding")
    figure(s, "fig_baselines.png", M, CY + 0.12, w=6.85)
    tf = tb(s, 7.8, CY, 4.93, 3.6)
    p = para(tf, True); p.space_after = Pt(12)
    run(p, "Two choices per item, so guessing gets 50%. But the right answer happens "
           "to be the second one ", size=15, color=GREY)
    run(p, "64.95%", size=15, color=RED, bold=True)
    run(p, " of the time.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(12)
    run(p, "A one-line program that always answers “the second one” scores "
           "64.95%. No model. No Arabic.", size=15, color=INK, bold=True)
    p = para(tf); p.space_after = Pt(12)
    run(p, "It is the same on all three splits, so it is baked into how the dataset "
           "was built.", size=14.5, color=GREY)
    p = para(tf)
    run(p, "A classical method from the 1980s adds almost nothing on top of it.",
        size=14.5, color=GREY)
    takeaway(s, "No paper on this benchmark reports a baseline. "
                "**The real result is 90.42 against 64.95 — not against 50.**")
    return s


def s_sft_metric(prs):
    s = blank(prs)
    title_of(s, "The second metric everyone reports is broken",
             kicker="Part 1 · Finding")
    tf = tb(s, M, CY, 6.4, 2.9)
    p = para(tf, True); p.space_after = Pt(12)
    run(p, "Macro-F1 averages the score over every class. Here every one of the 3,110 "
           "items has its own unique label — so there are 3,110 classes with one "
           "item each.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(14)
    run(p, "When the model gives a wrong answer, that wrong label counts as a brand new "
           "class scoring zero. The more different kinds of mistake, the more classes "
           "appear.", size=15, color=GREY)
    p = para(tf)
    run(p, "macro-recall   =   correct  /  number of labels seen",
        size=16, color=DEEP, bold=True, font=HEAD)

    card = rect(s, 7.2, CY - 0.04, 5.53, 2.96, fill=WHITE, line=LINE)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.22); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(8)
    run(p, "You can check it in three lines", size=15, color=INK, bold=True)
    for line, col, bold in [("0.9042 × 3,110  =  2,812 correct", GREY, False),
                            ("2,812 / 0.8379  =  3,356.0  exactly", GREY, False),
                            ("3,356 − 3,110  =  246 invented labels", RED, True)]:
        p = tfc.add_paragraph(); p.space_after = Pt(7)
        run(p, line, size=14, color=col, bold=bold, font=HEAD)
    p = tfc.add_paragraph()
    run(p, "Those 246 came from the model's own wrong answers. It is punished for how "
           "varied its mistakes are, not just how many.", size=12.5, color=GREY)

    takeaway(s, "So macro-F1 here is just **accuracy divided by a number the model "
                "moves itself**. It adds nothing — and every paper reports it.")
    return s


def s_sft_errors(prs):
    s = blank(prs)
    title_of(s, "What the 298 mistakes look like", kicker="Part 1 · Error analysis")
    figure(s, "fig_errors.png", M, CY + 0.1, w=6.8)
    tf = tb(s, 7.75, CY, 4.98, 3.7)
    for i, (lead, rest) in enumerate([
        ("Nothing came out broken. ", "All 3,110 answers were real sense IDs. The "
         "mistakes are genuine confusions, not formatting failures."),
        ("It did not take the shortcut. ", "The right answer is second 65% of the time, "
         "yet its mistakes split evenly. It learned the task, not the trick."),
        ("It fails where the two meanings overlap. ", "Some pairs are near-duplicates "
         "that a person would also struggle with. That caps how high anyone can score."),
    ]):
        p = para(tf, i == 0); p.space_after = Pt(13)
        run(p, lead, size=14.5, color=INK, bold=True)
        run(p, rest, size=14.5, color=GREY)
    takeaway(s, "A 2B model keeps up with 9B because this task is "
                "**telling two visible answers apart, not recalling facts**. "
                "Size buys knowledge — and knowledge is not what was missing.")
    return s
