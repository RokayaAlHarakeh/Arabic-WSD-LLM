# -*- coding: utf-8 -*-
"""Part 2 (CPT) and the conclusion. Imported by deck.py."""
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from build_slides import (
    blank, tb, run, para, rect, title_of, takeaway, stat, bullets, table, figure,
    NAVY, DEEP, INK, GREY, MUTE, TINT, LINE, WHITE, GREEN, RED, HEAD,
    W, H, M, CY, JURY,
)


def s_cpt_setup(prs):
    s = blank(prs)
    title_of(s, "A legal corpus, and a correction to the pipeline",
             kicker="Part 2 · CPT")
    table(s, M, CY, 5.9, [["The corpus", ""],
                          ["Documents · tokens", "43,680  ·  27.2M"],
                          ["Sources", "gazette, legislation, rulings"],
                          ["Split", "90 / 5 / 5 by document"],
                          ["Packed blocks", "13,262 of 2,048 tokens"]],
          [2.9, 3.0], size=13, row_h=0.44)

    card = rect(s, 7.2, CY - 0.04, 5.53, 2.3, fill=WHITE, line=LINE)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.2); tfc.margin_top = Inches(0.16)
    p = tfc.paragraphs[0]; p.space_after = Pt(7)
    run(p, "Why split by document, not by record", size=14.5, color=INK, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(7)
    run(p, "Long documents are stored as overlapping chunks. Splitting by record puts "
           "overlapping text on both sides.", size=13, color=GREY)
    p = tfc.add_paragraph()
    run(p, "Measured: 5.9% of validation and 5.6% of test would have leaked.",
        size=13.5, color=RED, bold=True)

    tf = tb(s, M, 4.32, W - 2 * M, 1.9)
    p = para(tf, True); p.space_after = Pt(9)
    run(p, "A correction to the supplied pipeline. ", size=15, color=INK, bold=True)
    run(p, "The specification grouped records into documents by source_id alone. "
           "Auditing the corpus first showed 396 identifiers shared across different "
           "source collections — none of them the same document.", size=15, color=GREY)
    p = para(tf)
    run(p, "Using the composite key gives 43,680 documents rather than 43,284. Without "
           "it, unrelated documents merge and the held-out guarantee quietly weakens.",
        size=14.5, color=GREY)
    takeaway(s, "The split is the foundation of every held-out number that follows. "
                "**It reproduced byte-identically on three machines.**")
    return s


def s_cpt_training(prs):
    s = blank(prs)
    title_of(s, "One epoch, 3h42m, $2.10", kicker="Part 2 · Training")
    figure(s, "fig_cpt_loss.png", M, CY + 0.06, w=7.3)
    tf = tb(s, 8.2, CY, 4.53, 3.7)
    p = para(tf, True); p.space_after = Pt(12)
    run(p, "Held-out loss fell at all 17 evaluation points with no uptick, so the saved "
           "adapter is genuinely the best of the run.", size=14.5, color=GREY)
    p = para(tf); p.space_after = Pt(12)
    run(p, "What the flat tail does not show. ", size=14.5, color=INK, bold=True)
    run(p, "The cosine schedule drives the learning rate to about 1.5e-8 by the final "
           "steps. A model barely updating cannot visibly improve.", size=14.5, color=GREY)
    p = para(tf)
    run(p, "So the tail is an artefact of the schedule — it is evidence neither for "
           "convergence nor against it.", size=14, color=RED)
    takeaway(s, "27.2M of roughly 134M available tokens, in one epoch. "
                "**Every number that follows is a lower bound.**")
    return s


def s_cpt_results(prs):
    s = blank(prs)
    title_of(s, "Perplexity and next-token prediction", kicker="Part 2 · Result")
    stat(s, M, 1.42, 3.88, "806 → 4.14", "Held-out perplexity", "on legal test documents",
         color=GREEN, h=1.32)
    stat(s, M + 4.08, 1.42, 3.88, "21.3 → 69.8%", "Next-token top-1", "+48.5 points",
         color=GREEN, h=1.32)
    stat(s, M + 8.16, 1.42, 3.93, "1705 → 17", "Rank of the gold token",
         "in a 256k vocabulary", color=GREEN, h=1.32)
    figure(s, "fig_cpt_nexttoken.png", M, 2.88, h=3.24)
    tf = tb(s, 6.30, 2.92, 6.43, 3.2)
    p = para(tf, True); p.space_after = Pt(12)
    run(p, "Every source improves. ", size=15, color=INK, bold=True)
    run(p, "Two sources supply 74% of the corpus, so an aggregate gain could in "
           "principle come from one template-heavy collection.", size=14.5, color=GREY)
    p = para(tf); p.space_after = Pt(12)
    run(p, "It does not. Bibliographic rulings — the shortest and least formulaic "
           "source — gains 41.3 points.", size=14.5, color=GREY)
    p = para(tf)
    run(p, "Hatched bars have fewer than 100 sampled positions and are not individually "
           "interpretable.", size=13, color=MUTE, italic=True)
    takeaway(s, "Perplexity alone would invite the circularity objection. "
                "**Next-token accuracy per source answers it.**")
    return s


def s_cpt_cloze(prs):
    s = blank(prs)
    title_of(s, "Did it learn legal knowledge, or just legal style?",
             kicker="Part 2 · The strict test")
    tf = tb(s, M, CY, 6.2, 1.15)
    p = para(tf, True)
    run(p, "300 cloze items built automatically from held-out documents only. Legal "
           "terms are masked and scored against a fixed candidate set, so chance is "
           "defined and formatting cannot distort the result.", size=14.5, color=GREY)

    table(s, M, 2.76, 6.2, [["Category", "Base", "CPT", "Majority"],
                            ["Collocation", "23.0%", "95.0%", "11.0%"],
                            ["Defined term", "65.0%", "92.0%", "11.0%"],
                            ["Statute instrument", "42.0%", "66.0%", "40.0%"],
                            ["Overall", "43.3%", "84.3%", "14.7%"]],
          [2.3, 1.3, 1.3, 1.3], size=13, row_h=0.44, bold_rows=(4,))

    card = rect(s, 7.05, CY - 0.04, 5.68, 3.5, fill=WHITE, line=LINE)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.22); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(9)
    run(p, "The statute row is the one that matters", size=15, color=INK, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(9)
    run(p, "The correct legal instrument is removed from the context and a competing "
           "one is usually present.", size=14, color=GREY)
    p = tfc.add_paragraph(); p.space_after = Pt(9)
    run(p, "The base model scores 42.0% against a 40.0% majority baseline — it knows "
           "nothing beyond which instrument is commonest.", size=14, color=GREY)
    p = tfc.add_paragraph(); p.space_after = Pt(9)
    run(p, "CPT reaches 66.0%, clearing that baseline by 26 points.",
        size=14.5, color=GREEN, bold=True)
    p = tfc.add_paragraph()
    run(p, "Paired McNemar on the same 300 items: 136 gained, 13 lost, p < 10⁻¹².",
        size=13.5, color=DEEP, bold=True)

    takeaway(s, "72% of the remaining errors are statute items — the expected "
                "residual. **Phrasing is absorbed first; instrument identity is "
                "bounded by corpus scale.**")
    return s


def s_cpt_retention(prs):
    s = blank(prs)
    title_of(s, "Did domain adaptation damage the model?",
             kicker="Part 2 · The decisive result")
    tf = tb(s, M, CY, 12.1, 0.8)
    p = para(tf, True)
    run(p, "The CPT model has never seen a WSD prompt. Evaluated zero-shot on "
           "Dataset A, its accuracy ", size=15, color=GREY)
    run(p, "rose", size=15, color=GREEN, bold=True)
    run(p, " from 24.9% to 36.5% — no forgetting. Read naively, that looks like the "
           "model got better at the task. It did not.", size=15, color=GREY)

    table(s, M, 2.52, 7.4, [["", "Produces an answer", "Correct when it answers", "Overall"],
                            ["Base", "52.5%", "47.4%", "24.9%"],
                            ["CPT", "75.1%", "48.6%", "36.5%"],
                            ["Change", "+22.6 pp", "+1.2 pp  (n.s.)", "+11.6 pp"]],
          [1.5, 2.2, 2.4, 1.3], size=13, row_h=0.46, bold_rows=(3,))

    card = rect(s, 8.25, 2.48, 4.48, 2.5, fill=TINT)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.22); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(9)
    run(p, "The whole gain is format", size=15, color=DEEP, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(9)
    run(p, "Willingness to emit a parseable answer rose 22.6 points.", size=14, color=INK)
    p = tfc.add_paragraph()
    run(p, "Accuracy among answered items moved 1.2 points against a pooled SE of 2.8. "
           "Both confidence intervals contain chance.", size=14, color=GREY)

    tf = tb(s, M, 5.26, W - 2 * M, 1.0)
    p = para(tf, True)
    run(p, "CPT made the model answer — not reason. ", size=15.5, color=INK, bold=True)
    run(p, "An epoch of EOS-terminated blocks dense with citation patterns made it far "
           "likelier to terminate cleanly and emit ID-shaped tokens. It did not make it "
           "better at choosing between two glosses.", size=15, color=GREY)
    takeaway(s, "This was the one pre-registered prediction that came out **wrong** — "
                "and chasing it produced the central result of the project.")
    return s


# ============================================================ CONCLUSION ===
def s_dissociation(prs):
    s = blank(prs)
    title_of(s, "What each objective actually changed", kicker="Conclusion")
    table(s, M, CY, W - 2 * M, [["", "Task competence", "Domain distribution"],
                                ["Supervised fine-tuning",
                                 "CHANGED   24.9% → 90.42%", "unchanged — no domain data"],
                                ["Continued pre-training",
                                 "unchanged   47.4% → 48.6%", "CHANGED   PPL 806 → 4.14"]],
          [3.9, 4.6, 3.59], size=14, row_h=0.72, head_h=0.48)

    tf = tb(s, M, 3.52, W - 2 * M, 1.5)
    p = para(tf, True); p.space_after = Pt(9)
    run(p, "A double dissociation, measured rather than argued. ", size=16,
        color=INK, bold=True)
    run(p, "The weak version of this claim would be that CPT improved on legal metrics "
           "and SFT on WSD metrics, each on its own ground — which is circular.",
        size=15, color=GREY)
    p = para(tf)
    run(p, "The retention decomposition avoids that: both objectives are measured on "
           "the same task, the same test set, and the same split of behaviour into "
           "format and discrimination.", size=15, color=GREY)

    card = rect(s, M, 5.18, W - 2 * M, 1.14, fill=TINT)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.24); tfc.margin_top = Inches(0.14)
    tfc.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tfc.paragraphs[0]
    run(p, "The conclusion holds independently of effect size. ", size=15,
        color=DEEP, bold=True)
    run(p, "Even had the CPT gain been modest, moving format compliance by 22.6 points "
           "while moving discrimination by 1.2 would carry the same meaning.",
        size=15, color=DEEP)
    return s


def s_usecase(prs):
    s = blank(prs)
    title_of(s, "What each bought, for these two use cases", kicker="Conclusion")

    c1 = rect(s, M, CY, 6.0, 3.52, fill=WHITE, line=LINE)
    tf = c1.text_frame
    tf.margin_left = tf.margin_right = Inches(0.26); tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]; p.space_after = Pt(6)
    run(p, "SFT — Arabic WSD", size=17, color=NAVY, bold=True, font=HEAD)
    p = tf.add_paragraph(); p.space_after = Pt(10)
    run(p, "the information was already in the prompt", size=13, color=MUTE, italic=True)
    for lead, rest in [("90.42% accuracy, ", "matching published 8B models and beating "
                        "the same-family 9B."),
                       ("25.5 points over the real floor ", "of 64.95%, not 40 over 50."),
                       ("Zero malformed outputs, ", "no positional bias."),
                       ("Cost: ", "a free Colab T4.")]:
        p = tf.add_paragraph(); p.space_after = Pt(9)
        run(p, "—  ", size=14, color=NAVY, bold=True)
        run(p, lead, size=14, color=INK, bold=True)
        run(p, rest, size=14, color=GREY)
    p = tf.add_paragraph()
    run(p, "Verdict: SFT was the right and sufficient tool. Capacity was never the "
           "bottleneck — format was.", size=14, color=DEEP, bold=True)

    c2 = rect(s, 7.0, CY, 5.73, 3.52, fill=WHITE, line=LINE)
    tf = c2.text_frame
    tf.margin_left = tf.margin_right = Inches(0.26); tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]; p.space_after = Pt(6)
    run(p, "CPT — Lebanese legal Arabic", size=17, color=NAVY, bold=True, font=HEAD)
    p = tf.add_paragraph(); p.space_after = Pt(10)
    run(p, "the knowledge had to come from the weights", size=13, color=MUTE, italic=True)
    for lead, rest in [("Perplexity 806 → 4.14, ", "next-token +48.5 points, every "
                        "source improved."),
                       ("Cloze 43.3 → 84.3%, ", "p < 10⁻¹² on a paired test."),
                       ("No forgetting — ", "general ability rose, not fell."),
                       ("Cost: ", "$5.40 of rented GPU.")]:
        p = tf.add_paragraph(); p.space_after = Pt(9)
        run(p, "—  ", size=14, color=NAVY, bold=True)
        run(p, lead, size=14, color=INK, bold=True)
        run(p, rest, size=14, color=GREY)
    p = tf.add_paragraph()
    run(p, "Verdict: CPT bought domain fluency and vocabulary — but no task ability "
           "whatsoever.", size=14, color=DEEP, bold=True)

    takeaway(s, "If the information is in the prompt, use SFT. If it must come from the "
                "weights, use CPT. **A legal WSD system would need both, in that order.**",
             y=5.42)
    return s


def s_limits(prs):
    s = blank(prs)
    title_of(s, "What this does not establish, and what comes next",
             kicker="Limitations · Future work")
    tfh = tb(s, M, CY, 6.0, 0.3)
    run(para(tfh, True), "Limitations", size=16, color=INK, bold=True)
    bullets(s, M, CY + 0.42, 5.9, [
        ("No matched-budget control. ", "No instruction-SFT run on the legal corpus at "
         "equal tokens, so the SFT/CPT comparison is qualitative."),
        ("One run, one seed, rank 16. ", "Whether more adapter capacity would absorb "
         "more of the domain is untested."),
        ("27.2M of ~134M tokens. ", "Corpus scale bounds the effect."),
        ("Perplexity measures register. ", "Legal text is formulaic; the cloze probe is "
         "the better evidence of knowledge."),
    ], size=13.5, gap=7)

    tfh2 = tb(s, 7.05, CY, 5.7, 0.3)
    run(para(tfh2, True), "Future work", size=16, color=INK, bold=True)
    bullets(s, 7.05, CY + 0.42, 5.68, [
        ("The matched-budget control. ", "The single most valuable missing experiment."),
        ("The full corpus. ", "Five times the tokens, to see how much was left."),
        ("Train the embeddings. ", "Legal terms fragment at 2.337 tokens per word."),
        ("Rank and quantization ablation, ", "and an encoder cross-encoder baseline on "
         "the same split."),
    ], size=13.5, gap=7)

    takeaway(s, "Three measurement faults were caught by checking numbers that looked "
                "**good**: a document key that merged unrelated documents, an extractor "
                "that reported 0.0% for a model that was answering, and a decoding confound.")
    return s


def s_thanks(prs, v):
    s = blank(prs)
    bg = rect(s, 0, 0, W, H, fill=DEEP, shape=MSO_SHAPE.RECTANGLE)
    bg.shadow.inherit = False
    tf = tb(s, M, 2.5, W - 2 * M, 1.0, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER
    run(p, "Thank you", size=46, color=WHITE, bold=True, font=HEAD)
    tf = tb(s, M, 3.62, W - 2 * M, 0.5, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER
    run(p, "Questions", size=20, color=RED if False else WHITE)
    tf = tb(s, M, 4.6, W - 2 * M, 1.6, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER; p.space_after = Pt(8)
    run(p, "Rokaya Al Harakeh", size=16, color=WHITE, bold=True)
    p = para(tf); p.alignment = PP_ALIGN.CENTER
    run(p, "  ·  ".join([t for t in v["institutions"] if t][:2]),
        size=13, color=RGB_LIGHT())
    return s


def RGB_LIGHT():
    from pptx.dml.color import RGBColor
    return RGBColor(0xC9, 0xD8, 0xE6)
