# -*- coding: utf-8 -*-
"""Part 2 (CPT) and the conclusion. Plain language; the numbers do the work."""
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

from build_slides import (
    blank, tb, run, para, rect, title_of, takeaway, stat, bullets, table, figure, ar,
    NAVY, DEEP, INK, GREY, MUTE, TINT, LINE, WHITE, GREEN, RED, HEAD,
    W, H, M, CY,
)

CODE_BG = RGBColor(0xF7, 0xF9, 0xFB)

# a real excerpt: a decree moving budget from the education ministry to build a school
LEGAL = ("تحويل اعتماد من "
         "موازنة وزارة "
         "التربية والتعليم "
         "العالي لعام 2004 "
         "الى مجلس "
         "الانماء والاعمار")
LEGAL2 = ("مرسوم رقم 13934 · "
          "تاريخ 04/01/2005 · "
          "الجريدة الرسمية")


def s_cpt_setup(prs):
    s = blank(prs)
    title_of(s, "The corpus, and what it looks like", kicker="Part 2 · CPT")

    card = rect(s, M, CY - 0.02, 7.1, 1.86, fill=CODE_BG, line=LINE)
    tf = card.text_frame
    tf.margin_left = tf.margin_right = Inches(0.24); tf.margin_top = Inches(0.16)
    p = tf.paragraphs[0]; p.space_after = Pt(8)
    run(p, "A TYPICAL DOCUMENT — NO LABELS, JUST TEXT", size=11, color=MUTE, bold=True)
    p = tf.add_paragraph(); p.space_after = Pt(8)
    ar(p, LEGAL, size=15, color=INK)
    p = tf.add_paragraph()
    ar(p, LEGAL2, size=13, color=GREY)

    table(s, 7.86, CY - 0.02, 4.87, [["The corpus", ""],
                                     ["Documents", "43,680"],
                                     ["Words", "27.2 million"],
                                     ["Sources", "gazette, laws, rulings"],
                                     ["Split", "90 / 5 / 5"]],
          [2.2, 2.67], size=13, row_h=0.42)

    tf = tb(s, M, 3.66, 6.1, 2.3)
    p = para(tf, True); p.space_after = Pt(11)
    run(p, "We hide 5% of the documents and never train on them. ", size=15,
        color=INK, bold=True)
    run(p, "Every number later is measured only on those.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(11)
    run(p, "Hidden by document, not by paragraph. ", size=15, color=INK, bold=True)
    run(p, "Long documents are stored in overlapping pieces. Split them carelessly and "
           "the same text lands on both sides — you end up testing the model on what "
           "it already read.", size=15, color=GREY)
    p = para(tf)
    run(p, "Measured: 5.9% of the hidden set would have leaked that way.",
        size=14.5, color=RED, bold=True)

    card2 = rect(s, 7.0, 3.66, 5.73, 2.46, fill=WHITE, line=LINE)
    tfc = card2.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.24); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(10)
    run(p, "A bug I found in the supplied pipeline", size=15, color=INK, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(10)
    run(p, "The plan said to group pieces into documents by their ID. I checked first: "
           "396 IDs are reused across different collections, and none are really the "
           "same document.", size=14, color=GREY)
    p = tfc.add_paragraph()
    run(p, "Unrelated documents would have been glued together. Using collection + ID "
           "gives 43,680 documents, not 43,284.", size=14, color=DEEP, bold=True)

    takeaway(s, "The split is the foundation of every later number. "
                "**It came out identical on three different machines.**")
    return s


def s_cpt_training(prs):
    s = blank(prs)
    title_of(s, "One pass over the corpus: 3h42m, $2.10", kicker="Part 2 · Training")
    figure(s, "fig_cpt_loss.png", M, CY + 0.06, w=7.3)
    tf = tb(s, 8.2, CY, 4.53, 3.7)
    p = para(tf, True); p.space_after = Pt(14)
    run(p, "The red line is error on documents the model never saw. It falls at all 17 "
           "checkpoints and never turns back up — so it was still learning, not just "
           "memorising.", size=14.5, color=GREY)
    p = para(tf); p.space_after = Pt(14)
    run(p, "What the flat ending does not mean. ", size=14.5, color=INK, bold=True)
    run(p, "The learning rate is wound down to almost nothing by the end, on purpose. "
           "A model that has stopped changing cannot show improvement.",
        size=14.5, color=GREY)
    p = para(tf)
    run(p, "So the flat tail tells us nothing either way. That is the schedule, not "
           "the model.", size=14, color=RED)
    takeaway(s, "We used 27 million of about 134 million available words, once. "
                "**Everything that follows is a floor, not a ceiling.**")
    return s


def s_cpt_results(prs):
    s = blank(prs)
    title_of(s, "Does it read legal Arabic better?", kicker="Part 2 · Result")
    stat(s, M, 1.42, 3.88, "806 → 4.14", "Perplexity",
         "how surprised it is by legal text", color=GREEN, h=1.32)
    stat(s, M + 4.08, 1.42, 3.88, "21 → 70%", "Next word predicted right",
         "+48.5 points", color=GREEN, h=1.32)
    stat(s, M + 8.16, 1.42, 3.93, "1705 → 17", "Where the right word ranked",
         "out of 256,000 options", color=GREEN, h=1.32)
    figure(s, "fig_cpt_nexttoken.png", M, 2.88, h=3.24)
    tf = tb(s, 6.30, 2.92, 6.43, 3.2)
    p = para(tf, True); p.space_after = Pt(14)
    run(p, "On its own this would be a weak claim. ", size=15, color=INK, bold=True)
    run(p, "Of course a model gets better at text it was trained on. So we split the "
           "result by source.", size=15, color=GREY)
    p = para(tf); p.space_after = Pt(14)
    run(p, "Every single source improved. ", size=15, color=GREEN, bold=True)
    run(p, "Two sources are 74% of the corpus, so the whole gain could have come from "
           "repetitive boilerplate. It did not — court rulings, the shortest and least "
           "formulaic source, gained 41 points.", size=15, color=GREY)
    p = para(tf)
    run(p, "Striped bars have very few test points and should not be read alone.",
        size=13, color=MUTE, italic=True)
    takeaway(s, "It reads legal Arabic far better. "
                "**But does it know anything, or has it only learned the style?**")
    return s


def s_cpt_cloze(prs):
    s = blank(prs)
    title_of(s, "Knowledge, or just style?", kicker="Part 2 · The strict test")
    tf = tb(s, M, CY - 0.02, 6.2, 1.05)
    p = para(tf, True)
    run(p, "We hide a legal term in a sentence the model has never seen, and ask it to "
           "fill the gap from a fixed list of options. A fixed list means we know "
           "exactly what pure guessing would score.", size=14.5, color=GREY)

    table(s, M, 2.72, 6.2, [["What we hide", "Before", "After", "Guessing"],
                            ["A set phrase", "23.0%", "95.0%", "11.0%"],
                            ["A defined term", "65.0%", "92.0%", "11.0%"],
                            ["Which law is cited", "42.0%", "66.0%", "40.0%"],
                            ["Overall", "43.3%", "84.3%", "14.7%"]],
          [2.3, 1.3, 1.3, 1.3], size=13, row_h=0.44, bold_rows=(4,))

    card = rect(s, 7.05, CY - 0.04, 5.68, 3.46, fill=WHITE, line=LINE)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.22); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(10)
    run(p, "The third row is the one that matters", size=15, color=INK, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(10)
    run(p, "We remove the name of the law being cited — and usually a different law is "
           "mentioned nearby, to mislead it.", size=14, color=GREY)
    p = tfc.add_paragraph(); p.space_after = Pt(10)
    run(p, "Before training it scored 42%. Simply always guessing the commonest answer "
           "scores 40%. So it knew essentially nothing.", size=14, color=GREY)
    p = tfc.add_paragraph(); p.space_after = Pt(10)
    run(p, "After training: 66% — 26 points clear of guessing.",
        size=14.5, color=GREEN, bold=True)
    p = tfc.add_paragraph()
    run(p, "Same 300 questions for both. The chance of that being luck is below one "
           "in a trillion.", size=13.5, color=DEEP, bold=True)

    takeaway(s, "So it learned which law is which — not only how legal writing sounds. "
                "**Most of what it still gets wrong is in that hardest row, which is "
                "exactly what you would expect.**")
    return s


def s_cpt_retention(prs):
    s = blank(prs)
    title_of(s, "Did teaching it law break anything?",
             kicker="Part 2 · The decisive result")
    tf = tb(s, M, CY, 12.1, 0.8)
    p = para(tf, True)
    run(p, "We gave the legal model the Part 1 word-sense task, which it was never "
           "trained on. Its score ", size=15, color=GREY)
    run(p, "went up", size=15, color=GREEN, bold=True)
    run(p, ", from 24.9% to 36.5%. Nothing was damaged. At first glance it even looks "
           "like it got better at the task. It did not.", size=15, color=GREY)

    table(s, M, 2.52, 7.4, [["", "Answered at all", "Right when it answered", "Score"],
                            ["Before", "52.5%", "47.4%", "24.9%"],
                            ["After", "75.1%", "48.6%", "36.5%"],
                            ["Change", "+22.6 pts", "+1.2 pts  (noise)", "+11.6 pts"]],
          [1.5, 2.3, 2.3, 1.3], size=13, row_h=0.46, bold_rows=(3,))

    card = rect(s, 8.25, 2.48, 4.48, 2.5, fill=TINT)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.22); tfc.margin_top = Inches(0.18)
    p = tfc.paragraphs[0]; p.space_after = Pt(10)
    run(p, "The whole gain is willingness", size=15, color=DEEP, bold=True)
    p = tfc.add_paragraph(); p.space_after = Pt(10)
    run(p, "It answers far more often — up 22.6 points.", size=14, color=INK)
    p = tfc.add_paragraph()
    run(p, "But when it does answer it is no better than before, and no better than a "
           "coin flip. Those 1.2 points are noise.", size=14, color=GREY)

    tf = tb(s, M, 5.26, W - 2 * M, 1.0)
    p = para(tf, True)
    run(p, "Reading law made it answer — not think. ", size=15.5, color=INK, bold=True)
    run(p, "A corpus full of “Decree no. 1234” made it far likelier to stop cleanly "
           "and produce something number-shaped. It did not make it any better at "
           "choosing between two meanings.", size=15, color=GREY)
    takeaway(s, "I wrote three predictions down before training. This is the one I got "
                "**wrong** — and chasing it produced the main result of the project.")
    return s


# ============================================================ CONCLUSION ===
def s_dissociation(prs):
    s = blank(prs)
    title_of(s, "What each one actually changed", kicker="Conclusion")
    table(s, M, CY, W - 2 * M,
          [["", "Ability to do the task", "Knowledge of the field"],
           ["Worked examples (SFT)", "CHANGED   24.9% → 90.42%",
            "unchanged — saw no legal text"],
           ["Reading the field (CPT)", "unchanged   47.4% → 48.6%",
            "CHANGED   806 → 4.14"]],
          [3.9, 4.6, 3.59], size=14, row_h=0.72, head_h=0.48)

    tf = tb(s, M, 3.52, W - 2 * M, 1.5)
    p = para(tf, True); p.space_after = Pt(10)
    run(p, "Each one moves a different column and leaves the other alone. ", size=16,
        color=INK, bold=True)
    run(p, "The weak way to argue this would be: the legal model did better on legal "
           "tests, the word-sense model did better on word-sense tests. That is "
           "circular — each was only measured on its own ground.", size=15, color=GREY)
    p = para(tf)
    run(p, "The last slide of Part 2 avoids that. Both are measured on the same task, "
           "the same questions, and the same split between answering and being right.",
        size=15, color=GREY)

    card = rect(s, M, 5.18, W - 2 * M, 1.14, fill=TINT)
    tfc = card.text_frame
    tfc.margin_left = tfc.margin_right = Inches(0.24); tfc.margin_top = Inches(0.14)
    tfc.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tfc.paragraphs[0]
    run(p, "And this holds however big the gains had turned out to be. ", size=15,
        color=DEEP, bold=True)
    run(p, "Even if the legal training had barely worked, moving willingness by 22.6 "
           "points while moving real ability by 1.2 would mean the same thing.",
        size=15, color=DEEP)
    return s


def s_usecase(prs):
    s = blank(prs)
    title_of(s, "What each one bought me, for each job", kicker="Conclusion")

    c1 = rect(s, M, CY, 6.0, 3.4, fill=WHITE, line=LINE)
    tf = c1.text_frame
    tf.margin_left = tf.margin_right = Inches(0.26); tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]; p.space_after = Pt(5)
    run(p, "Worked examples — Arabic word senses", size=16, color=NAVY, bold=True,
        font=HEAD)
    p = tf.add_paragraph(); p.space_after = Pt(11)
    run(p, "the answer was already in the question", size=13, color=MUTE, italic=True)
    for lead, rest in [("90.42%, ", "level with 8B models and ahead of the 9B one."),
                       ("25 points above the real floor ", "of 64.95%."),
                       ("Nothing malformed, ", "and it took no shortcuts."),
                       ("Cost: ", "a free GPU.")]:
        p = tf.add_paragraph(); p.space_after = Pt(8)
        run(p, "—  ", size=14, color=NAVY, bold=True)
        run(p, lead, size=14, color=INK, bold=True)
        run(p, rest, size=14, color=GREY)
    p = tf.add_paragraph()
    run(p, "Verdict: the right tool, and enough on its own. Size was never the problem "
           "here — format was.", size=14, color=DEEP, bold=True)

    c2 = rect(s, 7.0, CY, 5.73, 3.4, fill=WHITE, line=LINE)
    tf = c2.text_frame
    tf.margin_left = tf.margin_right = Inches(0.26); tf.margin_top = Inches(0.2)
    p = tf.paragraphs[0]; p.space_after = Pt(5)
    run(p, "Reading the field — Lebanese law", size=16, color=NAVY, bold=True,
        font=HEAD)
    p = tf.add_paragraph(); p.space_after = Pt(11)
    run(p, "the answer had to come from memory", size=13, color=MUTE, italic=True)
    for lead, rest in [("Perplexity 806 → 4.14, ", "and every source improved."),
                       ("Knows which law is which: ", "42% → 66%, clear of guessing."),
                       ("Nothing was damaged — ", "general ability rose, not fell."),
                       ("Cost: ", "$5.40 of rented GPU.")]:
        p = tf.add_paragraph(); p.space_after = Pt(8)
        run(p, "—  ", size=14, color=NAVY, bold=True)
        run(p, lead, size=14, color=INK, bold=True)
        run(p, rest, size=14, color=GREY)
    p = tf.add_paragraph()
    run(p, "Verdict: bought fluency and vocabulary in the field — but no ability to "
           "do a task.", size=14, color=DEEP, bold=True)

    takeaway(s, "If the answer is already in the question, use worked examples. If it "
                "has to come from memory, let it read the field. "
                "**A legal word-sense system would need both, in that order.**",
             y=5.22)
    return s


def s_limits(prs):
    s = blank(prs)
    title_of(s, "What this does not prove, and what I would do next",
             kicker="Limitations · Future work")
    tfh = tb(s, M, CY, 6.0, 0.3)
    run(para(tfh, True), "What it does not prove", size=16, color=INK, bold=True)
    bullets(s, M, CY + 0.42, 5.9, [
        ("The two halves are not a fair race. ", "Different data, different tests. I "
         "compare what each changes, never which one wins."),
        ("One run, one setting. ", "A bigger adapter might have absorbed more law — "
         "I cannot say."),
        ("27 of 134 million words. ", "The corpus size caps the effect."),
        ("Legal writing is repetitive, ", "so perplexity flatters it. The hidden-term "
         "test is the honest evidence."),
    ], size=13.5, gap=6)

    tfh2 = tb(s, 7.05, CY, 5.7, 0.3)
    run(para(tfh2, True), "What I would do next", size=16, color=INK, bold=True)
    bullets(s, 7.05, CY + 0.42, 5.68, [
        ("The fair race: ", "worked examples on the legal corpus at the same word "
         "budget. The one missing experiment."),
        ("The full corpus — ", "five times the text."),
        ("Let it learn new words properly. ", "The tokenizer chops legal terms into "
         "pieces."),
        ("Bigger adapters, ", "and an older-style model as a control."),
    ], size=13.5, gap=6)

    takeaway(s, "Three measuring mistakes were caught by checking numbers that looked "
                "**good**: a key that merged unrelated documents, a scorer that reported "
                "0% for a model that was in fact answering, and a setting that flattered "
                "the trained model.")
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
    run(p, "Questions", size=20, color=RGBColor(0xC9, 0xD8, 0xE6))
    tf = tb(s, M, 4.6, W - 2 * M, 1.6, align=PP_ALIGN.CENTER)
    p = para(tf, True); p.alignment = PP_ALIGN.CENTER; p.space_after = Pt(8)
    run(p, "Rokaya Al Harakeh", size=16, color=WHITE, bold=True)
    p = para(tf); p.alignment = PP_ALIGN.CENTER
    run(p, "  ·  ".join([t for t in v["institutions"] if t][:2]),
        size=13, color=RGBColor(0xC9, 0xD8, 0xE6))
    return s
