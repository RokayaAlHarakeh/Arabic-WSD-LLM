# -*- coding: utf-8 -*-
"""
Build both defense decks.

    cd slides && ../venv/Scripts/python.exe make_deck.py
"""
import os

from pptx import Presentation
from pptx.util import Inches

from build_slides import VARIANTS, footer, divider, W, H
import deck as D1
import deck_part2 as D2

HERE = os.path.dirname(os.path.abspath(__file__))

NOTES = {
    1: "Start concrete. Read the Arabic sentence, point at the two meanings, say which "
       "one is right. Then the key line: both answers are already in front of the model.",
    2: "Pre-training is Google's job and costs millions. Fine-tuning is ours and costs "
       "hours. Two ways to do it - that sets up the whole talk.",
    3: "One number against another: 41.7 against 16. Then say the memory goes on "
       "bookkeeping, not on the model.",
    5: "This is the pivot. Everything before it is setup, everything after is evidence "
       "for the bottom row. Read the four goals quickly.",
    7: "Show the prompt, then land the point: both answers are inside it.",
    8: "Lead with 90.42 matching 8B. Then the margin of error - say plainly you are not "
       "claiming to be better.",
    9: "Strongest finding of Part 1. Nobody reports a baseline. 64.95 percent with no "
       "model at all.",
    10: "Walk the three lines on the right slowly. 2812 over 0.8379 is exactly 3356.",
    13: "Show the Arabic document, then the bug you caught before running anything.",
    16: "The third row. Before training it was level with guessing - so it knew nothing.",
    17: "The most important slide. The score went up, but all of the gain is answering "
        "more often, none of it is being right more often.",
    18: "Read the table across. Then: this holds whatever the effect size.",
    19: "This answers 'what did each one buy me'. Left, right, verdict under each.",
}


def build(variant_key):
    v = VARIANTS[variant_key]
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    slides = [D1.s_title(prs, v)]
    slides.append(D1.s_task(prs))          # one real item, in Arabic
    slides.append(D1.s_what_is_ft(prs))    # what fine-tuning is, before the obstacle
    slides.append(D1.s_finetuning(prs))
    slides.append(D1.s_lora(prs))
    slides.append(D1.s_two(prs))           # SFT vs CPT + the four goals

    slides.append(divider(prs, "1", "Supervised fine-tuning", [
        "Arabic Word Sense Disambiguation, Dataset A",
        "Benchmarked against a published study",
        "What a 2B model can do when the answer is already in the prompt"]))
    slides.append(D1.s_sft_setup(prs))
    slides.append(D1.s_sft_headline(prs))
    slides.append(D1.s_sft_baselines(prs))
    slides.append(D1.s_sft_metric(prs))
    slides.append(D1.s_sft_errors(prs))

    slides.append(divider(prs, "2", "Continued pre-training", [
        "27.2 million tokens of Lebanese legal Arabic",
        "Benchmarked against the untrained base model",
        "What changes when the knowledge has to come from the weights"]))
    slides.append(D2.s_cpt_setup(prs))
    slides.append(D2.s_cpt_training(prs))
    slides.append(D2.s_cpt_results(prs))
    slides.append(D2.s_cpt_cloze(prs))
    slides.append(D2.s_cpt_retention(prs))

    slides.append(D2.s_dissociation(prs))
    slides.append(D2.s_usecase(prs))
    slides.append(D2.s_limits(prs))
    slides.append(D2.s_thanks(prs, v))

    # footers on content slides only -- not the title, dividers or the closing slide
    plain = {0, 6, 12, len(slides) - 1}
    for i, s in enumerate(slides):
        if i not in plain:
            footer(s, i + 1)
        if i in NOTES:
            s.notes_slide.notes_text_frame.text = NOTES[i]

    out = os.path.join(HERE, v["file"])
    prs.save(out)
    print("wrote %-26s %2d slides" % (v["file"], len(slides)))
    return out


if __name__ == "__main__":
    for k in ("M2", "ULFG"):
        build(k)
