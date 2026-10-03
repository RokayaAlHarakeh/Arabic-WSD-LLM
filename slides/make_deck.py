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
    2: "Open here. The task in one sentence, then the four objectives. Do not dwell - "
       "the numbers are the talk.",
    3: "The memory wall is why everything that follows is QLoRA. One minute.",
    5: "This is the pivot slide. Everything before it is setup; everything after it is "
       "evidence for the last row.",
    8: "Lead with 90.42 matching 8B. Then immediately the confidence interval - say "
       "plainly that you are not claiming to be better.",
    9: "The strongest contribution of Part 1. Nobody in the literature reports a "
       "baseline. 64.95 percent with no model at all.",
    10: "Show the arithmetic on the right. 2812 over 0.8379 is exactly 3356. That is "
        "the whole argument.",
    14: "The statute row is the one to narrate. Base is level with its own majority "
        "baseline, so it knows nothing; CPT clears it by 26 points.",
    15: "The most important slide in Part 2. Accuracy rose, but decomposing it shows "
        "the gain is entirely format. Say that CPT made it answer, not reason.",
    16: "The conclusion. Read the table across, then make the point that the claim "
        "survives regardless of effect size.",
    17: "This answers 'what did each one buy me'. Left SFT, right CPT, verdict at the "
        "bottom of each.",
}


def build(variant_key):
    v = VARIANTS[variant_key]
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)

    slides = [D1.s_title(prs, v)]
    slides.append(D1.s_objectives(prs))
    slides.append(D1.s_finetuning(prs))
    slides.append(D1.s_lora(prs))
    slides.append(D1.s_two(prs))

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
    plain = {0, 5, 11, len(slides) - 1}
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
