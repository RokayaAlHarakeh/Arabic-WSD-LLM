# -*- coding: utf-8 -*-
"""
Defense deck — both variants.

    ../venv/Scripts/python.exe build_slides.py

Writes two files that differ only in the second logo and the institution block:

    FYP_Defense_M2.pptx     LU + Faculty of Sciences + WhiteStork
    FYP_Defense_ULFG.pptx   LU + Faculty of Engineering + WhiteStork

Logos are read from ../thesis_word/figures/. A missing one leaves a visible
placeholder naming the file it wants, so the gap is obvious and nothing shifts.

Figures are the report's own PNGs, so the deck and the thesis cannot disagree.
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "thesis_word", "figures")

# ---- palette ---------------------------------------------------------------
NAVY = RGBColor(0x1F, 0x4E, 0x79)
DEEP = RGBColor(0x14, 0x33, 0x50)
INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x5A, 0x5A, 0x5A)
MUTE = RGBColor(0x8A, 0x8A, 0x8A)
TINT = RGBColor(0xEE, 0xF3, 0xF8)
LINE = RGBColor(0xD5, 0xDD, 0xE5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
RED = RGBColor(0xC0, 0x00, 0x00)

HEAD = "Cambria"
BODY = "Calibri"

W, H = 13.333, 7.5
M = 0.62                 # slide margin
TY = 0.52                # title band y
CY = 1.52                # content start y

# placeholders the author fills once the dates are fixed
DEFENSE_DATE = "[ Defense date ]"
JURY = ["Dr. [ Jury member 1 ]", "Dr. [ Jury member 2 ]", "Dr. [ Jury member 3 ]"]

VARIANTS = {
    "M2": {
        "file": "FYP_Defense_M2.pptx",
        "second_logo": "logo_institute.png",
        "banner": "MASTER 2 — FINAL YEAR PROJECT DEFENSE",
        "institutions": ["Lebanese University — Faculty of Sciences",
                         "Professional Master in Artificial Intelligence",
                         "and Data Engineering", "WhiteStork Software Solutions"],
    },
    "ULFG": {
        "file": "FYP_Defense_ULFG.pptx",
        "second_logo": "ulfg_logo.png",
        "banner": "FINAL YEAR PROJECT DEFENSE",
        "institutions": ["Lebanese University — Faculty of Engineering III",
                         "Computer and Communication Engineering",
                         "", "WhiteStork Software Solutions"],
    },
}


# ---- primitives ------------------------------------------------------------
def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return tf


def run(p, text, size=16, color=INK, bold=False, italic=False, font=BODY, space=0):
    r = p.add_run()
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color
    return r


def para(tf, first=False):
    return tf.paragraphs[0] if first else tf.add_paragraph()


def rect(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         radius=0.04):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(0.75)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            s.adjustments[0] = radius
        except (IndexError, AttributeError):
            pass
    s.shadow.inherit = False
    s.text_frame.word_wrap = True
    return s


def logo(slide, stem, x, y, h):
    """Place figures/<stem>; if absent, a labelled placeholder of the same height."""
    path = os.path.join(FIG, stem)
    if os.path.exists(path):
        from PIL import Image
        iw, ih = Image.open(path).size
        w = h * iw / ih
        slide.shapes.add_picture(path, Inches(x), Inches(y), height=Inches(h))
        return w
    w = h * 1.9
    box = rect(slide, x, y, w, h, fill=RGBColor(0xFB, 0xEA, 0xEA), line=RED)
    tf = box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, "logo missing:\n" + stem, size=7, color=RED, italic=True)
    return w


ARFONT = "Arial"          # ships with Office, shapes Arabic correctly on a projector


def ar(p, text, size=16, color=INK, bold=False, align_right=True):
    """An Arabic run in a right-to-left paragraph.

    python-pptx has no API for this, so the two attributes go straight onto the
    paragraph properties. Without rtl the glyphs still shape, but punctuation and
    any Latin or digits inside the line come out in the wrong order.
    """
    pPr = p._p.get_or_add_pPr()
    pPr.set("rtl", "1")
    if align_right:
        pPr.set("algn", "r")
    return run(p, text, size=size, color=color, bold=bold, font=ARFONT)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def title_of(slide, text, kicker=None):
    """Content-slide title. Same place, size and alignment on every content slide."""
    if kicker:
        tf = tb(slide, M, TY - 0.04, W - 2 * M, 0.3)
        run(para(tf, True), kicker.upper(), size=11.5, color=NAVY, bold=True)
        tf2 = tb(slide, M, TY + 0.28, W - 2 * M, 0.72)
        run(para(tf2, True), text, size=30, color=INK, bold=True, font=HEAD)
    else:
        tf = tb(slide, M, TY, W - 2 * M, 0.8)
        run(para(tf, True), text, size=32, color=INK, bold=True, font=HEAD)


def takeaway(slide, text, y=6.24):
    """One-line conclusion strip at the foot of a result slide."""
    box = rect(slide, M, y, W - 2 * M, 0.60, fill=TINT)
    tf = box.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.22)
    tf.margin_right = Inches(0.18)
    p = tf.paragraphs[0]
    bits = text.split("**")
    for i, b in enumerate(bits):
        if b:
            run(p, b, size=14.5, color=DEEP, bold=(i % 2 == 1))
    return box


def stat(slide, x, y, w, value, label, sub=None, color=NAVY, h=1.72):
    card = rect(slide, x, y, w, h, fill=WHITE, line=LINE)
    tf = card.text_frame
    tf.margin_left = tf.margin_right = Inches(0.16)
    tf.margin_top = Inches(0.14)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run(p, value, size=40, color=color, bold=True, font=HEAD)
    p2 = tf.add_paragraph()
    run(p2, label, size=12.5, color=INK, bold=True)
    if sub:
        p3 = tf.add_paragraph()
        run(p3, sub, size=10.5, color=MUTE)
    return card


def bullets(slide, x, y, w, items, size=15.5, gap=10, h=None):
    """items: list of (bold_lead, rest) or plain strings."""
    tf = tb(slide, x, y, w, h if h else min(4.4, H - 0.72 - y))
    for i, it in enumerate(items):
        p = para(tf, i == 0)
        p.space_after = Pt(gap)
        run(p, "—  ", size=size, color=NAVY, bold=True)
        if isinstance(it, tuple):
            run(p, it[0], size=size, color=INK, bold=True)
            run(p, it[1], size=size, color=GREY)
        else:
            run(p, it, size=size, color=GREY)
    return tf


def table(slide, x, y, w, rows, col_w, head_fill=DEEP, row_h=0.42, head_h=0.44,
          size=13, bold_rows=()):
    n_r, n_c = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                                   Inches(head_h + row_h * (n_r - 1)))
    t = shape.table
    for j, cw in enumerate(col_w):
        t.columns[j].width = Emu(int(Inches(cw)))
    t.rows[0].height = Inches(head_h)
    for i in range(1, n_r):
        t.rows[i].height = Inches(row_h)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            c.margin_left = c.margin_right = Inches(0.1)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = head_fill
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else TINT
            p = c.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            txt = str(val)
            col = WHITE if i == 0 else INK
            bold = (i == 0) or (i in bold_rows)
            if txt.startswith("+") and i != 0:
                col = GREEN
                bold = True
            elif txt.startswith("↓") or txt.startswith("−"):
                col = GREEN
                bold = True
            run(p, txt, size=size, color=col, bold=bold)
    return t


def figure(slide, name, x, y, w=None, h=None):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        box = rect(slide, x, y, w or 5, h or 3, fill=RGBColor(0xFB, 0xEA, 0xEA), line=RED)
        run(box.text_frame.paragraphs[0], "figure missing: " + name, size=10, color=RED)
        return
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)


def footer(slide, n):
    tf = tb(slide, M, H - 0.46, 8.0, 0.26)
    run(para(tf, True), "Parameter-Efficient Adaptation of Gemma 2-2B  ·  "
                        "Arabic WSD and Lebanese Legal CPT", size=10, color=MUTE)
    tf2 = tb(slide, W - M - 1.0, H - 0.46, 1.0, 0.26, align=PP_ALIGN.RIGHT)
    run(para(tf2, True), str(n), size=10, color=MUTE, bold=True)


def divider(prs, number, title, points):
    s = blank(prs)
    bg = rect(s, 0, 0, W, H, fill=DEEP, shape=MSO_SHAPE.RECTANGLE)
    bg.shadow.inherit = False
    circ = slide_circle(s, M, 2.42, 1.16)
    tf = circ.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run(p, number, size=38, color=DEEP, bold=True, font=HEAD)
    tfa = tb(s, M + 1.52, 2.46, 8.6, 1.1)
    run(para(tfa, True), title, size=38, color=WHITE, bold=True, font=HEAD)
    tfb = tb(s, M + 1.52, 3.62, 9.4, 1.4)
    for i, q in enumerate(points):
        p = para(tfb, i == 0)
        p.space_after = Pt(7)
        run(p, "—  " + q, size=15, color=RGBColor(0xC9, 0xD8, 0xE6))
    return s


def slide_circle(slide, x, y, d):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    c.fill.solid()
    c.fill.fore_color.rgb = WHITE
    c.line.fill.background()
    c.shadow.inherit = False
    return c
