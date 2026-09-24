# Word version of the report

Two files are produced. **They are the same report** — only the title page differs:

| File | Title page |
|---|---|
| `FYP_Report.docx` | Faculty of Sciences layout: *Master Thesis / Professional Master in Artificial Intelligence and Data Engineering / Supervisor + Reviewers* |
| `FYP_Report_Engineering.docx` | Faculty of Engineering Branch III layout: *Major / Option / Supervised by / jury with dot leaders* |

Pick whichever your department expects and delete the other.

## First three things to do in Word

1. **Build the four auto-lists.** Press `Ctrl+A`, then `F9`, and answer **Update entire table** to each
   prompt. This fills the Table of Contents, List of Tables, List of Figures and List of Listings.
   Do it once more immediately before printing, after the page numbers have settled.
   Table, figure and listing numbers are already correct before you do this — `F9` only recomputes
   the same values and fills the four lists.
2. **Check the Arabic renders.** Look at Table 3.1 (§3.1) and the examples in §2.1, §2.3 and §6.2.
   If Arabic shows as boxes, select it and change the font to **Traditional Arabic**, **Amiri**,
   or **Simplified Arabic**. Every Arabic run is already marked right-to-left by the builder.
3. **Turn on the Navigation pane** (View → Navigation Pane). Every heading is a real Word
   Heading 1/2/3, so you can jump around and reorder chapters by dragging.

## What's in it

~115 pages, 64 tables, 15 figures, 21 code/data listings.

| Part | State |
|---|---|
| Title page, Acknowledgements, Abstract, Abbreviations, auto-lists | Drafted — fill the `[bracketed]` placeholders |
| General Introduction | ✅ Written |
| **Chapter 1 — Introduction** | ✅ Written |
| **Chapter 2 — Transformers and Language Models** | ✅ Written |
| **Chapter 3 — Arabic NLP and Word Sense Disambiguation** | ✅ Written |
| **Chapter 4 — Model Adaptation (LoRA / QLoRA)** | ✅ Written |
| **Chapter 5 — Related Work** | ✅ Written |
| **Chapter 6 — Methodology: Supervised Fine-Tuning** | ✅ Written |
| **Chapter 7 — Results: Supervised Fine-Tuning** | ✅ Written |
| **Chapter 8 — Methodology: Continued Pre-Training** | ✅ Written |
| **Chapter 9 — Results: Continued Pre-Training** | ✅ Written |
| **Chapter 10 — Discussion: What Each Objective Changes** | ✅ Written |
| **Chapter 11 — Limitations** | ✅ Written |
| **General Conclusion** | ✅ Written, both halves |
| **Appendix A — Pipeline Source Code** | ✅ Read live from the repo at build time |
| **Appendix B — Recovered Run Configuration** | ✅ Written |
| **Appendix C — CPT Run Configuration and Logs** | ✅ Read live from the run artifacts |
| **Appendix D — Arabic Generation Samples** | ✅ Read live from the run artifacts |
| References | 29 entries; those marked `[VERIFY]` were reconstructed and must be checked |


Four red **TODO** notes remain. All four are things only you can do: personalise the
acknowledgements, update the Word fields, manage the citations, and confirm the Colab GPU model.

## Logos

Each title page looks for two files and inserts them automatically. `.png`, `.jpg` and `.jpeg` all
work — only the name before the extension matters:

```
figures/logo_lu          Lebanese University emblem   (left, both variants)
figures/logo_institute   Faculty of Sciences logo     (right, FYP_Report.docx)
figures/ulfg_logo        Faculty of Engineering logo  (right, FYP_Report_Engineering.docx)
```

A missing logo leaves a red note in its slot naming the file it wants, so nothing shifts and the
gap is obvious. Save the file and rebuild and it fills itself in.

Heights are chosen per image, not fixed: a logo taller than it is wide is assumed to carry a
caption under its emblem (the Faculty of Sciences one spends its bottom fifth on three lines of
text) and is scaled up so its *emblem* matches the square LU emblem instead of coming out a fifth
too small. The two sit in a borderless table so their tops align.

Both current logos are ~1000 px and ~500 px, which is 600–1000 DPI at print size — comfortably
above what printing needs.

## Title-page placeholders

Everything in red brackets is a name or date I could not know. Edit them in
`build_docx.py` (`title_page`, near the top of the file) and rebuild, or just type over them in Word.

| `FYP_Report.docx` | `FYP_Report_Engineering.docx` |
|---|---|
| `On [Weekday, DD Month 2026]` | `Dr. [Academic supervisor]` |
| `Dr. [Reviewer 1]`, `Dr. [Reviewer 2]` | `Defended on [DD / MM / 2026]` |
| | `Dr. [Jury member 1..3]` |

`AUTHOR` and `TITLE` are constants at the top of `build_docx.py`, shared by both variants — change
the title once and both files follow.

## Regenerating

```
cd thesis_word
../venv/Scripts/python.exe make_figures.py    # only if figure data changed
../venv/Scripts/python.exe build_docx.py
```

- `build_docx.py` — page setup, styles, the `Doc` helper class, title page, front matter,
  General Introduction, Chapter 1, and the build order
- `build_chapters.py` — Chapters 2–7, references, data samples, and the section expansions
- `build_cpt_chapters.py` — Chapters 8–11 (the continued pre-training half)
- `build_cpt_appendices.py` — Appendices C and D, read from `CPT/run_gemma2_2b/` at build time
- `build_appendices.py` — General Conclusion, Appendix A, Appendix B
- `make_figures.py` — regenerates the 15 PNGs in `figures/`

One run writes **both** files. To build only one, call it directly:

```
../venv/Scripts/python.exe -c "import build_docx as b; b.build('engineering', b.OUT2)"
```

⚠️ **Regenerating overwrites both .docx files.** Once you start editing in Word, either stop
regenerating or save your edits under a different filename. If a file is open in Word when you
rebuild, the builder writes `FYP_Report_NEW.docx` beside it instead of failing — close Word and
rerun to replace the original.

**Appendix A is read from the repository at build time**, not transcribed — so the code printed in
the report cannot drift from the code in `Gemma/Fine-tuning/Dataset-A/`. Local absolute paths are
rewritten to `<REPO>` for printing; nothing else is altered.

## Known limitations of the Word route

- **Two equations** are still plain centred text, flagged with a small red `[EQ-TODO]`.
  Replace them using **Insert → Equation** for the final version. The rest are fine as text.
- **Citations are manual.** Use Word's References tab → Manage Sources, or the Zotero/Mendeley
  Word plug-in. The numbered list at the end is a starting point, not a managed bibliography.

Table, figure and listing numbering *is* automatic — captions use real Word `SEQ` fields, which is
what makes the three lists build themselves.

The LaTeX version is still in `../thesis/` if you ever want to switch back, but it is now well
behind this one.
