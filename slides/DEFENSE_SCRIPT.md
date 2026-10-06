# Slide-by-slide defense script

**ULFG defense: Friday 9 October · Company rehearsal: Thursday 8 October**

Written on one assumption: **the jury does not work in this field.** Every explanation here is
for someone hearing "fine-tuning" for the first time.

Each slide has four parts:

| | |
|---|---|
| **ON SCREEN** | what you are standing in front of |
| **SAY** | the actual words — not to memorise, to have said once |
| **UNDERSTAND** | the idea underneath, in plain language. **This is the part that makes you able to answer anything.** |
| **IF ASKED** | the question most likely to come at this slide |

---

## The three sentences

If everything else goes, these carry the defense.

> **1.** "I took one small model and adapted it in two completely different ways — and I
> measured what each way actually changed."
>
> **2.** "The first way taught it **how to answer a question**. 90.42%, matching models four
> times bigger, on a free GPU."
>
> **3.** "The second taught it **what Lebanese legal writing looks like**. It learned the field
> — and gained no ability to answer questions at all. That contrast is the project."

---

## Two-day plan

| When | Do |
|---|---|
| **Mon night** | Read slides 1–13 of this document aloud, with the deck open beside you. |
| **Tue morning** | Slides 14–23. Then the whole thing once, standing, timing yourself. |
| **Tue night** | §Questions at the end. Say the answers out loud — reading them is not the same skill. |
| **Wed 8/10** | Company rehearsal. Write down every question they ask. |
| **Wed night** | Fix only what the rehearsal exposed. Do not rewrite the talk. |
| **Thu night** | The three sentences, and §Do not say. Then stop. |

---
---

# SLIDE 1 — Title · 20 seconds

**ON SCREEN** — three logos, the title, your name, your two academic supervisors and
Dr. Baalbaki.

**SAY**
> "Good morning. My project is about adapting a small language model for Arabic — in two
> different ways — and measuring what each way changes. My supervisors are Dr. Zein Alabidin
> Ibrahim and Dr. Abed Ellatif Samhat, and the work was done at WhiteStork with
> Dr. Daoud Baalbaki."

Then move on. Do not read the title aloud; they can read.

---

# SLIDE 2 — Objectives · 1 min 20

**ON SCREEN** — the problem, four numbered goals, and four big numbers at the bottom:
90.42% · 4.14 · $0 · $5.40

**SAY**
> "The published work on this task all uses models of seven billion parameters and up. Those are
> expensive to run and expensive to adapt. My question was whether a model four times smaller
> can keep up — and what it actually takes to adapt one.
>
> [point at the list] Four things. Match a big model with a small one on a free GPU. Find the
> real floor of this benchmark, because nobody who published on it reported one. Check that the
> metrics measure what everyone assumes they measure. And teach the model Lebanese law, to see
> what that changes.
>
> [point at the numbers] That row is where it ended up. I will come back to every one of them."

**UNDERSTAND**
Why open with the results? Because a jury that knows where you are going follows the middle far
better than one being kept in suspense. This is not a detective story.

**IF ASKED — "why Gemma 2-2B specifically?"**
> "Because the paper I am comparing against used Gemma 2-9B. Same family, same tokenizer, same
> pre-training — so the only thing that differs is size. If I had used a different family, any
> difference could have been the family rather than the size."

---

# SLIDE 3 — What fine-tuning means · 1 min

**ON SCREEN** — three numbered cards: Pre-training → Fine-tuning → Two ways to adapt.

**SAY**
> "A language model is first pre-trained on an enormous amount of general text. That is where it
> learns language itself. It costs millions and takes months, so nobody repeats it — not me, not
> anyone here.
>
> What we do instead is take the finished model and adapt it, with a small amount of our own
> data. Hours, not months.
>
> And there are two ways to adapt it. You can show it worked examples of a task, or you can let
> it keep reading text from a new field. Those are the two halves of this project."

**UNDERSTAND**
This is the slide that decides whether the jury follows the rest. If someone looks lost here,
stop and use the student analogy:

> "Think of a student who has read a huge number of books — they know the language. Now, one
> option is to give them worked exam questions with the answers. Another is to hand them a
> library from one subject and let them read. Both change the student. They change **different
> things**."

---

# SLIDE 4 — Why it needs a trick · 1 min

**ON SCREEN** — 41.7 GB vs 16 GB, and the memory figure.

**SAY**
> "But even adapting is not free. Training a two-billion-parameter model the normal way needs
> about 41.7 gigabytes, and a free GPU gives you 16.
>
> The interesting part is *where* that memory goes. Only 5.2 gigabytes is the model itself. The
> other 36 is bookkeeping — the optimiser keeps two extra numbers for every weight it is
> training.
>
> So the memory does not grow with how big the model is. It grows with **how many weights you
> train.** Which means: train fewer weights, and the problem disappears."

**UNDERSTAND**
The optimiser is the algorithm that decides how to change each weight. Adam, the standard one,
remembers two running statistics per weight so it can adapt the step size. Three numbers per
weight instead of one — that is the 36 GB.

**IF ASKED — "why not just use a smaller batch?"**
> "That reduces the activation memory, not the optimiser state. The optimiser state is fixed by
> the number of trainable parameters — it does not shrink no matter how little data you push
> through at a time."

---

# SLIDE 5 — LoRA and QLoRA · 1 min 20

**ON SCREEN** — the LoRA diagram, 1.6% and 16×.

**SAY**
> "That is what LoRA does. Freeze the whole model — do not touch it. Add two small matrices
> beside it, and train only those. It turns out the change a model needs for one specific task
> is simple enough to fit in something that small. Here it is 1.6% of the weights.
>
> QLoRA adds a second saving on top: store the frozen model in four bits instead of sixteen.
> A quarter of the space. The small part being trained stays at full precision, so you are not
> losing accuracy where it matters.
>
> Together that is sixteen times less memory — which is what let this run on a free GPU. And
> afterwards the small part folds back into the model, so nothing runs slower."

**UNDERSTAND**
*Low-rank* means: instead of learning a big grid of numbers, learn two thin ones whose product
is the same shape. A 2048×2048 grid is 4 million numbers; two 2048×32 strips are 131 thousand.
That is the whole saving.

The reason it works at all: research showed that the change fine-tuning makes to a model is
"low-rank" in practice — simple and structured, not arbitrary. So a thin approximation loses
almost nothing.

**IF ASKED — "what does the rank control?"**
> "How much freedom the change has. Higher rank can express a more complicated adaptation, at
> more memory. I used 32 for the first half and 16 for the second."

**IF ASKED — "doesn't 4-bit lose accuracy?"**
> "It loses precision in the frozen part, and the part being trained stays full precision. The
> QLoRA paper reports that 4-bit matches 16-bit. I could not verify that myself — I had no
> machine that could run the 16-bit comparison — and I say so in my limitations."

---

# SLIDE 6 — Two ways to fine-tune · 1 min 30 ⭐

**ON SCREEN** — the comparison table. Bottom row: *how to answer* vs *what the field sounds
like*.

**SAY** — slow down here.
> "So here are the two ways side by side.
>
> On the left, you show it pairs — a question and its right answer. Nine thousand nine hundred
> of them, in my case. On the right, you give it plain text with no labels at all — twenty-seven
> million words of Lebanese law — and just let it read.
>
> [point at the bottom row] **This is the point of the whole talk.** One of them teaches the
> model how to answer a question. The other teaches it what a field of writing sounds like.
> They are not interchangeable.
>
> Part 1 of my talk is the first column. Part 2 is the second. And at the end I will show you
> that each one moved exactly one of those two things — and left the other alone."

**UNDERSTAND**
Everything before this slide is setup. Everything after it is evidence for that bottom row. If
the jury takes away one slide, it should be this one.

**IF ASKED — "so which one is better?"** ⭐ *Expect this.*
> "Neither — they do different jobs. It depends on what is missing. If the information the model
> needs is already in the question, the problem is format, and worked examples fix it. If the
> information has to come from memory, no number of worked examples will put it there."

---

# SLIDE 7 — Part 1 divider · 10 seconds

**SAY**
> "Part one. Arabic word sense disambiguation."

---

# SLIDE 8 — The task in one example · 1 min 30 ⭐

**ON SCREEN** — the Arabic sentence, the target word جاب, the two candidate meanings, 4548
marked as correct.

**SAY** — your strongest minute. Do not rush it.
> "Here is one real question from the dataset.
>
> [read the sentence, or point at it] The target word is جاب. In Arabic this one word has two
> quite different meanings — to bore through rock, or for news to spread through a country.
> In this sentence it is the first one. The answer is sense 4548.
>
> Arabic makes this harder than it would be in English. Words change shape a great deal, small
> words attach to bigger ones, and the short vowels that would separate the two meanings are
> simply not written down.
>
> [pause] But now notice something about this question. **Both meanings are printed right there
> in it.** The model does not need to know anything about Arabic law or history. It needs to
> choose between two things it can already see.
>
> Remember that — it explains almost everything in Part 1."

**UNDERSTAND**
This is the seed of your whole thesis. Because the answer is *visible*, the task is
discrimination rather than recall. Big models are big because they store more knowledge — and
knowledge is not what this task is short of. That is why 2B keeps up with 9B, and you will say
exactly that on slide 13.

**IF ASKED — "how many choices are there always?"**
> "Always exactly two in this dataset. That matters for the baseline, which is slide 11."

---

# SLIDE 9 — What the model actually sees · 1 min

**ON SCREEN** — the real training example: `### Instruction` / `### Input` / `### Response`,
with the same جاب example inside it, and the run configuration on the right.

**SAY**
> "This is the real training example, exactly as my code builds it — not a simplified version.
>
> There is an instruction telling it what the task is, then the input: the sentence, the target
> word, and both candidate senses with their IDs. And the response is just the number.
>
> [point at the table] Gemma 2-2B in four bits, nine thousand nine hundred examples, three
> passes, on a free Colab GPU, at no cost.
>
> One detail I checked carefully: this exact same text has to be rebuilt at test time. If it
> differs even slightly the model sees something it was not trained on. I verified it character
> for character — zero mismatches across all 3,110 test items."

**UNDERSTAND**
The model is a text predictor. Training it on a task means wrapping the task in text and
teaching it to produce the right continuation. There is no classifier head, no special
machinery — just text in, text out.

**IF ASKED — "why this template?"**
> "It is the Alpaca format, a standard instruction layout. The specific template matters less
> than using the identical one at training and at test time."

---

# SLIDE 10 — 90.42% · 1 min 30

**ON SCREEN** — three stat cards, the comparison table, and the margin-of-error box.

**SAY**
> "The result: 2,812 of 3,110 correct — 90.42%. And nothing malformed; every single output was a
> usable sense ID.
>
> Against the published results on the same test: the 9-billion model got 89.39, the 8-billion
> got 90.42, the 7-billion got 90.77, and mine, at 2 billion, got 90.42.
>
> [immediately, before they ask] But I want to be careful here. With 3,110 questions the margin
> of error is about half a point — so the true value is somewhere between 89.4 and 91.4. Every
> published score sits inside that range.
>
> **So I am not claiming my model is better. I am claiming it is not worse — at a quarter of the
> size, and on a free GPU.**"

**UNDERSTAND — what "margin of error" means here.**
Each question is like a coin flip you either get right or wrong. With 3,110 of them, the score
you measure wobbles around the true score by about ±0.53 points just from which questions
happened to be in the test. So two scores one point apart are not distinguishable.

**IF ASKED — "could you prove yours is better?"**
> "Only with a paired test — comparing the two models question by question on the same items.
> That needs their per-question answers, and those were never published."

---

# SLIDE 11 — The baseline · 1 min 30 ⭐

**ON SCREEN** — the baseline chart, and 64.95%.

**SAY**
> "Now, 90.42% sounds good. But good compared to what?
>
> Every question has two choices, so guessing gets you 50%. But I measured something else: the
> correct answer happens to be **the second one listed** 64.95% of the time.
>
> So a one-line program that always answers 'the second one' — a program that cannot read, that
> knows no Arabic at all — scores 64.95% on this benchmark.
>
> And it is the same on all three splits of the data, so it is not luck. It is baked into how
> the dataset was built.
>
> **No paper published on this benchmark reports any baseline.** They present 90% as though the
> floor were 50. It is not. The real result is 90.42 against 64.95."

**UNDERSTAND — what a baseline is, in one line.**
> A baseline is the score you can get **without doing the work**. It is the bar your result has
> to clear before it means anything.

The exam analogy, if they look unsure:
> "In a two-option multiple-choice exam, guessing gets 50%. But if someone notices the answer is
> B about two-thirds of the time, a student who answers B on every question — without reading
> anything — scores 65%. If I then score 70%, I have barely beaten someone who cannot read the
> exam."

**IF ASKED — "did your model just learn that trick?"** ⭐
> "No, and I checked specifically. If it had, its mistakes would be concentrated on the
> questions where the answer was first. They are not — the mistakes split evenly between the two
> positions. It learned the task, not the shortcut. That is slide 13."

---

# SLIDE 12 — Macro-F1 is broken · 1 min 30 ⭐

**ON SCREEN** — the explanation, and the three-line arithmetic box.

**SAY**
> "The second thing every paper reports is macro-F1. I want to show you that on this dataset it
> carries no information at all.
>
> Macro-F1 averages the score separately over every class, so that a rare class counts as much
> as a common one. It exists to stop a model looking good by only handling the common cases.
>
> But here, every one of the 3,110 questions has its own unique answer ID. So there are 3,110
> classes with exactly one example each. There is nothing to average.
>
> And it gets worse. [point at the box] Watch: accuracy times items gives 2,812 correct. Divide
> by the reported macro-recall and you get 3,356 — exactly. Subtract the 3,110 real classes and
> you are left with 246.
>
> Those 246 are classes the model **invented**. Every time it gave a wrong answer ID, that wrong
> ID counted as a brand-new class scoring zero.
>
> **So macro-F1 here is just accuracy divided by a number the model moves itself** — the more
> *varied* its mistakes, the worse it looks. It adds nothing. And every paper reports it."

**UNDERSTAND — what F1 and "macro" normally are.**
- **Precision**: when you say yes, how often are you right?
- **Recall**: of all the real yes-cases, how many did you catch?
- **F1**: a single number combining the two.
- **Macro**: compute it per class, then average — so small classes are not drowned out.

All of that assumes classes with several examples each. With one example per class the machinery
degenerates into the formula above.

**IF ASKED — "then why do you report it?"**
> "Only so my table can be compared with theirs. I state in the report that it should not be read
> as a second opinion on quality."

---

# SLIDE 13 — The 298 mistakes · 1 min

**ON SCREEN** — the error figure and three findings.

**SAY**
> "Three things about what it got wrong.
>
> First, nothing came out broken — all 3,110 answers were real sense IDs. So these are genuine
> confusions, not formatting failures.
>
> Second, it did not take the shortcut. The right answer is second 65% of the time, and if it
> had learned that, its errors would be lopsided. They are not.
>
> Third, it fails where the two meanings genuinely overlap — where the two definitions share
> most of their words. Some of those pairs are near-duplicates that a person would also struggle
> with. That caps how high anyone can score on this benchmark.
>
> [then, slowly] And this is where Part 1 lands. A 2-billion model keeps up with a 9-billion one
> because this task is **telling two visible answers apart — not recalling facts.** Size buys
> knowledge. Knowledge is not what was missing here."

---

# SLIDE 14 — Part 2 divider · 15 seconds

**SAY**
> "Which raises the opposite question. What happens when the knowledge **does** have to come from
> the model's memory? That is Part 2."

---

# SLIDE 15 — The corpus · 1 min 20

**ON SCREEN** — a real Lebanese decree in Arabic, the corpus table, the split, and the bug box.

**SAY**
> "The company gave me 27 million words of Lebanese legal text — the official gazette, laws,
> court rulings. [point at the Arabic] This is a typical document: a decree moving budget from
> the education ministry to build a school. No labels, no questions. Just text.
>
> We hide 5% of the documents and never train on them. Every number I show you afterwards is
> measured only on those.
>
> One detail matters: we hide them **by document**, not by paragraph. Long documents are stored
> as overlapping pieces, so splitting carelessly puts the same text on both sides — you end up
> testing the model on what it already read. I measured it: 5.9% of the held-out set would have
> leaked that way.
>
> [the bug box] And one thing I want to mention because it is my own contribution. The plan I was
> given said to group pieces into documents by their ID. I checked before running anything, and
> 396 IDs are reused across different collections — they are not the same document. Grouping by
> ID alone would have glued unrelated documents together and quietly weakened that guarantee."

**UNDERSTAND**
"Leakage" means test material appearing in training. It makes your results look good for the
wrong reason. Checking for it before running is what makes the later numbers trustworthy.

**IF ASKED — "how do you know there is no leakage?"**
> "Two things. The split is at document level with an automated check that asserts zero overlap.
> And the whole split is deterministic — it reproduced identically on three different machines."

---

# SLIDE 16 — Training · 1 min

**ON SCREEN** — the loss curve with the learning rate dashed behind it.

**SAY**
> "One pass over the corpus. Three hours forty minutes, about two dollars of rented GPU.
>
> The red line is the error on documents the model never saw. It falls at all seventeen
> checkpoints and never turns back up. That matters — if it had started rising, it would mean
> the model was memorising the training documents instead of learning the language of the field.
>
> [then, honestly] One thing I want to be careful about. The curve goes flat at the end, and it
> is tempting to call that convergence. It is not. The grey dashed line is the learning rate,
> and it is deliberately wound down to almost nothing by the end. A model that has stopped
> changing cannot show improvement. So that flat tail tells us nothing in either direction."

**UNDERSTAND — what the loss actually is.** *(This is the question you said you wanted to be
sure of.)*

At every position in the text the model predicts the next word — not one answer, but a
probability for every possible word. The loss asks one thing: **what probability did you give
the word that was actually there?**

> loss = −log( probability it gave the correct word )

High probability → small loss. Low probability → big loss. Four anchors worth memorising:

| Loss | Meaning |
|---|---|
| **12.45** | Knows nothing. The vocabulary is 256,000 words and ln(256,000) = 12.45 — a model guessing uniformly scores exactly this. |
| **6.69** | Our base model on Lebanese legal text, before training. |
| **1.42** | The same model after reading the corpus. |
| **0** | Perfect. |

**Why it goes down:** the model predicts, we measure how wrong it was, and we adjust the weights
a small step in the direction that would have made the correct word more likely. Then repeat
with the next batch. We did that 1,658 times.

**IF ASKED — "how do you know which direction to adjust?"**
> "Calculus — we take the derivative of the error with respect to each weight. That tells us
> which way to move each one. That is what gradient descent means."

---

# SLIDE 17 — Does it read legal Arabic better · 1 min 20

**ON SCREEN** — three stat cards (806→4.14, 21→70%, 1705→17) and the per-source bar chart.

**SAY**
> "Three measurements, all on documents it never saw.
>
> Perplexity went from 806 to 4.14. Next word predicted exactly right: from 21% to 70%. And the
> position of the correct word in its ranked list: from 1,705th to 17th, out of 256,000 options.
>
> [then the defence] Now, on its own this would be a weak claim — of course a model gets better
> at text you trained it on. So I split the result by source.
>
> **Every single source improved.** That matters because two of the seven sources are 74% of the
> corpus and are very repetitive — the whole gain could have come from boilerplate. It did not.
> Court rulings, which are the shortest and least formulaic, gained 41 points."

**UNDERSTAND — what perplexity is.** *(Say this version, it lands.)*

> **Perplexity is how many options the model is effectively torn between.**

It is just the loss in a friendlier unit: **perplexity = e^loss**.

| Loss | Perplexity | Read as |
|---|---|---|
| 6.69 | **806** | as confused as someone picking blindly from 806 words |
| 1.42 | **4.14** | effectively choosing between about four |

**IF ASKED — "why do you quote 3.71 somewhere and 4.14 here?"**
> "Same measurement on two slightly different held-out samples — 3.71 on the packed test blocks,
> 4.14 on the thousand sampled positions. Both say the same thing."

---

# SLIDE 18 — Knowledge or style · 1 min 30 ⭐

**ON SCREEN** — the cloze table. Third row: *which law is cited*, 42% → 66%, guessing 40%.

**SAY**
> "But reading well is not the same as knowing anything. So here is the strict test.
>
> We take a sentence from a document the model has never seen, delete one legal term, and ask it
> to fill the gap from a fixed list of options. The fixed list matters: it means we know exactly
> what pure guessing is worth.
>
> [point at the third row] **This is the row that matters, and it is not the biggest one.**
>
> Here we remove the name of the law being cited — and usually a different law is mentioned
> nearby, to mislead it. Before training it scored 42%. But simply always guessing the commonest
> law scores 40%. So before training it knew essentially **nothing** — it was just naming the
> most frequent law.
>
> After training: 66%. Twenty-six points clear of guessing.
>
> **That is the evidence that it learned which law is which — not only how legal writing
> sounds.**"

**UNDERSTAND**
The top two rows look more impressive (+72 and +27) but prove less — phrasing is the easiest
thing to absorb. The third row is the one that separates *knowledge* from *style*, which is why
it is the row to narrate.

**IF ASKED — "is 300 questions enough?"**
> "We ran a paired test — the same 300 questions for both models. 136 it got right only after
> training; 13 only before. The chance of that pattern arising by luck is below one in a
> trillion."

---

# SLIDE 19 — Did it break anything · 2 min ⭐⭐

**ON SCREEN** — the decomposition table: *answered at all* / *right when it answered* / *score*.

**This is the most important slide in the talk. Deliver it slowest.**

**SAY**
> "Last question for Part 2. Teaching it law might have damaged what it could already do. So we
> gave the legal model the Part 1 word-sense task — which it was never trained on, not once.
>
> Its score went **up**. From 24.9% to 36.5%. So nothing was damaged.
>
> [pause] And at first glance that looks like it got better at the task. It did not. Here is why.
>
> [point at the table] Split the score into two separate questions. First: did it produce a
> usable answer at all? Second: when it did, was the answer right?
>
> Answering at all went from 52.5% to 75.1% — up 22.6 points. But being right when it answered
> went from 47.4 to 48.6. That is 1.2 points, and the margin of error is about 2.8. That is
> noise. And both numbers are sitting on 50% — which, with two choices, is a coin flip.
>
> **So reading law made it answer. It did not make it think.**
>
> The reason is almost funny: the legal corpus is full of 'Decree number 1234'. A full pass over
> that made the model far more likely to stop cleanly and produce something number-shaped — and
> the answers in Part 1 are numbers. So it produces a well-formed answer much more often. It is
> just not a better answer.
>
> [last line] I wrote three predictions down before I started training. This is the one I got
> wrong — I expected this number to stay flat. Chasing why it did not is what produced the main
> result of the project."

**UNDERSTAND — why decomposing mattered.**
The headline number, 36.5%, mixes two completely different abilities. Separated, they tell
opposite stories. If you had reported only 36.5% you would have claimed CPT improved the task —
which is false.

**IF ASKED — "why does the base model score below 50% when there are only two choices?"** ⭐
> "Because it often produces nothing usable — on 47.5% of items. Those count as wrong. When it
> does answer it is at chance. It is not worse than guessing; it is silent half the time."

**IF ASKED — "how do you know 1.2 points is noise?"**
> "The standard error on that comparison is about 2.8 points, so 1.2 is well inside it. And the
> confidence intervals for both models contain 50% — chance."

---

# SLIDE 20 — What each one changed · 1 min 20

**ON SCREEN** — the 2×2 table. SFT: task CHANGED, knowledge unchanged. CPT: the reverse.

**SAY**
> "So, putting both halves together.
>
> [read across] Worked examples changed its ability to do the task — 24.9 to 90.42 — and left its
> knowledge of Lebanese law exactly where it was, because it never saw any legal text.
>
> Reading the field changed its knowledge of law — perplexity 806 to 4.14 — and left its ability
> to do the task exactly where it was: 47.4 to 48.6.
>
> **Each one moves one column and leaves the other alone.**
>
> And I want to say why I trust that. The weak way to argue it would be: the legal model did
> better on legal tests, the word-sense model did better on word-sense tests. That is circular —
> each was only ever measured on its own ground. The retention test avoids that, because both
> models are measured on the **same** task, the **same** questions, and the same split between
> answering and being right.
>
> [final] And this holds whatever the sizes had turned out to be. Even if the legal training had
> barely worked, moving willingness by 22 points while moving real ability by 1 would mean the
> same thing."

---

# SLIDE 21 — What each one bought · 1 min 20

**ON SCREEN** — two cards side by side, with a verdict line under each.

**SAY**
> "In practical terms, for the two jobs I had.
>
> [left] For Arabic word senses, where the answer was already in the question: 90.42%, level with
> 8-billion models, 25 points above the real floor, nothing malformed, no shortcuts — and it cost
> nothing. Worked examples were the right tool and enough on their own. Size was never the
> problem here. Format was.
>
> [right] For Lebanese law, where the answer had to come from memory: perplexity 806 to 4.14,
> every source improved, and it learned which law is which — 42 to 66%. Nothing was damaged. Five
> dollars forty. But it bought fluency and vocabulary in the field, and no ability to do a task.
>
> [the rule] So: if the answer is already in the question, use worked examples. If it has to come
> from memory, let it read the field. A legal word-sense system would need both — in that order."

---

# SLIDE 22 — Limitations · 1 min

**ON SCREEN** — what it does not prove, and what comes next.

**SAY**
> "What this does not prove.
>
> The two halves are not a fair race — different data, different tests. That is why I only ever
> compare what each one changes, never which one wins.
>
> One run, one setting. A bigger adapter might have absorbed more law; I cannot say.
>
> And I used 27 of about 134 million available words, so every number is a floor, not a ceiling.
>
> [next] The first thing I would do next is the fair race: worked examples on the legal corpus at
> the same word budget. That is the one experiment that would turn my comparison from qualitative
> into controlled.
>
> [if time] And one thing I would say about process. Three measuring mistakes were caught during
> this project — every one of them by checking a number that looked **good**. A key that merged
> unrelated documents, a scorer that reported 0% for a model that was in fact answering
> correctly, and a setting that flattered the trained model. Each would have given a tidier story
> than the truth."

---

# SLIDE 23 — Thank you · 10 seconds

**SAY**
> "Thank you. I am happy to take questions."

Then **stop talking.** Do not fill the silence.

---
---

# Questions they will ask

Answer in two sentences, then stop. If they want more they will ask.

| Question | Answer |
|---|---|
| **"Your 2B beat their 9B — is that real?"** ⭐ | "No, and I say so. The margin of error is half a point and every published result sits inside it. The honest claim is it is not worse, at a quarter the size." |
| **"Why not train the whole model?"** | "Memory — 41.7 GB needed against 16 available. And it is the optimiser's bookkeeping, not the model, so training 1.6% of the weights solves it." |
| **"What is perplexity?"** ⭐ | "How many options the model is effectively torn between. 806 before, about 4 after." |
| **"Why is macro-F1 meaningless here?"** ⭐ | "Every item has its own unique label, so there are 3,110 classes of one. And wrong answers invent new classes — 246 of them. It is accuracy divided by a number the model moves itself." |
| **"Why did the legal model get better at word senses?"** ⭐ | "It did not get better at the task — it got better at *answering*. When it answers it is no better than before, and no better than a coin flip." |
| **"How do you know it learned, not memorised?"** | "The test documents were separated at document level with an asserted zero overlap, and the fill-the-gap test is built only from documents it never saw, with a known guessing rate." |
| **"Could you do both — law then worked examples?"** | "That is exactly the right next step and I did not run it. My project measured each separately, which is what lets me say what each changes." |
| **"What went wrong?"** | "Three measurement mistakes, all caught by checking a number that looked good. The clearest: my scorer was written for the trained model's output format, so on the untrained model it reported 0% while the model was answering correctly." |
| **"What would you do differently?"** | "Run the matched comparison — worked examples on the legal corpus at equal budget. It is the one experiment that would make the comparison controlled." |
| **"How much did it cost?"** | "The first half nothing — a free Colab GPU. The second, about two dollars of training and five forty including all evaluation." |

---

# Do not say

| Do not say | Say |
|---|---|
| "My model beat the 9B model." | "It is not worse, at a quarter the size." |
| "CPT improved accuracy on Dataset A." | "It made it answer more often. Accuracy when answering did not move." |
| "The loss curve shows it converged." | "The flat tail is the schedule winding down — it tells us nothing either way." |
| "CPT is better than SFT." | "They change different things." |
| "Macro-F1 of 0.83 shows it is balanced." | "Macro-F1 is degenerate here — and here is the arithmetic." |
| "The model understands legal Arabic." | "It predicts legal Arabic far better, and learned which law is which. I did not test understanding." |

---

# If you blank

1. **Go back to the student analogy** — worked exam questions versus a library. It resets you and
   the room.
2. **Read a number off the slide and say what it means.** "806 to 4.14 — from being torn between
   800 options to about four." You are never lost while explaining a number.
3. **"That is in the report — let me give you the short version."** Then give it.

**And if you genuinely do not know:** *"I did not test that, so I would be guessing."* That is a
strong answer. You have used it correctly all the way through this project — it is why the work
holds up.
