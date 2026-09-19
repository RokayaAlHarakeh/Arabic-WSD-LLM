"""
Build a legal cloze probe from held-out documents.

Day-7 deliverable in FYP_Timeline.pdf: mask legal terms in documents the model never
saw, and measure whether it can predict them. Fully automatic, no manual annotation.

WHY CLOZE AND NOT PERPLEXITY ALONE
----------------------------------
"We trained on legal text and legal perplexity dropped" invites the circularity
objection. Cloze asks a different and stronger question: can the model predict
*domain-characteristic terms* in documents it has never seen?

It is also distinct from next_token_eval_gemma.py, which samples positions at random.
Random positions are dominated by function words and general Arabic, where a base model
already does well. Here every target is a legal term, so the probe is concentrated
exactly where continued pretraining should help.

WHAT IS MASKED, AND WHAT IS DELIBERATELY NOT
--------------------------------------------
Three categories, all mined automatically:

  statute_term  the legal instrument noun in a citation
                "بناء على ____ رقم 14953"  -> المرسوم
  collocation   the final content word of a frequent domain phrase
                "الجريدة الرسمية ____"      -> اللبنانية
  defined_term  a domain-characteristic term frequent in TRAIN, masked in TEST

Phrases are mined from the TRAIN split, so they are known to be domain-typical; every
masked *instance* comes from a TEST document, so the item itself is unseen.

Arbitrary identifiers -- decree numbers, dates, case numbers -- are NOT masked. No model
can predict "14953" from context, so such items would score ~0% for both the base and the
CPT model and discriminate nothing. Masking them would make the probe look rigorous while
measuring noise. This is a methodology point worth stating in Chapter 8.

Usage (CPU only):
    python build_legal_cloze.py \
        --test  splits/test_doclevel.jsonl \
        --train splits/train_doclevel.jsonl \
        --out   eval/legal_cloze.json \
        --n 300 --seed 42
"""

import argparse
import collections
import json
import random
import re
import unicodedata
from pathlib import Path

# Same normalisation as next_word_inference_sonnet_bedrock_REFERENCE.py, so scores here
# are comparable with the project's existing Arabic matching.
ARABIC_DIACRITICS = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
ARABIC_WORD = re.compile(r"[ء-ي]+")
TAG = re.compile(r"<[^>]*>")


def normalize(value):
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("ـ", "")           # tatweel
    value = ARABIC_DIACRITICS.sub("", value)
    return value.strip()


def strip_tags(text):
    """Remove the <source:...><type:...> header tags the corpus prefixes to each chunk."""
    return TAG.sub("", text)


# The legal instruments a citation can name. Masking one of these asks the model which
# kind of instrument is being cited -- domain knowledge, and genuinely predictable.
STATUTE_TERMS = [
    "المرسوم",       # decree
    "القانون",       # law
    "القرار",        # decision
    "المادة",        # article
    "التعميم",       # circular
    "الاتفاقية",     # convention
]
STATUTE_RE = re.compile(
    r"(?P<term>" + "|".join(STATUTE_TERMS) + r")\s+(?:الاشتراعي\s+)?رقم\s+\d"
)


def mine_collocations(train_path, max_docs, min_count, top_k):
    """Frequent 2- and 3-grams of real Arabic words in TRAIN -- the domain's phrasing."""
    counts = collections.Counter()
    with train_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_docs:
                break
            words = ARABIC_WORD.findall(strip_tags(json.loads(line)["text"]))
            for n in (2, 3):
                for j in range(len(words) - n + 1):
                    gram = words[j:j + n]
                    if all(len(w) > 2 for w in gram):
                        counts[" ".join(gram)] += 1

    # Keep both lengths, half each. 3-grams give `collocation` items (the final word is
    # strongly determined by two preceding words); 2-grams give `defined_term` items
    # (a looser, harder target). Sorting by length alone would starve one category.
    half = max(1, top_k // 2)
    out = []
    for n in (3, 2):
        ranked = [
            g for g, c in counts.most_common()
            if c >= min_count and len(g.split()) == n
        ]
        out.extend(ranked[:half])
    return out


def build_items(test_path, collocations, n, seed, context_chars, per_source_cap):
    rng = random.Random(seed)
    coll_res = [
        (c, re.compile(re.escape(c) + r"(?![ء-ي])"))
        for c in collocations
    ]

    candidates = []
    with test_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = strip_tags(rec["text"])
            source = rec["final_cpt_source"]
            doc_id = f"{source}/{rec['source_id']}"

            # --- statute_term -------------------------------------------------
            for m in STATUTE_RE.finditer(text):
                s, e = m.start("term"), m.end("term")
                if s < context_chars // 2:
                    continue
                candidates.append({
                    "category": "statute_term",
                    "context": text[max(0, s - context_chars):s].rstrip(),
                    "answer": m.group("term"),
                    "source": source,
                    "doc_id": doc_id,
                })

            # --- collocation / defined_term -----------------------------------
            for phrase, rx in coll_res:
                m = rx.search(text)
                if not m:
                    continue
                words = phrase.split()
                answer = words[-1]
                # cut so the context ends just before the final word
                cut = m.start() + len(phrase) - len(answer)
                if cut < context_chars // 2:
                    continue
                candidates.append({
                    "category": "collocation" if len(words) > 2 else "defined_term",
                    "context": text[max(0, cut - context_chars):cut].rstrip(),
                    "answer": answer,
                    "phrase": phrase,
                    "source": source,
                    "doc_id": doc_id,
                })

    # Drop items whose answer is already visible in their own context -- the model could
    # copy it rather than know it.
    kept = []
    for c in candidates:
        if normalize(c["answer"]) not in normalize(c["context"]):
            kept.append(c)
    dropped_copyable = len(candidates) - len(kept)

    # Balance across categories, then across sources, then cap per document so a few long
    # documents cannot dominate the probe.
    by_cat = collections.defaultdict(list)
    for c in kept:
        by_cat[c["category"]].append(c)

    items, seen_docs = [], collections.Counter()
    per_cat = max(1, n // max(1, len(by_cat)))
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        rng.shuffle(pool)
        by_src = collections.defaultdict(list)
        for c in pool:
            by_src[c["source"]].append(c)

        taken, idx = 0, 0
        srcs = sorted(by_src)
        while taken < per_cat and any(by_src.values()):
            src = srcs[idx % len(srcs)]
            idx += 1
            if not by_src[src]:
                continue
            c = by_src[src].pop()
            if seen_docs[c["doc_id"]] >= per_source_cap:
                continue
            seen_docs[c["doc_id"]] += 1
            items.append(c)
            taken += 1

    rng.shuffle(items)
    items = items[:n]
    for i, c in enumerate(items):
        c["id"] = i
    return items, dropped_copyable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="splits/test_doclevel.jsonl")
    ap.add_argument("--train", default="splits/train_doclevel.jsonl")
    ap.add_argument("--out", default="eval/legal_cloze.json")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context_chars", type=int, default=600)
    ap.add_argument("--mine_docs", type=int, default=4000)
    ap.add_argument("--min_count", type=int, default=40)
    ap.add_argument("--top_phrases", type=int, default=120)
    ap.add_argument("--per_doc_cap", type=int, default=2)
    args = ap.parse_args()

    test_path, train_path = Path(args.test), Path(args.train)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Mining domain phrases from {args.mine_docs:,} TRAIN records...")
    collocations = mine_collocations(
        train_path, args.mine_docs, args.min_count, args.top_phrases
    )
    print(f"  kept {len(collocations)} phrases, e.g.:")
    for c in collocations[:5]:
        print(f"    {c}")

    print(f"\nBuilding cloze items from held-out TEST documents...")
    items, dropped = build_items(
        test_path, collocations, args.n, args.seed,
        args.context_chars, args.per_doc_cap,
    )

    payload = {
        "meta": {
            "n": len(items),
            "seed": args.seed,
            "context_chars": args.context_chars,
            "built_from": str(test_path),
            "phrases_mined_from": str(train_path),
            "note": "Targets are legal terms only. Arbitrary identifiers "
                    "(decree numbers, dates, case numbers) are deliberately not masked: "
                    "they are unpredictable from context and would measure noise.",
        },
        "items": items,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)

    print(f"\nWrote {len(items)} items -> {out_path}")
    print(f"  dropped (answer visible in its own context): {dropped:,}")
    print("\n  by category:")
    for k, v in collections.Counter(i["category"] for i in items).most_common():
        print(f"    {k:14} {v:>5}")
    print("\n  by source:")
    for k, v in collections.Counter(i["source"] for i in items).most_common():
        print(f"    {k:28} {v:>5}")
    print("\n  distinct documents:",
          len({i["doc_id"] for i in items}))

    if len(items) < 200:
        print("\n  WARNING: fewer than the 200-item floor in the timeline. "
              "Raise --top_phrases or --per_doc_cap.")


if __name__ == "__main__":
    main()
