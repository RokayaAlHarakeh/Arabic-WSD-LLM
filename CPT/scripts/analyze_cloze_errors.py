# -*- coding: utf-8 -*-
"""
Paired error analysis of the legal cloze probe: base vs CPT on the same items.

The Part 1 counterpart is Gemma/Fine-tuning/Dataset-A/error_analysis/, which
decomposes one model's errors. This does something the SFT analysis could not:
because both models were scored on the *same 300 items, aligned by id*, the
comparison is paired, so McNemar's test applies and the improvement can be
tested rather than merely reported.

CPU only. Reads the evaluation outputs already downloaded from the run; no GPU,
no model load, nothing to re-infer.

    python scripts/analyze_cloze_errors.py

Writes into CPT/error_analysis/:
    per_item_cloze.csv      all 300 items, both models, both scorings
    errors_cpt_only.csv     items the CPT model still gets wrong
    regressions.csv         items the base got right and CPT got wrong
    cloze_analysis.json     the summary tables and the test results
"""

import csv
import io
import json
import os
from collections import Counter, defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "run_gemma2_2b")
OUT = os.path.join(HERE, "..", "error_analysis")


def load(name):
    with io.open(os.path.join(RUN, name), encoding="utf-8") as f:
        return json.load(f)


def mcnemar_exact(b, c):
    """Exact two-sided McNemar on the two discordant cells.

    b = base correct, CPT wrong;  c = base wrong, CPT correct.
    Under H0 each discordant item is a fair coin, so the count follows
    Binomial(b + c, 0.5). Exact rather than chi-square because b is small here
    and the chi-square approximation is unreliable below about 25 discordants.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.96):
    """Wilson score interval -- behaves sensibly near 0 and 1, unlike normal."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    os.makedirs(OUT, exist_ok=True)

    con_b = {r["id"]: r for r in load("cloze_base_constrained.json")["predictions"]}
    con_c = {r["id"]: r for r in load("cloze_cpt_constrained.json")["predictions"]}
    try:
        free_b = {r["id"]: r for r in load("cloze_base.json")["predictions"]}
        free_c = {r["id"]: r for r in load("cloze_cpt.json")["predictions"]}
    except (OSError, KeyError):
        free_b = free_c = {}

    ids = sorted(con_b)
    assert ids == sorted(con_c), "base and CPT item sets differ -- not paired"

    rows = []
    for i in ids:
        b, c = con_b[i], con_c[i]
        rows.append({
            "id": i,
            "category": b["category"],
            "source": b["source"],
            "n_candidates": b.get("n_candidates", ""),
            "answer": b["answer"],
            "base_pred": b["predicted"],
            "cpt_pred": c["predicted"],
            "base_correct": int(bool(b["correct"])),
            "cpt_correct": int(bool(c["correct"])),
            "base_free_pred": (free_b.get(i) or {}).get("predicted", ""),
            "cpt_free_pred": (free_c.get(i) or {}).get("predicted", ""),
            "base_free_correct": int(bool((free_b.get(i) or {}).get("correct", 0))) if free_b else "",
            "cpt_free_correct": int(bool((free_c.get(i) or {}).get("correct", 0))) if free_c else "",
            "outcome": ("both correct" if b["correct"] and c["correct"] else
                        "CPT gained" if c["correct"] else
                        "CPT regressed" if b["correct"] else "both wrong"),
        })

    def write_csv(path, data, fields):
        # utf-8-sig so Excel reads the Arabic columns without mojibake,
        # matching the Part 1 error_analysis CSVs.
        with io.open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in data:
                w.writerow({k: r.get(k, "") for k in fields})

    fields = list(rows[0].keys())
    write_csv(os.path.join(OUT, "per_item_cloze.csv"), rows, fields)
    write_csv(os.path.join(OUT, "errors_cpt_only.csv"),
              [r for r in rows if not r["cpt_correct"]], fields)
    write_csv(os.path.join(OUT, "regressions.csv"),
              [r for r in rows if r["outcome"] == "CPT regressed"], fields)

    # ---- the 2x2 ----------------------------------------------------
    cells = Counter(r["outcome"] for r in rows)
    gained, regressed = cells["CPT gained"], cells["CPT regressed"]
    p = mcnemar_exact(regressed, gained)
    n = len(rows)
    base_k = sum(r["base_correct"] for r in rows)
    cpt_k = sum(r["cpt_correct"] for r in rows)

    print("=" * 66)
    print("PAIRED CLOZE ANALYSIS  (constrained scoring, n = %d)" % n)
    print("=" * 66)
    print("  base accuracy  %5.1f%%   95%% CI [%.1f, %.1f]"
          % (100 * base_k / n, *[100 * x for x in wilson(base_k, n)]))
    print("  CPT  accuracy  %5.1f%%   95%% CI [%.1f, %.1f]"
          % (100 * cpt_k / n, *[100 * x for x in wilson(cpt_k, n)]))
    print()
    print("  both correct    %3d" % cells["both correct"])
    print("  CPT gained      %3d   <- base wrong, CPT right" % gained)
    print("  CPT regressed   %3d   <- base right, CPT wrong" % regressed)
    print("  both wrong      %3d" % cells["both wrong"])
    print()
    print("  McNemar exact two-sided p = %s"
          % ("< 1e-12" if p < 1e-12 else "%.3g" % p))
    print()

    # ---- breakdowns -------------------------------------------------
    def breakdown(key):
        agg = defaultdict(lambda: [0, 0, 0, 0])   # n, base, cpt, regressions
        for r in rows:
            a = agg[r[key]]
            a[0] += 1
            a[1] += r["base_correct"]
            a[2] += r["cpt_correct"]
            a[3] += (r["outcome"] == "CPT regressed")
        return agg

    out = {
        "n": n,
        "base_accuracy": round(100.0 * base_k / n, 1),
        "cpt_accuracy": round(100.0 * cpt_k / n, 1),
        "contingency": {"both_correct": cells["both correct"], "cpt_gained": gained,
                        "cpt_regressed": regressed, "both_wrong": cells["both wrong"]},
        "mcnemar_exact_p": p,
    }

    for key, title in (("category", "BY CATEGORY"), ("source", "BY SOURCE")):
        agg = breakdown(key)
        print(title)
        print("  %-26s %5s %8s %8s %8s %7s" % ("", "n", "base", "CPT", "delta", "regr."))
        block = {}
        for name, (cnt, bk, ck, rg) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print("  %-26s %5d %7.1f%% %7.1f%% %+7.1fpp %7d"
                  % (name, cnt, 100 * bk / cnt, 100 * ck / cnt,
                     100 * (ck - bk) / cnt, rg))
            block[name] = {"n": cnt, "base": round(100 * bk / cnt, 1),
                           "cpt": round(100 * ck / cnt, 1), "regressions": rg}
        out["by_" + key] = block
        print()

    # ---- what CPT still gets wrong ----------------------------------
    wrong = [r for r in rows if not r["cpt_correct"]]
    print("CPT ERRORS: %d of %d" % (len(wrong), n))
    cat = Counter(r["category"] for r in wrong)
    for c, k in cat.most_common():
        print("  %-16s %3d  (%.0f%% of all CPT errors)" % (c, k, 100 * k / len(wrong)))
    out["cpt_error_categories"] = dict(cat)

    with io.open(os.path.join(OUT, "cloze_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nwrote 4 files -> %s" % os.path.normpath(OUT))


if __name__ == "__main__":
    main()
