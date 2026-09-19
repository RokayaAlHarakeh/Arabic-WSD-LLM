"""
Subsample the Lebanese legal CPT corpus by WHOLE DOCUMENTS before transfer.

Run this on the laptop that has train_cleanheaders.jsonl.
It keeps entire documents together (never splits a document's chunks)
and preserves the per-source proportions of the full corpus.

Usage:
    python subsample_corpus.py \
        --inputs train_cleanheaders.jsonl val_cleanheaders.jsonl test_cleanheaders.jsonl \
        --output legal_corpus_subset.jsonl \
        --target_tokens 30000000
"""
import argparse, json, random, collections


def est_tokens(text):
    # ~2.6 chars per Gemma token on this corpus; close enough for sizing.
    return round(len(text) / 2.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--target_tokens", type=int, default=30_000_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    # ---- load, grouping records into documents -------------------------
    docs = collections.defaultdict(list)          # source_id -> [records]
    doc_source = {}                               # source_id -> corpus source
    total_tokens = 0

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                text = r.get("text", "") or ""
                if not text.strip():
                    continue
                sid = r.get("source_id") or r.get("id")
                docs[sid].append(r)
                doc_source[sid] = r.get("final_cpt_source") or r.get("source") or "unknown"
                total_tokens += est_tokens(text)

    doc_tokens = {
        sid: sum(est_tokens(x.get("text", "")) for x in recs)
        for sid, recs in docs.items()
    }

    print(f"documents           : {len(docs):,}")
    print(f"records             : {sum(len(v) for v in docs.values()):,}")
    print(f"estimated tokens    : {total_tokens:,}")
    print(f"target tokens       : {args.target_tokens:,}")
    print(f"keep fraction       : {args.target_tokens / max(1, total_tokens):.1%}\n")

    # ---- per-source proportional quota ---------------------------------
    by_source = collections.defaultdict(list)
    for sid in docs:
        by_source[doc_source[sid]].append(sid)

    source_tokens = {
        s: sum(doc_tokens[sid] for sid in sids) for s, sids in by_source.items()
    }

    kept, kept_tokens = [], 0
    print(f"{'source':<28}{'docs':>8}{'kept':>8}{'tokens':>14}")
    print("-" * 58)

    for source, sids in sorted(by_source.items()):
        quota = args.target_tokens * (source_tokens[source] / total_tokens)
        random.shuffle(sids)
        taken, t = 0, 0
        for sid in sids:
            if t >= quota:
                break
            kept.append(sid)
            t += doc_tokens[sid]
            taken += 1
        kept_tokens += t
        print(f"{source:<28}{len(sids):>8}{taken:>8}{t:>14,}")

    print("-" * 58)
    print(f"{'TOTAL':<28}{len(docs):>8}{len(kept):>8}{kept_tokens:>14,}\n")

    # ---- write, documents contiguous and chunk order preserved ---------
    random.shuffle(kept)
    n_records = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for sid in kept:
            for r in sorted(docs[sid], key=lambda x: x.get("chunk_index") or 0):
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_records += 1

    print(f"wrote {n_records:,} records from {len(kept):,} documents -> {args.output}")
    print("\nNext: gzip it, then upload to Drive.")
    print("  gzip -9 " + args.output)


if __name__ == "__main__":
    main()
