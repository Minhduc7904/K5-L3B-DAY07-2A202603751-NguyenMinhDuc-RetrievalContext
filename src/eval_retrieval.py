"""Benchmark the retrieval methods on answerable questions: Hit@1, Hit@3, Hit@5, MRR@10, latency.

Queries and filters come from data/questions_public.json; gold evidence from data/questions.json is used ONLY
here, to score. A hit = a retrieved chunk with the gold source_id and page that contains the evidence verbatim
(any `gold` alternative counts), the same rule as starter/baseline.py.

    python src/eval_retrieval.py --split dev --filters none     # unfiltered
    python src/eval_retrieval.py --split dev --filters question # use each question's user-supplied filter
The test split is refused unless --final is passed, and the output folder must not exist yet.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever, load_chunks  # noqa: E402

DEPTH = 10  # ranks scored per query; top-3 is the headline cutoff


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and g["evidence"] in chunk["text"]
               for g in gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--filters", choices=["none", "question"], default="none")
    ap.add_argument("--methods", default="dense,bm25,hybrid")
    ap.add_argument("--reranker", default="minilm", help="rerank method: minilm | bge | model name (hybrid pool -> cross-encoder)")
    ap.add_argument("--pool", type=int, default=30, help="rerank method: size of the hybrid candidate pool")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--final", action="store_true", help="required for the test split; run it once")
    args = ap.parse_args()
    out = args.output or ROOT / "results" / "retrieval" / f"{args.split}-filters-{args.filters}"
    if args.split == "test":
        if not args.final:
            sys.exit("refusing to score the test split without --final (choose the configuration on dev first)")
        if out.exists():
            sys.exit(f"{out} already exists: the test split is scored once")
    chunks = load_chunks()
    public = {q["id"]: q for q in json.loads((ROOT / "data" / "questions_public.json").read_text(encoding="utf-8"))}
    gold_q = [q for q in json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8")) if q["split"] == args.split]

    t0 = time.time()
    dense, bm25 = DenseRetriever(chunks), BM25Retriever(chunks)
    setup_s = round(time.time() - t0, 2)
    all_methods = {"dense": dense, "bm25": bm25, "hybrid": HybridRetriever(dense, bm25)}
    if "rerank" in args.methods.split(","):
        all_methods["rerank"] = RerankRetriever(all_methods["hybrid"], args.reranker, args.pool)
    methods = {m: all_methods[m] for m in args.methods.split(",")}
    dense.search("warm-up query", 1)  # exclude one-off model/connection warm-up from the latency numbers
    if "rerank" in methods:
        methods["rerank"].search("warm-up query", 1)

    summary, traces = {}, []
    for name, retriever in methods.items():
        ranks, times = {}, []
        for q in gold_q:
            pub = public[q["id"]]  # the ONLY view of the question the retriever gets
            filters = pub.get("filter") if args.filters == "question" else None
            t = time.perf_counter()
            results = retriever.search(pub["question"], DEPTH, filters)
            times.append((time.perf_counter() - t) * 1000)
            rank = next((r["rank"] for r in results if is_hit(r, q["gold"])), None)
            ranks[q["id"]] = rank
            traces.append({"method": name, "question_id": q["id"], "question": pub["question"], "filter": filters,
                           "hit_rank": rank, "latency_ms": round(times[-1], 1),
                           "results": [{k: r[k] for k in ("rank", "id", "page", "source_id", "score")}
                                       | ({"components": r["components"]} if "components" in r else {}) for r in results]})
        n = len(gold_q)
        count = lambda k: sum(1 for r in ranks.values() if r is not None and r <= k)  # noqa: E731
        summary[name] = {"n": n, "hit@1": f"{count(1)}/{n}", "hit@3": f"{count(3)}/{n}", "hit@5": f"{count(5)}/{n}",
                         "hit@10": f"{count(10)}/{n}",
                         "mrr@10": round(sum(1 / r for r in ranks.values() if r) / n, 3),
                         "median_ms": round(statistics.median(times), 1), "cost_usd": 0.0,
                         "misses_top3": sorted(i for i, r in ranks.items() if r is None or r > 3)}
    out.mkdir(parents=True, exist_ok=True)
    meta = {"split": args.split, "filters": args.filters, "depth": DEPTH, "setup_seconds": setup_s,
            "n_chunks": len(chunks), "bm25": {"k1": bm25.k1, "b": bm25.b}, "rrf": {"k": 60, "candidates": 50},
            "embedding_model": dense.model_name}
    if "rerank" in methods:
        meta["rerank"] = {"cross_encoder": methods["rerank"].model_name, "pool": args.pool}
    (out / "summary.json").write_text(json.dumps({"meta": meta, "methods": summary}, indent=2), encoding="utf-8")
    (out / "traces.json").write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"split={args.split} filters={args.filters} n={len(gold_q)} setup={setup_s}s -> {out}")
    print(f"{'method':8} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR@10':>7} {'ms':>7}  misses@3")
    for name, s in summary.items():
        print(f"{name:8} {s['hit@1']:>7} {s['hit@3']:>7} {s['hit@5']:>7} {s['hit@10']:>7} {s['mrr@10']:>7} "
              f"{s['median_ms']:>7}  {s['misses_top3']}")


if __name__ == "__main__":
    main()
