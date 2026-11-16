"""Run the real-model agent over one split with one retrieval method; write traces and a manual review sheet.

    python src/run_agent_eval.py --split dev --method hybrid --tag base

Outputs go to results/agent/<split>-<method>-<tag>/:
  traces.json            one trace per question (query_id, status, events, citations, context, answer, tokens, cost ...)
  traces.partial.jsonl   written after EVERY question while the run is going (so a crash loses nothing); removed at the end
  cases.json             [{id, split, answerable, question}] in the format starter/agent_metrics.py expects
  summary.json           automatic metrics only (see below) + the full configuration
  reviews_template.json  BLANK manual labels (context_grade, citation_support, answer_grade / abstention_correct) for a human grader

Automatic metrics: Context Evidence Rate (does the FINAL context contain a gold evidence string verbatim in a chunk of the
right source and page; gold is read only here, after the agent ran, and is never passed to the agent), tool-use rate,
search-count distribution, answered/abstained/error counts, citation-id validity (both "accepted only" and "attempted"),
latency, tokens, estimated cost. Answers and citations are NOT graded automatically: the human fills reviews_template.json,
then starter/agent_metrics.py aggregates it. CER is a context metric, NOT answer quality.

Statuses: answered | abstain | error. `error` = provider / retriever / infrastructure failure (excluded from quality metrics and
reported separately). Model-side format failures end as `abstain` (reason model_format_error / truncated_answer) and DO count
against the agent.

The test split is refused unless --final is given, the output directory is the default one and does not exist yet, --only is
not used, and fewer than 2 test runs exist IN TOTAL (any method; the plan is exactly two hand-run test runs: the baseline and the
configuration chosen on dev). Dev output is not overwritten
unless --overwrite is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent import (AGENT_VERSION, FALLBACK_PRICE, PRICES_VERIFIED, PROMPTS, OpenAIController, RetrieverAdapter,  # noqa: E402
                   estimate_cost, load_llm_env, price_for, redact, run_agent)

RESULTS = ROOT / "results" / "agent"
DEFAULTS = {"top_k": 5, "budget": 10000, "max_searches": 3}
MAX_TEST_RUNS = 2  # GLOBAL cap over every method: the baseline and the configuration chosen on dev


def check_split_policy(split: str, final: bool, out_dir: Path, *, method: str | None = None, only: str | None = None,
                       default_output: bool = True, overwrite: bool = False) -> None:
    """Refuse the test split unless it is a single, whole, default-named, new run; protect dev baselines from overwriting."""
    if split == "test":
        if not final:
            sys.exit("refusing to run the test split without --final (choose the configuration on dev first)")
        if only:
            sys.exit("refusing --only with the test split: it is scored whole, once")
        if not default_output:
            sys.exit("refusing a custom --output for the test split")
        if out_dir.exists():
            sys.exit(f"{out_dir} already exists: the test split is run once")
        done = [d for d in out_dir.parent.glob("test-*") if d.is_dir()]
        if len(done) >= MAX_TEST_RUNS:
            sys.exit(f"{len(done)} test runs already exist ({MAX_TEST_RUNS} allowed in total, whatever the method: baseline and chosen configuration)")
    elif out_dir.exists() and (out_dir / "summary.json").exists() and not overwrite:
        sys.exit(f"{out_dir} already holds results: use another --tag or pass --overwrite")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_hit(chunk: dict, gold: list[dict]) -> bool:
    return any(chunk["source_id"] == g["source_id"] and chunk["page"] == g["page"] and g["evidence"] in chunk["text"] for g in gold)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_retriever(method: str, chunks: list[dict], reranker: str = "minilm", pool: int = 30):
    from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, RerankRetriever
    if method == "bm25":
        return BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    dense.search("warm-up query", 1)  # exclude one-off model/connection warm-up from latency numbers
    if method == "dense":
        return dense
    hybrid = HybridRetriever(dense, BM25Retriever(chunks))
    if method == "hybrid":
        return hybrid
    rerank = RerankRetriever(hybrid, reranker, pool)
    rerank.search("warm-up query", 1)
    return rerank


def run_questions(questions: list[dict], controller, retriever, *, filters_mode: str, top_k: int, budget: int,
                  max_searches: int, model: str, max_cost: float, on_trace=None, quiet: bool = False,
                  repair_ids: bool = False) -> tuple[list[dict], bool]:
    """Only public fields (id, question, filter) reach the agent. Returns (traces, aborted_for_cost)."""
    traces, spent = [], 0.0
    for q in questions:
        filters = q.get("filter") if filters_mode == "question" else None
        trace: dict = {}
        t0 = time.perf_counter()
        try:
            run_agent(q["question"], controller, RetrieverAdapter(retriever), filters=filters, max_searches=max_searches,
                      top_k=top_k, budget=budget, trace=trace, repair_ids=repair_ids)
        except Exception as exc:  # provider / retriever / infrastructure failure: recorded, never replaced by a fake result
            err = {"type": type(exc).__name__, "message": redact(str(exc))[:400]}
            if hasattr(exc, "failed_attempts"):
                err["failed_attempts"] = exc.failed_attempts
            trace.update(status="error", answer="", raw_answer="", citations=[], reason="error", error=err)
            for key, default in (("context", ""), ("context_ids", []), ("context_chars", 0), ("n_searches", 0), ("events", []),
                                 ("attempted_citations", []), ("attempted_citations_raw", []), ("invalid_citations", [])):
                trace.setdefault(key, default)  # run_agent keeps the partial context up to date, so it survives the error
        usage = trace.get("usage") or {"prompt_tokens": 0, "completion_tokens": 0}
        cost = estimate_cost(model, usage["prompt_tokens"], usage["completion_tokens"])
        trace.update(query_id=q["id"], split=q["split"], question=q["question"], filter=filters, model=trace.get("model") or model,
                     elapsed_s=round(time.perf_counter() - t0, 3), cost_usd=round(cost, 6), error=trace.get("error"))
        traces.append(trace)
        if on_trace:
            on_trace(trace)
        spent += cost
        if not quiet:
            print(f"  {q['id']:4} {trace['status']:8} searches={trace['n_searches']} {trace['elapsed_s']:6.1f}s "
                  f"tok={usage['prompt_tokens']}+{usage['completion_tokens']} ${cost:.5f} {trace.get('reason', '')}", flush=True)
        if spent > max_cost:
            print(f"STOP: estimated spend ${spent:.4f} exceeded --max-cost ${max_cost:.2f}", flush=True)
            return traces, True
    return traces, False


def evaluate(traces: list[dict], split: str, chunks_by_id: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Attach gold-based automatic checks (evaluator side only). Returns (cases, review_items)."""
    gold = {q["id"]: q for q in load_json(ROOT / "data" / "questions.json") if q["split"] == split}
    absent = {q["id"]: q for q in load_json(ROOT / "data" / "absent_questions.json") if q["split"] == split}
    cases, reviews = [], []
    for t in traces:
        qid = t["query_id"]
        answerable = qid in gold
        if not answerable and qid not in absent:
            raise ValueError(f"{qid} is in neither questions.json nor absent_questions.json for split {split}")
        t["answerable"] = answerable
        if answerable:
            if t["status"] == "error":  # no valid run: neither a hit nor a miss (reported separately)
                t["evidence_in_context"] = None
            else:
                g = gold[qid]["gold"]
                in_ctx = [i for i in t["context_ids"] if is_hit(chunks_by_id[i], g)]
                t["evidence_in_context"] = bool(in_ctx)
                t["evidence_chunk_ids_in_context"] = in_ctx
                searches = [e for e in t["events"] if e.get("event") == "search"]
                retrieved = {r["id"] for e in searches for r in e["results"] if is_hit(chunks_by_id[r["id"]], g)}
                t["evidence_retrieved_not_in_context"] = sorted(retrieved - set(t["context_ids"]))
                t["first_search_evidence_rank"] = next((r["rank"] for r in (searches[0]["results"] if searches else []) if r["id"] in retrieved), None)
                t["evidence_after_first_search"] = bool(searches) and any(i in retrieved for i in searches[0]["new_ids"])
        cases.append({"id": qid, "split": split, "answerable": answerable, "question": t["question"]})
        if t["status"] == "error":
            continue  # provider/retrieval errors need no labels (starter/agent_metrics.py)
        item = {"query_id": qid, "question": t["question"], "status": t["status"], "abstain_reason": t.get("reason"), "answerable": answerable,
                "answer": t["answer"], "model_text": t.get("raw_answer"), "citations": t["citations"],
                "attempted_citations": t.get("attempted_citations", []),
                "attempted_citations_raw": t.get("attempted_citations_raw", t.get("attempted_citations", [])), "invalid_citations": t.get("invalid_citations", []),
                "context_grade": None, "citation_support": [None] * len(t["citations"]),
                "_how_to": ("MANUAL grading, not filled by any script. context_grade (BENCHMARK.md section 2): 0 no evidence, 1 evidence but a "
                            "condition or source missing, 2 enough evidence and source in the final context ('context' in traces.json). "
                            "citation_support: true/false per accepted citation (does that passage support the claim it is attached to?). "
                            + ("answer_grade: 0 wrong/missing evidence/abstained, 1 partly right, 2 right and complete (BENCHMARK.md section 3)."
                               if answerable else "abstention_correct: true only if the agent really abstained and added no unsupported claim."))}
        if answerable:
            item["answer_grade"] = None
            item["grader_reference"] = {"expected_answer": gold[qid]["expected_answer"],
                                        "gold": [{k: x[k] for k in ("source_id", "page", "evidence")} for x in gold[qid]["gold"]]}
        else:
            item["abstention_correct"] = None
            item["grader_reference"] = {k: absent[qid].get(k) for k in ("absence_reason", "absence_check", "grader_note")}
        reviews.append(item)
    return cases, reviews


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("openai", "psycopg", "sentence-transformers", "python-dotenv", "torch"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def summarize(traces: list[dict], args, model: str, aborted: bool, setup: dict, controller_settings: dict) -> dict:
    n = len(traces)
    ok = [t for t in traces if t["status"] != "error"]
    answerable = [t for t in traces if t["answerable"]]
    ans_ok = [t for t in answerable if t["status"] != "error"]
    absent = [t for t in traces if not t["answerable"]]
    ptok = sum(t["usage"]["prompt_tokens"] for t in traces if t.get("usage"))
    ctok = sum(t["usage"]["completion_tokens"] for t in traces if t.get("usage"))
    (p_in, p_out), listed = price_for(model)
    lat = [t["elapsed_s"] for t in ok]
    searches = Counter(t["n_searches"] for t in ok)
    cer = sum(bool(t["evidence_in_context"]) for t in ans_ok)
    attempted = [i for t in ok for i in t.get("attempted_citations", [])]
    attempted_raw = [i for t in ok for i in t.get("attempted_citations_raw", t.get("attempted_citations", []))]
    invalid = [i for t in ok for i in t.get("invalid_citations", [])]
    accepted = [i for t in ok for i in t["citations"]]
    snapshots = sorted({e["model"] for t in traces for e in t["events"] if e.get("event") == "model_call" and e.get("model")})
    zero_new = sum(1 for t in ok for e in t["events"] if e.get("event") == "search" and e["step"] > 1 and not e["new_ids"])
    return {
        "agent_version": AGENT_VERSION, "split": args.split, "method": args.method, "tag": args.tag, "model": model,
        "model_snapshots_seen": snapshots, "aborted_for_cost": aborted,
        "config": {"top_k": args.top_k, "context_budget": args.budget, "context_budget_unit": "characters (formatted passage blocks, whole run)",
                   "max_searches": args.max_searches, "filters": args.filters, "repair_ids": getattr(args, "repair_ids", False), "dedup": "by chunk id across searches",
                   "stop": "model answers without a tool call, or max_searches reached then one forced answer turn (tool_choice=none)",
                   "citation_rule": "all cited [chunk-id] must be in the final context and in the answer text, else abstain",
                   **controller_settings, "filters_provenance": (
                       "filters come from data/questions_public.json: 5 of 21 dev answerable questions (d05, d06, d13, d18, d21) carry a user-supplied "
                       "document/chapter scope; no absent question and no test question has one, so dev with filters is optimistic for those 5 "
                       "compared with a no-filter setting. --filters none ignores them."),
                   "agent_code_sha256": sha256_file(ROOT / "src" / "agent.py"),
                   "system_prompt_sha256": sha256_text(controller_settings.get("system_prompt", "")),
                   "chunks_sha256": sha256_file(ROOT / "data" / "chunks.json"),
                   "questions_public_sha256": sha256_file(ROOT / "data" / "questions_public.json"),
                   **setup},
        "packages": package_versions(),
        "price_usd_per_1M_tokens": {"input": p_in, "output": p_out, "listed_in_PRICES_table": listed, "verified_against_openai": PRICES_VERIFIED,
                                    "note": "UNVERIFIED: verify against OpenAI pricing. Fallback constant (gpt-4o level) "
                                            f"{FALLBACK_PRICE} is used for unknown models; cached-input discounts ignored"},
        "n": n, "n_answerable": len(answerable), "n_absent": len(absent),
        "status": {"answered": sum(t["status"] == "answered" for t in traces),
                   "abstain": sum(t["status"] == "abstain" for t in traces),
                   "error": sum(t["status"] == "error" for t in traces)},
        "abstain_reasons": dict(Counter(t["reason"] for t in traces if t["status"] == "abstain")),
        "errors": [{"query_id": t["query_id"], **t["error"]} for t in traces if t["status"] == "error"],
        "context_evidence_rate": {"n": cer, "N": len(ans_ok), "rate": round(cer / len(ans_ok), 3) if ans_ok else None,
                                  "answerable_total": len(answerable), "errors_excluded": len(answerable) - len(ans_ok),
                                  "note": "context metric, NOT answer quality; provider errors are excluded and reported separately; "
                                          "evidence = gold string verbatim in a final-context chunk of the gold source+page"},
        "evidence_after_first_search": {"n": sum(bool(t.get("evidence_after_first_search")) for t in ans_ok), "N": len(ans_ok)},
        "tool_use": {"n": sum(t["n_searches"] > 0 for t in ok), "N": len(ok)},
        "search_count_distribution": {str(k): v for k, v in sorted(searches.items())},
        "mean_searches": round(statistics.mean(t["n_searches"] for t in ok), 2) if ok else None,
        "searches_with_zero_new_chunks": zero_new,
        "citation_id_validity": {"accepted_ids": len(accepted), "attempted_ids": len(attempted), "attempted_ids_raw": len(attempted_raw), "invalid_ids": len(invalid), "repaired_ids": sum(len(t.get("repaired_citations", [])) for t in ok),
                                 "answers_with_invalid_citation": sorted(t["query_id"] for t in ok if t.get("invalid_citations")),
                                 "note": "id check only (id is in the final context and in the text). Attempted counts ids of abstained answers too (after the optional para:-prefix repair; attempted_ids_raw = as the model wrote them); "
                                         "semantic support is manual (reviews_template.json)"},
        "abstained_on_answerable": [t["query_id"] for t in answerable if t["status"] == "abstain"],
        "answered_on_absent": [t["query_id"] for t in absent if t["status"] == "answered"],
        "evidence_missing_from_context": [t["query_id"] for t in ans_ok if not t["evidence_in_context"]],
        "mean_context_chars": round(statistics.mean(t.get("context_chars", 0) for t in ok), 1) if ok else None,
        "latency_s": {"median": round(statistics.median(lat), 2) if lat else None, "mean": round(statistics.mean(lat), 2) if lat else None,
                      "max": max(lat) if lat else None, "total": round(sum(lat), 1), "note": "use the median; single outliers occur"},
        "tokens": {"prompt": ptok, "completion": ctok}, "estimated_cost_usd": round(estimate_cost(model, ptok, ctok), 5),
        "statistical_note": "single run per configuration, n=21 answerable and 2 absent questions: differences between methods are not significant.",
        "manual_grading": "NOT DONE. context_grade, answer_grade, citation_support and abstention_correct are blank in reviews_template.json for a human grader.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--method", choices=["dense", "bm25", "hybrid", "rerank"], required=True)
    ap.add_argument("--reranker", default="minilm", help="rerank method only: minilm | bge | a CrossEncoder model name")
    ap.add_argument("--pool", type=int, default=30, help="rerank method only: candidates taken from hybrid before reranking")
    ap.add_argument("--repair-ids", action="store_true", help="variant C: repair a cited id that is the unique suffix of a shown id (logged)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="base", help="system prompt variant (config B = grounded)")
    ap.add_argument("--tag", default="base", help="config name, part of the output folder")
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    ap.add_argument("--budget", type=int, default=DEFAULTS["budget"], help="context budget in characters")
    ap.add_argument("--max-searches", type=int, default=DEFAULTS["max_searches"])
    ap.add_argument("--filters", choices=["none", "question"], default="question", help="question = user filter from questions_public.json")
    ap.add_argument("--only", help="comma-separated question ids (smoke test, dev only)")
    ap.add_argument("--max-cost", type=float, default=1.50, help="stop if the estimated spend of this run exceeds this many USD")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--overwrite", action="store_true", help="allow overwriting an existing dev result folder")
    ap.add_argument("--final", action="store_true", help="required for the test split")
    args = ap.parse_args()
    out = args.output or RESULTS / f"{args.split}-{args.method}-{args.tag}"
    check_split_policy(args.split, args.final, out, method=args.method, only=args.only, default_output=args.output is None, overwrite=args.overwrite)

    from retrieval import load_chunks
    chunks = load_chunks()
    chunks_by_id = {c["id"]: c for c in chunks}
    questions = [q for q in load_json(ROOT / "data" / "questions_public.json") if q["split"] == args.split]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    if not questions:
        sys.exit("no questions selected")

    key, model = load_llm_env()
    import openai
    client = openai.OpenAI(api_key=key, max_retries=0, timeout=60.0)
    temperature = None if model.startswith(("gpt-5", "o1", "o3", "o4")) else 0.0
    controller = OpenAIController(client, model, temperature=temperature, prompt=args.prompt)
    retriever = make_retriever(args.method, chunks, args.reranker, args.pool)
    (p_in, p_out), listed = price_for(model)
    print(f"agent={AGENT_VERSION} model={model} price(in/out per 1M)={p_in}/{p_out} (UNVERIFIED, listed={listed}) method={args.method} split={args.split} "
          f"n={len(questions)} top_k={args.top_k} budget={args.budget}chars max_searches={args.max_searches} filters={args.filters}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    partial = out / "traces.partial.jsonl"
    partial.write_text("", encoding="utf-8")

    def save_partial(trace: dict) -> None:
        with partial.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

    traces, aborted = run_questions(questions, controller, retriever, filters_mode=args.filters, top_k=args.top_k, budget=args.budget,
                                    max_searches=args.max_searches, model=model, max_cost=args.max_cost, on_trace=save_partial,
                                    repair_ids=args.repair_ids)
    cases, reviews = evaluate(traces, args.split, chunks_by_id)
    setup: dict = {"retrieval": args.method, "n_chunks": len(chunks)}
    base = retriever.base if args.method == "rerank" else retriever  # the hybrid under a reranker
    dense_part = base if args.method == "dense" else getattr(base, "dense", None)
    if dense_part is not None:
        setup.update(embedding_model=dense_part.model_name, vector_store="Day 6 pgvector, strategy B, exact cosine")
    if args.method != "dense":
        bm = base if args.method == "bm25" else base.bm25
        setup["bm25"] = {"k1": bm.k1, "b": bm.b}
    if args.method in ("hybrid", "rerank"):
        setup["rrf"] = {"k": base.rrf_k, "candidates": base.candidates}
    if args.method == "rerank":
        setup["rerank"] = {"cross_encoder": retriever.model_name, "pool": retriever.pool, "returns": args.top_k}
    summary = summarize(traces, args, model, aborted, setup, controller.settings())

    for name, obj in (("traces.json", traces), ("cases.json", cases), ("summary.json", summary), ("reviews_template.json", reviews)):
        (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    s = summary
    print(f"\n{args.split}/{args.method}/{args.tag}: answered={s['status']['answered']} abstain={s['status']['abstain']} error={s['status']['error']} "
          f"CER={s['context_evidence_rate']['n']}/{s['context_evidence_rate']['N']} tool_use={s['tool_use']['n']}/{s['tool_use']['N']} "
          f"searches={s['search_count_distribution']} mean_searches={s['mean_searches']} median_latency={s['latency_s']['median']}s "
          f"tokens={s['tokens']['prompt']}+{s['tokens']['completion']} cost=${s['estimated_cost_usd']}")
    print(f"abstained_on_answerable={s['abstained_on_answerable']} answered_on_absent={s['answered_on_absent']} "
          f"invalid_citations={s['citation_id_validity']['answers_with_invalid_citation']} -> {out}")


if __name__ == "__main__":
    main()
