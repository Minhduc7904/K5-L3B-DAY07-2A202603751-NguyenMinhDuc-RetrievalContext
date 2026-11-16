"""Collect every dev agent run into results/dev_comparison.json and results/DEV_COMPARISON.md and apply the preregistered rule
of results/SELECTION.md.

    python src/compare_runs.py

Automatic metrics come from each run's summary.json and from results/retrieval/. Graded metrics (answer-grade sum, supported
citations, abstention_correct) are read from `reviews.json` in a run folder when it exists (independent labels) and are skipped
otherwise, so the script can be run again after grading; the decision computed with labels is the binding one.
The rule implemented is results/SELECTION.md section 3 as amended by amendment 4 (25% tie-break, L3 as a share, clause R, a significantly worse
challenger never wins). A decision is FINAL only when every compared run and every noise pair is graded; otherwise it is PROVISIONAL with warnings.
Hand-written commentary lives in results/dev_comparison_reading.md and is copied into the report. Test split: never read.
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "results" / "agent"
RETR = ROOT / "results" / "retrieval"
RERANK_DIR = {"cross-encoder/ms-marco-MiniLM-L-6-v2": "minilm", "BAAI/bge-reranker-base": "bge"}
SKIP = re.compile(r"-(v0|v1a)$")
BASELINE = "dev-hybrid-base-nofilter"
BASELINE_REP = "dev-hybrid-base-nofilter-rep2"
# challengers in the order fixed in SELECTION.md section 3.3 (folder names); AB only if it exists
CHALLENGERS = ["dev-bm25-base-nofilter", "dev-dense-base-nofilter", "dev-rerank-minilm-nofilter", "dev-hybrid-grounded-nofilter",
               "dev-hybrid-repair-nofilter", "dev-hybrid-grounded-repair-nofilter", "dev-rerank-minilm-grounded-repair-nofilter"]  # SELECTION.md amendment 3


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_hits(method: str, filters: str, summary: dict) -> dict | None:
    """Hit@k of the retrieval method on the raw question (retrieval-only benchmark), independent of the agent."""
    if method == "rerank":
        name = RERANK_DIR.get(summary["config"].get("rerank", {}).get("cross_encoder"))
        path = RETR / f"dev-filters-{filters}-rerank-{name}" / "summary.json"
    else:
        path = RETR / f"dev-filters-{filters}" / "summary.json"
    if not path.exists():
        return None
    m = load(path)["methods"].get(method)
    return {k: m[k] for k in ("hit@1", "hit@3", "hit@5", "mrr@10")} if m else None


def graded(folder: Path, traces: list[dict]) -> dict | None:
    path = folder / "reviews.json"
    if not path.exists():
        return None
    reviews = {r["query_id"]: r for r in load(path)}
    ans = [t for t in traces if t["answerable"] and t["status"] != "error"]
    absent = [t for t in traces if not t["answerable"] and t["status"] != "error"]
    pts = sum(reviews[t["query_id"]]["answer_grade"] for t in ans if t["query_id"] in reviews)
    sup = sum(sum(bool(x) for x in r["citation_support"]) for r in reviews.values())
    tot = sum(len(r["citation_support"]) for r in reviews.values())
    abst = sum(bool(reviews[t["query_id"]].get("abstention_correct")) for t in absent if t["query_id"] in reviews)
    ctx = [reviews[t["query_id"]].get("context_grade") for t in ans if t["query_id"] in reviews]
    return {"answer_points": pts, "answer_points_max": 2 * len(ans), "supported_citations": sup, "accepted_citations": tot,
            "abstention_correct": abst, "n_absent": len(absent), "context_grade_2": sum(c == 2 for c in ctx if c is not None), "context_graded": sum(c is not None for c in ctx)}


def row(folder: Path) -> dict:
    s = load(folder / "summary.json")
    traces = load(folder / "traces.json")
    filters = s["config"]["filters"]
    ans = [t for t in traces if t["answerable"]]
    absent = [t for t in traces if not t["answerable"]]
    cnt = lambda ts, st: sum(t["status"] == st for t in ts)  # noqa: E731
    cer = s["context_evidence_rate"]
    return {
        "run": folder.name, "method": s["method"], "tag": s["tag"], "filters": filters, "prompt": s["config"].get("prompt", "base"),
        "repair_ids": bool(s["config"].get("repair_ids", False)), "repaired": s["citation_id_validity"].get("repaired_ids", 0),
        "agent_version": s["agent_version"],
        "reranker": s["config"].get("rerank", {}).get("cross_encoder"),
        "retrieval_dev": retrieval_hits(s["method"], filters, s),
        "cer": {"n": cer["n"], "N": cer["N"]}, "tool_use": s["tool_use"], "search_counts": s["search_count_distribution"], "mean_searches": s["mean_searches"],
        "answerable": {"answered": cnt(ans, "answered"), "abstain": cnt(ans, "abstain"), "error": cnt(ans, "error"), "N": len(ans)},
        "absent": {"answered": cnt(absent, "answered"), "abstain": cnt(absent, "abstain"), "error": cnt(absent, "error"), "N": len(absent),
                   "abstain_by_model": sum(t["status"] == "abstain" and t["reason"] == "model_abstained" for t in absent)},
        "abstain_reasons": s["abstain_reasons"], "abstained_on_answerable": s["abstained_on_answerable"], "answered_on_absent": s["answered_on_absent"],
        "evidence_missing_from_context": s["evidence_missing_from_context"],
        "median_latency_s": s["latency_s"]["median"], "tokens": s["tokens"], "cost_usd": s["estimated_cost_usd"], "errors": len(s["errors"]),
        "graded": graded(folder, traces),
    }


# Constants of SELECTION.md (amendment 4). They are written down before the graded recomputation and are not tuned afterwards.
MIN_T = 2                 # levels L1, L2: T = max(2, run-to-run noise of the identical baseline pair)
MIN_T_SHARE = 0.10        # level L3 (share of accepted citations that are supported): T = max(0.10, noise)
TIEBREAK_PCT = 0.25       # latency / cost tie-break: more than 25% apart (the baseline pair itself differs by ~10% in latency)
R_MIN_HIT1_GAIN = 3       # retrieval-gain clause R: Hit@1 better by at least 3 questions ...
R_MAX_LATENCY_RATIO = 1.4  # ... at most 40% more median latency, Hit@3 not lower, and (gates) no loss at agent level


def count_of(value) -> int | None:
    """'16/21' -> 16 (the retrieval summaries store Hit@k as 'n/N')."""
    if value is None:
        return None
    return int(str(value).split("/")[0])


def share(g: dict | None) -> float | None:
    return None if not g or not g["accepted_citations"] else g["supported_citations"] / g["accepted_citations"]


def method_noise(rows: dict[str, dict], method: str, prompt_base: str = "base") -> dict:
    """Run-to-run difference of the two identical base-nofilter runs of a method (SELECTION.md amendment 1: methods of both compared runs).
    `graded_pair` tells whether both runs of the pair carry labels (needed for a FINAL decision)."""
    a, b = rows.get(f"dev-{method}-base-nofilter"), rows.get(f"dev-{method}-base-nofilter-rep2")
    if not (a and b):
        return {}
    out = {"cer": abs(a["cer"]["n"] - b["cer"]["n"]), "graded_pair": bool(a["graded"] and b["graded"])}
    if a["graded"] and b["graded"]:
        out["answer_points"] = abs(a["graded"]["answer_points"] - b["graded"]["answer_points"])
        out["supported_citations"] = abs(a["graded"]["supported_citations"] - b["graded"]["supported_citations"])
        sa, sb = share(a["graded"]), share(b["graded"])
        if sa is not None and sb is not None:
            out["support_share"] = round(abs(sa - sb), 3)
    return out


def gates(c: dict, inc: dict) -> dict:
    g1 = c["cer"]["n"] >= inc["cer"]["n"] - 1
    g2 = c["errors"] == 0 and c["answerable"]["abstain"] <= inc["answerable"]["abstain"] + 1
    if c["graded"] and inc["graded"]:
        g3, g3_basis = c["graded"]["abstention_correct"] >= inc["graded"]["abstention_correct"], "graded"
    else:
        g3, g3_basis = c["absent"]["abstain_by_model"] >= inc["absent"]["abstain_by_model"], "proxy (status abstain, reason model_abstained)"
    return {"G1_cer": g1, "G2_errors_and_false_abstentions": g2, "G3_absent_abstentions": g3, "G3_basis": g3_basis, "pass": g1 and g2 and g3}


def retrieval_gain(c: dict, inc: dict) -> dict:
    """Clause R (amendment 4): a clear retrieval-only gain with no loss at agent level. Applied only when no level L1-L3 decided.
    Its thresholds were proposed AFTER the dev retrieval numbers of the reranker were known (not a blind preregistration)."""
    hc, hi = c.get("retrieval_dev") or {}, inc.get("retrieval_dev") or {}
    h1c, h1i, h3c, h3i = (count_of(h.get(k)) for h, k in ((hc, "hit@1"), (hi, "hit@1"), (hc, "hit@3"), (hi, "hit@3")))
    if None in (h1c, h1i, h3c, h3i):
        return {"applies": False, "why": "retrieval benchmark numbers missing"}
    lat_c, lat_i = c["median_latency_s"], inc["median_latency_s"]
    ratio = lat_c / lat_i if lat_i else float("inf")
    ok = h1c - h1i >= R_MIN_HIT1_GAIN and h3c >= h3i and ratio <= R_MAX_LATENCY_RATIO
    return {"applies": ok, "hit1": [h1c, h1i], "hit3": [h3c, h3i], "latency_ratio": round(ratio, 3),
            "why": f"Hit@1 {h1c} vs {h1i} (need +{R_MIN_HIT1_GAIN}), Hit@3 {h3c} vs {h3i} (not lower), latency x{ratio:.2f} (max x{R_MAX_LATENCY_RATIO})"}


def challenge(c: dict, inc: dict, rows: dict) -> dict:
    """Section 3.2 of SELECTION.md as amended (amendment 4). Returns the verdict, the reasoning steps and `decided_by`.

    Order: gates; L1 / L2 / L3 (the first level where the runs differ by at least T decides, in EITHER direction: a significantly worse
    challenger never wins); then clause R (retrieval gain); then the latency / cost tie-break (25%); otherwise the incumbent stays."""
    steps, gt = [], gates(c, inc)

    def out(replaces: bool, decided_by: str) -> dict:
        return {"challenger": c["run"], "incumbent": inc["run"], "gates": gt, "replaces": replaces, "decided_by": decided_by, "steps": steps}

    if not gt["pass"]:
        steps.append("gate failed")
        return out(False, "gate")
    base_methods = {inc["method"] if inc["method"] != "rerank" else "hybrid", c["method"] if c["method"] != "rerank" else "hybrid"}
    noises = [method_noise(rows, m) for m in base_methods]
    thr = lambda key, floor: max([floor] + [n.get(key, 0) for n in noises])  # noqa: E731
    levels = [("L1 answer-grade sum", "answer_points", MIN_T, lambda r: r["graded"]["answer_points"] if r["graded"] else None),
              ("L2 CER", "cer", MIN_T, lambda r: r["cer"]["n"]),
              ("L3 supported-citation share", "support_share", MIN_T_SHARE, lambda r: share(r["graded"]))]
    for name, key, floor, getter in levels:
        a, b = getter(c), getter(inc)
        if a is None or b is None:
            steps.append(f"{name}: skipped (no labels)")
            continue
        t = thr(key, floor)
        steps.append(f"{name}: challenger {a:.3g} vs incumbent {b:.3g}, threshold T={t:.3g}")
        if abs(a - b) >= t - 1e-9:
            win = a > b
            steps.append(f"  -> decided at {name}: {'challenger' if win else 'incumbent'} is higher" + ("" if win else " (a significantly worse challenger never wins)"))
            return out(win, name.split()[0])
    r = retrieval_gain(c, inc)
    steps.append(f"clause R (retrieval gain; thresholds fixed after rerank's retrieval numbers were known): {r['why']} -> {'applies' if r['applies'] else 'does not apply'}")
    if r["applies"]:
        return out(True, "R")
    lat_c, lat_i = c["median_latency_s"], inc["median_latency_s"]
    if lat_i and abs(lat_c - lat_i) / lat_i > TIEBREAK_PCT:
        win = lat_c < lat_i
        steps.append(f"tie-break latency: challenger {lat_c}s vs incumbent {lat_i}s (more than {TIEBREAK_PCT:.0%} apart) -> {'challenger' if win else 'incumbent'}")
        return out(win, "latency")
    cost_c, cost_i = c["cost_usd"], inc["cost_usd"]
    win = bool(cost_i) and abs(cost_c - cost_i) / cost_i > TIEBREAK_PCT and cost_c < cost_i
    steps.append(f"tie-break latency within {TIEBREAK_PCT:.0%} ({lat_c}s vs {lat_i}s); cost {cost_c} vs {cost_i} -> "
                 f"{'challenger' if win else 'inconclusive: incumbent retained by default (not a finding about quality)'}")
    return out(win, "cost" if win else "default")


def select(rows: dict[str, dict]) -> dict:
    if BASELINE not in rows:
        return {"error": f"{BASELINE} missing"}
    incumbent, log, warnings = rows[BASELINE], [], []
    for name in CHALLENGERS:
        if name not in rows:
            log.append({"challenger": name, "skipped": "run does not exist"})
            continue
        res = challenge(rows[name], incumbent, rows)
        log.append(res)
        if res["replaces"]:
            incumbent = rows[name]
    present = [n for n in [BASELINE, *CHALLENGERS] if n in rows]
    ungraded = [n for n in present if not rows[n]["graded"]]
    if ungraded:
        warnings.append("runs without labels: " + ", ".join(ungraded))
    involved = {"hybrid"} | {rows[n]["method"] if rows[n]["method"] != "rerank" else "hybrid" for n in present}
    for m in sorted(involved):
        nz = method_noise(rows, m)
        if not nz:
            warnings.append(f"noise pair of {m} (base-nofilter and base-nofilter-rep2) missing: T falls back to its floor")
        elif not nz["graded_pair"]:
            warnings.append(f"noise pair of {m} is not fully graded: the noise on L1/L3 counts as 0 and T falls back to its floor")
    labels = not warnings
    return {"final_incumbent": incumbent["run"], "log": log, "labels_available_for_all_compared_runs": labels, "warnings": warnings,
            "status": "FINAL (every compared run and every noise pair graded)" if labels else
            "PROVISIONAL (" + "; ".join(warnings) + ") - graded levels may be skipped, G3 may use a proxy, noise may be understated"}


def retrieval_table() -> list[dict]:
    """Retrieval-only benchmark before/after the reranker, from results/retrieval/dev-filters-<f>-rerank-<model>/ (dense/bm25/hybrid rows are the same in both)."""
    out = []
    for flt in ("none", "question"):
        for model in ("minilm", "bge"):
            path = RETR / f"dev-filters-{flt}-rerank-{model}" / "summary.json"
            if not path.exists():
                continue
            d = load(path)
            for m, v in d["methods"].items():
                if m != "rerank" and model == "bge":
                    continue  # identical to the minilm file; listed once
                label = m if m != "rerank" else f"rerank ({d['meta']['rerank']['cross_encoder'].split('/')[-1]}, pool {d['meta']['rerank']['pool']})"
                out.append({"filters": flt, "method": label, **{k: v[k] for k in ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "median_ms", "misses_top3")}})
    return out


def fmt_counts(d: dict) -> str:
    return " / ".join(f"{k}:{v}" for k, v in d.items())


def render(rows: list[dict], sel: dict, nz: dict, reading: str, retr: list[dict] | None = None) -> str:
    L = ["# Dev comparison (agent v1b, gpt-4o-mini, 3 searches max, top_k 5, 10000-char budget)\n",
         "Generated by `python src/compare_runs.py` from `results/agent/*/summary.json` and `results/retrieval/`. Single runs, n = 21 answerable + 2 absent: "
         "read differences of 1 question as noise (see the noise rows). No hand grading in this file.\n",
         "Retrieval columns are the retrieval-only benchmark (raw question as query, depth 10) of the method under the same filter setting; CER is the agent's "
         "final context. `abst/ans` = abstentions on the 21 answerable questions (false abstentions); `x01,x02` = status on the two absent questions "
         "(A = answered, AB = abstained).\n"]
    for title, flt in (("Primary: filters none (mimics test)", "none"), ("Secondary: filters = user question filter", "question")):
        L += [f"## {title}\n", "| run | method / prompt | Hit@1 | Hit@3 | MRR@10 | CER | tool use | searches (n:count) | answered / abst / err on 21 | abst/ans ids | x01,x02 | median s | tokens (in+out) | cost USD |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["filters"] != flt:
                continue
            h = r["retrieval_dev"] or {}
            absent = f"{r['absent']['answered']} A / {r['absent']['abstain']} AB"
            L.append(f"| {r['run']} | {r['method']}{'+' + r['reranker'].split('/')[-1] if r['reranker'] else ''} / {r['prompt']}{' + id-repair' if r['repair_ids'] else ''} | {h.get('hit@1', 'n/a')} | {h.get('hit@3', 'n/a')} | {h.get('mrr@10', 'n/a')} | "
                     f"{r['cer']['n']}/{r['cer']['N']} | {r['tool_use']['n']}/{r['tool_use']['N']} | {fmt_counts(r['search_counts'])} | "
                     f"{r['answerable']['answered']} / {r['answerable']['abstain']} / {r['answerable']['error']} | {','.join(r['abstained_on_answerable']) or '-'} | "
                     f"{absent}{' (answered ' + ','.join(r['answered_on_absent']) + ')' if r['answered_on_absent'] else ''} | {r['median_latency_s']} | "
                     f"{r['tokens']['prompt']}+{r['tokens']['completion']} | {r['cost_usd']} |")
        L.append("")
    L += ["## Retrieval-only benchmark before / after the reranker (dev, 21 answerable, raw question as query, depth 10)\n",
          "| filters | method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR@10 | median ms / query | misses at 3 |", "|---|---|---|---|---|---|---|---|---|"]
    for v in retr or []:
        L.append(f"| {v['filters']} | {v['method']} | {v['hit@1']} | {v['hit@3']} | {v['hit@5']} | {v['hit@10']} | {v['mrr@10']} | {v['median_ms']} | {','.join(v['misses_top3']) or '-'} |")
    L += ["", "## Noise (identical baseline run repeated, filters none)\n", "| method | CER run 1 | CER run 2 | median s run 1 / run 2 | answered-on-absent run 1 / run 2 |", "|---|---|---|---|---|"]
    byname = {r["run"]: r for r in rows}
    for m in ("hybrid", "bm25", "dense"):
        a, b = byname.get(f"dev-{m}-base-nofilter"), byname.get(f"dev-{m}-base-nofilter-rep2")
        if a and b:
            L.append(f"| {m} | {a['cer']['n']}/{a['cer']['N']} | {b['cer']['n']}/{b['cer']['N']} | {a['median_latency_s']} / {b['median_latency_s']} | "
                     f"{','.join(a['answered_on_absent']) or '-'} / {','.join(b['answered_on_absent']) or '-'} |")
    L += ["", f"## Preregistered selection rule applied: {sel.get('status', '')}\n", f"Incumbent at the end: **{sel.get('final_incumbent')}**\n"]
    for e in sel.get("log", []):
        if "skipped" in e:
            L.append(f"* {e['challenger']}: {e['skipped']}")
            continue
        gt = e["gates"]
        L.append(f"* {e['challenger']} vs {e['incumbent']}: gates G1={gt['G1_cer']} G2={gt['G2_errors_and_false_abstentions']} G3={gt['G3_absent_abstentions']} ({gt['G3_basis']}) -> "
                 f"{'REPLACES' if e['replaces'] else 'does not replace'} (decided by {e['decided_by']})")
        L += [f"    * {s}" for s in e["steps"]]
    for w in sel.get("warnings", []):
        L.append(f"* WARNING: {w}")
    g = [r for r in rows if r["graded"]]
    if g:
        L += ["", "## Graded metrics (independent labels found in the run folder)\n", "| run | answer points | supported citations | abstention correct | context grade 2 |", "|---|---|---|---|---|"]
        for r in g:
            x = r["graded"]
            L.append(f"| {r['run']} | {x['answer_points']}/{x['answer_points_max']} | {x['supported_citations']}/{x['accepted_citations']} | {x['abstention_correct']}/{x['n_absent']} | {x['context_grade_2']}/{x['context_graded']} |")
    if reading:
        L += ["", reading.strip(), ""]
    return "\n".join(L) + "\n"


def main() -> None:
    folders = [f for f in sorted(AGENT.glob("dev-*")) if (f / "summary.json").exists() and not SKIP.search(f.name)]
    rows = [row(f) for f in folders]
    by = {r["run"]: r for r in rows}
    sel = select(by)
    nz = {m: method_noise(by, m) for m in ("hybrid", "bm25", "dense")}
    reading_path = ROOT / "results" / "dev_comparison_reading.md"
    reading = reading_path.read_text(encoding="utf-8") if reading_path.exists() else ""
    (ROOT / "results" / "dev_comparison.json").write_text(json.dumps({"runs": rows, "selection": sel, "noise": nz, "retrieval_only": retrieval_table()}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / "results" / "DEV_COMPARISON.md").write_text(render(rows, sel, nz, reading, retrieval_table()), encoding="utf-8")
    print(f"{len(rows)} runs -> results/DEV_COMPARISON.md; {sel.get('status')}; incumbent {sel.get('final_incumbent')}")


if __name__ == "__main__":
    main()
