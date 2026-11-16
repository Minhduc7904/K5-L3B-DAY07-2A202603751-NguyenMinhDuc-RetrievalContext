"""Export Day 6 data (pgvector chunks + benchmark) to the Day 7 JSON contract.

Chunks: strategy 'B' rows of the Day 6 `chunks` table (662 chunks, structure-aware chunking).
Questions: Day 6 benchmark/questions.json; `alt_valid` entries become extra `gold` alternatives.
Vectors are NOT exported: the dense retriever queries the live Day 6 pgvector store.

Usage (from the Day 7 repo root):
    python src/export_day6.py --day6 <path-to-Day6-repo> --output data
DATABASE_URL is read from the environment, else from <day6>/starter/.env.
"""
import argparse
import json
import os
from pathlib import Path

import psycopg
from dotenv import dotenv_values

STRATEGY = "B"


def export_chunks(url: str) -> list[dict]:
    sql = """SELECT chunk_id, document_id, source, page, chapter, section, text, metadata, model
             FROM chunks WHERE strategy = %s ORDER BY document_id, page, chunk_id"""
    with psycopg.connect(url, connect_timeout=5) as conn, conn.cursor() as cur:
        cur.execute(sql, (STRATEGY,))
        rows = cur.fetchall()
    chunks = []
    for chunk_id, doc, source, page, chapter, section, text, meta, model in rows:
        metadata = dict(meta or {})
        metadata.update({"chapter": chapter, "section": section, "strategy": STRATEGY,
                         "embedding_model": model, "document_id": doc})
        chunks.append({"id": chunk_id, "source_id": doc, "source": source, "page": page,
                       "text": text, "metadata": metadata})
    return chunks


_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _canon_with_map(text: str) -> tuple[str, list[int]]:
    """Day 6 canonical form (quote/dash style, whitespace, case) plus canon-index -> original-index."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # also strips leading whitespace
    for i, ch in enumerate(text):
        if ch == "\u00ad":
            continue
        if ch.isspace():
            if not prev_space:
                out.append(" ")
                idx.append(i)
            prev_space = True
            continue
        prev_space = False
        for c in ch.translate(_QUOTES).lower():
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def align_evidence(chunks: list[dict], source_id: str, page: int, evidence: str) -> str:
    """Return the literal span of a chunk on (source, page) that matches the Day 6 quote.

    Day 6 matched quotes after normalising quote style/whitespace/case; Day 7 `hit` needs an exact
    substring, so we replace the quote by the literal text found in the chunk.
    """
    target, _ = _canon_with_map(evidence)
    for c in chunks:
        if c["source_id"] != source_id or c["page"] != page:
            continue
        canon_text, idx = _canon_with_map(c["text"])
        pos = canon_text.find(target.rstrip())
        if pos >= 0:
            end = pos + len(target.rstrip()) - 1
            return c["text"][idx[pos]: idx[end] + 1]
    raise SystemExit(f"evidence not found inside one chunk: {source_id} p{page}: {evidence[:60]!r}")


def convert_question(q: dict, chunks: list[dict]) -> dict:
    gold = [{"source_id": q["gold_source_id"], "page": q["gold_page"], "evidence": q["evidence_quote"]}]
    for alt in q.get("alt_valid") or []:
        gold.append({"source_id": alt["source_id"], "page": alt["page"], "evidence": alt["quote"]})
    for g in gold:
        g["evidence_original"] = g["evidence"]
        g["evidence"] = align_evidence(chunks, g["source_id"], g["page"], g["evidence"])
    out = {"id": q["id"], "split": q["split"], "question": q["question"],
           "expected_answer": q["expected_answer"], "gold": gold}
    if q.get("filter"):
        out["filter"] = q["filter"]
    # Analysis-only fields; never pass them to the agent or retriever.
    out["type"] = q.get("type")
    out["notes"] = q.get("notes", "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day6", required=True, type=Path, help="root of the Day 6 repo")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    url = os.getenv("DATABASE_URL") or dotenv_values(args.day6 / "starter" / ".env").get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL missing (env var or <day6>/starter/.env)")
    chunks = export_chunks(url)
    ids = [c["id"] for c in chunks]
    if not chunks or len(set(ids)) != len(ids):
        raise SystemExit(f"bad export: {len(chunks)} chunks, {len(set(ids))} unique ids")
    raw = json.loads((args.day6 / "benchmark" / "questions.json").read_text(encoding="utf-8"))
    questions = []
    next_dev = 1 + max(int(q["id"][1:]) for q in raw if q["split"] == "dev")
    for q in raw:
        item = convert_question(q, chunks)
        if q["split"] == "test":
            # The Day 6 test split was already evaluated once in Day 6 and discussed in its report, so it is
            # no longer an independent test: reuse it as extra dev data, keep the Day 6 id for traceability.
            item.update(id=f"d{next_dev}", split="dev", day6_id=q["id"], day6_split="test")
            next_dev += 1
        questions.append(item)
    authored_path = args.output / "day7_authored.json"  # fresh test questions written for Day 7
    if authored_path.exists():
        questions += [convert_question(q, chunks) for q in json.loads(authored_path.read_text(encoding="utf-8"))]
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate question ids")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("chunks.json", chunks), ("questions.json", questions)):
        (args.output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Model-facing view: only what a real user would type. Agent/retriever code must read THIS file,
    # never questions.json / absent_questions.json (gold, expected answers, notes, absence reasons).
    absent_path = args.output / "absent_questions.json"
    absent = json.loads(absent_path.read_text(encoding="utf-8")) if absent_path.exists() else []
    public = [{k: q[k] for k in ("id", "split", "question", "filter") if k in q} for q in questions + absent]
    (args.output / "questions_public.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"chunks={len(chunks)} questions={len(questions)} absent={len(absent)} -> {args.output}")


if __name__ == "__main__":
    main()
