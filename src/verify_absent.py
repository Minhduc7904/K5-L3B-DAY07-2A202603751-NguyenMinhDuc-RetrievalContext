"""Check that every `absence_check` pattern of data/absent_questions.json matches nothing.

Text is normalised first (soft hyphens removed, every whitespace run -> one space, so a line break
inside a phrase cannot hide it). Patterns are case-insensitive and may span sentences (DOTALL).

Scans (1) the indexed chunks and (2) the full PDFs given by --pdf-dir, so an absence claim does not
rest on one retriever's top-k. Without --pdf-dir only the chunks are scanned: this is reported loudly
and is not enough to claim the whole book lacks the answer (pass --chunks-only to accept that).

`decoy_check` lists near-miss patterns that are EXPECTED to match specific chunks (documented decoys):
the script fails if they match other chunks or no longer match the expected ones.
Exit code 1 on any problem.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PDFS = {"Introduction_to_Philosophy-WEB.pdf": 421, "business-ethics_-_WEB.pdf": 355,
        "Business_Law_I_Essentials_2e_-_WEB.pdf": 164}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("­", ""))


def compile_pattern(pat: str) -> re.Pattern:
    return re.compile(r"(?<![A-Za-z])(?:" + pat + ")", re.I | re.S)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/chunks.json", type=Path)
    ap.add_argument("--absent", default="data/absent_questions.json", type=Path)
    ap.add_argument("--pdf-dir", type=Path, help="folder with the 3 Day 6 PDFs (whole-book check)")
    ap.add_argument("--chunks-only", action="store_true", help="accept a chunk-only scan")
    args = ap.parse_args()
    if not args.pdf_dir and not args.chunks_only:
        sys.exit("refusing to run: pass --pdf-dir for a whole-book check, or --chunks-only to scan chunks only")
    cases = json.loads(args.absent.read_text(encoding="utf-8"))
    texts = {f"chunk:{c['id']}": norm(c["text"]) for c in json.loads(args.chunks.read_text(encoding="utf-8"))}
    problems = 0
    if args.pdf_dir:
        import fitz  # PyMuPDF
        for name, expected_pages in PDFS.items():
            doc = fitz.open(args.pdf_dir / name)
            nonempty = 0
            for i, page in enumerate(doc, 1):
                t = norm(page.get_text())
                nonempty += bool(t.strip())
                texts[f"{name}:p{i}"] = t
            if len(doc) != expected_pages or nonempty < len(doc) - 10:
                print(f"PROBLEM {name}: pages={len(doc)} (expected {expected_pages}), non-empty={nonempty}")
                problems += 1
    else:
        print("WARNING: chunk-only scan; the PDFs outside the indexed page ranges were NOT checked")
    for case in cases:
        for pat in case["absence_check"]:
            hits = [k for k, t in texts.items() if compile_pattern(pat).search(t)]
            problems += bool(hits)
            print(f"{case['id']} {'ok' if not hits else 'FOUND':5} {pat!r} hits={hits[:5]}")
        for decoy in case.get("decoy_check", []):
            hits = sorted(k for k, t in texts.items() if k.startswith("chunk:") and compile_pattern(decoy["pattern"]).search(t))
            expected = sorted(f"chunk:{i}" for i in decoy["expected_chunks"])
            good = hits == expected
            problems += not good
            print(f"{case['id']} {'decoy' if good else 'DECOY-MISMATCH':5} {decoy['pattern']!r} hits={hits} expected={expected}")
    print(f"scanned={len(texts)} texts; problems={problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
