# Code

| File | Purpose |
|---|---|
| `export_day6.py` | Exports the Day 6 chunks (strategy B, 662 rows) to `data/chunks.json` |
| `retrieval.py` | Four retrievers: `dense` (Day 6 pgvector + bge-base-en-v1.5), `bm25`, `hybrid` (RRF of the two), `rerank` (hybrid pool of 30 re-ordered by a local cross-encoder) |
| `eval_retrieval.py` | Retrieval-only benchmark (Hit@k, MRR) written to `results/retrieval/` |
| `agent.py` | Real-model agent loop (OpenAI function calling, one `search` tool), citation check, trace, cost table |
| `run_agent_eval.py` | CLI that runs the agent over a split and writes traces, summary and a blank manual review sheet |
| `compare_runs.py` | Collects all dev runs into `results/DEV_COMPARISON.md` / `dev_comparison.json` and applies the preregistered rule of `results/SELECTION.md` |
| `freeze_info.py` | Prints file/prompt hashes, versions and git head for `results/FROZEN_CONFIG.md` (no API call) |

## How to run the agent

Prerequisites: Python 3.12 with the packages `openai`, `psycopg`, `python-dotenv` and `sentence-transformers`,
`.env` copied from `.env.example` (needs `OPENAI_API_KEY`; `OPENAI_MODEL` defaults to `gpt-4o-mini`), and, for `dense` and
`hybrid`, the Day 6 pgvector container (`docker compose -p day6_student_starter up -d --wait` in the Day 6 `starter/` folder).
`bm25` needs neither Docker nor the embedding model.

```bash
# dev split, one config tag per experiment; results go to results/agent/<split>-<method>-<tag>/
python src/run_agent_eval.py --split dev --method dense  --tag base
python src/run_agent_eval.py --split dev --method bm25   --tag base
python src/run_agent_eval.py --split dev --method hybrid --tag base
python src/run_agent_eval.py --split dev --method hybrid --tag base-nofilter --filters none   # same, ignoring the user filters
# options: --top-k 5 --budget 10000 (characters) --max-searches 3 --filters question|none --only d01,x01 --max-cost 1.5 --overwrite
```

Defaults (see the docstring of `agent.py` for the reasons): `top_k=5` per search, context budget `10000` characters summed
over the whole run (chunks are never truncated, duplicates are shown once, a chunk that does not fit is skipped and logged),
at most `3` searches then one forced answer turn, temperature 0. The user-supplied filter in `data/questions_public.json` is
passed to the retriever by the harness (`--filters question`); the model only writes the query text. Only 5 of the 21 dev
answerable questions (d05, d06, d13, d18, d21) carry such a user-supplied filter, and no absent or test question does, so dev
results with filters are optimistic for those 5: report `base` (filters) and `base-nofilter` side by side.

Each run writes:

* `traces.json`: one object per question with every tool call (query, filter, ranked results with score and page, ids kept in
  context, dropped ids), the final context, answer, citations, status (`answered` / `abstain` / `error`), abstain reason, number of
  searches, wall time, tokens and estimated cost. Evidence checks (`evidence_in_context`, ...) are added by the evaluator only.
* `summary.json`: automatic metrics (Context Evidence Rate, tool-use rate, search-count distribution, abstentions and their reasons,
  errors, citation-id validity counted both on accepted citations and on all attempted ones, latency, tokens, estimated cost) and the
  full configuration (real controller settings, model snapshot, package versions, system prompt, and the price constants as recorded at run time).
  Context Evidence Rate is a context metric, not answer quality. A run is a single sample on 21 + 2 questions: differences between
  methods are not significant.
* `traces.partial.jsonl` is written after every question during a run (crash safety) and removed when the run completes.
* `cases.json` and `reviews_template.json`: inputs for `starter/agent_metrics.py`. The review sheet is **blank on purpose**:
  `context_grade` (0/1/2, BENCHMARK.md section 2), `citation_support`, `answer_grade` and `abstention_correct` must be filled in by a human who reads each cited passage. No script and no
  LLM grades answers or citations. Then aggregate with
  `python starter/agent_metrics.py --cases <dir>/cases.json --traces <dir>/traces.json --reviews <dir>/reviews.json --split dev --output <dir>/agent_metrics.json`.

Cost is estimated from usage tokens with the `PRICES` table in `agent.py`; its gpt-4o-mini rates were verified against the official OpenAI model page after the frozen runs (the historical summaries retain their original `UNVERIFIED` metadata). A run stops
if its estimated spend exceeds `--max-cost` (default USD 1.50). Transient API errors (connection, timeout, 429, 5xx) are retried up
to 4 times; any remaining provider/retriever failure is recorded as `status: "error"` (excluded from quality metrics and reported
separately) and the run continues. Model-side format failures (unknown tool, malformed arguments, empty query, truncated answer)
are not errors: they end as `abstain` (reason `model_format_error` / `truncated_answer`) and count against the agent. An answer whose
citations are not all in the final context is an `abstain` with reason `invalid_citation`; the ids it tried to cite are kept in
`attempted_citations` / `invalid_citations`. Nothing is faked.

The `test` split is refused unless `--final` is passed, the output directory is the default one and new, `--only` is not used, and
fewer than 2 `results/agent/test-*` folders exist IN TOTAL (global cap, whatever the method: the plan is exactly two hand-run test
runs, the baseline and the configuration chosen on dev; a crashed test run also uses up one of the two slots). Dev results are not overwritten without
`--overwrite`. `results/agent/dev-*-base-v0` are the pre-review runs (agent v0), kept as evidence; see their `_note`.

## Configurations and how the flags map to them

Primary comparison setting is `--filters none` (no test question has a user filter); `--filters question` is secondary. Reference
baseline B0 = `--method hybrid` with the base prompt. The rule used to choose between configurations is in `results/SELECTION.md`
(fixed before the new runs), the error analysis that motivated them is in `results/ERROR_ANALYSIS.md`, the table is `results/DEV_COMPARISON.md`.

| Config | Flags (all with `--split dev`) | Output folder |
|---|---|---|
| baseline per method | `--method dense\|bm25\|hybrid --tag base-nofilter --filters none` | `dev-<method>-base-nofilter` |
| baseline repeat (noise) | same with `--tag base-nofilter-rep2` | `dev-<method>-base-nofilter-rep2` |
| A. rerank | `--method rerank --reranker minilm --pool 30 --tag minilm-nofilter --filters none` | `dev-rerank-minilm-nofilter` |
| B. grounded prompt (as specified) | `--method hybrid --prompt grounded --tag grounded-nofilter --filters none` | `dev-hybrid-grounded-nofilter` |
| C. id repair (SELECTION amendment 3) | `--method hybrid --repair-ids --tag repair-nofilter --filters none` | `dev-hybrid-repair-nofilter` |
| B+C | `--method hybrid --prompt grounded --repair-ids --tag grounded-repair-nofilter --filters none` | `dev-hybrid-grounded-repair-nofilter` |
| A+B+C | `--method rerank --reranker minilm --prompt grounded --repair-ids --tag minilm-grounded-repair-nofilter --filters none` | `dev-rerank-minilm-grounded-repair-nofilter` |
| secondary (filters = question) | A: `--tag minilm`; C: `--tag repair`; B+C: `--tag grounded-repair`; A+B+C: `--tag minilm-grounded-repair` (same flags, without `--filters none`) | `dev-rerank-minilm`, `dev-hybrid-repair`, `dev-hybrid-grounded-repair`, `dev-rerank-minilm-grounded-repair` |

Not run: B alone (without `--repair-ids`) with filters = question, and the dropped A+B combination without repair; the A+B+C row replaces the
latter (SELECTION.md amendment 3). Runs `dev-*-base-v0` / `-v1a` are earlier harness versions kept as evidence and skipped by `compare_runs.py`.

`--reranker` is `minilm` (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~88 MB, ~1 s per 30-passage pool on CPU) or `bge`
(`BAAI/bge-reranker-base`, ~1.1 GB, ~7 s per pool); both download once from the Hugging Face hub into its cache. Retrieval-only benchmark of the
four methods (Hit@1/3/5, MRR@10, latency): `python src/eval_retrieval.py --split dev --filters none --methods dense,bm25,hybrid,rerank --reranker minilm --output results/retrieval/<new-folder>`.
After independent labels (`reviews.json`) are placed in a run folder, `python src/compare_runs.py` recomputes the table and the selection with the graded levels.

## Frozen harness (agent v1c-frozen)

What is fixed for any run that is quoted as a final result (the freeze record is `results/FROZEN_CONFIG.md`; `python src/freeze_info.py` prints the hashes, no API call):

* Deterministic parts: chunk order (`data/chunks.json`), BM25 and dense scoring, RRF fusion (ties broken by chunk id), rerank (sorted by (-score, id)), the context
  (rank order, dedup by id, a chunk is never truncated), the citation check, the abstention rules, the output files. No random number is used anywhere in the harness.
* Not deterministic: the model API. Temperature 0 does not guarantee identical output (the repeated baseline runs differ by one or two questions). No `seed` parameter is
  sent (its effect could not be checked without a paid call), so any single run is one sample.
* Changes made in v1c-frozen with respect to the runs already in `results/agent/dev-*` (v1b): (1) the id repair of variant C only restores the `para:` prefix
  (`X` -> `para:X`, only if `para:X` is a shown id; before, any unique suffix match was accepted; all 17 past repairs were `para:` repairs, so no past outcome changes);
  `attempted_citations` holds the ids after repair and the new `attempted_citations_raw` the ids exactly as the model wrote them (summary: `attempted_ids_raw`);
  (2) an answer that contains `INSUFFICIENT_EVIDENCE` anywhere but at the start is an abstain with reason `mixed_marker` (before it was `answered` when it had valid
  citations; the only past case is d18 of `dev-dense-base-nofilter`). **Existing run folders keep their old status**: they were not regenerated, `dev-dense-base-nofilter` d18
  is still `answered` there. Both changes apply identically to every configuration, baseline and chosen alike, on the test split.

Unit tests (offline, no key, no Docker): `python -m unittest discover -s tests -v`.
