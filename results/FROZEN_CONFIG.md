# Frozen configuration for the two test runs

**Rules and chronology of the freeze.** Exactly TWO test runs were allowed (the CLI refuses a third `results/agent/test-*` folder, whatever the method): the baseline and the chosen configuration, run once each by the student. At the moment this file was first saved, the test split had not been run, scored or opened and no `results/agent/test-*` folder existed. The two frozen runs were subsequently completed and are recorded in section 6. No configuration, prompt, agent code, data, threshold or selection rule was changed between the freeze and those runs.

## 1. Date and who froze it
* Frozen on 2026-11-16 by student Nguyen Minh Duc. Time of day: see the file time of this file.
* Test split read or run before this file was complete: **no**.

## 2. Code and data fingerprints (SHA-256, from `python src/freeze_info.py`)
| file | SHA-256 |
|---|---|
| `src/agent.py` (AGENT_VERSION `v1c-frozen`) | `d27b6c595ca81fb9a550e121bdbb6596c081a5c4a61ef85efd671da503f784ac` |
| `src/run_agent_eval.py` | `df12a7fe9bc0172175ec3fe9c09be931322058022fe4fb63b46d70b8bf31bf62` |
| `src/retrieval.py` | `ef429958650dcc510d9e53ff160ddd73d4f95d6ab9a5c5762f5fd989764840ff` |
| `src/compare_runs.py` (the version that produced section 5) | `b5bfdc322f463da368ccbcf0c4e43edc15b1a6907f319eba8e72e535f68f01e9` |
| `results/SELECTION.md` (submission copy; wording updated after the runs, rules/results unchanged) | `013b19cc5ac64956cde0970f3993f21462ba43fa6cfca1ce8a416cdfdac9cd42` |
| `results/dev_comparison.json` | `722f39c585f4434db8af5943b634005f2ae2993e0b72f0cf64c7d7d0032517c8` |
| `results/DEV_COMPARISON.md` (submission copy; attribution wording updated) | `ee639616b4909e41231b228f92b855e19987211be3511f208cd8606bc84157c4` |
| `data/chunks.json` (662 chunks) | `8e3d3d2d2b536ebd79a466c34faac218d83df00570382c1f02de33e53b1d92bc` |
| `data/questions_public.json` (the only question file the agent reads; 7 test questions, none with a filter) | `82318c71f62085cbed25d2f18191cb23c6baf7b7afb83d56db1b5f6aa7bb9f43` |
| `data/questions.json` (gold; read only by the evaluator after the run) | `2719cdcd2a581d3cc2146fcd7d247c5b5596f89e786676b8e4fc046ca862aec4` |
| `data/absent_questions.json` | `502a7f44fc83ff2a8a03bd85e492c3b37a55ebdd99d51410f631857ed3518bda` |
* At freeze time, git HEAD was `c67ec6292cd6d15e6e6cbaa2c7e7260398bb5a19` and the lab files had not yet been committed. The completed submission is now committed on `main`; the submission commit is the current git HEAD. The frozen code/data identity is established by the SHA-256 values above and by the matching `config.agent_code_sha256`, `system_prompt_sha256`, `chunks_sha256` and `questions_public_sha256` stored in both test `summary.json` files.
* Unit tests (`python -m unittest discover -s tests`): 69 tests, OK (rerun at freeze time, see section 9 if it differs).
* Packages: python 3.12.6, openai 3.46.0, psycopg 3.3.8, sentence-transformers 6.2.0, python-dotenv 1.2.4, torch 2.15.0. Dense/hybrid/rerank need the Day 6 pgvector container (`day6_student_starter-postgres-1`, image `pgvector/pgvector:0.8.7-pg16`, 127.0.0.1:55433) running and healthy; it was up when this file was written.

## 3. Prompt and model
* Model: `gpt-4o-mini` (from `OPENAI_MODEL` in `.env` or the default); the snapshot seen in all dev summaries: `gpt-4o-mini-2024-07-18`. The snapshot of each test run is recorded in `summary.json` (`model_snapshots_seen`); if it differs from this one, say so next to the results. temperature 0; max completion tokens 600; `parallel_tool_calls` false; API attempts 4 (backoff 2 s); no `seed` parameter (the model API is not deterministic at temperature 0).
* Prompt text SHA-256: base `3c00e31013fc8ff1d503d2b0a9844f8c08dd79528e249d9edd8ef3e5f7dc6f96`, grounded `641bd4b6f274732da108d996d225ac00209d1957b5080fb9361b1ffe0f7f03ee`. **Both test runs use the base prompt** (no `--prompt` flag). The full text is stored in each `summary.json` (`config.system_prompt`).
* Price table (`PRICES` in `agent.py`): gpt-4o-mini 0.15 / 0.60 USD per 1M input / output tokens. These rates were verified after the frozen runs against the official OpenAI model page; cached-input discounts are ignored.

## 4. Fixed agent and retrieval parameters (both test runs)
* `top_k` 5, context budget 10000 characters over the whole run, `max_searches` 3, dedup by chunk id, a chunk is never truncated, filters `none` (the 7 test questions carry no user filter, so this equals the `questions_public.json` filters), `--max-cost 0.50` per run (the estimated spend of a 7-question run is about USD 0.003).
* Retrieval: hybrid = RRF (k 60, 50 candidates per side) of dense (`BAAI/bge-base-en-v1.5` over the Day 6 pgvector store, strategy B, 662 chunks) and BM25 (k1 1.5, b 0.75). Chosen configuration: `rerank` = the same hybrid pool of 30 candidates re-ordered by the local cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`, top 5 returned.
* Harness behaviour, identical in both runs: no `--repair-ids` (the chosen configuration does not use id repair, so nothing to apply to both); an answer containing `INSUFFICIENT_EVIDENCE` anywhere is an abstention (`mixed_marker`); answers need valid citations in the final context.

## 5. Selection decision (from `python src/compare_runs.py` on the student's manual labels; details in `results/SELECTION.md`, "Results of applying the rule")
* Status printed by the script: **PROVISIONAL**. Warnings, verbatim: "noise pair of bm25 is not fully graded", "noise pair of dense is not fully graded", "noise pair of hybrid is not fully graded" (the `-rep2` repeats of the three base-nofilter runs have no labels, so T sits at its floors; it cannot change a verdict, see SELECTION.md).
* Baseline B0: `dev-hybrid-base-nofilter` (hybrid, base prompt). Rule output: final incumbent **`dev-rerank-minilm-nofilter`** (A: rerank MiniLM pool 30, base prompt, top 5).
* **The replacement of the baseline by rerank was decided by clause R** (retrieval gain), not by L1-L3, and R's thresholds (Hit@1 +3 questions, <= +40% latency) were proposed after the reranker's retrieval numbers were known: it is NOT a blind preregistration and may be fitted to rerank. The replacement is evidence of a retrieval gain with no measured agent-level loss, not of better answers.

| run (dev, filters none) | answer sum /42 | supported / accepted citations (share) | CER /21 | abstention correct /2 | Hit@1 | Hit@3 | MRR@10 | median s | cost USD | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| hybrid base (B0) | 41 | 20/23 (0.870) | 21 | 2 | 16 | 20 | 0.861 | 3.34 | 0.00844 | start incumbent |
| bm25 base | 39 | 17/23 | 21 | 1 | 18 | 20 | 0.914 | 3.54 | 0.00957 | gate G3 fails |
| dense base | 37 | 19/24 | 19 | 2 | 15 | 18 | 0.798 | 4.14 | 0.00936 | gate G1 fails |
| A rerank (chosen) | 42 | 20/23 (0.870) | 21 | 2 | 19 | 21 | 0.944 | 4.46 | 0.0085 | replaces B0, decided by R |
| B grounded | 31 | 17/21 | 21 | 2 | 16 | 20 | 0.861 | 3.08 | 0.0093 | gate G2 fails (4 false abstentions) |
| C repair | 41 | 21/23 (0.913) | 21 | 2 | 16 | 20 | 0.861 | 3.69 | 0.00841 | retained by default (inconclusive) |
| B+C | 42 | 21/27 (0.778) | 21 | 2 | 16 | 20 | 0.861 | 3.49 | 0.00938 | retained by default (inconclusive) |
| A+B+C | 42 | 23/27 (0.852) | 21 | 2 | 19 | 21 | 0.944 | 4.37 | 0.00909 | retained by default (inconclusive) |

* Noise used: the unlabelled repeats give CER noise 0 (hybrid), 0 (bm25), 2 (dense); graded noise unknown (counted 0); thresholds T1 = 2 points, T2 = 2 questions, T3 = 0.10 share; latency of the two hybrid baselines 3.34 / 3.54 s.
* Deviations from the preregistered rule (all disclosed in SELECTION.md with times of day): amendment 1 and 2 (noise per method involved; reranker choice), amendment 3 (variant C and its challengers added after seeing B), amendment 4 (tie-break 25%, L3 as a share, clause R, the first deciding level decides in either direction, FINAL needs graded noise pairs, global test cap), `compare_runs.py` rewritten after the results existed. The rule's own output is the chosen configuration above; the only element that is not "rule output in the blind sense" is clause R. If clause R is rejected, the rule's output is "no challenger beats the baseline" and the baseline alone would be the incumbent.
* Limits that apply: dev saturated (CER 21/21, answers 41-42 of 42), single runs of 21 + 2 questions, and ONE student manual grader with no second grader.

## 6. Completed frozen test runs
Both commands used `--split test --final`, filters none and the defaults of section 4. They were run once each after the freeze and produced `results/agent/test-hybrid-base` and `results/agent/test-rerank-minilm`. Both summaries record `agent_version = v1c-frozen`, model snapshot `gpt-4o-mini-2024-07-18`, and matching frozen code/data hashes.

(a) Baseline B0 (hybrid, base prompt, no user filters because no test question has one):
```
python src/run_agent_eval.py --split test --final --method hybrid --tag base --filters none --max-cost 0.50
```
(b) Chosen configuration A (rerank, MiniLM, pool 30, base prompt, no id repair):
```
python src/run_agent_eval.py --split test --final --method rerank --reranker minilm --pool 30 --tag minilm --filters none --max-cost 0.50
```
* Outcomes: both runs answered 5/5 answerable questions, correctly abstained on 2/2 absent questions, had 0 provider/retriever errors and CER 5/5. Estimated cost was USD 0.00284 for hybrid and USD 0.00279 for rerank. Manual grading gave answer score 10/10 for each; supported citations were 4/6 for hybrid and 5/6 for rerank. The runs were not repeated and no third test run was made.

## 7. What is compared after the runs
Graded manually by the student before the two are compared: answer-grade sum, supported-citation share, abstention correctness (if absent test questions exist), context grade; plus CER, latency and cost, for baseline and chosen configuration. The 7-question test set cannot support a significance claim; no further configuration is added or tuned after seeing test numbers.

## 8. Known limits to state next to any test result
Single runs; model API not deterministic at temperature 0; very small test split (7 questions); dev selection rule weakly discriminating (amendment 4 limitation); the chosen configuration was chosen through clause R, fitted after the fact; labels manually reviewed by one student with no second grader; dev runs of A were made with agent code v1b and the test runs use v1c-frozen (the differences do not matter without `--repair-ids` apart from the `mixed_marker` abstention rule, which applies to both runs).

## 9. Notes made after freezing (append only; never edit the sections above)
* 2026-11-16: The gpt-4o-mini prices used by the frozen code (USD 0.15 / 0.60 per 1M input / output tokens) were checked against the official OpenAI model page and match. Historical `summary.json` files retain `verified_against_openai: false` because that field records the state at run time. Source: https://developers.openai.com/api/docs/models/gpt-4o-mini.
* 2026-11-16: Submission documentation was reconciled after both test runs. The frozen agent, prompt and data hashes did not change. `SELECTION.md` and `DEV_COMPARISON.md` received attribution/clarity edits, so their submission hashes in section 2 supersede the original documentation-only hashes. The test folders and outcomes are now recorded in section 6 instead of being described as future work.
