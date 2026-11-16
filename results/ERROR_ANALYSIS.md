# Dev error analysis (agent v1b baseline, written BEFORE any new configuration was run)

Sources: `results/agent/dev-<method>-base[-nofilter]/traces.json` (ranks, chunk ids, answer text), `results/retrieval/dev-filters-none/`
(Hit@k, gold ranks), and the student's manual `reviews_reasons.json` for the three `-base` runs (filters on). Gold chunk ids come from
`data/questions.json` and are used here only for analysis. Primary setting going forward: `--filters none` (no test question has a filter).
`nofilter` agent runs have not been graded; statements about their answers below are my reading of the trace against `expected_answer`, not a grade.

## E1. Retrieval miss without the scope filter: d17 and d18 (dense, filters none) — ROOT CAUSE: ranking

* **d17** "If a person injured by a defective product is found partly responsible but may still collect damages, how is the amount they can recover
  determined?" Gold chunk `para:bizlaw:p0074:02` (the "comparative negligence" passage). Retrieval-only benchmark, filters none: dense rank **10**, hybrid rank 2,
  bm25 rank 1. The question says "partly responsible"; the gold text says "comparative negligence", so the dense embedding ranks the neighbouring
  `p0074:03`, `p0074:00` (a "contributory negligence" passage), `p0072:03` first.
* Agent trace `dev-dense-base-nofilter` d17: search 1 returned `p0074:03, p0074:00, p0075:00, p0073:02, p0073:01` (gold not among them); the model then
  rewrote its own query twice to "contributory negligence ..." (searches 2 and 3 keep returning `p0074:00`, `p0075:00`) and answered that damages are
  "adjusted ... contributory negligence", citing `p0074:00` and `p0075:00`. The expected answer is comparative negligence (shares of fault are compared). CER = miss, answer wrong
  in substance. The drift to "contributory" is a consequence of what the first retrieval showed, not of the model ignoring evidence.
* **d18** "How much advance notice must employers give before closing a plant ... and how large must the employer be?" Gold `para:bizlaw:p0102:00`. Gold rank: dense **8**, hybrid 4,
  bm25 2. In `dev-dense-base-nofilter` d18 the single search returned `bizethics:p0157:01/00/02` (a WARN Act passage without the size threshold), then `bizlaw:p0098:02`,
  `bizlaw:p0100:03`; gold absent from the 5 results and from the context (CER miss). The model answered "sixty days ... size requirement not detailed" and cited `p0157:01`.
  With the gold chunk at hybrid rank 4 or bm25 rank 2, the other methods did get it (`dev-hybrid-base-nofilter`: CER 21/21; the agent's own first query put the d17 and d18 evidence at hybrid rank 4 in both).
* [Phase B correction: E1 is a failure of DENSE retrieval. The selection baseline is hybrid, which has CER 21/21 without the filter (both runs), so rerank repairs a ranking weakness that did not show up as a CER loss in the baseline; its measurable effect is on Hit@1/Hit@3 and rank, not on CER.]
* Pattern: all CER misses without a filter are ranking failures (gold in top-10 in 21/21 for every method, Hit@10 = 21/21) not coverage failures. Without the
  `bizlaw`/`bizethics` scope, passages from the sibling book compete.
* **Hypothesis / fix (config A, rerank):** retrieve a wider pool (hybrid, 30 candidates, all gold are inside rank 10) and reorder with a cross-encoder that reads
  query and passage together, so lexically distant but semantically exact passages (d17 "partly responsible" vs "comparative negligence") and
  precise passages (d18 size threshold) move up. Expected: Hit@1/Hit@3 and dense-like CER rise on filters none; cost: extra latency per search (CPU cross-encoder over 30 passages).

## E2. Failed abstention on an absent question: x02, bm25 (all runs) — ROOT CAUSE: generation/grounding

* x02 "What ethical objections to human reproductive cloning does the philosophy textbook discuss?" The corpus never discusses cloning (`absence_check`).
* `dev-bm25-base` x02: search 1 `phil:p0328:05, p0331:01, p0327:01, p0307:00, p0332:00`; two more searches; the model answered "sanctity of human life, exploitation, autonomy
  and consent, ... biodiversity" citing `p0325:01` (abortion/euthanasia), `p0328:01` (clinical-trial consent) and `p0332:00` (inheritable gene modification). The grader marked the
  abstention failed and all 3 citations unsupported. All ids are valid and in context: the id check cannot catch this.
* bm25 answered x02 in 3 of its 4 base runs (base, rep2, nofilter) and abstained in the fourth (`base-nofilter-rep2`: the failure is unstable, not constant); dense and hybrid abstained in all their runs. [Phase B correction: the first text said "all 3 v1b runs" and left the fourth, abstaining run out.] So the evidence the model sees (lexical matches on "ethical", "human",
  "objections") is on neighbouring topics and gpt-4o-mini treats a neighbouring-topic passage as evidence when the prompt only says "passages on a related topic are not evidence" once.
* **Hypothesis / fix (config B, grounded prompt):** make the topic check explicit and generic (the passage must state the specific subject the question asks about; a neighbouring topic is
  not evidence; answer only what was asked). No topic names from dev are put in the prompt. Expected: bm25 (and any method) abstains on x02, fewer unsupported extras; risk: more false abstentions on answerable questions. [Phase B correction: B was only ever run on hybrid, which already abstains on x02 in every run, so the bm25 failure this fix was aimed at was never tested.]

## E3. Unsupported extra claims on answerable questions: d08 (bm25, hybrid) and d20 (dense, hybrid) — ROOT CAUSE: generation/grounding

* d08 `dev-bm25-base`: gold `para:bizethics:p0195:00` is in the context at rank 3 of search 1 and the core answer ("non-compete unenforceable in California") is right. The answer adds
  "This law reflects the approach that most states may take"; the cited passage only poses this as a question. Grader: capped at 1, citation false under the strict rule.
* d20 `dev-dense-base`: gold `para:phil:p0332:01` at rank 4. The answer adds a utilitarian sentence ("may oppose it if ... costly") citing `p0331:04`, which never mentions cost, and the sentence is
  off-question. Grader: capped at 1.
* In both cases retrieval and context were perfect (context grade 2); the model embellishes. Rule 2 of the v1 prompt already says "do not add background facts", so a one-line reminder is not strong enough.
* **Hypothesis / fix (config B):** state explicitly that a question, an example or a "may/might" in a passage is not a fact, that every sentence (not only the first) must be stated by the passage cited
  right after it, and that statements outside the question must be left out. Expected: fewer grade-1 answers (d08, d20), same CER.

## E4. Citation pointing to the wrong chunk: d01 (hybrid) — ROOT CAUSE: generation (citation placement)

* `dev-hybrid-base` d01: search 1 returned `phil:p0329:05` (rank 1) and the gold `phil:p0291:03` (rank 2). The answer states "treat persons as ends ... including ourselves ... rational beings" and puts `[p0329:05]` after it;
  only `p0291:03` contains "in your own person as well as in the person of any other". The grader: answer correct overall but the first citation does not support "including ourselves".
* Cause: the model attaches one id per paragraph rather than per supported claim. Same fix family as E3 (cite the passage that itself states each claim; cite both if two are needed). A reranker could also
  put `p0291:03` first, but both chunks were in context, so retrieval is not the cause.

## E5. Format/protocol slips (observed, not primary targets)

* `dev-bm25-base-nofilter` d13 cited the shortened id `bizlaw:p0104:03` (missing "para:") although the prompt now defines the id; correct answer, run ended `invalid_citation` (counted as an abstention
  on an answerable question). Cause: model editing ids. Rule 3 wording of config B repeats "copy the id exactly".
* `dev-dense-base-nofilter` d18 produced one answer containing both a cited statement and a trailing line "INSUFFICIENT_EVIDENCE: ...". The loop only treats the marker at the start of the answer as an
  abstention, so this is `answered`; a mixed answer is neither a clean answer nor a clean abstention. [Phase B: the frozen harness (agent v1c-frozen) treats a marker anywhere in the answer as an abstention with reason `mixed_marker`; the existing run folders keep their old status, d18 of `dev-dense-base-nofilter` is the only affected case.]

## Summary of causes and fixes

| id | case | cause | fix tested |
|---|---|---|---|
| E1 | d17, d18 (dense, no filter) | ranking (gold at rank 10 / 8, in top 10) | A: hybrid pool 30 + cross-encoder rerank |
| E2 | x02 (bm25) | grounding: neighbouring-topic passages treated as evidence | B: grounded prompt |
| E3 | d08, d20 | grounding: unsupported extra sentences | B: grounded prompt |
| E4 | d01 (hybrid) | citation placement | B: grounded prompt (per-claim citation) |
| E5 | d13, d18 | id editing / mixed marker | noted only |

## Phase B corrections (2026-11-16, after an independent review)

* E2: bm25 answered x02 in 3 of 4 base runs, not in all of them (see the inline note); B was only tested on hybrid, which already abstains on x02, so the claim "B should make bm25 abstain" was never tested. Whether B helps x02 on bm25 is unknown.
* E1: a dense-only failure; the baseline in the selection rule is hybrid (CER 21/21 without filter).
* E5: shortened ids are a cost that the grounded prompt increased (B alone: 4 abstentions on answerable questions; B+C: 3 repairs nofilter, 8 filters=question; A+B+C: 2 and 4; base prompt on hybrid: 0), and C hides it rather than removing it.
* d08 under B+C: the "2017 California law" sentence is TRUE and stated in `para:bizethics:p0194:03` (in context); it is cited to `p0195:04` (nonsolicitation), so it is a mis-citation, not a fabricated claim. It is still off-question, so E3 persists under B+C (E4-style mis-citation).
* E5 mixed marker: handled by the harness from v1c-frozen on (see above).
