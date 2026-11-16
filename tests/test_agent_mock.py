"""Offline tests of src/agent.py and src/run_agent_eval.py with fake controllers, retrievers and a fake OpenAI client.

No network, no API key, no Docker. They check the loop's behaviour, not model quality.
"""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import agent  # noqa: E402
import run_agent_eval as harness  # noqa: E402
from agent import OpenAIController, run_agent  # noqa: E402

C1 = {"id": "para:doc:p0001:00", "text": "Refunds take 7 days.", "source_id": "doc", "source": "doc.pdf", "page": 1, "score": 0.9, "rank": 1, "metadata": {}}
C2 = {"id": "para:doc:p0002:00", "text": "Shipping is free.", "source_id": "doc", "source": "doc.pdf", "page": 2, "score": 0.8, "rank": 2, "metadata": {}}
ABSTAIN = {"action": "answer", "answer": "INSUFFICIENT_EVIDENCE: x", "citations": []}


class FakeController:
    def __init__(self, actions):
        self.actions, self.calls = list(actions), []

    def decide(self, question, history, *, allow_search=True):
        self.calls.append((question, [dict(h) for h in history], allow_search))
        return self.actions.pop(0) if len(self.actions) > 1 else self.actions[0]


class FakeRetriever:
    def __init__(self, results):
        self.results, self.calls = results, []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        return self.results[:top_k]


SEARCH = {"action": "search", "query": "refund time"}


def answer(*ids, text="7 days"):
    return {"action": "answer", "answer": text + "".join(f" [{i}]" for i in ids), "citations": list(ids)}


class LoopTests(unittest.TestCase):
    def test_valid_citation_is_answered_and_context_recorded(self):
        ctl = FakeController([SEARCH, {**answer(C1["id"]), "usage": {"prompt_tokens": 10, "completion_tokens": 3, "latency_s": 0.1, "retries": 0, "model": "m"}}])
        ret = FakeRetriever([C1, C2])
        r = run_agent("Refund?", ctl, ret, filters={"document_id": "doc"}, top_k=2, budget=1000)
        self.assertEqual(r["status"], "answered")
        self.assertEqual(r["citations"], [C1["id"]])
        self.assertEqual(r["attempted_citations"], [C1["id"]])
        self.assertEqual(ret.calls, [("refund time", {"document_id": "doc"}, 2)])
        self.assertEqual(r["context_ids"], [C1["id"], C2["id"]])
        self.assertIn("Refunds take 7 days.", r["context"])
        self.assertEqual(r["n_searches"], 1)
        self.assertEqual(r["usage"]["prompt_tokens"], 10)
        search = [e for e in r["events"] if e["event"] == "search"][0]
        self.assertEqual(search["results"][0]["rank"], 1)
        self.assertEqual(search["filter"], {"document_id": "doc"})

    def test_citation_not_in_context_abstains(self):
        r = run_agent("Refund?", FakeController([SEARCH, answer("para:doc:p9999:00")]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"], r["citations"]), ("abstain", "invalid_citation", []))

    def test_citation_missing_from_text_or_absent_abstains(self):
        r = run_agent("q", FakeController([SEARCH, {"action": "answer", "answer": "7 days", "citations": [C1["id"]]}]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"]), ("abstain", "invalid_citation"))
        r = run_agent("q", FakeController([SEARCH, {"action": "answer", "answer": "7 days", "citations": []}]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"]), ("abstain", "no_citation"))

    def test_invalid_citation_keeps_attempted_and_invalid_ids(self):
        fake = "para:book:p0001:00"  # looks like an id, is not in the context
        r = run_agent("q", FakeController([SEARCH, answer(fake, C1["id"])]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"], r["citations"]), ("abstain", "invalid_citation", []))
        self.assertEqual(r["attempted_citations"], [fake, C1["id"]])
        self.assertEqual(r["invalid_citations"], [fake])
        self.assertIn(fake, r["raw_answer"])

    def test_grouped_citations_are_accepted(self):
        ans = f"7 days and free shipping [{C1['id']}, {C2['id']}]"
        r = run_agent("q", FakeController([SEARCH, {"action": "answer", "answer": ans, "citations": agent.extract_citations(ans)}]), FakeRetriever([C1, C2]))
        self.assertEqual((r["status"], r["citations"]), ("answered", [C1["id"], C2["id"]]))
        self.assertEqual(r["invalid_citations"], [])

    def test_answer_without_any_search_cannot_be_cited(self):
        r = run_agent("q", FakeController([answer(C1["id"])]), FakeRetriever([C1]))
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(r["n_searches"], 0)

    def test_model_abstain_marker(self):
        r = run_agent("q", FakeController([SEARCH, {"action": "answer", "answer": "INSUFFICIENT_EVIDENCE: nothing about it", "citations": []}]),
                      FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"]), ("abstain", "model_abstained"))
        self.assertIn("INSUFFICIENT_EVIDENCE", r["answer"])

    def test_marker_after_cited_text_is_abstain_mixed_marker_not_answered(self):
        mixed = f"Refunds take 7 days [{C1['id']}].\nINSUFFICIENT_EVIDENCE: the rest is not covered."
        r = run_agent("q", FakeController([SEARCH, {"action": "answer", "answer": mixed, "citations": [C1["id"]]}]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"], r["citations"]), ("abstain", "mixed_marker", []))
        self.assertEqual(r["raw_answer"], mixed)  # raw text kept
        self.assertEqual(r["attempted_citations"], [C1["id"]])
        self.assertTrue(any(e["event"] == "mixed_marker" for e in r["events"]))
        lower = {"action": "answer", "answer": f"7 days [{C1['id']}] (insufficient_evidence for the rest)", "citations": [C1["id"]]}
        self.assertEqual(run_agent("q", FakeController([SEARCH, lower]), FakeRetriever([C1]))["reason"], "mixed_marker")  # case-insensitive
        # a marker-free answer and a marker-first abstention are unchanged
        self.assertEqual(run_agent("q", FakeController([SEARCH, answer(C1["id"])]), FakeRetriever([C1]))["status"], "answered")
        self.assertEqual(run_agent("q", FakeController([SEARCH, ABSTAIN]), FakeRetriever([C1]))["reason"], "model_abstained")

    def test_model_format_errors_are_quality_failures_not_errors(self):
        for bad in ({"action": "search", "query": None, "format_error": "malformed search arguments"}, {"action": "search", "query": "  "}):
            r = run_agent("q", FakeController([bad]), FakeRetriever([C1]))
            self.assertEqual((r["status"], r["reason"]), ("abstain", "model_format_error"))
            self.assertEqual(r["events"][-1]["event"], "model_format_error")

    def test_truncated_answer_abstains(self):
        r = run_agent("q", FakeController([SEARCH, {**answer(C1["id"]), "finish_reason": "length"}]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"]), ("abstain", "truncated_answer"))

    def test_search_limit_abstains_and_last_turn_disables_tool(self):
        ctl, ret = FakeController([SEARCH]), FakeRetriever([C1])
        r = run_agent("q", ctl, ret, max_searches=2)
        self.assertEqual((r["status"], r["reason"], r["n_searches"]), ("abstain", "search_limit", 2))
        self.assertEqual(len(ret.calls), 2)
        self.assertEqual([c[2] for c in ctl.calls], [True, True, False])  # allow_search is False on the forced answer turn
        self.assertEqual(r["events"][-1]["event"], "search_limit")

    def test_dedup_and_budget(self):
        big = {**C2, "text": "x" * 500}
        ctl = FakeController([SEARCH, {"action": "search", "query": "again"}, answer(C1["id"])])
        r = run_agent("q", ctl, FakeRetriever([C1, big]), top_k=2, budget=200)
        searches = [e for e in r["events"] if e["event"] == "search"]
        self.assertEqual(searches[0]["new_ids"], [C1["id"]])
        self.assertEqual(searches[0]["dropped_budget"], [big["id"]])
        self.assertEqual(searches[1]["repeated_ids"], [C1["id"]])
        self.assertEqual(r["context_ids"], [C1["id"]])
        self.assertLessEqual(r["context_chars"], 200)
        self.assertEqual(r["context"].count(C1["text"]), 1)

    def test_budget_boundary_is_inclusive(self):
        need = len(agent.format_block(C1))
        ok = run_agent("q", FakeController([SEARCH, ABSTAIN]), FakeRetriever([C1]), budget=need)
        self.assertEqual((ok["context_ids"], ok["context_chars"]), ([C1["id"]], need))
        short = run_agent("q", FakeController([SEARCH, ABSTAIN]), FakeRetriever([C1]), budget=need - 1)
        self.assertEqual(short["context_ids"], [])
        both = len(agent.format_block(C1)) + 2 + len(agent.format_block(C2))  # the "\n\n" separator counts towards the budget
        r = run_agent("q", FakeController([SEARCH, ABSTAIN]), FakeRetriever([C1, C2]), top_k=2, budget=both)
        self.assertEqual((r["context_ids"], r["context_chars"], len(r["context"])), ([C1["id"], C2["id"]], both, both))
        r = run_agent("q", FakeController([SEARCH, ABSTAIN]), FakeRetriever([C1, C2]), top_k=2, budget=both - 1)
        self.assertEqual(r["context_ids"], [C1["id"]])

    def test_observation_matches_final_context_and_says_when_nothing_is_new(self):
        ctl = FakeController([SEARCH, {"action": "search", "query": "again"}, answer(C1["id"])])
        r = run_agent("q", ctl, FakeRetriever([C1, C2]), top_k=2, max_searches=3)
        history = ctl.calls[-1][1]
        self.assertEqual(len(history), 2)
        first, second = history[0]["observation"], history[1]["observation"]
        self.assertIn(r["context"], first)  # everything the model ever saw as a passage == the final context
        self.assertNotIn("<passages>", second)
        self.assertIn("No new passages", second)
        self.assertNotIn("Searches left", first + second)
        self.assertNotIn("answer now", first + second)  # only the last allowed search says so

    def test_last_search_observation_says_answer_now(self):
        ctl = FakeController([SEARCH, answer(C1["id"])])
        run_agent("q", ctl, FakeRetriever([C1]), max_searches=1)
        self.assertIn("answer now", ctl.calls[-1][1][0]["observation"])
        self.assertFalse(ctl.calls[-1][2])  # and tools are disabled on that turn

    def test_errors_propagate_and_partial_trace_survives(self):
        class Boom:
            def __init__(self):
                self.n = 0

            def search(self, query, *, filters, top_k):
                self.n += 1
                if self.n == 2:
                    raise RuntimeError("db down")
                return [C1]
        trace = {}
        with self.assertRaises(RuntimeError):
            run_agent("q", FakeController([SEARCH]), Boom(), trace=trace)
        self.assertEqual([e["event"] for e in trace["events"]], ["search"])
        self.assertEqual(trace["context_ids"], [C1["id"]])  # context built so far is kept for the error trace
        self.assertIn(C1["text"], trace["context"])
        self.assertEqual(trace["n_searches"], 1)
        with self.assertRaises(ValueError):
            run_agent("q", FakeController([{"action": "dance"}]), FakeRetriever([C1]))

    def test_extract_citations_variants(self):
        ex = agent.extract_citations
        self.assertEqual(ex("a [para:x:p1:00] b [1] [para:x:p1:00] [para:y:p2:01]"), ["para:x:p1:00", "para:y:p2:01"])
        self.assertEqual(ex("grouped [para:x:p1:00, para:y:p2:01]; [para:z:p3:02; para:x:p1:00]"), ["para:x:p1:00", "para:y:p2:01", "para:z:p3:02"])
        self.assertEqual(ex("copied header [para:x:p1:00 | data/book.pdf | page 1]"), ["para:x:p1:00"])
        self.assertEqual(ex("see [https://example.com/a:b] and [Note:1] and [1] and [para:x] and (para:x:p1:00)"), [])
        self.assertEqual(ex("edited id [bizlaw:p0104:03] and [p0104]"), ["bizlaw:p0104:03"])  # shortened ids are attempts (invalid), not 'no citation'
        self.assertEqual(ex(""), [])

    def test_redact(self):
        self.assertNotIn("abcdefghijklmnop", agent.redact("bad key sk-proj-abcdefghijklmnop12345 here"))
        agent._SECRETS.add("plainsecretvalue99")
        try:
            self.assertEqual(agent.redact("x plainsecretvalue99 y"), "x [REDACTED] y")
        finally:
            agent._SECRETS.discard("plainsecretvalue99")

    def test_shortened_id_is_an_invalid_citation_not_no_citation(self):
        text = "7 days [bizlaw:p0001:00]"
        r = run_agent("q", FakeController([SEARCH, {"action": "answer", "answer": text, "citations": agent.extract_citations(text)}]), FakeRetriever([C1]))
        self.assertEqual((r["status"], r["reason"], r["invalid_citations"]), ("abstain", "invalid_citation", ["bizlaw:p0001:00"]))

    def test_system_prompt_has_no_corpus_specific_example_id(self):
        self.assertNotIn("para:", agent.SYSTEM_PROMPT)
        self.assertIn("[<chunk-id>]", agent.SYSTEM_PROMPT)


class FakeOpenAI:
    """Records every request; raises transient errors first if asked."""

    def __init__(self, replies, fail_first=0, cycle=False):
        self.requests, self.replies, self.fail_first, self.cycle = [], list(replies), fail_first, cycle
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.requests.append(json.loads(json.dumps(kw)))
        if self.fail_first > 0:
            self.fail_first -= 1
            import httpx
            import openai
            raise openai.APIConnectionError(request=httpx.Request("POST", "http://x"))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if self.cycle:
            self.replies.append(reply)
        call = [SimpleNamespace(function=SimpleNamespace(name="search", arguments=json.dumps({"query": reply["search"]})))] if "search" in reply else None
        msg = SimpleNamespace(content=reply.get("text"), tool_calls=call)
        return SimpleNamespace(model="fake-model", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10),
                               choices=[SimpleNamespace(message=msg, finish_reason="tool_calls" if call else "stop")])


class OpenAIControllerTests(unittest.TestCase):
    def test_full_run_with_fake_client_and_retry(self):
        client = FakeOpenAI([{"search": "refund"}, {"text": f"7 days [{C1['id']}]"}], fail_first=2)
        ctl = OpenAIController(client, "gpt-4o-mini", sleep=lambda s: None)
        r = run_agent("Refund?", ctl, FakeRetriever([C1, C2]), top_k=2)
        self.assertEqual((r["status"], r["citations"]), ("answered", [C1["id"]]))
        self.assertEqual(r["usage"]["retries"], 2)
        self.assertEqual(r["usage"]["prompt_tokens"], 200)
        first, second = client.requests[2], client.requests[3]  # two failed attempts, then the real calls
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(second["messages"][-1]["role"], "tool")
        self.assertIn(C1["id"], second["messages"][-1]["content"])

    def test_retry_gives_up_and_records_attempts(self):
        import openai
        client = FakeOpenAI([{"text": "x"}], fail_first=99)
        with self.assertRaises(openai.APIConnectionError) as cm:
            OpenAIController(client, "m", attempts=3, sleep=lambda s: None).decide("q", [])
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(cm.exception.failed_attempts, 3)

    def test_forced_answer_turn_disables_tool_choice(self):
        client = FakeOpenAI([{"text": "INSUFFICIENT_EVIDENCE: x"}])
        OpenAIController(client, "m").decide("q", [{"query": "a", "observation": "obs"}], allow_search=False)
        self.assertEqual(client.requests[0]["tool_choice"], "none")

    def test_unknown_tool_and_malformed_arguments_become_format_errors(self):
        class Raw(FakeOpenAI):
            def create(self, **kw):
                self.requests.append(kw)
                call = [SimpleNamespace(function=SimpleNamespace(name=self.replies[0]["name"], arguments=self.replies[0]["args"]))]
                msg = SimpleNamespace(content=None, tool_calls=call)
                return SimpleNamespace(model="m", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1), choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")])
        for reply in ({"name": "delete_all", "args": "{}"}, {"name": "search", "args": "{not json"}, {"name": "search", "args": "{}"}):
            r = run_agent("q", OpenAIController(Raw([reply]), "m"), FakeRetriever([C1]))
            self.assertEqual((r["status"], r["reason"]), ("abstain", "model_format_error"))
            self.assertEqual(r["usage"]["prompt_tokens"], 5)  # the call was paid for and is counted

    def test_no_gold_or_labels_in_prompts(self):
        """Run the harness path (run_questions) over every public dev question, max_searches=3, with a BM25 retriever and a fake client:
        nothing from questions.json (expected answer, notes, type) or absent_questions.json (reasons, notes, checks) may reach the model."""
        from retrieval import BM25Retriever, load_chunks
        all_public = json.loads((ROOT / "data" / "questions_public.json").read_text(encoding="utf-8"))
        self.assertTrue(all(set(q) <= {"id", "split", "question", "filter"} for q in all_public))
        public = [q for q in all_public if q["split"] == "dev"]  # the held-out test split is never touched, not even here
        forbidden = []

        def add(v):
            if isinstance(v, (list, tuple)):
                for x in v:
                    add(x)
            elif isinstance(v, str) and len(v) >= 15:  # shorter labels (e.g. type names) are ordinary words that legitimately occur in passages
                forbidden.append(v)
        for q in (q for q in json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8")) if q["split"] == "dev"):
            for k in ("expected_answer", "notes", "type"):
                add(q.get(k))
        for q in (q for q in json.loads((ROOT / "data" / "absent_questions.json").read_text(encoding="utf-8")) if q["split"] == "dev"):
            for k in ("absence_reason", "absence_check", "grader_note", "related_topic", "expected_behavior", "decoy_check"):
                add(q.get(k))
        self.assertGreater(len(forbidden), 20)
        client = FakeOpenAI([{"search": "evidence"}, {"search": "more evidence"}, {"search": "even more"}, {"text": "INSUFFICIENT_EVIDENCE: fake"}], cycle=True)
        seen = []

        class Spy:
            def __init__(self, inner):
                self.inner = inner

            def search(self, query, top_k=3, filters=None):
                seen.append(filters)
                return self.inner.search(query, top_k, filters)
        traces, _ = harness.run_questions(public, OpenAIController(client, "gpt-4o-mini"), Spy(BM25Retriever(load_chunks())), filters_mode="question",
                                          top_k=5, budget=10000, max_searches=3, model="gpt-4o-mini", max_cost=1.0, quiet=True)
        self.assertEqual(len(traces), len(public))
        self.assertEqual({t["status"] for t in traces}, {"abstain"})
        blob = json.dumps(client.requests, ensure_ascii=False)
        self.assertEqual(len(client.requests), 4 * len(public))
        for f in forbidden:
            self.assertNotIn(f, blob)
        self.assertEqual(sorted(map(str, seen)), sorted(str(q.get("filter")) for q in public for _ in range(3)))  # only the public filter reaches retrieval
        for word in ("gold", "expected_answer", "grader_note", "absence_reason"):
            self.assertNotIn(word, agent.SYSTEM_PROMPT)


class SplitPolicyTests(unittest.TestCase):
    def test_test_split_refused_without_final(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                harness.check_split_policy("test", False, Path(d) / "new")

    def test_test_split_refused_when_output_exists(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                harness.check_split_policy("test", True, Path(d))

    def test_test_split_allowed_once_with_final_and_new_dir_and_dev_is_free(self):
        with tempfile.TemporaryDirectory() as d:
            harness.check_split_policy("test", True, Path(d) / "new")  # does not raise
            harness.check_split_policy("dev", False, Path(d))

    def test_test_split_extra_guards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(SystemExit):
                harness.check_split_policy("test", True, root / "test-bm25-x", method="bm25", only="t06")
            with self.assertRaises(SystemExit):
                harness.check_split_policy("test", True, root / "test-bm25-x", method="bm25", default_output=False)
            harness.check_split_policy("test", True, root / "test-bm25-a", method="bm25")
            harness.check_split_policy("test", True, root / "test-dense-a", method="dense")  # first run of any method: fine
            (root / "test-bm25-a").mkdir()
            (root / "test-bm25-b").mkdir()
            with self.assertRaises(SystemExit):  # a third test run (baseline + chosen are the two allowed IN TOTAL)
                harness.check_split_policy("test", True, root / "test-bm25-c", method="bm25")
            with self.assertRaises(SystemExit):  # global cap: another method does not get its own allowance
                harness.check_split_policy("test", True, root / "test-dense-a", method="dense")
            (root / "dev-dense-x").mkdir()  # dev folders and plain files do not count
            (root / "test-notes.txt").write_text("x", encoding="utf-8")
            self.assertEqual(harness.MAX_TEST_RUNS, 2)

    def test_dev_results_are_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "dev-bm25-base"
            harness.check_split_policy("dev", False, out, method="bm25")  # new folder: fine
            out.mkdir()
            harness.check_split_policy("dev", False, out, method="bm25")  # empty folder (e.g. crashed run): fine
            (out / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                harness.check_split_policy("dev", False, out, method="bm25")
            harness.check_split_policy("dev", False, out, method="bm25", overwrite=True)


class EvaluatorTests(unittest.TestCase):
    def test_evaluator_marks_context_evidence_and_blank_reviews(self):
        # evaluate() reads the real gold files for the dev split; feed it synthetic traces (no model involved)
        chunks = {c["id"]: c for c in json.loads((ROOT / "data" / "chunks.json").read_text(encoding="utf-8"))}
        gold = [q for q in json.loads((ROOT / "data" / "questions.json").read_text(encoding="utf-8")) if q["id"] == "d01"][0]
        hit_id = next(i for i, c in chunks.items() if harness.is_hit(c, gold["gold"]))
        other = next(i for i, c in chunks.items() if not harness.is_hit(c, gold["gold"]))
        base = {"split": "dev", "question": "q", "events": [], "usage": {}, "citations": [], "answer": "a", "raw_answer": "a", "status": "abstain"}
        traces = [{**base, "query_id": "d01", "context_ids": [hit_id]}, {**base, "query_id": "d02", "context_ids": [other]},
                  {**base, "query_id": "d03", "context_ids": [hit_id], "status": "error"}, {**base, "query_id": "x01", "context_ids": []}]
        cases, reviews = harness.evaluate(traces, "dev", chunks)
        self.assertIs(traces[0]["evidence_in_context"], True)
        self.assertIs(traces[1]["evidence_in_context"], False)  # a context without the gold evidence is a miss
        self.assertIsNone(traces[2]["evidence_in_context"])  # a provider error is neither hit nor miss
        self.assertNotIn("evidence_in_context", traces[3])  # absent question: not part of the retrieval/context metrics
        self.assertEqual({c["id"]: c["answerable"] for c in cases}, {"d01": True, "d02": True, "d03": True, "x01": False})
        self.assertEqual([r["query_id"] for r in reviews], ["d01", "d02", "x01"])  # no review item for the error run
        self.assertIsNone(reviews[0]["answer_grade"])
        self.assertIsNone(reviews[0]["context_grade"])
        self.assertIsNone(reviews[2]["abstention_correct"])
        self.assertNotIn("answer_grade", reviews[2])

    def test_pipeline_error_branch_incremental_trace_summary_and_cost_stop(self):
        from retrieval import BM25Retriever, load_chunks
        chunks = load_chunks()
        by_id = {c["id"]: c for c in chunks}
        inner = BM25Retriever(chunks)
        public = {q["id"]: q for q in json.loads((ROOT / "data" / "questions_public.json").read_text(encoding="utf-8"))}
        qs = [public["d01"], public["d02"], public["x01"]]

        class Flaky:
            def search(self, query, top_k=3, filters=None):
                if "euthanasia" in query:  # d02 only
                    raise RuntimeError("store down sk-proj-abcdefghijklmnop123456")
                return inner.search(query, top_k, filters)

        class Ctl:  # searches with the question text, then cites the first passage it was shown
            def decide(self, question, history, *, allow_search=True):
                usage = {"prompt_tokens": 1000, "completion_tokens": 100, "latency_s": 0.01, "retries": 0, "model": "gpt-4o-mini"}
                if not history:
                    return {"action": "search", "query": question, "usage": usage}
                first = re.search(r"<passages>\n\[(\S+) \|", history[-1]["observation"]).group(1)
                return {"action": "answer", "answer": f"x [{first}]", "citations": [first], "usage": usage}

        saved = []
        traces, aborted = harness.run_questions(qs, Ctl(), Flaky(), filters_mode="question", top_k=5, budget=10000, max_searches=3,
                                                model="gpt-4o-mini", max_cost=1.0, on_trace=saved.append, quiet=True)
        self.assertFalse(aborted)
        self.assertEqual([t["query_id"] for t in saved], ["d01", "d02", "x01"])  # written after every question
        self.assertEqual([t["status"] for t in traces], ["answered", "error", "answered"])
        err = traces[1]
        self.assertNotIn("abcdefghijklmnop", json.dumps(err))  # key-looking text is redacted
        self.assertEqual(err["error"]["type"], "RuntimeError")
        self.assertEqual(err["usage"]["prompt_tokens"], 1000)  # the paid model call before the failure is counted
        harness.evaluate(traces, "dev", by_id)
        args = SimpleNamespace(split="dev", method="bm25", tag="t", top_k=5, budget=10000, max_searches=3, filters="question")
        s = harness.summarize(traces, args, "gpt-4o-mini", aborted, {}, {"temperature": 0.0})
        self.assertEqual(s["status"], {"answered": 2, "abstain": 0, "error": 1})
        self.assertEqual(s["context_evidence_rate"]["N"], 1)  # d02 errored: excluded from the denominator and reported
        self.assertEqual((s["context_evidence_rate"]["errors_excluded"], s["context_evidence_rate"]["answerable_total"]), (1, 2))
        self.assertEqual(s["tool_use"], {"n": 2, "N": 2})
        self.assertEqual(s["tokens"], {"prompt": 5000, "completion": 500})
        self.assertEqual(s["estimated_cost_usd"], round((5000 * 0.15 + 500 * 0.60) / 1e6, 5))
        self.assertFalse(s["price_usd_per_1M_tokens"]["verified_against_openai"])
        self.assertEqual(s["citation_id_validity"]["accepted_ids"], 2)

        class Pricey(Ctl):
            def decide(self, question, history, *, allow_search=True):
                out = super().decide(question, history, allow_search=allow_search)
                out["usage"]["prompt_tokens"] = 10_000_000  # ~USD 1.50 per call
                return out
        traces, aborted = harness.run_questions(qs, Pricey(), inner, filters_mode="question", top_k=5, budget=10000, max_searches=3,
                                                model="gpt-4o-mini", max_cost=1.0, quiet=True)
        self.assertTrue(aborted)
        self.assertEqual(len(traces), 1)

    def test_summary_counts_invalid_citations_both_ways(self):
        t = {"query_id": "d01", "answerable": True, "status": "abstain", "reason": "invalid_citation", "events": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1},
             "n_searches": 1, "elapsed_s": 1.0, "citations": [], "attempted_citations": ["para:book:p1:00"], "invalid_citations": ["para:book:p1:00"],
             "evidence_in_context": True, "context_chars": 10}
        args = SimpleNamespace(split="dev", method="bm25", tag="t", top_k=5, budget=10000, max_searches=3, filters="question")
        s = harness.summarize([t], args, "gpt-4o-mini", False, {}, {})
        self.assertEqual(s["citation_id_validity"]["attempted_ids"], 1)
        self.assertEqual(s["citation_id_validity"]["attempted_ids_raw"], 1)  # no raw list in an old-style trace: falls back to attempted
        self.assertEqual(s["citation_id_validity"]["accepted_ids"], 0)
        self.assertEqual(s["citation_id_validity"]["answers_with_invalid_citation"], ["d01"])
        self.assertEqual(s["abstain_reasons"], {"invalid_citation": 1})


class RepairIdTests(unittest.TestCase):
    short = "doc:p0001:00"  # the model dropped the "para:" prefix of C1

    def _run(self, repair, ans=None, cites=None, results=(C1, C2)):
        ans = ans or f"7 days [{self.short}]"
        ctl = FakeController([SEARCH, {"action": "answer", "answer": ans, "citations": cites if cites is not None else agent.extract_citations(ans)}])
        return run_agent("q", ctl, FakeRetriever(list(results)), top_k=2, repair_ids=repair)

    def test_off_by_default_the_shortened_id_is_invalid(self):
        r = self._run(False)
        self.assertEqual(r["attempted_citations_raw"], r["attempted_citations"])
        self.assertEqual((r["status"], r["reason"], r["invalid_citations"]), ("abstain", "invalid_citation", [self.short]))
        self.assertEqual(r["repaired_citations"], [])

    def test_repair_replaces_a_unique_suffix_and_logs_it(self):
        r = self._run(True)
        self.assertEqual((r["status"], r["citations"]), ("answered", [C1["id"]]))
        self.assertEqual(r["repaired_citations"], [{"from": self.short, "to": C1["id"]}])
        self.assertEqual(r["attempted_citations"], [C1["id"]])  # after repair ...
        self.assertEqual(r["attempted_citations_raw"], [self.short])  # ... and exactly what the model wrote
        self.assertIn(f"[{C1['id']}]", r["answer"])
        self.assertIn(f"[{self.short}]", r["raw_answer"])  # the model's own text is kept
        self.assertTrue(any(e["event"] == "citation_repair" for e in r["events"]))

    def test_repair_does_not_touch_valid_ids_and_dedups(self):
        ans = f"a [{C1['id']}] b [{self.short}]"
        r = self._run(True, ans, [C1["id"], self.short])
        self.assertEqual(r["citations"], [C1["id"]])
        self.assertEqual(r["answer"].count(f"[{C1['id']}]"), 2)

    def test_only_the_para_prefix_is_repaired_any_other_suffix_match_stays_invalid(self):
        other = {**C2, "id": "other:doc:p0001:00", "page": 1}  # shown id that merely ENDS with the cited id
        r = self._run(True, results=(other,))
        self.assertEqual((r["status"], r["invalid_citations"], r["repaired_citations"]), ("abstain", [self.short], []))
        self.assertEqual(agent.repair_citations("x", ["doc:p0001:00"], ["other:doc:p0001:00", "xpara:doc:p0001:00"])[2], [])
        self.assertEqual(agent.repair_citations("x", ["para:doc:p9:00"], ["para:doc:p9:00"])[2], [])  # already valid
        self.assertEqual(agent.repair_citations("x", ["para:doc:p9:01"], ["para:para:doc:p9:01"])[2], [])  # cited id already has the prefix
        r = self._run(True, "x [doc:p0099:00]")
        self.assertEqual((r["status"], r["invalid_citations"]), ("abstain", ["doc:p0099:00"]))

    def test_repair_only_applies_to_ids_in_the_final_context(self):
        r = self._run(True, f"7 days [doc:p0002:00]", results=(C1,))  # C2 was never shown
        self.assertEqual(r["status"], "abstain")
        self.assertEqual(agent.repair_citations("x [doc:p1:0]", ["doc:p1:0"], ["para:doc:p1:0"])[2], [{"from": "doc:p1:0", "to": "para:doc:p1:0"}])


class RerankTests(unittest.TestCase):
    def _make(self, scores, pool=4):
        from retrieval import RerankRetriever

        class Base:
            def __init__(self):
                self.calls = []

            def search(self, query, top_k=3, filters=None):
                self.calls.append((query, top_k, filters))
                return [{**C1, "id": f"para:doc:p000{i}:00", "text": f"passage {i}", "rank": i, "score": 1.0 / i} for i in range(1, 5)][:top_k]

        class Model:
            def predict(self, pairs, show_progress_bar=False):
                self.pairs = pairs
                return scores
        r = RerankRetriever.__new__(RerankRetriever)  # no model download in unit tests
        r.base, r.pool, r.model_name, r._model = Base(), pool, "fake", Model()
        return r

    def test_rerank_reorders_pool_and_keeps_hybrid_rank(self):
        r = self._make([0.1, 0.2, 0.9, 0.5])
        out = r.search("q", 2, {"document_id": "doc"})
        self.assertEqual([x["id"] for x in out], ["para:doc:p0003:00", "para:doc:p0004:00"])
        self.assertEqual([x["rank"] for x in out], [1, 2])
        self.assertEqual(out[0]["components"], {"hybrid_rank": 3, "hybrid_score": 1.0 / 3})
        self.assertEqual(r.base.calls, [("q", 4, {"document_id": "doc"})])  # the pool, with the user filter, is taken from hybrid
        self.assertEqual(r._model.pairs[0], ("q", "passage 1"))
        self.assertEqual(len(r.search("q", 10)), 4)  # never more than the pool holds

    def test_rerank_is_deterministic_on_ties_and_works_in_the_agent_loop(self):
        r = self._make([0.5, 0.5, 0.5, 0.5])
        self.assertEqual([x["id"] for x in r.search("q", 3)], ["para:doc:p0001:00", "para:doc:p0002:00", "para:doc:p0003:00"])
        ctl = FakeController([SEARCH, ABSTAIN])
        res = run_agent("q", ctl, agent.RetrieverAdapter(r), top_k=2)
        self.assertEqual(len(res["context_ids"]), 2)

    def test_rerank_empty_pool(self):
        r = self._make([])
        r.base.search = lambda *a, **k: []
        self.assertEqual(r.search("q", 3), [])


class PromptVariantTests(unittest.TestCase):
    def test_prompt_registry_and_controller_uses_selected_prompt(self):
        self.assertEqual(set(agent.PROMPTS), {"base", "grounded"})
        client = FakeOpenAI([{"text": "INSUFFICIENT_EVIDENCE: x"}])
        OpenAIController(client, "m", prompt="grounded").decide("q", [])
        self.assertEqual(client.requests[0]["messages"][0]["content"], agent.GROUNDED_PROMPT)
        client = FakeOpenAI([{"text": "INSUFFICIENT_EVIDENCE: x"}])
        OpenAIController(client, "m").decide("q", [])
        self.assertEqual(client.requests[0]["messages"][0]["content"], agent.SYSTEM_PROMPT)
        self.assertEqual(OpenAIController(client, "m", prompt="grounded").settings()["prompt"], "grounded")
        with self.assertRaises(ValueError):
            OpenAIController(client, "m", prompt="nope")

    def test_grounded_prompt_keeps_the_protocol_and_names_no_dev_topic(self):
        g = agent.GROUNDED_PROMPT
        self.assertIn(agent.ABSTAIN_MARKER, g)
        self.assertIn("[<chunk-id>]", g)
        self.assertNotIn("para:", g)
        for topic in ("clon", "euthanasia", "Kant", "California", "OSHA", "WARN", "negligence", "gene editing"):  # dev subjects must not be written into the prompt
            self.assertNotIn(topic.lower(), g.lower())

    def test_summary_prompt_text_is_the_text_actually_sent_to_the_model(self):
        t = {"query_id": "d01", "answerable": True, "status": "answered", "reason": "ok", "events": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1},
             "n_searches": 1, "elapsed_s": 1.0, "citations": ["a"], "attempted_citations": ["a"], "invalid_citations": [],
             "evidence_in_context": True, "context_chars": 10}
        args = SimpleNamespace(split="dev", method="hybrid", tag="t", top_k=5, budget=10000, max_searches=3, filters="none")
        recorded = {}
        for name in ("base", "grounded"):
            client = FakeOpenAI([{"text": "INSUFFICIENT_EVIDENCE: x"}])
            ctl = OpenAIController(client, "gpt-4o-mini", prompt=name)
            ctl.decide("q", [])
            sent = client.requests[0]["messages"][0]["content"]  # what the (fake) API really received
            s = harness.summarize([t], args, "gpt-4o-mini", False, {}, ctl.settings())
            self.assertEqual((s["config"]["prompt"], s["config"]["system_prompt"]), (name, sent))
            recorded[name] = s["config"]["system_prompt"]
        self.assertNotEqual(recorded["base"], recorded["grounded"])
        self.assertEqual(recorded["base"], agent.SYSTEM_PROMPT)
        self.assertEqual(recorded["grounded"], agent.GROUNDED_PROMPT)


if __name__ == "__main__":
    unittest.main()
