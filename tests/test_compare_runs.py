"""Offline tests of the preregistered selection rule (SELECTION.md with amendment 4) implemented in src/compare_runs.py.
Synthetic rows, no results folders (except the temp dirs built here)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import compare_runs as cr  # noqa: E402


def mk(run, method="hybrid", cer=21, false_abst=0, absent_ab=2, lat=3.0, cost=0.01, points=None, supported=None, accepted=24, abst_ok=2, errors=0,
       hit1=None, hit3=None):
    graded = None if points is None else {"answer_points": points, "answer_points_max": 42, "supported_citations": 20 if supported is None else supported,
                                          "accepted_citations": accepted, "abstention_correct": abst_ok, "n_absent": 2, "context_grade_2": 21, "context_graded": 21}
    retr = None if hit1 is None else {"hit@1": f"{hit1}/21", "hit@3": f"{hit3}/21", "hit@5": "21/21", "mrr@10": 0.9}
    return {"run": run, "method": method, "cer": {"n": cer, "N": 21}, "errors": errors, "answerable": {"abstain": false_abst},
            "absent": {"abstain_by_model": absent_ab}, "median_latency_s": lat, "cost_usd": cost, "graded": graded, "retrieval_dev": retr}


def rows(*items):
    return {r["run"]: r for r in items}


BASE, REP = "dev-hybrid-base-nofilter", "dev-hybrid-base-nofilter-rep2"


class RuleTests(unittest.TestCase):
    def test_gates(self):
        inc = mk(BASE)
        self.assertTrue(cr.gates(mk("c", cer=20), inc)["pass"])  # may lose one question
        self.assertFalse(cr.gates(mk("c", cer=19), inc)["pass"])
        self.assertFalse(cr.gates(mk("c", false_abst=2), inc)["pass"])  # at most +1 false abstention
        self.assertTrue(cr.gates(mk("c", false_abst=1), inc)["pass"])
        self.assertFalse(cr.gates(mk("c", errors=1), inc)["pass"])
        self.assertFalse(cr.gates(mk("c", absent_ab=1), inc)["pass"])  # proxy for G3
        g = cr.gates(mk("c", points=40, abst_ok=1, absent_ab=2), mk(BASE, points=40, abst_ok=2))
        self.assertFalse(g["G3_absent_abstentions"])  # graded labels replace the proxy when both runs have them
        self.assertTrue(g["G3_basis"].startswith("graded"))

    def test_one_question_is_never_significant_and_tiebreak_needs_more_than_25_percent(self):
        by = rows(mk(BASE), mk(REP))
        res = cr.challenge(mk("c", cer=21, lat=2.0), mk(BASE), by)  # same CER, 33% faster
        self.assertTrue(res["replaces"])
        self.assertEqual(res["decided_by"], "latency")
        self.assertFalse(cr.challenge(mk("c", lat=2.5), mk(BASE), by)["replaces"])  # 17% faster: inside the noise of the baseline pair -> incumbent
        res = cr.challenge(mk("c", lat=3.2), mk(BASE), by)  # same cost -> incumbent retained by default
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "default"))
        self.assertIn("by default", res["steps"][-1])
        self.assertFalse(cr.challenge(mk("c", lat=3.1, cost=0.008), mk(BASE), by)["replaces"])  # 20% cheaper: not more than 25%
        res = cr.challenge(mk("c", lat=3.1, cost=0.007), mk(BASE), by)  # 30% cheaper, latency within 25%
        self.assertEqual((res["replaces"], res["decided_by"]), (True, "cost"))
        self.assertFalse(cr.challenge(mk("c", lat=4.0), mk(BASE), by)["replaces"])  # slower by more than 25% -> incumbent

    def test_graded_level_decides_before_cer_and_uses_threshold_with_noise(self):
        by = rows(mk(BASE, points=38), mk(REP, points=38))  # no noise: T = 2
        res = cr.challenge(mk("c", points=40, lat=9.0), mk(BASE, points=38), by)
        self.assertEqual((res["replaces"], res["decided_by"]), (True, "L1"))  # +2 points beats a slower run
        self.assertFalse(cr.challenge(mk("c", points=39, lat=9.0), mk(BASE, points=38), by)["replaces"])  # +1 point: not significant -> latency -> incumbent
        noisy = rows(mk(BASE, points=38), mk(REP, points=41))  # the two identical runs differ by 3 points: T = 3
        self.assertFalse(cr.challenge(mk("c", points=40, lat=9.0), mk(BASE, points=38), noisy)["replaces"])
        self.assertTrue(cr.challenge(mk("c", points=41, lat=9.0), mk(BASE, points=38), noisy)["replaces"])

    def test_incumbent_favoured_level_ends_the_comparison_even_if_challenger_is_faster_or_retrieves_better(self):
        """Text and code agree (amendment 4): the first level with a difference >= T decides in EITHER direction."""
        by = rows(mk(BASE, points=38), mk(REP, points=38))
        res = cr.challenge(mk("c", points=36, lat=1.0), mk(BASE, points=38), by)  # -2 points but 3x faster
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "L1"))
        self.assertIn("never wins", res["steps"][-1])
        res = cr.challenge(mk("c", points=36, lat=3.0, hit1=19, hit3=21), mk(BASE, points=38, hit1=16, hit3=20), by)  # clause R must not rescue it
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "L1"))
        by2 = rows(mk(BASE, cer=21), mk(REP, cer=21))
        res = cr.challenge(mk("c", cer=21, lat=3.0), mk(BASE, cer=22), by2)  # CER -1 is inside T=2
        self.assertEqual(res["decided_by"], "default")

    def test_cer_level_with_noise_threshold(self):
        by = rows(mk(BASE, cer=19), mk(REP, cer=21))  # noise 2 on CER
        self.assertTrue(cr.challenge(mk("c", cer=21, lat=9.0), mk(BASE, cer=19), by)["replaces"])
        self.assertFalse(cr.challenge(mk("c", cer=20, lat=9.0), mk(BASE, cer=19), by)["replaces"])

    def test_l3_is_the_share_of_supported_citations_not_their_number(self):
        by = rows(mk(BASE, points=38, supported=20, accepted=24), mk(REP, points=38, supported=20, accepted=24))
        inc = mk(BASE, points=38, supported=20, accepted=24)  # share 0.833
        more = mk("c", points=38, supported=33, accepted=40, lat=3.0)  # 13 more supported citations, share 0.825: volume must not win
        res = cr.challenge(more, inc, by)
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "default"))
        better = mk("c", points=38, supported=24, accepted=24, lat=9.0)  # share 1.0 vs 0.833: +0.167 >= 0.10
        self.assertEqual((cr.challenge(better, inc, by)["replaces"], cr.challenge(better, inc, by)["decided_by"]), (True, "L3"))
        worse = mk("c", points=38, supported=12, accepted=24, lat=1.0)  # share 0.5, faster: significantly worse -> incumbent
        res = cr.challenge(worse, inc, by)
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "L3"))
        noisy = rows(mk(BASE, points=38, supported=20, accepted=24), mk(REP, points=38, supported=17, accepted=24))  # pair differs by 0.125 share
        self.assertFalse(cr.challenge(mk("c", points=38, supported=22, accepted=24, lat=9.0), inc, noisy)["replaces"])  # +0.083 < T=0.125
        self.assertEqual(cr.method_noise(noisy, "hybrid")["support_share"], 0.125)

    def test_retrieval_clause_r_needs_hit1_gain_hit3_not_lower_and_latency_ratio(self):
        by = rows(mk(BASE), mk(REP))
        inc = mk(BASE, hit1=16, hit3=20, lat=3.0)
        res = cr.challenge(mk("c", method="rerank", hit1=19, hit3=21, lat=3.9), inc, by)  # +3 Hit@1, x1.3 latency, nothing else decides
        self.assertEqual((res["replaces"], res["decided_by"]), (True, "R"))
        self.assertEqual(cr.challenge(mk("c", hit1=18, hit3=21, lat=3.0), inc, by)["decided_by"], "default")  # +2: not enough
        self.assertEqual(cr.challenge(mk("c", hit1=19, hit3=19, lat=3.0), inc, by)["decided_by"], "default")  # Hit@3 lower
        res = cr.challenge(mk("c", hit1=19, hit3=21, lat=4.3), inc, by)  # x1.43 latency: R does not apply, slower than 25% -> incumbent
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "latency"))
        self.assertEqual(cr.challenge(mk("c", hit1=19, hit3=21, lat=3.0), mk(BASE, lat=3.0), by)["decided_by"], "default")  # no retrieval numbers: R silent
        self.assertFalse(cr.retrieval_gain(mk("c"), mk(BASE))["applies"])
        res = cr.challenge(mk("c", cer=19, hit1=19, hit3=21, lat=3.0), inc, by)  # G1 fails: R can never override a gate
        self.assertEqual((res["replaces"], res["decided_by"]), (False, "gate"))

    def test_select_processes_challengers_in_order_and_flags_provisional(self):
        by = rows(mk(BASE), mk(REP), mk("dev-rerank-minilm-nofilter", method="rerank", lat=9.0),
                  mk("dev-hybrid-grounded-nofilter", false_abst=4), mk("dev-hybrid-repair-nofilter", lat=1.5))
        sel = cr.select(by)
        self.assertEqual(sel["final_incumbent"], "dev-hybrid-repair-nofilter")  # the faster tie wins, the failing and slow ones do not
        self.assertTrue(sel["status"].startswith("PROVISIONAL"))
        self.assertTrue(sel["warnings"])
        self.assertEqual([e["challenger"] for e in sel["log"][:3]], ["dev-bm25-base-nofilter", "dev-dense-base-nofilter", "dev-rerank-minilm-nofilter"])
        self.assertIn("skipped", sel["log"][0])  # missing runs are reported, not invented
        graded = {k: mk(k, points=38) for k in (BASE, REP, "dev-hybrid-repair-nofilter")}
        sel = cr.select(graded)
        self.assertTrue(sel["status"].startswith("FINAL"))
        self.assertEqual(sel["warnings"], [])

    def test_ungraded_noise_pair_makes_the_decision_provisional_not_final(self):
        """Finding 8: without graded rep2 the L1/L3 noise silently becomes 0 (T = floor); that must be flagged."""
        by = rows(mk(BASE, points=38), mk(REP), mk("dev-hybrid-repair-nofilter", points=38))  # rep2 has no labels
        self.assertEqual(cr.method_noise(by, "hybrid"), {"cer": 0, "graded_pair": False})
        sel = cr.select(by)
        self.assertTrue(sel["status"].startswith("PROVISIONAL"))
        self.assertTrue(any("not fully graded" in w for w in sel["warnings"]))
        missing = rows(mk(BASE, points=38), mk("dev-hybrid-repair-nofilter", points=38))  # no rep2 at all
        sel = cr.select(missing)
        self.assertTrue(sel["status"].startswith("PROVISIONAL"))
        self.assertTrue(any("missing" in w for w in sel["warnings"]))
        bm = rows(mk(BASE, points=38), mk(REP, points=38), mk("dev-bm25-base-nofilter", method="bm25", points=38))  # bm25 involved but its pair is absent
        self.assertTrue(cr.select(bm)["status"].startswith("PROVISIONAL"))

    def test_retrieval_hits_reads_the_retrieval_benchmark_and_the_reranker_file(self):
        def summary(hybrid, rerank=None):
            m = {"hybrid": {"hit@1": hybrid, "hit@3": "20/21", "hit@5": "21/21", "mrr@10": 0.861}}
            if rerank:
                m["rerank"] = {"hit@1": rerank, "hit@3": "21/21", "hit@5": "21/21", "mrr@10": 0.95}
            return {"methods": m}
        old = cr.RETR
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for sub, data in (("dev-filters-none", summary("16/21")), ("dev-filters-none-rerank-minilm", summary("16/21", "19/21"))):
                (root / sub).mkdir()
                (root / sub / "summary.json").write_text(json.dumps(data), encoding="utf-8")
            cr.RETR = root
            try:
                self.assertEqual(cr.retrieval_hits("hybrid", "none", {"config": {}}), {"hit@1": "16/21", "hit@3": "20/21", "hit@5": "21/21", "mrr@10": 0.861})
                rr = cr.retrieval_hits("rerank", "none", {"config": {"rerank": {"cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2"}}})
                self.assertEqual((rr["hit@1"], rr["hit@3"]), ("19/21", "21/21"))
                self.assertIsNone(cr.retrieval_hits("hybrid", "question", {"config": {}}))  # no such file
                self.assertIsNone(cr.retrieval_hits("rerank", "none", {"config": {"rerank": {"cross_encoder": "BAAI/bge-reranker-base"}}}))
                self.assertIsNone(cr.retrieval_hits("bm25", "none", {"config": {}}))  # method not in the file
            finally:
                cr.RETR = old
        self.assertEqual((cr.count_of("19/21"), cr.count_of(None)), (19, None))


if __name__ == "__main__":
    unittest.main()
