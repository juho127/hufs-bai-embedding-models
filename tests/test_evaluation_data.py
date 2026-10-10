"""Small checks for ranking correctness, comparison integrity and split leakage."""
import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runs import compare_reports
from evaluate import exact_topk, score_ranking
from prepare_data import assert_no_leakage, read_aihub_qa, split_rows


class RetrievalTests(unittest.TestCase):
    def test_full_corpus_chunked_ranking_and_ties(self):
        ids, scores = exact_topk([[1., 0.], [0., 1.]],
                                 [[1., 0.], [0., 1.], [1., 1.], [1., 0.]],
                                 k=4, query_chunk_size=1, corpus_chunk_size=1)
        self.assertEqual(ids, [[0, 3, 2, 1], [1, 2, 0, 3]])
        self.assertAlmostEqual(scores[0][2], 1 / math.sqrt(2), places=6)

    def test_multiple_positives_graded_relevance(self):
        scores = score_ranking(["other", "a", "b"], {"a": 2, "b": 1, "missing": 1})
        expected = (3 / math.log2(3) + 1 / math.log2(4)) / (3 + 1 / math.log2(3) + 1 / math.log2(4))
        self.assertAlmostEqual(scores["ndcg@10"], expected)
        self.assertEqual(scores["mrr@10"], .5)
        self.assertAlmostEqual(scores["recall@50"], 2 / 3)

    def test_no_positive_does_not_silently_pass(self):
        with self.assertRaises(ValueError):
            score_ranking(["a"], {"a": 0})


class SplitTests(unittest.TestCase):
    def fixture(self):
        return [{"query": f"question {i}", "positive": f"passage {i}",
                 "query_id": f"q{i}", "positive_id": f"p{i}", "group_id": f"g{i}"}
                for i in range(20)]

    def test_duplicate_documents_force_same_split(self):
        rows = self.fixture()
        rows.append({"query": "alternate wording", "positive": " passage   0 ",
                     "query_id": "q-other", "positive_id": "p-other", "group_id": "g-other"})
        parts = split_rows(rows)
        assert_no_leakage(parts)
        owner = {row["query_id"]: name for name, data in parts.items() for row in data}
        self.assertEqual(owner["q0"], owner["q-other"])
        self.assertEqual(parts, split_rows(rows))

    def test_same_query_multiple_positives_force_same_split(self):
        rows = self.fixture()
        rows.append(dict(rows[0], positive="other positive", positive_id="p-multi", group_id="g-multi"))
        parts = split_rows(rows)
        found = [(name, row) for name, data in parts.items() for row in data if row["query_id"] == "q0"]
        self.assertEqual(len(found), 2)
        self.assertEqual(found[0][0], found[1][0])

    def test_inconsistent_document_id_rejected(self):
        rows = self.fixture()
        rows[1]["positive_id"] = rows[0]["positive_id"]
        with self.assertRaisesRegex(ValueError, "Conflicting positive"):
            split_rows(rows)


class ComparisonTests(unittest.TestCase):
    def report(self):
        return {"schema_version": 1, "purpose": "synthetic_smoke_only", "model": "fixture",
                "corpus_hash": "c", "queries_hash": "q", "qrels_hash": "r", "protocol": {},
                "per_query": [{"query_id": f"q{i}", "group_id": f"g{i}",
                               "ndcg@10": .25, "mrr@10": .5, "recall@10": .5, "recall@50": 1.}
                              for i in range(3)]}

    def test_identical_runs_zero_delta(self):
        report = self.report()
        result = compare_reports(report, report, replicates=100)
        self.assertEqual(result["metrics"]["ndcg@10"]["delta_ci95"], [0., 0.])

    def test_changed_corpus_rejected(self):
        left, right = self.report(), self.report()
        right["corpus_hash"] = "other"
        with self.assertRaisesRegex(ValueError, "corpus_hash"):
            compare_reports(left, right, replicates=100)

    def test_paired_positive_delta(self):
        left, right = self.report(), copy.deepcopy(self.report())
        for row in right["per_query"]:
            row["ndcg@10"] += .25
        result = compare_reports(left, right, replicates=100)
        self.assertEqual(result["metrics"]["ndcg@10"]["delta_ci95"], [.25, .25])


class AIHubAdapterTests(unittest.TestCase):
    def parse_fixture(self, fixture):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "fixture.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            return read_aihub_qa(path)

    def parse(self, qas):
        fixture = {"data": [{"doc_id": "FL-example", "paragraphs": [
            {"context_id": "C1", "context": "Parent paragraph is not necessarily the evidence.",
             "qas": qas, "tbs": [{"table_id": "T1", "table": "table contents"}]},
            {"context_id": "C2", "context": "A different paragraph contains the legal basis.", "qas": []}]}]}
        return self.parse_fixture(fixture)

    def test_released_null_tables_allow_text_evidence(self):
        fixture = {"Dataset": {"Identifier": "synthetic-schema-fixture"}, "data": [{
            "doc_id": "synthetic-law-1", "doc_class": {"class": "법률/사회과학/법학일반"},
            "paragraphs": [{"context_id": "C1", "context": "Synthetic legal evidence.",
                            "tbs": None, "qas": [{"qa_type": 1, "question_id": "Q1",
                            "question": "Where is the evidence?",
                            "answer": {"source": "C1", "text": "legal evidence",
                                       "cell_text": None, "options": None}}]}]}]}
        rows, skipped = self.parse_fixture(fixture)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["positive_id"], "aihub:synthetic-law-1:context:C1")
        self.assertEqual(skipped, {})

    def test_released_context_table_id_collision_skips_table(self):
        fixture = {"data": [{"doc_id": "synthetic-table-1", "paragraphs": [{
            "context_id": "C1", "context": "Synthetic table caption.",
            "tbs": [{"table_id": "C1", "table_title": "Example", "table": "<table />"}],
            "qas": [{"qa_type": 3, "question_id": "Q1", "question": "What is the table value?",
                     "answer": {"source": "C1", "text": "value"}}]}]}]}
        rows, skipped = self.parse_fixture(fixture)
        self.assertEqual(rows, [])
        self.assertEqual(skipped, {"table_requires_reviewed_serialization": 1})

    def test_official_singular_answer_resolves_other_context(self):
        rows, skipped = self.parse([{"question_id": "Q1", "question": "Where is the legal basis?",
                                    "answer": {"source": "C2", "text": "legal basis"}}])
        self.assertEqual(len(rows), 1)
        self.assertIn("different paragraph", rows[0]["positive"])
        self.assertEqual(rows[0]["positive_id"], "aihub:FL-example:context:C2")
        self.assertEqual(skipped, {})

    def test_tables_and_missing_sources_are_never_parent_positives(self):
        rows, skipped = self.parse([
            {"question_id": "Q1", "question": "table question", "answer": {"source": "T1", "text": "x"}},
            {"question_id": "Q2", "question": "unknown source", "answer": {"source": "C404", "text": "y"}},
            {"question_id": "Q3", "question": "missing source", "answer": {"text": "z"}}])
        self.assertEqual(rows, [])
        self.assertEqual(skipped, {"table_requires_reviewed_serialization": 1,
                                   "unresolved_answer_source": 1, "missing_answer_source": 1})

    def test_plural_answers_require_extract_in_parent(self):
        rows, skipped = self.parse([
            {"question_id": "Q1", "question": "extractive question", "answers": [{"text": "Parent paragraph"}]},
            {"question_id": "Q2", "question": "bad extract", "answers": [{"text": "not actually present"}]}])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["positive_id"], "aihub:FL-example:context:C1")
        self.assertEqual(skipped, {"answer_not_found_in_parent_context": 1})


if __name__ == "__main__":
    unittest.main()
