"""Meaningful release invariants with synthetic, redistributable fixtures."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_legal_release import (ReleaseBuilder, select_evaluation_queries,
                                   split_families, version_title)


def document(doc_id, typ=1, domain="법률/법학", title=None, text=None, query=None):
    return {"doc_id": doc_id, "doc_title": title or "Title " + doc_id, "doc_source": "Synthetic agency",
            "doc_class": {"class": domain}, "paragraphs": [{"context_id": "c" + doc_id,
            "context": text or "Evidence " + doc_id, "tbs": None,
            "qas": [{"qa_type": typ, "question_id": "q" + doc_id, "question": query or "Question " + doc_id,
                     "answer": {"source": "c" + doc_id, "text": "Evidence"}}]}]}


class LegalReleaseTests(unittest.TestCase):
    def test_domain_and_type_filter_preserves_distractors(self):
        builder = ReleaseBuilder()
        for i, typ, domain in [(1, 1, "법률/법학"), (2, 5, "법률/행정"), (3, 2, "법률/법학"),
                               (4, 4, "법률/법학"), (5, 3, "법률/법학"), (6, 1, "금융/경제")]:
            builder.add_document(document(str(i), typ, domain), "Training")
        rows, corpus, *_ = builder.materialize()
        self.assertEqual({row["qa_type"] for row in rows}, {1, 5})
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(corpus), 4)
        self.assertNotIn("6", builder.docs)
        corpus_by_id = {row["id"]: row["text"] for row in corpus}
        for row in rows:
            self.assertEqual(row["positive"], corpus_by_id[row["positive_id"]])

    def test_title_passage_and_query_connections_protect_reviewed_family(self):
        builder = ReleaseBuilder(reviewed_docs={"a"})
        docs = [document("a", title="Shared title"), document("b", title=" Shared   title ", text="bridge"),
                document("c", text=" bridge ", query="bridge question"), document("d", query="bridge question")]
        docs += [document(str(i)) for i in range(35)]
        for doc in docs:
            builder.add_document(doc, "Training")
        rows, _, mapping, _, reviewed = builder.materialize()
        self.assertEqual(len({mapping[did] for did in ("a", "b", "c", "d")}), 1)
        parts = split_families(rows, reviewed)
        self.assertNotIn(mapping["a"], {row["group_id"] for row in parts["test"]})
        self.assertEqual(parts, split_families(rows, reviewed))

    def test_review_exclusion_propagates_to_duplicate_pair_without_rewriting(self):
        builder = ReleaseBuilder(quality_exclusions={("a", "qa"): "conditional"})
        builder.add_document(document("a", text=" Same passage ", query=" Same question "), "Training")
        builder.add_document(document("b", text="Same passage", query="Same question"), "Validation")
        builder.add_document(document("c", text="  Kept original  ", query=" Original query? "), "Training")
        rows, corpus, *_ = builder.materialize()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], " Original query? ")
        self.assertEqual(rows[0]["positive"], "  Kept original  ")
        self.assertEqual(len(corpus), 2)
        # An excluded QA type may establish the original corpus representative.
        other = ReleaseBuilder()
        other.add_document(document("a", typ=2, text="  Same original "), "Training")
        other.add_document(document("b", typ=1, text="Same original"), "Training")
        pairs, candidates, *_ = other.materialize()
        self.assertEqual(pairs[0]["positive"], candidates[0]["text"])
        self.assertEqual(pairs[0]["positive"], "  Same original ")

    def test_selection_is_proportional_diverse_and_preserves_all_positives(self):
        rows = []
        for i in range(100):
            rows.append({"query_id": f"q{i}", "positive_id": f"p{i}", "group_id": f"g{i % 30}",
                         "qa_type": 1 if i < 60 else 5})
        rows.append(dict(rows[0], positive_id="another-positive"))
        selected = select_evaluation_queries(rows, 20, 42)
        queries = {row["query_id"]: row for row in selected}
        self.assertEqual(len(queries), 20)
        self.assertEqual(sum(row["qa_type"] == 1 for row in queries.values()), 12)
        self.assertEqual(len({row["group_id"] for row in queries.values()}), 20)
        self.assertEqual(selected, select_evaluation_queries(rows, 20, 42))
        if "q0" in queries:
            self.assertEqual(sum(row["query_id"] == "q0" for row in selected), 2)

    def test_edition_key_does_not_strip_meaningful_numbers(self):
        self.assertEqual(version_title("법률 실무 (개정판)"), "법률 실무")
        self.assertNotEqual(version_title("2021년 법률 실무"), version_title("2022년 법률 실무"))
        self.assertNotEqual(version_title("제1조 해설"), version_title("제2조 해설"))


if __name__ == "__main__":
    unittest.main()
