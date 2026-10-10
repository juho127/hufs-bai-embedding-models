"""Cache integrity and query-weighted paired-bootstrap regression checks."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_runs import compare_reports, sampled_group_means
from evaluate import METRICS, cached_corpus_embeddings, checkpoint_fingerprint, file_hash, exact_topk


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.identity = {"checkpoint": "weights", "corpus_hash": "ordered-corpus", "max_seq_length": 512,
                         "prompts": {"document": "passage: "}, "dtype": "float32"}
        self.values = np.asarray([[1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)

    def encode(self, encoder=None, identity=None):
        return cached_corpus_embeddings(self.cache, identity or self.identity, 3, 2,
                                        encoder or (lambda: self.values.copy()))

    def test_reuse_skips_encoder_and_preserves_exact_ranking(self):
        first, written = self.encode()
        fail_if_called = Mock(side_effect=AssertionError("Cache hit must skip document encoding"))
        second, reused = self.encode(fail_if_called)
        self.assertEqual(written["status"], "stored")
        self.assertEqual(reused["status"], "hit")
        fail_if_called.assert_not_called()
        np.testing.assert_array_equal(first, second)
        self.assertEqual(exact_topk([[1., 0.]], first), exact_topk([[1., 0.]], second))

    def test_configuration_and_corpus_changes_require_reencoding(self):
        _, initial = self.encode()
        for name, replacement in (("checkpoint", "newweights"), ("corpus_hash", "reordered"),
                                  ("max_seq_length", 1024), ("prompts", {"document": "changed"}),
                                  ("dtype", "bfloat16")):
            with self.subTest(name=name):
                identity = dict(self.identity, **{name: replacement})
                encoder = Mock(return_value=self.values)
                _, status = self.encode(encoder, identity)
                self.assertEqual(status["status"], "stored")
                self.assertNotEqual(initial["key"], status["key"])
                encoder.assert_called_once()

    def test_truncated_array_is_recomputed(self):
        _, initial = self.encode()
        (self.cache / f"{initial['key']}.npy").write_bytes(b"truncated")
        encoder = Mock(return_value=self.values)
        with self.assertWarnsRegex(RuntimeWarning, "checksum"):
            _, status = self.encode(encoder)
        self.assertEqual(status["status"], "recomputed")
        encoder.assert_called_once()

    def test_shape_and_finite_checks_even_with_matching_checksum(self):
        for bad_array in (np.ones((2, 3), dtype=np.float32),
                          np.full((3, 2), np.nan, dtype=np.float32),
                          np.zeros((3, 2), dtype=np.float32)):
            _, initial = self.encode()
            array_path = self.cache / f"{initial['key']}.npy"
            metadata_path = self.cache / f"{initial['key']}.json"
            np.save(array_path, bad_array)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["array_sha256"] = file_hash(array_path)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertWarns(RuntimeWarning):
                result, status = self.encode()
            self.assertEqual(status["status"], "recomputed")
            np.testing.assert_array_equal(result, self.values)

    def test_uncommitted_array_does_not_count_as_cache_hit(self):
        _, initial = self.encode()
        (self.cache / f"{initial['key']}.json").unlink()
        with self.assertWarns(RuntimeWarning):
            _, status = self.encode()
        self.assertEqual(status["status"], "recomputed")

    def test_write_failure_preserves_fresh_embeddings_and_cleans_temporaries(self):
        with patch("evaluate.os.replace", side_effect=PermissionError("read-only cache")):
            with self.assertWarnsRegex(RuntimeWarning, "evaluation continues"):
                result, status = self.encode()
        self.assertEqual(status["status"], "write_failed")
        np.testing.assert_array_equal(result, self.values)
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_encoder_failure_is_not_hidden_as_cache_failure(self):
        with self.assertRaisesRegex(RuntimeError, "encoder failed"):
            self.encode(Mock(side_effect=RuntimeError("encoder failed")))

    def test_checkpoint_bytes_not_size_or_timestamp_determine_identity(self):
        model = self.root / "model"
        model.mkdir()
        weight = model / "model.safetensors"
        weight.write_bytes(b"original")
        before = weight.stat()
        first = checkpoint_fingerprint(model)
        weight.write_bytes(b"modified")
        os.utime(weight, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertNotEqual(first["sha256"], checkpoint_fingerprint(model)["sha256"])

    def test_adapter_cache_depends_on_base_weights(self):
        base, adapter = self.root / "base", self.root / "adapter"
        base.mkdir()
        adapter.mkdir()
        (base / "model.safetensors").write_bytes(b"base-one")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(base)}), encoding="utf-8")
        first = checkpoint_fingerprint(adapter)
        (base / "model.safetensors").write_bytes(b"base-two")
        self.assertNotEqual(first["sha256"], checkpoint_fingerprint(adapter)["sha256"])


class BootstrapScalingTests(unittest.TestCase):
    def fixture(self):
        baseline = {"schema_version": 1, "purpose": "synthetic_smoke_only", "model": "base",
                    "corpus_hash": "c", "queries_hash": "q", "qrels_hash": "r", "protocol": {},
                    "per_query": [{"query_id": f"q{i}", "group_id": "a" if i == 0 else "b",
                                   **{metric: 0. for metric in METRICS}} for i in range(4)]}
        candidate = copy.deepcopy(baseline)
        candidate["model"] = "candidate"
        for i, row in enumerate(candidate["per_query"]):
            for metric in METRICS:
                row[metric] = [.1, .4, .8, .9][i]
        return baseline, candidate

    def test_unequal_groups_match_manual_expansion_of_every_selected_query(self):
        # Group a has one query, group b has three: their mean of means is wrong.
        blocks = [[.1], [.4, .8, .9]]
        selected = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1], [1, 0]])
        actual = sampled_group_means([[sum(block)] for block in blocks], [1, 3], selected)[:, 0]
        expected = [np.mean([value for index in sample for value in blocks[index]]) for sample in selected]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)
        self.assertAlmostEqual(actual[1], .55)
        self.assertNotAlmostEqual(actual[1], (.1 + .7) / 2)

    def test_ci_matches_manual_pcg64_draws_for_group_and_query_units(self):
        baseline, candidate = self.fixture()
        for unit in ("group", "query"):
            with self.subTest(unit=unit):
                blocks = [[.1], [.4, .8, .9]] if unit == "group" else [[.1], [.4], [.8], [.9]]
                rng = np.random.Generator(np.random.PCG64(17))
                samples = rng.integers(0, len(blocks), size=(513, len(blocks)))
                expanded = [np.mean([value for index in sample for value in blocks[index]]) for sample in samples]
                result = compare_reports(baseline, candidate, replicates=513, seed=17, unit=unit)
                np.testing.assert_allclose(result["metrics"]["ndcg@10"]["delta_ci95"],
                                           np.quantile(expanded, [.025, .975]), atol=1e-15)
                self.assertAlmostEqual(result["metrics"]["ndcg@10"]["delta"], .55)
                self.assertEqual(result["bootstrap_rng"]["algorithm"], "numpy.random.PCG64")

    def test_report_order_does_not_change_seeded_comparison(self):
        baseline, candidate = self.fixture()
        first = compare_reports(baseline, candidate, replicates=100)
        baseline["per_query"].reverse()
        candidate["per_query"].reverse()
        self.assertEqual(first, compare_reports(baseline, candidate, replicates=100))

    def test_invalid_unit_and_nonfinite_metric_are_rejected(self):
        baseline, candidate = self.fixture()
        with self.assertRaisesRegex(ValueError, "unit"):
            compare_reports(baseline, candidate, replicates=100, unit="invalid")
        candidate["per_query"][0]["ndcg@10"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            compare_reports(baseline, candidate, replicates=100)


if __name__ == "__main__":
    unittest.main()
