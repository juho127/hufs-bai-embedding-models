"""Paired bootstrap confidence intervals for fixed-protocol retrieval reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate import METRICS


def percentile(values, quantile):
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (1 - position + lower) + values[upper] * (position - lower)


def sampled_group_means(group_sums, group_sizes, selected):
    """Query-weighted means for paired, with-replacement group selections.

    A group containing three queries contributes three observations every time
    it is selected. Averaging group means would silently change the estimand.
    """
    import numpy as np
    sums = np.asarray(group_sums, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.int64)
    selected = np.asarray(selected, dtype=np.int64)
    if sums.ndim != 2 or sizes.shape != (len(sums),) or selected.ndim != 2:
        raise ValueError("Incompatible group sums, sizes or selections")
    if not len(sums) or not np.isfinite(sums).all() or np.any(sizes < 1):
        raise ValueError("Groups must have finite sums and positive sizes")
    if not selected.shape[1] or np.any(selected < 0) or np.any(selected >= len(sums)):
        raise ValueError("Invalid selected group index")
    return sums[selected].sum(axis=1) / sizes[selected].sum(axis=1, keepdims=True)


def compare_reports(baseline, candidate, replicates=10000, seed=42, unit="group"):
    import numpy as np
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    if unit not in ("group", "query"):
        raise ValueError("Resampling unit must be group or query")
    for key in ("schema_version", "purpose", "corpus_hash", "queries_hash", "qrels_hash", "protocol"):
        if baseline.get(key) != candidate.get(key) or key not in baseline:
            raise ValueError(f"Cannot compare reports with different/missing {key}")
    base = {row["query_id"]: row for row in baseline["per_query"]}
    cand = {row["query_id"]: row for row in candidate["per_query"]}
    if len(base) != len(baseline["per_query"]) or len(cand) != len(candidate["per_query"]):
        raise ValueError("Duplicate query ids in report")
    if not base or set(base) != set(cand):
        raise ValueError("Both reports must contain the same nonempty query set")
    groups = {}
    for qid in sorted(base):
        if base[qid].get("group_id", qid) != cand[qid].get("group_id", qid):
            raise ValueError(f"Group mismatch for query {qid}")
        group = base[qid].get("group_id", qid) if unit == "group" else qid
        groups.setdefault(str(group), []).append(qid)
    blocks = list(groups.values())
    base_values = np.asarray([[base[qid][metric] for metric in METRICS]
                              for qid in sorted(base)], dtype=np.float64)
    cand_values = np.asarray([[cand[qid][metric] for metric in METRICS]
                              for qid in sorted(base)], dtype=np.float64)
    if not np.isfinite(base_values).all() or not np.isfinite(cand_values).all():
        raise ValueError("Metrics must be finite")
    offsets = {qid: i for i, qid in enumerate(sorted(base))}
    differences = cand_values - base_values
    group_sums = np.stack([differences[[offsets[qid] for qid in block]].sum(axis=0)
                           for block in blocks])
    group_sizes = np.asarray([len(block) for block in blocks], dtype=np.int64)
    # Explicit PCG64 replaces the former Python random.Random stream. The
    # resampling distribution is unchanged; version/algorithm are reported.
    rng = np.random.Generator(np.random.PCG64(seed))
    sampled = np.empty((replicates, len(METRICS)), dtype=np.float64)
    # Bounded chunks avoid allocating replicates * groups * metrics at once.
    chunk_size = max(1, min(256, 1_000_000 // len(blocks)))
    for start in range(0, replicates, chunk_size):
        count = min(chunk_size, replicates - start)
        selected = rng.integers(0, len(blocks), size=(count, len(blocks)))
        sampled[start:start + count] = sampled_group_means(group_sums, group_sizes, selected)
    stats = {}
    for index, metric in enumerate(METRICS):
        values = np.sort(sampled[:, index])
        old = float(base_values[:, index].mean())
        new = float(cand_values[:, index].mean())
        stats[metric] = {"baseline": old, "candidate": new, "delta": new - old,
                         "delta_ci95": [float(percentile(values, .025)), float(percentile(values, .975))]}
    return {"schema_version": 1, "purpose": baseline["purpose"], "baseline_model": baseline["model"],
            "candidate_model": candidate["model"], "query_count": len(base), "resampling_unit": unit,
            "unit_count": len(blocks), "replicates": replicates, "seed": seed,
            "bootstrap_rng": {"algorithm": "numpy.random.PCG64", "numpy_version": np.__version__,
                              "implementation": "query_weighted_group_sums_v1",
                              "stream_changed_from": "python.random.Random"},
            "primary_metric": "ndcg@10", "metrics": stats,
            "interpretation": "Paired percentile bootstrap; positive CI lower bound supports improvement on this fixed evaluation set. Synthetic smoke scores are not model-quality evidence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unit", choices=["group", "query"], default="group")
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    report = compare_reports(baseline, candidate, args.replicates, args.seed, args.unit)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
