"""Create leakage-aware train/dev/test pairs and frozen retrieval inputs.

Canonical input JSONL requires query, positive, query_id, positive_id, group_id.
group_id must identify the parent case/document or all versions of one document.
Duplicate document/question text and shared IDs are joined before splitting.
These exact checks do not detect paraphrases; review source families separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import content_hash, load_collection, read_jsonl


REQUIRED = ("query", "positive", "query_id", "positive_id", "group_id")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, index):
        while index != self.parent[index]:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left, right):
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def canonicalize(rows):
    if not rows:
        raise ValueError("No input pairs")
    for position, row in enumerate(rows, 1):
        for field in REQUIRED:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Row {position}: {field} must be a nonempty string")
    # Reused IDs with incompatible texts cannot be silently merged.
    for id_field, text_field in (("query_id", "query"), ("positive_id", "positive")):
        ids = {}
        text_to_id = {}
        for row in rows:
            key, text = row[id_field], normalized(row[text_field])
            if key in ids and ids[key] != text:
                raise ValueError(f"Conflicting {text_field} for {id_field} {key!r}")
            ids[key] = text
            text_to_id[text] = min(key, text_to_id.get(text, key))
        for row in rows:
            row[id_field] = text_to_id[normalized(row[text_field])]
    return rows


def connected_groups(rows):
    uf = UnionFind(len(rows))
    seen = {}
    for index, row in enumerate(rows):
        keys = [("source_group", row["group_id"]), ("positive_id", row["positive_id"]),
                ("query_id", row["query_id"]), ("positive_text", digest(normalized(row["positive"]))),
                ("query_text", digest(normalized(row["query"])))]
        for key in keys:
            if key in seen:
                uf.union(index, seen[key])
            else:
                seen[key] = index
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[uf.find(index)].append(row)
    components = []
    for component in grouped.values():
        original_groups = sorted({row["group_id"] for row in component})
        component_id = "group-" + digest(json.dumps(original_groups, ensure_ascii=False))[:20]
        # The component group is used for both false-negative control and bootstrap.
        components.append((component_id, [dict(row, source_group_id=row["group_id"], group_id=component_id)
                                          for row in component]))
    return sorted(components, key=lambda item: item[0])


def split_rows(rows, seed=42, dev_fraction=.1, test_fraction=.1):
    if not (0 < dev_fraction < 1 and 0 < test_fraction < 1 and dev_fraction + test_fraction < 1):
        raise ValueError("Dev and test fractions must be positive and sum to less than 1")
    components = connected_groups(canonicalize([dict(row) for row in rows]))
    if len(components) < 3:
        raise ValueError("At least three independent document/case groups are needed for train/dev/test")
    random.Random(seed).shuffle(components)
    # Allocate components, never independently shuffled QA rows.
    dev_count = max(1, round(len(components) * dev_fraction))
    test_count = max(1, round(len(components) * test_fraction))
    while dev_count + test_count >= len(components):
        if dev_count >= test_count and dev_count > 1:
            dev_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Insufficient independent groups")
    parts = {"dev": components[:dev_count], "test": components[dev_count:dev_count + test_count],
             "train": components[dev_count + test_count:]}
    splits = {}
    for name, selected in parts.items():
        unique = {}
        for _, component in selected:
            for row in component:
                unique[(row["query_id"], row["positive_id"])] = row
        splits[name] = [unique[key] for key in sorted(unique)]
    assert_no_leakage(splits)
    return splits


def assert_no_leakage(splits):
    for field in ("group_id", "positive_id", "query_id", "positive", "query"):
        sets = {name: {normalized(row[field]) if field in ("positive", "query") else row[field]
                       for row in rows} for name, rows in splits.items()}
        names = list(sets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = sets[left] & sets[right]
                if overlap:
                    raise ValueError(f"Leakage between {left}/{right} for {field}: {len(overlap)}")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_retrieval_rows(rows):
    queries, qrels = {}, {}
    for row in rows:
        queries[row["query_id"]] = {"id": row["query_id"], "text": row["query"], "group_id": row["group_id"]}
        key = (row["query_id"], row["positive_id"])
        qrels[key] = {"query_id": row["query_id"], "corpus_id": row["positive_id"], "relevance": 1}
    return [queries[key] for key in sorted(queries)], [qrels[key] for key in sorted(qrels)]


def read_aihub_qa(path):
    """Resolve AI Hub answer.source against the document's actual evidence.

The official 71610 schema uses singular ``answer`` and may point to a different
context or a table. Tables are deliberately skipped until a reviewed table
serializer is added. A nested QA is not proof that its parent is the evidence.
Plural SQuAD-style answers without source require an extractive text match.
"""
    source = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    documents = source.get("data") if isinstance(source, dict) else None
    if not isinstance(documents, list):
        raise ValueError(f"{path}: expected data[]; inspect an actual licensed sample before adapting")
    rows, skipped = [], Counter()
    for document in documents:
        doc_id = document.get("doc_id", document.get("id"))
        if doc_id is None or not isinstance(document.get("paragraphs"), list):
            raise ValueError(f"{path}: each document needs doc_id/id and paragraphs[]")
        group_id = "aihub:" + str(doc_id)
        evidence = {}
        table_ids = set()
        for paragraph in document["paragraphs"]:
            context, qas = paragraph.get("context"), paragraph.get("qas")
            if not isinstance(context, str) or not context.strip() or not isinstance(qas, list):
                raise ValueError(f"{path}: expected string context and qas[]")
            context_id = paragraph.get("context_id", paragraph.get("id"))
            positive_id = group_id + ":context:" + (str(context_id) if context_id is not None else digest(normalized(context))[:20])
            paragraph["_positive_id"] = positive_id
            if context_id is not None:
                key = str(context_id)
                if key in evidence:
                    raise ValueError(f"{path}: duplicate evidence id {key!r}")
                evidence[key] = {"kind": "context", "text": context, "positive_id": positive_id}
            # Released labels use null for paragraphs without tables. Context
            # and table IDs can also be equal, so keep their namespaces apart.
            for table in paragraph.get("tbs") or []:
                table_id = table.get("table_id")
                if table_id is not None:
                    key = str(table_id)
                    if key in table_ids:
                        raise ValueError(f"{path}: duplicate table evidence id {key!r}")
                    table_ids.add(key)
        for paragraph in document["paragraphs"]:
            context, qas = paragraph["context"], paragraph["qas"]
            for qa in qas:
                answers = qa.get("answer", qa.get("answers"))
                if qa.get("is_impossible") is True or answers == [] or answers == {}:
                    skipped["unanswerable"] += 1
                    continue
                query = qa.get("question")
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"{path}: nonempty question string required")
                if answers is None:
                    skipped["missing_answer"] += 1
                    continue
                answer_items = answers if isinstance(answers, list) else [answers]
                if not all(isinstance(answer, dict) for answer in answer_items):
                    raise ValueError(f"{path}: answer must be an object or list of objects")
                positives = {}
                skip_reason = None
                if qa.get("qa_type") in (3, "3"):
                    skip_reason = "table_requires_reviewed_serialization"
                for answer in answer_items:
                    source_id = answer.get("source")
                    if source_id is not None and not isinstance(source_id, (str, int)):
                        skip_reason = "unsupported_source_type"
                        break
                    if source_id is not None and str(source_id).strip():
                        if str(source_id) in table_ids:
                            skip_reason = "table_requires_reviewed_serialization"
                            break
                        item = evidence.get(str(source_id))
                        if item is None:
                            skip_reason = "unresolved_answer_source"
                            break
                    else:
                        # SQuAD-style plural annotations can prove an extractive
                        # relation to their parent. Singular 71610 must name source.
                        answer_text = answer.get("text")
                        if "answer" in qa or not isinstance(answer_text, str) or not answer_text.strip():
                            skip_reason = "missing_answer_source"
                            break
                        if normalized(answer_text) not in normalized(context):
                            skip_reason = "answer_not_found_in_parent_context"
                            break
                        item = {"text": context, "positive_id": paragraph["_positive_id"]}
                    positives[item["positive_id"]] = item["text"]
                if skip_reason:
                    skipped[skip_reason] += 1
                    continue
                qid = qa.get("id", qa.get("question_id"))
                query_id = group_id + ":query:" + (str(qid) if qid is not None else digest(normalized(query))[:20])
                for positive_id, positive in positives.items():
                    rows.append({"query": query, "positive": positive, "query_id": query_id,
                                 "positive_id": positive_id, "group_id": group_id})
    return rows, dict(skipped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical JSONL or directory of JSONL; QA format accepts JSON")
    parser.add_argument("--format", choices=["canonical", "aihub-qa"], default="canonical")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--corpus", help="Optional already frozen corpus JSONL; positives must exist, never injected")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=.1)
    parser.add_argument("--test-fraction", type=float, default=.1)
    parser.add_argument("--smoke", action="store_true", help="Explicitly mark synthetic/runtime inputs")
    args = parser.parse_args()
    source = Path(args.input)
    pattern = "*.jsonl" if args.format == "canonical" else "*.json"
    files = sorted(source.rglob(pattern)) if source.is_dir() else [source]
    if not files:
        parser.error("No input files found")
    rows, skipped = [], Counter()
    for path in files:
        if args.format == "canonical":
            rows.extend(read_jsonl(path))
        else:
            converted, ignored = read_aihub_qa(path)
            rows.extend(converted)
            skipped.update(ignored)
    splits = split_rows(rows, args.seed, args.dev_fraction, args.test_fraction)
    all_rows = [row for name in ("train", "dev", "test") for row in splits[name]]
    if args.corpus:
        corpus = load_collection(args.corpus)
        candidates = {row["id"]: normalized(row["text"]) for row in corpus}
        for row in all_rows:
            if candidates.get(row["positive_id"]) != normalized(row["positive"]):
                raise ValueError(f"Supplied corpus lacks matching positive id/text: {row['positive_id']}")
        corpus_origin = "external_frozen_corpus"
    else:
        docs = {row["positive_id"]: {"id": row["positive_id"], "text": row["positive"], "group_id": row["group_id"]}
                for row in all_rows}
        corpus = [docs[key] for key in sorted(docs)]
        corpus_origin = "global_dataset_passage_pool_not_production_corpus"
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to preserve frozen splits; choose a new output path")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus)
    for name, data in splits.items():
        write_jsonl(output / f"{name}.jsonl", data)
        if name in ("dev", "test"):
            queries, qrels = make_retrieval_rows(data)
            write_jsonl(output / name / "queries.jsonl", queries)
            write_jsonl(output / name / "qrels.jsonl", qrels)
    manifest = {"schema_version": 1, "purpose": "synthetic_smoke_only" if args.smoke else "legal_retrieval",
                "input_hash": content_hash(rows), "input_files": [str(path) for path in files],
                "seed": args.seed, "requested_group_fractions": {"dev": args.dev_fraction, "test": args.test_fraction},
                "corpus_origin": corpus_origin, "corpus_hash": content_hash(corpus), "corpus_size": len(corpus),
                "skipped_by_reason": dict(skipped),
                "split_counts": {name: {"pairs": len(data), "groups": len({r['group_id'] for r in data}),
                                        "queries": len({r['query_id'] for r in data})} for name, data in splits.items()},
                "leakage_checks": ["source group", "positive id", "query id", "normalized document text", "normalized query text"],
                "limitations": ["Exact text checks do not detect paraphrase or case/version families omitted from group_id.",
                                "Document-group splitting does not create a chronological or external test set.",
                                "Qrels may be incomplete; unjudged relevant passages require human review.",
                                "Use dev for selection; preserve test for final comparison, never for tuning."]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
