"""Freeze the local licensed AI Hub 71610 legal retrieval experiment.

Read label ZIPs in place, preserve the supplied question/passage text, and join
document families before any train/dev/test selection. Outputs contain licensed
text and belong under ignored data/, never in a public source repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from evaluate import content_hash, load_collection, load_qrels
from prepare_data import (UnionFind, assert_no_leakage, digest, make_retrieval_rows,
                          normalized, write_jsonl)


TRAIN_TYPES = {1, 5}
CORPUS_TYPES = {1, 2, 4, 5}
VERSION_SUFFIX = re.compile(r"\s*[\[(]\s*(?:개정판|수정판|증보판|제\s*\d+\s*판)\s*[\])]\s*$")


def stable_id(prefix, value):
    return prefix + digest(normalized(value))


def version_title(title):
    """Only explicit edition suffixes; preserve years, statute numbers and dates."""
    return VERSION_SUFFIX.sub("", normalized(title)).strip()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_quality_reviews(reports_dir):
    root = Path(reports_dir)
    group_path = root / "aihub71610_quality_reviewed_groups.json"
    group_report = json.loads(group_path.read_text(encoding="utf-8"))
    reviewed = {str(item["doc_id"]) for item in group_report["groups"]}
    excluded, inputs = {}, [{"file": group_path.name, "sha256": file_hash(group_path)}]
    for name in ("aihub71610_quality_extractive_review.json", "aihub71610_quality_yesno_mc_review.json"):
        path = root / name
        report = json.loads(path.read_text(encoding="utf-8"))
        inputs.append({"file": name, "sha256": file_hash(path)})
        for item in report.get("items", report.get("rows", [])):
            if item["retrieval_pair"] in ("conditional", "reject"):
                excluded[(str(item["doc_id"]), str(item["question_id"]))] = item["retrieval_pair"]
    return reviewed, excluded, inputs


class ReleaseBuilder:
    def __init__(self, reviewed_docs=(), quality_exclusions=None):
        self.reviewed_docs = set(reviewed_docs)
        self.quality_exclusions = quality_exclusions or {}
        self.docs = {}
        self.passages = {}
        self.query_texts = {}
        self.pairs = {}
        self.excluded_pairs = set()
        self.exclusion_records = []
        self.counts = Counter()
        self.original_id_texts = defaultdict(set)

    def add_document(self, document, original_split):
        domain = document.get("doc_class", {}).get("class", "")
        if not domain.startswith("법률/"):
            self.counts["excluded_nonlegal_document_records"] += 1
            return
        doc_id = str(document["doc_id"])
        meta = self.docs.setdefault(doc_id, {"titles": set(), "sources": set(), "passages": set(), "queries": set()})
        title, source = normalized(document.get("doc_title") or ""), normalized(document.get("doc_source") or "")
        if title:
            meta["titles"].add(title)
        if source:
            meta["sources"].add(source)
        evidence, tables = {}, set()
        for paragraph in document["paragraphs"]:
            text = paragraph.get("context")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty context: {doc_id}")
            cid = str(paragraph["context_id"])
            pid = stable_id("passage-", text)
            if cid in evidence and evidence[cid][0] != pid:
                raise ValueError(f"Conflicting within-record context id: {doc_id}/{cid}")
            evidence[cid] = (pid, text)
            for table in paragraph.get("tbs") or []:
                if table.get("table_id") is not None:
                    tables.add(str(table["table_id"]))
            qa_types = {int(qa["qa_type"]) for qa in paragraph["qas"]}
            if qa_types & CORPUS_TYPES and cid not in tables:
                item = self.passages.setdefault(pid, {"text": text, "doc_ids": set(), "qa_types": set()})
                if normalized(item["text"]) != normalized(text):
                    raise ValueError("Passage hash collision")
                item["doc_ids"].add(doc_id)
                item["qa_types"].update(qa_types & CORPUS_TYPES)
                meta["passages"].add(pid)
                self.original_id_texts[("passage", doc_id, cid)].add(pid)
        for paragraph in document["paragraphs"]:
            for qa in paragraph["qas"]:
                typ = int(qa["qa_type"])
                self.counts[f"legal_qa_type_{typ}"] += 1
                query = qa.get("question")
                if typ not in CORPUS_TYPES:
                    self.counts["excluded_table_qa"] += 1
                    continue
                if not isinstance(query, str) or not query.strip():
                    raise ValueError(f"Empty question: {doc_id}")
                qid = stable_id("query-", query)
                representative = self.query_texts.setdefault(qid, query)
                if normalized(representative) != normalized(query):
                    raise ValueError("Query hash collision")
                source_qid = str(qa.get("question_id", qa.get("id", "")))
                meta["queries"].add(qid)
                self.original_id_texts[("query", doc_id, source_qid)].add(qid)
                if typ not in TRAIN_TYPES:
                    self.counts["excluded_training_qa_type_2_or_4"] += 1
                    continue
                answer = qa.get("answer")
                answers = answer if isinstance(answer, list) else [answer]
                if not answers or any(not isinstance(a, dict) for a in answers):
                    raise ValueError(f"Missing answer: {doc_id}/{source_qid}")
                for answer in answers:
                    source_id = str(answer.get("source", ""))
                    if source_id in tables:
                        raise ValueError(f"Selected non-table question points to table: {doc_id}/{source_qid}")
                    if source_id not in evidence:
                        raise ValueError(f"Unresolved answer.source: {doc_id}/{source_qid}")
                    pid, positive = evidence[source_id]
                    if pid not in self.passages:
                        raise ValueError(f"Positive missing from corpus: {doc_id}/{source_qid}")
                    key = (qid, pid)
                    row = self.pairs.setdefault(key, {"query": query, "positive": positive, "query_id": qid,
                        "positive_id": pid, "doc_ids": set(), "qa_types": set(), "source_refs": set()})
                    row["doc_ids"].add(doc_id)
                    row["qa_types"].add(typ)
                    row["source_refs"].add((doc_id, source_qid, original_split, typ))
                    self.counts["selected_type_qa_before_dedup_and_review"] += 1
                    if (doc_id, source_qid) in self.quality_exclusions:
                        self.excluded_pairs.add(key)
                        self.exclusion_records.append({"doc_id": doc_id, "question_id": source_qid,
                            "query_id": qid, "positive_id": pid,
                            "review_result": self.quality_exclusions[(doc_id, source_qid)]})

    def group_documents(self):
        doc_ids = sorted(self.docs)
        uf, seen, joins = UnionFind(len(doc_ids)), {}, Counter()
        for index, doc_id in enumerate(doc_ids):
            meta = self.docs[doc_id]
            keys = [("title", title) for title in meta["titles"]]
            keys += [("passage", pid) for pid in meta["passages"]]
            keys += [("query", qid) for qid in meta["queries"]]
            keys += [("explicit_edition", source + "\0" + version_title(title))
                     for title in meta["titles"] for source in meta["sources"]]
            for key in keys:
                if key in seen:
                    if uf.find(index) != uf.find(seen[key]):
                        joins[key[0]] += 1
                    uf.union(index, seen[key])
                else:
                    seen[key] = index
        components = defaultdict(list)
        for index, doc_id in enumerate(doc_ids):
            components[uf.find(index)].append(doc_id)
        mapping, families = {}, {}
        for ids in components.values():
            gid = "family-" + digest(json.dumps(ids, ensure_ascii=False))
            families[gid] = ids
            mapping.update({did: gid for did in ids})
        self.family_join_counts = dict(joins)
        return mapping, families

    def materialize(self):
        mapping, families = self.group_documents()
        reviewed_families = {mapping[doc] for doc in self.reviewed_docs if doc in mapping}
        rows = []
        for key in sorted(self.pairs):
            if key in self.excluded_pairs:
                continue
            source = self.pairs[key]
            groups = {mapping[doc] for doc in source["doc_ids"]}
            if len(groups) != 1:
                raise ValueError("Duplicate pair spans unjoined document families")
            row = {key: source[key] for key in ("query", "positive", "query_id", "positive_id")}
            row["query"] = self.query_texts[row["query_id"]]
            row["positive"] = self.passages[row["positive_id"]]["text"]
            self.counts["canonical_positive_representation_reused"] += row["positive"] != source["positive"]
            self.counts["canonical_query_representation_reused"] += row["query"] != source["query"]
            row.update(group_id=next(iter(groups)), qa_type=min(source["qa_types"]),
                       qa_types=sorted(source["qa_types"]), source_doc_ids=sorted(source["doc_ids"]))
            rows.append(row)
        corpus = []
        for pid in sorted(self.passages):
            passage = self.passages[pid]
            groups = {mapping[doc] for doc in passage["doc_ids"]}
            if len(groups) != 1:
                raise ValueError("Duplicate corpus passage spans document families")
            corpus.append({"id": pid, "text": passage["text"], "group_id": next(iter(groups))})
        return rows, corpus, mapping, families, reviewed_families


def split_families(rows, reviewed_families, seed=42, dev_fraction=.1, test_fraction=.1):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    ids = sorted(grouped)
    if len(ids) < 3:
        raise ValueError("At least three independent families are required")
    ndev, ntest = max(1, round(len(ids) * dev_fraction)), max(1, round(len(ids) * test_fraction))
    if ndev + ntest >= len(ids):
        raise ValueError("Too few families for requested fractions")
    test_eligible = sorted(set(ids) - set(reviewed_families))
    random.Random(seed).shuffle(test_eligible)
    if len(test_eligible) < ntest:
        raise ValueError("Insufficient untouched families for test")
    test_ids = set(test_eligible[:ntest])
    remaining = sorted(set(ids) - test_ids)
    random.Random(seed + 1).shuffle(remaining)
    dev_ids = set(remaining[:ndev])
    splits = {"train": [], "dev": [], "test": []}
    for gid in ids:
        name = "test" if gid in test_ids else "dev" if gid in dev_ids else "train"
        splits[name].extend(grouped[gid])
    for part in splits.values():
        part.sort(key=lambda row: (row["query_id"], row["positive_id"]))
    assert_no_leakage(splits)
    if test_ids & set(reviewed_families):
        raise ValueError("Reviewed family leaked into test")
    return splits


def select_evaluation_queries(rows, budget, seed):
    """Proportional question types, cycling document families before repetition."""
    queries = defaultdict(list)
    for row in rows:
        queries[row["query_id"]].append(row)
    if len(queries) <= budget:
        return list(rows)
    by_type = defaultdict(lambda: defaultdict(list))
    for qid in sorted(queries):
        sample = queries[qid][0]
        by_type[sample["qa_type"]][sample["group_id"]].append(qid)
    totals = {typ: sum(map(len, groups.values())) for typ, groups in by_type.items()}
    quotas = {typ: int(budget * total / len(queries)) for typ, total in totals.items()}
    for typ in sorted(totals, key=lambda typ: (-(budget * totals[typ] / len(queries) - quotas[typ]), typ))[:budget - sum(quotas.values())]:
        quotas[typ] += 1
    rng, selected, globally_seen_groups = random.Random(seed), set(), set()
    for typ in sorted(by_type, key=lambda typ: (totals[typ], typ)):
        groups = by_type[typ]
        group_ids = sorted(groups)
        rng.shuffle(group_ids)
        group_ids.sort(key=lambda gid: gid in globally_seen_groups)
        for values in groups.values():
            rng.shuffle(values)
        remaining = quotas[typ]
        while remaining:
            progressed = False
            for gid in group_ids:
                if groups[gid] and remaining:
                    selected.add(groups[gid].pop())
                    globally_seen_groups.add(gid)
                    remaining -= 1
                    progressed = True
            if not progressed:
                raise ValueError("Unable to satisfy evaluation sample quotas")
    return [row for row in rows if row["query_id"] in selected]


def split_counts(rows):
    queries = {}
    for row in rows:
        queries[row["query_id"]] = row
    return {"pairs": len(rows), "queries": len(queries), "groups": len({row["group_id"] for row in rows}),
            "query_types": dict(sorted(Counter(row["qa_type"] for row in queries.values()).items()))}


def write_retrieval(output, name, rows):
    write_jsonl(output / f"{name}.jsonl", rows)
    queries, qrels = make_retrieval_rows(rows)
    write_jsonl(output / name / "queries.jsonl", queries)
    write_jsonl(output / name / "qrels.jsonl", qrels)


def prepare_release(input_dir, output_dir, reports_dir, seed=42, dev_queries=500, test_queries=1000):
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Frozen output already exists; choose a new release directory")
    reviewed, exclusions, review_inputs = load_quality_reviews(reports_dir)
    builder, inputs = ReleaseBuilder(reviewed, exclusions), []
    files = [p for p in sorted(Path(input_dir).rglob("*.zip")) if p.name.startswith(("TL_", "VL_"))]
    if len(files) != 10:
        raise ValueError(f"Expected 10 released label ZIPs, found {len(files)}")
    for path in files:
        split = "Training" if path.name.startswith("TL_") else "Validation"
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".json")]
            if len(members) != 1:
                raise ValueError(f"Expected one JSON member: {path}")
            data = json.loads(archive.read(members[0]))
        inputs.append({"file": path.name, "sha256": file_hash(path), "member": members[0]})
        for document in data["data"]:
            builder.add_document(document, split)
    rows, corpus, doc_families, families, reviewed_families = builder.materialize()
    splits = split_families(rows, reviewed_families, seed)
    selected = {"dev": select_evaluation_queries(splits["dev"], dev_queries, seed + 100),
                "test": select_evaluation_queries(splits["test"], test_queries, seed + 200)}
    # All corpus ownership is checked against positive families, including types
    # used solely as distractors. Corpus itself is global across all splits.
    corpus_groups = {row["id"]: row["group_id"] for row in corpus}
    corpus_texts = {row["id"]: row["text"] for row in corpus}
    for row in rows:
        if corpus_groups[row["positive_id"]] != row["group_id"]:
            raise ValueError("Positive and corpus family mismatch")
        if corpus_texts[row["positive_id"]] != row["positive"]:
            raise ValueError("Training and evaluation use different positive text representatives")
    source_owners = {}
    for name, part in splits.items():
        for row in part:
            for doc_id in families[row["group_id"]]:
                if doc_id in source_owners and source_owners[doc_id] != name:
                    raise ValueError("Source document appears in multiple splits")
                source_owners[doc_id] = name
    sizes = Counter(row["group_id"] for row in rows)
    largest = [{"group_id": gid, "pairs": size, "documents": len(families[gid]),
                "reviewed": gid in reviewed_families} for gid, size in sizes.most_common(10)]
    if largest and largest[0]["pairs"] > len(rows) * .25:
        raise ValueError(f"Large connected component needs review before splitting: {largest[0]}")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "corpus.jsonl", corpus)
    write_jsonl(output / "train.jsonl", splits["train"])
    for name in ("dev", "test"):
        write_retrieval(output, name + "_pool", splits[name])
        write_retrieval(output, name, selected[name])
    # This audit file contains IDs only. All raw text remains in ignored data/.
    audit = {"document_to_family": doc_families,
             "reviewed_document_ids": sorted(reviewed),
             "reviewed_family_ids": sorted(reviewed_families),
             "excluded_quality_records": sorted(builder.exclusion_records, key=lambda r: (r["doc_id"], r["question_id"]))}
    (output / "lineage.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "release": "aihub71610_legal_v1", "purpose": "legal_retrieval",
        "seed": seed, "input_files": inputs, "quality_review_inputs": review_inputs,
        "source_domain": "doc_class.class startswith 법률/", "training_qa_types": sorted(TRAIN_TYPES),
        "corpus_qa_types": sorted(CORPUS_TYPES), "source_qa_counts": dict(sorted(builder.counts.items())),
        "selected_pairs_before_review": len(builder.pairs), "excluded_quality_pairs": len(builder.excluded_pairs),
        "selected_pairs_after_review": len(rows), "id_scheme": "SHA256 of NFKC+whitespace-normalized text; original text unchanged",
        "text_representative_policy": "First original text per normalized text ID in sorted input order, shared exactly by train pairs and frozen corpus; no text rewrite",
        "normalized_id_collision_check_passed": True,
        "conflicting_original_ids_safely_replaced": sum(len(v) > 1 for v in builder.original_id_texts.values()),
        "input_hash": content_hash(rows), "corpus_origin": "frozen_global_all_legal_non_table_label_passages",
        "corpus_size": len(corpus), "corpus_hash": content_hash(corpus),
        "document_count": len(doc_families), "document_family_count": len(families),
        "family_join_counts": builder.family_join_counts, "largest_components": largest,
        "requested_group_fractions": {"train": .8, "dev": .1, "test": .1},
        "split_counts": {name: split_counts(part) for name, part in splits.items()},
        "evaluation_selection": {"method": "type-proportional, family-cycling, text- and model-independent fixed seed",
                                 "dev": split_counts(selected["dev"]), "test": split_counts(selected["test"]),
                                 "seed_dev": seed + 100, "seed_test": seed + 200},
        "reviewed_source_documents": len(reviewed), "reviewed_families": len(reviewed_families),
        "test_reviewed_family_overlap": 0, "test_model_scores_seen": False,
        "test_selection_frozen_before_model_evaluation": True,
        "input_text_policy": "Original question and full label answer.source passage unchanged; no heuristic semantic rewrite/removal",
        "quality_exclusion_policy": "Only already-reviewed conditional/reject selected pairs excluded, with IDs in lineage.json; reviewed families barred from test",
        "grouping_policy": ["same source doc_id", "exact normalized nonempty title across sources",
                            "same normalized non-table passage", "same normalized non-table question",
                            "same source + exact title after removal of explicit terminal edition marker only"],
        "leakage_checks": ["connected family", "all source doc IDs", "positive ID", "query ID", "normalized positive text", "normalized query text", "corpus-positive family", "corpus-positive identical text representative", "reviewed family excluded from test"],
        "checks_passed": True,
        "limitations": ["Grouping does not guarantee detection of paraphrases or differently titled legal/case versions.",
                        "Global corpus contains train and held-out passages, but only train pairs are used for gradient updates.",
                        "Annotation qrels may omit other relevant passages; this is source-annotation retrieval evaluation.",
                        "Legal category includes policy and administrative reports; no external case-law benchmark claim.",
                        "Quality review is an LLM sample, not expert legal adjudication; no full-corpus semantic cleaning.",
                        "The test query pool is untouched by model selection, not a chronological or external-domain test.",
                        "Licensed source and derived text remain local/server training data and must not be redistributed as repository files."]}
    manifest["output_files"] = {str(path.relative_to(output)).replace("\\", "/"): {"bytes": path.stat().st_size, "sha256": file_hash(path)}
                                for path in sorted(output.rglob("*")) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded_corpus = load_collection(output / "corpus.jsonl")
    for name in ("dev", "test", "dev_pool", "test_pool"):
        queries = load_collection(output / name / "queries.jsonl")
        load_qrels(output / name / "qrels.jsonl", queries, loaded_corpus)
    print(json.dumps({key: manifest[key] for key in ("release", "corpus_size", "selected_pairs_after_review", "split_counts", "evaluation_selection", "largest_components", "checks_passed")}, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed/aihub71610_legal_v1")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-queries", type=int, default=500)
    parser.add_argument("--test-queries", type=int, default=1000)
    args = parser.parse_args()
    prepare_release(args.input_dir, args.output_dir, args.reports_dir, args.seed, args.dev_queries, args.test_queries)


if __name__ == "__main__":
    main()
