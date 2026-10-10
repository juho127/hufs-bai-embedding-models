"""Evaluate exact cosine retrieval against an independently frozen corpus.

Corpus/query JSONL: {"id": "...", "text": "...", "group_id": "..." (optional)}.
Qrels JSONL: {"query_id": "...", "corpus_id": "...", "relevance": 1}.
No gold document is ever added to the supplied candidate corpus by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path


METRICS = ("ndcg@10", "mrr@10", "recall@10", "recall@50")


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected object")
                    rows.append(row)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON object") from exc
    return rows


def content_hash(rows):
    """Hash the ordered semantic input, independent of JSON whitespace."""
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checkpoint_fingerprint(model_name_or_path, *, revision=None, resolved_revision=None):
    """Hash checkpoint content, including any locally cached adapter base.

    This never downloads a missing dependency. An unresolvable fingerprint
    disables the optional cache rather than guessing that a model name is an
    immutable identity. File size/mtime alone are never accepted as identity.
    """
    visited = set()

    def inspect(source, requested_revision):
        root = Path(source)
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(str(source), revision=requested_revision,
                                          local_files_only=True))
        root = root.resolve()
        if root in visited:
            raise ValueError(f"Cyclic checkpoint dependency: {root}")
        visited.add(root)
        manifest, dependencies = [], []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
                continue
            before = path.stat()
            digest = file_hash(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Checkpoint changed while hashing: {path}")
            manifest.append({"path": path.relative_to(root).as_posix(),
                             "bytes": before.st_size, "sha256": digest})
            if path.name == "adapter_config.json":
                adapter = json.loads(path.read_text(encoding="utf-8"))
                base = adapter.get("base_model_name_or_path")
                if not base:
                    raise ValueError(f"Adapter base is unspecified: {path}")
                # A relative local base may be relative to the adapter folder.
                if not Path(base).is_dir() and (path.parent / base).is_dir():
                    base = str(path.parent / base)
                dependencies.append(inspect(base, adapter.get("revision") or resolved_revision))
        if not manifest:
            raise ValueError(f"Empty checkpoint: {root}")
        visited.remove(root)
        return {"files": manifest, "dependencies": dependencies}

    manifest = inspect(model_name_or_path, resolved_revision or revision)
    return {"sha256": content_hash(manifest), "manifest": manifest}


def valid_embeddings(embeddings, rows, dimension):
    import numpy as np
    values = np.asarray(embeddings)
    if values.shape != (rows, dimension) or values.dtype != np.dtype("float32"):
        raise ValueError(f"Invalid embedding shape/dtype: {values.shape}/{values.dtype}")
    if not np.isfinite(values).all() or np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Nonfinite or zero-length cached embeddings")
    return values


def cached_corpus_embeddings(cache_dir, identity, rows, dimension, encode):
    """Reuse verified arrays; cache I/O failure never discards a fresh encoding.

    Both files are replaced atomically, with metadata last as the commit record.
    Its array checksum rejects stale, incomplete or interleaved writes. Cache
    failures are surfaced in the returned report and as stderr warnings.
    """
    import numpy as np
    key = content_hash(identity)
    directory = Path(cache_dir)
    array_path, metadata_path = directory / f"{key}.npy", directory / f"{key}.json"
    status = {"enabled": True, "key": key, "status": "miss", "directory": str(directory)}
    try:
        if array_path.exists() or metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (metadata.get("schema_version") != 1 or metadata.get("key") != key
                    or metadata.get("identity") != identity
                    or metadata.get("shape") != [rows, dimension]
                    or metadata.get("dtype") != "float32"):
                raise ValueError("Cache metadata does not match the encoding request")
            if file_hash(array_path) != metadata.get("array_sha256"):
                raise ValueError("Cache array checksum mismatch")
            values = valid_embeddings(np.load(array_path, allow_pickle=False), rows, dimension)
            status["status"] = "hit"
            return values, status
    except Exception as exc:
        status["read_error"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Ignoring invalid embedding cache: {exc}", RuntimeWarning)
    # Encoder errors must propagate. Only optional cache I/O is best-effort.
    values = valid_embeddings(np.asarray(encode(), dtype=np.float32), rows, dimension)
    temporary = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{key}.", suffix=".npy",
                                         delete=False) as stream:
            temp_array = Path(stream.name)
            temporary.append(temp_array)
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        metadata = {"schema_version": 1, "key": key, "identity": identity,
                    "shape": [rows, dimension], "dtype": "float32",
                    "array_sha256": file_hash(temp_array),
                    "created_at": datetime.now(timezone.utc).isoformat()}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=f".{key}.", suffix=".json", delete=False) as stream:
            temp_metadata = Path(stream.name)
            temporary.append(temp_metadata)
            json.dump(metadata, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_array, array_path)
        os.replace(temp_metadata, metadata_path)
        status["status"] = "recomputed" if "read_error" in status else "stored"
    except Exception as exc:
        status["status"] = "write_failed"
        status["write_error"] = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Could not save optional embedding cache; evaluation continues: {exc}", RuntimeWarning)
    finally:
        for path in temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    return values, status


def load_collection(path):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty collection: {path}")
    found = set()
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"Every id must be a nonempty string: {path}")
        if row["id"] in found:
            raise ValueError(f"Duplicate id {row['id']!r} in {path}")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError(f"Empty text for {row['id']!r} in {path}")
        found.add(row["id"])
    return rows


def load_qrels(path, queries, corpus):
    query_ids = {row["id"] for row in queries}
    corpus_ids = {row["id"] for row in corpus}
    qrels = {qid: {} for qid in query_ids}
    for row in read_jsonl(path):
        qid, did = row.get("query_id"), row.get("corpus_id")
        rel = row.get("relevance")
        if qid not in query_ids or did not in corpus_ids:
            raise ValueError(f"Qrel references missing query/corpus id: {row}")
        if not isinstance(rel, (int, float)) or not math.isfinite(rel) or rel < 0:
            raise ValueError(f"Relevance must be finite and nonnegative: {row}")
        if did in qrels[qid] and qrels[qid][did] != rel:
            raise ValueError(f"Conflicting relevance labels: {row}")
        qrels[qid][did] = rel
    missing = sorted(qid for qid, rels in qrels.items() if not any(v > 0 for v in rels.values()))
    if missing:
        raise ValueError(f"Queries without a positive qrel (not silently omitted): {missing[:10]}")
    return qrels


def score_ranking(ranked_ids, qrels):
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("Ranking contains duplicate corpus ids")
    positives = {did for did, grade in qrels.items() if grade > 0}
    if not positives:
        raise ValueError("At least one positive qrel is required")
    ideal_grades = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)[:10]
    # Graded gain follows the common 2**relevance - 1 convention.
    ideal = sum((2.0 ** grade - 1.0) / math.log2(rank + 2) for rank, grade in enumerate(ideal_grades))
    dcg = sum((2.0 ** qrels.get(did, 0) - 1.0) / math.log2(rank + 2)
              for rank, did in enumerate(ranked_ids[:10]))
    rr = next((1.0 / rank for rank, did in enumerate(ranked_ids[:10], 1) if did in positives), 0.0)
    return {"ndcg@10": dcg / ideal, "mrr@10": rr,
            "recall@10": len(positives.intersection(ranked_ids[:10])) / len(positives),
            "recall@50": len(positives.intersection(ranked_ids[:50])) / len(positives)}


def exact_topk(query_embeddings, corpus_embeddings, k=50, query_chunk_size=32, corpus_chunk_size=8192):
    """Chunked full-corpus dot product on normalized vectors; ties use corpus order."""
    import numpy as np
    q = np.asarray(query_embeddings, dtype=np.float32)
    c = np.asarray(corpus_embeddings, dtype=np.float32)
    if q.ndim != 2 or c.ndim != 2 or q.shape[1] != c.shape[1] or not len(c):
        raise ValueError("Expected nonempty compatible 2-D embedding matrices")
    if not np.isfinite(q).all() or not np.isfinite(c).all():
        raise ValueError("Nonfinite embeddings")
    qnorm, cnorm = np.linalg.norm(q, axis=1, keepdims=True), np.linalg.norm(c, axis=1, keepdims=True)
    if np.any(qnorm == 0) or np.any(cnorm == 0):
        raise ValueError("Zero-length embeddings")
    q, c = q / qnorm, c / cnorm
    if min(k, query_chunk_size, corpus_chunk_size) < 1:
        raise ValueError("Chunk sizes and k must be positive")
    k = min(k, len(c))
    all_ids, all_scores = [], []
    for qstart in range(0, len(q), query_chunk_size):
        block = q[qstart:qstart + query_chunk_size]
        best_ids = np.empty((len(block), 0), dtype=np.int64)
        best_scores = np.empty((len(block), 0), dtype=np.float32)
        for cstart in range(0, len(c), corpus_chunk_size):
            scores = block @ c[cstart:cstart + corpus_chunk_size].T
            ids = np.broadcast_to(np.arange(cstart, cstart + scores.shape[1]), scores.shape)
            scores = np.concatenate((best_scores, scores), axis=1)
            ids = np.concatenate((best_ids, ids), axis=1)
            # Secondary stable corpus index makes identical-score ties reproducible.
            order = np.lexsort((ids, -scores), axis=1)[:, :k]
            best_ids = np.take_along_axis(ids, order, axis=1)
            best_scores = np.take_along_axis(scores, order, axis=1)
        all_ids.extend(best_ids.tolist())
        all_scores.extend(best_scores.tolist())
    return all_ids, all_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--output", required=True, help="Report JSON; rankings also written beside it")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32",
                        help="Use float32 for all three methods in a controlled quality comparison")
    parser.add_argument("--autocast", choices=["bfloat16", "none"], default="bfloat16")
    parser.add_argument("--attention", choices=["sdpa", "eager", "flash_attention_2"], default="sdpa")
    parser.add_argument("--query-chunk-size", type=int, default=32)
    parser.add_argument("--corpus-chunk-size", type=int, default=8192)
    parser.add_argument("--embedding-cache-dir",
                        help="Optional verified document-embedding cache shared across dev/test evaluations")
    parser.add_argument("--smoke", action="store_true", help="Label this synthetic/software check, not quality evidence")
    args = parser.parse_args()
    if min(args.batch_size, args.max_seq_length, args.query_chunk_size, args.corpus_chunk_size) < 1:
        parser.error("Lengths, batch size and chunk sizes must be positive")
    corpus, queries = load_collection(args.corpus), load_collection(args.queries)
    qrels = load_qrels(args.qrels, queries, corpus)
    import torch
    import sentence_transformers
    from model_loading import load_sentence_transformer
    start = time.perf_counter()
    model, loading_metadata = load_sentence_transformer(
        args.model, revision=args.revision, device=args.device,
        dtype=getattr(torch, args.dtype), attention=args.attention)
    model.max_seq_length = args.max_seq_length
    model.eval()
    cuda = args.device.startswith("cuda")
    use_autocast = cuda and args.autocast == "bfloat16"
    if use_autocast and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 autocast is unavailable")
    kwargs = dict(batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=True,
                  convert_to_numpy=True)
    corpus_hash = content_hash(corpus)
    cache_identity = None
    cache_status = {"enabled": bool(args.embedding_cache_dir), "status": "disabled"}
    if args.embedding_cache_dir:
        try:
            from importlib.metadata import PackageNotFoundError, version
            versions = {}
            for name in ("torch", "sentence-transformers", "transformers", "peft", "tokenizers", "numpy"):
                try:
                    versions[name] = version(name)
                except PackageNotFoundError:
                    versions[name] = None
            config = model[0].auto_model.config
            tokenizer = model.tokenizer
            cache_identity = {"schema_version": 1,
                "checkpoint": checkpoint_fingerprint(args.model, revision=args.revision,
                    resolved_revision=getattr(config, "_commit_hash", None)),
                "config": config.to_dict(), "modules": str(model),
                "tokenizer": {"class": type(tokenizer).__name__, "init_kwargs": tokenizer.init_kwargs,
                              "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side},
                "corpus_hash": corpus_hash, "corpus_size": len(corpus),
                "dimension": model.get_sentence_embedding_dimension(), "prompts": model.prompts,
                "default_prompt_name": getattr(model, "default_prompt_name", None),
                "max_seq_length": args.max_seq_length, "batch_size": args.batch_size,
                "requested_load_dtype": args.dtype,
                "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                "autocast": "bfloat16" if use_autocast else "none", "attention": args.attention,
                "document_api": "encode_document", "normalize_embeddings": True,
                "versions": versions, "python": platform.python_version(), "device": args.device,
                "cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(args.device) if cuda else platform.processor(),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
            # Configs may contain torch dtypes or tokenizer AddedToken objects.
            cache_identity = json.loads(json.dumps(cache_identity, ensure_ascii=False, default=str))
        except Exception as exc:
            cache_identity = None
            cache_status.update(status="fingerprint_failed", error=f"{type(exc).__name__}: {exc}")
            warnings.warn(f"Optional embedding cache disabled; encoding documents afresh: {exc}", RuntimeWarning)
    with torch.autocast(device_type=torch.device(args.device).type,
                        dtype=torch.bfloat16, enabled=use_autocast):
        if cache_identity is None:
            corpus_embeddings = model.encode_document([row["text"] for row in corpus], **kwargs)
        else:
            corpus_embeddings, cache_status = cached_corpus_embeddings(
                args.embedding_cache_dir, cache_identity, len(corpus),
                model.get_sentence_embedding_dimension(),
                lambda: model.encode_document([row["text"] for row in corpus], **kwargs))
        query_embeddings = model.encode_query([row["text"] for row in queries], **kwargs)
    encoding_seconds = time.perf_counter() - start
    top_ids, top_scores = exact_topk(query_embeddings, corpus_embeddings, 50,
                                    args.query_chunk_size, args.corpus_chunk_size)
    per_query, rankings = [], []
    for query, indices, scores in zip(queries, top_ids, top_scores):
        ids = [corpus[i]["id"] for i in indices]
        per_query.append({"query_id": query["id"], "group_id": query.get("group_id", query["id"]),
                          **score_ranking(ids, qrels[query["id"]])})
        rankings.append({"query_id": query["id"], "hits": [
            {"corpus_id": did, "score": score, "rank": rank,
             "relevance": qrels[query["id"]].get(did, 0)}
            for rank, (did, score) in enumerate(zip(ids, scores), 1)]})
    protocol = {"retrieval": "exact_cosine_full_corpus", "top_k": 50,
                "gain": "2**relevance-1", "unjudged": "nonrelevant",
                "max_seq_length": args.max_seq_length, "tie_break": "corpus_order",
                "query_api": "encode_query", "document_api": "encode_document",
                "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
                "requested_load_dtype": args.dtype,
                "autocast": "bfloat16" if use_autocast else "none",
                "attention": args.attention, "batch_size": args.batch_size,
                "prompts": model.prompts}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "purpose": "synthetic_smoke_only" if args.smoke else "retrieval_evaluation",
              "model": args.model, "requested_revision": args.revision,
              "resolved_revision": getattr(model[0].auto_model.config, "_commit_hash", None),
              "prompts": model.prompts, "protocol": protocol,
              "loading_metadata": loading_metadata,
              "corpus_hash": corpus_hash, "queries_hash": content_hash(queries),
              "qrels_hash": content_hash(qrels), "corpus_size": len(corpus), "query_count": len(queries),
              "metrics": {name: sum(row[name] for row in per_query) / len(per_query) for name in METRICS},
              "per_query": per_query, "encoding_seconds": encoding_seconds,
              "embedding_cache": cache_status,
              "total_seconds": time.perf_counter() - start,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "sentence_transformers": sentence_transformers.__version__,
                              "device": args.device, "dtype": args.dtype, "attention": args.attention,
                              "autocast": "bfloat16" if use_autocast else "none"}}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".rankings.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rankings), encoding="utf-8")
    print(json.dumps({"report": str(output), "purpose": report["purpose"], "metrics": report["metrics"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
