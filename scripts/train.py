#!/usr/bin/env python3
"""Single-GPU LoRA or full training with query-to-document Cached MNRL.

Input JSONL: query, positive, query_id, positive_id, group_id (strings).
group_id must identify a source/evidence family, not an individual QA row.
The synthetic smoke dataset checks execution only, never retrieval quality.
"""
from __future__ import annotations

import argparse
from collections import deque
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import time
import unicodedata


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def read_rows(path: Path) -> list[dict]:
    rows, query_ids, documents = [], {}, {}
    required = ("query", "positive", "query_id", "positive_id", "group_id")
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            for key in required:
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{path}:{line_number}: non-empty string {key!r} required")
            for id_key, text_key, seen in (("query_id", "query", query_ids),
                                           ("positive_id", "positive", documents)):
                value = normalized(row[text_key])
                if row[id_key] in seen and seen[row[id_key]] != value:
                    raise ValueError(f"{path}:{line_number}: {id_key} maps to different text")
                seen[row[id_key]] = value
            row = dict(row)
            row["_query_key"] = normalized(row["query"])
            row["_positive_key"] = normalized(row["positive"])
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("At least two training pairs are required")
    return rows


def safe_batches(rows: list[dict], batch_size: int, seed: int):
    """Yield index lists; no source family, positive ID, or exact text conflicts.

    Conflicts are deferred to later batches. A final incompatible remainder is
    omitted, because a singleton produces no in-batch negatives. Coverage is
    reported by the caller; no text is silently labeled as a known negative.
    """
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    pending = deque(indices)
    while pending:
        batch, groups, positive_ids, texts = [], set(), set(), set()
        for _ in range(len(pending)):
            idx = pending.popleft()
            row = rows[idx]
            row_texts = {row["_query_key"], row["_positive_key"]}
            if (row["group_id"] in groups or row["positive_id"] in positive_ids
                    or not texts.isdisjoint(row_texts)):
                pending.append(idx)
                continue
            batch.append(idx)
            groups.add(row["group_id"])
            positive_ids.add(row["positive_id"])
            texts.update(row_texts)
            if len(batch) == batch_size:
                break
        if len(batch) < 2:
            # This singleton has no compatible partner. Other pending rows may
            # still be mutually compatible (through different conflict keys),
            # so retry the remainder. Randomize order again in the next epoch.
            continue
        yield batch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", default="nvidia/Nemotron-3-Embed-1B-BF16")
    parser.add_argument("--train", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tuning-mode", choices=["lora", "full"], default="lora")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", "--max_steps", type=int, default=0)
    parser.add_argument("--logical-batch", "--logical_batch", type=int, default=128)
    parser.add_argument("--micro-batch", "--micro_batch", type=int, default=8)
    parser.add_argument("--max-seq-length", "--max_seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        settings = json.loads(preliminary.config.read_text(encoding="utf-8-sig"))
        unknown = set(settings) - {a.dest for a in parser._actions}
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        parser.set_defaults(**settings)
    args = parser.parse_args()
    if args.train is None or args.output is None:
        parser.error("--train and --output are required, directly or through --config")
    args.train, args.output = Path(args.train), Path(args.output)
    if args.logical_batch < 2 or args.micro_batch < 1:
        parser.error("logical_batch >= 2 and micro_batch >= 1 are required")
    if args.epochs < 1 or args.max_steps < 0 or args.max_seq_length < 8:
        parser.error("Invalid epochs, max_steps, or max_seq_length")
    if not (0 <= args.warmup_ratio < 1):
        parser.error("warmup_ratio must be in [0, 1)")
    if args.smoke and not 1 <= args.max_steps <= 3:
        parser.error("--smoke requires --max-steps between 1 and 3")
    return args


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def final_parameter_snapshot(trainable, initial):
    """Replace the initial CPU snapshot one tensor at a time; bound scratch RAM."""
    import torch

    final, updated, largest = {}, 0, 0.0
    for name, parameter in trainable.items():
        current = parameter.detach().to(device="cpu", copy=True)
        previous = initial.pop(name)
        flat_current, flat_previous = current.reshape(-1), previous.reshape(-1)
        tensor_delta = 0.0
        for start in range(0, current.numel(), 1_048_576):
            part = flat_current[start:start + 1_048_576]
            if not torch.isfinite(part).all():
                raise FloatingPointError(f"Nonfinite trained parameter: {name}")
            tensor_delta = max(tensor_delta, (part - flat_previous[start:start + 1_048_576]).abs().max().item())
        updated += int(tensor_delta > 0)
        largest = max(largest, tensor_delta)
        final[name] = current
        del previous, flat_previous
    return final, {"max_abs_update": largest, "updated_tensor_count": updated,
                   "trainable_tensor_count": len(trainable)}


def verify_full_checkpoint(reference, directory):
    """Compare every full-model parameter to safetensors without a second full load."""
    import torch
    from safetensors import safe_open
    from model_loading import canonical_lora_key

    canonical = {canonical_lora_key(name): value for name, value in reference.items()}
    if len(canonical) != len(reference):
        raise RuntimeError("Ambiguous full-model parameter keys")
    files = sorted(Path(directory).rglob("model*.safetensors"))
    if not files:
        raise RuntimeError("Full checkpoint contains no model safetensors")
    seen, extra = set(), []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as stream:
            for stored_key in stream.keys():
                key = canonical_lora_key(stored_key)
                if key not in canonical:
                    extra.append(stored_key)
                    continue
                if key in seen:
                    raise RuntimeError(f"Duplicate full checkpoint parameter: {key}")
                saved = stream.get_tensor(stored_key)
                if saved.dtype != torch.float32 or not torch.equal(canonical[key], saved):
                    raise RuntimeError(f"Full checkpoint changed parameter value or precision: {key}")
                seen.add(key)
                del saved
    missing = set(canonical) - seen
    if missing:
        raise RuntimeError(f"Full checkpoint omitted parameters: {sorted(missing)[:5]}")
    return {"values_exact": True, "tensor_count": len(seen), "dtype": "torch.float32",
            "extra_saved_tensors": extra, "checkpoint_files": [str(file) for file in files],
            "checkpoint_weight_bytes": sum(file.stat().st_size for file in files)}


def main():
    args = parse_args()
    rows = read_rows(args.train)
    batches = list(safe_batches(rows, args.logical_batch, args.seed))
    if not batches:
        raise ValueError("Cannot construct a batch with >=2 distinct evidence groups")
    coverage = sum(map(len, batches))
    if coverage < len(rows) * 0.9 and not args.smoke:
        raise ValueError(f"Only {coverage}/{len(rows)} rows fit safe batches. "
                         "Balance evidence families before training.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "resolved_config.json", vars(args))

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, TaskType
    from sentence_transformers import losses
    from transformers import get_linear_schedule_with_warmup
    from model_loading import (active_adapter_metadata, compare_adapter_states,
                               load_sentence_transformer, local_adapter_state, lora_parameters,
                               canonical_lora_key, PRECISION_FILENAME)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = args.device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This recipe requires CUDA BF16 support")
    dtype = torch.bfloat16 if cuda else torch.float32
    device = torch.device(args.device)
    load_dtype = torch.float32 if args.tuning_mode == "full" else dtype
    model, initial_loading_metadata = load_sentence_transformer(args.model, device=args.device, dtype=load_dtype)
    model.max_seq_length = args.max_seq_length
    backbone = model[0].auto_model
    if getattr(backbone.config, "is_causal", None) is not False:
        raise RuntimeError("Expected Nemotron bidirectional is_causal=false configuration")
    backbone.config.use_cache = False
    for prompt, expected in (("query", "query: "), ("document", "passage: ")):
        if model.prompts.get(prompt) != expected:
            raise RuntimeError(f"Unexpected saved {prompt} prompt: {model.prompts.get(prompt)!r}")
    for module in model:
        if getattr(module, "include_prompt", True) is False:
            raise RuntimeError("This recipe expects mean pooling that includes prefix tokens")
    if lora_parameters(model) or getattr(backbone, "peft_config", None):
        raise RuntimeError("Start from an adapter-free base model for this comparison recipe")
    matched_modules = []
    if args.tuning_mode == "lora":
        linear_names = {name.rsplit(".", 1)[-1] for name, module in backbone.named_modules()
                        if isinstance(module, torch.nn.Linear)}
        missing = set(args.target_modules) - linear_names
        if missing:
            raise RuntimeError(f"Requested LoRA modules do not exist: {sorted(missing)}")
        matched_modules = [name for name, module in backbone.named_modules()
                           if isinstance(module, torch.nn.Linear)
                           and name.rsplit(".", 1)[-1] in args.target_modules]
        model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                                    inference_mode=False, r=args.lora_rank,
                                    lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                    target_modules=args.target_modules, bias="none"))
    else:
        model.float()
        model.requires_grad_(True)
    if args.gradient_checkpointing:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not trainable or (args.tuning_mode == "lora" and any("lora_" not in name for name in trainable)):
        raise RuntimeError("Trainable parameters do not match the selected tuning mode")
    # FP32 trainable parameters and Adam states preserve small updates.
    for parameter in trainable.values():
        parameter.data = parameter.data.float()
    initial = {name: parameter.detach().to(device="cpu", copy=True) for name, parameter in trainable.items()}
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in trainable.values())
    base_parameter_dtype = str(next(parameter.dtype for name, parameter in model.named_parameters()
                                    if "lora_" not in name)).removeprefix("torch.")
    if args.tuning_mode == "full" and trainable_parameters != total_parameters:
        raise RuntimeError("Full tuning requires every parameter to be trainable")
    # Adapter initialization consumes RNG. Training randomness begins from the
    # same seed in both modes; safe_batches already uses an independent RNG.
    torch.manual_seed(args.seed)

    def fp32_cosine(left, right):
        with torch.autocast(device_type=device.type, enabled=False):
            return F.normalize(left.float(), dim=-1) @ F.normalize(right.float(), dim=-1).T

    loss_kwargs = {"mini_batch_size": args.micro_batch, "scale": args.scale,
                   "similarity_fct": fp32_cosine, "show_progress_bar": False}
    if "directions" in inspect.signature(losses.CachedMultipleNegativesRankingLoss).parameters:
        loss_kwargs["directions"] = ("query_to_doc",)
    loss_fn = losses.CachedMultipleNegativesRankingLoss(model, **loss_kwargs)
    parameters = list(trainable.values())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    planned_steps = sum(sum(1 for _ in safe_batches(rows, args.logical_batch, args.seed + epoch))
                        for epoch in range(args.epochs))
    if args.max_steps:
        planned_steps = min(planned_steps, args.max_steps)
    warmup_steps = min(math.ceil(planned_steps * args.warmup_ratio), max(planned_steps - 1, 0))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, planned_steps)
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    step, history, epoch_stats = 0, [], []
    started = time.perf_counter()
    log_path = args.output / "training_steps.jsonl"
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        for epoch in range(args.epochs):
            used = 0
            epoch_batches = list(safe_batches(rows, args.logical_batch, args.seed + epoch))
            for indices in epoch_batches:
                started_step = time.perf_counter()
                selected = [rows[index] for index in indices]
                features = []
                for key, prompt in (("query", "query"), ("positive", "document")):
                    tokenized = model.tokenize([model.prompts[prompt] + row[key] for row in selected])
                    features.append({key: value.to(device) if isinstance(value, torch.Tensor) else value
                                     for key, value in tokenized.items()})
                optimizer.zero_grad(set_to_none=True)
                # CachedMNRL re-embeds inside its backward hook, so both passes
                # must execute under the same autocast context.
                with torch.autocast(device_type=device.type, dtype=dtype, enabled=cuda):
                    loss = loss_fn(features, labels=None)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step + 1}")
                    loss.backward()
                if not any(parameter.grad is not None for parameter in parameters):
                    raise RuntimeError("No trainable-parameter gradients were produced")
                gradient_tensor_count = sum(parameter.grad is not None for parameter in parameters)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                if grad_norm.item() == 0:
                    raise RuntimeError(f"Zero gradient norm at step {step + 1}")
                learning_rate = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                if cuda:
                    torch.cuda.synchronize(device)
                step += 1
                used += len(indices)
                record = {"step": step, "epoch": epoch + 1, "pairs": len(indices),
                          "batch_indices_sha256": hashlib.sha256(
                              json.dumps(indices, separators=(",", ":")).encode("utf-8")).hexdigest(),
                          "negative_candidates_per_query": len(indices) - 1,
                          "query_tokens_max": int(features[0]["attention_mask"].sum(dim=1).max().item()),
                          "document_tokens_max": int(features[1]["attention_mask"].sum(dim=1).max().item()),
                          "loss": loss.item(), "grad_norm": grad_norm.item(),
                          "gradient_tensor_count": gradient_tensor_count,
                          "lr": learning_rate, "seconds": time.perf_counter() - started_step}
                history.append(record)
                log.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)
                del features, tokenized, loss
                if step >= planned_steps:
                    break
            epoch_stats.append({"epoch": epoch + 1, "rows_used": used,
                                "safe_batch_rows_available": sum(map(len, epoch_batches)),
                                "rows_excluded_by_batch_conflicts": len(rows) - sum(map(len, epoch_batches))})
            if step >= planned_steps:
                break
    if step == 0:
        raise RuntimeError("No optimizer step executed")
    peak_memory = torch.cuda.max_memory_allocated(device) if cuda else None
    train_seconds = time.perf_counter() - started
    optimizer_state_dtypes = sorted({str(state[key].dtype) for state in optimizer.state.values()
                                    for key in ("exp_avg", "exp_avg_sq") if key in state})
    if optimizer_state_dtypes != ["torch.float32"]:
        raise RuntimeError(f"AdamW states must be FP32: {optimizer_state_dtypes}")
    final_snapshot, update_stats = final_parameter_snapshot(trainable, initial)
    delta = update_stats["max_abs_update"]
    if not math.isfinite(delta) or delta == 0:
        raise RuntimeError("Weights did not change by a finite non-zero amount")
    # The final snapshot replaces the initial snapshot. Release gradient and
    # Adam storage before encoding/saving/reloading the full model.
    optimizer.zero_grad(set_to_none=True)
    del initial, optimizer, scheduler, loss_fn, parameters
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    model.eval()
    probes = [rows[0]["query"], rows[1]["query"]]
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=cuda):
        expected = model.encode_query(probes, batch_size=2, convert_to_tensor=True,
                                      normalize_embeddings=True, show_progress_bar=False).float().cpu()
    model_path = args.output / "model"
    before_save_active = active_adapter_metadata(model)
    model.save_pretrained(str(model_path))
    precision = {"schema_version": 1, "tuning_mode": args.tuning_mode,
                 "base_parameter_dtype": base_parameter_dtype,
                 "adapter_parameter_dtype": "float32" if args.tuning_mode == "lora" else None,
                 "inference_autocast_dtype": "bfloat16" if cuda else None}
    write_json(model_path / PRECISION_FILENAME, precision)
    canonical_snapshot = {canonical_lora_key(name): value for name, value in final_snapshot.items()}
    if len(canonical_snapshot) != len(final_snapshot):
        raise RuntimeError("Parameter normalization produced duplicate keys")
    if args.tuning_mode == "lora":
        saved_adapters, saved_adapter_files = local_adapter_state(model_path)
        save_verification = compare_adapter_states(canonical_snapshot, saved_adapters)
        del saved_adapters
    else:
        save_verification = verify_full_checkpoint(final_snapshot, model_path)
    if not save_verification["values_exact"]:
        raise RuntimeError(f"Saving changed parameter values: {save_verification}")
    model_checkpoint_bytes = sum(file.stat().st_size for file in model_path.rglob("*") if file.is_file())
    adapter_checkpoint_bytes = sum(file.stat().st_size for file in model_path.rglob("adapter_model.safetensors"))
    # Free the old model before loading another full copy. final_snapshot and
    # canonical_snapshot are two mappings referencing the same CPU tensors.
    del trainable, backbone, model
    gc.collect()
    if cuda:
        torch.cuda.empty_cache()
    reloaded, reload_metadata = load_sentence_transformer(model_path, device=args.device, dtype=dtype)
    reloaded_parameters = (lora_parameters(reloaded) if args.tuning_mode == "lora" else
                           {canonical_lora_key(name): value for name, value in reloaded.named_parameters()})
    exact_reload = compare_adapter_states(canonical_snapshot, reloaded_parameters)
    if not exact_reload["values_exact"]:
        raise RuntimeError(f"Reloaded parameters differ from trained values: {exact_reload}")
    verification_filename = "adapter_verification.json" if args.tuning_mode == "lora" else "weight_verification.json"
    write_json(args.output / verification_filename, {
        "before_save_active": before_save_active, "save": save_verification,
        "reload": exact_reload, "loading": reload_metadata})
    del final_snapshot, canonical_snapshot, reloaded_parameters
    reloaded.eval()
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=cuda):
        actual = reloaded.encode_query(probes, batch_size=2, convert_to_tensor=True,
                                       normalize_embeddings=True, show_progress_bar=False).float().cpu()
    difference = (expected - actual).abs().max().item()
    if not torch.isfinite(actual).all() or not torch.allclose(expected, actual, atol=2e-3, rtol=1e-2):
        raise RuntimeError(f"Saved model reload mismatch: max absolute error={difference}")
    versions = {package: importlib.metadata.version(package) for package in
                ("torch", "transformers", "sentence-transformers", "peft")}
    report = {"status": "passed", "kind": "execution_smoke_only" if args.smoke else "training_run",
              "tuning_mode": args.tuning_mode,
              "retrieval_quality_verified": False, "steps": step, "training_rows": len(rows),
              "loss": "CachedMultipleNegativesRankingLoss", "directions": ["query_to_doc"],
              "logical_batch": args.logical_batch, "micro_batch": args.micro_batch,
              "max_seq_length": args.max_seq_length, "is_causal": False,
              "lora_target_module_count": len(matched_modules), "lora_target_modules": matched_modules,
              "trainable_parameter_count": trainable_parameters, "total_parameter_count": total_parameters,
              "trainable_parameter_ratio": trainable_parameters / total_parameters,
              "updated_tensor_count": update_stats["updated_tensor_count"],
              "trainable_tensor_count": update_stats["trainable_tensor_count"],
              "parameter_max_abs_update": delta, "reload_max_abs_error": difference,
              "model_checkpoint_bytes": model_checkpoint_bytes,
              "adapter_checkpoint_bytes": adapter_checkpoint_bytes,
              "optimizer_state_dtypes": optimizer_state_dtypes,
              "precision": precision,
              "initial_loading": initial_loading_metadata,
              "weight_save_verification": save_verification,
              "weight_reload_verification": exact_reload,
              "loading": reload_metadata,
              "embedding_dimension": actual.shape[-1], "train_seconds": train_seconds,
              "peak_cuda_allocated_gib": peak_memory / 2**30 if peak_memory is not None else None,
              "epoch_stats": epoch_stats, "first_loss": history[0]["loss"], "last_loss": history[-1]["loss"],
              "base_model": args.model, "model_output": str(model_path), "versions": versions,
              "training_file_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
              "note": "Loss/gradient/update/reload checks establish execution correctness, not legal retrieval accuracy."}
    if args.tuning_mode == "lora":
        # Preserve existing LoRA-report fields for callers of the first recipe.
        report.update({"adapter_max_abs_update": delta, "adapter_save_verification": save_verification,
                       "adapter_reload_verification": exact_reload, "adapter_before_save": before_save_active,
                       "adapter_loading": reload_metadata})
    write_json(args.output / "training_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
