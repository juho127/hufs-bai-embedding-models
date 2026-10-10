"""Shard a full FP32 Sentence Transformers checkpoint without changing weights.

The output uses the standard Hugging Face safetensors index format. Only model
configuration and tokenizer files are copied; training data and logs are not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


CONFIG_FILES = (
    "config.json", "config_sentence_transformers.json", "modules.json",
    "sentence_bert_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "tokenizer.model", "added_tokens.json",
    "training_precision.json", "1_Pooling/config.json", "2_Normalize/config.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(source: Path, output: Path, max_shard_bytes: int,
            license_file: Path | None = None, base_model_card: Path | None = None) -> dict:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, output = source.resolve(), output.resolve()
    checkpoint = source / "model.safetensors"
    if not checkpoint.is_file():
        raise ValueError("Source must contain a single full model.safetensors checkpoint")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Source and output must be separate directory trees")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be absent or empty; existing releases are not overwritten")
    if max_shard_bytes < 1_000_000 or max_shard_bytes >= 2_000_000_000:
        raise ValueError("Use a shard size from 1 MB to less than 2,000,000,000 bytes")
    for optional_file in (license_file, base_model_card):
        if optional_file is not None and not optional_file.is_file():
            raise ValueError(f"Missing attribution file: {optional_file.name}")

    # Reserve room for the safetensors JSON header in the requested file limit.
    header_reserve = 1_000_000
    groups, current, current_bytes, total_bytes = [], [], 0, 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            tensor = handle.get_tensor(name)
            if tensor.dtype != torch.float32:
                raise ValueError(f"Expected full FP32 weights: {name} is {tensor.dtype}")
            size = tensor.numel() * tensor.element_size()
            if size + header_reserve >= max_shard_bytes:
                raise ValueError(f"One tensor exceeds the shard limit: {name}")
            if current and current_bytes + size + header_reserve >= max_shard_bytes:
                groups.append(current)
                current, current_bytes = [], 0
            current.append(name)
            current_bytes += size
            total_bytes += size
        if current:
            groups.append(current)
        if not groups:
            raise ValueError("Checkpoint is empty")

        output.mkdir(parents=True, exist_ok=True)
        weight_map = {}
        for index, names in enumerate(groups, start=1):
            filename = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            shard = output / filename
            save_file({name: handle.get_tensor(name) for name in names},
                      str(shard), metadata={"format": "pt"})
            if shard.stat().st_size >= max_shard_bytes:
                raise RuntimeError(f"Shard exceeds requested file limit: {filename}")
            weight_map.update({name: filename for name in names})

        verified = 0
        for filename in sorted(set(weight_map.values())):
            with safe_open(str(output / filename), framework="pt", device="cpu") as shard:
                for name in shard.keys():
                    original, restored = handle.get_tensor(name), shard.get_tensor(name)
                    if original.dtype != restored.dtype or not torch.equal(original, restored):
                        raise RuntimeError(f"Exact FP32 tensor verification failed: {name}")
                    verified += 1
        if verified != len(weight_map) or set(weight_map) != set(handle.keys()):
            raise RuntimeError("Shard tensor inventory differs from source")

    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for relative in CONFIG_FILES:
        origin = source / relative
        if origin.is_file():
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
    if license_file:
        shutil.copy2(license_file, output / "LICENSE")
    if base_model_card:
        shutil.copy2(base_model_card, output / "BASE_MODEL_CARD.md")

    return {
        "tensor_count": verified, "values_exact": True, "parameter_dtype": "float32",
        "weight_bytes": total_bytes, "shards": len(groups),
        "files": [{"path": str(path.relative_to(output)).replace("\\", "/"),
                   "bytes": path.stat().st_size, "sha256": sha256(path)}
                  for path in sorted(output.rglob("*")) if path.is_file()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shard-bytes", type=int, default=1_800_000_000)
    parser.add_argument("--license-file", type=Path)
    parser.add_argument("--base-model-card", type=Path)
    parser.add_argument("--report", type=Path,
                        help="Optional external packaging report (not part of the model)")
    args = parser.parse_args()
    report = package(args.source, args.output, args.max_shard_bytes,
                     args.license_file, args.base_model_card)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
