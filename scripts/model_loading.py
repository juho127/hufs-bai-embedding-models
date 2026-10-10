"""Load a BF16 base while preserving local LoRA weights exactly in FP32.

Some Transformers/PEFT loading paths cast saved FP32 adapters to the requested
base dtype. Recasting those rounded values to FP32 cannot recover the weights.
For a local adapter checkpoint, restore from its safetensors after loading.
Only adapter tensors are copied; the frozen base stays in its requested dtype.
"""
from __future__ import annotations

from pathlib import Path
import json
import re


PRECISION_FILENAME = "training_precision.json"


def read_training_precision(path):
    source = Path(path) / PRECISION_FILENAME
    if not source.is_file():
        return None
    metadata = json.loads(source.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("tuning_mode") not in ("lora", "full"):
        raise RuntimeError(f"Unsupported training precision metadata: {source}")
    if metadata.get("base_parameter_dtype") not in ("bfloat16", "float32"):
        raise RuntimeError(f"Unsupported base precision: {source}")
    if metadata["tuning_mode"] == "full" and metadata["base_parameter_dtype"] != "float32":
        raise RuntimeError("Full-training checkpoints must preserve FP32 parameters")
    return metadata


def canonical_lora_key(name):
    # ST 5 exposes 0.auto_model.*, while ST 6 uses 0.model.*.
    name = re.sub(r"^\d+\.(?:auto_model|model)\.", "", name)
    if ".auto_model." in name:
        name = name.split(".auto_model.", 1)[1]
    while name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    for kind in ("lora_A", "lora_B"):
        name = name.replace(f".{kind}.default.", f".{kind}.")
    return name


def lora_parameters(model):
    found = {}
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name and ".lora_B." not in name:
            continue
        key = canonical_lora_key(name)
        if key in found:
            raise RuntimeError(f"Ambiguous adapter parameter key: {key}")
        found[key] = parameter
    return found


def local_adapter_state(path):
    """Return canonical CPU tensors for exactly one local LoRA checkpoint."""
    from safetensors.torch import load_file

    path = Path(path)
    if not path.is_dir():
        return {}, []
    files = sorted(path.rglob("adapter_model.safetensors"))
    if not files:
        return {}, []
    if len(files) != 1:
        raise RuntimeError(f"Expected one adapter checkpoint under {path}, found {len(files)}")
    state = {}
    for name, tensor in load_file(str(files[0]), device="cpu").items():
        if ".lora_A." not in name and ".lora_B." not in name:
            raise RuntimeError(f"Unsupported non-LoRA tensor in adapter checkpoint: {name}")
        key = canonical_lora_key(name)
        if key in state:
            raise RuntimeError(f"Ambiguous saved adapter key: {key}")
        state[key] = tensor
    if not state:
        raise RuntimeError(f"Empty adapter checkpoint: {files[0]}")
    return state, [str(file) for file in files]


def compare_adapter_states(reference, candidate):
    """Compare parameter mappings exactly with bounded scratch memory."""
    import torch

    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))[:5]
        extra = sorted(set(candidate) - set(reference))[:5]
        raise RuntimeError(f"Parameter keys differ; missing={missing}, extra={extra}")
    largest, equal = 0.0, True
    for key in reference:
        left, right = reference[key].detach().cpu(), candidate[key].detach().cpu()
        if left.shape != right.shape:
            raise RuntimeError(f"Parameter shape mismatch: {key}: {left.shape} vs {right.shape}")
        flat_left, flat_right = left.reshape(-1), right.reshape(-1)
        for start in range(0, left.numel(), 1_048_576):
            part_left = flat_left[start:start + 1_048_576].float()
            part_right = flat_right[start:start + 1_048_576].float()
            if not torch.isfinite(part_left).all() or not torch.isfinite(part_right).all():
                raise RuntimeError(f"Nonfinite parameter weights: {key}")
            largest = max(largest, (part_left - part_right).abs().max().item())
            equal = equal and torch.equal(part_left, part_right)
    return {"tensor_count": len(reference), "values_exact": equal,
            "max_abs_error": largest,
            "reference_dtypes": sorted({str(value.dtype) for value in reference.values()}),
            "candidate_dtypes": sorted({str(value.dtype) for value in candidate.values()})}


def active_adapter_metadata(model):
    backbone = model[0].auto_model
    metadata = {"backbone_class": type(backbone).__name__}
    for attr in ("active_adapters", "active_adapter"):
        try:
            value = getattr(backbone, attr, None)
            value = value() if callable(value) else value
            if isinstance(value, (str, list, tuple)):
                metadata[attr] = list(value) if isinstance(value, tuple) else value
        except (ValueError, AttributeError):
            pass
    return metadata


def restore_local_lora_fp32(model, path):
    import torch

    saved, files = local_adapter_state(path)
    metadata = {"local_adapter_files": files, "fp32_adapter_restored": False,
                **active_adapter_metadata(model)}
    if not saved:
        return metadata
    parameters = lora_parameters(model)
    metadata["before_restore"] = compare_adapter_states(saved, parameters)
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.data = saved[key].to(device=parameter.device, dtype=torch.float32).clone()
    metadata["after_restore"] = compare_adapter_states(saved, parameters)
    if not metadata["after_restore"]["values_exact"]:
        raise RuntimeError("FP32 adapter restore failed exact weight verification")
    metadata["fp32_adapter_restored"] = True
    return metadata


def load_sentence_transformer(model_name_or_path, *, device="cuda", dtype=None,
                              attention="sdpa", revision=None):
    import torch
    from sentence_transformers import SentenceTransformer

    dtype = torch.bfloat16 if dtype is None else dtype
    requested_dtype = dtype
    precision = read_training_precision(model_name_or_path)
    # A lower precision reload would erase small full-training updates. LoRA
    # and original bases still honor explicit FP32 evaluation requests.
    if precision and precision["tuning_mode"] == "full":
        dtype = torch.float32
    model = SentenceTransformer(str(model_name_or_path), device=device, revision=revision,
                                model_kwargs={"dtype": dtype, "attn_implementation": attention})
    metadata = restore_local_lora_fp32(model, model_name_or_path)
    metadata.update({"precision": precision, "requested_base_dtype": str(requested_dtype),
                     "effective_base_dtype": str(dtype)})
    if precision and precision["tuning_mode"] == "full":
        if lora_parameters(model):
            raise RuntimeError("Full checkpoint unexpectedly contains a LoRA adapter")
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("Full checkpoint parameters did not retain FP32 precision")
    return model, metadata
