"""Small precision/integrity regressions; no model download or GPU required."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_loading
from train import final_parameter_snapshot, verify_full_checkpoint

try:
    import torch
    from safetensors.torch import load_file, save_file
    OPTIONAL_DEPENDENCIES = True
    DEPENDENCY_REASON = ""
except (ImportError, OSError) as exc:
    OPTIONAL_DEPENDENCIES = False
    DEPENDENCY_REASON = f"Optional torch/safetensors unavailable: {exc}"


def tiny_sentence_transformer(dtype=None, adapters=False):
    """Expose the same parameter paths/properties used by the model loader."""
    dtype = torch.float32 if dtype is None else dtype

    class FirstModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(3, 2, dtype=dtype)
            if adapters:
                self.model.lora_A = torch.nn.ModuleDict({
                    "default": torch.nn.Linear(3, 1, bias=False, dtype=dtype)})
                self.model.lora_B = torch.nn.ModuleDict({
                    "default": torch.nn.Linear(1, 2, bias=False, dtype=dtype)})

        @property
        def auto_model(self):
            return self.model

    return torch.nn.Sequential(FirstModule())


@unittest.skipUnless(OPTIONAL_DEPENDENCIES, DEPENDENCY_REASON)
class TrainingModeTests(unittest.TestCase):
    def setUp(self):
        # A workspace temp directory also works in restricted Windows sandboxes.
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = tiny_sentence_transformer()
        with torch.no_grad():
            for parameter in self.model.parameters():
                # These values intentionally cannot be represented exactly in BF16.
                parameter.fill_(1.000123)

    def write_precision(self, mode):
        metadata = {"schema_version": 1, "tuning_mode": mode,
                    "base_parameter_dtype": "float32" if mode == "full" else "bfloat16",
                    "adapter_parameter_dtype": None if mode == "full" else "float32",
                    "inference_autocast_dtype": "bfloat16"}
        (self.root / model_loading.PRECISION_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")

    def save_full_model(self):
        save_file(self.model[0].auto_model.state_dict(), str(self.root / "model.safetensors"))

    def load_without_download(self, requested_dtype):
        calls = []

        def tiny_loader(path, **kwargs):
            calls.append(kwargs)
            model = tiny_sentence_transformer(kwargs["model_kwargs"]["dtype"])
            model[0].auto_model.load_state_dict(load_file(str(Path(path) / "model.safetensors")))
            return model

        replacement = types.SimpleNamespace(SentenceTransformer=tiny_loader)
        with patch.dict(sys.modules, {"sentence_transformers": replacement}):
            loaded, metadata = model_loading.load_sentence_transformer(
                self.root, device="cpu", dtype=requested_dtype)
        return loaded, metadata, calls

    def test_full_checkpoint_rejects_tampered_parameter(self):
        reference = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        self.save_full_model()
        verified = verify_full_checkpoint(reference, self.root)
        self.assertTrue(verified["values_exact"])
        self.assertEqual(verified["tensor_count"], 2)
        with torch.no_grad():
            self.model[0].auto_model.weight[0, 0].add_(0.001)
        self.save_full_model()
        with self.assertRaisesRegex(RuntimeError, "changed parameter value or precision"):
            verify_full_checkpoint(reference, self.root)

    def test_full_checkpoint_preserves_fp32_despite_bf16_request(self):
        self.save_full_model()
        self.write_precision("full")
        loaded, metadata, calls = self.load_without_download(torch.bfloat16)
        self.assertEqual(calls[0]["model_kwargs"]["dtype"], torch.float32)
        self.assertEqual(metadata["effective_base_dtype"], "torch.float32")
        for actual, expected in zip(loaded.parameters(), self.model.parameters()):
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.equal(actual, expected))
            self.assertFalse(torch.equal(actual, expected.to(torch.bfloat16).float()))

    def test_lora_metadata_respects_explicit_fp32_base_request(self):
        self.save_full_model()
        self.write_precision("lora")
        loaded, metadata, calls = self.load_without_download(torch.float32)
        self.assertEqual(calls[0]["model_kwargs"]["dtype"], torch.float32)
        self.assertEqual(metadata["effective_base_dtype"], "torch.float32")
        self.assertTrue(all(parameter.dtype == torch.float32 for parameter in loaded.parameters()))

    def test_lora_restores_fp32_values_from_disk_after_rounded_load(self):
        rounded = tiny_sentence_transformer(torch.bfloat16, adapters=True)
        parameters = model_loading.lora_parameters(rounded)
        saved = {key: torch.full(parameter.shape, 1.000123, dtype=torch.float32)
                 for key, parameter in parameters.items()}
        with torch.no_grad():
            for key, parameter in parameters.items():
                parameter.copy_(saved[key])
        save_file({"base_model.model." + key: value for key, value in saved.items()},
                  str(self.root / "adapter_model.safetensors"))
        metadata = model_loading.restore_local_lora_fp32(rounded, self.root)
        self.assertGreater(metadata["before_restore"]["max_abs_error"], 0)
        self.assertTrue(metadata["after_restore"]["values_exact"])
        for key, parameter in model_loading.lora_parameters(rounded).items():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertTrue(torch.equal(parameter, saved[key]))
        # Restoring adapters must not promote the frozen base implicitly.
        self.assertEqual(rounded[0].auto_model.weight.dtype, torch.bfloat16)

    def test_update_summary_checks_all_trainable_tensors(self):
        initial = {name: parameter.detach().clone() for name, parameter in self.model.named_parameters()}
        with torch.no_grad():
            self.model[0].auto_model.weight.add_(0.125)
        final, summary = final_parameter_snapshot(dict(self.model.named_parameters()), initial)
        self.assertEqual(summary["trainable_tensor_count"], 2)
        self.assertEqual(summary["updated_tensor_count"], 1)
        self.assertAlmostEqual(summary["max_abs_update"], 0.125)
        self.assertEqual(initial, {})
        self.assertEqual(len(final), 2)


if __name__ == "__main__":
    unittest.main()
