"""Release download and integrity checks use tiny fixtures without network access."""
import copy
import hashlib
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from download_release import REPOSITORY, ReleaseError, download_release, validate_manifest


class ReleaseDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "model"
        self.payloads = [b"tiny synthetic weights", b'{"pooling_mode_mean_tokens":true}\n']
        self.manifest = {
            "schema_version": 1, "model_id": "hufs-bai-legal-embed-1b-v0.1",
            "repository": REPOSITORY, "release_tag": "v0.1.0", "files": [
                {"path": path, "asset": asset, "size_bytes": len(payload),
                 "sha256": hashlib.sha256(payload).hexdigest()}
                for path, asset, payload in zip(
                    ["model-00001-of-00003.safetensors", "1_Pooling/config.json"],
                    ["model-00001-of-00003.safetensors", "1_Pooling-config.json"], self.payloads)
            ],
        }

    def test_download_restores_nested_directory_and_verifies_bytes(self):
        opener = Mock(side_effect=[io.BytesIO(payload) for payload in self.payloads])
        result = download_release(self.manifest, self.output, opener=opener)
        self.assertEqual(result["downloaded"], 2)
        for entry, payload in zip(self.manifest["files"], self.payloads):
            self.assertEqual((self.output / entry["path"]).read_bytes(), payload)
        self.assertEqual(list(self.output.rglob("*.partial")), [])
        self.assertEqual(opener.call_args_list[1].args[0].full_url,
                         f"https://github.com/{REPOSITORY}/releases/download/v0.1.0/1_Pooling-config.json")

    def test_matching_existing_files_are_verified_and_skipped(self):
        for entry, payload in zip(self.manifest["files"], self.payloads):
            target = self.output / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        opener = Mock(side_effect=AssertionError("A verified file must not be downloaded again"))
        result = download_release(self.manifest, self.output, opener=opener)
        self.assertEqual(result["verified_existing"], 2)
        self.assertEqual(result["downloaded"], 0)
        opener.assert_not_called()

    def test_mismatching_existing_file_blocks_all_downloads(self):
        target = self.output / "1_Pooling/config.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"another model")
        opener = Mock()
        with self.assertRaisesRegex(ReleaseError, "Existing file differs"):
            download_release(self.manifest, self.output, opener=opener)
        opener.assert_not_called()
        self.assertEqual(target.read_bytes(), b"another model")
        self.assertFalse((self.output / self.manifest["files"][0]["path"]).exists())

    def test_checksum_mismatch_never_publishes_bad_file(self):
        corrupt = b"x" * len(self.payloads[0])
        with self.assertRaisesRegex(ReleaseError, "verification failed"):
            download_release(self.manifest, self.output, opener=Mock(return_value=io.BytesIO(corrupt)))
        self.assertEqual(list(self.output.rglob("*")), [])

    def test_wrong_size_never_publishes_bad_file(self):
        for payload in [b"short", self.payloads[0] + b"excess"]:
            with self.subTest(size=len(payload)):
                with self.assertRaises(ReleaseError):
                    download_release(self.manifest, self.output, opener=Mock(return_value=io.BytesIO(payload)))
                self.assertEqual(list(self.output.rglob("*")), [])

    def test_interrupted_download_restarts_partial_file(self):
        self.output.mkdir()
        partial = self.output / (self.manifest["files"][0]["path"] + ".partial")
        partial.write_bytes(b"incomplete previous download")
        opener = Mock(side_effect=[io.BytesIO(payload) for payload in self.payloads])
        self.assertEqual(download_release(self.manifest, self.output, opener=opener)["downloaded"], 2)
        self.assertFalse(partial.exists())

    def test_network_failure_cleans_partial_and_preserves_verified_file(self):
        opener = Mock(side_effect=[io.BytesIO(self.payloads[0]), OSError("simulated disconnection")])
        with self.assertRaisesRegex(OSError, "simulated disconnection"):
            download_release(self.manifest, self.output, opener=opener)
        self.assertEqual((self.output / self.manifest["files"][0]["path"]).read_bytes(), self.payloads[0])
        self.assertEqual(list(self.output.rglob("*.partial")), [])
        retry = Mock(return_value=io.BytesIO(self.payloads[1]))
        result = download_release(self.manifest, self.output, opener=retry)
        self.assertEqual(result["downloaded"], 1)
        self.assertEqual(result["verified_existing"], 1)

    def test_unsafe_paths_are_rejected_before_network_or_output_creation(self):
        for path in ["../outside", "/absolute", "C:/outside", "C:relative", "\\\\server\\file",
                     "sub/../../outside", "sub\\outside", "./config.json", "a//b", "a/CON.txt",
                     "trailing./file", "file:stream", "config.json\x00", "config.json "]:
            with self.subTest(path=path):
                manifest = copy.deepcopy(self.manifest)
                manifest["files"][0]["path"] = path
                opener = Mock()
                with self.assertRaises(ReleaseError):
                    download_release(manifest, self.output, opener=opener)
                opener.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_foreign_repository_tag_injection_and_asset_paths_are_rejected(self):
        for field, value in [("repository", "another/project"), ("release_tag", "../../v0.1.0")]:
            manifest = copy.deepcopy(self.manifest)
            manifest[field] = value
            with self.assertRaises(ReleaseError):
                validate_manifest(manifest)
        for asset in ["../config.json", "config.json?token=x", "CON", "/config.json"]:
            manifest = copy.deepcopy(self.manifest)
            manifest["files"][0]["asset"] = asset
            with self.assertRaises(ReleaseError):
                validate_manifest(manifest)

    def test_manifest_limits_duplicates_and_conflicts(self):
        for size in [-1, 2_000_000_000, True, 1.5]:
            manifest = copy.deepcopy(self.manifest)
            manifest["files"][0]["size_bytes"] = size
            with self.assertRaises(ReleaseError):
                validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["files"].append(copy.deepcopy(manifest["files"][0]))
        with self.assertRaises(ReleaseError):
            validate_manifest(manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest["files"][0]["path"] = "1_Pooling"
        with self.assertRaises(ReleaseError):
            validate_manifest(manifest)

    def test_unlisted_files_prevent_mixing_model_versions(self):
        self.output.mkdir()
        (self.output / "pytorch_model.bin").write_bytes(b"another model")
        opener = Mock()
        with self.assertRaisesRegex(ReleaseError, "Unexpected file"):
            download_release(self.manifest, self.output, opener=opener)
        opener.assert_not_called()

    def test_symlink_output_is_rejected(self):
        real_directory = self.root / "actual"
        real_directory.mkdir()
        try:
            self.output.symlink_to(real_directory, target_is_directory=True)
        except OSError:
            self.skipTest("Creating symbolic links requires additional Windows privileges.")
        opener = Mock()
        with self.assertRaisesRegex(ReleaseError, "Symbolic links"):
            download_release(self.manifest, self.output, opener=opener)
        opener.assert_not_called()
        self.assertEqual(list(real_directory.iterdir()), [])

    def test_reparse_point_is_rejected_on_all_test_platforms(self):
        actual_lstat = Path.lstat

        def lstat(path, *args, **kwargs):
            if path == self.output:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return actual_lstat(path, *args, **kwargs)

        opener = Mock()
        with patch.object(Path, "lstat", lstat):
            with self.assertRaisesRegex(ReleaseError, "junctions"):
                download_release(self.manifest, self.output, opener=opener)
        opener.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_checked_in_manifest_if_present(self):
        path = Path(__file__).resolve().parents[1] / "releases/v0.1.0/manifest.json"
        if not path.exists():
            self.skipTest("Release manifest has not been generated yet.")
        validate_manifest(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
