"""Download verified full-model assets from this project's public GitHub Release.

Only the Python standard library is required. Safetensors shards remain separate;
Transformers reads their index when loading the completed model directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


REPOSITORY = "juho127/hufs-bai-embedding-models"
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "releases/v0.1.0/manifest.json"
DEFAULT_OUTPUT = "models/hufs-bai-legal-embed-1b-v0.1"
MAX_ASSET_BYTES = 2_000_000_000
CHUNK_BYTES = 1024 * 1024
WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                    *(f"LPT{i}" for i in range(1, 10))}


class ReleaseError(ValueError):
    """The release manifest, destination, or received bytes are unsafe or invalid."""


def _safe_parts(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReleaseError(f"Invalid relative file path: {value!r}")
    parts = value.split("/")
    for part in parts:
        if (part in {"", ".", ".."} or part[-1:] in {" ", "."}
                or any(ord(char) < 32 or char in ':<>"|?*' for char in part)
                or part.split(".")[0].upper() in WINDOWS_RESERVED):
            raise ReleaseError(f"Invalid relative file path: {value!r}")
    return parts


def validate_manifest(manifest):
    """Validate the entire manifest before creating directories or fetching assets."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ReleaseError("Expected a schema_version 1 release manifest.")
    if manifest.get("repository") != REPOSITORY:
        raise ReleaseError(f"Only releases from {REPOSITORY} are accepted.")
    tag = manifest.get("release_tag")
    if not isinstance(tag, str) or not re.fullmatch(r"(?:(?:defense|legal)-)?v\d+\.\d+\.\d+(?:-[A-Za-z0-9.-]+)?", tag):
        raise ReleaseError("Invalid release_tag; expected a version such as v0.1.0 or defense-v0.2.0.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseError("The manifest must contain a nonempty files list.")
    paths, assets = set(), set()
    for item in files:
        if not isinstance(item, dict):
            raise ReleaseError("Each file entry must be an object.")
        path = item.get("path")
        _safe_parts(path)
        asset = item.get("asset")
        if not isinstance(asset, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", asset):
            raise ReleaseError(f"Invalid asset filename: {asset!r}")
        _safe_parts(asset)
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size < MAX_ASSET_BYTES:
            raise ReleaseError(f"{path}: size_bytes must be below 2,000,000,000.")
        digest = item.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ReleaseError(f"{path}: invalid SHA-256 digest.")
        key = path.casefold()
        if key in paths or asset.casefold() in assets:
            raise ReleaseError(f"Duplicate file path or asset: {path}")
        paths.add(key)
        assets.add(asset.casefold())
    for path in paths:
        if any("/".join(path.split("/")[:i]) in paths for i in range(1, len(path.split("/")))):
            raise ReleaseError(f"File/directory path conflict: {path}")
    return manifest


def _reject_links(path):
    """Check every existing ancestor, including Windows junctions/reparse points."""
    for component in [*reversed(path.parents), path]:
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        reparse = getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if stat.S_ISLNK(info.st_mode) or reparse:
            raise ReleaseError(f"Symbolic links and junctions are not allowed: {component}")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path, entry):
    return (path.is_file() and path.stat().st_size == entry["size_bytes"]
            and sha256_file(path) == entry["sha256"].lower())


def _preflight(output, files):
    _reject_links(output)
    if output.exists() and not output.is_dir():
        raise ReleaseError(f"Output must be a directory: {output}")
    existing = set()
    allowed = {item["path"].casefold() for item in files}
    partials = {(item["path"] + ".partial").casefold() for item in files}
    if allowed & partials:
        raise ReleaseError("Manifest paths conflict with temporary download filenames.")
    if output.exists():
        for current, directories, filenames in os.walk(output, followlinks=False):
            for name in directories + filenames:
                path = Path(current) / name
                _reject_links(path)
                if path.is_file() and path.relative_to(output).as_posix().casefold() not in allowed | partials:
                    raise ReleaseError(f"Unexpected file in model directory: {path}. Choose an empty output directory.")
    for item in files:
        destination = output.joinpath(*_safe_parts(item["path"]))
        _reject_links(destination)
        _reject_links(destination.with_name(destination.name + ".partial"))
        if destination.exists():
            if not _matches(destination, item):
                raise ReleaseError(f"Existing file differs from this release: {destination}. Choose another output directory.")
            existing.add(item["path"])
    return existing


def download_release(manifest, output, *, opener=None, progress=None):
    """Download missing files atomically; existing different files are never replaced."""
    validate_manifest(manifest)
    output = Path(os.path.abspath(output))
    existing = _preflight(output, manifest["files"])
    opener = opener or urlopen
    downloaded = 0
    for entry in manifest["files"]:
        relative = entry["path"]
        if relative in existing:
            if progress:
                print(f"Verified existing: {relative}", file=progress, flush=True)
            continue
        destination = output.joinpath(*_safe_parts(relative))
        _reject_links(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".partial")
        _reject_links(temporary)
        if temporary.exists():
            if not temporary.is_file():
                raise ReleaseError(f"Temporary path is not a regular file: {temporary}")
            temporary.unlink()
        url = f"https://github.com/{REPOSITORY}/releases/download/{quote(manifest['release_tag'], safe='')}/{quote(entry['asset'], safe='')}"
        request = Request(url, headers={"User-Agent": "HUFS-BAI-model-downloader/1", "Accept-Encoding": "identity"})
        digest, received, last_update = hashlib.sha256(), 0, 0.0
        try:
            with opener(request, timeout=60) as response, temporary.open("xb") as handle:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > entry["size_bytes"]:
                        raise ReleaseError(f"{relative}: received more bytes than the manifest allows.")
                    handle.write(chunk)
                    digest.update(chunk)
                    now = time.monotonic()
                    if progress and now - last_update >= 2:
                        print(f"\r{relative}: {received:,}/{entry['size_bytes']:,} bytes", end="", file=progress, flush=True)
                        last_update = now
                handle.flush()
                os.fsync(handle.fileno())
            if received != entry["size_bytes"] or digest.hexdigest() != entry["sha256"].lower():
                raise ReleaseError(f"{relative}: size or SHA-256 verification failed; rerun to download again.")
            _reject_links(destination)
            if destination.exists():
                raise ReleaseError(f"Destination appeared during download; refusing to overwrite: {destination}")
            os.replace(temporary, destination)
            downloaded += 1
            if progress:
                print(f"\rVerified: {relative} ({received:,} bytes)", file=progress, flush=True)
        finally:
            _reject_links(temporary)
            if temporary.exists():
                temporary.unlink()
    return {"downloaded": downloaded, "verified_existing": len(existing), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Empty model directory, or a previous download of this release")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Local release manifest JSON")
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = download_release(manifest, args.output, progress=sys.stderr)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
