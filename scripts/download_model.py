"""Download a pinned, public embedding checkpoint; no training or credentials required."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "nvidia/Nemotron-3-Embed-1B-BF16"
REVISION = "c0c9fea93ea424587517f2c59e20db9f1d6bf615"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/nemotron-3-embed-1b-bf16")
    args = parser.parse_args()
    target = Path(args.output)
    snapshot_download(
        repo_id=MODEL_ID, revision=REVISION, local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.model", "LICENSE*", "README.md"],
        max_workers=4,
    )
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    metadata = {"model_id": MODEL_ID, "revision": REVISION,
                "model_type": config.get("model_type"),
                "is_causal": config.get("is_causal"),
                "hidden_size": config.get("hidden_size")}
    (target / "download_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
