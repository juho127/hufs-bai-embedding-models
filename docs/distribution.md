# Download HUFS BAI Legal Embed 1B

[한국어](distribution.ko.md) · [Model overview](../README.md)

**HUFS BAI Legal Embed 1B v0.1** is the complete, fully fine-tuned Korean legal embedding model. Download the weights and tokenizer from [GitHub Releases v0.1.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0). No token, subscription, or separate download of the NVIDIA base model is required.

## Download

Clone this repository, then run the downloader with Python 3.10 or newer. The downloader itself needs no third-party packages.

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
```

The download is about **4.6 GB**. Allow at least 5 GB of free disk space for the model; the inference environment requires additional space. Each release asset is **smaller than 2,000,000,000 bytes**, below GitHub's 2 GiB per-asset limit. Files are stored as regular release assets, so downloading does not use the repository's Git LFS quota. GitHub documents no total release-size or download-bandwidth limit in its [Releases policy](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

The model uses three standard Safetensors shards and an index. **Keep all shards separate in the same model directory.** Transformers and Sentence Transformers read the index automatically. There is no archive to extract and no binary pieces to concatenate. The released weights retain the FP32 parameters used in evaluation.

| Weight file | Bytes | Approximate size |
| --- | ---: | ---: |
| `model-00001-of-00003.safetensors` | 1,778,445,520 | 1.78 GB |
| `model-00002-of-00003.safetensors` | 1,795,301,096 | 1.80 GB |
| `model-00003-of-00003.safetensors` | 989,942,248 | 0.99 GB |

The downloader reads [the versioned manifest](../releases/v0.1.0/manifest.json), restores nested paths such as `1_Pooling/config.json`, and checks the byte count and SHA-256 of every file before accepting it. Completed files are verified and skipped on rerun. If the connection fails, rerun the same command; only the interrupted file starts again. An existing file that differs from the release causes an error rather than an overwrite. Use a separate empty directory for another model or version.

## Load the model

After installing the [inference environment](setup.md), use the downloaded local path:

```bash
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

See the [model overview](../README.md) for query/document prefixes and retrieval examples.

## Manual download and verification

For manual downloads, save every file listed in [manifest.json](../releases/v0.1.0/manifest.json) from the [release assets](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0). The `asset` field is the release filename and `path` is its required destination relative to the model folder. For example, save `1_Pooling-config.json` as `1_Pooling/config.json`. Include the model index, tokenizer, pooling configuration, and license files alongside all three shards.

Compare a file's digest with its `sha256` entry in the manifest:

```bash
# Linux / macOS
shasum -a 256 models/hufs-bai-legal-embed-1b-v0.1/model-00001-of-00003.safetensors
```

```powershell
# Windows PowerShell
Get-FileHash models/hufs-bai-legal-embed-1b-v0.1/model-00001-of-00003.safetensors -Algorithm SHA256
```

The downloader is recommended because it verifies all files and restores the directory layout automatically.

## License and contact

Model weights and inherited NVIDIA model materials are distributed under [OpenMDW-1.1](https://openmdw.ai/license/1-1/), with attribution and license files included in the release. Repository code is licensed under [Apache-2.0](../LICENSE). See [NOTICE](../NOTICE.md) and the [model card](model-card.md) for the base model and AI Hub training-data attribution. Training and evaluation texts are not included in the model download.

For specialized domain tuning, model customization, or research collaboration, contact **[juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)**. [Homepage](https://juho-bai.pages.dev/).
