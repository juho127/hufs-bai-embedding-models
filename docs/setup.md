# Setup and local use

Run commands from the repository root. The released model was fully fine-tuned and evaluated on one NVIDIA RTX A6000 (x86_64, CUDA 12.8). Python 3.11 or newer is required by the scripts.

## 1. Install dependencies

Create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Choose **one** PyTorch installation matching the target platform:

```bash
# Linux x86_64, RTX A6000, CUDA 12.8 wheel
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128

# Linux ARM64, GB10, CUDA 13.0 wheel
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# CPU examples and unit tests
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

These versioned builds are listed in the official PyTorch wheel indexes for [CUDA 12.8](https://download.pytorch.org/whl/cu128/torch/), [CUDA 13.0](https://download.pytorch.org/whl/cu130/torch/), and [CPU](https://download.pytorch.org/whl/cpu/torch/). A compatible NVIDIA driver must already be installed for GPU use. Install the remaining pinned packages afterward:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`requirements.txt` intentionally leaves the PyTorch wheel choice to the platform. It pins the application packages used in the experiment; it is not a complete lockfile for every transitive dependency. The public installation instructions have not been validated as a fresh environment on every platform. The recorded experiments used these package versions on the devices above. The private experiment container is not needed or provided here.

## 2. Download the released model

```bash
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
```

The script downloads the complete `hufs-bai-legal-embed-1b-v0.1` model from GitHub Releases and checks file sizes and SHA-256 hashes. The standard safetensors index loads the three weight shards automatically. Keep all configuration, tokenizer and license files alongside the weights. `models/` is ignored by Git.

The complete FP32 model occupies about 4.58 GB. No separate base download is needed for inference. For original-model evaluation or new fine-tuning only, `python scripts/download_model.py` downloads NVIDIA's pinned base revision `c0c9fea93ea424587517f2c59e20db9f1d6bf615` into `models/nemotron-3-embed-1b-bf16`.

## 3. Try retrieval

```bash
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1 --device cpu
```

The example ranks two newly written, synthetic Korean passages. It uses the checkpoint's mean pooling, `query: ` and `passage: ` prompts through the Sentence Transformers APIs. It preserves FP32 full-model updates through `scripts/model_loading.py`, uses BF16 autocast on supported CUDA devices, and disables autocast on CPU. A synthetic example demonstrates execution only.

## 4. Prepare your own authorized data

Create `data/raw/pairs.jsonl` with one JSON object per line. Every field below is a non-empty string:

```json
{"query":"가상 문서의 배송 일정은?","positive":"가상 문서 A의 배송 예정일은 10월 20일이다.","query_id":"q-example","positive_id":"p-example","group_id":"source-family-example"}
```

This is a schema example, not a sufficient training dataset. Supply many independent source families. `group_id` identifies the source or evidence family, not the individual question. Questions and passages sharing a document, edition family, or other relevant source relationship should be grouped together before splitting.

```bash
python scripts/prepare_data.py --input data/raw/pairs.jsonl --format canonical --output-dir data/processed
```

The converter creates grouped train/dev/test splits, a global passage corpus, query files and qrels. It rejects inconsistent IDs and protects exact matching questions/passages across splits. Its checks do not discover all semantic duplicates or omitted family relationships. An existing frozen corpus can be supplied with `--corpus`.

The original experiment used an additional AI Hub-specific release procedure and private quality-review annotations. The generic command above prepares a **new dataset**, not the exact reported experimental split. AI Hub source text, derived pairs and review annotations are not distributed here. Obtain the source data through its authorized channel; do not upload it to GitHub.

## 5. Train one epoch

```bash
python scripts/train.py --config configs/train_full_mnrl.json
```

Select a free GPU before training. The recipe uses one epoch, seed 42, maximum length 1,536, logical batch 128 and microbatch 4. Cached Multiple Negatives Ranking Loss uses the other positives in the logical batch as negatives. The sampler avoids same-family and exact-duplicate conflicts within a batch. Incompatible singleton leftovers are excluded and reported.

The recipe updates all model weights with learning rate `2e-6`, using BF16 CUDA compute and FP32 master weights. Every saved checkpoint is checked for exact parameter preservation and reload behavior.

For a short software check with artificial data:

```bash
python scripts/make_smoke_data.py --output data/smoke --count 256
python scripts/train.py --config configs/train_full_mnrl.json --train data/smoke/pairs.jsonl --output runs/smoke_full --smoke --max-steps 2
```

The synthetic run cannot be used to claim improved legal retrieval. Actual memory and runtime depend on sequence lengths and hardware; use the reported experiment settings when comparing methods. Evaluation commands and result interpretation are documented separately in this repository.

## 6. Run CPU regression checks

```bash
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -p "test_*.py" -v
```

These tests use synthetic data and small tensors. No AI Hub data, trained checkpoint, GPU, or Hugging Face download is required. GitHub Actions runs the same suite in a CPU-only environment.
