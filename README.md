# HUFS BAI Legal Embed 1B

**English** · [한국어](README.ko.md)

**`hufs-bai-legal-embed-1b-v0.1`** is a Korean legal-domain embedding model from **HUFS Business AI Lab, Hankuk University of Foreign Studies**. It converts questions and document passages into 2,048-dimensional vectors for semantic search and retrieval-augmented generation (RAG).

Built on [NVIDIA Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16), the model was adapted through full-parameter fine-tuning on Korean legal question–passage pairs. On the reported legal retrieval development set, it achieves **0.8293 nDCG@10**, a **5.04% relative improvement** over the original model, and **95.0% Recall@10**.

[Download v0.1.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0) · [Model card](docs/model-card.md) · [Evaluation details](docs/evaluation.md) · [Contact](mailto:juho@hufs.ac.kr)

## Model at a glance

| Property | Value |
|---|---|
| Model ID | `hufs-bai-legal-embed-1b-v0.1` |
| Primary language / domain | Korean / legal documents |
| Intended uses | Question-to-passage search, legal document search, RAG retrieval |
| Parameters | 1.14 billion, all fine-tuned |
| Embedding dimension | 2,048 |
| Pooling / similarity | Mean pooling / cosine similarity with normalized vectors |
| Input length used in training and evaluation | 1,536 tokens, including the prompt |
| Query / document prompts | `query: ` / `passage: ` |
| Released weights | Complete FP32 model, approximately 4.58 GB |
| Model license | [OpenMDW 1.1](https://openmdw.ai/license/1-1/) |

The complete model is provided through GitHub Releases in files smaller than **2 GB each**. Weights use standard safetensors shards with an index for automatic loading; no manual concatenation is needed. The download script installs all required files, so a separate base-model download is unnecessary. See [distribution instructions](docs/distribution.md) for file details and integrity verification.

## Quick start

Use Python 3.11 or newer. Install the PyTorch build for your CPU or CUDA environment following the [setup guide](docs/setup.md), then run:

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
python -m pip install -r requirements.txt
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

For CPU inference, add `--device cpu` to the retrieval command. The example ranks synthetic Korean passages and preserves the checkpoint's FP32 parameters. On a supported CUDA device, it uses BF16 autocast for inference compute, matching the reported evaluation precision.

The checkpoint supplies its query and document prompts through `encode_query` and `encode_document`. Do not prepend these prompts again when using the provided example. Embed the indexed passages and user queries with the same model and use normalized vectors for cosine retrieval.

## Retrieval quality

Both models were evaluated on the same NVIDIA RTX A6000 with **500 development queries** and the full corpus of **66,895 passages**, using exact cosine search.

| Model | nDCG@10 ↑ | MRR@10 ↑ | Recall@10 ↑ | Recall@50 ↑ |
|---|---:|---:|---:|---:|
| Original Nemotron-3-Embed-1B | 0.7895 | 0.7441 | 0.928 | 0.980 |
| **HUFS BAI Legal Embed 1B v0.1** | **0.8293** | **0.7897** | **0.950** | **0.988** |

The nDCG@10 gain is **+0.0398**, with a paired document-group bootstrap 95% confidence interval of **[+0.0221, +0.0581]**. The source corpus contains Korean legal materials, including policy, legislative and administrative documents. These are single-seed development-set results; the reserved 1,000-query test set has not been evaluated. The [evaluation document](docs/evaluation.md) describes the retrieval protocol and statistical comparison, and [result JSON](results/initial_dev_results.json) provides the numerical results.

## Training and provenance

The model was fully fine-tuned for one epoch using `CachedMultipleNegativesRankingLoss` with in-batch negatives. Training used 89,898 question–passage pairs from the legal portion of **AI Hub's 금융·법률 문서 기계독해 데이터 (Financial and Legal Document Machine Reading Comprehension)**, [dataset 71610](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610). Source-family grouping and duplicate checks separate the data splits and reduce conflicting negatives within a batch.

See the [model card](docs/model-card.md) for the full training recipe. Training and evaluation code is available for users working with their own authorized data. The release contains model files and software; AI Hub source documents and derived training/evaluation text are not redistributed.

## License

The released model weights and inherited model materials are covered by [OpenMDW 1.1](https://openmdw.ai/license/1-1/). Repository code and documentation are provided under [Apache 2.0](LICENSE). Preserve the license and attribution files included with the model. Data provenance and the separate AI Hub terms are described in [NOTICE.md](NOTICE.md).

## Contact and domain-specific tuning

For **domain-specific embedding fine-tuning, customized retrieval models, and research or industry collaboration**, please contact **Prof. Juho Bai at [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)**.

**HUFS Business AI Lab · Hankuk University of Foreign Studies**

[Homepage](https://juho-bai.pages.dev/) · [GitHub issues](https://github.com/juho127/hufs-bai-embedding-models/issues)
