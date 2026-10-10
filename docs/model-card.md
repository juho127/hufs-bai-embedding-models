# Model card: HUFS BAI Legal Embed 1B v0.1

**English** · [한국어](model-card.ko.md)

## Model description

**`hufs-bai-legal-embed-1b-v0.1`** is a Korean legal-domain embedding model developed by **HUFS Business AI Lab, Hankuk University of Foreign Studies**. It embeds questions and passages for semantic retrieval and RAG. Full-parameter fine-tuning of NVIDIA Nemotron-3-Embed-1B-BF16 improves legal retrieval nDCG@10 from **0.7895 to 0.8293** on the reported development set.

| Property | Value |
|---|---|
| Release | [v0.1.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0) |
| Base model | [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) |
| Base revision | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| Parameters | 1,140,918,272 |
| Encoder | Ministral3 with bidirectional attention (`is_causal=false`) |
| Embedding dimension | 2,048 |
| Pooling | Mean pooling, including prompt tokens |
| Similarity | Cosine similarity after normalization |
| Query / document prompts | `query: ` / `passage: ` |
| Training and evaluation sequence length | 1,536 tokens, including the prompt |
| Weight precision | FP32 |
| Model license | OpenMDW 1.1 |

The 1,536-token setting is the length used for this release's training and evaluation; it is not a claim about the base architecture's maximum supported context length. Longer-input retrieval has not been evaluated for this model.

## Intended use

Use this model to retrieve Korean legal passages relevant to a natural-language question, search legal document collections, or supply retrieved evidence to a RAG system. The output is an embedding vector. The model does not generate legal answers or determine whether a source is current or legally authoritative.

The reported corpus includes legal policy, legislative and administrative materials. Performance on case-law-only collections, other professional domains, other languages and different corpus distributions should be evaluated on the intended retrieval task.

## Download and inference

The release provides the complete FP32 model in files smaller than 2 GB each, together with configuration, tokenizer, license and attribution files. Standard safetensors shards load automatically through the accompanying index; no manual concatenation or separate NVIDIA base-model download is required.

After installing the dependencies in the [setup guide](setup.md):

```bash
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

The provided loader preserves FP32 weights. Inference uses BF16 autocast on supported CUDA devices, or FP32 compute on CPU. Use the checkpoint's `encode_query` and `encode_document` methods so that each prompt is applied once. The example normalizes embeddings and computes cosine similarity. See [distribution instructions](distribution.md) for checksums and release packaging.

## Training data

Training used the legal subset of **금융·법률 문서 기계독해 데이터 (Financial and Legal Document Machine Reading Comprehension)**, provided by **AI Hub**, [dataset 71610](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610). Training pairs use passage-extraction and procedural question types 1 and 5. The retrieval corpus includes non-table passages from legal question types 1, 2, 4 and 5.

| Data item | Count |
|---|---:|
| Prepared training question–passage pairs | 89,979 |
| Pairs used in training | 89,898 |
| Singleton leftovers excluded by batch compatibility rules | 81 |
| Development queries | 500 |
| Reserved test queries | 1,000 |
| Retrieval corpus passages | 66,895 |
| Development document groups | 452 |

Splits were formed from connected source families using document IDs, titles, duplicate questions and duplicate passages. Document families used in quality review were excluded from the final test set. AI Hub source text, derived examples and private review annotations are not distributed. The published preparation code describes the processing logic; exact reproduction of the reported split also requires the source data and curation annotations.

## Training procedure

All model parameters were fine-tuned with query-to-passage `CachedMultipleNegativesRankingLoss`. Other positives in the logical batch act as negatives. The sampler separates pairs sharing a document family, passage ID, or exactly duplicated normalized query or passage. These checks reduce identifiable conflicts; they do not identify every semantically relevant alternative passage.

| Setting | Value |
|---|---|
| Epochs / seed | 1 / 42 |
| Optimization steps | 738 |
| Learning rate | `2e-6` |
| Logical batch / microbatch | 128 / 4 |
| Maximum sequence length | 1,536 |
| Loss scale | 20 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Gradient clipping norm | 1.0 |
| Optimizer | AdamW |
| Trainable parameters | 1,140,918,272 |
| Parameters and optimizer states | FP32 |
| Forward computation | BF16 autocast |
| Training device | One NVIDIA RTX A6000 48 GB |
| Training time | 18,446 seconds, approximately 5.12 hours |
| Peak PyTorch CUDA allocated memory | 29.54 GiB |
| PyTorch | `2.11.0+cu128` |

The original saved checkpoint occupied 4,580,826,739 bytes. Release packaging adds metadata and splits weight files while preserving the FP32 tensors. Peak CUDA allocated memory measures PyTorch tensor allocations, not total process or system memory. Parameter values were verified to match exactly after saving and reloading the trained checkpoint.

## Evaluation

The original and fine-tuned models were evaluated on the same RTX A6000, using all 66,895 candidate passages for 500 development queries. Both were loaded with FP32 parameters and evaluated with BF16 autocast, SDPA attention, batch size 8 and maximum sequence length 1,536. Ranking used exact cosine search with normalized embeddings.

| Metric | Original model | HUFS BAI Legal Embed 1B | Absolute gain | Gain 95% CI |
|---|---:|---:|---:|---:|
| nDCG@10 | 0.7895 | **0.8293** | +0.0398 | [+0.0221, +0.0581] |
| MRR@10 | 0.7441 | **0.7897** | +0.0455 | [+0.0241, +0.0674] |
| Recall@10 | 0.928 | **0.950** | +0.022 | [+0.0021, +0.0420] |
| Recall@50 | 0.980 | **0.988** | +0.008 | [−0.0020, +0.0197] |

Confidence intervals use 10,000 paired bootstrap resamples over 452 document groups, with query-weighted aggregation and seed 42. nDCG@10 improves by **5.04% relative**. The Recall@50 point estimate increases, but its interval includes zero.

These are development-set results from one training recipe and seed. Repeated-seed evaluation has not been run. [Evaluation details](evaluation.md) describe the retrieval protocol and statistical comparison; [result JSON](../results/initial_dev_results.json) contains full-precision aggregates.

## Licenses and attribution

- **Model weights and inherited model materials:** [OpenMDW License Agreement 1.1](https://openmdw.ai/license/1-1/), with the applicable NVIDIA and HUFS BAI notices included in the release.
- **Repository code and documentation:** [Apache License 2.0](../LICENSE).
- **Training data:** AI Hub 금융·법률 문서 기계독해 데이터, dataset 71610. AI Hub's separate data terms apply to obtaining and using the data; the dataset is not included in this release.

Retain the model license and applicable notices when redistributing the model. See [NOTICE.md](../NOTICE.md) for provenance and source-policy links.

## Contact

For **domain-specific embedding fine-tuning, customized retrieval models, and research or industry collaboration**, contact **Prof. Juho Bai at [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)**.

**HUFS Business AI Lab · Hankuk University of Foreign Studies**

[Homepage](https://juho-bai.pages.dev/)
