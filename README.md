# HUFS BAI Legal Embed 1B

Korean legal-domain embedding model · 한국어 법률 도메인 임베딩 모델

`hufs-bai-legal-embed-1b-v0.1` is an embedding model from HUFS Business AI Lab, Hankuk University of Foreign Studies. It converts questions and document passages into 2,048-dimensional vectors for semantic search and retrieval-augmented generation (RAG). Full-parameter fine-tuning of [NVIDIA Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) on Korean legal question–passage pairs achieves 0.8293 nDCG@10, a 5.04% relative improvement over the original model, and 95.0% Recall@10 on the reported development set.

`hufs-bai-legal-embed-1b-v0.1`은 한국외국어대학교 비즈니스AI연구실(HUFS Business AI Lab)의 임베딩 모델입니다. 질문과 문단을 2,048차원 벡터로 변환하여 의미 기반 검색과 검색 증강 생성(RAG)에 활용할 수 있습니다. NVIDIA Nemotron-3-Embed-1B-BF16의 전체 파라미터를 한국어 법률 질문·근거 문단 쌍으로 학습했으며, 법률 검색 검증셋에서 nDCG@10 0.8293을 기록하여 원본 대비 상대 5.04% 향상됐습니다. Recall@10은 95.0%입니다.

[Download / 다운로드](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0) · [Model card (EN)](docs/model-card.md) · [모델 카드 (KO)](docs/model-card.ko.md) · [Contact / 문의](mailto:juho@hufs.ac.kr)

## Model overview · 모델 정보

| Property / 항목 | Value / 내용 |
|---|---|
| Model ID / 모델 ID | `hufs-bai-legal-embed-1b-v0.1` |
| Language and domain / 언어·도메인 | Korean legal documents / 한국어 법률 문서 |
| Uses / 활용 | Question-to-passage search, legal document search, RAG retrieval / 근거 문단 검색, 법률 문서 검색, RAG 검색기 |
| Parameters / 파라미터 | 1.14 billion, all fine-tuned / 약 11.4억 개, 전체 학습 |
| Embedding dimension / 임베딩 차원 | 2,048 |
| Pooling and similarity / 풀링·유사도 | Mean pooling, normalized cosine similarity / 평균 풀링, 정규화된 벡터의 코사인 유사도 |
| Training and evaluation input length / 학습·평가 입력 길이 | 1,536 tokens including the prompt / 접두어 포함 1,536 tokens |
| Query and document prompts / 질문·문단 접두어 | `query: ` / `passage: ` |
| Released weights / 공개 가중치 | Complete FP32 model, approximately 4.58 GB / FP32 전체 모델, 약 4.58 GB |
| Model license / 모델 라이선스 | [OpenMDW 1.1](https://openmdw.ai/license/1-1/) |

## Free download and quick start · 무료 다운로드와 사용법

The complete model is available free of charge through GitHub Releases. Weights are provided as three standard safetensors shards, each smaller than 2 GB, with an index for automatic loading. The download script retrieves and verifies all required files. No token, subscription, manual file concatenation, or separate base-model download is required. See the [download guide (EN)](docs/distribution.md) for details.

전체 모델을 GitHub Releases에서 무료로 제공합니다. 가중치는 파일당 2 GB 미만인 표준 safetensors 파일 3개와 자동 로딩용 인덱스로 구성됩니다. 다운로드 스크립트가 필요한 파일을 받고 무결성을 확인합니다. 토큰이나 구독, 파일 병합, 기반 모델의 별도 다운로드가 필요하지 않습니다. 자세한 내용은 [다운로드 안내 (KO)](docs/distribution.ko.md)를 참고하세요.

Use Python 3.11 or newer. Install the PyTorch build for your CPU or CUDA environment following the [setup guide](docs/setup.md), then run the commands below.

Python 3.11 이상을 사용합니다. [환경 설정 안내](docs/setup.md)에 따라 CPU 또는 CUDA 환경에 맞는 PyTorch를 설치한 뒤 다음 명령을 실행하세요.

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
python -m pip install -r requirements.txt
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

For CPU inference, add `--device cpu` to the retrieval command. The example ranks synthetic Korean passages while preserving the FP32 parameters. Supported CUDA devices use BF16 autocast, matching the reported evaluation precision. `encode_query` and `encode_document` apply the checkpoint's prompts automatically; do not prepend them again. Use this same model for both queries and indexed passages, and normalize the vectors for cosine retrieval.

CPU에서 실행하려면 마지막 명령에 `--device cpu`를 추가합니다. 예제는 FP32 파라미터를 유지하면서 가상의 한국어 문단을 검색합니다. 지원되는 CUDA 장치에서는 공개 평가와 같은 BF16 autocast 연산을 사용합니다. `encode_query`와 `encode_document`가 접두어를 자동 적용하므로 입력 문자열에 다시 붙이지 마세요. 질문과 검색 대상 문단에는 같은 모델을 사용하고, 정규화한 벡터의 코사인 유사도로 검색합니다.

## Retrieval quality · 검색 성능

Both models were evaluated on the same NVIDIA RTX A6000 with 500 development queries and the full corpus of 66,895 passages, using exact cosine search.

동일한 NVIDIA RTX A6000 환경에서 검증 질문 500개와 전체 검색 후보 66,895개 문단을 사용하여 정확 코사인 검색으로 평가했습니다.

| Model / 모델 | nDCG@10 ↑ | MRR@10 ↑ | Recall@10 ↑ | Recall@50 ↑ |
|---|---:|---:|---:|---:|
| Original / 원본 Nemotron-3-Embed-1B | 0.7895 | 0.7441 | 0.928 | 0.980 |
| HUFS BAI Legal Embed 1B v0.1 | 0.8293 | 0.7897 | 0.950 | 0.988 |

The nDCG@10 gain is +0.0398, with a paired document-group bootstrap 95% confidence interval of [+0.0221, +0.0581]. The corpus includes Korean legal policy, legislative and administrative documents. These are single-seed development-set results.

nDCG@10 개선량은 +0.0398이며, 문서군 단위 paired bootstrap 95% 신뢰구간은 [+0.0221, +0.0581]입니다. 평가 자료에는 법률 관련 정책·입법·행정 문서가 포함됩니다. 단일 seed의 검증셋 결과입니다.

[Evaluation protocol / 평가 절차](docs/evaluation.md) · [Result JSON / 결과 JSON](results/initial_dev_results.json)

## Training and provenance · 학습과 데이터 출처

The model was fully fine-tuned for one epoch using `CachedMultipleNegativesRankingLoss` with in-batch negatives. Training used 89,898 question–passage pairs from the legal portion of AI Hub's 금융·법률 문서 기계독해 데이터 (Financial and Legal Document Machine Reading Comprehension), [dataset 71610](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610). Source-family grouping and duplicate checks separate the data splits and reduce conflicting negatives within a batch.

배치 내 다른 정답 문단을 네거티브로 사용하는 `CachedMultipleNegativesRankingLoss`로 전체 파라미터를 1 epoch 학습했습니다. AI Hub 금융·법률 문서 기계독해 데이터, 데이터셋 71610의 법률 자료에서 질문·근거 문단 89,898쌍을 사용했습니다. 문서군 단위 분리와 중복 검사를 통해 데이터셋을 나누고 배치 내 네거티브 충돌을 줄였습니다.

The model cards ([EN](docs/model-card.md) / [KO](docs/model-card.ko.md)) provide the full training recipe. Training and evaluation code is available for users working with their own authorized data. AI Hub source documents and derived training/evaluation text are not redistributed.

모델 카드에 상세 학습 설정을 제공합니다. 이용 권한을 확보한 데이터로 추가 학습과 평가를 수행할 수 있도록 코드를 공개하며, AI Hub 원문이나 가공한 학습·평가 텍스트는 재배포하지 않습니다.

## License · 라이선스

The released model weights and inherited model materials are covered by [OpenMDW 1.1](https://openmdw.ai/license/1-1/). Repository code and documentation are provided under [Apache 2.0](LICENSE). Preserve the model's license and attribution files. Data provenance and separate AI Hub terms are described in [NOTICE.md](NOTICE.md).

공개 모델 가중치와 기반 모델에서 가져온 구성요소에는 OpenMDW 1.1이 적용됩니다. 저장소의 코드와 문서는 Apache 2.0으로 공개합니다. 모델에 포함된 라이선스와 출처 고지를 유지해 주세요. 학습 데이터 출처와 별도 AI Hub 이용 조건은 NOTICE.md에 안내되어 있습니다.

## Contact and domain-specific tuning · 전문 도메인 튜닝 문의

For domain-specific embedding fine-tuning, customized retrieval models, and research or industry collaboration, please contact Prof. Juho Bai at [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr).

전문 도메인에 맞춘 임베딩 모델 튜닝, 맞춤형 검색 모델 구축, 산학·연구 협력에 관심이 있으시면 배주호 교수 [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)로 연락해 주세요.

HUFS Business AI Lab · 한국외국어대학교 비즈니스AI연구실

[Homepage / 홈페이지](https://juho-bai.pages.dev/) · [GitHub issues / 이슈](https://github.com/juho127/hufs-bai-embedding-models/issues)
