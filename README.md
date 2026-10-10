# HUFS BAI Embedding Models

Korean domain-specific embedding models · 한국어 전문 도메인 임베딩 모델

HUFS Business AI Lab, Hankuk University of Foreign Studies, provides embedding models for Korean legal and defense document retrieval. Each model converts questions and passages into 2,048-dimensional vectors for semantic search and retrieval-augmented generation (RAG). Both models were fully fine-tuned from [NVIDIA Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).

한국외국어대학교 비즈니스AI연구실(HUFS Business AI Lab)의 한국어 법률·국방 도메인 임베딩 모델입니다. 질문과 문단을 2,048차원 벡터로 변환하여 의미 기반 검색과 검색 증강 생성(RAG)에 활용할 수 있습니다. 두 모델 모두 NVIDIA Nemotron-3-Embed-1B-BF16의 전체 파라미터를 각 도메인의 질문·근거 문단 쌍으로 학습했습니다.

## Available models · 공개 모델

| Model / 모델 | Domain / 도메인 | Download / 다운로드 | Model card / 모델 카드 |
|---|---|---|---|
| `hufs-bai-defense-embed-1b-v0.2` | Korean public defense briefings and document retrieval / 한국어 공개 국방 브리핑·문서 검색 | [defense-v0.2.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/defense-v0.2.0) | [EN · KO](docs/defense-model-card.md) |
| `hufs-bai-legal-embed-1b-v0.1` | Korean legal documents / 한국어 법률 문서 | [v0.1.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0) | [EN](docs/model-card.md) · [KO](docs/model-card.ko.md) |

The defense model achieves 0.7947 nDCG@10 on its held-out test set, compared with 0.7397 for the original model. The legal model achieves 0.8293 nDCG@10 on its reported development set, compared with 0.7895 for the original model. These are separate retrieval tasks with different evaluation sets.

국방 모델은 보류 test에서 nDCG@10 0.7947을 기록했으며 원본 모델은 0.7397입니다. 법률 모델은 공개한 검증셋에서 nDCG@10 0.8293을 기록했으며 원본 모델은 0.7895입니다. 두 결과는 평가 자료가 서로 다른 별개의 검색 과제입니다.

## Model specifications · 모델 사양

| Property / 항목 | Both models / 두 모델 공통 |
|---|---|
| Parameters / 파라미터 | 1,140,918,272, all fine-tuned / 약 11.4억 개, 전체 학습 |
| Embedding dimension / 임베딩 차원 | 2,048 |
| Pooling and similarity / 풀링·유사도 | Mean pooling, normalized cosine similarity / 평균 풀링, 정규화된 벡터의 코사인 유사도 |
| Training and evaluation input length / 학습·평가 입력 길이 | 1,536 tokens including the prompt / 접두어 포함 1,536 tokens |
| Query and document prompts / 질문·문단 접두어 | `query: ` / `passage: ` |
| Released weights / 공개 가중치 | Complete FP32 model, approximately 4.58 GB per model / 모델별 FP32 전체 가중치, 약 4.58 GB |
| Model license / 모델 라이선스 | [OpenMDW 1.1](https://openmdw.ai/license/1-1/) |

## Free download and quick start · 무료 다운로드와 사용법

Each complete model is available free of charge through GitHub Releases. Its weights use three standard safetensors shards, each smaller than 2 GB, with an index for automatic loading. The download script verifies file sizes and SHA256 checksums. No token, subscription, manual file concatenation, or separate base-model download is required.

모델별 전체 가중치를 GitHub Releases에서 무료로 제공합니다. 가중치는 파일당 2 GB 미만인 표준 safetensors 파일 3개와 자동 로딩용 인덱스로 구성됩니다. 다운로드 스크립트가 파일 크기와 SHA256을 확인합니다. 토큰이나 구독, 파일 병합, 기반 모델의 별도 다운로드가 필요하지 않습니다.

Use Python 3.11 or newer. Install the PyTorch build for your CPU or CUDA environment following the [setup guide](docs/setup.md), then install the repository dependencies.

Python 3.11 이상을 사용합니다. [환경 설정 안내](docs/setup.md)에 따라 CPU 또는 CUDA 환경에 맞는 PyTorch를 설치한 뒤 저장소 의존성을 설치하세요.

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
python -m pip install -r requirements.txt
```

Defense model / 국방 모델:

```bash
python scripts/download_release.py --manifest releases/defense-v0.2.0/manifest.json --output models/hufs-bai-defense-embed-1b-v0.2
python examples/retrieve_defense.py --model models/hufs-bai-defense-embed-1b-v0.2
```

Legal model / 법률 모델:

```bash
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

For CPU inference, add `--device cpu` to the retrieval command. The examples rank synthetic Korean passages while preserving FP32 parameters. Supported CUDA devices use BF16 autocast, matching the reported evaluation precision. `encode_query` and `encode_document` apply the checkpoint's prompts automatically; do not prepend them again. Use the same model for both queries and indexed passages, and normalize the vectors for cosine retrieval.

CPU에서 실행하려면 검색 명령에 `--device cpu`를 추가합니다. 예제는 FP32 파라미터를 유지하면서 가상의 한국어 문단을 검색합니다. 지원되는 CUDA 장치에서는 공개 평가와 같은 BF16 autocast 연산을 사용합니다. `encode_query`와 `encode_document`가 접두어를 자동 적용하므로 입력 문자열에 다시 붙이지 마세요. 질문과 검색 대상 문단에는 같은 모델을 사용하고, 정규화한 벡터의 코사인 유사도로 검색합니다.

## Defense retrieval quality · 국방 검색 성능

The original and defense models were evaluated using the same 35 held-out test queries from 24 document groups and the same corpus of 4,904 passages. Exact cosine retrieval used FP32 parameters and BF16 autocast on an NVIDIA RTX A6000.

원본과 국방 모델을 동일한 보류 test 질문 35개, 문서군 24개, 검색 후보 4,904개 문단으로 평가했습니다. NVIDIA RTX A6000에서 FP32 파라미터와 BF16 autocast를 사용하여 정확 코사인 검색을 수행했습니다.

| Model / 모델 | nDCG@10 ↑ | MRR@10 ↑ | Recall@10 ↑ | Recall@50 ↑ |
|---|---:|---:|---:|---:|
| Original / 원본 Nemotron-3-Embed-1B | 0.7397 | 0.6946 | 0.8857 | 0.9143 |
| HUFS BAI Defense Embed 1B v0.2 | 0.7947 | 0.7548 | 0.9143 | 0.9714 |

The nDCG@10 gain is +0.0551, with a paired document-group bootstrap 95% confidence interval of [+0.0239, +0.0909]. The nDCG@10 and MRR@10 intervals exclude zero; the Recall intervals include zero. Results use one training seed and AI-reviewed relevance labels. The fixed epoch-5 checkpoint was evaluated once on test without test-based selection or further training. Splits separate document/source families; recurring events and some semantic content overlap across dated briefings. The results describe retrieval within this domain, with event-disjoint and temporal generalization untested.

nDCG@10 개선량은 +0.0551이며 문서군 단위 paired bootstrap 95% 신뢰구간은 [+0.0239, +0.0909]입니다. nDCG@10과 MRR@10의 신뢰구간은 0을 포함하지 않으며 Recall의 신뢰구간은 0을 포함합니다. 단일 학습 seed와 AI 의미 검수 정답을 사용한 결과입니다. 사전에 고정한 5 epoch 모델을 test에서 1회 평가했으며 test를 이용한 모델 선택이나 추가 학습은 하지 않았습니다. 문서·출처군을 분리했으나 다른 날짜의 브리핑 사이에 반복 사건과 일부 의미 내용이 겹칩니다. 결과의 범위는 해당 도메인 내 검색 성능이며 사건별 독립 평가와 시간 순서에 따른 일반화는 검증하지 않았습니다.

[Defense model card / 국방 모델 카드](docs/defense-model-card.md) · [Result JSON / 결과 JSON](results/defense_v2_test_results.json)

## Legal retrieval quality · 법률 검색 성능

The original and legal models were evaluated on the same NVIDIA RTX A6000 with 500 development queries and the full corpus of 66,895 passages, using exact cosine search.

동일한 NVIDIA RTX A6000 환경에서 검증 질문 500개와 전체 검색 후보 66,895개 문단을 사용하여 정확 코사인 검색으로 평가했습니다.

| Model / 모델 | nDCG@10 ↑ | MRR@10 ↑ | Recall@10 ↑ | Recall@50 ↑ |
|---|---:|---:|---:|---:|
| Original / 원본 Nemotron-3-Embed-1B | 0.7895 | 0.7441 | 0.928 | 0.980 |
| HUFS BAI Legal Embed 1B v0.1 | 0.8293 | 0.7897 | 0.950 | 0.988 |

The nDCG@10 gain is +0.0398, a 5.04% relative improvement, with a paired document-group bootstrap 95% confidence interval of [+0.0221, +0.0581]. The corpus includes Korean legal policy, legislative and administrative documents. These are single-seed development-set results.

nDCG@10 개선량은 +0.0398, 상대 향상률은 5.04%이며, 문서군 단위 paired bootstrap 95% 신뢰구간은 [+0.0221, +0.0581]입니다. 평가 자료에는 법률 관련 정책·입법·행정 문서가 포함됩니다. 단일 seed의 검증셋 결과입니다.

[Legal evaluation protocol / 법률 평가 절차](docs/evaluation.md) · [Result JSON / 결과 JSON](results/initial_dev_results.json)

## Training and provenance · 학습과 데이터 출처

Both models use query-to-passage `CachedMultipleNegativesRankingLoss`, with other positives in the logical batch acting as negatives. Source-family grouping and duplicate checks separate splits and reduce conflicting negatives within a batch.

두 모델은 배치 내 다른 정답 문단을 네거티브로 사용하는 `CachedMultipleNegativesRankingLoss`로 학습했습니다. 문서군 단위 분리와 중복 검사를 통해 데이터셋을 나누고 배치 내 네거티브 충돌을 줄였습니다.

| Model / 모델 | Training pairs / 학습 쌍 | Epochs | Training source / 학습 출처 |
|---|---:|---:|---|
| Defense v0.2 / 국방 | 283 | 5 | Ministry of National Defense daily briefing text under KOGL Type 1, via Korea Policy Briefing / 대한민국 정책브리핑에 공개된 국방부 일일 정례 브리핑, 공공누리 제1유형 본문 텍스트 |
| Legal v0.1 / 법률 | 89,898 | 1 | Legal portion of AI Hub [dataset 71610](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610), 금융·법률 문서 기계독해 데이터 / 해당 데이터의 법률 부분 |

For defense, semantic review accepted 353 of 1,200 candidate question–answer pairs; 283 were used for training and 35 each for development and test. Public defense policy and National Assembly Budget Office documents contributed retrieval candidates only, not training pairs. See the [training-source attribution](releases/defense-v0.2.0/TRAINING_SOURCES.json) and model cards for details. Source text and derived training/evaluation text are not included in these releases.

국방 자료는 후보 질답 1,200쌍을 의미 검수하여 353쌍을 승인했으며, 학습 283쌍과 dev·test 각 35쌍으로 나눴습니다. 공개 국방 정책문서와 국회예산정책처 자료는 검색 후보로만 사용했으며 학습 질답 수에 포함하지 않았습니다. 상세 출처는 [학습 자료 출처 고지](releases/defense-v0.2.0/TRAINING_SOURCES.json)와 모델 카드에서 확인할 수 있습니다. 원문이나 가공한 학습·평가 텍스트는 공개 패키지에 포함하지 않습니다.

## License · 라이선스

Released model weights and inherited model materials are covered by [OpenMDW 1.1](https://openmdw.ai/license/1-1/). Repository code and documentation are provided under [Apache 2.0](LICENSE). Preserve the model's license and applicable attribution files when redistributing. Defense data attribution and the separate AI Hub terms for the legal model are described in [NOTICE.md](NOTICE.md).

공개 모델 가중치와 기반 모델에서 가져온 구성요소에는 OpenMDW 1.1이 적용됩니다. 저장소의 코드와 문서는 Apache 2.0으로 공개합니다. 재배포 시 모델에 포함된 라이선스와 적용되는 출처 고지를 유지해 주세요. 국방 자료 출처와 법률 모델의 별도 AI Hub 이용 조건은 NOTICE.md에 안내되어 있습니다.

## Contact and domain-specific tuning · 전문 도메인 튜닝 문의

For domain-specific embedding fine-tuning, customized retrieval models, and research or industry collaboration, please contact Prof. Juho Bai at [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr).

전문 도메인에 맞춘 임베딩 모델 튜닝, 맞춤형 검색 모델 구축, 산학·연구 협력에 관심이 있으시면 배주호 교수 [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)로 연락해 주세요.

HUFS Business AI Lab · 한국외국어대학교 비즈니스AI연구실

[Homepage / 홈페이지](https://juho-bai.pages.dev/) · [GitHub issues / 이슈](https://github.com/juho127/hufs-bai-embedding-models/issues)
