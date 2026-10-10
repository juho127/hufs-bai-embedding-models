# HUFS BAI Defense Embed 1B v0.2

English and Korean model card · 한·영 통합 모델 카드

`hufs-bai-defense-embed-1b-v0.2` is a Korean defense-domain embedding model from HUFS Business AI Lab, Hankuk University of Foreign Studies. Full-parameter fine-tuning on public Ministry of National Defense briefing question–answer pairs improves nDCG@10 from 0.7397 to 0.7947 on the reported held-out retrieval test.

`hufs-bai-defense-embed-1b-v0.2`는 한국외국어대학교 비즈니스AI연구실의 한국어 국방 도메인 임베딩 모델입니다. 공개된 국방부 브리핑의 질문·답변 쌍으로 전체 파라미터를 학습했으며, 보류 검색 test에서 nDCG@10이 원본 0.7397에서 0.7947로 향상됐습니다.

## Model and intended use · 모델과 활용

| Property / 항목 | Value / 내용 |
|---|---|
| Release / 공개 버전 | [defense-v0.2.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/defense-v0.2.0) |
| Base model / 기반 모델 | [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) |
| Base revision / 기반 버전 | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| Parameters / 파라미터 | 1,140,918,272, all fine-tuned / 전체 파라미터 학습 |
| Embedding dimension / 임베딩 차원 | 2,048 |
| Pooling / 풀링 | Mean pooling including prompt tokens / 접두어를 포함한 평균 풀링 |
| Similarity / 유사도 | Cosine similarity after normalization / 정규화한 벡터의 코사인 유사도 |
| Query / document prompts / 질문·문단 접두어 | `query: ` / `passage: ` |
| Training and evaluation length / 학습·평가 길이 | 1,536 tokens including the prompt / 접두어 포함 |
| Released precision / 공개 정밀도 | FP32 |
| Model license / 모델 라이선스 | [OpenMDW 1.1](https://openmdw.ai/license/1-1/) |

Use the model for Korean public defense briefing search, relevant-passage retrieval and RAG retrieval over authorized document collections. It outputs vectors rather than generated answers. The 1,536-token setting is the evaluated input length; performance on longer inputs, other languages, operational documents and other corpus distributions has not been measured. Source statements reflect their publication dates.

한국어 공개 국방 브리핑 검색, 질문에 관련된 근거 문단 검색, 이용 권한을 확보한 문서에 대한 RAG 검색기로 활용할 수 있습니다. 모델의 출력은 임베딩 벡터입니다. 1,536 tokens는 학습·평가에 사용한 입력 길이이며 더 긴 입력, 다른 언어, 작전 문서 및 다른 분포의 자료는 평가하지 않았습니다. 검색되는 원문의 내용은 각 발행 시점을 기준으로 합니다.

## Download and inference · 다운로드와 추론

The complete FP32 model is provided as three safetensors shards, each smaller than 2 GB, plus an automatic-loading index. The package includes the tokenizer, configuration, license and attribution files. No separate base-model download is required. Install dependencies as described in the [repository setup guide](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/docs/setup.md), then run from the repository root:

FP32 전체 모델을 파일당 2 GB 미만인 safetensors 파일 3개와 자동 로딩 인덱스로 제공합니다. 토크나이저, 설정, 라이선스, 출처 고지를 함께 포함하며 기반 모델을 별도로 받을 필요가 없습니다. [저장소 환경 설정 안내](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/docs/setup.md)에 따라 의존성을 설치한 뒤 저장소 루트에서 실행합니다.

```bash
python scripts/download_release.py --manifest releases/defense-v0.2.0/manifest.json --output models/hufs-bai-defense-embed-1b-v0.2
python examples/retrieve_defense.py --model models/hufs-bai-defense-embed-1b-v0.2
```

| Weight file / 가중치 파일 | Bytes / 바이트 |
|---|---:|
| `model-00001-of-00003.safetensors` | 1,778,445,520 |
| `model-00002-of-00003.safetensors` | 1,795,301,096 |
| `model-00003-of-00003.safetensors` | 989,942,248 |

Keep all three shards and `model.safetensors.index.json` in the same model directory. Standard loading uses the index automatically; the files are not manually concatenated. File-by-file checksums are in the [release manifest](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/releases/defense-v0.2.0/manifest.json) and the release's `SHA256SUMS.txt`. Packaging preserves all 146 FP32 tensors exactly; CPU embeddings and cosine scores matched the source checkpoint with zero maximum absolute error on the synthetic packaging check.

가중치 파일 3개와 `model.safetensors.index.json`을 같은 모델 폴더에 둡니다. 표준 로더가 인덱스를 자동 사용하므로 파일을 직접 이어 붙이지 않습니다. 파일별 해시는 release manifest와 공개 패키지의 `SHA256SUMS.txt`에 있습니다. 분할 패키징은 FP32 텐서 146개의 값을 정확히 보존하며, 가상 문장으로 확인한 CPU 임베딩과 코사인 점수는 원래 체크포인트와 최대 절대 오차 0으로 일치했습니다.

The downloader verifies each file against the release manifest's SHA256 and size. The example preserves FP32 parameters and uses BF16 autocast on supported CUDA devices. Add `--device cpu` for FP32 CPU inference. `encode_query` and `encode_document` apply their prompts automatically; do not add these prompts twice. Encode queries and passages with the same model and normalize the vectors. Example passages are synthetic and are not evaluation questions or answers.

다운로더는 release manifest에 기록한 SHA256과 파일 크기를 확인합니다. 예제는 FP32 파라미터를 유지하며 지원되는 CUDA 장치에서 BF16 autocast를 사용합니다. `--device cpu`를 추가하면 FP32 CPU 추론을 수행합니다. `encode_query`와 `encode_document`가 접두어를 자동으로 적용하므로 중복해서 붙이지 마세요. 질문과 문단은 같은 모델로 인코딩하고 벡터를 정규화합니다. 사용 예제의 문단은 가상 자료이며 실제 평가 질답이 아닙니다.

## Training data · 학습 데이터

Training uses text from the Ministry of National Defense's daily briefings, published through Korea Policy Briefing with item-level KOGL Type 1 attribution notices. Semantic AI review read all 1,200 candidate question–answer pairs and accepted 353 that directly answer the questions' core requirements. Nonanswers, evasions, incomplete answers and unreviewed pairs were excluded. The reviewed material was split by document/source family.

대한민국 정책브리핑에서 문서별 공공누리 제1유형 출처표시 고지를 확인한 국방부 일일 정례 브리핑 본문을 사용했습니다. AI 의미 검수에서 후보 1,200쌍의 질문과 답변을 모두 읽고 핵심 요구에 직접 답하는 353쌍을 승인했습니다. 미답변, 회피, 부분 답변과 미검수 질답을 제외하고 문서·출처군 단위로 분리했습니다.

| Data / 자료 | Pairs or passages / 쌍 또는 문단 수 | Source groups / 문서군 |
|---|---:|---:|
| Training / 학습 | 283 | 185 |
| Development / 검증 | 35 | 23 |
| Held-out test / 보류 test | 35 | 24 |
| Shared retrieval corpus / 공통 검색 corpus | 4,904 passages / 문단 | |

Collection covered 597 licensed briefings, 3,150 public defense policy documents, and 341 selected pages from three National Assembly Budget Office reports. Policy and budget documents contributed only retrieval candidates; they were not counted as supervised training pairs and were not used for negative mining. The release contains source attribution metadata, not the underlying text or derived training/evaluation pairs. Training source titles, publishers, dates and links appear in [TRAINING_SOURCES.json](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/releases/defense-v0.2.0/TRAINING_SOURCES.json).

수집 자료는 이용조건을 확인한 브리핑 597건, 공개 국방 정책문서 3,150건, 국회예산정책처 보고서 3건에서 선별한 341쪽입니다. 정책·예산 자료는 검색 후보에만 포함했고 지도학습 질답 수에 포함하거나 네거티브 채굴에 사용하지 않았습니다. 공개 패키지는 원문이나 가공한 학습·평가 질답을 포함하지 않으며 출처 메타데이터를 제공합니다. 학습에 사용한 문서의 제목, 발행기관, 발행일, 원문 링크는 [TRAINING_SOURCES.json](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/releases/defense-v0.2.0/TRAINING_SOURCES.json)에 있습니다.

## Training procedure · 학습 방법

All parameters were fine-tuned from the original NVIDIA checkpoint using query-to-passage `CachedMultipleNegativesRankingLoss`. Other positives in a logical batch serve as negatives. The sampler separates pairs sharing a source family, positive passage or normalized duplicate question/answer. Loss inputs contain only the training queries and their positive passages.

원본 NVIDIA 체크포인트에서 `CachedMultipleNegativesRankingLoss`로 전체 파라미터를 학습했습니다. 논리 배치 내 다른 정답 문단을 네거티브로 사용하며, 같은 출처군·정답 문단·정규화한 중복 질답을 공유하는 쌍은 배치에서 분리합니다. 손실 계산에는 학습 질문과 해당 정답 문단만 사용했습니다.

| Setting / 설정 | Value / 값 |
|---|---|
| Epochs / seed | 5 / 42 |
| Checkpoint rule / 모델 선택 규칙 | Fixed final epoch 5 / 사전 고정한 마지막 5 epoch |
| Optimizer / 학습률 | AdamW / `2e-6` |
| Logical batch maximum / microbatch | 128 / 4 |
| Optimization steps / 학습 step | 30 |
| Loss scale / Warmup ratio / Weight decay | 20 / 0.1 / 0.01 |
| Maximum sequence length / 최대 입력 길이 | 1,536 |
| Parameters and optimizer states / 파라미터·옵티마이저 상태 | FP32 |
| Forward computation / 순전파 | BF16 autocast |
| Training device / 학습 장치 | One NVIDIA RTX A6000 48 GB / 1대 |

Each of the 283 training pairs was used once per epoch. All 146 trainable tensors were updated. Saved and reloaded parameters matched exactly. The learning rate, seed and final epoch were fixed before the run; development or test scores did not select the checkpoint, and no retraining followed the final test.

학습 283쌍을 매 epoch 한 번씩 사용했으며 학습 대상 텐서 146개가 모두 갱신됐습니다. 저장 후 재로딩한 파라미터가 정확히 일치함을 확인했습니다. 학습률, seed, 마지막 epoch를 실행 전에 고정했고 dev·test 점수로 체크포인트를 선택하거나 최종 test 이후 다시 학습하지 않았습니다.

## Held-out retrieval test · 보류 검색 test

Both models ranked the same 4,904 passages for 35 queries from 24 held-out source groups. Evaluation used exact cosine search, normalized embeddings, FP32 parameters, BF16 autocast, SDPA attention, batch size 8 and maximum input length 1,536 on the same NVIDIA RTX A6000. Each query has one labeled positive; unjudged passages are treated as nonrelevant, with ties resolved by corpus order.

원본과 국방 모델은 보류 문서군 24개에서 나온 질문 35개에 대해 동일한 문단 4,904개를 검색했습니다. 동일한 NVIDIA RTX A6000에서 벡터 정규화, 정확 코사인 검색, FP32 파라미터, BF16 autocast, SDPA attention, batch size 8, 최대 입력 길이 1,536을 적용했습니다. 질문마다 정답 문단 하나를 지정했으며 미판정 문단은 비관련 문단으로 처리하고 동점은 corpus 순서로 정렬했습니다.

| Metric / 지표 | Original / 원본 | Defense v0.2 / 국방 | Absolute gain / 개선량 | Gain 95% CI / 개선량 95% 신뢰구간 |
|---|---:|---:|---:|---|
| nDCG@10 | 0.7397 | 0.7947 | +0.0551 | [+0.0239, +0.0909] |
| MRR@10 | 0.6946 | 0.7548 | +0.0602 | [+0.0268, +0.0957] |
| Recall@10 | 0.8857 | 0.9143 | +0.0286 | [0.0000, +0.0968] |
| Recall@50 | 0.9143 | 0.9714 | +0.0571 | [0.0000, +0.1471] |

Intervals use 10,000 paired percentile bootstrap resamples over the 24 document groups, retaining all queries in each sampled group with query-weighted aggregation and seed 42. The nDCG@10 and MRR@10 intervals exclude zero; the Recall point estimates improve, but their intervals include zero. Full-precision aggregates are in [defense_v2_test_results.json](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/results/defense_v2_test_results.json).

문서군 24개를 단위로 paired percentile bootstrap을 10,000회 수행했습니다. 표집된 문서군의 모든 질문을 유지하고 질문 수로 가중 집계했으며 seed는 42입니다. nDCG@10과 MRR@10의 신뢰구간은 0을 포함하지 않습니다. Recall 점추정치는 향상됐지만 신뢰구간은 0을 포함합니다. 반올림하지 않은 집계 결과는 [defense_v2_test_results.json](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/results/defense_v2_test_results.json)에 있습니다.

## Evidence scope · 결과의 범위

These results use a single training seed, 35 held-out questions and AI semantic-reviewed labels, without expert annotation. They support improved retrieval on this fixed in-domain set. The group bootstrap does not measure variation across training seeds. Audit found no direct held-out supervision in training, no exact normalized question/answer overlap across splits and no near-literal overlap at a character-12-shingle Jaccard threshold of 0.90. Nevertheless, different dated briefings do share recurring events and some semantic content. The split is document/source-family based; event-disjoint and future-time generalization have not been established. Inclusion of these texts in the base model's pretraining is unknown.

단일 학습 seed, 보류 질문 35개, 전문가 주석이 아닌 AI 의미 검수 정답을 사용한 결과로, 이 고정된 도메인 내 평가셋의 검색 성능 향상을 보여줍니다. 문서군 bootstrap은 학습 seed 간 변동을 추정하지 않습니다. 점검에서는 보류 질답의 직접 학습 사용, 분할 간 정규화 질답의 완전 중복, 문자 12-shingle Jaccard 0.90 이상인 유사 중복이 발견되지 않았습니다. 다만 다른 날짜의 브리핑 사이에 반복 사건과 일부 의미 내용이 겹칩니다. 문서·출처군 단위 분리이며, 독립 사건이나 미래 시점에 대한 일반화는 확인하지 않았습니다. 기반 모델의 사전학습에 해당 텍스트가 포함됐는지는 알 수 없습니다.

## License, attribution and contact · 라이선스·출처·문의

Model weights and inherited model materials are provided under [OpenMDW 1.1](https://openmdw.ai/license/1-1/), with the NVIDIA license and original model card included. Repository code and documentation use Apache 2.0. The briefing text was used under [KOGL Type 1](https://www.kogl.or.kr/info/licenseType1.do), which permits commercial use and modification with attribution. Source-data conditions remain separate from the model license. Retain applicable notices when redistributing. See [NOTICE.md](https://github.com/juho127/hufs-bai-embedding-models/blob/defense-v0.2.0/NOTICE.md) for source attribution; the source institutions do not sponsor or endorse this model.

모델 가중치와 기반 모델 구성요소에는 OpenMDW 1.1을 적용하며 NVIDIA 라이선스와 원본 모델 카드를 함께 제공합니다. 저장소 코드와 문서는 Apache 2.0입니다. 브리핑 텍스트는 출처표시 조건으로 상업적 이용과 변경을 허용하는 공공누리 제1유형에 따라 사용했습니다. 원천 데이터 이용 조건과 모델 라이선스는 구분되며 재배포 시 적용되는 고지를 유지해야 합니다. 상세 출처는 NOTICE.md에 있습니다. 원천 자료 기관이 이 모델을 후원하거나 보증하는 것은 아닙니다.

For domain-specific fine-tuning, custom retrieval models and research or industry collaboration, contact Prof. Juho Bai at [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr).

전문 도메인 특화 튜닝, 맞춤형 검색 모델 구축, 산학·연구 협력 문의는 배주호 교수 [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)로 연락해 주세요.

HUFS Business AI Lab · 한국외국어대학교 비즈니스AI연구실 · [Homepage / 홈페이지](https://juho-bai.pages.dev/)
