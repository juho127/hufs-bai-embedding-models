# HUFS BAI Legal Embedding 모델 카드

## 개요

HUFS Business AI Lab은 한국어 법률 질문과 관련 근거 문단의 검색 정확도를 높이기 위해 NVIDIA Nemotron 임베딩 모델을 학습했습니다. 같은 데이터에 LoRA와 전체 파라미터 학습을 적용하고 각각 원본 대비 검색 성능을 측정합니다.

| 항목 | 내용 |
|---|---|
| 개발 | 한국외국어대학교 비즈니스AI연구실 / HUFS Business AI Lab |
| 도메인 | 한국어 법률 문서 검색 |
| 기반 모델 | `nvidia/Nemotron-3-Embed-1B-BF16` |
| 기반 revision | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| 기반 파라미터 수 | 1,140,918,272 |
| 임베딩 차원 | 2,048 |
| 인코더 | Ministral3, bidirectional attention (`is_causal=false`) |
| pooling | mean pooling, 접두어 토큰 포함 |
| 검색 | 정규화된 벡터의 코사인 유사도 |
| 실험 입력 길이 | 최대 1,536 tokens, 접두어 포함 |
| 질문 / 문단 접두어 | `query: ` / `passage: ` |

1,536 tokens는 이번 실험의 설정입니다. 원본 모델의 최대 지원 길이를 뜻하지 않습니다. 준비한 비표 법률 문단의 최대 길이는 접두어를 포함해 1,370 tokens로 측정됐습니다.

## 모델 상태

| 실험 모델 이름 | 학습 상태 | 가중치 공개 |
|---|---|---|
| `hufs_bai-legal-embed-1b-lora-v0.1` | 1 epoch 학습 및 검증셋 평가 완료 | 미공개 |
| `hufs_bai-legal-embed-1b-full-v0.1` | 1 epoch 학습 및 검증셋 평가 완료 | 미공개 |

위 이름은 프로젝트의 실험 모델 이름이며, 아직 Hugging Face 모델 ID가 아닙니다. 현재 제공하는 다운로드 스크립트는 NVIDIA 원본 모델을 받습니다. 향후 가중치 배포는 [배포 안내](distribution.md)에 기록합니다.

## 학습 데이터

[AI Hub 금융·법률 문서 기계독해 데이터, 데이터셋 71610](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610)의 법률 자료를 사용했습니다. 문단 추출 및 절차형 질문인 type 1·5를 학습 대상으로 삼고, 검색 후보에는 type 1·2·4·5의 비표 법률 문단을 포함했습니다.

| 항목 | 수 |
|---|---:|
| 준비된 학습 질문·문단 쌍 | 89,979 |
| seed 42에서 실제 사용한 쌍 | 89,898 |
| 배치 충돌로 남은 단독 쌍 | 81 |
| 학습 step | 738 |
| 검증 질문 | 500 |
| 최종 test 질문 | 1,000, 아직 평가하지 않음 |
| 검색 후보 문단 | 66,895 |

동일 문서 ID, 제목, 중복 질문·문단 등의 연결 관계를 이용해 문서군을 만든 후 train/dev/test를 분리했습니다. 품질 검토에 사용한 문서군은 최종 test에서 제외했습니다. 데이터 정제와 검색 평가 절차는 [평가 문서](evaluation.md)에 정리합니다.

## 학습 설정

공통 목적 함수는 query → passage 방향의 `CachedMultipleNegativesRankingLoss`입니다. 배치 내 다른 질문의 정답 문단을 네거티브로 사용합니다. 같은 문서군, 같은 문단 ID, 정규화 후 중복인 질문·문단은 한 배치에 함께 넣지 않습니다. 이 규칙은 명시적으로 확인 가능한 충돌을 줄이며, 의미상 관련된 모든 문단을 판별하는 규칙은 아닙니다.

| 설정 | LoRA | 전체 파라미터 학습 |
|---|---|---|
| epoch / seed | 1 / 42 | 1 / 42 |
| learning rate | `1e-5` | `2e-6` |
| logical / micro batch | 128 / 4 | 128 / 4 |
| 최대 입력 길이 | 1,536 | 1,536 |
| loss scale | 20 | 20 |
| warmup ratio | 0.1 | 0.1 |
| weight decay | 0.01 | 0.01 |
| gradient clipping norm | 1.0 | 1.0 |
| optimizer | AdamW | AdamW |
| 학습 파라미터 수 | 2,097,152 | 1,140,918,272 |
| 저장 파라미터 정밀도 | 고정 BF16 기반 모델 + FP32 adapter | FP32 |
| forward 연산 | BF16 autocast | BF16 autocast |
| optimizer state | FP32 | FP32 |

LoRA는 16개 layer의 `q_proj`와 `v_proj`에 rank 16, alpha 32, dropout 0으로 적용했습니다. 학습 파라미터 비율은 adapter를 포함한 모델 전체의 약 0.1835%입니다. 두 방법 모두 동일한 seed 42 배치 순서와 학습 쌍을 사용했고, 2개 이상의 호환 가능한 쌍이 남은 부분 배치를 유지했습니다.

## 자원 사용

다음은 실제 1 epoch 실행의 측정값입니다. 하드웨어가 달라 학습 시간의 차이를 방법 자체의 속도 차이로 해석할 수 없습니다.

| 항목 | LoRA | 전체 파라미터 학습 |
|---|---|---|
| 학습 장치 | NVIDIA GB10 | NVIDIA RTX A6000 48GB, 1장 |
| PyTorch | `2.11.0+cu130` | `2.11.0+cu128` |
| 학습 시간 | 23,455초, 약 6.52시간 | 18,446초, 약 5.12시간 |
| PyTorch 최대 CUDA allocated 메모리 | 9.67 GiB | 29.54 GiB |
| 저장 모델 디렉터리 크기 | 25,534,528 bytes, 기반 모델 별도 | 4,580,826,739 bytes |
| 그중 LoRA adapter 가중치 | 8,396,720 bytes | 해당 없음 |

메모리 수치는 프로세스 전체나 운영체제의 메모리 사용량이 아닙니다. LoRA 배포 파일만으로는 실행할 수 없으며 고정 revision의 기반 모델이 필요합니다. 저장·재로딩 후 학습 가중치 값의 일치를 두 방법 모두 확인했습니다.

## 평가와 사용 범위

전체 학습 모델의 검증셋 nDCG@10은 **0.8293**, 같은 환경에서 평가한 원본은 **0.7895**입니다. LoRA는 원본 **0.7904 → 0.7957**로 소폭 상승했지만, 개선량의 95% 신뢰구간 **[−0.0076, +0.0180]**에 0이 포함돼 개선을 통계적으로 확인하지 못했습니다. 수치와 신뢰구간, 결과의 적용 범위는 [평가 문서](evaluation.md)에 정리합니다.

이 모델의 출력은 검색용 벡터입니다. 법률 질문에 관련된 문단을 찾는 검색기 또는 RAG의 검색 구성요소로 사용할 수 있습니다. 검색 결과의 법적 타당성이나 현행성을 판정하는 기능은 별도로 평가해야 합니다.

## 라이선스와 출처

코드는 [Apache License 2.0](../LICENSE), NVIDIA 기반 모델은 [OpenMDW 1.1](https://openmdw.ai/license/1-1/)을 따릅니다. 학습 데이터는 AI Hub에서 제공하며 해당 데이터의 이용 조건이 별도로 적용됩니다. 원문과 가공 데이터는 이 저장소에서 재배포하지 않습니다. 학습 가중치의 공개 라이선스와 고지 사항은 가중치 배포 시 함께 제공합니다.

## 연락처

배주호 교수 (Juho Bai) · HUFS Business AI Lab

[홈페이지](https://juho-bai.pages.dev/) · [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)
