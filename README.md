# HUFS BAI Embedding Models

Korean domain-specific embedding models by HUFS Business AI Lab, starting with legal retrieval using NVIDIA Nemotron, LoRA, and full fine-tuning.

한국외국어대학교 비즈니스AI연구실의 한국어 도메인 특화 임베딩 모델 프로젝트입니다. 첫 과제는 **법률 질문에 필요한 근거 문단을 더 정확하게 검색하는 것**입니다. NVIDIA Nemotron을 기반으로 LoRA와 전체 파라미터 학습을 수행하고, 각각 원본 모델 대비 검색 성능을 평가합니다.

## 초기 결과

전체 파라미터를 1 epoch 학습한 모델은 검증셋에서 **nDCG@10이 0.7895에서 0.8293으로, 상대 5.04% 향상**됐습니다. 개선량의 문서군 단위 paired bootstrap 95% 신뢰구간은 **[+0.0221, +0.0581]**입니다.

| 모델 | 평가 환경 | nDCG@10 ↑ | MRR@10 ↑ | Recall@10 ↑ | Recall@50 ↑ |
|---|---|---:|---:|---:|---:|
| 원본 Nemotron | A6000 | 0.7895 | 0.7441 | 0.928 | 0.980 |
| HUFS BAI Legal Full | A6000 | **0.8293** | **0.7897** | **0.950** | **0.988** |
| 원본 Nemotron | GB10 | 0.7904 | 0.7454 | 0.928 | 0.980 |
| HUFS BAI Legal LoRA | GB10 | 0.7957 | 0.7497 | 0.936 | 0.980 |

LoRA의 nDCG@10은 0.7904에서 0.7957로 소폭 올랐지만, 개선량의 95% 신뢰구간 **[−0.0076, +0.0180]**에 0이 포함돼 이번 실험에서 개선을 통계적으로 확인하지 못했습니다.

2026-10-10 기준, 검증 질문 500개와 검색 문단 66,895개를 사용한 seed 42 초기 실험입니다. 각 모델은 같은 환경에서 평가한 원본과 비교합니다. 최종 test 1,000문항과 여러 seed의 반복 평가는 아직 수행하지 않았습니다. 평가 방법과 결과의 적용 범위는 [평가 문서](docs/evaluation.md), 수치는 [결과 JSON](results/initial_dev_results.json)에 정리합니다.

## 모델과 학습

| 항목 | 설정 |
|---|---|
| 기반 모델 | [NVIDIA Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) |
| 고정 revision | `c0c9fea93ea424587517f2c59e20db9f1d6bf615` |
| 임베딩 | 2,048차원, mean pooling, 정규화 후 코사인 유사도 |
| 학습 목적 | `CachedMultipleNegativesRankingLoss`, query → passage |
| 네거티브 | 배치 안의 다른 정답 문단을 사용하며 같은 문서군·중복 질문·중복 문단은 배치에서 분리 |
| 학습 데이터 | AI Hub 금융·법률 문서 기계독해의 법률 질문·근거 문단 89,979쌍 |
| 공통 설정 | 1 epoch, logical batch 128, micro batch 4, 최대 1,536 tokens |
| LoRA | rank 16, alpha 32, `q_proj`·`v_proj`, learning rate `1e-5` |
| 전체 학습 | 모든 파라미터 업데이트, learning rate `2e-6` |

모델 구조, 정밀도, 학습 자원은 [모델 카드](docs/model-card.md)에서 확인할 수 있습니다.

## 시작하기

Python 3.11 이상과 환경에 맞는 PyTorch를 설치합니다. CUDA 및 CPU 설치 방법은 [환경 설정](docs/setup.md)을 따릅니다.

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
pip install -r requirements.txt
python scripts/download_model.py
python examples/retrieve.py --model models/nemotron-3-embed-1b-bf16
```

이 예제는 고정 revision의 **원본 Nemotron 모델**을 다운로드해 검색합니다. HUFS BAI 학습 가중치는 아직 공개하지 않았습니다. 공개 시 모델 ID, 체크섬, 다운로드 방법을 [배포 안내](docs/distribution.md)에 추가합니다.

질문에는 `query: `, 문단에는 `passage: ` 접두어를 적용합니다. 제공된 예제와 평가 스크립트는 모델의 `encode_query`·`encode_document` 설정을 사용하므로 입력 문자열에 접두어를 다시 붙이지 않습니다.

## 데이터와 재현

실제 법률 실험에는 [AI Hub 금융·법률 문서 기계독해 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610)를 사용합니다. 사용자는 AI Hub에서 해당 자료의 이용 조건을 확인하고 직접 이용 승인을 받아야 합니다. 이 저장소에는 원문, 가공 법률 데이터, 모델 가중치를 포함하지 않습니다.

코드에는 학습, 검색 평가, 문서군 단위 bootstrap 비교, 가중치 저장·재로딩 검증이 포함됩니다. 직접 보유한 데이터의 준비와 실행은 [환경 설정](docs/setup.md), 평가 입력 형식과 비교 절차는 [평가 문서](docs/evaluation.md)를 참고하세요.

## 라이선스

이 저장소의 코드는 [Apache License 2.0](LICENSE)으로 공개합니다. 기반 모델의 [OpenMDW 1.1](https://openmdw.ai/license/1-1/)과 AI Hub 데이터 이용 조건은 각각 별도로 적용됩니다. 코드 라이선스가 기반 모델이나 데이터의 이용 조건을 대체하지 않습니다.

## 연구실

**HUFS Business AI Lab · 한국외국어대학교 비즈니스AI연구실**

연락처: 배주호 교수 (Juho Bai) · [홈페이지](https://juho-bai.pages.dev/) · [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)

문제 보고와 재현 결과는 [GitHub Issues](https://github.com/juho127/hufs-bai-embedding-models/issues)에 남겨 주세요. 법률 원문이나 개인정보를 포함한 데이터는 이슈에 첨부하지 마세요.
