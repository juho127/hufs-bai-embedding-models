# HUFS BAI Legal Embed 1B

[English](README.md) · **한국어**

**`hufs-bai-legal-embed-1b-v0.1`**은 **한국외국어대학교 비즈니스AI연구실(HUFS Business AI Lab)**이 공개하는 한국어 법률 도메인 임베딩 모델입니다. 질문과 문서의 문단을 2,048차원 벡터로 변환하여 의미 기반 검색과 검색 증강 생성(RAG)에 활용할 수 있습니다.

[NVIDIA Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)을 기반으로 한국어 법률 질문·근거 문단 쌍을 사용해 전체 파라미터를 학습했습니다. 법률 검색 검증셋에서 **nDCG@10 0.8293**을 기록하여 원본 대비 **상대 5.04% 향상**됐으며, **Recall@10은 95.0%**입니다.

[v0.1.0 다운로드](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0) · [모델 카드](docs/model-card.ko.md) · [평가 상세](docs/evaluation.md) · [문의](mailto:juho@hufs.ac.kr)

## 모델 정보

| 항목 | 내용 |
|---|---|
| 모델 ID | `hufs-bai-legal-embed-1b-v0.1` |
| 주 언어 / 도메인 | 한국어 / 법률 문서 |
| 활용 | 질문에 맞는 근거 문단 검색, 법률 문서 검색, RAG 검색기 |
| 파라미터 | 약 11.4억 개, 전체 파라미터 학습 |
| 임베딩 차원 | 2,048 |
| pooling / 유사도 | Mean pooling / 정규화된 벡터의 코사인 유사도 |
| 학습·평가 입력 길이 | 접두어를 포함하여 최대 1,536 tokens |
| 질문 / 문단 접두어 | `query: ` / `passage: ` |
| 공개 가중치 | FP32 전체 모델, 약 4.58 GB |
| 모델 라이선스 | [OpenMDW 1.1](https://openmdw.ai/license/1-1/) |

전체 모델을 **파일당 2 GB 미만**으로 나누어 GitHub Releases에서 제공합니다. 가중치는 표준 safetensors 분할 파일과 자동 로딩용 인덱스로 구성되어 직접 합칠 필요가 없습니다. 다운로드 스크립트가 실행에 필요한 파일을 모두 받으므로 기반 모델을 별도로 다운로드할 필요가 없습니다. 파일 구성과 무결성 검증 방법은 [배포 안내](docs/distribution.md)에 정리되어 있습니다.

## 바로 사용하기

Python 3.11 이상을 사용합니다. [환경 설정 안내](docs/setup.md)에 따라 CPU 또는 CUDA 환경에 맞는 PyTorch를 먼저 설치한 뒤 다음 명령을 실행하세요.

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
python -m pip install -r requirements.txt
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

CPU에서 실행하려면 마지막 명령에 `--device cpu`를 추가합니다. 예제는 가상의 한국어 문단을 검색하며 체크포인트의 FP32 파라미터를 유지합니다. 지원되는 CUDA 장치에서는 공개 평가와 같은 BF16 autocast 연산을 사용합니다.

체크포인트에 질문·문단 접두어가 설정되어 있으며 `encode_query`와 `encode_document`가 이를 적용합니다. 제공된 예제를 사용할 때 입력 문자열에 접두어를 다시 붙이지 마세요. 검색 대상 문단과 사용자 질문을 같은 모델로 임베딩하고, 정규화한 벡터의 코사인 유사도로 검색합니다.

## 검색 성능

동일한 NVIDIA RTX A6000 환경에서 **검증 질문 500개**와 **전체 검색 후보 66,895개 문단**을 사용하여 정확 코사인 검색으로 평가했습니다.

| 모델 | nDCG@10 ↑ | MRR@10 ↑ | Recall@10 ↑ | Recall@50 ↑ |
|---|---:|---:|---:|---:|
| 원본 Nemotron-3-Embed-1B | 0.7895 | 0.7441 | 0.928 | 0.980 |
| **HUFS BAI Legal Embed 1B v0.1** | **0.8293** | **0.7897** | **0.950** | **0.988** |

nDCG@10 개선량은 **+0.0398**이며, 문서군 단위 paired bootstrap 95% 신뢰구간은 **[+0.0221, +0.0581]**입니다. 평가 자료에는 법률 관련 정책·입법·행정 문서가 포함되어 있습니다. 단일 seed의 검증셋 결과이며, 별도로 확보한 최종 test 1,000문항은 아직 평가하지 않았습니다. 평가 절차와 통계적 비교 방법은 [평가 문서](docs/evaluation.md), 수치는 [결과 JSON](results/initial_dev_results.json)에서 확인할 수 있습니다.

## 학습과 데이터 출처

배치 내 다른 정답 문단을 네거티브로 사용하는 `CachedMultipleNegativesRankingLoss`로 전체 파라미터를 1 epoch 학습했습니다. **AI Hub 금융·법률 문서 기계독해 데이터**, [데이터셋 71610](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610)의 법률 자료에서 질문·근거 문단 89,898쌍을 사용했습니다. 문서군 단위 분리와 중복 검사를 통해 데이터셋을 나누고 배치 내 네거티브 충돌을 줄였습니다.

상세 학습 설정은 [모델 카드](docs/model-card.ko.md)에 있습니다. 이용 권한을 확보한 데이터로 추가 학습과 평가를 수행할 수 있도록 코드를 제공합니다. 배포 파일에는 모델과 소프트웨어가 포함되며 AI Hub 원문이나 가공한 학습·평가 텍스트는 재배포하지 않습니다.

## 라이선스

공개 모델 가중치와 기반 모델에서 가져온 구성요소에는 [OpenMDW 1.1](https://openmdw.ai/license/1-1/)이 적용됩니다. 저장소의 코드와 문서는 [Apache 2.0](LICENSE)으로 공개합니다. 모델에 포함된 라이선스와 출처 고지를 유지해 주세요. 학습 데이터 출처와 별도 AI Hub 이용 조건은 [NOTICE.md](NOTICE.md)에 안내되어 있습니다.

## 전문 도메인 튜닝 문의

**전문 도메인에 맞춘 임베딩 모델 튜닝, 맞춤형 검색 모델 구축, 산학·연구 협력**에 관심이 있으시면 **배주호 교수 [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)**로 연락해 주세요.

**한국외국어대학교 비즈니스AI연구실 · HUFS Business AI Lab**

[홈페이지](https://juho-bai.pages.dev/) · [GitHub 이슈](https://github.com/juho127/hufs-bai-embedding-models/issues)
