# Notices and attribution

HUFS BAI Embedding Models

Copyright 2026 HUFS Business AI Lab.

HUFS BAI Legal Embed 1B and HUFS BAI Defense Embed 1B are Korean domain-specific retrieval embedding models developed by HUFS Business AI Lab, Hankuk University of Foreign Studies.

HUFS BAI Legal Embed 1B와 HUFS BAI Defense Embed 1B는 한국외국어대학교 비즈니스AI연구실이 개발한 한국어 전문 도메인 검색 임베딩 모델입니다.

## Repository code and documentation

별도 고지가 없는 이 저장소의 자체 작성 코드와 문서에는 [Apache License 2.0](LICENSE)을 적용합니다. 외부 라이브러리는 각각의 라이선스를 따릅니다. 이 저장소의 라이선스가 외부 모델 또는 학습 데이터의 이용 조건을 대체하지 않습니다.

## NVIDIA base model

- Model: [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
- Provider: NVIDIA
- Pinned revision: `c0c9fea93ea424587517f2c59e20db9f1d6bf615`
- Model and associated configuration license: [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/)

원본 모델 및 관련 설정 파일은 NVIDIA가 별도로 제공합니다. OpenMDW-1.1은 모델 자료의 사용과 수정을 허용하며, 해당 자료의 일부를 배포할 때 라이선스 사본과 적용 가능한 저작권·출처 고지를 유지하도록 규정합니다. 이 코드 저장소에는 원본 모델 가중치를 포함하지 않습니다.

The fully fine-tuned models are distributed under OpenMDW-1.1 with the NVIDIA copyright notice and original public model card. The code's Apache-2.0 license does not replace the model license. Each release is a complete model and does not require a separate base-model download.

전체 파라미터를 학습한 공개 모델에는 OpenMDW-1.1을 적용하며, NVIDIA 저작권 고지와 원본 공개 모델 카드를 함께 배포합니다. 코드의 Apache-2.0 라이선스가 모델 라이선스를 대체하지 않습니다. 공개 패키지는 별도 기반 모델 다운로드가 필요 없는 전체 모델입니다.

## Legal model training data · 법률 모델 학습 자료

- Dataset: 금융·법률 문서 기계독해 데이터
- Source: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610), dataset ID `71610`
- Project usage: 법률 문서 부분을 이용한 임베딩 학습과 검색 성능 평가

AI Hub의 [공식 FAQ](https://aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104)는 학습 결과인 모델·서비스·연구 결과의 영리 및 비영리 활용을 허용하고, 데이터셋 정식 명칭과 AI Hub 출처 표시를 요구합니다. 원본 데이터와 단순 가공 데이터의 외부 공개·공유·재배포는 허용하지 않습니다. 해당 데이터셋의 개별 이용 조건도 적용됩니다.

The AI Hub terms apply to the legal model's source data; they do not describe the defense model's separate public briefing sources. This repository does not distribute source text, derived training/evaluation text, or logs containing source text. Users obtaining AI Hub data should obtain it directly and review the terms applicable to them.

AI Hub 이용 조건은 법률 모델의 원천 데이터에 적용되며 국방 모델의 별도 공개 브리핑 자료와는 구분됩니다. 이 저장소는 데이터 원문, 가공된 학습·평가 텍스트, 원문이 포함된 실행 로그를 배포하지 않습니다. AI Hub 자료가 필요한 이용자는 직접 확보하고 자신에게 적용되는 이용 조건을 확인해야 합니다.

## Defense model training data · 국방 모델 학습 자료

HUFS BAI Defense Embed 1B v0.2 uses text from 국방부 일일 정례 브리핑 (Ministry of National Defense Daily Briefing), authored by the Ministry of National Defense and published through [Korea Policy Briefing / 대한민국 정책브리핑](https://www.korea.kr/briefing/policyBriefingList.do). Training uses 283 question–answer pairs from 185 source documents published from July 5, 2022 through October 6, 2026. Titles, publishers, publication dates and direct source links are provided in [TRAINING_SOURCES.json](releases/defense-v0.2.0/TRAINING_SOURCES.json), also included with the model package.

국방 모델은 국방부가 작성하고 대한민국 정책브리핑에 공개한 국방부 일일 정례 브리핑의 본문 텍스트를 이용했습니다. 학습에는 2022년 7월 5일부터 2026년 10월 6일까지 발행한 문서 185건에서 질문·답변 283쌍을 사용했습니다. 저작물명, 기관, 발행일, 원문 주소를 TRAINING_SOURCES.json과 모델 패키지에 함께 제공합니다. 원문은 각 출처 주소에서 무료로 열람할 수 있습니다.

Each selected article's text carries a [KOGL Type 1 attribution notice](https://www.kogl.or.kr/info/licenseType1.do), which permits commercial use and modification with attribution. The source permission covers the article text; images, video, audio and third-party materials were not included. Source institutions do not sponsor or endorse this model. The defense release includes model weights, configuration and attribution metadata; it does not redistribute the source transcripts or derived training/evaluation text.

선별한 각 문서의 본문 텍스트에는 출처표시를 조건으로 상업적 이용과 변경을 허용하는 공공누리 제1유형 고지가 있습니다. 이용 범위는 본문 텍스트이며 사진·영상·음성 및 제3자 자료는 포함하지 않았습니다. 원천 자료 기관이 이 모델을 후원하거나 보증하는 것은 아닙니다. 국방 모델 공개 패키지는 가중치, 설정과 출처 메타데이터를 제공하며 원문이나 가공한 학습·평가 텍스트를 재배포하지 않습니다.

### Defense retrieval corpus sources · 국방 검색 후보 자료 출처

Evaluation also used KOGL Type 1 public policy text from the [Ministry of National Defense](https://www.mnd.go.kr/) and [Defense Acquisition Program Administration](https://www.dapa.go.kr/), published through Korea Policy Briefing. Selected defense and military-manpower chapters from the following National Assembly Budget Office (NABO / 국회예산정책처) reports contributed 341 pages:

평가 검색 후보에는 대한민국 정책브리핑에 게시된 국방부·방위사업청의 공공누리 제1유형 정책문서 본문과 아래 국회예산정책처 보고서의 국방부·병무청 관련 공개 부분 341쪽도 사용했습니다.

| Publisher / 발행기관 | Published / 발행일 | Work / 저작물 |
|---|---|---|
| NABO / 국회예산정책처 | 2024-10-28 | [2025년도 예산안 위원회별 분석[외교통일위원회·국방위원회]](https://www.nabo.go.kr/ko/report/analysisView.do?idx=8495&key=2509250001) |
| NABO / 국회예산정책처 | 2025-07-28 | [2024회계연도 결산 위원회별 분석 [외교통일위원회·국방위원회]](https://www.nabo.go.kr/ko/report/analysisView.do?idx=8825&key=2509250001) |
| NABO / 국회예산정책처 | 2025-10-31 | [2026년도 예산안 위원회별 분석[외교통일위원회·국방위원회]](https://www.nabo.go.kr/ko/report/analysisView.do?idx=8953&key=2509250001) |

The selected reports carry individual KOGL Type 1 notices; [NABO's copyright policy](https://nabo.go.kr/ko/content.do?key=2509260003) provides additional terms. Policy and budget material was used only as retrieval candidates, not as supervised training pairs or for negative mining. These source documents are not part of the public model package.

선별한 보고서는 개별 공공누리 제1유형 고지를 확인했으며 추가 이용 조건은 국회예산정책처 저작권 정책에 안내되어 있습니다. 정책·예산 자료는 검색 후보로만 사용했고 지도학습 질답이나 네거티브 채굴에 사용하지 않았습니다. 해당 원문은 공개 모델 패키지에 포함하지 않습니다.

## Maintainer

- Website: [Juho BAI](https://juho-bai.pages.dev/)
- Contact: [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)

For specialized domain fine-tuning, custom embedding models, and research collaboration, contact Prof. Juho Bai at [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr).

전문 도메인 특화 튜닝, 맞춤형 임베딩 모델 개발 및 연구 협업 문의는 배주호 교수 [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)로 연락해 주세요.

정책 링크와 배포 상태 확인일: 2026-10-10.
