# Notices and attribution

HUFS BAI Embedding Models

Copyright 2026 HUFS Business AI Lab.

이 프로젝트는 한국외국어대학교 비즈니스AI연구실(HUFS Business AI Lab)의 한국어 도메인 특화 임베딩 연구를 위한 코드와 문서를 제공합니다.

## Repository code and documentation

별도 고지가 없는 이 저장소의 자체 작성 코드와 문서에는 [Apache License 2.0](LICENSE)을 적용합니다. 외부 라이브러리는 각각의 라이선스를 따릅니다. 이 저장소의 라이선스가 외부 모델 또는 학습 데이터의 이용 조건을 대체하지 않습니다.

## NVIDIA base model

- Model: [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
- Provider: NVIDIA
- Pinned revision: `c0c9fea93ea424587517f2c59e20db9f1d6bf615`
- Model and associated configuration license: [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/)

원본 모델 및 관련 설정 파일은 NVIDIA가 별도로 제공합니다. OpenMDW-1.1은 모델 자료의 사용과 수정을 허용하며, 해당 자료의 일부를 배포할 때 라이선스 사본과 적용 가능한 저작권·출처 고지를 유지하도록 규정합니다. 이 코드 저장소에는 원본 모델 가중치를 포함하지 않습니다.

LoRA와 Full 학습 가중치는 아직 공개 배포하지 않았습니다. 향후 가중치 배포에는 OpenMDW-1.1 및 적용 가능한 원본 고지를 포함하고, 라이선스 범위와 원본 모델 의존성을 각 모델 카드에 명시합니다. 가중치는 이 저장소의 Apache-2.0 코드 라이선스에 일괄 포함되지 않습니다.

## Training data

- Dataset: **금융·법률 문서 기계독해 데이터**
- Source: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610), dataset ID `71610`
- Project usage: 법률 문서 부분을 이용한 임베딩 학습과 검색 성능 평가

AI Hub의 [공식 FAQ](https://aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104)는 학습 결과인 모델·서비스·연구 결과의 영리 및 비영리 활용을 허용하고, 데이터셋 정식 명칭과 AI Hub 출처 표시를 요구합니다. 원본 데이터와 단순 가공 데이터의 외부 공개·공유·재배포는 허용하지 않습니다. 해당 데이터셋의 개별 이용 조건도 적용됩니다.

이 저장소는 데이터 원문, 가공된 학습·평가 텍스트, 원문이 포함된 실행 로그를 배포하지 않습니다. 이용자는 AI Hub에서 자료를 직접 확보하고 자신에게 적용되는 이용 조건을 확인해야 합니다.

## Maintainer

- Website: [Juho BAI](https://juho-bai.pages.dev/)
- Contact: [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)

정책 링크와 배포 상태 확인일: 2026-10-10.
