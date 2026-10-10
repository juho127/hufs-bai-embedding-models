# Notices and attribution

HUFS BAI Embedding Models

Copyright 2026 HUFS Business AI Lab.

HUFS BAI Legal Embed 1B is a Korean legal retrieval embedding model developed by HUFS Business AI Lab, Hankuk University of Foreign Studies.

HUFS BAI Legal Embed 1B는 한국외국어대학교 비즈니스AI연구실이 개발한 한국어 법률 검색 임베딩 모델입니다.

## Repository code and documentation

별도 고지가 없는 이 저장소의 자체 작성 코드와 문서에는 [Apache License 2.0](LICENSE)을 적용합니다. 외부 라이브러리는 각각의 라이선스를 따릅니다. 이 저장소의 라이선스가 외부 모델 또는 학습 데이터의 이용 조건을 대체하지 않습니다.

## NVIDIA base model

- Model: [nvidia/Nemotron-3-Embed-1B-BF16](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16)
- Provider: NVIDIA
- Pinned revision: `c0c9fea93ea424587517f2c59e20db9f1d6bf615`
- Model and associated configuration license: [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/)

원본 모델 및 관련 설정 파일은 NVIDIA가 별도로 제공합니다. OpenMDW-1.1은 모델 자료의 사용과 수정을 허용하며, 해당 자료의 일부를 배포할 때 라이선스 사본과 적용 가능한 저작권·출처 고지를 유지하도록 규정합니다. 이 코드 저장소에는 원본 모델 가중치를 포함하지 않습니다.

The fully fine-tuned model is distributed under OpenMDW-1.1 with the NVIDIA copyright notice and original public model card. The code's Apache-2.0 license does not replace the model license. The release is a complete model and does not require a separate base-model download.

전체 파라미터를 학습한 공개 모델에는 OpenMDW-1.1을 적용하며, NVIDIA 저작권 고지와 원본 공개 모델 카드를 함께 배포합니다. 코드의 Apache-2.0 라이선스가 모델 라이선스를 대체하지 않습니다. 공개 패키지는 별도 기반 모델 다운로드가 필요 없는 전체 모델입니다.

## Training data

- Dataset: **금융·법률 문서 기계독해 데이터**
- Source: [AI Hub](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610), dataset ID `71610`
- Project usage: 법률 문서 부분을 이용한 임베딩 학습과 검색 성능 평가

AI Hub의 [공식 FAQ](https://aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104)는 학습 결과인 모델·서비스·연구 결과의 영리 및 비영리 활용을 허용하고, 데이터셋 정식 명칭과 AI Hub 출처 표시를 요구합니다. 원본 데이터와 단순 가공 데이터의 외부 공개·공유·재배포는 허용하지 않습니다. 해당 데이터셋의 개별 이용 조건도 적용됩니다.

이 저장소는 데이터 원문, 가공된 학습·평가 텍스트, 원문이 포함된 실행 로그를 배포하지 않습니다. 이용자는 AI Hub에서 자료를 직접 확보하고 자신에게 적용되는 이용 조건을 확인해야 합니다.

## Maintainer

- Website: [Juho BAI](https://juho-bai.pages.dev/)
- Contact: [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)

For specialized domain fine-tuning, custom embedding models, and research collaboration, contact **juho@hufs.ac.kr**.

전문 도메인 특화 튜닝, 맞춤형 임베딩 모델 개발 및 연구 협업 문의는 **juho@hufs.ac.kr**로 연락해 주세요.

정책 링크와 배포 상태 확인일: 2026-10-10.
