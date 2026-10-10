# 모델 다운로드와 배포

코드와 문서는 이 GitHub 저장소에서 관리하고, 모델 가중치는 별도 다운로드로 배포할 계획입니다. **2026-10-10 기준 학습된 가중치의 공개 다운로드는 아직 제공하지 않습니다.** 아래 용량은 완료된 실험의 저장 파일을 기준으로 합니다.

## 현재 파일 크기

| 항목 | 저장 크기 | 사용 시 필요한 파일 |
| --- | ---: | --- |
| LoRA 어댑터 가중치 | 8,396,720 bytes, 약 8.40 MB | 원본 Nemotron 모델과 어댑터 |
| LoRA 저장 폴더 전체 | 25,534,528 bytes, 약 25.53 MB | 설정·토크나이저 포함, 원본 모델은 별도 |
| Full 저장 폴더 전체 | 4,580,826,739 bytes, 약 4.58 GB | FP32로 저장된 전체 모델과 설정 |
| 원본 모델 | 약 2.3 GB | NVIDIA가 제공하는 BF16 모델 |

MB와 GB는 십진 단위입니다. LoRA는 원본 모델을 함께 불러오므로, 어댑터 파일 크기가 전체 설치 용량을 뜻하지 않습니다. Full 체크포인트는 평가에 사용한 FP32 가중치를 보존합니다. BF16 변환본을 제공하게 되면 별도 파일로 표시하고 변환 후 성능을 검증합니다.

## GitHub 무료 계정에서 가능한가요?

| 배포 방식 | 공식 제한 | 이 프로젝트에서의 판단 |
| --- | --- | --- |
| 일반 Git | 파일 하나가 100 MiB를 초과하면 업로드 차단 | 코드·문서 관리에 사용 |
| Git LFS, GitHub Free | 파일당 2 GB, 무료 저장 10 GiB·월 다운로드 10 GiB | Full 파일 분할이 필요하며 반복 다운로드 시 무료 전송량이 작음 |
| GitHub Releases | 파일당 2 GiB 미만, 릴리스 전체 크기와 다운로드 대역폭 제한 없음 | LoRA 배포 가능. Full은 보수적으로 조각당 2 GB 미만으로 분할해 배포 가능 |
| Hugging Face 공개 모델 저장소 | Free 공개 저장은 best-effort 정책 | 모델 카드·버전·라이브러리 연동을 갖춘 배포 경로로 권장 |

제한은 2026-10-10에 확인한 [GitHub 일반 파일 정책](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [LFS 파일 한도](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage), [LFS 과금 정책](https://docs.github.com/en/billing/concepts/product-billing/git-lfs), [Releases 정책](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), [Hugging Face 저장 정책](https://huggingface.co/docs/hub/en/storage-limits)을 따릅니다. Hugging Face 무료 공개 저장은 무제한 용량을 보장하는 계약이 아닙니다.

현재 Full 저장 폴더 약 4.27 GiB를 LFS로 배포하면, 다른 전송을 제외해도 완전한 다운로드 두 번에 약 8.53 GiB를 사용합니다. 세 번째 다운로드까지 무료 월 10 GiB에 담기는 어렵습니다. GitHub Releases에 조각 파일과 SHA-256 검증 목록을 올리거나 Hugging Face에 모델을 게시하면 일반 Git 저장소를 가볍게 유지할 수 있습니다.

## 공개 파일 구성

각 가중치 릴리스에는 모델 카드, 학습 설정, 원본 모델의 고정 revision, 평가 결과, 파일별 SHA-256, 적용 라이선스와 원본 고지를 함께 제공합니다. LoRA는 원본 모델 다운로드 경로를 명시합니다. Full을 분할 배포할 경우 모든 조각의 순서와 복원·검증 명령을 함께 제공합니다.

학습 데이터 원문, 가공된 학습 쌍·검색 문단·질문, 원문이 포함된 실행 로그는 이 저장소와 모델 릴리스에 포함하지 않습니다. 데이터 준비는 이용자가 [AI Hub의 금융·법률 문서 기계독해 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71610)를 직접 확보한 뒤 제공된 변환 코드를 실행하는 방식입니다. 원 실험의 비공개 품질 검토 주석은 포함하지 않으므로 일반 변환 명령만으로 동일한 분할을 재현하지는 못합니다. 입력 형식과 재현 범위는 [환경 설정](setup.md)에 설명합니다.

## 사용 조건

- 이 저장소에서 작성한 코드와 문서는 [Apache License 2.0](../LICENSE)을 적용합니다.
- 원본 NVIDIA 모델과 해당 모델을 포함하는 파생 모델의 배포에는 [OpenMDW-1.1](https://openmdw.ai/license/1-1/) 및 해당 원본 고지를 유지합니다. 예정된 학습 가중치의 라이선스는 모델 카드와 릴리스에 따로 표시합니다.
- AI Hub FAQ는 해당 데이터를 학습해 만든 모델·서비스의 상업적 활용을 허용하며 데이터셋 정식 명칭과 AI Hub 출처 표시를 요구합니다. 원본 데이터와 단순 가공 데이터의 외부 공유·재배포는 허용하지 않습니다. 데이터별 이용 조건도 함께 적용됩니다. [AI Hub 공식 FAQ](https://aihub.or.kr/aihubnews/faq/list.do?currMenu=146&topMenu=104)

자세한 출처와 라이선스 범위는 [NOTICE](../NOTICE.md)를 참조하세요.
