# HUFS BAI Legal Embed 1B 다운로드

[English](distribution.md) · [모델 소개](../README.ko.md)

**HUFS BAI Legal Embed 1B v0.1**은 전체 파라미터를 학습한 한국어 법률 임베딩 모델입니다. [GitHub Releases v0.1.0](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0)에서 가중치와 토크나이저를 무료로 제공합니다. 토큰이나 구독, NVIDIA 원본 모델의 별도 다운로드가 필요하지 않습니다.

## 다운로드

저장소를 받은 뒤 Python 3.10 이상에서 다운로드 스크립트를 실행하세요. 이 스크립트는 외부 패키지 설치 없이 동작합니다.

```bash
git clone https://github.com/juho127/hufs-bai-embedding-models.git
cd hufs-bai-embedding-models
python scripts/download_release.py --output models/hufs-bai-legal-embed-1b-v0.1
```

전체 다운로드 용량은 **약 4.6 GB**입니다. 모델 저장에 여유 공간 5 GB 이상을 확보하세요. 추론 환경 설치 공간은 별도로 필요합니다. 각 릴리스 파일은 **2,000,000,000 bytes 미만**으로, GitHub의 파일당 2 GiB 제한보다 작습니다. 일반 릴리스 첨부 파일이므로 Git LFS 전송량을 사용하지 않습니다. GitHub는 [Releases 공식 정책](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)에서 릴리스 전체 크기와 다운로드 대역폭에 제한을 두지 않는다고 안내합니다.

가중치는 표준 Safetensors 파일 3개와 인덱스로 구성됩니다. **분할된 가중치 파일은 합치지 않고 모두 같은 모델 폴더에 두세요.** Transformers와 Sentence Transformers가 인덱스를 읽어 자동으로 불러옵니다. 압축 해제나 바이너리 파일 병합이 필요하지 않습니다. 공개 가중치는 평가에 사용한 FP32 파라미터를 유지합니다.

| 가중치 파일 | 바이트 | 용량 |
| --- | ---: | ---: |
| `model-00001-of-00003.safetensors` | 1,778,445,520 | 약 1.78 GB |
| `model-00002-of-00003.safetensors` | 1,795,301,096 | 약 1.80 GB |
| `model-00003-of-00003.safetensors` | 989,942,248 | 약 0.99 GB |

다운로더는 [버전별 manifest](../releases/v0.1.0/manifest.json)를 읽고 `1_Pooling/config.json` 등의 폴더 구조를 복원한 뒤, 파일별 크기와 SHA-256이 맞는지 확인합니다. 다시 실행하면 이미 검증된 파일은 건너뜁니다. 연결이 끊기면 같은 명령을 다시 실행하세요. 중단된 파일만 처음부터 다운로드합니다. 기존 파일이 해당 릴리스와 다르면 덮어쓰지 않고 오류를 표시합니다. 다른 모델이나 버전에는 별도의 빈 폴더를 사용하세요.

## 모델 불러오기

[추론 환경](setup.md)을 설치한 뒤 다운로드한 로컬 경로를 지정하세요.

```bash
python examples/retrieve.py --model models/hufs-bai-legal-embed-1b-v0.1
```

질문·문서 접두어와 검색 예시는 [모델 소개](../README.ko.md)를 참고하세요.

## 직접 다운로드와 검증

직접 받으려면 [manifest.json](../releases/v0.1.0/manifest.json)에 나열된 파일을 모두 [릴리스 첨부 파일](https://github.com/juho127/hufs-bai-embedding-models/releases/tag/v0.1.0)에서 다운로드하세요. `asset`은 릴리스 파일명이고 `path`는 모델 폴더 안에 저장할 상대 경로입니다. 예를 들어 `1_Pooling-config.json`은 `1_Pooling/config.json`으로 저장합니다. 가중치 3개뿐 아니라 인덱스, 토크나이저, 풀링 설정과 라이선스 파일도 함께 있어야 합니다.

각 파일의 해시를 계산해 manifest의 `sha256` 값과 비교할 수 있습니다.

```bash
# Linux / macOS
shasum -a 256 models/hufs-bai-legal-embed-1b-v0.1/model-00001-of-00003.safetensors
```

```powershell
# Windows PowerShell
Get-FileHash models/hufs-bai-legal-embed-1b-v0.1/model-00001-of-00003.safetensors -Algorithm SHA256
```

다운로더를 사용하면 모든 파일의 검증과 폴더 구성이 자동으로 처리됩니다.

## 라이선스와 문의

모델 가중치와 NVIDIA 원본에서 상속한 모델 자료에는 [OpenMDW-1.1](https://openmdw.ai/license/1-1/)이 적용되며, 출처 고지와 라이선스 파일을 함께 제공합니다. 저장소 코드에는 [Apache-2.0](../LICENSE)이 적용됩니다. 원본 모델과 AI Hub 학습 데이터의 출처는 [NOTICE](../NOTICE.md)와 [모델 카드](model-card.ko.md)를 참고하세요. 학습·평가 텍스트는 모델 다운로드에 포함하지 않습니다.

**전문 도메인 튜닝, 모델 맞춤 개발, 연구 협업 문의는 [juho@hufs.ac.kr](mailto:juho@hufs.ac.kr)로 연락해 주세요.** [홈페이지](https://juho-bai.pages.dev/).
