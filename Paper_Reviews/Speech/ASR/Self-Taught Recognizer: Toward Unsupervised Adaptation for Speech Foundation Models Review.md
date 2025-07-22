# Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 연구 문제 정의
- 대형 ASR foundation models (Whisper 등)도 domain shift 상황(잡음, 억양 등)에서는 성능 저하 발생
- 라벨 없는 타겟 도메인 데이터만으로 모델을 효과적으로 적응시키는 방법 필요

### 기존 방법의 한계
- 기존 Unsupervised Domain Adaptation(UDA) 방법
- 대부분 라벨된 소스 데이터와 비라벨 타겟 데이터 둘 다 필요
- 현실에서는 소스 데이터 접근 불가능한 경우 많음 (보안, 저장, 프라이버시 문제 등)
- Softmax 기반 confidence score만으로 pseudo-label quality 판단
  - 신뢰성 떨어짐 (over-confidence 문제)
 
### 제안 방법 : STAR (Self-TAught Recognizer)
- Source-free UDA: 소스 데이터 없이, unlabeled 타겟 데이터만 활용
- Token-level pseudo-label quality 평가
  - 디코딩 중 self-attention 정보를 활용
  - pseudo-label의 품질을 더 잘 평가할 수 있는 새로운 지표 설계
- 효과적인 informed finetuning
  - 높은 품질 pseudo-label만 선별하여 모델 업데이트
- 모델 일반성
  - Whisper 외 Canary, SeamlessM4T 등 다른 speech foundation model에도 적용 가능

### 실험 세팅
- 다양한 target domain (잡음, 억양 등 14개 domain)에서 실험
- unlabeled data만 사용
- 평가 metric: WER(Word Error Rate)

### 주요 결과
- 평균 13.5% WER 상대 개선
- 일부 domain에서는 supervised adaptation upper bound에 근접
- 1시간 미만 unlabeled data만 필요 (데이터 효율성 높음)
- catastrophic forgetting 방지 효과도 확인
- **장점**
  - 소스 데이터 필요 없음 → practical, privacy-friendly
  - pseudo-label quality를 token 단위로 정교하게 평가
  - Whisper 등 다양한 model에 적용 가능
  - 작은 데이터만으로도 빠르게 적응 가능



