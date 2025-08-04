# Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models

## 요약 정리
### Problem


### Contributions


### Method


### Experiments & Setup


### Results


### Limitations


### Insights & Idea


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




<br>  
  
## 1. Introduction
### 연구 문제 정의
- 사람의 음성은 개인의 특성, 억양, 발화 스타일, 배경 소음 등으로 인해 다양하고 예측 불가능한 환경으로 인해 복잡함
- 이로 인해 ASR 시스템은 **도메인 간 차이(domain shift)** 에 매우 취약

### 최근 ASR 기술 발전 및 한계
- Whisper (OpenAI), SeamlessM4T (Meta), Canary (NVIDIA) 등 **대규모 사전학습 기반 ASR 모델**들이 공개
- 이 모델들은 **거대한 범용 데이터에 대해 학습**되어 있지만, **특정 도메인에서는 여전히 성능이 낮음**

### 도메인 적응 어려움
- 타겟 도메인 데이터에 라벨 붙이기는 시간·비용이 큼
- 기존 UDA(Unsupervised Domain Adaptation) 방법은 소스 도메인 라벨/데이터를 요구 → 현실성 낮음
  - UDA: 라벨이 없는 타겟 도메인 데이터만 가지고, 소스 도메인에서 학습된 모델을 타겟 도메인에 적응시키는 기술

### 소스 프리(Source-Free) UDA 및 자기 학습(Self-Training) 개념
- 인간이 익숙하지 않은 음성 도메인에 직면했을 때 **'자기 주도적' 능력**을 보이는 것처럼, ASR 시스템도 유사하게 학습할 수 있음
  - 첫째, 사전 학습된 모델이 타겟 도메인 데이터에 대해 '의사 라벨(pseudo label)'을 생성
  - 둘째, 이 의사 라벨이 있는 데이터를 신뢰도와 함께 사용하여 모델을 적응 
- 본 논문에서는 인간의 음성 인식 과정과 유사하게 어떠한 소스 데이터 없이도 사전 학습된 Whisper 모델 활용
  - **타겟 도메인의 소량의 라벨링되지 않은 데이터를 사용**하여 도메인별 음성 인식기로 적응시키는 '소스 프리 UDA' 문제 연구
  - 소스 데이터 사용 없이 ASR 모델을 적응시켜 광범위한 컴퓨팅 자원 소비를 피하고
  - 적은 양의 음성 샘플만으로 타겟 도메인에서 ASR 성능을 크게 향상시킬 수 있다는 점에서 중요한 가치를 지님
 
### 제안 방법 : STAR(Self-TAught Recognizer)
- **핵심 질문**: "Ground-truth 라벨이 없는 상황에서 자기 학습을 안내하기 위해 의사 라벨의 품질을 어떻게 평가할 것인가?"
  - 기존 ASR 모델의 **'신뢰도 점수(confidence scores)'** 는 소프트맥스(softmax) 함수에서 파생된 유사 사후 확률(pseudo posterior probabilities)로 근사화되며, 이는 **'과신(over-confident)' 문제로 인해 신뢰할 수 없을 수 있음**
- STAR는 자동 회귀 디코딩(auto-regressive decoding) 중 얻어지는 자기 어텐션(self-attention) 행렬 활용
  - 이 어텐션 행렬은 음성 입력에 기반하며 언어적 적합성에 초점을 맞추므로 의사 라벨 품질 측정에 더 신뢰할 수 있는 지표가 될 수 있음
  - 그러나 어텐션 스코어(attentive score)는 수치적 불안정성을 보일 수 있음
- STAR는 이러한 문제들을 해결하기 위해 신뢰도 점수와 어텐션 스코어의 장점을 통합하는 새로운 지표 제안
  - 이 지표는 이후의 미세 조정(finetuning) 프로세스를 안내하여 '지시적 적응(instructive adaptation)'을 수행
  - 지시적 적응: 모델을 무작위가 아닌, 신뢰도 높은 pseudo-label을 기반으로 학습시키는 방식

### 주요 결과
- 14개 타겟 도메인에 걸쳐 평균 13.5%의 상대적인 단어 오류율(Word Error Rate, WER) 감소를 달성했으며, 때로는 지도 학습 적응의 상한 성능에 근접
- 소스 도메인 데이터를 다시 불러오지 않고도 적응된 모델이 '파멸적 망각(catastrophic forgetting)' 문제에 빠지는 것을 방지
  - 파멸적 망각: 딥러닝 모델이 새로운 도메인에 적응하면서, **기존에 잘하던 도메인 성능이 급격히 저하되는 현상**
- 1시간 미만의 라벨링되지 않은 데이터만으로도 높은 데이터 효율성을 보이며, 인식 및 번역 작업에서 다른 대규모 음성 모델에도 원활하게 적용 가능

### 본 논문의 기여
- 실제 애플리케이션에 가장 근접한 설정인 ASR의 소스 프리 UDA에 중점을 두어, 사전 학습된 음성 파운데이션 모델과 라벨링되지 않은 음성 샘플만을 사용하여 특정 타겟 도메인에 적응시키는 방법을 제시함
- 소음, 억양 및 특정 시나리오를 포함한 광범위한 타겟 도메인에서 음성 파운데이션 모델의 도메인별 능력을 향상시킴
- 데이터 효율성 및 일반성 분석을 통해 음성 비서의 점진적 업데이트와 같은 실제 애플리케이션에서의 잠재력을 보여줌





<br>  
  
## 2. Related Work









