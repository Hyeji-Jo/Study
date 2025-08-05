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
### ASR에서의 비지도 도메인 적응 (Unsupervised Domain Adaptation, UDA)
- 타겟 도메인에서 **라벨링된 음성(transcriptions)** 확보는 시간과 비용이 많이 듬
- 따라서 기존 연구들은 **소스 도메인(label 포함) 데이터**를 기반으로, **라벨 없는 타겟 도메인 데이터**에 적응하는 방법 고안

- **주요 기법**
  - 타켓 음성 시뮬레이션: 타겟 도메인 스타일의 데이터를 합성해서 적응
  - 적대적 학습 (Adversarial Learning): 도메인 불변 표현을 학습하여 domain shift 최소화
  - Teacher-Student Learning: source 모델이 label 제공 → target 모델 학습
  - Self-supervised 모델 활용: wav2vec2 등으로 pseudo-label 생성하여 라벨 없이 적응
- **한계**
  - 위 접근법들은 대부분 **소스 도메인의 레이블이 필요**하므로 완전한 비지도 학습이 아님 → **semi-supervised에 가까움**
  - 소스 데이터 접근 자체가 어려운 경우가 많음 → **현실 적용 제약 존재**

### 소스 프리 비지도 도메인 적응 (Source-free Unsupervised Domain Adaptation)
- **소스 데이터를 완전히 배제**하고, 사전 학습된 모델만을 가지고 **라벨 없는 타겟 도메인에 적응하는 방법론**
- **필요성**
  - 소스 데이터에 개인정보나 민감 정보 포함 가능성
  - 저장/전달/재사용의 법적·윤리적 제약 → 소스 없이 적응하는 방법이 중요
- **주요 접근법**
  - Self-supervised Knowledge Distillation: 사전 학습된 모델의 내부 지식만 활용하여 적응
  - Contrastive Learning: 타겟 도메인 간 유사·비유사 샘플 간 간극을 학습
  - Hidden Structure Mining: 라벨 없이 데이터 간 구조 관계를 추출하여 학습
  - Uncertainty-guided Adaptation: 모델의 예측 불확실성을 활용해 신뢰도 높은 샘플 선택 (→ STAR가 속한 범주)
 
- 최근에는 **auto-regressive decoder 기반 불확실성 추정**에 대한 연구가 많아졌으나,
→ **Source-Free UDA에는 거의 적용되지 않음**

### STAR의 위치 및 차별성
- **문제 인식**
  - Whisper 등 speech foundation model은 사전 학습에 수십만 시간의 데이터를 사용
    - 소스 도메인의 범위가 넓고 복잡
  - 이 데이터를 보존·공유·재학습하는 것이 현실적으로 불가능
- **STAR의 핵심 아이디어**
  - 소스 데이터가 없는 환경에서, **모델이 생성한 pseudo-label의 품질을 평가**
  - 신뢰할 수 있는 토큰 중심으로 학습을 유도 → **지시적 자기학습 (instructive self-training)**
- **효과**
  - source 데이터 불필요
  - ground-truth vs pseudo-label 적응 성능 차이를 크게 줄임
  - 현실적인 Source-free UDA 달성
  - ASR 기반 실제 서비스에 적용 가능성↑ (예: 음성 비서 도메인 적응) 




<br>  
  
## 3. Methodology
### 3.1 Problem Setup
#### ASR 공식화 (ASR Formulation)
- ASR 모델 f는 음성 입력 $$x \in \mathbb{R}^T$$를 텍스트 시퀀스 $$y \in \mathbb{R}^L$$로 변환
- 학습은 **teacher-forcing + cross-entropy loss**로 이루어짐
  - $$\mathcal{L}{\text{ASR}}(x, y) = \sum{l=1}^{L} - \log P_{\theta} (y_l|y_{1:l-1}, x)$$
- 핵심: **순차적(auto-regressive) 예측**으로 **이전 토큰에 기반해 다음 토큰 예측**

#### UDA 설정
- 목적: 라벨 없는 **타겟 도메인 데이터** $$X^{(t)}$$ 만 사용해, **소스 모델 $$f^{(s)}$$** 를 **타겟 도메인 $$D^{(t)}$$** 에 맞게 적응
- Source-free UDA 시나리오
  - 소스 도메인 데이터 $$\{X^{(s)}, Y^{(s)}\}$$는 사용할 수 없음
  - 사전 학습된 모델 $$f^{(s)}$$만 주어진 상황에서 **라벨 없는 음성 $$X^{(t)}$$** 만으로 적응해야 함 

#### Self-Training 전략 (STAR의 핵심 구조)
- **Pseudo-label 생성**
  - unlabeled 데이터 $$X^{(t)} = \{x_i^{(t)}\}_{i=1}^{N^{(t)}}$$를 소스 모델 $$f^{(s)}$$에 넣어 pseudo-label $$\hat{Y}^{(t)} = \{\hat{y}i^{(t)}\}{i=1}^{N^{(t)}}$$ 생성
- **Informed Finetuning (정보 기반 미세 조정)**
  - $$\{X^{(t)}, \hat{Y}^{(t)}\}$$를 이용해 모델을 타겟 도메인으로 적응시키는 학습 수행
  - $$\mathcal{L}{\text{ST}}(X^{(t)}, \hat{Y}^{(t)}) = \sum{i=1}^{N^{(t)}} \mathcal{L}_{\text{ASR}}(x_i^{(t)}, \hat{y}_i^{(t)})$$


### 3.2 Token-level Assessment and Re-weighting
#### 배경 및 목표
- Self-training에서 생성된 pseudo-label의 품질은 다양
- 따라서 **각 토큰별로 품질을 평가하고, 좋은 토큰은 크게, 나쁜 토큰은 작게 학습에 반영해야 함**
- 이 유사 레이블의 품질을 평가하고, 이를 바탕으로 모델 업데이트를 안내하는 "지표 (indicator)"를 개발하는 것이 이 섹션의 주요 목표
- 이는 토큰 레벨 (단어 또는 하위 단어 단위)과 발화 레벨 (전체 문장 단위)에서 모두 이루어짐

#### Confidence Score의 한계
- $$C_l = \max P(\hat{y}_l|\hat{y}_{1:l-1}, x, \theta^*)$$
  - $$\hat{y}{1:l-1}$$: 이전에 예측된 토큰들
  - $$\theta^*$$: 모델의 학습된 파라미터
- 기존의 모델에서는 각 토큰 예측에 대한 "confidence score" $$\(C_l\)$$ (확신 점수)를 사용
- 이는 자동 회귀 디코딩 (auto-regressive decoding) 시 이전 예측된 토큰들에 기반하여 현재 토큰 $$\hat{y}_l$$의 후방 확률 (posterior probability) 중 가장 높은 값으로 정의
- **STAR 기반 재가중 학습 손실**: $$eLASR(x, \hat{y}) = \sum_{l=1}^{L} -\log P_{\theta}(\hat{y}_l|\hat{y}_{l-1:1}, x) \cdot C_l$$
  - L: 출력 텍스트 시퀀스의 길이
- **문제점**
  - 자동 회귀 디코딩에서 확신 점수가 예측 정확도를 정확하게 반영하지 못하는 경향 존재
  - 오류 누적 (error accumulation) 및 과도한 확신 (over-confident) 문제 때문 

#### Attentive Score




