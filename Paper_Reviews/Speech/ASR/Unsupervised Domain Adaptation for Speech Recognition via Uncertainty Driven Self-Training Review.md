# Unsupervised Domain Adaptation for Speech Recognition via Uncertainty Driven Self-Training
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
### 문제 정의
- ASR 시스템은 훈련 데이터와 실제 테스트 데이터(다른 도메인) 간 차이로 인해 성능이 저하됨
- ex) 깨끗한 뉴스 음성(WSJ)으로 학습 후, TED나 전화 통화(SWBD)에서 성능 하락
  
### 기존 Self-Training 방법의 한계
- teacher model이 unlabeled target data에 pseudo-label 부여 → student model 학습
- 한계: pseudo-label이 noisy하면 성능이 악화됨
- 기존 work들은 domain mismatch 없는 조건에서 filtering 없이 사용
  
### 제안 방법 : DUST(Dropout-based Uncertainty-driven Self-Training)
- **핵심 아이디어**
  - dropout 설정을 다르게 하여 얻은 여러 prediction들 간의 일치도를 통해 모델의 prediction 불확실성을 측정
  - **불확실성 높은 pseudo-label은 학습에서 제외**
- **장점**
  - Filtering 없는 **ST 대비 ASR 성능 크게 향상**
  - 학습 데이터셋 크기가 줄어 **training time 단축**
  
### 싦험
- Dataset
  - Source: WSJ (깨끗한 read speech)
  - Target: TED-LIUM 3 (강연), SWITCHBOARD (전화 대화)
- 결과
  - WSJ → TED: 최대 80% WER recovery
  - WSJ → SWBD: 최대 65% WER recovery 


<br>  
  
## 1. Introduction
### 문제 정의 및 필요성
- 훈련 도메인과 테스트 도메인의 mismatch
  - ASR 시스템이 학습한 데이터와 실제 사용할 환경의 조건(화자, 발음, 배경소음 등)이 달라서 생기는 일반적인 문제
- 이상적인 해결책: **target domain에 맞는 라벨 데이터를 수집해서 모델 fine-tuning**
  - 하지만 라벨링 비용 + 시간 부담 → 어려움
- **라벨 없는(unlabeled) target domain 데이터를 활용하는 unsupervised domain adaptation 기법**     
### 기존 연구: Distribution Alignment
- 최근에는 labeled data 없이도 source-target 간 분포를 맞추는 방법들 연구됨
  - **Optimal Transport (OT)**
  - **Domain Adversarial Training (GRL)**
    - domain 구분 불가능하게 모델 학습 
  - **Discrepancy loss 기반 방법**

### Self-Training(ST)의 개념
- labeled source data로 teacher model 학습
- 이 모델이 target domain의 **unlabeled data에 대해 pseudo-label 생성**
- student model이 labeled + pseudo-labeled data로 재학습

### ST의 문제점
- ST 성능이 **pseudo-label이 부정확하면 오히려 모델이 나빠**질 수 있음이 알려져 있음
- 그래서 **과거에는 filtering을 통해 quality 낮은 pseudo-label 제거했음**
- 그런데 최근 논문 [19]에서는 filtering 없이도 성능 좋았다고 보고
  - **Source와 target 도메인 간 mismatch가 없음**
  - Target 도메인에 충분한 텍스트가 있어 language model을 잘 만들 수 있음

### 제안 방법
- **본 논문에서 다루는 상황**
  - 도메인 mismatch 존재 (예: WSJ → TED, SWBD)
  - target domain에 ground-truth label이나 풍부한 텍스트 없음
  - 이런 상황에서는 teacher가 생성한 pseudo-label quality가 낮을 가능성 ↑
    - 따라서 **pseudo-label filtering이 중요**
- **DUST (Dropout-based Uncertainty-driven Self-Training)**
  - Dropout으로 다양한 prediction sample 생성
  - 각 sample들과 reference prediction(드롭아웃 없는) 간 **agreement (edit distance) 측정**
  - **불확실성 높은 sample은 학습 제외**



<br>  
  
## 2. DUST
### 핵심 아이디어
- 모델이 unlabeled target 데이터에 대해 **불확실한 예측을 하면 해당 pseudo-label은 학습에 사용하지 않도록 filtering**하는 self-training 방식
  - Dropout을 적용한 모델로 같은 입력을 여러 번 예측
  - 결과가 서로 다르면 불확실성이 높다고 판단

### Dropout을 통한 불확실성 측정
- Dropout은 학습 중 overfitting을 방지하기 위한 기법
- **테스트 시에도 dropout을 적용하면 매번 다른 결과가 나옴**
- 이것을 **Bayesian 추론의 근사**로 해석할 수 있음
- 즉, **dropout으로 모델의 예측 분포를 샘플링**할 수 있고, 그 **분포의 분산이 예측의 불확실성**이 됨






















