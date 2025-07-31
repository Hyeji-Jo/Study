# Consistency Based Unsupervised Self-training For ASR Personalisation
## 요약 정리
### Problem
- 대규모 음성 데이터로 학습된 ASR 시스템은 훈련에 등장하지 않은 사용자(화자)의 발화 특성(억양, 발음 등)이나 환경 조건(소음, 울림 등)에서 성능 저하 발생
- 이는 **도메인 시프트(domain shift)** 문제로, 개인화(personalisation)가 해결책이지만 기존 방식은 **라벨이 있는 사용자 데이터**를 필요로 함
- 실제 사용자 환경에서는 라벨 없는 데이터만 존재하는 경우가 많아, **완전 비지도(fully unsupervised) 개인화**가 매우 필요함


### Contributions
1. **Consistency Constraint (CC)** 기반의 새로운 unsupervised ASR personalisation 방식 제안
   - pseudo-label 생성과 모델 학습 양쪽에 perturbation을 적용하고, **예측 일관성(consistency)**을 유지하도록 훈련
2. **NCM 기반 filtering 방법과 결합**하여 pseudo-label의 품질을 개선
   - NCM은 LAS 디코더 출력 feature로부터 예측이 정확한지를 판단하는 lightweight binary classifier
3. 제안한 방식은 **DUST, CT 등 다른 filtering 기법과도 호환 가능(agnostic)**
4. 사전 학습된 모델을 fine-tune 하되, **모델 구조를 변경하지 않고 파라미터 효율적으로 업데이트**함



### Method
- 전체 파이프라인
  1. **Data Filtering**: NCM 등을 통해 레이블 없는 사용자 데이터 중 신뢰도 높은 샘플 선택
  2. **Pseudo-labeling**: filtered data에 perturbation(SpecAugment)을 가해 모델 예측을 의사 레이블로 생성
  3. **Consistency Training**: 동일 입력에 다른 perturbation을 적용한 경우에도 예측이 동일하도록 학습
- 손실 함수(RNN-T 기반)
  - pseudo-label $$\hat{y}$$, perturbed input $$\tilde{x}에 대해  
  - $$\mathcal{L} = -\log P(\hat{y} \mid \tilde{x})$$



### Experiments & Setup
- **Pre-training**: 20K 시간 영어 데이터 (LibriSpeech + in-house)
- **Personalisation test data**
  - 12명의 화자, 3가지 스타일: Apps, Contacts, Dictations
  - Apps & Contacts → 학습용 / Dictations → 테스트용 (held-out)
- **모델**: Two-pass ASR (1st: Conformer Transducer, 2nd: LAS)
- **Filtering 방법**: NCM (기본), DUST, Confidence Thresholding
- **Training 설정**
  - N 라운드 반복 학습
  - SpecAugment + dropout을 통한 dual-perturbation 적용


### Results
- **성능 요약 (NCM + CC 기준)**
  - Apps: 22.66 → **18.73% WER** (17.3% WERR)
  - Contacts: 23.49 → **21.79% WER**
  - Dictation: 9.43 → **8.67% WER** (8.1% WERR)

- **기타 실험 결과**
  - Filtering 방법이 달라도 CC는 강력하게 작동 → label noise에 robust
  - Epoch 수는 3이 적절 (1은 underfitting, 5는 overfitting)
  - 사용자별 WER이 최대 45%까지 다양 → 개인화 필요성 강조


### Limitations
- pseudo-label이 초기부터 크게 틀릴 경우, consistency 훈련도 불안정할 수 있음
- 데이터 perturbation 및 학습 스케줄(N, M)의 튜닝이 성능에 민감
- LAS (2nd pass)는 학습 중 업데이트하지 않으므로, 개선 여지 있음


### Insights & Idea
- **Consistency Constraint는 비지도 학습에서 매우 강력한 regularizer**로 작동하며, pseudo-label 품질이 일정 수준 이상만 되어도 안정적 훈련 가능
- filtering quality가 낮더라도 consistency 기반 self-training을 통해 일반화 성능을 끌어올릴 수 있음
- 향후
  - Semi-supervised 학습 확장
  - 코드스위칭, 다국어 환경 적용 가능성
  - lightweight ASR + on-device adaptation 시나리오에 직접 적용 가능


<br>  
  
## 0. Abstract
### 문제 배경
- 현대 ASR(Automatic Speech Recognition) 시스템은 대규모 사용자 데이터를 기반으로 학습되지만, 훈련 시 보지 못한 개별 사용자에게는 성능 저하 발생
- 주요 원인: domain shift
  - 사용자 발화 특성 (억양, 발음 등)
  - 환경적 음향 조건 (소음, 울림 등)

### 기존 방법
- ASR personalisation  
  - 개별 사용자 데이터로 모델을 fine-tuning 하여 성능 개선
  - 대부분 기존 방법은 **labelled user data 필요**
- 라벨 없는 상황에서의 개인화(unsupervised personalisation)는 특히 어려움
  - 데이터 수량 부족
  - 녹음 품질 문제

### 제안 방법
- pseudo-labeling + consistency 기반 training
- 라벨 없는 user data에서 안정적인 학습 가능

### 주요 성과
- 17.3% WER 감소 (training data 기준)
- 8.1% WER 감소 (held-out test data 기준)
- 기존 SOTA 방법들보다 우수한 성능




<br>  
  
## 1. Introduction
### 문제 배경 및 필요성
- ASR 모델은 훈련 과정에서 접하지 못한 사용자(개인)의 음성 특성(억양, 톤 등)이나 주변 환경의 음향 조건(노이즈, 잔향 등)에 직면했을 때 성능이 저하되는 문제 존재
  - 이는 사용자 데이터와 원래 훈련 데이터 사이의 **'도메인 시프트(domain shift)' 때문에 발생**
- 이러한 문제를 해결하기 위해 사용자 데이터를 활용하여 모델의 견고성(robustness)을 개선
  - **"개인화(personalisation)" 또는 "적응(adaptation)" 접근 방식 사용** 

### 비지도 개인화(unsupervised personalisation) 문제
- 대부분의 ASR 개인화 방법은 감독 학습(supervised learning)을 위해 **레이블이 지정된(labelled) 사용자 데이터를 가정**
- 하지만 실제 사용 사례에서는 레이블링된 데이터가 없거나, 데이터 크기가 제한적이고 녹음된 오디오 샘플의 품질이 좋지 않음
  - 레이블 없는 데이터(unlabelled data)만으로 개인화를 수행하는 것은 어려움 존재

### 본 연구의 접근 방식 및 기여
- **"의사 레이블링(pseudo-labelling)"을 통한 새로운 일관성 기반(Consistency Based) 훈련 방법을 개발**
  - 사전 훈련된(pre-trained) 모델 대비 레이블 없는 훈련 데이터에서 17.3%의 상대적인 WER(Word Error Rate) 감소
  - 홀드아웃(held-out) 데이터에서 8.1%의 WERR을 달성
  - 현재 최신(state-of-the-art, SOTA) 방법들을 능가하는 성능

- **일관성 제약(Consistency Constraint, CC)** 은 동일한 입력에 다양한 버전의 섭동(perturbation)을 가했을 때 모델이 동일한 결과를 예측하도록 강제하는 기법
  - 레이블 없는 데이터를 탐색하는 데 효과적 


<br>  
  
## 2. Related Work
### 데이터 필터링 (Data Filtering)
- 레이블이 없는(unlabelled) 데이터를 활용할 때 전처리 단계로 사용
- **목표**: 전체 레이블 없는 데이터셋 중에서 WER(Word Error Rate)이 낮은, 즉 품질이 좋은 레이블을 가진 샘플을 선택하는 것
- **신뢰도 기반 필터링(Confidence based filtering)** 방법이 주로 사용됨
  - **DUST**
     - dropout을 적용한 여러 번의 예측 결과의 편차를 통해 confidence 수준 추정
     - dropout 없는 예측과 비교 → Levenshtein 거리로 정확도 측정
     - 편집 거리가 특정 임계값 이상이면 해당 발화는 신뢰도가 낮다고 판단하여 제외
  - **NCM (Neural Confidence Measure)**
    - ASR 모델에서 파생된 중간 특징(intermediate features)을 사용하여 별도의 신경망 학습
    - 주어진 ASR 특징이 오류가 없는 의사 레이블(pseudo-label)에 해당하는지 예측

### 모델 적응 방식 (Model Adaptation Methods)
- **목적**: 전학습된 모델을 특정 사용자에 맞게 가볍게 fine-tune 하거나 보정하는 방법
- **Input feature를 변환하는 선형 변환 학습**: 훈련 분포에 맞게 입력을 보정
- **Batch normalization** 파라미터 적응
- **LHUC (Learning Hidden Unit Contribution)**
  - 고정된 차원의 임베딩 벡터를 학습하여 은닉 유닛 활성화(hidden unit activations)의 진폭 변경 
- **단점**: 어떤 layer를 적응할지 결정하기 어렵고, 모델 구조 변경이 필요

### 엔트로피 최소화 (Entropy Minimisation)
- **목적**: 레이블 없는 타겟 도메인(target domain) 샘플에 대한 모델의 불확실성을 줄이는 것
- 사전 학습된 모델에서 나오는 출력 확률 기반 엔트로피 손실(entropy loss)을 최소화하여 작동
- 이 방법의 한 가지 문제는 초기 예측이 부정확할 경우 모델이 잘못된 방향으로 "드리프트"될 수 있다는 것



<br>  
  
## 3. Background
<img width="762" height="597" alt="image" src="https://github.com/user-attachments/assets/8352539e-e667-41f6-98b5-601e4bdcc531" />

### 3.1 ASR model
- 사용된 모델: Streaming Two-Pass ASR 시스템
  - 이 모델은 두 개의 서브 모델로 구성
  - **Parent Model (첫 번째 통과 모델)**
    - Conformer 트랜스듀서(transducer) 기반
    - Transcription network (conformer blocks)
    - Prediction network (LSTM)
    - Joint network (dense layer)

  - **Second-Pass Model (두 번째 통과 모델)**
    - LSTM 기반의 인코더-디코더 모델(LAS)
    - 첫 번째 통과 예측을 보정하여 정확도를 높임

### 3.2 Data filtering methods
- 미리 학습된 2-pass ASR 모델이 생성한 전사(transcripts)를 필터링하기 위해 **NCM(Neural Confidence Measure) 이진 분류 모델 사용**
- **NCM의 기능**
  - ASR 결과에 대한 신뢰도를 추정하여, 생성된 의사 레이블(pseudo-labels)이 정확한지(WER=0) 또는 오류가 있는지(WER>0) 예측
  - WER=0으로 예측된 샘플만이 필터링된 데이터셋으로 선택
- **NCM 입력 특징**
  - ASR 모델에서 파생된 두 가지 유형의 중간 특징(intermediate features)
  - 즉 두 번째 통과 디코더 출력(second pass decoder output)과 빔 스코어(beam scores) 사용 
- 다른 필터링 방법
  - 논문에서는 DUST와 Confidence Thresholding (CT)이라는 두 가지 다른 데이터 필터링 기술과도 추가 실험 진행
    - **CT (Confidence Thresholding)**
      - 각 예측의 토큰별 log-softmax score를 합산
      - ROC curve를 통해 임계값(threshold)을 선정 → binary WER=0/1 예측 



<br>  
  
## 4. Proposed Method
### 전반적인 파이프라인
- **데이터 필터링**
  - 레이블이 없는 사용자 데이터 X를 필터링하여 품질이 좋은 데이터 $$\hat{X}$$를 선별
- **Consistency Constraint(CC) 기반 훈련**
  - 필터링된 데이터를 사용하여 사전 학습된 ASR 모델을 개인화
  - 이 과정에서 모델은 데이터에 다양한 변형(perturbation)을 가하더라도 일관된 결과를 예측하도록 학습

### Consistency 기반 self-training
- CC는 모델이 동일한 입력에 다양한 종류의 변형을 가했을 때에도 동일한 결과를 예측하도록 강제하는 기법
- 이는 모델의 일반화 성능을 향상시키는 데 효과적
- **적용 방식**
  - 이 연구에서는 CC를 유사 레이블(pseudo-labeling) 생성 과정과 모델 훈련 과정 모두에 적용
- **기존 연구와의 차별점**
  - 기존에는 CC가 주로 반지도 학습(semi-supervised learning)에서 보조 손실(auxiliary loss)로 사용
  - 이 논문에서는 **완전히 비지도(fully unsupervised) 설정**에서 ASR 개인화를 위한 유일한 손실(only loss)로 CC를 활용

### 훈련 과정 (그림 Fig. 2 & 알고리즘 1 기준)
- **Input**: 사전학습된 ASR 모델 f, 초기 파라미터 θ, 레이블이 없는 데이터 X
- **DataFilter(X)** -> X̂  -> 필터링된 데이터셋
- **Iterative self-training**: 총 N라운드 반복
  - D̂ ← $${(x̂, f(SPECAUG(x̂), θi))}$$
    - SpecAugment로 증강된 필터링된 오디오 샘플을 사용하여 유사 레이블 $$\hat{D} 생성
    - 즉, 입력에 무작위 변형을 가한 후 모델의 예측을 유사 레이블로 사용
  - θi+1 ← Train(f, θi, D̂, M epochs)
    - 생성된 유사 레이블 $$\hat{D}$$을 사용하여 모델 f를 M 에포크(epoch) 동안 훈련하여 가중치 $$\theta_{i+1}$$ 업데이트
- **손실 함수**
  - $$\mathcal{L} = -\log P(\hat{y} \mid \tilde{x}_{train})$$
  - 표준 RNN-T 손실에 CC를 통합한 형태




<br>  
  
## 5. Experiment Setup

### 5.1 Data

- **Pre-training Data**  
  - ASR 모델은 총 20,000시간 분량의 영어 음성 데이터로 사전 학습됨  
    - 공개 corpus: LibriSpeech
    - 자체 수집 데이터: 검색, 전화, 원거리 마이크 등 다양한 도메인 포함
  - Validation set  
    - 6시간 분량 / 5,000 문장 → WER 15.69% 기록

- **Personalisation Experiment Data**  
  - **사용자 환경 시나리오를 모사한 user data** 사용
  - 총 12명의 화자 / 3가지 발화 스타일
    1. **Apps**: 앱 실행/다운로드 명령  
    2. **Contacts**: 연락처 호출/메시지 명령  
    3. **Dictations**: 일반 voice assistant 명령어  
  - 평균 발화 길이
    - Apps/Contacts: 약 2초
    - Dictation: 약 6.5분 분량
  - 개인화 과정
    - Apps & Contacts → 학습에 사용  
    - Dictations → **보류 데이터**로 일반화 성능 평가


### 5.2 ASR Model Configuration

- **1st-pass: Conformer Transducer**
  - Transcription Network: 16개의 Conformer 블록
    - 각 block: FFN → Convolution → Multi-head Attention → FFN → LayerNorm
  - Prediction Network: LSTM 2개 (dim=640)
  - Joint Network: Dense layer

- **2nd-pass: LAS Rescorer**
  - Encoder: LSTM 1개 (dim=680)
  - Decoder: LSTM 2개 (dim=680)
  - Beam size
    - 1st-pass: 4
    - LAS: 1
  - Language Model: 사용하지 않음

- **Training Hyperparameters**
  - Optimizer: Adam
  - Batch size: 16
  - Learning rate:
    - ASR fine-tuning: `5e-6`
    - LHUC training: `1e-3`
  - SpecAugment 설정
    - Frequency mask: 1개, size=13
    - Time mask: 2개, size=12
  - Dropout: 0.1 (training 시 모델 perturbation)

### 5.3 Data Filtering Method Setup

- **NCM (Neural Confidence Measure) 설정**
  - 입력 특징
    - LAS 2nd-pass decoder의 Top-K logits (token별)
    - Beam scores (log-probability)
  - 네트워크 구조
    - FC block 1: Dense(64) × 2 + Tanh  
    - Self-attention layer → 토큰 차원에서 sum  
    - FC block 2: Dense(64) × 2 → Binary output
  - 학습
    - Dataset: in-house 6시간 (train:valid = 8:2)
    - Optimizer: Adam (`lr=1e-3`)
    - Scheduler: exponential decay (0.5 per 500 steps)
    - K 값 (top-K logits): 4
    - Loss: Binary cross-entropy

- **CT (Confidence Thresholding)**
  - log-softmax score 합산 → ROC 기반 threshold 선정
  - Positive: WER=0, Negative: WER>0
  - 임계값 선택: sensitivity × specificity의 기하 평균 최대값 기준

- **DUST (Dropout-based Uncertainty Filtering)**
  - Dropout rate: 0.2  
  - Hypotheses 수: 5개  
  - 편차 측정: Dropout 없이 예측된 transcript와 dropout 적용된 예측들 간의 Levenshtein 거리  
  - Threshold: 0.1 (최적값)





<br>  
  
## 6. Result and Analysis

### 6.1 ASR Personalisation Results

- **비교 대상 baseline**
  - **NST (Noisy Student Training)**: pseudo-labeling + SpecAugment (pseudo-label 시에는 augmentation 없음)
  - **EM (Entropy Minimisation)**: LAS decoder의 token-level posterior entropy 최소화
  - **LHUC (Learning Hidden Unit Contribution)**: hidden unit scaling vector 학습
  - **NCM+EM**, **NCM+LHUC**, **NCM+CC**, **NCM+CC+LHUC**

- **훈련 시 규칙**
  - Fair comparison 위해 NST, EM, CC 모두 first-pass model만 업데이트
  - LHUC는 transcription network 내 17개 layer에 적용 (conformer block + subsampling output)

- **결과 요약 (Table 2)**

| Method            | Apps WER (%) | Contacts WER (%) | Dictation WER (%) |
|-------------------|--------------|------------------|-------------------|
| Pre-trained       | 22.66        | 23.49            | 9.43              |
| NST               | 21.94        | 23.07            | 9.36              |
| EM [8]            | 20.26        | 23.23            | 9.53              |
| NCM + EM          | 19.12        | 22.22            | 8.86              |
| NCM + LHUC [10]   | 20.30        | 22.70            | 9.10              |
| NCM + CC + LHUC   | 19.30        | 21.99            | 8.64              |
| **NCM + CC**      | **18.73**    | **21.79**        | **8.67**          |

- **주요 claim**
  - NCM+CC가 모든 데이터셋에서 가장 낮은 WER을 달성 → 새로운 SOTA 달성
  - NCM+CC+LHUC 는 오히려 성능 향상이 없음 → CC 자체가 강력함
  - **Dictation**(held-out)에서 8.1% WERR → 일반화 성능 입증


### 6.2 Ablation Study

- 실험 목적: 다양한 데이터 필터링 방법/훈련 설정이 모델 성능에 어떤 영향을 주는지 분석

#### Ablation 1: Filtering 유무 및 종류 비교

| Method           | Apps WER (%) | Contacts WER (%) | Dictation WER (%) |
|------------------|--------------|------------------|-------------------|
| CC (Unfiltered)  | 21.25        | 22.71            | 9.04              |
| CC (WER=0 only)  | 20.38        | 22.40            | 8.87              |
| CT + CC          | 18.91        | 22.10            | 8.75              |
| DUST + CC        | 18.93        | 21.87            | 8.69              |
| **NCM + CC**     | **18.73**    | **21.79**        | **8.67**          |

- **분석 요약**
  - WER=0 샘플만으로 훈련한 모델이 전체 데이터로 훈련한 CC보다 더 좋음 → **pseudo-label quality 중요**
  - 하지만 DUST, CT, NCM으로 filtering 후 훈련한 CC는 **WER=0 기준보다도 성능 우수**
    - **Consistency constraint가 어느 정도의 label noise를 극복 가능**함을 의미

- **NCM의 장점**
  - 다른 방법 대비 lightweight & efficient → 디바이스 환경에 적합
  - CT는 threshold 설정이 어려움, DUST는 inference time이 길고 계산량 큼

#### Ablation 2: 라운드 수 / epoch 수 변화에 따른 성능

- 총 20 라운드까지 실험, 각 라운드당 epoch 수: {1, 3, 5}
- **결과 요약 (Figure 3 참고)**
  - Epoch = 5 → overfitting 발생 (특히 Dictation)
  - Epoch = 1 → underfitting, convergence 느림
  - Epoch = 3 → **가장 안정적인 성능 개선**
  - 전체적으로 **NST보다 CC가 훨씬 안정적이고 성능 좋음**

#### Ablation 3: 사용자별 성능 비교 (Figure 4 참고)
- 총 12명의 사용자 개별 WER 분석
- NCM+CC는 대부분 사용자에서 **NST, Pretrained보다 우수**
- 일부 사용자의 Dictation WER이 10% 미만 또는 45%까지 다양 → **개인화 필요성 강조**
- NCM+CC는 label noise가 있음에도 **robust하게 학습됨**을 입증



<br>  
  
## 7. Conclusions
- 본 논문은 **fully unsupervised ASR personalisation** 문제를 해결하기 위해 새로운 **consistency constraint 기반 self-training 방법**을 제안함

### 핵심 기여

1. **Pseudo-label 기반 self-training**에서 발생하는 불안정성을 **Consistency Constraint(CC)**로 완화
   - 동일 입력에 다양한 perturbation(교란)을 주고도 **일관된 예측을 하도록 강제**
   - 학습 안정성 향상 및 generalisation(일반화) 강화

2. **NCM (Neural Confidence Measure)** 기반 데이터 필터링
   - ASR의 중간 feature를 활용해 **WER=0인지 여부**를 예측
   - 라벨 없는 사용자 데이터에서 신뢰 높은 샘플만 선별 가능

3. **다양한 필터링 방법과의 호환성**
   - 제안한 consistency training은 DUST, CT, NCM 등 **여러 filtering 기법과 함께 사용 가능**
   - 특정 필터링 방식에 의존하지 않음 (agnostic)

### 실험 결과 요약
- 평균 **17.3% WERR (Apps)** / **8.1% WERR (Dictation, held-out)** → 기존 SOTA 대비 우수
- pseudo-label에 오류가 포함되어 있어도, consistency 학습을 통해 robust하게 학습 가능
- 다양한 발화 스타일과 억양을 가진 사용자에 대해서도 **개인화 성능 향상** 확인








