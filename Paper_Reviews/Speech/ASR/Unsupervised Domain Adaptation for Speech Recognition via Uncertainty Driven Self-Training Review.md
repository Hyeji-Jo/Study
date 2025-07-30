# Unsupervised Domain Adaptation for Speech Recognition via Uncertainty Driven Self-Training
## 요약 정리
### Problem
- ASR(자동 음성 인식) 시스템은 훈련 데이터와 테스트 데이터의 도메인이 다르면 성능이 크게 저하됨 (domain mismatch)
- Target domain에서 라벨링된 데이터를 얻는 것은 비용과 시간이 많이 드는 작업
- 라벨 없이도 target domain에 적응할 수 있는 **Unsupervised Domain Adaptation 방법**이 필요함


### Contributions
- **DUST**: Dropout을 활용한 불확실성 기반 self-training 기법 제안
- **Uncertainty filtering**을 통해 noisy pseudo-label을 효과적으로 제거
- 실험을 통해 기존 self-training(ST)보다 우수한 성능 입증
- LM 없이도 strong performance를 유지할 수 있음을 보임
- Low-resource 환경에서도 **wav2vec + DUST** 조합으로 강력한 성능 달성


### Method
- **Self-training 기반** 도메인 적응 기법
  - Labeled source data로 teacher 모델 학습
  - Unlabeled target data에 pseudo-label 생성
- **Dropout을 적용한 예측 다수 생성** → reference 예측과의 **edit distance** 비교
- **Filtering 기준**: max edit distance < τ × 문장 길이
  - 이 기준을 만족하는 예측만 학습에 사용
- 반복(iterative) self-training 구조


### Experiments & Setup
- **Source domain**: WSJ (뉴스 읽기 음성)
- **Target domains**: TED-LIUM 3 (강연), SWITCHBOARD (전화 대화)
- **ASR 모델**: CNN + Transformer encoder, CTC loss
- **Data augmentation**: noise simulation + SpecAugment
- **Language model**: character 10-gram LM (KenLM), shallow fusion
- **Dropout rate**: 0.1 (training & sampling)
- **Beam search**: width = 20


### Results
- **WSJ → TED**
  - DUST로 최대 **WER 66~80% 복원**
  - Filtering threshold τ=0.3일 때 최적
- **WSJ → SWBD**
  - Domain mismatch 가장 심한 경우에도 **65.5% WER 회복**
- **Low-resource + wav2vec**
  - 라벨 3시간만 있어도 WER 44.6%까지 회복 (baseline은 95.6%)
- **LM 없이도 DUST filtering은 효과적** (실험에서 확인됨)


### Limitations
- Filtering을 위해 dropout 기반 beam search를 T번 수행해야 함 → **연산량 증가**
- **Filtering threshold τ** 값은 실험적으로 튜닝해야 함
- 도메인 mismatch가 극심한 경우 **초기 iteration에서 LM 사용이 도움**될 수 있음


### Insights & Idea
- 불확실성 기반 filtering은 **self-training의 가장 큰 단점(noisy label)**을 보완하는 강력한 수단
- **dropout을 활용한 prediction variance → confidence estimation**이 효과적임을 입증
- filtering 기반 self-training은 **low-resource, cross-domain, no-label 환경 모두에 활용 가능**
- 향후 **GTC, wav2vec 등 다른 semi-supervised 기법과 결합 가능성 매우 높음**



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

### DUST 알고리즘 (Self-training + filtering)
- **입력**
  - Source domain의 labeled data: $$L = \{(x_i, y_i)\}$$
  - Target domain의 unlabeled data: $$U = \{x_j\}$$

- **과정**
  - Base model 학습
    - Dropout(p) 포함된 모델 $$f^p_\theta$$를 L로 학습
      - 파라미터 $$\theta$$를 가진 모델. 학습 시 dropout 확률 p를 적용 
  - Pseudo-label 예측
    - 각 target 예제 $$x_u$$에 대해 dropout 없이 예측한 reference hypothesis $$\hat{y}^{\text{ref}}_u$$ 생성
      - 이 결과는 **기준(reference) 역할**
      - 나중에 dropout을 적용해서 나온 여러 예측들과 비교 대상
    - deterministic forward pass
      - 모델이 예측할 때 **dropout을 끄고, 항상 같은 결과**를 내는 방식
      - 정답처럼 사용될 데이터이기에 안정적이고 reproducible(재현 가능) 해야 함
  - Dropout 예측 샘플링
    - dropout을 적용하여 T개의 예측 $$\hat{y}^t_u$$ 생성
    - stochastic forward pass
      - 예측할 때 **dropout을 켜고, 무작위로 뉴런 일부를 비활성화**해서 예측이 달라지는 방식 
    - 각 t는 서로 다른 random seed를 사용
  - Edit distance 계산
    - 각 $$\hat{y}^t_u$$와 $$\hat{y}^{\text{ref}}_u$$ 사이의 Levenshtein 거리 계산 → 집합 E 생성
      - **Levenshtein 거리**
        - 두 문자열 사이에서 한 문자열을 다른 문자열로 바꾸기 위해 필요한 최소한의 편집 횟수
        - 삽입 (insertion)/삭제 (deletion)/교체 (substitution) 
    - 거리를 reference 길이로 나눠 정규화함
    - $$|\hat{y}^{\text{ref}}_u|$$: 기준 예측 문장의 문자 수 (length)
  - Filtering 조건 확인
    - $$\max(E) < \tau \cdot |\hat{y}^{\text{ref}}_u|$$이면 → **pseudo-label 수용**
      - $$\max(E)$$: 가장 worst-case edit distance (즉, 가장 틀린 샘플) 
    - 아니면 → 제외
  - 모델 재학습
    - 수용된 pseudo-label 집합 P와 labeled set L을 합쳐 새 model 학습
    - 반복 가능 

### Filtering threshold τ
- 작은 τ → 더 엄격한 filtering → 더 신뢰도 높은 pseudo-label만 채택
- 실험에서 τ=0.3 정도가 좋은 성능 보임
  - 길이의 30% 이상 차이나면 불신 

### 구현상 고려 사항
- Beam search decoding을 T번 반복해야 하므로 연산량이 큼
- 그래서 실험에서는 T=3 (3개 샘플만 생성)으로 제한




<br>  
  
## 3. Experimental Setup
### Data
- **Source domain: WSJ (Wall Street Journal)**
  - 깨끗하게 녹음된 뉴스 음성
  - 약 80시간 분량, 280명 화자
  - 미국식 발음, 스튜디오 품질 → clean + read speech

- **Target domains**
  - TED-LIUM 3 (TED)
    - 450시간 분량의 TED 강연
    - 다양한 주제, 2000명 이상의 다양한 화자
    - 발화 스타일 다양 (강연체, 자연스러움)
    - 도메인 차이 중간 정도 
  - SWITCHBOARD (SWBD)
    - 전화 대화 데이터
    - 260시간, 양방향 대화, 302명 남성 + 241명 여성
    - 미국 전역에서 수집된 다양한 억양/발음
    - 도메인 mismatch 가장 심함 


### Architecture for ASR - Encoder 구조
- 모델은 입력 음성을 받아서 **Transformer 기반 feature representation 생성**
- **Input features**
  - 80-dim log-mel spectral energies + 3 pitch features 
- **EncPre(·) (pre-encoder)**
  - 2-layer CNN
  - 256 filters, stride 2, kernel size 3 \times 3
- **EncBody(·) (main encoder)**
  - 12-layer Transformer blocks
  - 각 block 구성
    - multi-head self-attention (4 heads, dim=256)
    - 2 fully connected layers (첫 번째 FC: 1024 dim)
    - ReLU, dropout, LayerNorm 
- **Dropout 설정**
  - 학습 시 dropout rate = 0.1
  - filtering용 sampling에도 동일 dropout rate 사용

### Data Augmentation
- 모델 학습의 일반화 성능 향상을 위해 두 가지 augmentation 적용
- **Offline augmentation**
  - Simulated room impulse responses + noise 추가
- **SpecAugment**
  - Frequency mask (width 30)
  - Time mask (width 40)
  - Time warping (factor = 5)

### Language Model (LM)
- 디코딩 성능 향상을 위해 **character-level 10-gram LM 사용**
- 훈련 도구: KenLM
- 디코딩 시 적용 방식: Shallow fusion

### 학습 및 디토딩 설정
- Loss function: CTC loss
- Optimizer: Adam
- Scheduler: Transformer-style learning rate 스케줄러 (Ref [33])
- Warmup: 25,000 iterations
- Epochs: 100
- 모델 선택: validation loss 기준 top-10 모델 평균
- 디코딩: Beam search (beam width = 20)


<br>  
  
## 4. Result
- 주요 비교 대상
  - **Baseline**: source domain (WSJ)에서만 학습된 모델
  - **Topline**: target domain의 ground-truth 라벨을 사용한 지도 학습
  - **ST (All)**: filtering 없이 모든 pseudo-label 사용
  - **DUST**: dropout 기반 filtering 사용

### 실험 결과 요약
#### 실험 설정
- Source: WSJ
- Target: TED-LIUM 3
- LM: source domain 텍스트로 훈련된 10-gram char LM 사용

#### Filtering threshold τ 효과 분석 (DUST1)
| τ 값 | 선택된 pseudo-label 수 (k) | TED/test WER (%) | WER Recovery Rate (%) |
|------|----------------------------|------------------|------------------------|
| 0.1  | 7                          | 30.0             | 24.7                   |
| 0.3  | 38                         | 26.8             | **40.6**               |
| 0.5  | 70                         | 27.6             | 36.6                   |
| 0.7  | 90                         | 27.9             | 35.1                   |
| ST1 (All) | 100                  | 27.7             | 36.1                   |

- **τ = 0.3**일 때 가장 낮은 WER (가장 효과적인 filtering 기준)

#### Iterative DUST 성능 향상 (DUST1 → DUST5)
| Iteration | TED/test WER (%) | WER Recovery (%) |
|-----------|------------------|------------------|
| DUST1     | 26.5             | 40.3             |
| DUST2     | 24.3             | 50.7             |
| DUST3     | 23.5             | 54.5             |
| DUST4     | 22.4             | 59.7             |
| DUST5     | 21.1             | **66.0**         |

- **DUST 반복할수록 더 많은 깨끗한 pseudo-label이 누적되어 성능 향상**

### LM 사용 여부의 영향

#### Pseudo-label 생성 시 LM 사용 여부
- LM을 사용하지 않아도 좋은 성능 가능 (특히 WSJ → TED)
- Filtering에만 dropout-based uncertainty 사용한다면, **LM 없이도 높은 WER recovery** 달성 가능

| LM 사용 | DUST5 WER (TED/test) | WER Recovery (%) |
|---------|-----------------------|------------------|
| 사용    | 21.1                  | 66.0             |
| 미사용  | 24.7                  | 80.2             |

- **LM 없이도 DUST5는 WER 24.7% 달성 (Topline 19.0%에 근접)**


### Domain mismatch 심한 경우: SWITCHBOARD 결과

| 모델 | SWBD/test WER (%) | WER Recovery (%) |
|-------|--------------------|------------------|
| Baseline | 64.1           | 0.0              |
| ST1     | 56.8            | 19.2             |
| DUST5   | 41.7            | 58.9             |
| DUST5 (w/o LM) | **39.2** | **65.5**         |
| Topline | 26.1            | 100              |

- domain mismatch가 심한 SWBD에서도 DUST는 **ST보다 훨씬 우수**




### Low-resource 시나리오 (3시간 라벨 데이터 + wav2vec feature)

| 모델 | WSJ/test WER | TED/test WER | WER Recovery (TED) |
|-------|---------------|--------------|---------------------|
| Baseline | 43.2        | 95.6         | 0.0                 |
| wav2vec  | 37.0        | 78.2         | 21.3                |
| DUST5 (w/o LM) | 35.4  | **44.6**     | **62.4**            |

- **wav2vec + DUST 조합으로 low-resource 환경에서도 높은 성능 회복**




<br>  
  
## 5. Conclusion
### 핵심 기여
- 본 논문은 **Unsupervised Domain Adaptation for ASR** 문제를 해결하기 위해 **dropout 기반 예측 불확실성**을 활용한 self-training 기법 **DUST**를 제안
  - 모델이 예측에 대해 **얼마나 확신하는지를 측정**
  - **불확실한 pseudo-label은 제거**
  - domain mismatch 상황에서도 **신뢰도 높은 학습 데이터로만 모델을 fine-tune**할 수 있도록 함
 
### 주요 성과
- DUST는 **기존 self-training(ST)**보다 다음 측면에서 더 뛰어난 성능을 보임
  - **불확실한 pseudo-label filtering**으로 더 정확한 label 사용
  - **적은 양의 pseudo-labeled data로도 성능 향상**
  - **학습 시간 단축 가능 (필요한 데이터 수 감소)**

- 실험 결과
  - **TED-LIUM 3** 타겟 도메인: 최대 **WER 66~80% 복원**
  - **SWITCHBOARD** 타겟 도메인: 최대 **WER 65.5% 복원**
  - **Low-resource setting + wav2vec**: 제한된 labeled data에서도 성능 확보

### 향후 연구 방향 (Future Work)
- **Self-supervised representation learning** (예: wav2vec2.0)과 DUST의 결합을 더 체계적으로 연구
- **Graph-based Temporal Classification (GTC)** 등 다른 semi-supervised 학습 방법과 DUST의 결합 가능성 탐색
- **pseudo-label 생성 시 LM 사용 여부**를 동적으로 조절하는 전략 (특히 severe domain mismatch에서)















