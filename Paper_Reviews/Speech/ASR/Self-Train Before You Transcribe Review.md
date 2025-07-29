# Self-Train Before You Transcribe

## 요약 정리
### Problem
- ASR 시스템은 훈련 데이터와 테스트 데이터의 도메인이 다르면(domain mismatch) 성능이 크게 저하됨
- 기존 self-training 방식은 **별도의 unlabeled target domain 데이터가 사전에 필요하다는 한계 존재**
- 테스트 시점(test-time)에 **도메인을 미리 알 수 없거나 데이터 수집이 불가능한 경우**를 해결하기 위한 실용적 **domain adaptation 방법이 부재**

### Contributions
- **NSTI (Noisy Student Teacher at Inference)**
  - test recording 자체에 noisy student training을 적용한 **test-time self-training 방식 제안**
  - 별도의 adaptation 데이터 없이 모델을 실시간 적응시킴
- **SpecAugment 기반 augmentation 활용**
  - frequency masking이 domain mismatch에 특히 효과적임을 보임
- **발화 간 context transfer 가능성 확인**
  - recording 전체를 활용함으로써 utterance 간 dependency를 반영 (기존 방식은 independent 처리)
- **성능 향상**
  - 기존 TTA 및 self-training 방법보다 뛰어난 성능
  - 최대 **32.2% WER 감소**, 특히 CHiME-6 같은 강한 domain shift 상황에서 탁월한 성능 

### Method
- **NSTI (Noisy Student Teacher at Inference)**
  - 하나의 test recording을 여러 segment로 나누고, 이를 기반으로 test-time에 self-training 수행
  - **Teacher와 Student는 동일 모델 파라미터 공유**
  - Teacher 예측을 pseudo-label로 사용하고, student는 noise가 섞인 input을 통해 이를 학습
  - 주요 augmentation: SpecAugment (frequency masking)
  - Segment 순서를 섞고(epoch 반복), 이후 업데이트된 모델로 최종 transcription 수행

### Experiments & Setup
- **Pretraining**: Spotify Podcast Corpus (58K 시간)
- **Evaluation Datasets**
  - TEDLIUM (중립적 dev set)
  - Earnings-22, CHiME-6 (out-of-domain)
  - Rev16 (in-domain)
- **모델**: 6-layer Fast Conformer (~90M 파라미터)
- **정규화**: Batch Renormalization (소규모 batch에 적합)
- **Optimizer**: MadGrad
- **Segmenting**: context window 162초 + 12.5% stride 

### Results
- **NSTI vs Unadapted**
  - 모든 도메인에서 WER 감소 (CHiME-6: 86.5 → 59.4) 
- **NSTI vs AWMC (기존 TTA)**
  - 모든 설정에서 NSTI가 우수 (특히 augmentation 미포함 시 AWMC 성능 급락) 
- **NSTI vs NST (self-training)**
  - 100배 적은 데이터로 더 나은 성능 (E-22 기준 14.7 vs 15.0) 
- **변환 기법 비교**
  - SpecAugment가 가장 안정적이며 강건함 
- **녹음 길이 실험**
  - 긴 recording일수록 context 효과로 성능 향상 증가


### Limitations
- 현재 방식은 utterance 간 순서 정보(sequential dependency)를 활용하지 않음
  - 시간적 흐름, 담화 구조 반영이 불가능
- test-time마다 NSTI 수행 → 추론 시간 증가 (real-time factor 높음) 

### Insights & Idea
- Test-time adaptation만으로도 domain mismatch 문제를 강력하게 해결할 수 있음
- augmentation (특히 frequency masking)은 모델의 강건성을 크게 높임
- local context 기반 적응이 global domain adaptation보다 효과적일 수 있음
- **향후**
  - sequential utterance 구조를 반영하는 TTA 방식
  - augmentation 전략 다양화 및 선택 최적화


<br>  
  
## 0. Abstract
### 배경 및 문제
- ASR 시스템은 훈련 데이터와 테스트 데이터의 도메인이 다를 경우(domain mismatch) 성능이 크게 저하됨
- 기존 self-training 방법은 별도의 unlabeled target domain 데이터가 필요

### 제안 방법
- **Test-Time Adaptation (TTA) 방식**
  - 테스트 시점에서 테스트 recording 자체에 noisy student teacher training (NST)를 적용
  - 별도 adaptation dataset 필요 없음 → 데이터 수집 비용/노력 절감
- **Dynamic evaluation analogy**
  - 발화(utterance) 경계 넘어 context transfer → 긴 recording에서 유리
  - Local context 활용 → 모델이 현재 recording에 빠르게 적응 가능
 
### 주요 성과
- 다양한 dataset에서 최대 32.2% WER 개선
- 기존 self-training (separate adaptation data 사용)보다 더 큰 효과



<br>  
  
## 1. Introduction
### Domain Mismatch
- ASR 모델은 **학습 이후 고정된 상태로 추론을 수행함** → 훈련 데이터 분포에만 최적화됨
- 테스트 시점에서 **도메인이 다르면 성능이 급격히 저하됨** (e.g., 잡음 환경, 말투, 주제 등)

### 기존 방법 : Self-Training
- **Self-training / Pseudo-labelling**
  - Source domain에서 학습한 모델이 target domain의 unlabeled 데이터를 pseudo-label로 학습시킴
  - labeled data 없이 domain adaptation 가능
- **한계**
  - target domain의 **unlabeled 데이터가 사전에 있어야 함**
  - 현실에서는 수집·공유가 어렵고 비용이 큼
  - 도메인 드리프트 발생 시 재수집 필요
 
### 새로운 접근 : Test-Time Adaptation (TTA)
- TTA는 **테스트 시점에만 모델을 적응**시키는 기법
  - **모델이 추론(inference) 중에 test data에 적응(adapt)하는 방법**
  - ex. 테스트 중에도 gradient descent로 모델 weight를 조금씩 바꾸는 것 
- target domain을 사전에 몰라도 됨
- 별도의 데이터 수집 없이 현재 recording만으로 적응 가능

### 제안
- 기존 pseudo-labelling을 확장한 **Noisy Student Teacher Training (NST)** 을 test-time에 적용
  - **NSTI 방법 제안**
  - **Noisy Student Teacher Training (NST)**
    - **Teacher 모델이 만든 pseudo-label을, noise가 섞인 input을 넣은 Student 모델에게 학습시키는 self-training 방식**
    - 즉, Student는 **noise가 추가된 input을 보고 teacher의 예측을 따라가도록 학습**
  - Pseudo-labeling을 noise가 포함된 student에 적용
  - 모델 regularization 및 generalization 효과
- 발화 간 정보 전이 가능 → utterance 독립 가정 완화
  - 일반 ASR의 경우 하나의 긴 녹음을 여러 **utterance 단위(예: 한 문장)** 로 나눠 **각각 독립적으로 처리**
  - 논문의 방식은 한 utterance를 학습한 후, 그 학습 결과가 **다음 utterance에도 영향을 미침**
  - 즉, 문맥이 이어짐
- **augmentation (noise 추가)** 로 일반화 성능 향상
- 언어 모델링에서의 **dynamic evaluation과 유사**
  - Dynamic Evaluation : **언어 모델이 테스트 시점에서 이전 단어/문장을 학습하며 실시간으로 자기 자신을 개선하는 방식**
  - ex. 문장 생성 모델이 “The stock market…”까지 입력
    - 이전 단어를 기반으로 **모델 파라미터를 업데이트**
    - 다음 단어 예측 시 더 나은 성능을 냄 (e.g., “crashed” or “rebounded”)




<br>  
  
## 2. Method
<img width="399" height="179" alt="image" src="https://github.com/user-attachments/assets/45fbbad0-c6a8-4ead-b2c9-ff384c8e61d2" />

### NSTI(Noisy Student Teacher at Inference)
- NST 방법을 **“test 시점”** 에 바로 적용하는 Test-Time Self-Training 방식
- **일반적인** NST (Noisy Student Teacher)는 **학습 데이터에서 사용**
- NSTI는 **테스트용 녹음(recording)에 대해 실시간으로 NST를 수행**

### 기본 구성
- 입력: 하나의 녹음(Recording)
  - 이를 여러 개의 **segment로 나눔 (utterance보다 길 수 있음)**
  - Segment들은 **랜덤하게 섞음** (shuffled)
  - 각 segment에 대해 **pseudo-label 기반 self-training 수행**

### 주요 구조
- **모델 구성**
  - **Teacher 모델 M** 과 **Student 모델 M’** 사용
    - 두 모델은 **동일한 구조와 파라미터를 공유함**
    - 즉, 모델 하나만 두 번 forward/backward 하면 됨
- **입력 데이터**
  - X: **원본 segment의 spectrogram**
  - X’ = $$\text{Transformation}(X)$$
    - **변형된 입력 (augmentation 적용된 student input)**
- **예측 흐름**
  - P = M(X): 원본 입력 -> teacher 모델 출력 (logits/probabilities)
  - $$Y^* = \text{Decode}(P)$$: teacher 예측을 **pseudo-label로 사용**
  - P’ = M’(X’): noise가 적용된 입력 -> student 모델 예측
  - $$\mathcal{L}(P’, Y^*)$$: student 예측과 pseudo-label 간 loss로 backprop
  - 모델 weight 업데이트
  - 이 과정을 n epochs 반복하며 모델을 해당 recording에 적응시킴

### Transformation (입력 변형 방법)
- **SpecAugment (frequency masking)**
  - 특정 주파수 영역을 가림 -> robust하게 학습됨
  - 가장 안정적이고 성능 좋음
- **Identity**
  - 아무 변형 없이 원본 그대로 사용 (baseline) 
- **Random noise**
  - Gaussian 노이즈 추가 
- **CutOut**
  - 스펙트로그램 일부 사각형 영역 제거
 



<br>  
  
## 3. Prior Work
### Self-training 기반 Domain Adaptation (기존 방식)
- 보통은 target domain에 대해 **별도의 adaptation set (unlabeled)** 를 사용해 모델을 학습
  - Teacher → Pseudo-label 생성
  - Student → 그 label을 따라 학습
  - 하지만 **사전 수집된 target domain data 필요**

### TTA (Test-Time Adaptation)의 등장
- TTA 방식은 테스트 시점에서 이용 가능한 데이터만으로 모델을 적응
- 별도 adaptation set 없이, 지금 처리 중인 utterance나 recording만 사용

### SUTA: Single-Utterance TTA 방법
- **한 utterance에 대해서만 모델을 적응시키고, 그 이후에는 모델을 다시 초기화**
- 각 utterance가 독립적일 때는 괜찮지만, 현실에서는 **녹음 내 여러 발화가 서로 상관관계가 높음**
  - 성능에 한계 존재
- 즉, utterance-level TTA는 발화 간 context를 활용하지 못함 

### AWMC: 기존 TTA + Pseudo-labeling
- **online 환경에서 이전 utterance 정보를 이용**하는 pseudo-label 기반 TTA 기법
- **구조**
  - Teacher 모델 = Student 모델의 EMA(지수 평균 가중치)
    - 모델 collapse 방지용
  - **Online 방식**: 현재 utterance 이전의 것만 사용 가능 (future context 불가)
- **본 연구와의 차이**
  - AWMC는 온라인 방식이고 이전 발화만 활용하는 반면, 본 연구는 오프라인 방식으로 미래 발화도 활용하여 성능을 개선
  - AWMC는 데이터 증강(augmentation) 사용 안 함
    - 도메인 mismatch에 약함
  - EMA teacher를 사용해보았지만, teacher가 천천히 업데이트되어 오히려 성능이 떨어졌음
    - 모델 collapse 문제가 없다면 **EMA는 불필요하고 오히려 방해될 수 있음**

### Dynamic Evaluation
- NLP에서도 유사한 TTA 연구 존재
  - Low-confidence 예측을 걸러내고
  - Pretrained weight와의 유사성을 유지하면서 모델을 test-time에 업데이트
- 과거 텍스트 히스토리를 기반으로 모델을 gradient descent로 실시간 업데이트하는 방식
  - 긴 시퀀스에서 반복되는 패턴(ex: stock market 문맥)을 더 잘 학습 가능 




<br>  
  
## 4. Experimental Setup
### 4.1 Datasets
#### Pre-training용 데이터
- Spotify Podcast Corpus 사용
- 총 58,000 시간 분량의 오디오
- 다양한 도메인 포함 → 범용성 높은 ASR 모델 사전 학습

#### 평가용 데이터셋 (도메인 다양성 고려하여 선정)
- **Earnings-22**
  - 기업 실적 발표 회의 오디오
  - Out-of-domain (긴 길이, 다양한 억양)
- **Rev16**
  - 팟캐스트 에피소드 (16개)
  - In-domain (훈련 데이터와 유사)
- **Tedlium**
  - TED 강연 (10–20분)
  - 중간 도메인 (학술적, 정제된 발화)
- **CHiME-6**
  - 집에서 여러 명이 대화하는 noisy 환경
  - 강한 Out-of-domain (다중화자 + 노이즈)
- 다양한 발화 스타일, 녹음 조건, 길이, 억양 포함
- NSTI의 도메인 적응 능력 평가를 위한 구성
 
### 4.2 Model Configuration (모델 구조)
#### 모델 아키텍처
- **Conformer 기반 Acoustic Model**
- Fast Conformer 구조 사용 (subsampling 포함)

#### 주요 구조 설명
- Layer 수: 6개
- 파라미터 수: 약 90M (standard ASR model 수준)
- Context window: 162초 길이 segment 단위로 학습
  - 비교적 긴 발화 단위까지 학습 가능 

#### Normalization 전략
- 기존의 BatchNorm → Batch Renormalization으로 대체
  - **Batch Renormalization**
    - BatchNorm의 개선 버전
    - BatchNorm의 경우 batch size가 작거나 1이면 작동이 불안정
      - Batch size가 작을수록 **평균/분산이 데이터에 민감하게 요동침**
      - 예: batch size = 2면 2개의 값으로 분산을 계산 → 매우 불안정
      - 배치가 1이면 분산이 0이 되어 통계량이 무의미
    - 훈련 중에도 moving average의 평균/분산과 현재 batch 통계가 크게 다르면 보정함
  - 이유: TTA 중에는 batch 통계가 불안정해질 수 있기 때문
  - 비슷한 접근: 다른 연구는 GroupNorm 사용 (그러나 unstable)
    - GroupNorm : Batch가 아니라 채널을 그룹으로 묶어서 정규화
  - 실험에서 Batch Renorm이 더 안정적이고 성능 좋았음 

### 4.3 Hyperparameters
#### 튜닝 전략
- Tedlium dev set 기준으로 랜덤 서치로 튜닝
  - **훈련에는 사용하지 않은 검증용(dev) 데이터이며, 중간 정도의 도메인 거리(domain shift)를 가진 데이터셋** 
- 이후 다른 도메인에도 동일 세팅 사용해 generality 검증
- 일부는 dataset-specific 튜닝도 시도 (→ §5.2)

#### 사용된 세팅
- **Optimizer**: MadGrad
  - Momentumized Adaptive Dual-Averaged Gradient
  - Adam처럼 **adaptive learning rate를 사용**
  - SGD나 RMSProp보다 더 잘 수렴하는 특성 
- **Learning rate**: $$9 \times 10^{-5}$$
- **Epochs**: 5
- **Batch size**: 2
- **Augmentation**: SpecAugment - frequency masking
  - **SpecAugment**
    - 스펙트로그램(시간×주파수 이미지)에 마스킹을 가해서 일부 정보 제거
    - **Frequency masking**: 주파수 축(frequency axis)에서 연속된 영역을 가림
      - 예: 특정 주파수 구간의 정보를 아예 제거 (노이즈, 잡음 등에 강건해짐)
    - Time masking: 시간 축을 가림 (일부 구간 사라진 것처럼) -> 오히려 성능 저하  
    - Time warping: 시간 축을 비선형으로 늘리거나 줄임
- **Mask 수/크기**: freq mask 6개, 최대 크기 34
  - 한 input에 대해 **주파수 축을 총 6번 가림**
  - 각 마스킹은 **최대 34개의 주파수 bin**까지 가릴 수 있음


#### Segmenting 방식
- 슬라이딩 윈도우 기반 segment 구성
  - 컨텍스트 길이: 162초
  - stride: 12.5% (즉, segment끼리 중첩 있음)
- 추론 시에는 겹치는 segment 예측 확률을 평균하여 결과 생성 





<br>  
  
## 5. Experimental Results
### 5.1 How effective is the method?
- NSTI (제안 방식)
  - Shuffled / Ordered / Online (세 가지 variation)
- AWMC
  - augmentation 포함/미포함
- Unadapted baseline: TTA 없이 모델 그대로 

| 방법            | TEDLIUM | Earnings-22 | CHiME-6 | Rev16 |
|-----------------|---------|-------------|---------|--------|
| Unadapted 모델  | 6.2     | 18.3        | 86.5    | 14.5   |
| NSTI (Shuffled) | **5.8** | **14.9**    | **59.4**| **14.2** |
| AWMC (Aug 사용) | 6.0     | 15.7        | 75.9    | 14.2   |

- NSTI가 모든 데이터셋에서 가장 성능이 뛰어남
- 도메인 불일치가 클수록 성능 향상 폭이 큼
- in-domain (Rev16)은 소폭 개선만 있음

| 설정     | 설명                                | 성능 |
|----------|-------------------------------------|------|
| Shuffled | Segment 순서를 랜덤하게 섞음       | **가장 우수** |
| Ordered  | 자연 순서로 segment 처리           | 약간 하락 |
| Online   | 과거 utterance만 사용 (future X)   | 가장 낮음 |

### 5.2 Dependency on hyperparameters
- TEDLIUM dev 기준으로 튜닝한 값을 다른 도메인에 그대로 적용해도 안정적인 성능
- CHiME-6에 대해 별도 튜닝 시 2~4% WER 추가 개선 가능
- **결론:** NSTI는 **튜닝 민감도가 낮고 일반화 성능이 좋음**

### 5.3 Comparison to Traditional Self-Training (NST)
| 방식        | 사용 데이터           | WER (Earnings-22) |
|-------------|------------------------|-------------------|
| 기존 NST    | 105시간 adaptation 데이터 | 15.0              |
| 제안된 NSTI | 테스트 녹음 1시간        | **14.7**          |

- NSTI는 **100배 적은 데이터로 더 나은 성능**
- NST 후 NSTI를 추가해도 성능 향상 없음 (이미 분포가 sharpen됨)

### 5.4 Comparison of transformation functions
| 변환 기법       | TED | E-22 | CHiME-6 | Rev16 |
|----------------|-----|------|---------|--------|
| SpecAugment    | 5.8 | 14.9 | 59.4    | 14.2   |
| Identity (변환 없음) | 6.1 | 17.4 | 100.0 | 15.0 |
| Gaussian Noise | 5.9 | 19.2 | 97.3  | 18.5 |
| CutOut         | 5.8 | 14.5 | 56.9    | **37.9** |

- **SpecAugment (frequency masking)**가 가장 안정적이고 성능이 좋음
- 다른 기법은 domain mismatch 환경에서 오히려 성능 저하

### 5.5 Impact of recording duration
- Earnings-22의 녹음을 2.7 ~ 66분으로 나눠 NSTI 수행
- 길이가 길수록 → 더 많은 context → 성능 향상

| 녹음 길이 | 성능 개선 (WER 감소) |
|-----------|----------------------|
| 2.7분 (context window와 동일) | <1% |
| 10.9분 이상 | 유의미한 개선 |
| 전체 녹음 (~66분) | **가장 높은 성능** |





<br>  
  
## 6. Conclusion
### 제안 방법 요약
- Noisy Student Teacher (NST) 방식의 self-training을 test-time에, test recording 자체에 적용함
- 별도의 적응 데이터가 필요 없고, 현재 입력된 녹음 하나만으로도 효과적인 적응 가능

### 실험 결과 요약
- 제안한 NSTI 방식은 **기존 TTA 방법들보다 성능이 높음**
  - 특히 도메인 mismatch가 큰 CHiME-6, Earnings-22에서 큰 향상
- 심지어는 105시간짜리 별도 데이터로 수행한 기존 NST self-training보다도 더 나은 성능을 달성
  - test recording 내의 utterance들이 서로 강하게 연관되어 있고, 그 local context가 효과적이기 때문  

### 방법의 한계 및 향후 방향 (Future Work)
- **현재 방법의 한계**
  - **Utterance 간의 순차적 구조(sequential nature)** 를 고려하지 않음
  - 즉, 발화가 순서대로 이어지는 정보 (예: 시간 순, 문맥 흐름)을 사용하지 않음 
- **향후 연구 방향**
  - Utterance의 순차성 정보까지 반영하는 적응 구조 설계
  - 다양한 augmentation 전략 실험을 더 해볼 계획 











