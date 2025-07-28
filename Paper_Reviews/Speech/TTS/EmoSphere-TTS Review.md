# EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector for Controllable Emotional Text-to-Speech

## 요약 정리
### Problem
- 기존 Emotional TTS는 이산 감정 레이블에 의존해 평균적인 감정표현 (ex. 슬픔, 기쁨..)
  - 감정의 **세밀한 강도와 스타일(방향성)** 조절이 어려움
- 감정 레이블 수가 제한되고, 연속적인 감정 표현에 필요한 **AVD annotation 데이터는 부족**
- 감정 표현과 음성 품질을 동시에 만족시키는 **정교한 제어 방식이 부재**

### Contributions
1. **Spherical Emotion Vector 제안**  
   - 감정 스타일과 강도를 **방향(θ, φ)** 과 **거리(r)** 로 분리해 표현  
2. **AVD 기반 pseudo-label 사용**  
   - 감정 주석 없이도 감정 표현이 가능하도록 음성에서 AVD 추출  
3. **Spherical Emotion Encoder 설계**  
   - 감정 스타일, 강도, 클래스 정보를 통합한 임베딩 벡터 구성  
4. **Dual Conditional GAN 적용**  
   - 감정 + 화자 조건을 동시에 고려한 훈련으로 **품질과 감정 표현력 동시 향상**  
5. **강도/스타일 조절 가능한 제어형 Emotional TTS 구현**  
   - Inference 시 감정 조절 가능

### Method
- **AVD 추출 (wav2vec2 기반 SER)** -> **pseudo-label로 사용**
- AVD → **구면 좌표계(Spherical Transformation)** 변환  
   - 거리 r: 감정 강도  
   - 각도 θ, φ: 감정 스타일
- 감정 스타일/강도/class 정보를 Spherical Emotion Encoder에서 정제 -> 감정 임베딩 생성
- 감정 임베딩 + 화자 임베딩 → FastSpeech2에 입력
- 음성 품질 향상을 위해 Dual Conditional GAN 구조로 학습

### Experiments & Setup
- **Dataset**: ESD (5 emotions × 10 speakers × 350 utterances = 17,500 samples)
- **Acoustic model**: FastSpeech 2
- **Vocoder**: BigVGAN 사용
- **SER 모델**: wav2vec 2.0 기반
- **Metrics**: UTMOS, nMOS, sMOS, WER, CER, RMSEf0, F1 V/UV, SECS, EER, ECA 등
- **Ablation**: 벡터 제거 / GAN 제거 시 성능 비교

### Results
- 제안한 EmoSphere-TTS가 전반적인 품질(nMOS/sMOS), 감정 표현력(ECA), 음성 정확도(WER/CER)에서 기존 방식보다 우수
- 감정 강도와 스타일을 조절했을 때 pitch 변화 등 **prosody의 명확한 차이** 발생
- Ablation 결과
  - Spherical Emotion Vector 제거 시 표현력 감소
  - GAN 제거 시 음질과 감정 전달력 감소
- **감정 강도 조절 실험에서 사용자 인식률 향상**

### Limitations
- 문장 수준(global level) 감정 표현만 가능  
  - 감정의 **미세한 부분(phoneme level control)** 은 다루지 않음
- AVD pseudo-label은 정확한 ground truth가 아니므로  
  - **정확도에 한계**가 있을 수 있음
- 모델이 복잡하고, inference 제어에 추가 입력이 필요함 (θ, φ, r)

### Insights & Idea
- 감정은 단순한 클래스(label)보다 **연속적이고 복합적인 벡터 표현**이 적합함
- **스타일과 강도의 분리 표현**은 감정 제어의 핵심
- Spherical coordinate 방식은 감정 공간을 **구조적이고 직관적으로 제어** 가능하게 함
- GAN 구조에서 **multi-aspect 조건(discriminative guidance)** 을 활용하면, 음성 품질과 감정 표현력을 동시에 향상시킬 수 있음
- 해당 구조는 **Emotional Voice Conversion, multi-lingual 감정 합성** 등으로 확장 가능성 있음


<br>  
  
## 0. Abstract
### 기존 연구의 한계
- 기존 감정 TTS는 고정된 감정 레이블(예: 기쁨, 슬픔 등)만 사용 → **세밀한 감정 표현 불가능**
- 같은 감정 레이블이라도 **실제 음성은 스타일과 강도가 다양**하지만 이를 반영하기 어려움

### 제안 방법
- **AVD (Arousal, Valence, Dominance)** 기반의 감정 정보를 사용하여 **구면 좌표계(Spherical Emotion Vector)** 로 감정 스타일(방향)과 강도(거리)를 정밀하게 조절
  - Arousal (각성도) : 감정의 에너지나 흥분 수준
  - Valence (정서가치) : 감정의 긍정/부정 정도
  - Dominance (지배성) : 감정에서 느껴지는 통제력 
- **사람의 감정 주석 없이도**, 음성에서 pseudo-label 형태로 AVD를 추출해 사용
  - Pseudo-label (의사 레이블) : 정답 레이블이 없는 경우, **모델이 예측한 결과를 임시 레이블처럼 사용**
  - 즉, 사람이 직접 입력하지 않고, 감정 인식 모델이 예측한 값을 pseudo-label로 활용
- **Dual Conditional GAN** 구조 도입 → 감정과 화자 특성을 모두 반영해 음성 품질 향상
  - **GAN**
    - Generator (G): 가짜 데이터 생성
    - Discriminator (D): 진짜/가짜 판별
  - **Conditional GAN**
    - 단순히 진짜/가짜만 구분하지 않고, **조건에 맞는 데이터인지도 확인**
  - **Dual Conditional GAN**
    - Discriminator에 두 가지 조건을 동시에 넣는 구조
    - 본 논문에서는 감정 정보 / 화자 정보 삽입
    - **이 음성이 주어진 감정과 화자 조건에 잘 맞는가 확인**

### 결과
- 제안한 모델은 감정의 **스타일과 강도를 세밀하게 제어 가능**
- 기존보다 **더 표현력 있는 고품질 감정 음성 합성을 달성**



<br>  
  
## 1. Introduction
### 연구 배경
- Emotional TTS는 최근 빠르게 발전하고 있으며, 감정을 포함한 음성 합성에 대한 관심이 높아지고 있음
- 하지만 아직도 **해석 가능하고 정교한 감정 조절에는 한계가 존재**
  - 같은 ‘슬픔’ 레이블을 붙인 음성이라도 실제 감정 표현은 다양함
  - (예: 외로움, 억울함, 체념 등) → 평균 스타일로만 합성하면 다양성 손실

### 기존 접근 방식의 한계
- **감정 레이블 기반 방법**
  - Discrete emotion labels (슬픔, 기쁨 등)을 입력으로 사용
  - 예시
    - Relative Attribute: 감정 강도 간 상대 순서를 학습
    - EmoQ-TTS: 선형 판별 분석(LDA) 기반 강도 양자화
  - 한계
    - 이산 레이블이 감정의 복잡성을 제대로 담지 못함
    - '슬픔’이라는 하나의 라벨 안에 너무 다양한 감정 존재
- **Reference 기반 방법**
  - **기준 음성(reference audio)**를 보고 감정을 모방
  - 예시: scaling factor를 곱해 감정 강도 조절
  - 한계
    - reference와 합성 음성 간 불일치 발생
    - scaling factor 조정이 불안정한 품질로 이어지기도 함
   
### 감정 차원 기반 접근 (Arousal, Valence, Dominance)
- Russell’s 2D 감정 모델 → AVD 3D 확장
- **AVD는 연속적이고 세밀한 감정 표현 가능**
- BUT
  - AVD 레이블이 있는 **데이터셋이 매우 제한적**
  - AVD 공간에서 모델이 감정 표현을 **직관적으로 제어하기 어려움**

### 제안
- **핵심 아이디어**
  - AVD 값을 기반으로 한 **구면 감정 벡터(Spherical Emotion Vector) 생성**
    - **감정의 스타일은 구면 좌표의 방향 (각도 θ, φ)**
    - **감정의 강도는 중심으로부터의 거리 r**
  - 감정 레이블 없이 **SER 모델로 추출한 pseudo-label AVD 사용**
  - 정규화된 구면 벡터를 통해 감정 스타일과 강도를 분리해서 조절 가능
- 품질 향상을 위한 추가 구조
  - **Dual Conditional Discriminator** 도입
  - 감정 정보 + 화자 정보 동시 반영 → 더 자연스러운 감정 표현 가능
- 성능
  - 실험 결과, 음성 품질과 감정 표현력 모두 향상
  - 공개 웹사이트에서 샘플 확인 가능 (https://EmoSphere-TTS.github.io/)
 
 

<br>  
  
## 2. EmoSphere-TTS
<img width="1551" height="619" alt="image" src="https://github.com/user-attachments/assets/a477b4a4-f082-4f87-86e7-597608fc5efd" />

### 1) Emotional style and intensity modeling
- 감정의 복잡한 특성을 정밀하게 표현하기 위해, 기존 **Cartesian 좌표계 (AVD)** 를 **Spherical 좌표계로 변환**
- 감정의 **강도(intensity)** 는 **중심점(중립 감정)에서의 거리**
- 감정의 **스타일(style)** 은 구면 좌표의 **방향 (각도 θ, φ)** 로 정의

#### 감정 벡터 정의
- wav2vec 2.0 기반의 SER -> $$e_{ki} = (d_a, d_v, d_d)$$
  - $$d_a$$: arousal
  - $$d_v$$: valence 
  - $$d_d$$: dominance
  - 이는 k-번째 감정의 i-번째 샘플
  - 각각 데카르트 좌표계에서 약 0에서 1 사이의 값

#### Step1 중립(Neutral) 감정 중심 정하기
- 중립 감정들을 평균 내어 기준점 $$\mathbf{M}$$으로 설정
- 즉, 감정이 없거나 평온한 상태들이 감정 공간에서 어디쯤에 분포하는지 평균 위치를 계산
- $$\mathbf{M} = \frac{1}{N_n} \sum_{i=1}^{N_n} e_{ni}$$
  - $$e_{ni}$$: neutral 감정의 pseudo-label 벡터들
  - $$N_n$$: 중립 샘플 수

#### Step2 기준점으로 이동 (원점 정렬)
- 모든 감정 벡터를 중립 중심으로 이동시킴
  - 모든 감정 벡터를 **중립 감정 중심 M 기준으로 평행 이동**함
  - 즉, “이 감정은 중립에서 얼마나 벗어나 있는가?”를 나타내는 상대 벡터로 변환
- 출발점을 (0,0,0)으로 이동
- $$e’{ki} = e{ki} - \mathbf{M}$$

#### Step3 Cartesian → Spherical 변환
- 이제 이 기준점 중심 벡터 $$e’_{ki} = (d’_a, d’_v, d’_d)$$를 **구면 좌표계로 변환**
- **감정 강도 r – 중심으로부터의 거리**
  - $$r = \sqrt{d_a’^2 + d_v’^2 + d_d’^2}$$
  - 감정의 강도, 즉 “중립으로부터 얼마나 멀리 떨어져 있는가”
  - 멀수록 강한 감정, 가까울수록 중립에 가까운 감정
- **감정 스타일 각도 ①: 세타 $$\vartheta (극각, θ)$$**
  - z축(dominance) 기준의 기울기 
  - $$\vartheta = \arccos\left(\frac{d_d’}{r}\right)$$
  - 지배성 방향으로 얼마나 기울었는지
- **감정 스타일 각도 ②: 파이 $$\varphi (방위각, φ)$$**
  - arousal과 valence 사이에서 어떤 방향을 가리키는지 
  - $$\varphi = \arctan\left(\frac{d_v’}{d_a’}\right)$$
  - arousal-valence 평면에서 감정의 방향성

#### Step4 감정 강도 정규화 (0~1 범위로)
- r 값을 **min-max normalization**으로 0~1 범위로 정규화
  - r는 raw 값이기 때문에 그대로 쓰면 범위가 너무 넓거나 한쪽에 쏠릴 수 있음 
- 이상치에 민감하지 않게 **IQR (interquartile range)** 기법 사용
  - $$\text{IQR} = Q_3 - Q_1$$
  - 중앙 50%의 데이터 범위

#### Step5 감정 스타일 양자화 (Octant segmentation)
- $$\vartheta, \varphi$$를 기준으로 **8개 옥탄트로 나눔**
- 3차원 좌표축(A/V/D)의 +,– 부호 조합


### 2) Spherical emotion encoder
- 앞에서 만든 구면 감정 정보 -> **TTS 모델에 입력 가능한 벡터로 인코딩**
- 즉, 감정 스타일 벡터 (방향 θ, φ) + 강도 벡터 (거리 r) + **감정 클래스 정보 (emotion ID)** 를 하나의 벡터로 정제

#### 입력 벡터의 종류
- $$h_{sty}$$ : 감정 스타일 벡터 (θ, φ) 기반
- $$h_{int}$$ : 감정 강도 벡터 (r)
- $$h_{cls}$$ : 감정 클래스 ID (예: 슬픔, 분노 등) 임베딩

#### 차원 정렬 (Projection layer)
- 각각의 벡터는 차원이 다를 수 있으므로, **projection layer (dense layer)** 를 거쳐 차원을 맞춤

#### 벡터 결합 후 활성화 및 정규화
- $$h_{emo} = \text{LN}\left(\text{softplus}(\text{concat}(h_{sty}, h_{cls}))\right) + h_{int}$$
  - $$\text{concat}(h_{sty}, h_{cls})$$: 감정 스타일 + 감정 클래스 결합
  - softplus: ReLU처럼 음수는 0으로 만들되, 기울기 부드럽게 만들어 학습 안정성 향상
    - softplus는 ReLU처럼 양수일수록 출력이 커지지만, 음수도 완전히 0이 되진 않음
    - **기울기(gradient)가 항상 존재**해서 학습이 부드럽게 이어짐 
  - LayerNorm: 레이어 정규화 → 학습 안정성, convergence 향상
  - 마지막에 **강도 벡터 $$h_{int}$$** 를 더함 → 스타일과 클래스에 강도를 반영

 
### 3) Dual conditional adversarial training
#### 목적
- 감정 표현력 + 음질을 동시에 높이기 위해 GAN 구조를 도입
  - **기존 TTS 시스템(FastSpeech2 같은)** 은 대부분 **MSE(평균제곱오차) 같은 회귀 기반 손실 함수** 사용
  - GAN의 경우 Discriminator가 **“이 음성이 진짜처럼 들리는가?”** 를 학습
  - Generator는 **더 생생하고 자연스러운 출력을 생성**하려고 학습
- **Dual Conditional Discriminator를 사용**

#### GAN 기본 구성 
- **Generator (G)** : 텍스트 + 감정 → Mel-spectrogram 생성
- **Discriminator (D)** : 진짜(Mel-clip)인지 가짜인지 판별

#### Dual Conditional 구조
- **감정 조건 (emo)** : 감정 스타일, 강도 등
- **화자 조건 (spk)** : 말하는 사람 정보 (speaker embedding)

#### Discriminator 구성
- 여러 개의 2D CNN stack (Conv2D layer + FC layer)
  - Mel-spectrogram이 2D니까   
- 입력: **랜덤한 길이의 Mel-spectrogram clip (Mel clip)**
- 조건 embedding (감정 / 화자)도 함께 입력됨
  - clip 길이에 맞춰 time axis로 확장한 뒤 concatenate

#### 손실 함수
- **Discriminator Loss $$(\mathcal{L}_D)$$**
  - $$\mathcal{L}D = \sum{c \in \{\text{spk}, \text{emo}\}} \sum_t \mathbb{E} \left[ (1 - D_t(y_t, c))^2 + D_t(\hat{y}_t, c)^2 \right]$$
  - $$y_t$$: 진짜 Mel-spectrogram (ground truth)
  - $$\hat{y}_t$$: Generator가 만든 가짜 Mel
  - $$D_t(\cdot, c)$$: 조건 c에 따른 Discriminator 출력
  - 진짜는 1, 가짜는 0 되도록 학습
- **Generator Loss $$(\mathcal{L}_G)$$**
   - $$\mathcal{L}G = \sum{c \in \{\text{spk}, \text{emo}\}} \sum_t \mathbb{E} \left[ (1 - D_t(\hat{y}_t, c))^2 \right]$$
   - G는 Discriminator가 “가짜를 진짜로 믿게” 만들도록 학습



 
### 4) TTS mode
#### 핵심 구조
- Base : FastSpeech 2
  - FastSpeech 2는 빠르고 안정적인 non-autoregressive TTS 모델
  - 텍스트(phoneme) → Mel-spectrogram → vocoder → waveform
- 추가된 입력 정보
  - $$h_{emo}$$ : 감정 임베딩 (감정 스타일 + 강도 + 클래스 정보 포함)
  - $$h_{spk}$$ : 화자 ID 임베딩

#### 통합 방식
- 감정 임베딩 $$h_{emo}$$과 화자 임베딩 $$h_{spk}$$을 concatenate해서 **variance adaptor 앞에 입력으로 넣어줌**
  - **variance adaptor**: FastSpeech 2에서 길이, pitch, energy 등을 조절하는 모듈
  - 여기에 감정/화자 정보를 주면, 감정 표현력과 발화 스타일이 달라짐

#### Inference
- 학습 시에는 감정 정보를 모델이 자동으로 예측
- **추론 시(inference)** 에는 사용자가 직접 원하는 감정 스타일(θ, φ) + 강도(r) 값을 수동으로 설정 가능
  - 감정 표현을 직접 조절 가능 



 

<br>  
  
## 3. Experiments and results
### 1) Experimental Setup
- **Dataset**: ESD (Emotional Speech Dataset) 사용  
  - 5가지 감정: neutral, happy, angry, sad, surprise  
  - 10명의 화자가 350개의 문장을 감정별로 말한 데이터셋 (총 17,500개 샘플)
- **Mel-spectrogram 파라미터**
  - STFT: hop size=256, window size=1024, FFT size=1024
  - Mel-filter: 80개 bin 사용
- **Optimizer**
  - AdamW (β1=0.9, β2=0.98)
  - TTS 학습률: 5×10⁻⁴ / Discriminator 학습률: 1×10⁻⁴
- **훈련 환경**: RTX A6000 GPU, 약 24시간 훈련
- **Vocoder**: BigVGAN 사용 (공식 사전학습 모델)


### 2) Implementation Details
- **Acoustic model** (FastSpeech 2 기반)
  - FFT 블록: 4개 레이어, hidden=256, filter=1024, kernel=9
- **AVD Encoder**
  - wav2vec 2.0 기반 SER 모델 사용 (pseudo AVD 추출)
- **Discriminator**
  - 각 condition에 대해 projection layer (hidden=128)
  - 다양한 시간 길이 ([32, 64, 96])로 랜덤 clip 생성하여 학습
 

### 3) Model Performance
  
#### 비교 모델
| 모델 | 설명 |
|------|------|
| FastSpeech 2 + Emotion Label | 레이블 기반 감정 입력 |
| FastSpeech 2 + Relative Attribute | 감정 강도 순서 학습 |
| FastSpeech 2 + Scaling Factor | 감정 임베딩에 스칼라 곱 |
| EmoSphere-TTS (Proposed) | 구면 감정 벡터 + dual GAN |
| w/o Spherical Emotion Vector | 감정 ID만 사용 (벡터 X) |
| w/o Dual Conditional Discriminator | GAN 없이 감정 벡터만 사용 |
  
#### 주요 평가 지표 (↑: 높을수록 좋음, ↓: 낮을수록 좋음)
- **UTMOS, nMOS, sMOS**: 자연스러움, 감정 표현력 평가
- **WER, CER**: 음성 인식 정확도 → 발음의 명확성
- **RMSE (f0, period), F1 V/UV**: prosody
- **SECS, EER**: 화자 유사성
- **ECA**: 감정 분류 정확도

#### 주요 결과
- **EmoSphere-TTS**가 전반적으로 가장 높은 **nMOS/sMOS** 달성
- ablation 실험에서
  - spherical 벡터 제거 시 품질 하락
  - GAN 제거 시 감정 표현력 하락


### 4) Emotion Intensity Controllability
- **목표**: 감정 강도(weak/medium/strong)가 음성에서 잘 표현되었는지 평가
- **방법**: 각 강도에 따라 합성된 음성쌍을 제시하고 “어느 쪽이 더 강한 감정인가?”를 사람이 선택
- **결과**
  
| 감정 | 모델 | Weak<Medium | Medium<Strong | Weak<Strong |
|------|------|--------------|----------------|---------------|
| Angry | EmoSphere-TTS | **0.72** | **0.75** | **0.79** |
| Sad   | EmoSphere-TTS | 0.62 | 0.42 | 0.48 |
| Happy | EmoSphere-TTS | **0.80** | **0.66** | **0.84** |
| Surprise | EmoSphere-TTS | **0.69** | **0.76** | **0.79** |
- **분석**
  - 기존 방식(Relative Attribute, Scaling Factor)은 감정별 성능 편차 큼
  - EmoSphere-TTS는 강도 표현이 전체적으로 가장 일관적이고 우수함
  - 특히 happy, angry, surprise 감정에서 pitch 변화와 일치


### 5) Emotion Style Shift
- **목표**: 감정 스타일(방향)을 바꾸면 음성 특성이 어떻게 바뀌는지 시각화
- **방법**: 동일한 문장, 동일한 감정 강도에서 감정 스타일(θ, φ)을 변경
  - pitch track 분석
- **결과 해석**
  - Arousal ↑ → pitch 상승
  - Valence ↑ → 평균 pitch 상승
  - Dominance ↑ → pitch 변화 폭 감소
  - 스타일 vector를 바꾸면 pitch 패턴도 명확하게 변화
    - 감정 스타일(방향성)이 실제 prosody에 반영되었음을 확인


 

<br>  
  
## 4. Conclusion
### 연구 핵심
- 감정의 **스타일(style)** 과 **강도(intensity)** 를 정밀하게 제어할 수 있는 시스템인 **EmoSphere-TTS**를 제안
- 기존 방식
  - 감정을 이산적인 레이블 분류
  - 평균적인 감정 표현 사용
  - 세밀한 감정 제어 어려움
- EmoSphere-TTS
  - **AVD (arousal, valence, dominance)** 값을 **pseudo-label**로 추출
  - 이를 **구면 좌표계(Spherical Coordinate System)**로 변환
  - 감정의 방향(스타일)과 거리(강도)를 **분리 표현**
  - **Dual Conditional GAN**을 통해 감정 + 화자 특성을 반영한 고품질 음성을 생성

### 실험 결과 요약
- 제안한 방법은 다음 측면에서 우수한 성능을 보임
  - 감정 스타일 및 강도 조절 능력
  - 감정 표현력 (emotion similarity)
  - 음성 자연스러움 (MOS)
  - 발음 정확도 및 화자 유사성

### 향후 연구 방향
- 현재는 **문장 수준**의 글로벌 감정 정보만 사용
- 향후에는 **음소(phoneme) 수준의 fine-grained 감정 제어**로 확장할 계획
  - 더 정교한 감정 표현 가능

- 또한, 본 방법은 감정 TTS뿐 아니라 **Emotional Voice Conversion (EVC)** 분야에도 적용 가능성이 높음
