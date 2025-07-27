# EmoSphere-TTS: Emotional Style and Intensity Modeling via Spherical Emotion Vector for Controllable Emotional Text-to-Speech

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



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
