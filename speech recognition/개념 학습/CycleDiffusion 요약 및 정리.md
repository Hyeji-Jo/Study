## 0. 초록 - Abstract
### 1) 목적
- **음성 변환 (Voice Conversion, VC)**: 한 화자의 음성을 다른 화자의 음성으로 바꾸되, 언어적 정보는 그대로 유지
- **응용 분야**: 음성 합성, 악센트 수정, 의학, 보안, 프라이버시 보호, 엔터테인먼트 등

### 2) 기존 한계점
- 기존 Diffusion Model(DM)은 **재구성(reconstruction)** 위주 학습 → GAN처럼 **변환 경로에 최적화되지 않음**
- 즉, 변환 음성의 품질이 제한됨

### 3) 제안 방법: CycleDiffusion
- **핵심 아이디어**: 서로 반대 방향으로 작동하는 두 개의 DM을 구성하여 **Cycle Consistency**를 학습
  - DM1: Source → Target 화자
  - DM2: Target → Source 화자
- **Cycle Consistency Loss**를 통해 두 방향을 동시에 학습 → 변환 + **재구성 품질 동시 향상**

### 4) 기여
- **양방향 변환 구조**와 **cycle consistency loss** 도입으로 기존 DM 대비 변환된 **음성의 품질 향상**
- **실험 검증**: VCTK 데이터셋에서 성능 입증


## 1. 개요 - Introduction
### 1) 음성 변환(Voice Conversion, VC)이란?
- **정의**: 화자의 음성을 다른 화자의 음성으로 변환하되 **언어적 정보(내용)** 는 유지하는 기술
- 활용 분야: 음성 합성, 억양/악센트 변경, 의료, 보안, 개인정보 보호, 엔터테인먼트 등
- **VC 일반 프로세스**
  - **특징 추출**: 일반적으로 spectrogram 사용
  - **특징 변환**: 화자의 음성 특성을 변경
  - **waveform 재구성**: 신경망 vocoder 활용

### 2) 기존 VC 방법과 한계
- **VAE 기반 VC**
  - 입력 음성을 저차원 latent vector로 인코딩 → target 화자 정보와 함께 디코딩
  - **장점**: 병렬 데이터 불필요, many-to-many 가능
  - **한계**: 변환 음질이 GAN 기반보다 낮음 (재구성 위주)
- **GAN 기반 VC (특히 CycleGAN)**
  - 두 개의 GAN을 이용한 **Cycle-consistency 학습**
    - A→B (소스→타겟), B→A (타겟→소스)
  - **장점**: 재구성 중심이 아닌 **변환 경로** 학습 → 음질 향상
  - **한계**: 학습 불안정성, mode collapse, many-to-many 시 계산 복잡도↑

### 3) Diffusion Models(DMs) 기반 VC의 등장
- **원리**
  - Forward: 점점 noise 추가 → 정규 분포
  - Reverse: 점진적 noise 제거 → 원 데이터 복원
- **장점**: 안정적 학습, 고품질 생성
- **적용 사례**: 이미지 생성에서 시작 → 음성 변환 분야에도 확장
- **문제점**: VAE처럼 **재구성 경로만 학습** → 변환 음질 한계 존재

### 4) 제안 방법: CycleDiffusion
- **아이디어**: 2개의 DM 사용 + cycle consistency loss 적용
  - **DM1**: 소스 음성 → 타겟 음성
  - **DM2**: 타겟 음성 → 소스 음성 복원
- **효과**:
  - 재구성 경로 + 변환 경로 모두 학습 가능
  - **고품질 음성 변환** 달성

### 5) 본 논문의 기여 요약
- CycleDiffusion 모델 제안: 변환 & 재구성 경로 모두 학습
- **안정적인 학습 알고리즘 설계**
- **병렬 데이터 없이** 음성 변환 적용 가능
- **VCTK 데이터셋 실험**을 통해 유용성 입증

### 6) 논문 구성 안내
- 2장: 관련 연구
- 3장: 제안 모델 상세 설명
- 4장: 실험 및 성능 분석
- 5장: 결론
- 부록: 표기법 명세

## 2. 관련 연구 - Related Works
### 1) Diffusion Model 기반 음성 변환
#### 핵심 개념
- **Diffusion Model (DM)** 은 데이터를 노이즈화하는 **순방향(diffusion)** 과정과, 그 노이즈를 제거해 원 데이터를 복원하는 **역방향(reverse)** 과정으로 구성됨
- 기존 VAE처럼 **입력 재구성에 중점**을 두어 학습되며, 이는 **변환된 음성 품질을 제한함**

#### 수학적 모델링 (확률 미분 방정식 기반)
- **순방향 과정 (Forward SDE)**
  - 𝑑𝑥ₜ = ½ βₜ (𝑥̄ − 𝑥ₜ) 𝑑𝑡 + √βₜ 𝑑𝑤ₜ
  - 𝑥₀: 입력 음성 특징 (언어 정보 포함)
  - 𝑥̄: 평균 상태 (보통 0), 𝑤ₜ: Wiener process
- **역방향 과정 (Reverse SDE)**
  - 𝑑𝑥̃ₜ = [½ (𝑥̄ − 𝑥̃ₜ) − ∇ log 𝑝(𝑥̃ₜ)] βₜ 𝑑𝑡 + √βₜ 𝑑𝑤̃ₜ
  - 𝑥̃ₜ: 재구성 과정의 상태
  - ∇ log 𝑝(·): score function (데이터 확률 밀도의 gradient)
 
#### 학습 메커니즘
- 목표: **forward trajectory**와 **reverse trajectory** 간의 차이를 줄이고, 𝑥₀ ≈ 𝑥̃₀ 가 되도록 학습
- 손실 함수 (Score Matching Loss): 𝐿_diffusion(𝑥₀^ζ) = 𝐸ₜ [ (1 − αₜ²) · 𝐸_{𝑥ₜ^ζ | 𝑥₀^ζ} [ ‖ S_θ(𝑥ₜ^ζ, 𝑥̄, ζ, t) − ∇ log 𝑝(𝑥ₜ^ζ | 𝑥₀^ζ) ‖²₂ ] ]
  - S_θ: 학습 가능한 score 함수 네트워크
  - 𝑥₀^ζ: 화자 ζ의 음성 특징
 
#### 변환 절차
- **화자 ζ**의 입력 𝑥₀^ζ 를 forward diffusion → 𝑥_T^ζ
  - 𝑥_T^ζ = 𝐹(𝑥₀^ζ)
- 𝑥_T^ζ 를 화자 ξ의 embedding을 주어 역방향 SDE 수행
  - 𝑥̃₀^{ζ→ξ} = 𝑅(𝑥_T^ζ, ξ)
  - score는 S_θ(𝑥̃ₜ, 𝑥̄, ξ, t) 로 대체됨
  - inference에서는 화자 임베딩만 바꿔줌
 
#### 한계점
- 학습 시 S_θ는 재구성 용도로 학습되며, inference 시에는 변환에 사용됨 → **사용 목적 불일치 문제 발생**
- 변환 목적에 맞게 S_θ 학습 방식 자체를 **변환 중심으로 전환할 필요 있음**

### 2) CycleGAN 기반 음성 변환
#### 핵심 개념
- **CycleGAN**은 **두 개의 GAN** (각각 Generator + Discriminator 쌍) 사용
  - Generator 1: 소스 → 타겟 화자
  - Generator 2: 타겟 → 소스 화자
- **Cycle Consistency Loss**를 활용해 언어 정보 보존

#### 학습 구조
- **Forward 변환**
  - Generator 𝐺₁: 𝑥_{ζ→ξ} = 𝐺₁(𝑥_ζ)
  - Discriminator 𝐷₁: adversarial loss → 𝐷₁(𝑥_{ζ→ξ})
- **Cycle 재구성**
  - Generator 𝐺₂: 𝑥_{ζ→ξ→ζ} = 𝐺₂(𝑥_{ζ→ξ})
  - Discriminator 𝐷₂: 𝐷₂(𝑥_{ζ→ξ→ζ})
- **Cycle Consistency Loss**
  - 언어 정보 보존을 위해: 𝐿_cycle(𝑥_ζ, 𝑥_{ζ→ξ→ζ}) = ‖𝑥_ζ − 𝑥_{ζ→ξ→ζ}‖₁
 
#### 양방향 학습
- 동일한 구조로 화자 ξ → ζ 방향도 학습
- 총 loss = **2× adversarial loss + 2× cycle consistency loss**

#### 한계점
- **학습 불안정성** 및 **mode collapse** 문제 있음
- 다대다 변환에서는 **계산 비용 증가**
- 파라미터 공유가 어렵고 훈련이 어려움

#### 확장 모델
- **MaskCycleGAN-VC** [12]: 보다 견고한 변환을 위해 마스킹 추가
- **wav2wav** [16]: waveform 단위 직접 변환 → 성능 향상


## 3. 제안 모델 - Proposed Method
### 1) Cycle-Consistent Diffusion (CycleDiffusion)
#### 배경 및 문제 인식
- 기존 **Diffusion Model(DM)** 기반 음성 변환은 **VAE처럼 재구성 경로만 학습함**
  - 변환 경로는 학습하지 않아 **변환된 음성의 품질이 제한됨**
- 해결책으로 **CycleGAN의 cycle consistency 개념**을 DM에 도입하여 **CycleDiffusion 제안**

#### 핵심 아이디어: Cycle Consistency Loss를 활용한 변환 경로 학습
- **목표**: 두 가지 경로 모두 학습
  - (1) **재구성 경로**: 입력 → 노이즈 → 다시 원래 음성 복원
  - (2) **변환 경로**: 소스 화자 음성 → 타겟 화자 → 다시 소스 화자
 
#### 전체 학습 프로세스
- **기본 재구성 경로 학습 (pretraining)**
  - 화자 ζ: 𝒙₍ζ₎ → 𝒙̃₍ζ₎ (DM 통과)
  - 손실: 𝓛_diffusion(𝒙₍ζ₎, ζ)
- **변환 경로 학습 (cycle loss 적용)**
  - 화자 ζ → ξ: 𝒙₍ζ→ξ₎ = ℜ(𝔽(𝒙₍ζ₎), ξ)
  - 다시 화자 ζ로: 𝒙₍ζ→ξ→ζ₎ = ℜ(𝔽(𝒙₍ζ→ξ₎), ζ)
  - cycle consistency loss 적용: 𝓛_cycle(𝒙₍ζ₎, 𝒙₍ζ→ξ→ζ₎) = ‖𝒙₍ζ₎ − 𝒙₍ζ→ξ→ζ₎‖₁
  - 동일한 과정이 화자 ξ에서도 반복됨
 
#### 최종 학습 손실 함수
- 𝓛 = 𝓛_diffusion(𝒙₍ζ₎, ζ) + 𝓛_diffusion(𝒙₍ξ₎, ξ) + λᵢ × (𝓛_cycle(𝒙₍ζ₎, 𝒙₍ζ→ξ→ζ₎) + 𝓛_cycle(𝒙₍ξ₎, 𝒙₍ξ→ζ→ξ₎))
- λᵢ: 학습 초반 0부터 시작해 점진적으로 증가하는 cycle loss 가중치

#### Figure 1 
![image](https://github.com/user-attachments/assets/3cc8ec95-d614-4253-9857-d08acf47c6a0)  

- 실선: 기존 DM 학습 (𝓛_diffusion)
- 점선/파선: 변환 경로 학습 (𝓛_cycle)
  - **파란 점선**: ζ → ξ
  - **파란 파선**: ξ → ζ
  - **빨간선**: 파란선의 역방향
- **점선은 고정된 경로**, **파선은 gradient가 업데이트되는 경로**

#### Cycle loss의 효과 (2가지 측면)
- **변환 경로 자체를 명시적으로 학습**
  - 𝒙_{ζ→ξ} → 𝒙_{ζ→ξ→ζ} 간의 차이를 최소화
- **언어 정보 보존**
  - 𝒙_ζ ↔ 𝒙_{ζ→ξ→ζ},
  - 𝒙_ξ ↔ 𝒙_{ξ→ζ→ξ} 간의 차이 최소화 → **같은 문장을 말한 것처럼 보존됨**
 
### 2) Training Algorithm
![image](https://github.com/user-attachments/assets/141f6505-f147-4d26-9024-f6727159d430)

- **Many-to-Many 음성 변환**을 위한 CycleDiffusion 알고리즘은 Algorithm 1에 명시됨
  - 학습률 η, loss 감소 기준 사용
  - 모든 화자 쌍 간에 위의 cycle consistency 과정을 적용 가능
 
### 3) 기존 연구와의 비교
- **[28]: 문서 개선(task)에 DM 기반 순환 변환 개념 도입**
  - 하지만 cycle consistency loss는 학습에 사용되지 않음
  - 단순히 두 도메인의 DM을 잠재 공간에서 연결
- **[29]: 이미지 변환(image-to-image translation)에 cycle consistency loss 적용**
  - denoising diffusion probabilistic model (DDPM) 사용 (이산 시간 기반)
  - 음성 변환(task)에 적용된 것은 아님
- **CycleDiffusion의 차별점**
  - **연속 시간 기반** score-based diffusion model 사용
  - **음성 변환(task)** 에 직접 적용됨
  - **cycle consistency loss를 학습 과정에 직접 활용**
  - **음성 변환 분야에서 최초 시도**
