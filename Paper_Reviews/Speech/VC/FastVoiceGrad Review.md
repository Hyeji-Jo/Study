# FastVoiceGrad: One-step Diffusion-Based Voice Conversion with Adversarial Conditional Diffusion Distillation
## 요약 정리
### Problem
### Contributions
### Method
### Experiments


<br>  
  
## 0. Abstract
#### 연구 배경
- VoiceGrad 같은 diffusion 기반 음성 변환(VC) 기법은 음질(speech quality) 및 화자 유사도(speaker similarity) 측면에서 성능이 우수함
- 그러나 역확산(reverse diffusion) 을 여러 단계 거치는 구조로 인해 추론 속도가 느리다는 한계 존재
#### 제안 방법
- 이 문제를 해결하기 위해 FastVoiceGrad를 제안
  - 기존 diffusion VC의 고성능을 유지하면서 다단계(iterative) 과정을 단 한 번(one-step) 으로 줄임
- 이를 위해 Adversarial Conditional Diffusion Distillation (ACDD) 방법을 도입
  - GAN과 diffusion model의 장점을 결합
  - 샘플링 초기 상태 설정을 재고(reconsidering initial states in sampling)함
- 실험 결과
  - One-shot any-to-any VC 실험
  - FastVoiceGrad는 기존 multi-step diffusion VC보다 우수하거나 동등한 성능을 보여줌
  - 동시에 추론 속도는 향상됨
 

<br>  
  
## 1. Introduction
### 1) 연구 배경
- **Voice Conversion (VC)** : 음성의 언어적 내용은 유지한 채, 화자 특성만 변환하는 기술
- 초기에는 **병렬 말뭉치(parallel corpus)** 를 사용한 **지도학습 방식**이었지만, 데이터 수집이 어렵다는 문제 존재
- 이후 **비병렬 VC**가 주목받음
  - 특히 **딥 생성 모델(VAE, GAN, Flow, Diffusion 기반)** 이 비약적인 발전을 이끌었음

### 2) 연구 대상 및 문제
- 이 논문은 특히 **diffusion 기반 VC에 집중함**
- 이유: 기존 대표 VC들보다 성능이 좋고, diffusion 모델은 다른 분야(예: 이미지/음성 합성) 에서도 빠르게 발전 중
- 문제점: **noise → 음향 특징** 변환 시 **역확산(reverse diffusion) 을 수십 단계 반복**해야 하므로 **추론 속도가 느림**

### 3) 제안 방법: FastVoiceGrad
- 기존의 고성능 diffusion VC(예: VoiceGrad)의 성능을 유지하면서도, **수십 단계 → 1단계(one-step)** 로 줄인 새로운 모델 **FastVoiceGrad** 제안
- 이를 위해 **ACDD (Adversarial Conditional Diffusion Distillation)** 기법을 도입
  - 기존 이미지 생성 분야의 ADD 기법을 확장
  - multi-step teacher diffusion 모델 → one-step student 모델로 knowledge distillation
  - GAN + Diffusion의 장점 결합
  - 샘플링 초기 상태를 재설계

### 4) 실험 구성 및 결과 개요
- One-shot any-to-any VC 실험에서 평가
- FastVoiceGrad는 **1단계 VoiceGrad보다 우수**
- **30단계 VoiceGrad와 유사한 성능**
- **DiffVC보다도 성능이 우수하거나 유사, 추론 속도는 더 빠름**

 

<br>  
  
## 2. Preliminary : VoiceGrad
### 1) 모델 개요
- **VoiceGrad는** diffusion 모델을 기반으로 한 **비병렬 음성 변환(VC) 모델**
  - 학습 시에 소스–타깃 음성 간 일대일 정렬된(parallel) 데이터가 필요 없기 때문
  - 학습 과정에서 음소 정보(phoneme embedding) 와 화자 정보(speaker embedding) 만을 조건(condition)으로 사용해서 mel spectrogram을 직접 예측 
- 두 가지 변형 존재
  - DSM(Denoising Score Matching) 기반
  - DDPM(Denoising Diffusion Probabilistic Model) 기반
- 본 논문에서는 DDPM 기반 VoiceGrad에 집중함
  - DSM 대비 반복 횟수를 줄이면서도 성능 유지 가능
  - DSM 기반 VoiceGrad는 Reverse 과정에서 약 100~1000 step을 필요로 함
  - DDPM기반 VocieGrad는 10step 수준으로도 비슷한 성능을 낼 수 있음
  - 즉, inference 속도 측면에서 효율적
 
### 2) Forward Diffusion
- 원 데이터 $`x_0`$ 를 노이즈 $`x_T`$ 로 점진적으로 변환
- $`q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_{t-1}, \beta_t I) \tag{1}`$
- 초기 $x_0$에서 $x_t$까지 직접 계산한 식
  - $`q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I) \tag{2}`$
  - $`x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I) \tag{3}`$


### 3) Reverse Diffusion
- 노이즈 $`x_T`$ 에서 실제 음성 특징 $`x_0`$ 로 복원
- $`p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, s, p), \sigma_t^2 I) \tag{4}`$
- $`x_{t-1} = \mu_\theta(x_t, t, s, p) + \sigma_t z, \quad z \sim \mathcal{N}(0, I) \tag{5}`$

- 평균 $\mu_\theta$는 예측된 노이즈 $\epsilon_\theta$로부터 계산
  - $`\mu_\theta(x_t, t, s, p) = \frac{1}{\sqrt{\alpha_t}} \left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t, s, p) \right) \tag{7}`$


### 4) 학습 방식
- 목표 : 모델 $\epsilon_\theta$가 실제 노이즈 $\epsilon$을 예측하도록 학습
- 손실 함수 : $`\mathcal{L}_{DDPM}(\theta) = \sum_{t=1}^T w_t \, \mathbb{E}_{x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, s, p) \|_1 \right] \tag{8}`$
- $w_t = 1$ (실험에서 고정), L1 손실 사용 (L2보다 효과적)

### 5) 음성 변환 알고리즘(Inference)
- 입력
  - $x_0^{src}$: 소스 mel-spectrogram
  - $s^{tgt}$: 타겟 화자 임베딩
  - $p^{src}$: 소스 음소 임베딩
- 알고리즘 개요
  <img width="433" alt="image" src="https://github.com/user-attachments/assets/af8b92cf-a6b4-4ff2-bce6-668701ed82de" />




 

<br>  
  
## 3. Proposal: FastVoiceGrad
### 1) Rethinking initial states in sampling
#### 배경 및 개념
- FastVoiceGrad는 단일 스텝(one-step) 역확산으로 음성 변환을 수행해야 하므로, 초기 상태 설정(sampling 시작점)인 **두 요소가 변환 성능에 영향을 미침**
  - $x$ (입력 스펙트로그램)
  - $t$ (reverse step 위치)
- 추론(inference) 시 어떤 초기값을 선택해야 source 정보는 잘 보존하면서도, 학습 분포와의 차이를 줄여 고품질 음성 변환이 가능한가를 실험적으로 분석함
- VoiceGrad에서는 $x_0^{src}$ (소스 mel-spectrogram) 를 초기값으로 사용함
  - 하지만 diffusion 모델은 학습 시 정규분포에서 시작된 샘플을 다루므로
  - 이렇게 깨끗한 데이터로 시작하면 분포 차이(gap) 가 발생
 
#### 기존 방식의 한계
- $x \sim \mathcal{N}(0, I)$
  - 순수 노이즈에서 시작
  - 학습과 추론 분포 일치 (장점)
  - 소스 정보가 완전히 사라짐 → 내용 왜곡 (단점)
- $x = x_0^{src}$
  - 깨끗한 소스 그대로 시작
  - 소스 내용 완벽 유지 (장점)
  - 학습 시에는 노이즈에서 시작했으므로 분포 불일치 (단점)

#### 제안 : 중간 노이즈 상태로 시작
- 소스 정보를 일부 보존하면서도, 학습 분포와 유사한 초기값을 생성
  - $`x^{src}{S_K} = \sqrt{\bar{\alpha}{S_K}} \cdot x^{src}0 + \sqrt{1 - \bar{\alpha}{S_K}} \cdot \epsilon \tag{9}`$
  - 이 식은 $x_0^{src}$에 노이즈를 일정 단계($S_K$)만큼 섞은 상태를 의미
  - 추론 시에는 이 $x^{src}_{S_K}$를 초기값 $x$로 사용

#### 결론
<img width="426" alt="image" src="https://github.com/user-attachments/assets/c4b98a10-4a36-4de1-9187-8fb64c35df29" />

- 실험결과를 바탕으로 논문에서는 $S_K = 950$ 일 때 음질과 화자 유사도가 가장 균형 잡힘을 확인
- 따라서 이후 모든 실험 설정값
  - 초기값 $x = x^{src}_{950}$
  - 초기 step $t = S_K = 950$

### 2) Adversarial conditional diffusion distillation
#### 배경 및 개념























