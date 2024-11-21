- What is a signal?
- what is signal processing?
- Philosophy(철학) behind signal processing
- Language of signal processing

## 1. Signals
- **정의**
  - 시간 또는 공간에 따라 변화하는 물리적 양
  - 시간, 공간 혹은 다른 독립 변수에 따라 변화하는 물리적 양을 의미하며, 정보를 전달하거나 표현하는데 사용됨
  - ex) 소리(음파), 빛(전자기파), 온도변화 등
  - A 'signal' describes how some physical quantity varies over time and/or space
- **Examples of Signals**
  - Sound pressure - 소리 압력(음악, 사람의 음성..)
    - 공기 중의 음파로, 시간에 따라 압력이 변함 
  - Radio or television broadcast - FM 라디오, TV 신호
    - 전자기파 형태로 전송되는 신호 
  - Movie
    - 연속된 프레임(이미지)과 오디오 신호로 구성 
  - Electrocardiogram - 심전도(ECG 그래프)
    - 심장의 전기적 활동을 기록한 신호 
  - Sunspot count - 태양 흑점 수(과거의 태양 활동 데이터)
    - 태양 표면의 흑점 개수를 시간에 따라 기록한 신호  
  - Accelerator position - 가속 페달 위치(자동차의 속도 제어 데이터)
    - 차량 가속 페달의 위치를 시간에 따라 측정한 신호 
- **유형**
  - **연속 신호** : 시간이나 공간이 연속적인 값으로 표현됨 (ex. 아날로그 신호)
    - 음성 신호, 온도 변화, 전압 신호, 빛의 강도, 심전도, 자연 소리, 진동 신호 등  
  - **이산 신호** : 시간이나 공간이 이산적인 값으로 표현됨 (ex. 디지털 신호)
    - 디지털 오디오, 디지털 비디오, 컴퓨터 데이터, 디지털 통신신호, 디지털 센서 데이터 등 
- 연속 신호와 이산 신호의 차이  

| 구분             | 연속 신호 (Continuous Signal)       | 이산 신호 (Discrete Signal)        |
|------------------|-------------------------------------|-------------------------------------|
| **시간 축**      | 연속적                              | 이산적                              |
| **값**          | 연속적                              | 이산적 또는 연속적                  |
| **예시**        | 아날로그 오디오, 심전도             | 디지털 오디오, 주식 데이터          |
| **정보 표현**    | 무한히 많은 정보 표현 가능          | 유한한 정보만 표현 가능             |
| **변환**        | 샘플링 후 디지털 신호로 변환 가능    | 적절한 보간으로 연속 신호로 복원 가능 |

- 아날로그 신호와 디지털 신호의 차이  

| 특징              | 아날로그 신호                      | 디지털 신호                         |
|-------------------|------------------------------------|-------------------------------------|
| **값의 표현**     | 연속적                             | 이산적                              |
| **시간의 표현**   | 연속적                             | 이산적                              |
| **노이즈 민감성** | 노이즈에 민감함                   | 노이즈에 강함                       |
| **저장 및 처리**  | 아날로그 장치를 사용              | 컴퓨터 및 디지털 장치를 사용        |
| **전송 효율성**   | 대역폭 요구량이 큼                | 데이터 압축 가능, 효율적 전송       |

## 2. Signal Processing
- **정의**
  - 신호의 분석, 변환, 조작을 통해 유용한 정보를 추출하거나 신호의 품질을 개선하는 기술
  - 신호를 조작하여 특성을 변경하거나 정보를 추출함
  - 측정된 신호의 노이즈를 줄이고, 왜곡을 수정하며, 신호에서 일부 정보를 추출하는것으로 주로 사용됨
  - Manipulating a signal to change its characteristics or extract information
- **Performed by**
  - **Computer**
    - 디지털 신호 처리를 위한 가장 일반적인 도구
    - 유연성과 정확성이 높으며, 복잡한 연산 가능(Fourier Transform, 필터링 등)
    - ex. 음성 인식 시스템, 이미지 처리, 머신러닝 기반 신호 분석 
  - **Special purpose integrated circuits** - 특수 목적 집적 회로
    - 특정한 신호 처리 작업을 수행하도록 설계된 하드웨어 칩
    - 고속 처리와 에너지 효율이 뛰어나며, 실시간 신호 처리에 적합
    - 휴대기기, 의료기기, 통신 시스템 등에서 주로 사용
    - ex. DSP 칩, FPGA, GPU 
  - **Analog electrical circuits** - 아날로그 전기 회로
    - 전기 신호의 아날로그 변환을 직접 수행하는 하드웨어 회로
    - 아날로그 신호를 실시간으로 처리 가능
    - 디지털 변환이 필요 없는 간단한 필터링 작업에 적합
    - ex. 라디오 주파수 필터, 아날로그 증폭기

### 1) 응용 분야
- **소비자 전자기기 (Consumer Electronics)**
  - ex. HDTV, 휴대폰, 카메라 등
  - 영상 및 음향 품질을 개선 + 데이터 압축 및 복원 기능 제공

- **교통 (Transportation)**
  - ex. GPS, 엔진 제어, 항공기 추적 등
  - 실시간 위치 추적 가능 + 차량 시스템 최적화 및 비행 경로 모니터링 지원

- **의료 (Medical)**
  - ex. 영상 처리, 모니터링(EEG, ECG) 등
  - MRI, CT 스캔, 심전도와 같은 의료 데이터 분석 및 노이즈 제거

- **군사 (Military)**
  - ex. 목표 추적, 감시 등
  - 적의 위치 추적 + 무기 시스템 및 감시 시스템 최적화

### 2) 일반적인 신호처리 문제
- **잡음 제거(Eliminating Noise)**
  - 신호 데이터에는 환경적, 기기적 요인으로 인해 불필요한 잡음(Noisy Data)이 포함될 수 있음
  - 신호 처리를 통해 잡음을 제거하여 유용한 데이터를 복원
  - ex. 심전도(ECG) 신호에서 잡음을 제거하여 깨끗한 신호(Clean ECG)로 변환
![image](https://github.com/user-attachments/assets/22b447a0-6b0c-40e7-becb-dd5a37cb5e96)

- **왜곡 보정 (Correcting Distortion)**
  - 신호 또는 데이터는 환경적, 기기적 한계로 인해 왜곡(Distortion) 발생
  - 신호 처리 기술을 사용하여 왜곡된 데이터를 보정하고 원래의 신호를 복원
  - ex. 위성 이미지나 사진에서 발생하는 흐림(Blur) 현상을 수정하여 선명한 이미지로 복원 

- **측정 신호에서 간접적인 양 추출 (Extracting an Indirect Quantity from Measured Signals)**
  - 직접 측정이 어려운 데이터를 간접적으로 계산하거나 추정
  - 신호 처리 기법을 통해 신호를 분석하고 필요한 정보를 도출
  - ex. 항공기의 위치와 속도 추정
    - 레이더 반사 신호 측정: 항공기에서 반사된 레이더 신호를 수집
    - 위치 및 속도 추정: 레이더 데이터를 기반으로 항공기의 정확한 위치와 속도를 계산   

## 3. Philosophy behind signal processing (신호 처리의 철학)
- 신호 처리는 모델을 기반으로 신호와 데이터를 이해하고 분석
  - **신호와 잡음의 특성화 (Characterize "signal" and "noise")**
  - **왜곡 설명 (Describe distortion)**
    - 수식 : `y[n] = H{x[n]} + w[n]`
    - `H{x[n]}`: 신호에 적용된 시스템의 영향
    - `w[n]`: 잡음
  - **측정 데이터와 원하는 양의 관계 정의 (Relate desired quantity to measured data)**
- 신호 처리 모델은 기존의 지식을 기반으로 생성됨

### 1) Modeling Issues
- **Poor Model → Poor Performance:**
  - 잘못된 모델은 신호 처리 시스템의 성능 저하로 이어짐
  - 신호를 잘 설명하지 못하는 모델은 결과의 정확성과 신뢰도를 낮춤
- **모델 복잡성과 성능 (Model Complexity vs Performance)**
  - 복잡한 모델 vs 단순한 모델 (Detailed vs Simple Models)
  - 복잡한 모델
    - ex. `y[n] = a₁y[n-1] + a₂y[n-2] + ... + b₁w[n-1] + b₂₀w[n-20]`
    - 긴 수식과 다수의 매개변수를 포함하는 모델
    - 정확도는 잘 나올 수 있지만, 높은 계산 비용(Computation Cost) 요구
  - 단순한 모델
    - ex. `y[n] = a₁y[n-1] + w[n]` 
    - 짧고 간단한 수식을 사용하는 모델
    - 계산 비용이 적지만, 정확도가 낮을 가능성이 있음
- 모델 선택 시 계산 비용과 정확성 간의 트레이드오프를 고려해야 함

## 4. Language of signal processing (신호 처리의 언어)
- **수학 (Mathematics)**
  - 신호 처리를 이해하고 구현하는 데 필수적인 도구
  - 미적분(Calculus)
    - Fourier Transform : $X(\omega) = \int x(t)e^{-j\omega t}\dt$
  - 선형대수(Linear Algebra)
    - $\hat{x}[n] = (H^{T}H)^{-1}H^{T}y[n]$
    - 행렬 연산과 내적(Dot product)을 사용하여 신호 분석 
 
- **확률과 통계 (Probability and Statistics)**
  - 신호의 잡음과 불확실성을 모델링하고 분석
  - 잡음 및 불확실성 모델링(Model Noise and Uncertainty)
    - 신호에 포함된 잡음을 분석하기 위한 확률 분포 모델
    - 가우시안 분포 : $f(w) = \dfrac{1}{\sqrt{2\pi\sigma^{2}}} e^{ -\dfrac{(w - m)^{2}}{2\sigma^{2}} }$
  - 결과의 신뢰도 특성화(Characterize Confidence of Results)
    - 신호 처리 결과에 대한 신뢰 구간과 확률적 특성 평가 
