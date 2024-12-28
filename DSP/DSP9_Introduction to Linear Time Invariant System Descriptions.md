
# Signal Processing: System Overview
<img width="789" alt="image" src="https://github.com/user-attachments/assets/61323e98-bcd3-4ed6-bbcb-a2a18998628f" />

## 1. System의 역할
- 시스템 \(H\)는 입력 신호 \(x(t)\)를 출력 신호 \(y(t)\)로 변환
- 입력과 출력 신호는 시간 연속형 (\(x(t), y(t)\)) 또는 이산형 (\(x[n], y[n]\))일 수 있습니다


## 2. 시스템의 주요 목적
### 1) 물리적 현상을 모델링 (Model Physical Phenomenon)
- **예시: 무선 통신에서의 다중 경로 전파(Multipath Propagation)**  
  - 신호는 물리적 장애물(빌딩 등)로 인해 반사 신호가 섞임
  - 수학적으로 다음과 같이 표현됩니다: $\[y(t) = x(t) + \alpha x(t-\tau)\]$
    - $\(\alpha\)$ : 감쇠 계수 $(\(|\alpha| < 1\))$
    - $\(\tau\)$ : 시간 지연(Time Delay)

### 2) 원하는 효과 구현 (Implement Desired Effect)
- 시스템은 특정 입력 신호를 원하는 출력 신호로 변환하는 데 사용
- **예시: 신호 복원 과정**
  - $\[z(t) = y(t) - \alpha y(t-\tau) + \alpha^2 y(t-2\tau) - \alpha^3 y(t-3\tau)\]$
  - 위 과정을 통해 \(z(t)\)는 원 신호 \(x(t)\)에 가까워짐


## 3. 주요 응용
- **다중 경로 모델링** : 실제 세계의 물리적 신호 왜곡 현상을 수학적으로 모델링
- **신호 복원** : 입력 신호에서 원하는 정보를 추출하거나 원래 신호를 복원

### 수식 설명
1. **모델링된 물리적 신호** : $\[y(t) = x(t) + \alpha x(t-\tau)\]$

2. **신호 복원 과정** : $\[z(t) = x(t) \approx y(t) - \alpha y(t-\tau) + \alpha^2 y(t-2\tau) - \alpha^3 y(t-3\tau)\]$



# Signal Processing: Linear and Time-Invariant Systems

<img width="741" alt="image" src="https://github.com/user-attachments/assets/c4e40b02-268e-4cf2-bb11-0a5643fa987c" />
## 1. Linear System
- **선형 시스템의 정의**: 입력 신호의 합은 출력 신호의 합으로 나타남
- **Superposition (중첩 원리)**
  - $\( x_1[n] \xrightarrow{H} y_1[n] \)$
  - $\( x_2[n] \xrightarrow{H} y_2[n] \)$
  - $\( a \cdot x_1[n] + b \cdot x_2[n] \xrightarrow{H} a \cdot y_1[n] + b \cdot y_2[n] \)$
- 이는 시스템이 입력 신호의 선형 결합에 대해 출력 신호의 선형 결합으로 응답함


## 2. Time-Invariant System
- **시간 불변 시스템의 정의**: 시스템의 응답이 시간에 따라 변하지 않음
- **시간 불변성 (Time Invariance)**:
  - 입력 신호가 시간 지연을 받을 경우, 출력 신호도 동일한 시간 지연을 받음
  - $\( x[n] \xrightarrow{H} y[n] \)$
  - $\( x[n - n_0] \xrightarrow{H} y[n - n_0] \)$
- 이는 시스템이 입력 신호의 시간적 이동에 대해 동일하게 반응함


# Signal Processing: Causal and Non-Causal Systems

<img width="753" alt="image" src="https://github.com/user-attachments/assets/8cfa8c87-dda0-4c7a-b8c5-05032c39cd40" />

## 1. Causal System
- **정의**: 출력은 과거 또는 현재의 입력 신호에만 의존
- **특징**
  - 미래의 입력 값에 의존하지 않음
  - 실시간 시스템에서 구현 가능

- **예시** : $\[y[n] = \frac{1}{2} (x[n] + x[n-1])\]$
  - 현재 입력 \(x[n]\)과 과거 입력 \(x[n-1]\)에만 의존
  - 이는 **Causal** 시스템

## 2. Non-Causal System
- **정의**: 현재 출력이 미래의 입력 신호에 의존
- **특징**:
  - 실시간으로 구현 불가능
  - 주로 저장된 데이터나 사전 정보가 필요

- **예시**: $\[y[n] = \frac{1}{2} (x[n+1] + x[n])\]$
  - 미래의 입력 \(x[n+1]\)에 의존
  - 이는 **Non-Causal** 시스템


# Four LTI System Descriptions
<img width="763" alt="image" src="https://github.com/user-attachments/assets/5c273019-c6c9-4051-837a-960f2c289d36" />

## 1. Overview  
LTI (Linear Time-Invariant) 시스템은 네 가지 주요 방식으로 설명될 수 있습니다  
1. Difference Equation  
2. Impulse Response  
3. Frequency Response  
4. System Function (Poles/Zeros)  

## 2. Description Comparison Table

| **Description**         | **Computation**                          | **Intuition**                         |
|--------------------------|------------------------------------------|---------------------------------------|
| **Difference Equation**  | ⭐⭐⭐⭐⭐                                   | ⭐                                     |
| **Impulse Response**     | ⭐⭐⭐                                      | ⭐⭐⭐⭐                                  |
| **Frequency Response**   | ⭐                                        | ⭐⭐⭐⭐                                  |
| **System Function**      | ⭐                                        | ⭐⭐⭐⭐                                  |


## 3. Explanation
### 1. Difference Equation
- **Computation** : 매우 효율적이며, 시스템의 연산을 직접적으로 정의
- **Intuition** : 수식 기반으로 직관적인 이해가 어려움
- **예시 수식** : $\[y[n] = -\sum_{k=1}^{N} a_k y[n-k] + \sum_{q=0}^{M} b_q x[n-q]\]$

### 2. Impulse Response
- **Computation**: 계산 효율성 중간
- **Intuition**: 시스템의 동작을 직관적으로 이해하기 쉬움
- **시스템의 응답**: $\[\delta[n] \xrightarrow{H} h[n]\]$

### 3. Frequency Response
- **Computation**: 계산 효율성 낮음
- **Intuition**: 주파수 영역에서의 동작을 이해하기 쉬움
- **시스템의 응답**: $\[e^{j\omega n} \xrightarrow{H} H(e^{j\omega}) e^{j\omega n}\]$

### 4. System Function (Poles/Zeros)
- **Computation**: 계산 효율성 낮음
- **Intuition**: 시스템의 전체 동작을 이해하는 데 가장 직관적
- **시스템의 응답**: $\[z^n \xrightarrow{H} H(z) z^n\]$


## 4. 요약
- **Difference Equation**: 계산적으로 강력하지만 직관적이지 않음
- **Impulse Response**: 중간 수준의 계산 효율성과 높은 직관성 제공
- **Frequency Response**: 주파수 영역에서 시스템 동작을 이해하기 매우 직관적
- **System Function**: 전체 시스템의 특성을 이해하기에 적합
