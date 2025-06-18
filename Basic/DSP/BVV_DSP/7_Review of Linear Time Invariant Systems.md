## 1. LTI Systems 개요
![image](https://github.com/user-attachments/assets/e1c73a3f-5d6f-41e5-9739-f51dcf45af4f)

- LTI 시스템은 입력 신호를 선형적이고 시불변적인 방식으로 처리하여 출력 신호를 생성하는 시스템
- 입력 신호를 출력 신호로 매핑하는 시스템
  - 입력 : x(t)(연속 시간 신호) 또는 x[n](이산 시간 신호)
  - 출력 : y(t)(연속 시간 출력) 또는 y[n](이산 시간 출력)
  - 시스템 : H, 입력 신호를 변환하는 연산자
- 주요 특징
  - **선형성 (Linearity)**
   - 입력 신호의 선형 결합이 출력에서도 동일하게 반영됨
   - $\[H\{a \cdot x_1(t) + b \cdot x_2(t)\} = a \cdot H\{x_1(t)\} + b \cdot H\{x_2(t)\}\]$

  - **시불변성 (Time-Invariance)**
   - 시스템의 특성이 시간에 따라 변하지 않음
   - $\[H\{x(t - t_0)\} = y(t - t_0)\]$

## 2. System: Modeling and Implementation (시스템: 모델링과 구현)

![image](https://github.com/user-attachments/assets/dfdc4d07-cbbd-4b19-b25e-ae6d69e82b60)

### 1) 시스템의 역할
1. **Model a Physical Phenomenon (물리적 현상의 모델링)**  
  - 시스템은 실제 물리적 현상을 수학적으로 표현합니다
  - 예: 신호의 전송 또는 반사

2. **Implement Desired Characteristics (필요한 특성 구현)**  
  - 특정 목적에 맞는 시스템 동작을 설계합니다
  - 예: 필터 설계, 신호 증폭

### 2) 수학적 표현
1. **시스템 입력과 출력 관계**
  - 시스템은 입력 x(t)와 과거 상태를 사용하여 출력을 생성
  - $\[y(t) = x(t) + b x(t - \tau)\]$

2. **반복적 출력 변환**
  - z(t)를 y(t)의 변환으로 정의
  - $\[z(t) = y(t) - b y(t - \tau) + b^2 y(t - 2\tau) - b^3 y(t - 3\tau) + \dots\]$

3. **입력 신호의 확장 표현**
  - z(t)를 x(t)로 확장
  - $\[z(t) = x(t) + b x(t - \tau) - b x(t - \tau) - b^2 x(t - 2\tau) + b^2 x(t - 2\tau) + b^3 x(t - 3\tau) - \dots\]$

### 3) 시스템 블록 다이어그램
1. **입력과 출력 관계**
   - 입력 x(t)가 시스템 H를 통해 변환되어 출력 z(t)를 생성
2. **과거 시간 상태 사용**
   - 과거 시간 신호 $\( x(t - \tau), x(t - 2\tau), \dots \)$을 사용하여 출력 생성

## 3. LTI Systems (선형 시불변 시스템)

![image](https://github.com/user-attachments/assets/412af773-f5e0-4c7d-95aa-7b71a308ab43)

### 1) Linear System (선형 시스템)
- **중첩 원리 (Superposition Principle)**
  - 입력 신호의 합이 출력 신호의 합으로 반영됩니다
  - 수식: $\[H\{a x_1[n] + b x_2[n]\} = a H\{x_1[n]\} + b H\{x_2[n]\}\]$
  - 입력과 출력의 관계
    - $\( x_1[n] \to H \to y_1[n] \)$
    - $\( x_2[n] \to H \to y_2[n] \)$
    - 결과적으로 $\( a x_1[n] + b x_2[n] \to H \to a y_1[n] + b y_2[n] \)$

### 2) Time Invariance (시불변성)
- **시간 변환 불변성**
  - 시스템은 입력 신호의 시간 이동에 동일한 방식으로 반응합니다
  - 수식: $\[x[n - n_0] \to H \to y[n - n_0]\]$
  - 입력 신호 x[n]가 $\( n_0 \)$만큼 이동하면, 출력 신호도 동일하게 $\( n_0 \)$만큼 이동

### 3) LTI 시스템의 특징
- **LTI 시스템은 선형성과 시불변성을 모두 만족**합니다
  1. 입력 신호의 선형 결합에 대해 동일한 선형 결합 출력 생성
  2. 시간 이동에 대해 동일한 시스템 응답 유지


## 4. I/O for LTI Systems (LTI 시스템의 입출력 관계)

![image](https://github.com/user-attachments/assets/2391b2bf-3164-4639-8fe3-02e00c612cc1)

### 1) Impulse Response (임펄스 응답)
- **정의**:
  - 임펄스 입력 $\( \delta[n] \)$를 넣었을 때의 출력 h[n]로, 시스템의 모든 동작 특성을 나타냅니다

### 2) 입력과 출력 관계
- **컨볼루션 (Convolution)**
  - 입력 x[n]와 임펄스 응답 h[n]의 컨볼루션으로 출력 y[n]이 계산됩니다
  - 수식: $\[y[n] = x[n] * h[n]\]$
  - $\[y[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k]\]$
  - $\[y[n] = \sum_{k=-\infty}^{\infty} h[k] x[n-k]\]$

### 3) 시스템 속성
1. **Causal (인과성)**
  - $\( h[n] = 0 \) for \( n < 0 \)$
  - 시스템은 과거와 현재의 입력에만 반응하며, 미래의 입력에는 영향을 받지 않습니다

2. **Stable (안정성)**
  - 임펄스 응답의 절대값 합이 유한해야 합니다: $\[\sum_{n=-\infty}^{\infty} |h[n]| < \infty\]$

### 4) 블록 다이어그램 설명
1. **임펄스 입력 $(\( \delta[n] \))$**
   - $\( \delta[n] \)$ 입력에 대한 출력 h[n]을 통해 시스템의 동작 특성을 분석
2. **컨볼루션 계산**
   - 임의의 입력 x[n]에 대해 h[n]과의 컨볼루션을 통해 출력 y[n] 생성

## 5. Difference Equations (차분 방정식)

![image](https://github.com/user-attachments/assets/af906bf7-0cec-4045-acad-5b35860edd6d)

### 1) 정의
- 차분 방정식은 LTI 시스템의 중요한 클래스 중 하나
- 시스템의 입력 x[n]과 출력 y[n]간의 관계를 수학적으로 표현
- 수식: $\[\sum_{k=0}^N a_k y[n-k] = \sum_{k=0}^M b_k x[n-k]\]$
  - $\( a_k \)$ : 출력 신호 계수
  - $\( b_k \)$ : 입력 신호 계수
  - $\( N, M \)$ : 과거 출력 및 입력 샘플의 최대 차수

### 2) 역할
1. **모델링 (Model Physical Systems)**
   - 물리적 시스템의 동작을 수학적으로 표현
2. **필터 설계 (Design Filters)**
   - 디지털 필터의 설계에 사용
3. **필터 구현 (Implement Filters)**
   - 필터를 계산 및 구현

### 3) 예제

#### 차분 방정식
- $\( y[n] - \frac{1}{2}y[n-1] = x[n] \)$

#### 임펄스 응답 계산
- 입력 $\( x[n] = \delta[n] \)$ (단위 임펄스 입력)
- 초기 조건 : $\( y[n] = 0 \)$ for $\( n < 0 \)$
- 계산 과정 : $\[y[0] = \frac{1}{2}y[-1] + x[0] = 0 + 1 = 1\]$ 
  - $\[y[1] = \frac{1}{2}y[0] + x[1] = \frac{1}{2} \cdot 1 + 0 = \frac{1}{2}\]$
  - $\[y[2] = \frac{1}{2}y[1] + x[2] = \frac{1}{2} \cdot \frac{1}{2} + 0 = \frac{1}{4}\]$
  - 결과 : $\( y[n] = \{1, \frac{1}{2}, \frac{1}{4}, \dots\} \)$

### 4) 주요 특징
- 차분 방정식은 LTI 시스템의 동작을 간단히 설명하며, 필터 설계 및 구현에서 핵심적 역할을 합니다
- 과거 입력 및 출력 값을 사용하여 현재 출력을 계산합니다
