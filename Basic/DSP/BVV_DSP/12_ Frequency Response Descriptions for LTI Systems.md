# 주파수 응답 (Frequency Response)

<img width="642" alt="image" src="https://github.com/user-attachments/assets/b9596cba-51d3-4b09-a1b9-4ee49ba05d47" />

## 1. 개념
- 주파수 응답은 시스템이 다양한 주파수를 가지는 사인파에 어떻게 반응하는지 나타냅니다
- 입력 신호가 복소 지수 함수 $\(x[n] = e^{j\omega n}\)$일 때, 시스템의 출력은:$\[y[n] = H(e^{j\omega}) e^{j\omega n}\]$

## 2. 주파수 응답의 정의
1. **주파수 응답**:$\[H(e^{j\omega}) = |H(e^{j\omega})| e^{j\angle H(e^{j\omega})}\]$
   - **크기 응답(Magnitude Response)**: $\( |H(e^{j\omega})| \)$
   - **위상 응답(Phase Response)**: $\( \angle H(e^{j\omega}) \)$

2. **출력 신호**:$\[y[n] = |H(e^{j\omega})| e^{j(\omega n + \angle H(e^{j\omega}))}\]$
   - 입력 신호의 **진폭**은 $\( |H(e^{j\omega})| \)$에 의해 변경됩니다.
   - 입력 신호의 **위상**은 $\( \angle H(e^{j\omega}) \)$에 의해 변경됩니다.

## 3. 선형성(Linearity)와 복합 입력
- 입력 신호가 두 개 이상의 주파수를 포함하는 경우: $\[x[n] = \alpha_1 e^{j\omega_1 n} + \alpha_2 e^{j\omega_2 n}\]$
- 출력은 각 성분의 응답을 더한 값: $\[y[n] = \alpha_1 H(e^{j\omega_1}) e^{j\omega_1 n} + \alpha_2 H(e^{j\omega_2}) e^{j\omega_2 n}\]$


# 임펄스 응답과 주파수 응답의 관계

<img width="628" alt="image" src="https://github.com/user-attachments/assets/baa82b91-31d4-4d4f-b77d-bf4ca63a171f" />

## 1. 개요
- 임펄스 응답 \(h[n]\)과 주파수 응답 $\(H(e^{j\omega})\)$ 사이의 관계는 **이산 시간 푸리에 변환(DTFT)**를 통해 설명됩니다:
- $\[H(e^{j\omega}) = \sum_{k=-\infty}^{\infty} h[k] e^{-j\omega k}\]$
- 이는 임펄스 응답을 주파수 영역으로 변환하는 수학적 도구입니다.

## 2. 수학적 관계
1. **출력 신호와 임펄스 응답**:$\[y[n] = \sum_{k=-\infty}^{\infty} h[k] x[n-k]\]$

2. **주파수 응답**
   - DTFT를 적용하면:$\[H(e^{j\omega}) = \sum_{k=-\infty}^{\infty} h[k] e^{-j\omega k}\]$

## 3. 예제

### (1) 예제 1
- **시스템 1**:$\[y_1[n] = \frac{1}{2}x[n] + \frac{1}{2}x[n-1]\]$
  - 임펄스 응답: $\[ h_1[n] = \begin{cases} \frac{1}{2}, & n = 0, 1 \\ 0, & \text{otherwise} \end{cases} \]$
  - 주파수 응답: $\[H_1(e^{j\omega}) = \frac{1}{2} + \frac{1}{2}e^{-j\omega} = e^{-j\omega/2} \cos(\omega/2)\]$

### (2) 예제 2
- **시스템 2**: $\[y_2[n] = \frac{1}{2}x[n] - \frac{1}{2}x[n-1]\]$
  - 임펄스 응답: $\[ h_2[n] = \begin{cases} \frac{1}{2}, & n = 0 \\ -\frac{1}{2}, & n = 1 \\ 0, & \text{otherwise} \end{cases} \]$
  - 주파수 응답: $\[H_2(e^{j\omega}) = \frac{1}{2} - \frac{1}{2}e^{-j\omega} = je^{-j\omega/2} \sin(\omega/2)\]$

## 4. 주파수 응답의 크기 그래프
- **시스템 1** $(\(|H_1(e^{j\omega})|\))$
  - 주파수 $\(\omega\)$에 따라 코사인 형태로 변동
- **시스템 2** $(\(|H_2(e^{j\omega})|\))$
  - 주파수 $\(\omega\)$에 따라 사인 형태로 변동


# 합성곱-곱셈 성질과 이상적인 필터 (Convolution-Multiplication Property & Ideal Filters)

<img width="625" alt="image" src="https://github.com/user-attachments/assets/2897f7ab-4edc-4559-8b71-92567d04da4c" />

## 1. 합성곱-곱셈 성질
- 시간 영역에서의 합성곱은 주파수 영역에서 곱셈으로 변환됩니다.
- 수식 관계: $\[y[n] = h[n] * x[n] \quad \xrightarrow{\text{DTFT}} \quad Y(e^{j\omega}) = H(e^{j\omega}) X(e^{j\omega})\]$
  - \(h[n]\): 임펄스 응답
  - \(x[n]\): 입력 신호
  - \(y[n]\): 출력 신호
  - $\(H(e^{j\omega})\)$, $\(X(e^{j\omega})\)$, $\(Y(e^{j\omega})\)$: 주파수 영역 표현

## 2. 이상적인 필터 (Ideal Filters)
이상적인 필터는 주파수 영역에서 특정 주파수 대역만 통과시키고 나머지 대역은 제거합니다

### (1) 저역통과 필터 (Low-Pass Filter)
- $\(H(e^{j\omega})\)$: $\[ H(e^{j\omega}) = \begin{cases} 1, & |\omega| \leq \omega_c \\ 0, & |\omega| > \omega_c \end{cases}\]$
- 특징: 저주파 성분을 통과시키고 고주파 성분을 제거

### (2) 고역통과 필터 (High-Pass Filter)
- $\(H(e^{j\omega})\)$: $\[ H(e^{j\omega}) = \begin{cases} 0, & |\omega| \leq \omega_c \\ 1, & |\omega| > \omega_c \end{cases} \]$
- 특징: 고주파 성분을 통과시키고 저주파 성분을 제거

### (3) 대역통과 필터 (Band-Pass Filter)
- $\(H(e^{j\omega})\)$: $\[ H(e^{j\omega}) = \begin{cases} 1, & \omega_1 \leq |\omega| \leq \omega_2 \\ 0, & \text{otherwise} \end{cases} \]$
- 특징: 특정 주파수 대역($\([\omega_1, \omega_2]\)$)만 통과

### (4) 대역저지 필터 (Band-Stop Filter)
- $\(H(e^{j\omega})\)$: $\[ H(e^{j\omega}) = \begin{cases} 0, & \omega_1 \leq |\omega| \leq \omega_2 \\ 1, & \text{otherwise} \end{cases}\]$
- 특징: 특정 주파수 대역($\([\omega_1, \omega_2]\)$)을 제거하고 나머지는 통과

## 3. 요약
- 시간 영역의 합성곱은 주파수 영역에서 곱셈으로 표현됩니다
- 이상적인 필터는 주파수 응답을 통해 특정 주파수 대역만 통과하거나 제거합니다
  - 저역통과, 고역통과, 대역통과, 대역저지 필터
- 실제 응용에서는 이상적인 필터 대신 근사화된 필터가 사용됩니다

# 주파수 응답과 차분 방정식의 관계

<img width="631" alt="image" src="https://github.com/user-attachments/assets/646c51cc-5a33-4c8d-82ea-42378cea8002" />

## 1. 차분 방정식 (Difference Equation)
- 주어진 차분 방정식:$\[\sum_{k=0}^{N} a_k y[n-k] = \sum_{k=0}^{M} b_k x[n-k], \quad a_0 = 1\]$
  - $\(a_k\)$: 출력 신호 \(y[n]\)의 계수
  - $\(b_k\)$: 입력 신호 \(x[n]\)의 계수

## 2. 입력이 복소 지수 함수일 때 ($\(x[n] = e^{j\omega n}\)$)
1. 출력 신호: $\[y[n] = H(e^{j\omega}) e^{j\omega n}\]$
  - 여기서 $\(H(e^{j\omega})\)$는 시스템의 주파수 응답.

2. 차분 방정식에 대입:$\[\sum_{k=0}^{N} a_k H(e^{j\omega}) e^{j\omega(n-k)} = \sum_{k=0}^{M} b_k e^{j\omega(n-k)}\]$

3. 단순화하면:$\[\sum_{k=0}^{N} a_k H(e^{j\omega}) e^{-j\omega k} = \sum_{k=0}^{M} b_k e^{-j\omega k}\]$

4. 주파수 응답:$\[H(e^{j\omega}) = \frac{\sum_{k=0}^{M} b_k e^{-j\omega k}}{\sum_{k=0}^{N} a_k e^{-j\omega k}}\]$

## 3. 주파수 응답의 의미
- $\(H(e^{j\omega})\)$는 **두 다항식의 비율**로 표현됩니다:$\[H(e^{j\omega}) = \frac{\text{입력 계수의 다항식}}{\text{출력 계수의 다항식}}\]$
  - 분자: 입력 신호 계수의 합
  - 분모: 출력 신호 계수의 합

# 예제: 6 포인트 평균 필터 및 차분 필터의 주파수 응답

<img width="618" alt="image" src="https://github.com/user-attachments/assets/b740b5f1-f096-4b0f-bc5e-586b6a661736" />

## 1. 6 포인트 평균 필터 (6-Point Averaging Filter)
- **수식**:$\[y[n] = \frac{1}{6} \{ x[n] + x[n-1] + x[n-2] + x[n-3] + x[n-4] + x[n-5] \}\]$
- **특징**:
  - 신호의 저주파 성분을 강조하여 부드럽게 만듦
  - 잡음 제거와 신호 평활화에 유용
- **주파수 응답**:
  - **크기 응답 (Magnitude Response)**: 저주파 영역에서 큰 값을 가지며, 고주파 영역에서는 점차 감소
  - **위상 응답 (Phase Response)**: 선형적인 위상 변화

## 2. 6 포인트 차분 필터 (6-Point Differencing Filter)
- **수식**:$\[y[n] = \frac{1}{6} \{ x[n] - x[n-1] + x[n-2] - x[n-3] + x[n-4] - x[n-5] \}\]$
- **특징**:
  - 신호의 고주파 성분을 강조하여 변화나 경계를 탐지
  - 잡음 증폭 가능성 있음
- **주파수 응답**
  - **크기 응답 (Magnitude Response)**: 고주파 영역에서 큰 값을 가지며, 저주파 영역에서는 점차 감소
  - **위상 응답 (Phase Response)**: 선형적인 위상 변화

## 3. 주파수 응답 그래프
### 6 포인트 평균 필터
- **크기 응답**
  - 주파수 $\(\omega\)$가 작을수록 높은 이득을 보임
- **위상 응답**
  - 위상이 선형적으로 감소하며, 저주파에서 고주파로 갈수록 점차 변화

### 6 포인트 차분 필터
- **크기 응답**
  - 고주파에서 높은 이득을 보이며, 저주파에서는 감소
- **위상 응답**
  - 선형적으로 변화하며, 고주파에서 큰 위상 이동이 나타남


# 예제: 재귀 필터의 주파수 응답 (Frequency Response for Recursive Filters)

<img width="630" alt="image" src="https://github.com/user-attachments/assets/d833e749-39dd-483c-a030-795c22b501e1" />

## 1. 필터 정의

### (1) 재귀 저역통과 필터 (Recursive Low-Pass Filter)
- **수식**:$\[y[n] = 0.95y[n-1] + 0.05x[n]\]$

- **특징**:
  - 저주파 성분을 강조하여 고주파 잡음을 억제
  - 신호를 부드럽게 만듦

### (2) 재귀 고역통과 필터 (Recursive High-Pass Filter)
- **수식**:$\[y[n] = -0.95y[n-1] + 0.05x[n]\]$

- **특징**:
  - 고주파 성분을 강조하고 저주파 성분을 제거
  - 신호의 경계와 빠른 변화를 감지

## 2. 주파수 응답 그래프

### (1) 재귀 저역통과 필터
- **크기 응답 (Magnitude Response)**
  - 저주파 성분에서 높은 이득을 가지며, 고주파로 갈수록 감소
- **위상 응답 (Phase Response)**
  - 주파수에 따라 선형적으로 증가하며, 저주파에서 위상 변화가 작음

### (2) 재귀 고역통과 필터
- **크기 응답 (Magnitude Response)**
  - 고주파 성분에서 높은 이득을 가지며, 저주파로 갈수록 감소
- **위상 응답 (Phase Response)**
  - 주파수에 따라 선형적으로 증가하며, 고주파에서 위상 변화가 큼


# 주파수 응답과 연속 시간 시스템 (Frequency Response and Continuous-Time Systems)

<img width="617" alt="image" src="https://github.com/user-attachments/assets/4cd1affc-e434-430d-9388-0691e6eadf69" />

## 1. 개념
연속 시간 시스템에서 주파수 응답은 입력 신호가 사인파 또는 복소 지수 신호일 때 시스템의 동작을 설명합니다.

- 입력 신호:$\[x(t) = e^{j\Omega t}\]$
- 시스템 응답:$\[y(t) = H(\Omega) e^{j\Omega t} = |H(\Omega)| e^{j(\Omega t + \angle H(\Omega))}\]$

## 2. 합성곱과 주파수 영역
- 시간 영역에서의 출력 신호:$\[y(t) = h(t) * x(t)\]$
  - 여기서 $\( * \)$는 합성곱을 의미합니다.

- 주파수 영역에서의 관계:$\[Y(\Omega) = H(\Omega) X(\Omega)\]$
  - $\(H(\Omega)\)$: 시스템의 주파수 응답.
  - $\(X(\Omega)\)$, $\(Y(\Omega)\)$: 입력 및 출력 신호의 주파수 변환.

## 3. 임펄스 응답과 주파수 응답의 관계
- 임펄스 응답 \(h(t)\)와 주파수 응답 $\(H(\Omega)\)$는 푸리에 변환을 통해 상호 변환됩니다:$\[H(\Omega) = \mathcal{F} \{ h(t) \}, \quad h(t) = \mathcal{F}^{-1} \{ H(\Omega) \}\]$
  - $\( \mathcal{F} \)$: 푸리에 변환.
  - $\( \mathcal{F}^{-1} \)$: 역푸리에 변환.

## 4. 필터 설계와 응용
- 주파수 응답 $\(H(\Omega)\)$는 필터 설계에 활용됩니다
- **필터 구현**
  - 전기 회로를 사용하여 구현 가능 (예: 저역통과, 고역통과 필터)
