# Difference Equation in Discrete-Time Systems

<img width="755" alt="image" src="https://github.com/user-attachments/assets/211c5295-29e2-4179-aca7-90564fb8e0a1" />

## 1. Overview
- Difference Equation은 **연속 시간 미분 방정식**의 이산 시간 버전
- 시스템의 출력 \(y[n]\)을 계산하는 데 있어 가장 효율적인 방식으로, "King of Computation"이라 불림

## 2. Difference Equation
### 일반적인 형태: $\[y[n] + a_1 y[n-1] + a_2 y[n-2] + \dots + a_N y[n-N] = b_0 x[n] + b_1 x[n-1] + \dots + b_M x[n-M]\]$
- **현재 출력**: \(y[n]\)
- **과거 출력**: $\(y[n-1], y[n-2], \dots\)$
- **현재 및 과거 입력**: $\(x[n], x[n-1], \dots\)$

## 3. Parameters
- 시스템은 $\(N + M + 1\)$개의 매개변수 $\(\{a_k, b_k\}\)$로 완전히 정의
- **조건**: $\(a_0 = 1\)$ (일반적으로 정규화됨)

## 4. Computation
- Difference Equation을 활용한 출력 계산: $\[y[n] = -\sum_{k=1}^{N} a_k y[n-k] + \sum_{k=0}^{M} b_k x[n-k]\]$
- **현재 출력** (\(y[n]\))은 다음에 의존
  - **Past Outputs**: $\(-\sum_{k=1}^{N} a_k y[n-k]\)$
  - **Inputs**: $\(\sum_{k=0}^{M} b_k x[n-k]\)$

## 5. Order of the System
- **Nth Order**
  - 시스템의 차수는 최대 \(N\)과 \(M\) 중 더 큰 값에 따라 결정
  - \(N\): 출력 항의 개수
  - \(M\): 입력 항의 개수

 
# 예제: 차분 방정식(Difference Equation) 반복 계산

<img width="775" alt="image" src="https://github.com/user-attachments/assets/2179795e-b057-4a22-8b67-723503400a2d" />

## 1. 문제 정의
- 주어진 차분 방정식: $\[y[n] - \frac{1}{2}y[n-1] = \frac{1}{4}x[n] + \frac{1}{4}x[n-1]\]$
- 이를 다음과 같이 다시 쓸 수 있습니다:$\[y[n] = \frac{1}{2}y[n-1] + \frac{1}{4}x[n] + \frac{1}{4}x[n-1]\]$

## 2. 초기 조건
- 가정
  - $\(x[n] = 0\) for \(n < 0\), \(x[n] = 1\)$ for $\(n \geq 0\)$
  - $\(y[-1] = 0\)$ (초기 조건)

## 3. 반복 계산
### 단계별 계산
1. \(n = 0\): $\[y[0] = \frac{1}{2}y[-1] + \frac{1}{4}x[0] + \frac{1}{4}x[-1]\]$
   - 초기 조건 \(y[-1] = 0\), \(x[0] = 1\), \(x[-1] = 0\)을 대입하면:$\[y[0] = \frac{1}{2}(0) + \frac{1}{4}(1) + \frac{1}{4}(0) = \frac{1}{4}\]$

2. \(n = 1\):$\[y[1] = \frac{1}{2}y[0] + \frac{1}{4}x[1] + \frac{1}{4}x[0]\]$
   - $\(y[0] = \frac{1}{4}\)$, \(x[1] = 1\), \(x[0] = 1\)을 대입하면:$\[y[1] = \frac{1}{2}\left(\frac{1}{4}\right) + \frac{1}{4}(1) + \frac{1}{4}(1) = \frac{1}{8} + \frac{1}{4} + \frac{1}{4} = \frac{5}{8}\]$


## 4. 결과 요약
| \(n\)  | \(y[n]\)               |
|-------|-----------------------|
| \(0\) | \(\frac{1}{4}\)       |
| \(1\) | \(\frac{5}{8}\)       |
| \(2\) | \(\frac{13}{16}\)     |
| \(3\) | \(\frac{29}{32}\)     |


# 예제: 신호 처리 필터 설계

<img width="793" alt="image" src="https://github.com/user-attachments/assets/844d3793-9664-452f-846b-614df777176a" />

## 1. 6 포인트 평균 필터 (6 Point Averaging)
- $\[y[n] = \frac{1}{6} \{ x[n] + x[n-1] + x[n-2] + x[n-3] + x[n-4] + x[n-5] \}\]$
- **설명**: 입력 신호의 최근 6개의 샘플 값을 평균하여 부드럽게 만듭니다
- **특징**: 저역통과 필터(Low-Pass Filter)로 작용하며 잡음을 줄이는 데 유용합니다

## 2. 6 포인트 차분 필터 (6 Point Differencing)
- $\[y[n] = \frac{1}{6} \{ x[n] - x[n-1] - x[n-2] - x[n-3] - x[n-4] - x[n-5] \}\]$
- **설명**: 입력 신호의 차이를 계산하여 고주파 성분을 강조합니다
- **특징**: 고역통과 필터(High-Pass Filter)로 작용하며 신호의 변화 감지에 적합합니다

## 3. 재귀 저역통과 필터 (Recursive Low-Pass Filter)
- $\[y[n] = 0.95y[n-1] + 0.05x[n]\]$
- **설명**: 출력 신호의 이전 값을 사용하여 현재 출력을 계산합니다
- **특징**: 신호의 평균값을 추적하며 저주파 성분을 강조합니다

## 4. 재귀 고역통과 필터 (Recursive High-Pass Filter)
- $\[y[n] = 0.95y[n-1] + 0.05x[n]\]$
- **설명**: 재귀 방식으로 고주파 성분을 강조하여 신호를 필터링합니다
- **특징**: 신호의 빠른 변화를 추적하며 고주파 성분을 강조합니다


# 예제: 6 포인트 평균 및 차분 필터의 입력 및 출력

<img width="778" alt="image" src="https://github.com/user-attachments/assets/f92545ff-b123-49bd-bf0e-bfbeea0cc133" />

## 1. 입력 신호 정의
- 세 가지 입력 신호가 사용
1. **단계 입력(Step Input)**: $\[x[n] =\begin{cases} 0, & n < 0 \\ 1, & n \geq 0\end{cases}\]$

2. **저주파 코사인 입력(Low-Frequency Cosine Input)**: $\[x[n] = \begin{cases} 0, & n < 0 \\ \cos\left(\frac{\pi n}{8}\right), & n \geq 0 \end{cases} \]$

3. **고주파 코사인 입력(High-Frequency Cosine Input)**: $\[x[n] = \begin{cases} 0, & n < 0 \\ \cos\left(\frac{7\pi n}{8}\right), & n \geq 0 \end{cases} \]$

## 2. 필터 정의
- 6 포인트 평균 필터 (6 Point Averaging Filter)
  - $\[y[n] = \frac{1}{6} \{ x[n] + x[n-1] + x[n-2] + x[n-3] + x[n-4] + x[n-5] \}\]$
- 6 포인트 차분 필터 (6 Point Differencing Filter)
  - $\[y[n] = \frac{1}{6} \{ x[n] - x[n-1] - x[n-2] - x[n-3] - x[n-4] - x[n-5] \}\]$

## 3. 출력 그래프 요약

### 6 포인트 평균 필터 결과
1. **단계 입력**
   - 출력은 점진적으로 증가하여 \(n \geq 5\)에서 포화 상태에 도달
2. **저주파 코사인 입력**
   - 저주파 성분을 유지하며 필터링
3. **고주파 코사인 입력**
   - 고주파 성분이 부드럽게 감소

### 6 포인트 차분 필터 결과
1. **단계 입력**
   - \(n = 0\)에서 초기 피크를 보이고, 그 이후 값은 0으로 유지
2. **저주파 코사인 입력**
   - 주파수 성분이 약화되며 변화가 더 부드러워짐
3. **고주파 코사인 입력**
   - 고주파 성분이 상대적으로 강조되어 유지됨

## 4. 요약
- **6 포인트 평균 필터**
  - 저주파 성분을 강조하고 고주파 성분을 억제
  - 단계 입력에서 부드러운 응답을 보임
  
- **6 포인트 차분 필터**
  - 고주파 성분을 강조하고 저주파 성분을 억제
  - 단계 입력에서 초기 변화만 강하게 반응


# 예제: 재귀 필터 (Recursive Filters)의 입력 및 출력

<img width="775" alt="image" src="https://github.com/user-attachments/assets/bd128ae1-bcb4-45dd-ab01-56113a1363af" />

## 1. 필터 정의

- 재귀 저역통과 필터 (Recursive Low-Pass Filter)
  - $\[y[n] = 0.95y[n-1] + 0.05x[n]\]$

- 재귀 고역통과 필터 (Recursive High-Pass Filter)
  - $\[y[n] = -0.95y[n-1] + 0.05x[n]\]$

## 2. 출력 그래프 요약

### 1. 재귀 저역통과 필터 (Low-Pass Filter)
1. **단계 입력**
   - 출력이 점진적으로 증가하며 \(n\)이 커질수록 \(1\)에 가까워짐
2. **저주파 코사인 입력**
   - 저주파 성분을 유지하며 필터링
3. **고주파 코사인 입력**
   - 고주파 성분이 크게 억제됨

### 2. 재귀 고역통과 필터 (High-Pass Filter)
1. **단계 입력**
   - 초기 피크 이후 값이 감소하며 0에 가까워짐
2. **저주파 코사인 입력**
   - 저주파 성분이 억제되며 고주파 성분이 상대적으로 유지됨
3. **고주파 코사인 입력**
   - 고주파 성분이 강조되어 출력에서 유지됨

## 3. 요약
- **재귀 저역통과 필터**
  - 신호의 저주파 성분을 강조하고 고주파 성분을 억제
  - 단계 입력에서는 부드러운 증가를 보임
  
- **재귀 고역통과 필터**
  - 신호의 고주파 성분을 강조하고 저주파 성분을 억제
  - 단계 입력에서는 초기 변화만 반응
