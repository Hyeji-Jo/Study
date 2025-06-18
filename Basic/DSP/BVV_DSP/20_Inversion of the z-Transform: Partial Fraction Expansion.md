## 부분 분수 전개 (Partial Fraction Expansion)
![image](https://github.com/user-attachments/assets/b57acc44-e499-48b1-9b2b-8d20ca8c4164)

- **목적**: $X(z)$를 간단한 형식으로 분해하여 쉽게 역변환할 수 있도록 변환합니다

- 주어진 함수:$\[  X(z) = \frac{1 + 2z^{-1} + z^{-2}}{1 - \frac{3}{2}z^{-1} + \frac{1}{2}z^{-2}}, \quad |z| > 1  \]$

-  분수 전개:$\[  X(z) = \frac{8}{1 - z^{-1}} - \frac{9}{1 - \frac{1}{2}z^{-1}} + 2  \]$

### 과정 설명
1. 분모를 간단한 인수들로 분리합니다
2. 각각의 항은 역변환하기 쉬운 형식으로 작성됩니다


# 분수 함수의 전개 및 처리 (Fraction Expansion and Processing)
![image](https://github.com/user-attachments/assets/a4b1ab22-fa2b-4ba2-82cc-771eb630892d)

## 1. 일반적인 표현
- 주어진 함수 \(X(z)\)는 다음과 같이 표현됩니다:$\[  X(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}}  \]$

## 2. 처리 단계

### (1) $\(M \geq N\)$일 경우: Long Division 사용
- $\(M \geq N\)$인 경우, 분자를 분모로 나누어 \(X(z)\)를 다항식과 나머지 항의 조합으로 작성합니다
  - $\[  X(z) = \sum_{r=0}^{M-N} B_r z^{-r} + \frac{\sum_{k=0}^{N-1} b'_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}}  \]$

### (2) 분모를 1차항의 곱으로 분해
- 분모를 1차항의 곱으로 표현합니다
  - $\[  \frac{\sum_{k=0}^{N-1} b'_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}} = \frac{\sum_{k=0}^{N-1} b'_k z^{-k}}{a_0 \prod_{k=1}^{N} (1 - d_k z^{-1})}  \]$

- 이 표현은 **부분 분수 전개(Partial Fraction Expansion)**에 유용하게 사용됩니다

## 주요 아이디어
1. **Long Division**: $\(M \geq N\)$일 경우 사용하여 \(X(z)\)를 다항식과 나머지 항으로 나눕니다
2. **분모 분해**: 분모를 곱으로 분해하여 각 항을 역변환하기 쉽게 만듭니다

## 요약
- 분수 함수를 효율적으로 다루기 위해 $\(M \geq N\)$일 경우 long division을 사용하며, 분모는 곱으로 분해합니다
- 이 과정을 통해 \(X(z)\)를 간단한 형식으로 변환하여 분석하거나 역변환할 수 있습니다


# Partial Fraction Expansion: Rational Functions

![image](https://github.com/user-attachments/assets/1079a0ae-911b-4095-a420-e2e5d6fd34b5)

## 3. 분수 함수의 1차항 전개 (Partial Fraction Expansion in First-Order Terms)
- 주어진 분수 함수를 1차항의 합으로 전개합니다:$\[  \frac{\sum_{k=0}^{N-1} b'_k z^{-k}}{a_0 \prod_{k=1}^{N} (1 - d_k z^{-1})} = \sum_{k=1}^{N} \frac{A_k}{1 - d_k z^{-1}} \quad (\text{assuming distinct } d_k)  \]$

- $\(d_k\)$가 반복되는 경우:$\[  \sum_{r=1}^{s} \frac{A_r}{(1 - d_i z^{-1})^r}  \]$

## 4. ROC를 사용한 각 항의 역변환 (Invert Each Term Using ROC)

### (1) 단일 항의 역변환
- 주어진 항 $\(\frac{A_k}{1 - d_k z^{-1}}\)$
  - $\(|z| > |d_k|\)$: $\[ A_k d_k^n u[n] \]$
  - $\(|z| < |d_k|\)$: $\[ -A_k d_k^n u[-n-1] \]$

### (2) 반복된 항의 역변환
- 주어진 반복된 항 $\(\frac{A}{(1 - d_k z^{-1})^p}\)$
  - $\(|z| > |d_k|\)$:$\[  A \frac{(n+1)(n+2)\cdots(n+p-1)}{(p-1)!} d_k^n u[n]  \]$
  - $\(|z| < |d_k|\)$:$\[  -A \frac{(n+1)(n+2)\cdots(n+p-1)}{(p-1)!} d_k^n u[-n-1]  \]$

## 요약
- **1차항 전개**를 통해 복잡한 분수 함수를 단순화합니다
- **ROC**(Region of Convergence)를 활용하여 각 항을 적절히 역변환합니다

## 예제: 부분 분수 전개를 활용한 $\(X(z)\)$의 분해
![image](https://github.com/user-attachments/assets/6db2bc31-3e90-49d5-87e2-a42c199e9f2b)

- 주어진 식:$\[ X(z) = \frac{1 - z^{-1} + z^{-2}}{(1 - \frac{1}{2} z^{-1})(1 - 2 z^{-1})(1 - z^{-1})}, \quad \text{ROC: } 1 < |z| < 2 \]$

### 1. 조건: $\(M = 2 < N = 3\)$
- 분자의 차수가 분모의 차수보다 작으므로 **장제법(Long Division)**은 필요하지 않음

### 2. 분모 인수 분해
- 분모는 이미 다음과 같이 인수 분해됨:$\[ (1 - \frac{1}{2} z^{-1})(1 - 2 z^{-1})(1 - z^{-1}) \]$

### 3. 부분 분수 전개
- $\(X(z)\)$를 다음과 같이 전개:$\[ X(z) = \frac{A_1}{1 - \frac{1}{2} z^{-1}} + \frac{A_2}{1 - 2 z^{-1}} + \frac{A_3}{1 - z^{-1}} \]$

#### 각 계수 $\(A_1, A_2, A_3\)$ 계산
- $\(A_1\)$:$\[ A_1 = X(z)(1 - \frac{1}{2} z^{-1}) \Big|_{z^{-1} = \frac{1}{2}} = 1 \]$
- $\(A_2\)$:$\[ A_2 = X(z)(1 - 2 z^{-1}) \Big|_{z^{-1} = 2} = 2 \]$
- $\(A_3\)$:$\[ A_3 = X(z)(1 - z^{-1}) \Big|_{z^{-1} = 1} = -2 \]$

### 4. 최종 결과:$\[ X(z) = \frac{1}{1 - \frac{1}{2} z^{-1}} + \frac{2}{1 - 2 z^{-1}} - \frac{2}{1 - z^{-1}} \]$

### 요약:
1. $\(M < N\)$이므로 장제법을 사용하지 않아도 됨
2. 분모는 서로 다른 1차 항의 곱으로 인수 분해됨
3. $\(A_1, A_2, A_3\)$는 각각 해당 $\(z^{-1}\)$ 값을 대입하여 계산

## 4. 각 항을 ROC 정보를 사용하여 변환하기
![image](https://github.com/user-attachments/assets/4b97ccb7-f874-4c66-a9e6-c5b70b01ec87)

- 주어진 정보:**ROC**: $\(1 < |z| < 2\)$

### 항 별 변환
1. $\[ \frac{1}{1 - \frac{1}{2} z^{-1}} \leftrightarrow \left(\frac{1}{2}\right)^n u[n] \]$
2. $\[ \frac{2}{1 - 2 z^{-1}} \leftrightarrow -2 \cdot 2^n u[-n-1] \]$
3. $\[ \frac{-2}{1 - z^{-1}} \leftrightarrow -2 \cdot u[n] \]$

### 최종 결과
- $\[ x[n] = \left(\frac{1}{2}\right)^n u[n] - 2 \cdot 2^n u[-n-1] - 2 \cdot u[n] \]$

### 요약
- 각 항은 부분 분수 전개를 기반으로 $z$-변환의 역변환을 수행하여 변환됨
- **ROC**에 따라 오른쪽 신호($u[n]$)와 왼쪽 신호($u[-n-1]$)로 나뉨


## Example: \(X(z)\)

![image](https://github.com/user-attachments/assets/3f61b70b-fb99-4fcb-9bcb-f341d52d93ad)

- 주어진 식:$\[ X(z) = \frac{z^3 - 10z^2 - 4z + 4}{2z^2 - 2z - 4}, \quad |z| < 1 \]$

### 단계별 풀이
#### 0) \(z^{-1}\)의 거듭제곱으로 표현
- $\[ X(z) = \frac{1 - 10z^{-1} - 4z^{-2} + 4z^{-3}}{2 - 2z^{-1} - 4z^{-2}} \]$

#### 1) \(M = 3 > N = 2\), **장제법(long division)** 사용
- $\[ X(z) = -z^{-1} + \frac{3}{2} + \frac{-5z^{-1} - 2}{2 - 2z^{-1} - 4z^{-2}} \]$

### 최종 표현식
- $\[ X(z) = -z^{-1} + \frac{3}{2} + \frac{-5z^{-1} - 2}{2 - 2z^{-1} - 4z^{-2}} \]$

### 요약
- \(X(z)\)를 $\(z^{-1}\)$의 항으로 재정리하여 \(z\)-변환의 역변환 계산을 간소화
- 장제법을 활용하여 상위 차수 \(M > N\)의 분자/분모를 나눔


![image](https://github.com/user-attachments/assets/3cba15ee-491d-42e3-8248-6b6b074cc2ec)
- 주어진 식:$\[ X(z) = -z^{-1} + \frac{3}{2} + \frac{-5z^{-1} - 2}{2(1+z^{-1})(1-2z^{-1})}, \quad |z| < 1 \]$

### 단계별 풀이
#### 2) 분모를 인수분해
- $\[ X(z) = -z^{-1} + \frac{3}{2} + \frac{-5z^{-1} - 2}{2(1+z^{-1})(1-2z^{-1})} \]$

#### 3) 부분 분수 전개 (Partial Fraction Expansion)
- $\[ \frac{-5z^{-1} - 2}{2(1+z^{-1})(1-2z^{-1})} = \frac{A_1}{1+z^{-1}} + \frac{A_2}{1-2z^{-1}}, \quad |z| < 1 \]$

  - $\(A_1\)$ 계산:$\[ A_1 = \left(\frac{-5z^{-1} - 2}{1-2z^{-1}}\right)\bigg|_{z^{-1} = -1} = \frac{1}{2} \]$
  
  - $\(A_2\)$ 계산:$\[ A_2 = \left(\frac{-5z^{-1} - 2}{1+z^{-1}}\right)\bigg|_{z^{-1} = \frac{1}{2}} = -\frac{3}{2} \]$
  
  - 전개 후 \(X(z)\):$\[ X(z) = -z^{-1} + \frac{3}{2} + \frac{1/2}{1+z^{-1}} - \frac{3/2}{1-2z^{-1}} \]$

#### 4) ROC 정보를 이용한 각 항의 역변환
- 첫 번째 항:$\[ \frac{1}{1+z^{-1}} \quad \xrightarrow{z} \quad -\frac{1}{2}(-1)^n u[-n-1] \]$
- 두 번째 항:$\[ \frac{1}{1-2z^{-1}} \quad \xrightarrow{z} \quad \frac{3}{2}(2)^n u[-n-1] \]$
- 전체 역변환:$\[ x[n] = -\delta[n-1] + \frac{3}{2}\delta[n] - \frac{1}{2}(-1)^n u[-n-1] + \frac{3}{2}(2)^n u[-n-1] \]$


### 결과
- $\(x[n]\)$: $\[ x[n] = -\delta[n-1] + \frac{3}{2}\delta[n] - \frac{1}{2}(-1)^n u[-n-1] + \frac{3}{2}(2)^n u[-n-1] \]$


# 부분 분수 전개와 ROC (Region of Convergence) 관련 정리

![image](https://github.com/user-attachments/assets/10e0dcb0-de29-4d9e-91cf-29937605f5a5)
## 1. $\(X(z)\)$의 극과 계수

- **$\(X(z)\)$의 극**
  - $\(X(z)\)$의 계수가 실수(real)라면, 극(pole)은 **켤레 복소수(conjugate pair)** 형태로 나타남:$\(d_1 = d_2^*\)$

- **부분 분수 전개(Partial Fraction Expansion, PFE)**
  - PFE의 계수 역시 **켤레 복소수 쌍**으로 나타나야 함:$\( \frac{A_1}{1 - d_1 z^{-1}} + \frac{A_2}{1 - d_2 z^{-1}} = \frac{A_R + jA_I}{1 - d_1 z^{-1}} + \frac{A_R - jA_I}{1 - d_2 z^{-1}} \)$
  - 여기서
    - $\(A_R\)$: 실수(real) 부분
    - $\(A_I\)$: 허수(imaginary) 부분

- **설명**
  - $\(X(z)\)$의 계수가 실수라면, 시스템이 대칭적인 특성을 가지므로, 극 역시 켤레 복소수 형태로 배치됩니다
  - PFE의 계수도 같은 대칭성을 가지기 때문에, 극의 형태와 계수 간의 관계를 이해하면 시스템의 특성을 더 잘 분석할 수 있습니다

## 2. ROC와 추가 정보

- **인과 신호(Causal Signal)**
  - 신호가 인과적(causal)이라면, **우측 신호(right-sided signal)**로 나타나야 함
  - 이는 주로 $\(u[n]\)$ 형태로 표현됩니다

- **안정 신호(Stable Signal)**
  - 신호가 안정적(stable)이라면, **ROC(수렴 영역)**은 단위 원($\(|z| = 1\)$)을 포함해야 함

- **설명**
  - 인과 신호는 현재 또는 과거 정보로부터 미래 출력을 계산할 수 있음을 의미합니다. 이는 디지털 필터 설계에서 중요한 속성입니다.
  - 안정 신호는 시스템이 발산하지 않고 출력이 안정적으로 유지된다는 것을 보장합니다. 이를 위해 ROC는 단위 원을 포함해야 하며, 이는 신호가 에너지적으로 수렴한다는 의미입니다.

## 예제: 우측 신호의 시간 영역 표현
- $\( \frac{z}{z - d} \)$
  - Z-변환으로부터 시간 영역 신호로 변환하면:$\( \frac{z}{z - d} \xrightarrow{z} d^n u[n] \)$

- **설명**
  - $\( \frac{z}{z - d} \)$는 우측 신호의 Z-변환 표현입니다
  - 시간 영역에서 $\(d^n u[n]\)$로 변환되며, 이는 신호가 $\(n \geq 0\)$에서만 존재함을 나타냅니다
  - 계수 $\(d\)$는 신호의 감쇠 또는 증폭을 결정합니다. $\( |d| < 1 \)$인 경우 신호는 감쇠하고, $\( |d| > 1 \)$인 경우 신호는 점차 커집니다

### 요약
1. **극(pole)과 계수**
   - 극이 켤레 복소수 형태라면, PFE 계수도 대칭적인 형태를 가짐
2. **인과적(Causal) 신호**
   - 우측 신호로 표현되며 $\(u[n]\)$ 형태
3. **안정적(Stable) 신호**
   - ROC가 단위 원을 포함해야 신호가 발산하지 않음
