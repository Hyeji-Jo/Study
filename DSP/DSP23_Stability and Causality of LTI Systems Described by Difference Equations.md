# 시스템 특성과 차분 방정식 (System Properties and Difference Equation)

![image](https://github.com/user-attachments/assets/a0b96fb4-a32b-427a-8775-e08319fddb1f)

## 1. 시스템 특성
### 안정성 (Stability)
- 시스템이 안정적이려면 충족해야 할 조건:$\( \sum_{n=-\infty}^\infty |h[n]| < \infty \)$
  - 이는 임펄스 응답 $\(h[n]\)$이 절대적으로 수렴해야 함을 의미합니다

### 인과성 (Causality)
- 시스템이 인과적이려면 충족해야 할 조건:$\( h[n] = 0, \, n < 0 \)$
  - 이는 현재 및 과거의 입력만이 출력에 영향을 미쳐야 함을 나타냅니다

## 2. 차분 방정식으로 시스템 설명
- 시스템은 아래와 같은 차분 방정식으로 표현될 수 있습니다:$\( \sum_{k=0}^N a_k y[n-k] = \sum_{k=0}^M b_k x[n-k] \)$
  - 여기서
    - $\(a_k\)$와 $\(b_k\)$는 시스템 계수입니다
    - $\(x[n]\)$: 입력 신호
    - $\(y[n]\)$: 출력 신호

## 3. 전달 함수 $\(H(z)\)$
- $\(H(z)\)$는 시스템의 전달 함수 또는 주파수 응답을 나타냅니다:$\( H(z) = \frac{\sum_{k=0}^M b_k z^{-k}}{\sum_{k=0}^N a_k z^{-k}} \)$
- $\(H(z)\)$와 $\(h[n]\)$는 $z$ 변환을 통해 상호 변환 가능합니다:$\( H(z) \leftrightarrow h[n] \)$

### 그림 설명
1. **블록 다이어그램**
   - 입력 신호 $\(x[n]\)$이 시스템 $\(h[n]\)$을 통과하여 출력 $\(y[n]\)$ 생성
   - $\( y[n] = x[n] * h[n] \)$: 입력과 임펄스 응답의 컨볼루션
2. **안정성 및 인과성**
   - 안정성: $\(h[n]\)$의 절대적 합이 유한해야 함
   - 인과성: $\(h[n]\)$이 $\(n < 0\)$에서 0이어야 함
  
# 전달 함수와 ROC (Region of Convergence) 분석
![image](https://github.com/user-attachments/assets/f1480311-1c57-4496-861a-ad898f093d1d)

## 전달 함수 $\(H(z)\)$
- $\(H(z)\)$는 아래와 같이 정의됩니다:$\( H(z) = \frac{\sum_{k=0}^M b_k z^{-k}}{\sum_{k=0}^N a_k z^{-k}} = \frac{b_0}{a_0} \prod_{k=1}^M (1 - c_k z^{-1}) \Big/ \prod_{k=1}^N (1 - d_k z^{-1}) \)$
  - $\(c_k\)$: 영점 (zeros)
  - $\(d_k\)$: 극점 (poles)

## 1. 인과성 (Causality)
- 시스템이 인과적이려면 모든 극점에 대해 오른쪽 변환 (Right-Sided Inverse Transform)을 가져야 합니다
- **ROC (Region of Convergence)**
  - 가장 큰 반지름을 가진 극점에서 바깥 방향으로 확장됩니다
  - $\( |z| > \max |d_k| \)$

## 2. 안정성 (Stability)
- 시스템이 안정적이려면 임펄스 응답 $\(h[n]\)$이 절대적으로 수렴해야 합니다
  - $\( \sum_{n=-\infty}^\infty |h[n]| < \infty \)$
- 절대적 수렴성을 주파수 영역으로 표현
  - $\( \sum_{n=-\infty}^\infty |h[n]| = \sum_{n=-\infty}^\infty |h[n] z^{-n}| \Big|_{|z|=1} = |H(z)|_{|z|=1} \)$
- **결론**
  - $\(H(z)\)$가 단위원 (unit circle)에서 유한한 값을 가지면 $\(h[n]\)$은 안정적입니다
  - 따라서, $\(H(z)\)$의 ROC는 단위원을 포함해야 합니다


### 그림 설명
1. **극점과 영점의 배치**
   - $z$-평면에서 극점과 영점의 배치가 시스템 특성을 결정합니다
2. **ROC 시각화**
   - 가장 큰 극점의 반지름 외부가 ROC로 표시됨
   - 안정성을 위해 단위원이 ROC에 포함됩니다
  

# 예제: 시스템의 안정성 및 인과성 분석
![image](https://github.com/user-attachments/assets/102f1de6-a29c-402c-8b18-24460fbbff0c)

## 차이 방정식 (Difference Equation)
- $\( y[n] - \frac{3}{2}y[n-1] - y[n-2] = 2x[n] - x[n-1] \)$

## 전달 함수 (Transfer Function)
1. 전달 함수 $\(H(z)\)$는 아래와 같이 계산됩니다
   - $\( H(z) = \frac{2 - z^{-1}}{1 - \frac{3}{2}z^{-1} - z^{-2}} = \frac{2}{1 - 2z^{-1}} + \frac{1}{1 + \frac{1}{2}z^{-1}} \)$

2. 극점
   - $\(z = 2\)$
   - $\(z = -\frac{1}{2}\)$

3. 분해
   - 첫 번째 항: $\( \frac{2}{1 - 2z^{-1}} \)$
   - 두 번째 항: $\( \frac{1}{1 + \frac{1}{2}z^{-1}} \)$

## 분석

### 안정한 경우 (Stable Case)
- **ROC**
  - $\( |z| > 2 \)$
  - 단위원 $\( |z| = 1 \)$ 포함
- **임펄스 응답 $\(h[n]\)$**
  - $\( h[n] = \left(-\frac{1}{2}\right)^n u[n] - 2(2)^n u[-n-1] \)$
- **특징**
  - 안정성은 충족하지만, 인과성은 충족하지 않음 (not causal)


### 인과적인 경우 (Causal Case)
- **ROC**
  - $\( |z| > \frac{1}{2} \)$
- **임펄스 응답 $\(h[n]\)$**
  - $\( h[n] = \left(-\frac{1}{2}\right)^n u[n] + 2(2)^n u[n] \)$
- **특징**
  - 인과성은 충족하지만, 안정성은 충족하지 않음 (not stable)

## 결론
- 시스템이 안정하려면 ROC가 단위원 $\( |z| = 1 \)$을 포함해야 함
- 시스템이 인과적이려면 ROC가 극점 외부를 포함해야 함
- 이 예제에서는 안정성과 인과성을 동시에 충족할 수 없음

### 그림 설명
1. **Stable Case**
   - ROC는 $\( |z| > 2 \)$
   - 단위원을 포함하므로 안정적
   - 하지만 임펄스 응답에서 인과성이 보장되지 않음

2. **Causal Case**
   - ROC는 $\( |z| > \frac{1}{2} \)$
   - 인과성을 만족하지만, 단위원을 포함하지 않아 안정적이지 않음


# 안정성과 인과성 (Stability and Causality)

![image](https://github.com/user-attachments/assets/ec17fdb3-c1d5-4bfd-a90f-76cc69c9151d)

## 안정성과 인과성의 관계
- 안정하고 인과적인 시스템은 모든 극점이 단위원 내부 $\( |z| = 1 \)$에 위치해야 합니다
- 하지만 안정성과 인과성은 항상 동시에 만족되지 않을 수 있습니다.

## 예제 1: 안정하고 인과적인 시스템
### 차이 방정식
- $\( y[n] + \frac{3}{4}y[n-1] + \frac{1}{8}y[n-2] = x[n] - x[n-1] \)$

### 전달 함수 (Transfer Function)
- $\( H(z) = \frac{1 - z^{-1}}{1 + \frac{3}{4}z^{-1} + \frac{1}{8}z^{-2}} \)$

### 분석
1. 극점 (Poles)
   - $\( z = -\frac{1}{2} \)$
   - $\( z = -\frac{1}{4} \)$
2. ROC
   - $\( |z| > \text{largest pole radius} = \frac{1}{4} \)$
3. 특성
   - 모든 극점이 단위원 내부에 위치하며 ROC가 적절하므로 안정성과 인과성을 모두 만족합니다

## 예제 2: 안정적이지 않고 인과적인 시스템
### 전달 함수 (Transfer Function)
- $\( H(z) = \frac{z^2 + 2z + 1}{z - \frac{1}{2}} \)$

### 분석
1. 극점 (Poles)
   - $\( z = \frac{1}{2} \)$
   - $\( z = \infty \)$
2. ROC
   - $\( |z| > \frac{1}{2} \)$ (인과적 조건을 만족)
3. 특성
   - 단위원을 포함하지 않으므로 안정적이지 않음
   - 따라서 안정성과 인과성을 동시에 만족할 수 없습니다

## 결론
1. 안정성과 인과성을 동시에 만족하기 위해서는 모든 극점이 단위원 내부에 위치해야 하며, ROC가 단위원을 포함해야 합니다.
2. 하지만 일부 시스템에서는 두 조건을 동시에 만족시키는 것이 불가능합니다.


### 관련 임펄스 응답 $\(h[n]\)$
1. 안정하고 인과적인 경우:$\( h[n] = \delta[n] - \frac{3}{4}\delta[n-1] - \frac{1}{8}\delta[n-2] \)$

2. 안정적이지 않고 인과적인 경우:$\( h[n] = \delta[n+1] - 2\delta[n] + \frac{7}{2}\left(\frac{1}{2}\right)^n u[n] \)$
