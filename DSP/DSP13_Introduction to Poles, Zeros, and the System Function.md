# 주파수 응답과 Z 변환 (Frequency Response and Z-Transform)

<img width="618" alt="image" src="https://github.com/user-attachments/assets/d98be564-e0aa-49d3-b77b-cc84be8488bf" />

## 1. LTI 시스템과 주파수 응답
- 주파수 응답 $\(H(e^{j\omega})\)$는 LTI 시스템 \(H\)에서 복소 지수 입력 $\(x[n] = e^{j\omega n}\)$에 대해 출력 \(y[n]\)을 설명합니다: $\[y[n] = H(e^{j\omega}) e^{j\omega n}\]$

## 2. Z 변환과 시스템 함수
- **Z 변환 정의**:$\[z = re^{j\omega}, \quad z^n = r^n e^{j\omega n}\]$
  - $\(r = |z|\)$: 반지름 (크기)
  - $\(\omega\)$: 각도 (주파수)

- Z 변환을 사용한 시스템 응답:$\[H(z) = \sum_{k=-\infty}^{\infty} h[k] z^{-k}\]$
  - 여기서 \(H(z)\)는 **시스템 함수**이며, \(z\)-평면에서 시스템의 특성을 설명합니다

## 3. 주파수 응답과 Z 변환의 관계
- Z 변환에서 \(r = 1\) ($\(z = e^{j\omega}\)$)일 때:$\[H(e^{j\omega}) = H(z) \big|_{z=e^{j\omega}}\]$
  - 이는 단위 원 (Unit Circle) 상에서의 Z 변환 값을 의미합니다

## 4. 단위 원과 Z 평면
- Z 변환에서 주파수 응답은 단위 원 $\(|z| = 1\)$에서 측정됩니다
  - **단위 원의 중요성**
    - 주파수 응답 $\(H(e^{j\omega})\)$는 Z-평면의 단위 원 위의 값으로 결정됩니다

## 5. 출력 신호 계산
- Z 변환을 이용한 출력 신호:$\[y[n] = \sum_{k=-\infty}^{\infty} h[k] x[n-k] = z^n \sum_{k=-\infty}^{\infty} h[k] z^{-k}\]$
  - 여기서 \(H(z)\)는 임펄스 응답 \(h[k]\)의 Z 변환입니다


# Z 변환과 극-영점 (Poles and Zeros in Z-Transform)

<img width="635" alt="image" src="https://github.com/user-attachments/assets/a28cf40f-8a9c-4022-aac4-4eb66c3c5f8e" />

## 1. Z 변환 시스템 함수
- 주어진 차분 방정식:$\[\sum_{k=0}^{N} a_k y[n-k] = \sum_{k=0}^{M} b_k x[n-k}\]$

- Z 변환을 통해 시스템 함수 \(H(z)\)는 다음과 같이 정의됩니다:$\[H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}}\]$
- 이는 **유리형 시스템 함수**(Rational System Function)를 나타냅니다.

## 2. 극(Pole)과 영점(Zero)
1. **극(Pole)**
  - 분모가 0이 되는 \(z\) 값:$\[\sum_{k=0}^{N} a_k z^{-k} = 0\]$
  - $\(H(z) \to \infty\)$일 때

2. **영점(Zero)**
  - 분자가 0이 되는 \(z\) 값:$\[\sum_{k=0}^{M} b_k z^{-k} = 0\]$
  - $\(H(z) \to 0\)$일 때

## 3. 시스템 함수의 팩토링 (Factored Form)
- 시스템 함수 \(H(z)\)는 극과 영점을 사용하여 다음과 같이 팩토링할 수 있습니다:$\[H(z) = \frac{b_0 \prod_{k=1}^{M} (1 - c_k z^{-1})}{a_0 \prod_{k=1}^{N} (1 - d_k z^{-1})}\]$
  - $\(c_k\)$: 영점 (Zeros)
  - $\(d_k\)$: 극 (Poles)

## 4. Z-평면에서의 극과 영점
- Z-평면 상에서
  - **영점(Zeros)**: 원점 또는 단위 원 안의 위치에 "o"로 표시
  - **극(Poles)**: 단위 원 안, 밖, 또는 위에 위치하며 "x"로 표시

- **단위 원**
  - 시스템의 안정성은 극이 단위 원 내부에 위치할 때 보장됩니다

 
# 예제: Z 변환, 극(Poles), 영점(Zeros) 계산

<img width="634" alt="image" src="https://github.com/user-attachments/assets/9ac59160-f247-4b8d-a8d1-b2a5b3bee9ed" />

## 1. 문제 정의
- 주어진 차분 방정식:$\[y[n] - \frac{3}{8}y[n-1] - \frac{7}{16}y[n-2] = x[n] + x[n-2]\]$
- 계수
  - $\(a_0 = 1, a_1 = -\frac{3}{8}, a_2 = -\frac{7}{16}\)$
  - $\(b_0 = 1, b_1 = 0, b_2 = 1\)$

## 2. Z 변환 시스템 함수
- 시스템 함수 \(H(z)\):$\[H(z) = \frac{\sum_{k=0}^{M} b_k z^{-k}}{\sum_{k=0}^{N} a_k z^{-k}} = \frac{1 + z^{-2}}{1 - \frac{3}{8}z^{-1} - \frac{7}{16}z^{-2}}\]$

- 팩토링된 형태:$\[H(z) = \frac{(1 + jz^{-1})(1 - jz^{-1})}{\left(1 - \frac{7}{8}z^{-1}\right)\left(1 + \frac{1}{2}z^{-1}\right)}\]$

## 3. 극(Poles)와 영점(Zeros)
1. **영점(Zeros)**
   - $\(z = \pm j\)$

2. **극(Poles)**
   - $\(z = \frac{7}{8}\)$
   - $\(z = -\frac{1}{2}\)$

## 4. 극과 영점의 성질
- 계수 $\(a_k\)$와 $\(b_k\)$가 실수일 때
  - 극과 영점은 항상 **켤레 복소수 쌍**(Conjugate Pairs)으로 나타납니다
- Z-평면에서
  - 영점은 단위 원 상에 $\(z = \pm j\)$에 위치
  - 극은 $\(z = \frac{7}{8}\)$ (단위 원 내부)와 $\(z = -\frac{1}{2}\)$ (단위 원 내부)에 위치

## 5. Z-평면 극과 영점 그래프
- Z-평면
  - 단위 원 $\(|z| = 1\)$ 상에 영점(⊕)이 위치
  - 극(×)은 단위 원 내부에 위치

# 극(Pole)과 영점(Zero) 위치가 제공하는 정보

<img width="633" alt="image" src="https://github.com/user-attachments/assets/f368cb4b-7849-415b-813e-63f44a93b541" />

## 1. 안정성 (Stability)
- 시스템이 **안정적**이고 **인과적(Causal)**이기 위한 조건:$\[\text{모든 극(Pole)} \quad |z| < 1 \quad \text{(단위 원 내부에 위치)}\]$
- 모든 극이 단위 원 내부에 위치하면, 시스템은 안정적입니다

## 2. 응답 시간 (Response Time)
- 시스템의 초기 상태 \(n = 0\)에서의 응답은 다음과 같이 나타낼 수 있습니다:$\[\text{응답} = \sum_{k=1}^{N} \alpha_k d_k^n + \text{Steady State}\]$
  - $\(d_k\)$: 극의 위치
  - $\(\alpha_k\)$: 극의 계수

- 극의 위치에 따른 응답 속도
  - **극이 $\(z = 0\)$에 가까울수록**: 빠른 응답(Fast Response)
  - **극이 단위 원 $\(|z| = 1\)$에 가까울수록**: 느린 응답(Slow Response)

## 3. 주파수 응답 (Frequency Response)
- 주파수 응답은 극(Pole)과 영점(Zero)의 위치와 단위 원 사이의 거리로 결정됩니다
- 극과 영점이 단위 원에 가까울수록 주파수 응답에 더 큰 영향을 미칩니다

