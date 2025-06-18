# Z-변환 소개 (Introduction to the Z-Transform)

![image](https://github.com/user-attachments/assets/b61ca60c-567d-4a20-be13-d0f9c10f6811)


## 1. Z-변환의 개요
- Z-변환은 이산 푸리에 변환(DTFT)의 일반화입니다.
- **DTFT가 존재하지 않는 신호**에 대해 사용할 수 있습니다.
- Z-변환은 새로운 개념을 도입합니다:
  - **안정성(Stability)** 분석.
  - **인과성(Causality)** 분석.

## 2. Z-변환의 정의
- Z-변환:$\[X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}, \quad z \in \mathbb{C} \text{ (복소수 영역)}\]$

- DTFT와의 관계:$\[X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n} \quad \Rightarrow \quad X(z)\big|_{z=e^{j\omega}} = X(e^{j\omega})\]$

## 3. Z-변환과 DTFT의 차이점
1. **DTFT**:
   - 실수 값 주파수 $\(\omega\)$에 대한 복소수 함수
   - 주파수 성분에 대한 분석에 초점

2. **Z-변환**:
   - 복소수 값 \(z\)에 대한 복소수 함수
   - 신호의 안정성과 인과성 분석에 적합

## 4. Z-변환과 라플라스 변환의 관계
- Z-변환은 **이산 시간 라플라스 변환**의 대응물입니다
- DTFT는 Z-변환의 단위 원 \(|z| = 1\)에서의 특수한 경우입니다


# 복소수 평면 (The Complex Plane, z-plane)

![image](https://github.com/user-attachments/assets/dffc89d0-4436-48a5-a16a-d31a13248778)

## 1. Z-평면과 단위 원 (Unit Circle)
- Z-변환 \(X(z)\)는 복소수 평면 \(z\)-평면에서 정의됩니다.
- DTFT $\(X(e^{j\omega})\)$는 $\(z = e^{j\omega}\)$로 표현되며, 단위 원 \(|z| = 1\) 상에서 정의됩니다
  - $\[X(z) \quad \text{(Z-평면 전체에서 정의)}\]$
  - $\[X(e^{j\omega}) \quad \text{(단위 원 \(|z| = 1\) 상에서 정의)}\]$

## 2. 단위 원의 특징
- $\(z = e^{j\omega}\)$가 단위 원을 한 바퀴 회전하면서 $\(-\pi < \omega < \pi\)$ 범위를 탐색
- 이로 인해 DTFT의 주기성 ($\(2\pi\)$ -주기)을 확인할 수 있습니다

## 3. Z-변환과 DTFT의 관계
1. **Z-평면**
   - Z-변환은 복소수 평면 전체를 탐색하며, 신호의 안정성과 인과성을 분석하는 데 사용됩니다
2. **단위 원**
   - DTFT는 단위 원 상에서 신호의 주파수 응답을 나타냅니다


# Z-변환의 표기법과 예제 (Notation and Examples of Z-Transform)

![image](https://github.com/user-attachments/assets/fce98900-92bf-4081-87a9-60f794bdce66)

## 1. Z-변환 표기법
- 이산 신호 \(x[n]\)의 Z-변환:$\[x[n] \xrightarrow{Z} X(z)\]$
- 수식으로 표현:$\[X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}\]$

## 2. 예제 (Examples)

### (1) 예제 1: \(X(z) = z\)
- \(X(z)\)가 단순한 선형 함수 \(z\)일 때
  - Z-평면에서의 크기와 위상이 선형적으로 변화
  - 3D 그래프에서 \(X(z) = z\)는 평면적으로 증가하는 모습을 보여줌

### (2) 예제 2: $\(X(z) = \frac{1}{z - 0.58}\)$
- \(X(z)\)는 극(Pole)을 포함한 유리형 함수
  - 극은 \(z = 0.58\)에 위치
  - 3D 그래프에서 극(Pole) 주위의 값이 무한대로 발산하는 형태를 보임

## 3. Z-변환의 시각화
- **Z-평면에서의 표현**
  - Z-변환의 크기와 위상은 복소수 \(z\)에 대한 함수로 나타납니다
  - 극(Pole)과 영점(Zero)의 위치는 Z-변환의 특성을 직관적으로 이해하는 데 도움을 줍니다

- **3D 그래프**
  - 첫 번째 그래프 $\(X(z) = z\)$: 선형적으로 증가하는 평면
  - 두 번째 그래프 $\(X(z) = \frac{1}{z - 0.58}\)$: 극 주위에서 발산하는 형태

---

## 4. 요약
1. Z-변환은 이산 신호를 복소수 영역으로 변환하여 신호의 특성을 분석합니다.
2. Z-평면 상에서의 시각화는 Z-변환 함수의 극과 영점의 영향을 이해하는 데 유용합니다.
3. 예제는 Z-변환의 다양한 동작을 보여줍니다:
   - 단순 선형 함수.
   - 극(Pole)을 포함한 유리형 함수.
