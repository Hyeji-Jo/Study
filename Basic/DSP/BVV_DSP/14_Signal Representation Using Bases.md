
# 신호 표현과 기저 신호 (Signal Representation and Basis Signals)

<img width="627" alt="image" src="https://github.com/user-attachments/assets/63bcb0e1-6bd6-495d-ba65-d1ae640002ce" />

## 1. 신호의 기저 표현
- 신호 \(x[n]\)은 기저 신호 $\(\psi_k[n]\)$를 사용하여 표현할 수 있습니다:$\[x[n] = \sum_{k} a_k \psi_k[n]\]$
- **기저 신호** $\(\psi_k[n]\)$
  - 신호를 구성하는 빌딩 블록
  - 예: 푸리에 변환에서 복소 지수 함수

## 2. 예제: 이산 푸리에 변환 (Discrete Fourier Transform, DFT)
- DFT의 기저 신호:$\[\psi_k[n] = e^{j \frac{2\pi k}{N} n}\]$
- 신호 \(x[n]\)의 표현:$\[x[n] = \sum_{k=0}^{N-1} X[k] e^{j \frac{2\pi k}{N} n}\]$
  - \(X[k]\): 주파수 성분의 계수.

## 3. 목표: 저차원 표현 (Low-Dimensional Representation)
- 기저를 사용하여 신호를 저차원(소수의 비영 계수)으로 표현하는 것이 목표입니다
  - 관심 있는 신호의 표현을 간결하게 유지
  - 비영 계수 \(a_k\)의 수를 줄임

## 4. 응용
1. **압축 (Compression)**
   - 작은 계수 $\(a_k\)$와 관련된 성분 제거
   - 데이터 크기를 줄이면서 정보 유지

2. **필터링 (Filtering)**
   - 신호와 관련된 $\(a_k\)$를 유지하고, 잡음과 관련된 $\(a_k\)$를 제거

3. **간섭 제거 (Interference Reduction)**
   - 간섭이 포함된 $\(a_k\)$를 0으로 설정하여 제거


# 기저(basis)의 종류와 선택

<img width="626" alt="image" src="https://github.com/user-attachments/assets/dae378ee-2773-4599-bcb0-b68391023e1a" />

## 1. 기저의 최적성에 대한 개요
- **보편적으로 최적화된 기저는 존재하지 않습니다!**
  - 각 기저는 특정 응용 분야나 데이터의 특성에 따라 적합성이 다릅니다

## 2. 기저의 분류 (Categories of Bases)

### (1) 고정 또는 결정적 기저 (Fixed or Deterministic Bases)
- 주어진 주파수를 가지는 사인파(sinusoids)와 웨이블릿(wavelets)과 같은 사전 정의된 기저
- **대표적인 예**
  - **이산 푸리에 변환(DFT)**
    - 주파수 기반 신호 분석에 사용

### (2) 데이터 적응형 기저 (Data-Adaptive Bases)
- 데이터의 통계적 특성을 활용하여 최적의 기저를 학습
- **대표적인 기법**
  - **주성분 분석(PCA)**: 데이터 분산을 최대화하는 기저를 찾음
  - **정준 상관 분석(CCA)**: 두 데이터 세트 간의 상관성을 분석
  - **독립 성분 분석(ICA)**: 독립적인 신호 성분을 분리

### (3) 매개변수 기반 기저 (Parametric Bases)
- 미지의 매개변수에 의존하는 함수형 기저
- **예**
  - 주파수가 알려지지 않은 사인파
  - 시작 시간이 알려지지 않은 임펄스 신호

# 선형 독립성과 신호 표현

<img width="648" alt="image" src="https://github.com/user-attachments/assets/eae3286f-5cd3-4140-b49f-5db9c2fb9e6d" />

## 1. 기저 신호의 정의
- 기저 신호 $\(\psi_k[n]\)$이 주어졌을 때, 신호 \(x[n]\)은 다음과 같이 표현됩니다:$\[x = \sum_{k=1}^{N} a_k \psi_k = \Psi a\]$
  - \(x\): $\(N \times 1\)$ 벡터 (신호)
  - $\(\Psi\)$: $\(N \times N\)$ 행렬 (기저 신호를 열로 구성)
  - \(a\): $\(N \times 1\)$ 계수 벡터

## 2. 선형 독립성
- **기저 신호가 선형 독립일 경우**
  - \(N\)개의 기저 신호만으로 모든 $\(N \times 1\)$ 신호를 표현할 수 있습니다
  - 행렬 $\(\Psi\)$는 **가역(Invertible)**입니다

## 3. 신호 표현과 복원
1. **계수 추정**:$\[a = \Psi^{-1} x\]$
   - 주어진 신호 \(x\)로부터 계수 \(a\)를 계산

2. **신호 복원**:$\[x = \Psi a\]$


# 중요한 사례: 직교 기저 (Orthogonal Bases)

<img width="640" alt="image" src="https://github.com/user-attachments/assets/ba56e3f6-cd10-4c7b-99d8-9acbf8068e06" />


## 1. 직교 기저의 정의
- 기저 신호 $\(\psi_k[n]\)$가 직교할 경우: $\[\sum_n \psi_k[n] \psi_\ell^*[n] = 0 \quad \text{(if \(k \neq \ell\))} \]$
  - $\[\psi_k^H \psi_\ell = 0 \quad \text{(if \(k \neq \ell\))} \]$
  - $\(\psi_k^H\)$: $\(\psi_k\)$의 켤레 전치 (Hermitian Transpose)

## 2. 계수 추정과 신호 표현
1. **행렬 표현**:$\[a = \Psi^{-1} x\]$

2. **계수 계산**:$\[a_k = \frac{1}{\psi_k^H \psi_k} \psi_k^H x\]$
  - 이는 다음과 같이 전개됩니다:$\[a_k = \frac{1}{\psi_k^H \psi_k} \sum_{n=0}^{N-1} \psi_k^*[n] x[n]\]$

## 3. 예제: 이산 푸리에 변환 (DFT)
- DFT의 기저 신호:$\[\psi_k[n] = e^{j\frac{2\pi k}{N} n}\]$

- 직교 조건: $\[ \psi_\ell^H \psi_k = \sum_{n=0}^{N-1} e^{-j\frac{2\pi \ell}{N} n} e^{j\frac{2\pi k}{N} n} = \begin{cases} 0, & \ell \neq k \\ N, & \ell = k \end{cases} \]$

- DFT 계수 계산: $\[ a_k = \frac{1}{N} \sum_{n=0}^{N-1} e^{-j\frac{2\pi k}{N} n} x[n] \]$
