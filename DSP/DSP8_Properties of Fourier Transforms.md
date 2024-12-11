# FT and DTFT Properties(푸리에 변환 및 DTFT의 특성)

## 1. 컨볼루션-곱셈 성질 (Convolution-Multiplication Property)

![image](https://github.com/user-attachments/assets/4b29a926-0dbb-4276-b54e-b32a5b18ea39)

#### 연속 시간 (Continuous-Time)
- 출력 y(t)는 입력 x(t)와 시스템 임펄스 응답 h(t)의 컨볼루션으로 정의
  - $\[y(t) = x(t) * h(t) = \int_{-\infty}^\infty x(\tau) h(t-\tau) d\tau\]$
- 푸리에 변환(FT)을 적용하면 컨볼루션이 곱셈으로 변환
  - $\[Y(\Omega) = X(\Omega) H(\Omega)\]$

#### 이산 시간 (Discrete-Time)
- 출력 y[n]는 입력 x[n]와 시스템 임펄스 응답 h[n]의 컨볼루션으로 정의
  - $\[y[n] = x[n] * h[n] = \sum_{k=-\infty}^\infty x[k] h[n-k]\]$
- DTFT를 적용하면 컨볼루션이 곱셈으로 변환
  - $\[Y(e^{j\omega}) = X(e^{j\omega}) H(e^{j\omega})\]$

### 1) 블록 다이어그램
- 입력 신호 x가 시스템 h를 통과하면 출력 y가 생성
- FT와 DTFT는 시간 영역의 컨볼루션을 주파수 영역의 곱셈으로 변환

### 2) 주파수 응답 (Frequency Response)
- X : 입력 신호의 주파수 성분
- H : 시스템의 주파수 응답 (필터)
- Y : 출력 신호의 주파수 성분
- 주파수 영역에서 $\( X(\Omega) \)$, $\( H(\Omega) \)$, $\( Y(\Omega) \)$의 곱셈 관계로 표현
  - $\( Y(\Omega) = X(\Omega) H(\Omega) \)$
  - 또는 $\( Y(e^{j\omega}) = X(e^{j\omega}) H(e^{j\omega}) \)$

### 3) 그래프 설명
1. 입력 신호 X : 주파수 도메인에서 정의된 신호
2. 시스템 H : 필터 역할을 하는 주파수 응답
3. 출력 신호 Y : 입력 신호와 필터의 곱셈 결과

## 2. Multiplication-Convolution Property (곱셈-컨볼루션 성질)

![image](https://github.com/user-attachments/assets/d4bd4af0-669c-415d-8513-63f91e621464)

### 1) 개요
- 시간 영역에서 신호 곱셈은 주파수 영역에서 컨볼루션으로 변환됩니다
- 주로 **윈도잉(Windowing)** 기법으로 활용됩니다

### 2) 수학적 관계
#### 연속 시간 (Continuous-Time)
- 시간 영역에서 : $\[z(t) = x(t) w(t)\]$
- 푸리에 변환(FT) 적용 : $\[Z(\Omega) = \frac{1}{2\pi} X(\Omega) * W(\Omega)\]$

#### 이산 시간 (Discrete-Time)
- 시간 영역에서 : $\[z[n] = x[n] w[n]\]$
- DTFT 적용 : $\[Z(e^{j\omega}) = \frac{1}{2\pi} \int_{-\pi}^\pi X(e^{j\theta}) W(e^{j(\omega-\theta)}) d\theta\]$
  - 또는 : $\[Z(e^{j\omega}) = \frac{1}{2\pi} X(e^{j\omega}) * W(e^{j\omega})\]$

### 3) 블록 다이어그램
1. 입력 신호 x(t)와 윈도우 함수 w(t)를 곱합니다
2. 출력 z(t)는 시간 영역의 곱셈 결과로 생성됩니다
3. 주파수 영역에서 $\( X(\Omega) \)$와 $\( W(\Omega) \)$의 컨볼루션을 통해 $\( Z(\Omega) \)$를 계산합니다

### 4) 그래프 설명
1. 시간 영역
   - 입력 신호 x(t) : 주파수 성분을 포함한 원 신호
   - 윈도우 w(t) : 신호를 특정 구간으로 제한하는 함수
   - 결과 z(t) : x(t)와 w(t)의 곱
2. 주파수 영역
   - $\( X(\Omega) \)$ : 입력 신호의 주파수 표현
   - $\( W(\Omega) \)$ : 윈도우 함수의 주파수 표현
   - $\( Z(\Omega) \)$ : $\( X(\Omega) \)$와 $\( W(\Omega) \)$의 컨볼루션

### 5) 주요 특징
1. 시간 제한 : 윈도우 함수 w(t)를 통해 신호를 제한된 구간으로 축소
2. 주파수 확장 : 윈도우가 주파수 영역에서 신호를 확장시키는 효과를 가짐


## 3. Time Shift Property (시간 이동 성질)

![image](https://github.com/user-attachments/assets/dc4f309f-7979-4464-93bd-41ea562eafeb)

- 시간 영역에서 신호가 이동하면 주파수 영역에서는 위상이 변화합니다

### 1) 수학적 관계
#### 연속 시간 (Continuous-Time)
- 시간 영역에서 : $\[y(t) = x(t - t_0)\]
- 푸리에 변환(FT) : $\[Y(\Omega) = e^{-j\Omega t_0} X(\Omega)\]$

#### 이산 시간 (Discrete-Time)
- 시간 영역에서 : $\[y[n] = x[n - n_0]\]$
- DTFT : $\[Y(e^{j\omega}) = e^{-j\omega n_0} X(e^{j\omega})\]$

### 2) 블록 다이어그램
1. 시간 영역
   - 입력 신호 x(t)또는 x[n]를 $\( t_0 \)$ 또는 $\( n_0 \)$만큼 이동
   - 출력 신호 y(t) 또는 y[n] 생성
2. 주파수 영역
   - 입력 $\( X(\Omega) \)$ 또는 $\( X(e^{j\omega}) \)$에 지수 함수 $\( e^{-j\Omega t_0} \)$ 또는 $\( e^{-j\omega n_0} \)$를 곱하여 위상 이동

### 3) 그래프 설명
1. 시간 영역
   - x(t) : 원래 신호
   - $\( y(t) = x(t - t_0) \)$ : 시간 $\( t_0 \)$만큼 이동된 신호
2. 주파수 영역
   - 크기 $(\( |X(\Omega)| \), \( |Y(\Omega)| \))$ : 동일
   - 위상 $(\( \arg\{X(\Omega)\} \), \( \arg\{Y(\Omega)\} \))$ : 위상이 $\( t_0 \)$ 또는 $\( n_0 \)$에 비례하여 변화

### 4) 주요 특징
1. **시간 이동**
   - 시간 영역에서 신호의 이동이 주파수 영역에서 위상 변화로 반영
2. **크기 보존**
   - 주파수 스펙트럼의 크기는 변하지 않고, 위상만 변화


## 4. FT Representation for Periodic Signals (주기 신호의 푸리에 변환 표현)

![image](https://github.com/user-attachments/assets/190f1c9d-2ebf-4c64-b982-f673ba08898f)

### 1) 정의
- x(t) : 기본 주기 T를 가진 주기 신호
- 푸리에 변환(FT)을 통해 주기 신호는 다음과 같이 표현됩니다
  - $\[X(\Omega) = 2\pi \sum_{k=-\infty}^{\infty} X[k] \delta(\Omega - k\Omega_0)\]$
    - $\( \Omega_0 = \frac{2\pi}{T} \)$ : 기본 주파수
    - $\( X[k] \)$ : 푸리에 계수

### 2) 푸리에 계수 계산
- 푸리에 계수 X[k]는 주기 신호의 한 주기 동안의 적분으로 계산됩니다
  - $\[X[k] = \frac{1}{T} \int_{T} x(t) e^{-jk\Omega_0 t} dt\]$

### 3) 예제
#### Example : $\( s(t) = \sum_{l=-\infty}^{\infty} \delta(t - lT) \)$
1. **푸리에 계수 계산** : $\[S[k] = \frac{1}{T} \int_{-T/2}^{T/2} s(t) e^{-jk\Omega_0 t} dt\]$
   - s(t)가 단위 임펄스이므로 적분값은 $\( \frac{1}{T} \)$ : $\[S[k] = \frac{1}{T}\]$

2. **푸리에 변환**: $\[S(\Omega) = 2\pi \sum_{k=-\infty}^{\infty} S[k] \delta(\Omega - k\Omega_0)\]$
   - $\( S[k] = \frac{1}{T} \)$ 를 대입하면 : $\[S(\Omega) = 2\pi \sum_{k=-\infty}^{\infty} \frac{1}{T} \delta(\Omega - k\Omega_0)\]$

### 4) 주요 특징
1. **주기 신호의 스펙트럼**
   - 주기 신호의 푸리에 변환은 이산 주파수 성분(델타 함수)으로 구성
2. **푸리에 계수**
   - 푸리에 계수 X[k]는 주기 신호의 주파수 도메인 특성을 나타냄
