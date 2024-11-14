# 푸리에 변환(Fourier transform)
![image](https://github.com/user-attachments/assets/bc4e8fc3-46d3-4248-a2a6-00a2b4d66706)

![image](https://github.com/user-attachments/assets/e5b6cc82-92b7-4adc-915b-1430a2763f3f)

- 임의의 입력 신호를 다양한 주파수를 갖는 주기함수(복수 지수함수)들의 합으로 분해하여 표현하는 것
- 비주기적인 신호(또는 무한히 긴 주기를 가진 신호)를 주파수 성분으로 분석하기 위한 기법
  - 푸리에 급수는 주기적인 신호를 대상으로, 비주기적 신호는 푸리에 변환 
- 그리고 각 주기함수들의 진폭과 위상을 구하는 과정
  - 주기(period) : 파동이 한번 진동하는데 걸리는 시간 또는 그 길이
  - 주파수(frequency) : 1초동안의 진동횟수  

- 푸리에 급수 : $$y(t)=\sum_{k=-\infty}^\infty A_k \, \exp \left( i\cdot 2\pi\frac{k}{T} t \right)$$
  - 주기가 T인 신호를 기본 주파수 $\frac{1}{T}$의 정수배 주파수 성분으로 분해
  - 주파수 성분이 이산적
  - 주파수에서의 신호 성분을 나타내는 복소수 값 : y(t) 
  - 주기 함수들의 수 : $k$는 $-\infty ~ \infty$의 범위
  - 사인함수의 진폭 : $A_k$
  - 회전하는 속도 : $\exp()$ 괄호 안의 값

- 진폭 수식 : $$A_k = \frac{1}{T} \int_{-\frac{T}{2}}^\frac{T}{2} f(t) \ \exp \left( -i\cdot 2\pi \frac{k}{T} t \right)dt$$
  - 각 주파수의 성분이 신호에서 어느 정도의 크기로 기여하는지 나타내는 것
  - 시계방향으로 돌기 때문에 $\exp()$ 괄호 안의 값이 **마이너스**
  - 원형으로 회전하는 복소수 $\exp()$에 f(t)를 곱해 커졌다, 작아졌다를 반복 -> 원에 그래프가 감긴 형태 생성
  - $\frac{1}{T}$을 통해 복소수로서 모두 더한 값을 점의 개수로 나눔 (평균을 구하는것)
  - 주기 함수의 합으로 표현된다고 했는데, 왜 지수함수 형태로 작성되어 있나?
    - 오일러 공식 : $$e^{i\theta} = \cos{\theta} + i\sin{\theta}$$
      - 다른 표현 : $$\exp \left( i\cdot 2\pi\frac{k}{T} t \right) = \cos\left({2\pi\frac{k}{T}}\right) + i\sin\left({2\pi\frac{k}{T}}\right)$$
      - 사인 함수와 코사인 함수를 각각 다루지 않고, 하나의 복소수로 통합(cos:복소수의 실수부, sin:복소수의 허수부)
      - 사인 함수와 코사인 함수는 각도에 따라 독립적으로 변하므로, 방정식이나 계산을 수행할 때 각 함수별 따로 처리해야함
        - 예를 들어, 미분이나 적분을 할 때 따로따로 계산해야 하므로 복잡도 상승
      - 복소 지수 함수를 사용하므로서 미분과 적분이 단순해짐
        - 복소 지수 함수의 미분은 지수가 곱해지기만 하면됨
        - $\[\frac{d}{dt} e^{i \omega t} = i \omega e^{i \omega t}\]$
        - $\[\frac{d}{dt} \cos(\omega t) = -\omega \sin(\omega t)\]$, $\[\frac{d}{dt} \sin(\omega t) = \omega \cos(\omega t)\]$
        - 적분은 지수 형태 그대로 남음
        - $\[\int e^{i \omega t}dt = \frac{e^{i \omega t}}{i \omega}\]$
        - $\[\int \cos(\omega t)dt = \frac{\sin(\omega t)}{\omega}\]$, $\[\int \sin(\omega t)dt = -\frac{\cos(\omega t)}{\omega}\]$
      
- 푸리에 변환 : $\[X(f) = \int_{-\infty}^{\infty} f(t) e^{-i 2 \pi f t} \, dt\]$
  - 주파수 $\( f \)$에서의 신호 성분 : $\( X(f) \)$
  - 시간 영역의 신호 : $\( f(t) \)$
  - 주파수 $\( f \)$에 해당하는 복소 지수 함수 : $\( e^{-i 2 \pi f t} \)$

- 진폭 수식 : $\[X(f) = \int_{-\infty}^{\infty} f(t) e^{-i 2 \pi f t}dt\]$

## DFT(Discrete Fourier Transform)
- 주기가 무한대인 푸리에 변환
  - **Inverse Fourier Transform**: $\[x(t) = \frac{1}{2\pi} \int_{-\infty}^{\infty} X(j\omega)e^{j\omega t} d\omega\]$
  - **Fourier Transform**: $\[X(j\omega) = \int_{-\infty}^{\infty} x(t)e^{-j\omega t} dt\]$
- **sampling한 신호는 시간의 간격과 소리의 amplitude가 모두 descrete한 데이터**
- 주기가 무한대인 푸리에 변환식을 **discrete한 영역으로 변환**해야함
- 수집한 데이터 x[t]에서, 이산 시계열 데이터가 주기 N으로 반복한다고 할때, DFT는 주파수와 진폭이 서로 다른 N개의 사인 함수의 합으로 표현 가능
  - **Inverse Discrete Fourier Transform**: $\[x[t] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j\omega_k n}\]$
  - **Discrete Fourier Transform**: $\[X[k] = \sum_{n=0}^{N-1} x[n] e^{-j\omega_k n}\]$
  - x[t] : input signal
  - n : discrete time index
  - k : discrete frequency index
  - X[k] : k 번째 frequency에 대한 spectrum 값
- 일정한 시간 간격으로 샘플링된 심호를 이산적인 주파수 성분으로 분해
- 시간 신호 x[n]을 주파수 성분 X[k]로 변환
- 계산량이 많아 속도가 느리다는 단점 존재 - O(nlogn)의 시간복잡도
```py
# DFT 함수 정의
def DFT(x):
    N = len(x)
    X = np.array([]) #-> 계산된 주파수 성분(복소수 값)이 배열에 추가됨
    nv = np.arange(N) #-> 0부터 N-1까지의 정수 배열 생성

    for k in range(N):
        s = np.exp(1j*2*np.pi*k/N*nv) #-> 인덱스 k에 해당하는 복소지수함수 s 계산
        X = np.append(X, sum(x*np.conjugate(s))) #-> 입력 신호 x와 복소지수함수 s를 곱한 뒤 그 합을 계산해 X 배열에 추가
    return X
```

## FFT(Fast Fourier Transform)
![image](https://github.com/user-attachments/assets/f2dabd5c-2697-4bf5-84ac-fffd6cfcfdff)

- 복잡한 연산을 최적화하여 DFT를 빠르게 수행가능
  - 만약 N개의 샘플을 가진 신호에 대해 DFT를 계산하려면 $\[O(N^2)\]$번의 연산이 필요
  - 그렇듯 신호 길이가 길어질수록 기하급수적으로 증가
  - FFT는 O(NlogN) 시간복잡도 - 긴 신호에 대해서도 빠르게 계산 가능
- **"분할 정복(divide and conquer)"** 방식으로 DFT를 계산
  - 긴 신호를 더 작은 신호로 분할하고, 각 부분에서 계산한 결과를 다시 합치는 방식
- 가장 일반적으로 사용되는 알고리즘 = 쿨리-튜키 알고리즘(Cooley-Tukey algorithm)
  - 보통 크기 n을 재귀적으로 2등분하여 분할 정복을 적용
- 우함수와 기함수의 성질을 활용하여 계산량을 줄이는데 기여
  - 우함수(Even Function) : f(x)=f(−x) ex)cos(x)
  - 우함수 신호의 푸리에 변환은 순수한 코사인 성분으로만 구성
  - 기함수(Odd Function) : f(x)=−f(−x) ex)sin(x)
  - 기함수 신호의 푸리에 변환은 순수한 사인 성분으로만 구성

## STFT(Short-Time Fourier Transform)
