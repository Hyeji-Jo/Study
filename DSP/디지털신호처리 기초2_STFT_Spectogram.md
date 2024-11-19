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
- **정의**
  - **시간 영역에서 이산적인 신호를 주파수 영역으로 변환하는 수학적 기법**
  - 주어진 신호를 다양한 주파수 성분으로 분해하여, 신호의 진폭(크기)과 위상 정보를 구함
- **필요성**
  - 디지털 신호 처리는 주로 이산적인 데이터를 다루기 때문에, 연속 푸리에 변환을 사용할 수 없음
  - 이산적인 데이터에서 주파수 성분을 분석하려면 DFT가 필요
- **단점**
  - 계산량이 많아 속도가 느리다는 단점 존재
  - 각 k에 대해 n을 N번 순회하여 곱하고 합산해야함 즉, 모든 샘플 쌍에 대해 계산을 수행해야 하므로 $\[O(N^2)\]$ 시간복잡도를 가짐
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
- 계산량이 많아 속도가 느리다는 단점 존재
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
- **정의**
  - 이산 푸리에 변환(DFT)을 빠르게 계산하기 위한 알고리즘
  - DFT가 가지고 있는 **대칭성과 주기성**을 활용해 불필요한 계산을 줄이고, 신호를 작은 부분으로 나누어 계산하는 **분할 정복(divide and conquer) 방식**을 사용하여 속도를 개선
  - 대규모 데이터나 실시간 신호처리에 적합
- 복잡한 연산을 최적화하여 DFT를 빠르게 수행가능
  - 만약 N개의 샘플을 가진 신호에 대해 DFT를 계산하려면 $\[O(N^2)\]$번의 연산이 필요
  - 그렇듯 신호 길이가 길어질수록 기하급수적으로 증가
  - FFT는 O(NlogN) 시간복잡도 - 긴 신호에 대해서도 빠르게 계산 가능
- **대칭성(Symmetry)**
  - 주파수 성분이 나이퀴스트 주파수를 기준으로 대칭
  - 전체 주파수 성분을 계산할 필요 없이 절반만 계산하면 나머지는 바로 알 수 있음
- **주기성(Periodicity)**
  ![image](https://github.com/user-attachments/assets/088c2b8e-0288-47f6-a8e4-a3d0ebb339f1)
  - 복소수 지수 함수가 주기 N을 가진다는 점을 의미
- **"분할 정복(divide and conquer)"** 방식으로 DFT를 계산
  - 긴 신호를 더 작은 신호로 분할하고, 각 부분에서 계산한 결과를 다시 합치는 방식
- 가장 일반적으로 사용되는 알고리즘 = 쿨리-튜키 알고리즘(Cooley-Tukey algorithm)
  - 보통 크기 n을 재귀적으로 2등분하여 분할 정복을 적용
- 우함수와 기함수의 성질을 활용하여 계산량을 줄이는데 기여
  - 우함수(Even Function) : f(x)=f(−x) ex)cos(x)
  - 우함수 신호의 푸리에 변환은 순수한 코사인 성분으로만 구성
  - 기함수(Odd Function) : f(x)=−f(−x) ex)sin(x)
  - 기함수 신호의 푸리에 변환은 순수한 사인 성분으로만 구성

## STFT(Short-Time Fourier Transform) - 단시간 푸리에 변환
- **정의**
  - 긴 신호를 짧은 시간 구간(윈도우)으로 나누고, 각 구간에 대해 푸리에 변환을 수행하여 **시간-주파수 정보를 동시에 분석하는 기법**
  - 시계열 데이터를 일정한 시간 구간 (window size)로 나누고, 각 구간에 대해서 스펙트럼을 구하는 데이터
  - 푸리에 변환은 신호의 주파수 성분을 분석할 수 있지만 시간 정보를 제공하지 않기 때문에, 신호가 시간에 따라 변화하는 경우 STFT를 사용
    - **일반** 푸리에 변환, DFT, FFT는 주파수 성분이 **언제 발생했는지 알 수 없음**
  - **결과 : 시간의 흐름(Window)에 따른 Frequency 영역별 Amplitude를 반환**
- **수식**
  - $\[X(t, f) = \int_{-\infty}^\infty x(\tau) \cdot w(\tau - t) \cdot e^{-j 2 \pi f \tau} \, d\tau\]$
  - \( x(\tau) \): 시간 영역의 원래 신호
  - \( w(\tau - t) \): **윈도우 함수**로, 신호를 특정 시간 구간 \( t \)에서 자르는 역할
    - 일반적으로 Hann window 사용
    - 윈도우는 일반적으로 1/4 정도를 겹치게 함 
  - \( e^{-j 2 \pi f \tau} \): 푸리에 변환의 복소수 지수 함수로, 주파수 \( f \) 성분을 추출
  - \( X(t, f) \): 시간 \( t \)와 주파수 \( f \)에서의 STFT 결과
- **작동 원리**
  - **신호 분할(윈도우)**
    - 긴 신호를 일정한 길이의 짧은 구간으로 나눔
    - 윈도우 함수(예: Hamming, Hanning, Gaussian 등)를 사용해 신호를 부드럽게 자르기
  - **푸리에 변환 수행**
    - 각 구간(윈도우)에 대해 푸리에 변환을 수행하여 주파수 성분 계산
    - 각 구간에서만 푸리에 변환을 수행하므로, 시간 정보 유지
  - **시간-주파수 결합**
    - 모든 구간의 푸리에 변환 결과를 시간 순서대로 결합해, 신호의 시간-주파수 스펙트럼 생성
- **단점**
  - **시간-주파수 해상도의 트레이드오프**
    - 짧은 윈도우: 시간 해상도는 좋지만 주파수 해상도가 나쁨
    - 긴 윈도우: 주파수 해상도는 좋지만 시간 해상도가 나쁨
  - **고정 윈도우**
    - 모든 구간에서 동일한 윈도우 크기를 사용하므로, 다양한 시간-주파수 스케일을 분석하기 어려움
```py
# 필요 변수
audioData = test_dataset[1][0][0]
sr = test_dataset[1][1]
audioData, audioData.shape
audio_np = audioData.numpy()

# STFT
S = librosa.core.stft(audio_np, n_fft=1024, hop_length=512, win_length=1024)
#-> n_fft:윈도우 크기, Window를 얼마나 많은 주파수 밴드로 나누는가 / hop_length:윈도우가 이동하는 오버랩 간격 / win_length:윈도우 함수의 길이, Window 함수에 들어가는 Sample의 양
S.shape, len(S[0]), S[0][0]
#input shape 중요!
# -> S.shape = ((n_fft/2)+1, N)
## N = (Signal Length - win_length)/hop_length + 1
```
![image](https://github.com/user-attachments/assets/b09de2c1-aca9-4eb7-a6d9-0f2f6c9edd0e)


### 윈도우 Function과 Size는 왜 쓰는 것이며 어떨때 쓰는 것일까요?
- Window function의 주된 기능은 main-lobe의 width와 side-lobe의 레벨의 Trade-off 를 제어해 준다는 장점
  - Main-lobe: 주파수 성분의 중심, 원하는 신호의 정보를 표현
  - Side-lobe: 주변 잡음 성분, 원치 않는 에너지를 표현
- 깁스 현상을 막아줌
  - 신호의 경계에서 발생하는 급격한 변화(불연속성)가 FFT에서 인위적인 진동을 유발 
<img width="1550" alt="image" src="https://github.com/user-attachments/assets/30a2c960-d77e-41fa-921d-225c6575ab08">

- 첫번째 사진처럼 windowing을 적용하기전 plot은 끝부분이 다 다르지만
- 두번째 사진처럼 windowing을 지나고 나서 나오는 plot은 끝이 0 으로 일치한다는 특성

# Spectrogram
![image](https://github.com/user-attachments/assets/06aa5f52-5417-4677-8f74-7dd32d68e1e6)

- **시간에 따라 신호의 주파수 성분이 어떻게 변화하는지 시각적으로 나타낸 그래프**
- 시간, 주파수, 진폭 정보 포함
  - x축-시간 / y축-주파수 / 색상-진폭(색상이 밝을수록 에너지가 높고, 어두울수록 에너지가 낮음)  
- 푸리에 변환을 실시한 결과를 그래프로 나타낸 것

- **추출 과정**
  - 프로세스는 입력신호에 대해서 window function을 통과
  - window size만큼 sampling 된 data를 받아서 푸리에 변환을 거침
  - Frequency와 Amplitude의 영역을 가지는 Spectrum
  - 이후 이를 90도로 회전시켜서, time domain으로 stack
- **Spectrogram의 종류** - Frequency Scale에 대해서 Scaling 진행
  - Linear Spectrogram
    - 전 주파수 대역이 고르게 표현, 주로 신호 처리 연구나 분석에 사용   
  - Log Spectrogram
    - 주파수 축이 로그 스케일로 나타남
    - 저주파 영역을 더 자세히 보여주며, 음성, 음악 신호 분석에 유용 
  - Mel Spectrogram
    - $\[m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)\]$
    - 낮은 주파수에서는 민감하게, 높은 주파수에서는 둔감하게 반응
    - 사람의 청각 감각에 맞춘 Mel Scale로 변환된 Spectrogram
    - 음성 인식, 음향 분석에서 자주 사용 
```py
# 오디오 데이터를 짧은 시간 구간(윈도우)으로 나누고, 각 구간에 대해 푸리에 변환을 수행
## -> S : 복소수 행렬, 진폭과 위상 정보 포함
## 복소수의 크기 = 진폭(Amplitude), 복소수의 각도 = 위상(Phase)
S = librosa.core.stft(audio_np, n_fft=1024, hop_length=512, win_length=1024)

# Phase 정보 제거 및 파워 스펙트럼 계산
## np.abs(S) : 복소수의 크기(진폭) 계산
## 진폭의 제곱을 구해 파워 스펙트럼 생성
## 각 주파수의 성분의 에너지
D = np.abs(S)**2

# Mel 필터 생성
## librosa.filters.mel - Mel 스케일로 변환하기 위한 Mel 필터뱅크 생성
## n_mels - Mel 스케일로 변환한 후의 Mel bin의 개수 (주파수 대역 수)
## 음성 인식 - 40, 음성 합성 - 80, 음악 분석 - 128 주로 사용
## 출력 - (n_mels, FFT bin) 크기의 행렬
mel_basis = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=40)
mel_basis.shape #-> (40, 513) : (n_mels,n_fft/2+1)

# Mel Spectrogram 계산
mel_S = np.dot(mel_basis, D)
mel_S.shape #-> (40, 103)

#-----------------------------------------------------------------

# Mel Spectrogram 계산
## 입력된 오디오 신호를 Mel 스케일로 변환된 스펙트로그램으로 계산
## STFT 변환, 파워 스펙트럼 계산, Mel 필터뱅크 적용을 단일 함수로 적용
S = librosa.feature.melspectrogram(y=audio_np, sr=sr, n_mels=256)

# 데시벨 스케일로 변환
## librosa.power_to_db - Mel Spectrogram을 데시벨(dB) 스케일로 변환
## 에너지가 큰 값과 작은 값 사이의 차이를 로그 스케일로 변환해 쉽게 시각화할 수 있도록 만듦
## 값이 큰 주파수 성분은 더 밝게, 값이 작은 주파수 성분은 어둡게 표시
log_S = librosa.power_to_db(S, ref=np.max)

# 그래프 생성
plt.figure(figsize=(12, 4))
librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel power spectrogram')
plt.colorbar(format='%+02.0f dB')
plt.tight_layout()
```
- 일반적으로 저주파의 소리의 정보가 풍부함
  - 저주파 : 주파수가 낮은 신호, 약 20 Hz ~ 300 Hz가 저주파
  - 대부분의 자연 신호(음성, 음악, 자연소리 등)의 에너지가 저주파 대역에 집중되어 있기 때문
- 저주파는 파장이 길고 에너지가 더 넓게 분산되기 때문에, 먼 거리에서도 잘 전달 

## µ-law Encoding (Mu-law Encoding)
- 오디오 신호의 다이내믹 레인지(dynamic range)를 압축하는 비선형 양자화(non-linear quantization) 기술
- 디지털 오디오 압축에서 사용
- **낮은 진폭의 신호(소리)를 더 정밀하게 표현하고, 높은 진폭의 신호는 덜 정밀하게 표현**
```py
# 함수 정의
def mu_law(x, mu=255):
    return np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)

x = np.linspace(-1, 1, 1000)
x_mu = mu_law(x)

plt.figure(figsize=[6, 4])
plt.plot(x)
plt.plot(x_mu)
plt.show()
```

# 기타
## 윈도우 함수의 예
### 1. Rectangular Window (직사각형 윈도우)
- 신호의 특정 구간만 단순히 잘라냄
- **수식**: $\[w(t) = 1\]$(구간 내에서 일정)

### 2. Hanning Window (해닝 윈도우)
- 경계에서 부드럽게 감쇠
- **수식**: $\[w(t) = 0.5 \left( 1 - \cos\left(\frac{2\pi t}{N-1}\right) \right)\]$

### 3. Hamming Window (해밍 윈도우)
- 경계에서 감쇠 효과가 더 크며, 스펙트럼 누출 감소
- **수식**: $\[w(t) = 0.54 - 0.46 \cos\left(\frac{2\pi t}{N-1}\right)\]$

### 4. Gaussian Window (가우시안 윈도우)
- 중앙에서 부드럽게 강조하고, 점점 감쇠
- **수식**: $\[w(t) = e^{-\frac{1}{2} \left( \frac{t}{\sigma} \right)^2}\]$

## 왜 windowing을 해서 시작과 끝을 0으로 맞추는게 좋은지
- 푸리에 변환(또는 STFT)은 신호가 한 번 반복된 후 무한히 계속 반복된다고 가정
  - 만약 신호의 시작과 끝이 서로 연결되지 않으면, **잘리는 부분에서 "불연속점"** 이 발생
- 즉, 시작과 끝이 자연스럽게 연결되며, 분석 과정에서 생기는 왜곡이 줄어듬
- 경계를 부드럽게 처리해 **깁스 현상과 스펙트럼 누출을 완화**
- 신호를 종이 테이프에 비유하면 쉽게 이해가능

### 깁스 현상(Gibbs Phenomenon)
- 불연속점 때문에 그래프에 불필요한 진동이 생기는 것
- 신호가 갑자기 끊기면, 분석 과정에서 실제 신호에는 없던 **이상한 파형(진동)** 이 추가됨
- 양끝을 0으로 맞춤으로서 해당 문제 해결
### 스펙트럼 누출(Spectral Leakage)
- 신호에 불연속점이 존재하면, 분석 결과에서 주파수 성분이 번져 보이는 현상
- 예를 들어, 원래는 특정 주파수 하나만 강하게 나타나야 하는데, 주변 주파수에도 에너지가 섞여 보임
- 양끝을 0으로 만들어 경계를 부드럽게 하면 이런 "번짐"이 줄어듬

![image](https://github.com/user-attachments/assets/bb7726e2-452c-4c8e-b9dd-bfc97f8bec2e)
