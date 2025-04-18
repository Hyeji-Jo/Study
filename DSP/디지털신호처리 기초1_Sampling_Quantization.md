# 0. 개요
## Sound
- 소리는 일반적으로 **진동으로 인한 공기의 압축으로 생성**
- 압축이 얼마나 됬느냐에 따라서 표현되는것이 **wave(파동)**
  - 파동은 진동하며 공간/매질을 전파해 나가는 현상
  - 파동이 전파할 때 **매질은 움직이지 않는다**
  - 파동은 눈으로 볼 수 없다
- 파동을 통해 얻을 수 있는 정보 3가지 (파형그래프를 통해 나타낼 수 있음)
  ![image](https://github.com/user-attachments/assets/b3967b6a-b9b2-4321-babf-b9f2c815d24a)

  - **Phase(Degress of displacement) : 위상**
  - **Amplitude(Intensity) : 진폭(소리의 크기와 관련)**
  - **Frequency : 주파수(소리의 높낮이)**
  
![image](https://github.com/user-attachments/assets/c8cebd45-932d-404f-bf9a-4d44b453c051)

- 물리 음향
  - Itensity : 소리 진폭의 세기
  - Frequency : 소리 떨림의 빠르기
  - Tone-Color : 소리 파동의 모양
- 심리 음향
  - Loudness : 소리 크기
  - Pitch : 음정, 소리의 높낮이/진동수
  - Timbre : 음색, 소리 감각     

## Audio Task
- Sound
  - Sound Classification & Auto-tagging
    - ex) 다양한 기계로부터 수집한 사운드를 통해 어느 위치에 있는지 파악하는 프로젝트 
- Speech
  - Speech Recognition(STT) - 음성인식
    - 음성을 인식하여 나오는 텍스트가 사용될 곳이 많음
    - LAS 모델이 유명
    - 딥러닝 시대 이후로는 End-to-end 모델로 나아가고 있음
  - Speech Synthesis(TTS) - 음성합성
    - Tacotron 모델이 유명 
  - Speech Style Transfer(STS) - 음성변환

## Computer가 소리를 이해하는 과정
- 연속적인 아날로그 신호를 **표본화(Sampling), 양자화(Quantizing), 부호화(Encoding)** 을 거쳐 **이진 디지털 신호(Binary Digital Signal)로 변화**시켜 인식

- **표본화(Sampling)**
  - 샘플링 단계에서 초당 샘플링 횟수를 정하는데, 이를 **Sampling rate**라고 함
  - 1초의 연속적인 시그널을 몇개의 숫자로 표현 할 것인가?
  - 샘플링 레이트가 최대 frequency의 2배 보다 커져야 한다는 것
    - **𝑓𝑠>2𝑓𝑚**  여기서  𝑓𝑠 는 sampling rate, 그리고  𝑓𝑚 은 maximum frequency
    - Nyqusit rate = 2𝑓𝑚
    - **Nyqusit frequency** =  𝑓𝑠/2 , **sampling rate의 절반**
  - 일반적으로 Sampling은 인간의 청각 영역에 맞게 형성됨
    - Audio CD : 44.1 kHz(44100 sample/second)
    - Speech communication : 8 kHz(8000 sample/second)  
- **양자화(Quantizing)**
  - 양자화 단계에서는 amplitude의 real valued를 기준으로 시그널의 값을 조절
  - **Amplitude를 이산적인 구간으로 나누고**, signal 데이터의 amplitude를 **반올림하게** 됨


# 1. Digital Signal Processing
- 목적 : 소리 signal를 어떠한 데이터 타입으로 표현하며, 소리와 관련된 task를 해결

## 데이터 로드 및 확인
```py
# 필요 패키지 설치
!pip install torch
!pip install torchaudio

# 패키지 로드
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchaudio

# 데이터셋 다운로드
test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

# 데이터 확인
test_dataset[1] # (tensor([오디오 샘플 데이터]), 샘플링 주파수, 메타데이터)
# (tensor([0.123, -0.234, ...]), 16000, {'utterance_id': '19', 'speaker_id': '103', 'chapter_id': '5', 'transcription': 'Hello, world!', 'original_text': 'Hello, world!'})

# duration 계산 - 오디오 신호의 재생 시간을 초 단위로 계산 (오디오 샘플 데이터 수 / sampling rate)
audioData = test_dataset[1][0][0]
sr = test_dataset[1][1]
len(audioData) / sr

# 오디오 표시
import IPython.display as ipd
ipd.Audio(audioData, rate=sr)

# 오디오 데이터 시각화
import librosa.display #-> 오디오 신호를 시각화하는 툴
audio_np = audioData.numpy() #-> numpy 배열을 필요로 하기에 바꾸기
fig = plt.figure(figsize = (14,5))
librosa.display.waveshow(audio_np[0:100000], sr=sr) #-> 오디오 신호의 파형 그리기
```
<img width="1443" alt="image" src="https://github.com/user-attachments/assets/e9e50f69-46cc-4de1-b2a4-485d33962e73">


## Resampling
- 샘플링된 데이터를 다시 더 높은 sampling rate 혹은 더 낮은 sampling rate로 resampling 가능
  - 일반적으로 interpolation(보간)을 할때는 low-pass filter를 사용 
- **보편적으로 사용되는 Sampling rate**
  - 96 kHz, 192 kHz: 고해상도 오디오 및 전문 녹음
  - 48 kHz: 비디오 및 프로 오디오에 사용 (영화, 방송)
  - **44.1 kHz**: 음악 및 오디오 파일에 사용 (CD 품질, MP3, 스트리밍 오디오)
  - **22.05 kHz**
  - **16 kHz**: 음성 인식에 사용
  - 8 kHz: 전화 통화에서 사용되는 낮은 샘플링 레이트
```py
import torchaudio
import librosa

# torchaudio를 사용해 Resampling 수행
resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
y_8k = resampler(audioData)  # 텐서 형태로 유지됨
ipd.Audio(y_8k.numpy(), rate=8000)

# librosa로 Resampling 수행
y_8k = librosa.core.resample(audioData.numpy(), orig_sr=sr, target_sr=8000) #torch에서 바로 import 해서 tensor로 되어 있음 -> numpy 붙이기
ipd.Audio(y_8k, rate=8000)
```

## Normalization & Quantization
- 최근에는 보통 Quantization은 하지 않음
  - **대규모 데이터 처리, 실시간 처리, 경량화 모델에서 주로 사용**
- 이산적 구간 나누기
  - **B bit의 Quantization** :  **$-2^{B-1}$ ~ $2^{B-1}-1$**
  - Audio CD의 Quantization (16 bits) : $-2^{15}$ ~ $2^{15}-1$
  - 위 값들은 보통 -1.0 ~ 1.0 영역으로 scaling되기도 합니다
- **정규화**
  - 진폭을 **-1에서 1사이로 조정**
  - 오디오 볼륨이 조절되고, 왜곡이 줄어듬 
```py
# Normalization
audio_np = audioData.numpy()
normed_wav = audio_np / max(np.abs(audio_np)) #-> 오디오 신호/각 샘플의 절대값 중 최댓값(-1에서 1 사이의 값으로 정규화)
ipd.Audio(normed_wav, rate=sr)

# Quantization
## quantization 하면 음질은 떨어지지만 light한 자료형이 된다.
Bit = 8 #-> 비트 수 설정, 양자화를 8비트 정수로 수행하겠다는 의미
max_value = 2 ** (Bit-1) #-> 오디오의 최댓값, 최솟값 설정 (-128,127)

quantized_8_wav = normed_wav * max_value #-> 정규화된 오디오 신호의 각 샘플을 설정 범위로 스케일링
quantized_8_wav = np.round(quantized_8_wav).astype(int) #-> 반올림하여 정수형으로 변환
quantized_8_wav = np.clip(quantized_8_wav, -max_value, max_value-1) #-> 데이터 값의 범위를 넘어가는 값은 잘라내서(clipping) 오버플로 방지
ipd.Audio(quantized_8_wav, rate=sr) #-> 음원에 기계음이 섞여서 들림
## 어떤 샘플 값이 -150이라면, 이 값은 -128로 클리핑 / 어떤 샘플 값이 130이라면, 이 값은 127로 클리핑
```

# 2. Complex Wave
- 우리가 사용하는 대부분의 소리들은 복합파
- 복합파는 복수의 서로 다른 정현파들의 합으로 이루어진 파형
  - 여러 주파수 성분이 합쳐진 복잡한 파형
- 푸리에 변환을 통해 분석 가능
  - 개별 주파수 성분으로 분해하여 파형을 구성하는 각각의 주파수, 진폭, 위상을 파악할 수 있게 함

## Sinusoidal Wave(정현파)
- 일종의 복소 주기함수
- 주기신호를 총칭하는 말
- $\( x(n) \approx \sum_{k=0}^{K} a_k(n) \cos(\varphi_k(n)) + e(n) \)$
  - $\( a_k = \text{instantaneous amplitude} \)$
  - $\( \varphi_k = \text{instantaneous phase} \)$
  - $\( e(n) = \text{residual (noise)} \)$

```py
# 초기 값 설정
A = 0.9 #-> 진폭(Amplitude), 진폭이 클수록 소리의 음향이 커짐
f = 340 # 계이름 '라' #-> 주파수(Frequency) 340 Hz로 설정
phi = np.pi/2 #-> 초기 위상(Phase), 위상이 0이면 정현파가 원점에서 시작하지만 현재 값으로는 사인파가 최대값에서 시작
fs = 22050
t = 1 #-> 신호의 지속 시간
#여러개의 정현파를 합치면 소리의 신호. 반대로 소리에서 정현파를 분리할 수 있다.

# 정현파 신호 생성 함수
def Sinusoid(A, f, phi, fs, t):
    t = np.arange(0, t, 1.0/fs) #-> 시간벡터 생성
    x = A * np.cos(2 * np.pi * f * t + phi) #-> 정현파 신호 생성(코사인 함수 생성)
    return x

# 값 대입
sin = Sinusoid(A,f,phi,fs,t)
ipd.Audio(sin, rate=fs)
```


# 기타 자료 조사
### Phase(위상)
![image](https://github.com/user-attachments/assets/3171f171-a058-4837-aa31-d7798f39235a)

- **동일 주파수**에서 얼마나 어긋나있는가
  - 파동이 처음 시작하는 위치를 기준으로 다른 지점에서 파동이 얼마나 이동했는지 
- 출발 위치를 결정할 수 있는 값
- 위상은 각도로 나타내며, 일반적으로 **도(degree)** 나 **라디안(radian)** 으로 표현
  - 한 주기는 360도(또는 2π 라디안) 
- v1 = sin(wt) -> v2 = **sin(wt-$\theta$)**
  - theta만큼 뒤처지면 -, 빠르면 + 

### Amplitude(진폭)
![image](https://github.com/user-attachments/assets/f861c9a7-1530-4647-949f-6387e45c5bc9)

- **파동의 높이**로 이해할 수 있음
- 소리의 경우 **진폭이 크면 소리가 더 크고, 진폭이 작으면 소리가 작게 들림**
  - 음압이 높아질수록 공기의 진동이 커지기 때문에 사람의 귀에 더 크게 들림
- 파동이 가진 에너지를 나타냄
  - 진폭이 클수록 파동이 전달하는 에너지가 커짐
  - 진폭이 큰 물결은 더 강한 힘으로 해안에 부딪힐 수 있음
- 사인파, 코사인파와 같은 주기적 파동에서는 진폭이 **파동의 최고점과 최저점 간의 거리를 절반으로 나눈 값**
- 진폭의 변화를 통해 감정, 말투, 강세 등을 분석할 수 있음
  - 진폭이 갑자기 커진다면 화를 내는 감정이나 흥분 상태임을 유추가능
- sin(wt) -> **3sin(wt)**

### Frequency(주파수)
![image](https://github.com/user-attachments/assets/079ae6ad-759a-453a-8af8-57456be5c2ea)

- **파동이 1초 동안 반복되는 횟수**
- 보통 **헤르츠(Hz)** 단위 사용
- 주파수가 높을수록 파동이 빠르게 반복되고, 낮을수록 느리게 반복
  - 주파수 **100 Hz의 파동은 1초에 100번 진동**하는 것
- **소리의 높낮이**를 결정하는 중요한 요소
  - 높은 주파수는 높은 음(고음)을 생성, 낮은 주파수는 낮은 음(저음)을 생성
  - 남성의 목소리는 대체로 저주파, 여성의 목소리는 고주파 성분을 더 많이 포함
- 인간이 들을 수 있는 주파수의 범위 : 20 Hz에서 20,000 Hz(20 kHz)까지 - 가청 주파수
- **주파수와 파장은 서로 반비례 관계**
  - 주파수가 높을수록 파장은 짧아지고, 낮을수록 파장은 길어짐
- **주기(period)와 주파수는 역수 관계**
  - **주기 : 파동이 한 번 진동하는데 걸리는 시간**, 또는 길이 
  - 주기 = 1/주파수
  - 일반적으로 **sin** 함수의 주기는 **2$π$/w**
- sin(wt) -> **sin(3wt)**

### Sound와 Speech의 차이
- **Sound**
  - 일반적인 물리적 현상
  - 꼭 **의미를 가질필요 없으며**, 사람, 동물, 물체, 자연현상 등 다양한 원천에서 발생 가능
- **Speech**
  - **의미를 전달하기 위한 사람의 음성 신호**
    - 사람이 특정한 언어를 통해 의사를 표현하기 위해 만들어낸 소리

### 표본화(Sampling)
![image](https://github.com/user-attachments/assets/4fcd0f46-e926-475a-9eb5-1004ff34efba)

- 연속적인 아날로그 신호를 이산적인 디지털 신호로 변환하는 과정
![image](https://github.com/user-attachments/assets/76d19ec2-6968-41a1-8795-5f0385e158db)

- 신호를 측정하는 간격 = Sampling rate
  - 높을수록 음질이 더 좋다
- **나이퀴스트 샘플링 이론(Nyquist Sampling Theorem)** 에 근거
  - 원본 신호의 최대 주파수의 2배 이상이어야 함
  - aliasing 방지를 위한 이론
![image](https://github.com/user-attachments/assets/b46a981d-92ef-4a3e-bc86-87835a1851e0)
- Sampling rate 값이 너무 작으면 **Aliasing** 문제 발생

### Aliasing
![image](https://github.com/user-attachments/assets/47ae2458-c91a-4bb6-882c-e2bb6d43072c)

- 신호를 샘플링할 때 샘플링 레이트가 충분히 높지 않아 생기는 왜곡 현상
- 샘플링 주파수가 신호의 최대 주파수 성분의 두배보다 낮을 때 발생
- 표본화 과정에서 원신호를 정상적으로 복원하지 못하고 일그러짐이 발생하는것
- 즉, 신호의 왜곡이 발생
- **방지 방법**  
  - **적절한 샘플링 레이트 사용** : 원본 신호의 최대 주파수의 2배 이상 샘플링 레이트를 선택
  - **반올림 필터(Anti-Aliasing Filter)** : 샘플링 전에 신호의 고주파 성분을 제거하는 저역 통과 필터를 사용하는 방법

### 반올림 필터(Anti-Aliasing Filter)
- 신호를 샘플링하기 전에 고주파 성분을 제거하여 에일리어싱(Aliasing)을 방지하는 필터
- 저역 통과 필터(Low-pass Filter)로 구현되며, 샘플링 주파수의 절반 이상인 주파수 성분을 걸러냄
- 반올림 필터는 신호의 고주파 성분을 제거하면서, 신호를 더 스무드하게 만듬
  - 사람이 필요로 하지 않거나 신호 복원에 방해되는 고주파 성분만을 제거하는 것
  - 전체 음질이나 의미를 크게 훼손하지 않도록 설계
  - 그리고 제거 되더라도 보통 인간이 인지하지 못할 정도로 미미한 수준

### 양자화(Quantization)
- 연속적인 아날로그 신호 값을 이산적인 디지털 값으로 변환하는 과정
  - 샘플링된 신호의 각 값을 가까운 이산 값으로 반올림하여 디지털화
  - 컴퓨터는 0과 1로 구성된 이진 데이터인 이산적 데이터만 처리가 가능하기에
- 이 과정에서 샘플링된 값은 특정 비트(bit) 수로 표현됨
  - 샘플링된 값을 이진수로 표현할 수 있는 고정된 비트 수로 변환 
  - 예를 들어 8비트 양자화는 진폭을 256개의 값으로
  - 16비트 양자화는 65,536개의 값으로(-32,768에서 +32,767 사이의 범위)
- **디지털 신호의 장점**
  - 데이터 크기를 줄이고 저장 공간을 절약할 수 있다
  - 아날로그 신호보다 노이즈에 강하고, 안정적으로 전송할 수 있음
    - 아날로그 신호는 시간과 진폭이 연속적인 신호로 모든 값이 연속적으로 변함
      - 그래서 미세한 노이즈나 왜곡이 그대로 반영됨
    - 디지털 신호는 이산적인 0과 1로 구성된 신호로 특정 임계값을 기준으로 해석
      - 임계값을 넘으면 1, 아니면 0으로 해석하기 때문에 작은 노이즈가 있어도 원래값으로 복원 가능
    - 디지털 신호는 오류 검출 및 오류 정정 코드를 추가할 수 있음
- MP3 파일은 디지털 데이터 형식
  - 아날로그 신호를 디지털로 변환하여 압축한 형태 
- **현대적 사용 경향**
  - 최근에는 주로 대규모 데이터 처리나 라이트 네트워크, 실시간 처리가 필요한 경우 사용
  - 모델의 크기와 연산을 줄여주는 장점이 있지만, 고성능이 필수적인 작업에서는 성능 저하를 유발할 수 있기에
  - 양자화는 부동소수점(floating-point) 대신 정수(integer) 연산을 사용하므로 모델 정확도가 약간 저하될 수 있음
    - 부동소수점 연산은 매우 정밀한 숫자 표현을 가능하게 하지만
    - 정수 연산은 표현할 수 있는 숫자의 범위와 정밀도가 제한적, 값을 근사치로 처리
  - 일반적으로 신경망의 가중치나 활성화 함수 출력을 부동소수점으로 표현

### 양자화 오류(Quantization Error)
- 원본 아날로그 신호와 양자화된 디지털 신호 사이의 차이에서 비롯
  - 낮은 비트로 양자화할 때 왜곡이 더 크게 나타남
- 비트 수가 많을수록 진폭의 표현 범위가 더 세밀해져, 원본 아날로그 신호에 가깝게 표현
  - 8비트 오디오 신호는 낮은 해상도를 가지기 때문에 잡음이나 왜곡이 발생하기 쉽고
  - 16비트나 24비트 오디오 신호는 훨씬 더 고음질을 제공
- 왜곡을 줄이기 위해 디더링(dithering)과 같은 기법 사용가능

### Overflow(오버플로)
- 데이터가 표현할 수 있는 최대 범위를 초과하여 값을 저장할 수 없게 되는 상황
- 표현할 수 없는 값이 잘못된 값으로 저장되어 신호의 원래 형태가 왜곡될 수 있음
  - 어떤 신호가 오버플로로 인해 값이 반전되거나, 원래와 다른 값으로 저장될 수 있음
  - 음성 신호의 경우, 원래의 음성 정보가 손실되거나 왜곡된 소리로 변질될 수 있음   

### 정현파 관련 공식
- **A: 진폭, ω0: 주파수(=2πf), Φ: 위상**
- $\( e^{j\theta} = \cos \theta + j \sin \theta \)$
- 삼각함수에 의한 표현: $\( x(t) = A \cos(\omega_0 t + \Phi) \)$
- 복소지수에 의한 표현: $\( x(t) = \text{Re} \{ A e^{j(\omega_0 t + \Phi)} \} = \frac{A e^{j(\omega_0 t + \Phi)} + A e^{-j(\omega_0 t + \Phi)}}{2} \)$ 
