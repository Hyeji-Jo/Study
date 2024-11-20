- 크게 3가지 대분류 존재
1. 신호 변형(Signal-Level Augmentation)
  - 원본 오디오 신호를 직접 변형
  - ex) 잡음 추가, 시간 축 변화(Time Stretching), 피치 변환(Pitch Shifting), 볼륨 조정 등
2. 스펙트럼 변형(Spectrum-Level Augmentation)
  - STFT, Mel-spectrogram 등 주파수 변환 후의 데이터를 수정
  - ex) Time Masking, Frequency Masking, SpecAugment
3. 데이터 믹싱(Data Mixing) 혹은 Split
  - 여러 신호를 혼합하거나 새로운 데이터셋 생성
  - ex) Mixup, Noise Injection, Room Impulse Response (RIR) 적용
- Train 데이터에서만 사용해야함

### 피치 변환(Pitch Shifting) - 중요(큰 도움이 됨)
```py
def change_pitch(data, sr):
    y_pitch = data.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), sr, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)
    return y_pitch

def waveform_aug(waveform,sr):
  y = change_pitch(waveform, sr)
  fig = plt.figure(figsize = (14,5))
  librosa.display.waveplot(y, sr=sr)
  ipd.display(ipd.Audio(data=y, rate=sr))
  return y, sr
```

### Amplitude 변환
- 입력 오디오 신호의 진폭을 임의로 조정하여 다양한 음량(볼륨) 조건을 모사
- 이 함수는 모델이 다양한 음량 조건에서도 견고하게 학습할 수 있도록 데이터를 변형
```py
def value_aug(data):
    y_aug = data.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    y_aug = y_aug * dyn_change
    return y_aug
```

### Noise 추가
- 입력 오디오 신호에 랜덤한 잡음을 추가하여 노이즈가 포함된 데이터를 생성
- 모델이 다양한 노이즈 환경에서도 잘 작동할 수 있도록 학습 데이터를 증강
```py
def add_noise(data):
    noise = np.random.randn(len(data))
    data_noise = data + 0.005 * noise
    return data_noise
```

### 멜로디와 리듬 분리
- 오디오 신호를 하모닉(Harmonic) 성분과 퍼커시브(Percussive) 성분으로 분리
  - 하모닉 성분
    - 시간이 지나도 주파수가 일정하게 유지
    - 주로 음높이(pitch)가 있는 악기의 멜로디나 화음 
  - 퍼커시브 성분
    - 시간이 짧고 주파수 변화가 급격한 성분
    - 주로 타악기나 리듬 
- 멜로디와 리듬을 개별적으로 분석하거나 강조하는 데 유용
```py
# # 보통 harmonic part만 사용
def hpss(data):
    y_harmonic, y_percussive = librosa.effects.hpss(data.astype('float64'))
    return y_harmonic, y_percussive
```

### 신호를 이동시키는 것
- 입력 오디오 신호를 시간적으로 이동(Shift) 시키는 것
- 모델이 시간적 변동에 강건한(robust) 성능을 가지도록 도와줌
```py
def shift(data):
    return np.roll(data, 1600) #-> 오른쪽으로 1600 샘플만큼 이동 / 양수는 오른쪽, 음수는 왼쪽
```
![image](https://github.com/user-attachments/assets/0592ec6b-2666-409c-864e-1709787c3baf)


### 오디오 신호 조절 - 중요(큰 도움이 됨)
- 입력 오디오 신호를 시간적으로 늘리거나 줄이는(Time Stretching) 작업을 수행
```py
def stretch(data, rate=1):
    input_length = len(data)
    streching = librosa.effects.time_stretch(y=data, rate=rate) #-> rate<1 : 신호를 느리게(늘림) / rate>1 : 신호를 빠르게(줄임)
    if len(streching) > input_length: #-> 길이가 길면 앞부분만 자르고, 짧으면 뒤에 0으로 채워 패딩해서 길이 맞춤
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching
```

### 재생속도와 피치 동시 조율
- 오디오 신호의 **재생 속도(speed)** 와 **피치(pitch)** 를 동시에 변경하는 데이터 증강(Data Augmentation) 함수
- 모델이 다양한 속도와 음높이 변화에 강건한 성능을 가지도록 데이터를 변형
```py
def change_pitch_and_speed(data):
    y_pitch_speed = data.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high=1) #-> 0.8~1사이의 랜덤 값 선택
    speed_fac = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac), np.arange(0, len(y_pitch_speed)), y_pitch_speed)
    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]
    return y_pitch_speed
```

## 스펙트럼 변형
```py
train_audio_transforms = torch.nn.Sequential(
      # Mel Spectrogram 생성
      torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
      # Frequency Masking
      torchaudio.transforms.FrequencyMasking(freq_mask_param=15, iid_masks=False),
      # Time Masking
      torchaudio.transforms.TimeMasking(time_mask_param=35, iid_masks=False)
)
```
- **Frequency Masking**
  - Mel Spectrogram에서 특정 주파수 영역을 무작위로 가림
  - freq_mask_param=15: 마스킹할 주파수 대역의 최대 크기
  - iid_masks=False: 독립적으로 마스크를 적용하지 않고 입력에 대해 동일한 마스크 사용
  - 특정 주파수 정보가 사라져도 모델이 다른 주파수 정보를 사용해 학습하도록 유도
- **Time Masking**
  -  Mel Spectrogram에서 특정 시간 구간을 무작위로 가림
  -  time_mask_param=35: 마스킹할 시간 구간의 최대 크기
  -  iid_masks=False: 독립적으로 마스크를 적용하지 않고 동일한 마스크를 사용
