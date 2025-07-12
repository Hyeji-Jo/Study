# FASTSPEECH 2: FAST AND HIGH-QUALITY END-TOEND TEXT TO SPEECH
https://github.com/ming024/FastSpeech2/tree/master

## 요약 정리
### Problem
- 기존 TTS 시스템의 대부분은 자기회귀(autoregressive) 방식
  - 고품질 음성이 생성 가능하지만 추론 속도가 느리며, robustness 문제 좀재
- **FastSpeech**
  - 비자기회귀(non-autoregressive) 방식으로 속도/robustness 문제를 해결했지만
  - 복잡한 Teacher-Student 학습 구조
  - Teacher output 기반 duration/mel-spectrogram의 정보 손실 → 품질 한계
  - Teacher attention 기반 duration 부정확성 → 리듬/타이밍 품질 저하
- TTS의 근본적인 one-to-many mapping 문제
  - 동일 텍스트도 pitch, energy, duration 등에 따라 여러 스타일로 발화 가능
  - 텍스트만 input으로는 발화의 변이를 충분히 표현하기 어려움 → 모델 과적합 위험
   
### Contributions
- **FastSpeech 2**
  - teacher-student distillation 제거
  - ground-truth mel-spectrogram 직접 사용
  - 정확한 duration(MFA)과 pitch, energy variance 정보 추가 -> **one-to-many mapping 완화**
  - CWT 기반 pitch prediction 설계
- **FastSpeech 2s**
  - 최초의 fully non-autoregressive, text-to-waveform end-to-end 모델
  - mel-spectrogram 단계 제거
  - 완전 병렬 waveform 생성
  - inference latency 대폭 감소
- 풍부한 variance 정보 활용
  - 기존 연구보다 더 fine-grained하고 다채로운 variance 정보(duration, pitch, energy)를 input으로 활용
  - 음질 개선 + 더 자연스러운 prosody 표현  

### Method
1. FastSpeech 2 구조
  - **Encoder → Variance Adaptor(duration, pitch, energy 추가) → Decoder**
  - Duration Predictor : MFA 기반 정확한 phoneme duration 사용
  - Pitch Predictor : CWT 사용 → time-domain pitch contour를 frequency domain에서 안정적 예측
  - Energy Predictor : frame-level energy 예측 → prosody control 개선

2. FastSpeech 2s 설계
  - WaveNet 구조 기반 waveform decoder : non-causal convolution, gated activation, transposed 1D-convolution
  - Parallel WaveGAN-style discriminator : dilated 1D-convolution, Leaky ReLU
  - Loss: multi-resolution STFT loss + adversarial loss
  - mel-spectrogram decoder는 text feature extractor 역할로만 사용

### Experiments
- Dataset: LJSpeech (13,100 samples, 약 24시간)
- Audio Quality (MOS)
  - FastSpeech 2 → Tacotron 2 / Transformer TTS보다 우수
  - FastSpeech 2s → autoregressive 모델 수준의 품질 달성 
- Speed
  - FastSpeech 2 → FastSpeech 대비 3.12배 빠른 학습
  - FastSpeech 2 → Transformer TTS 대비 47.8배 빠른 inference
  - FastSpeech 2s → Transformer TTS 대비 51.8배 빠른 inference
- Variance Information 효과 분석
  - pitch, energy variance 정보 → 모두 음질 개선에 기여
  - CWT 사용 → pitch prediction 안정성, 정확도 개선
  - MFA 기반 duration → alignment 정확도 및 음질 개선
  - mel-spectrogram decoder → FastSpeech 2s의 고품질 waveform generation에 중요  


<br>  
  
## 0. Abstract  
### 1) 연구 배경
- 기존 TTS(Text-to-Speech)의 대표적인 방법은 **자기회귀(autoregressive) 방식** 
  - 품질은 우수하나 추론 속도가 느리고, 반복/누락 등의 문제 발생
- FastSpeech (2019)
  - **비자기회귀(non-autoregressive)** 방식으로 빠른 음성 합성이 가능
  - 하지만 **Teacher 모델에 의존하는 복잡한 학습 구조와 품질 한계 존재**
#### 자기회귀 방식 vs 비자기회귀 방식
1. 자기회귀 방식
  - 텍스트 입력을 받아서 음성을 한 프레임씩 순차적으로 생성
  - **이전에 생성한** 프레임이 다음 프레임 **생성에 조건으로 사용됨**
  - 이 방식은 **품질이 좋지만**, 순차적이라 **속도가 느리고 긴 종속성 문제(robustness issue)** 가 생길 수 있음
2. 비자기회귀 방식
  - 모든 프레임을 **한꺼번에 병렬로 생성**
  - 따라서 빠르고 안정적, GPU 병렬처리에 유리
  
### 2) FastSpeech의 문제점
1. Teacher-Student distillation pipeline
  - 학습 과정이 복잡하고 시간 소모적
2. Teacher로부터 추출한 duration과 mel-spectrogram
  - 정확도가 떨어지고 정보 손실이 발생 → 음질 저하

#### FastSpeech
A. Teacher 모델
  - 텍스트를 입력받아 mel-spectrogram을 autoregressive 방식으로 생성
  - 이 teacher 모델의 attention map으로부터 duration 정보도 추출
B. Student 모델
  - Teacher 모델이 생성한 mel-spectrogram을 학습 target으로 사용
  - Teacher가 만든 duration 정보로 text sequence를 mel-spectrogram length로 늘려줌(length regulator)

  
### 3) 주요 개선점
1. 학습 개선
  - Teacher-Student 구조 제거
  - ground-truth mel-spectrogram 직접 사용하여 학습 단순화 및 품질 개선
    - ground-truth mel-spectrogram = 실제 녹음된 음성에서 추출한 mel-spectrogram (정답 데이터) 
2. One-to-many mapping 문제 완화
  - One-to-many mapping 문제 : 하나의 입력에 여러 개의 올바른 출력이 존재하는 문제
    - ex) 안녕하세요 -> 천천히, 빠르게, 높낮이나 목소리 크기 다르게 
  - Speech의 **변이 정보(Variance Information) 도입**
    - Pitch (음높이)
    - Energy (에너지)
    - Duration (발음 길이) 
  - 이러한 변이 정보를 조건부 입력으로 추가하여 텍스트로부터 다양한 음성 변이를 보다 잘 표현
  
### 4) FastSpeech 2s
- FastSpeech 2의 확장 모델
- 세계 최초로 parallel 방식의 Text-to-Waveform end-to-end 모델
  - Mel-spectrogram 없이 텍스트로부터 바로 waveform 생성
  - 추론 지연(latency) 대폭 감소
  
### 5) 주요 성과
1. 학습 속도
  - FastSpeech 2 → FastSpeech 대비 3배 빠른 학습
  - FastSpeech 2s → 더욱 빠른 추론 속도
2. 음성 품질
  - FastSpeech 2 / 2s 모두 FastSpeech보다 품질 우수
  - FastSpeech 2는 자기회귀 모델도 능가



<br>  
  
## 1. Introduction  
### 1) FastSpeech  
- 기존 TTS 시스템은 대체로 **자기회귀(autoregressive) 방식**
  - mel-spectrogram을 텍스트에서 순차적으로 생성 → vocoder로 waveform 생성
  - 추론 속도가 느리며, robustness(단어 누락 및 반복) 문제 존재
- 비자기회귀 방식의 FastSpeech
  - 병렬적 mel-spectrogram 생성 → 빠름 + robustness 개선
  - 음질은 기존 autoregressive 방식과 유사
  
### 2) FastSpeech의 한계
1. Teacher-Student distillation pipeline
  - 훈련 절차 복잡, 시간 많이 소요
2. Teacher로부터 얻은 mel-spectrogram의 정보 손실
  - 단순화된 output → pitch, prosody 등 세부 정보 부족 → 음질 저하
3. Teacher attention map 기반 duration의 정확성 부족
  - phoneme-align의 부정확성 → 리듬/타이밍 품질 저하
#### Prosody(운율)
- 발화의 억양(intonation), 리듬(rhythm), 강세(stress) 등의 전체적인 패턴을 의미
- pitch + duration + energy = prosody


<br>  
  
## 2. FASTSPEECH2 AND 2S  
### 1) Motivation
#### TTS의 근본적 문제: One-to-Many Mapping
- 이유: 하나의 텍스트 문장도 발화자의 스타일, pitch, duration, energy, prosody에 따라 여러 가지 음성으로 표현 가능
- 같은 문장이라도 빠르게, 느리게, 높게, 낮게, 크게, 작게 다양하게 발음될 수 있음
#### Non-autoregressive TTS의 한계
- **입력 정보가 텍스트뿐** → 발화의 다양한 변이를 충분히 설명할 정보 부족
- **훈련 데이터에 있는 특정 스타일에 과적합(overfitting)할 위험**
- 새로운 상황에서는 일반화 성능 저하
#### FastSpeech의 한계
- 훈련 파이프라인 복잡 (Teacher-Student 구조)
- Teacher가 만든 mel-spectrogram은 **ground-truth보다 정보 손실 발생** → 음질 저하
- Teacher의 attention 기반 duration **부정확 → 음성의 리듬/타이밍 문제**
  
### 2) Model Overview
<img width="176" height="278" alt="image" src="https://github.com/user-attachments/assets/f635529c-c46b-43c9-93dd-ce64a83bbd76" />

#### 아키텍처
1. Encoder
  - 입력된 phoneme embedding sequence → phoneme hidden sequence로 변환
    - 입력: phoneme sequence (예: “a”, “n”, “n”, “y”, “e”, “o”, “n”, “g”, “h”, “a”, “s”, “e”, “y”, “o”)
    - 이 sequence를 학습된 embedding vector로 변환
    - self-attention, convolution 등을 사용해서 phoneme들 사이의 관계(문맥, 의미)를 반영한 hidden sequence로 인코딩
2. Variance Adaptor
  - hidden sequence에 duration, pitch, energy와 같은 variance information을 추가
  - duration predictor, pitch predictor, energy predictor가 여기 포함
3. Decoder
  - 수정된 hidden sequence → mel-spectrogram으로 병렬 변환

#### 기존 FastSpeech와의 차별점
- Teacher output 대신 ground-truth mel-spectrogram을 직접 target으로 사용 → 정보 손실 방지, 품질 상한선(upper bound) 향상
- Teacher의 attention 기반 duration 대신, **Montreal Forced Aligner(MFA)에서 얻은 정확한 duration 사용**
- pitch predictor, energy predictor 추가
  - one-to-many mapping 문제 완화
  - 음질/자연스러움/표현력 향상 

### 3) Variance Adaptor
<img width="291" height="281" alt="image" src="https://github.com/user-attachments/assets/150ea9e3-85a0-44e3-b748-3592a0be0bb7" />

#### Variance Information
1. Duration
  - 각 음소(phoneme)가 얼마나 길게 발음되는지 (타이밍, 리듬)
2. Pitch
  - 음의 높이 → 감정 표현, prosody(운율)에 중요한 요소
    - **pitch = 기본 주파수(F0)** -> 음성이 높거나 낮게 들리는 가장 중요한 acoustic cue
    - 질문형 문장 끝은 pitch가 올라감 / 평서문은 pitch가 일정하게 내려감
    - 자연스러운 발화에서는 pitch contour가 자연스러운 억양 패턴(intonation)을 만들어서 prosody의 핵심 요소 
3. Energy
  - 프레임 단위 mel-spectrogram의 magnitude → 볼륨, 강조, prosody에 영향
  - 발화의 볼륨 및 강세
- 추가적으로 emotion, style, speaker 정보도 확장 가능 (향후 연구 과제로 남김)

#### Variance Adaptor 구성 요소
1. Duration Predictor (Length Regulator)
2. Pitch Predictor
3. Energy Predictor
- 모든 predictor는 유사한 네트워크 구조
  - 2-layer 1D-Convolutional Network + ReLU 활성화
  - Layer Normalization + Dropout
  - Linear Layer → output sequence로 변환 

#### 학습과 추론에서의 동작 방식
- training
  - ground-truth duration, pitch, energy를 hidden sequence에 직접 입력하여 target mel-spectrogram을 예측
  - 동시에 duration/pitch/energy predictor들은 ground-truth variance 값으로 supervised 학습 
- inference
  - predictor들이 duration, pitch, energy를 예측하여 hidden sequence에 추가 → 음성 합성
 
#### Duration Predictor
- phoneme hidden sequence 입력 → phoneme별 duration 예측
  - **duration** = 해당 phoneme이 몇 개의 mel-spectrogram frame에 대응하는지
- 기존 FastSpeech: Teacher 모델의 attention map에서 duration 추출
- FastSpeech 2: **Montreal Forced Aligner(MFA)** 를 사용해 duration 추출
  - MFA - 음성과 텍스트 간의 자동 정렬 도구
  - 음성 녹음 파일과 음성의 텍스트를 입력하면 각 phoneme이나 word가 녹음에서 정확히 언제 시작되고 끝나는지(timing, duration) 표시
  - 유명한 음성 인식 toolkit(Kaldi) 기반으로 만들어졌고, 높은 정확도와 빠른 속도를 자랑
- loss 함수: MSE(mean square error) loss

#### Pitch Predictor
- pitch contour(음높이의 시간에 따른 변화) 예측 → 자연스러운 prosody에 필수
- 기존 방식: pitch contour를 time domain에서 직접 예측 -> **변동성이 큼**
- FastSpeech 2: **Continuous Wavelet Transform(CWT)** 사용
  - **CWT** : 신호(time-domain signal)를 다양한 scale에서 분석해서 time-frequency representation으로 변환하는 방법
  - pitch contour (time-series) → frequency domain의 pitch spectrogram으로 변환하여 예측
  - 예측된 spectrogram → inverse CWT(iCWT)로 다시 pitch contour로 변환
- pitch F0 값은 log-scale에서 256개 bin으로 양자화 → embedding vector로 변환 후 hidden sequence에 추가
- loss 함수: MSE loss

#### Energy Predictor
- 각 frame의 energy(음성 크기/강세)를 예측
  - energy = STFT frame의 amplitude의 L2-norm
  - **STFT** : 긴 신호(예: 음성)를 짧은 window(프레임) 단위로 잘라서 각 구간의 주파수 성분을 분석
- energy 값도 256개 bin으로 uniform quantization → embedding vector로 변환 후 hidden sequence에 추가
- loss 함수: MSE loss
- 양자화된 값이 아닌 원래 energy 값을 직접 예측
  
### 4) FastSpeech 2S
#### 기존 방식의 한계
1. waveform은 mel-spectrogram보다 더 많은 variance 정보(특히 phase)를 포함
  - text → waveform 간 정보 격차 큼
2. waveform 길이가 매우 길어 전체 audio를 한 번에 학습하기 어려움
  - GPU memory 제한 → partial clip 단위로만 학습 가능
  - phoneme 간 긴 dependency 학습 어려움 → text feature extraction에 불리

#### 주요 설계
1. adversarial training 도입
  - Phase 정보 예측은 어렵기 때문에, adversarial loss를 추가하여 decoder가 phase를 implicit하게 복원하도록 유도
  - **Phase** 정보는 mel-spectrogram에 없는 waveform 고유의 중요한 정보
  - Adversarial Training : discriminator(판별자)가 “진짜 waveform과 생성 waveform을 구분”하도록 학습
  - 이 과정에서 waveform decoder는 **phase 정보를 자동으로 잘 복원하도록 압박 받음**
2. FastSpeech 2의 mel-spectrogram decoder 활용
  - mel-spectrogram decoder는 전체 텍스트 시퀀스에 대해 학습됨 -> **전체 문장을 입력으로 받아 좋은 text feature를 추출**
  - waveform decoder가 이 feature에 condition을 걸어 clip 단위로 학습 가능하게 함
  - **mel-spectrogram decoder의 출력은 “중간 hidden feature”이지, mel-spectrogram 자체가 아니다!**
3. WaveNet 구조 기반 waveform decoder 설계
  - non-causal convolution + gated activation
    - non-causal convolution : 현재 timestep뿐 아니라 **양방향 context(과거+미래)를 모두 참조**
    - gated activation : 두 가지 활성화 함수를 결합하면서 중요한 정보를 선택적으로 통과시키는 구조
  - transposed 1D-convolution으로 upsampling → waveform 길이에 맞춤
4. Parallel WaveGAN-style discriminator 사용
  - dilated 1D-convolution으로 멀리 떨어진 dependency까지 잘 처리
    - dilated convolution : convolution filter가 데이터를 **“뛰어넘어서 간격을 두고 보는 방식”**
  - Leaky ReLU activation
5. Loss 함수
  - multi-resolution STFT loss
    - waveform의 time-frequency 특성을 여러 해상도에서 비교 → 더 정교한 waveform 생성 
  - LSGAN discriminator loss
    - adversarial training에서 사용하는 loss
    - generator(=waveform decoder)가 더 “realistic한 waveform”을 만들도록 유도
  
### 5) Discussions
#### 기존 방법과의 차별점
- Deep Voice / Deep Voice 2 등 기존 autoregressive 방식
  - waveform을 순차적으로 생성
  - duration, pitch도 예측했지만 속도 느림
- FastSpeech 2 / 2s
  - Self-Attention 기반 feed-forward 네트워크 사용 → 병렬적 생성 → 빠름
#### 기존 non-autoregressive acoustic model과의 차별점
- 기존 non-AR acoustic model : duration 정확도 개선에 집중
- FastSpeech 2 / 2s : duration + pitch + energy까지 variation 정보를 input으로 활용
- input-output 정보 격차 축소, 더 자연스러운 음성 품질
#### 기존 pitch 예측 방식과의 차별점
- Concurrent work : phoneme-level pitch 예측
- FastSpeech 2 / 2s : frame-level fine-grained pitch contour 예측 + CWT 적용
#### 기존 text-to-waveform 모델과의 차별점
- ClariNet : autoregressive acoustic model + non-AR vocoder를 joint training
- FastSpeech 2s : 완전한 non-autoregressive end-to-end 설계
#### EATS (Donahue et al. 2020)와의 차별점
- EATS : duration 예측에 초점
- FastSpeech 2s : duration뿐 아니라 pitch, energy까지 variation 정보 추가
#### 기존 non-AR vocoder와의 차별점
- 기존 non-AR vocoder : time-aligned linguistic feature → waveform 변환
  - 별도의 acoustic/linguistic model 필요
- FastSpeech 2s :  phoneme sequence → waveform 직접 end-to-end, parallel 방식으로 최초 실현





<br>  
  
## 3. Experiments and Results
### 1) Experimental Setup
#### 데이터셋
- **LJSpeech dataset** 사용 (Ito, 2017) : 13,100개의 영어 오디오 클립 (약 24시간 분량)
- **Training** - 12228 samples
- **Validation** - 349 samples(with LJ003)
- **Test** - 523 samples(with LJ001, LJ002)
- **주관적 평가용 샘플** - Test set에서 100개 샘플 랜덤 선택

#### 입력 데이터 전처리
1. 발음 오류 방지
  - 텍스트 → phoneme sequence 변환
  - open-source grapheme-to-phoneme tool 사용
2. Waveform → mel-spectrogram 변환
  - Shen et al. (2018) 방식
  - Sample rate = 22050 Hz
  - Frame size = 1024
  - Hop size = 256

#### 모델 구성
1. FastSpeech 2 구조
  - Encoder: 4개의 feed-forward Transformer(FFT) blocks
  - Decoder: 4개의 FFT blocks
2. Output layer
  - 80차원 mel-spectrogram 생성
3. Loss 함수
  - Mean Absolute Error (MAE) 사용

  
### 2) Results
#### Model Performance
<img width="382" height="198" alt="image" src="https://github.com/user-attachments/assets/c7cd3ccf-43e8-4e38-82c7-c8909be8166e" />

- **평가 방법**
  - Mean Opinion Score (MOS) 방식
  - 20명의 영어 모국어 화자 참여
  - 동일 텍스트로 시스템별 음질만 비교 
- **비교 모델**
  - GT: 실제 녹음
  - GT (Mel + PWG): GT → mel-spectrogram → Parallel WaveGAN 복원
  - Tacotron 2 (Mel + PWG)
  - Transformer TTS (Mel + PWG)
  - FastSpeech (Mel + PWG)
  - FastSpeech 2 (Mel + PWG)
  - FastSpeech 2s
- **결과 요약**
  - FastSpeech 2: Tacotron 2 / Transformer TTS보다 높은 MOS → autoregressive 모델보다 더 나은 음질
  - FastSpeech 2s: Tacotron 2 / Transformer TTS와 비슷한 수준의 MOS
  - FastSpeech 2는 FastSpeech보다 음질 개선 → variance 정보(pitch, energy, 정확한 duration) 활용 효과 확인

<img width="609" height="118" alt="image" src="https://github.com/user-attachments/assets/96eb7b4d-3ca7-47e2-bb63-15907a346012" />

- Training Speed 비교
  - FastSpeech 대비 3.12배 빠른 학습 시간
- Inference Speed 비교
  - FastSpeech 2 : Transformer TTS보다 47.8배 빠름
  - FastSpeech 2s : Transformer TTS보다 51.8배 빠름

#### Analyses on Variance Information
<img width="343" height="168" alt="image" src="https://github.com/user-attachments/assets/3b321858-0cf9-42d4-9188-522b5f0fbaf8" />

- **Pitch 분석**
  - pitch 분포의 통계(moment) : 표준편차(σ), 왜도(γ), 첨도(K)
  - 평균 Dynamic Time Warping(DTW) 거리
  - FastSpeech 2 / 2s의 pitch moments와 DTW 거리가 ground-truth에 더 가까움
  - FastSpeech보다 더 자연스러운 pitch contour

<img width="343" height="62" alt="image" src="https://github.com/user-attachments/assets/7f958285-768e-4c70-80c1-fb0740f024c5" />

- **Energy 분석**
  - frame-wise energy의 Mean Absolute Error(MAE)
  - FastSpeech 2 / 2s의 energy MAE가 FastSpeech보다 작음
  - ground-truth와 유사한 energy 표현

<img width="559" height="95" alt="image" src="https://github.com/user-attachments/assets/9609b467-3d3e-4380-8ffb-9dec70027323" />

- **Duration 정확도 및 효과 분석**
  - FastSpeech의 teacher model 기반 duration vs. MFA 기반 duration
    - phoneme boundary 차이(align 정확도)
    - CMOS(Comparison MOS) 테스트로 음질 비교
  - MFA 기반 duration이 teacher model 기반보다 alignment 더 정확
  - MFA duration 사용 시 FastSpeech 음질이 개선됨 → MFA duration의 효과 검증
 
#### Ablation Study
<img width="549" height="135" alt="image" src="https://github.com/user-attachments/assets/ec1095cf-0b70-4d71-ae47-f780d4de0a78" />

- Pitch와 Energy Input의 효과
  - Pitch, Energy, 둘 다 제거한 경우에 대해 CMOS(Comparison MOS) 평가
  - **Pitch와 Energy는 모두 성능(음질) 개선에 기여**
  - 특히 FastSpeech 2s에서 Pitch의 기여도가 매우 큼
- CWT 사용의 효과 (Pitch Predictor)
  - CWT 없이 Time-domain에서 pitch contour 직접 예측
  - CMOS 평가 + pitch 통계(moment, DTW) 비교
  - CWT는 pitch modeling 안정성과 정확성 향상에 효과적 → prosody 개선에 기여
- Mel-spectrogram Decoder의 효과
  - FastSpeech 2s에서 mel-spectrogram decoder 제거하여 실험
  - CMOS 감소 : -0.285
  - text feature extraction 품질을 높여 FastSpeech 2s의 고품질 waveform generation에 중요한 역할을 함
 





<br>  
  
## 4. Conclusion  
### 1) FastSpeech 2의 주요 기여
- FastSpeech의 한계(복잡한 학습, 정보 손실 등)를 해결하기 위해 설계
- Ground-truth mel-spectrogram을 직접 사용 → teacher-student distillation 제거 → 정보 손실 방지 + 학습 단순화
- 정확한 duration(MFA 사용) + pitch, energy variance 정보 추가 → one-to-many mapping 문제 완화
- CWT 사용으로 pitch prediction 정확도 개선

### 2) FastSpeech 2s의 기여
- FastSpeech 2 기반으로 발전 → 최초의 fully non-autoregressive text-to-waveform end-to-end 모델
- 추론 속도 대폭 향상
- mel-spectrogram 단계 제거 → pipeline 단순화

### 3) 실험 결과
- FastSpeech 2
  - FastSpeech보다 더 나은 음질
  - Autoregressive 모델(Tacotron 2, Transformer TTS)보다도 우수한 품질
  - 훨씬 빠르고 단순한 학습 pipeline
- FastSpeech 2s
  - FastSpeech 2보다 더 빠른 추론 속도
  - Tacotron 2 수준의 음질 달성
 
### 4) 향후 계획
- 완전히 외부 도구 없이 self-contained end-to-end TTS 실현 (e.g., alignment, pitch 추출 도구 제거)
- 더 다양한 variance 정보 추가 고려 → 음질 향상 + inference 속도 개선
- 경량화된 모델 설계 → 더 빠르고 효율적인 실시간 TTS 가능
