# Self-training and Pre-training are Complementary for Speech Recognition
## 요약 정리
### Problem


### Contributions


### Method


### Experiments & Setup


### Results


### Insights


### Limitations


### Idea



<br>  
  
## 0. Abstract
### 연구 목표 및 배경
- 최근 ASR에서 **Self-training (자기 지도 학습)** 과 **unsupervised pre-training (비지도 사전학습)** 이 주목
  - **Self-training** : labeled data로 학습한 초기 모델을 이용해 unlabeled data에 pseudo-label 생성 → 추가 학습
  - **Pre-training** : 대규모 unlabeled data에서 representation 학습 후, 소량 labeled data로 fine-tuning
  - 두 방법이 **비슷한 패턴을 학습하는지**, 또는 **효과적으로 결합될 수 있는지는 명확하지 않음**

### 제안 및 실험 내용
- **제안 방법**
  - wav2vec 2.0 모델로 unlabeled data에 대해 self-supervised pre-training
  - 소량 labeled data로 fine-tuning
  - fine-tuned 모델을 사용해 unlabeled data에 pseudo-label 생성
  - pseudo-label + labeled data로 최종 모델 재학습
- **실험 데이터**
  - labeled : LibriSpeech (10min, 1h, 10h, 100h, 960h)
  - unlabeled : LibriSpeech 960h, LibriVox 53,000h

### 주요 성과
- 10분 labeled data + 53k시간 unlabeled data 사용
  - WER 2.8% (clean) / 4.8% (other) 달성
  - 이 성능은 1년 전 960시간 labeled data로만 훈련한 최고 성능과 같은 수준
- 960시간 labeled data 사용
  - WER 1.5% / 3.1% 달성 (당시 최고 수준)

### Insight
- Self-training과 pre-training은 겹치는 것이 아니라 **서로 다른 정보를 학습하며 결합하면 효과적**
- 특히 **극소량 labeled data 환경에서 큰 성능 향상**


<br>  
  
## 1. Introduction
### 연구 배경 및 동기
- 최근 음성 인식(ASR)은 **레이블된 음성 데이터**를 이용한 모델 훈련을 통해 크게 발전
- **지도학습 기반 모델은 대량의 labeled data에 의존**, 이는 **영어와 일부 자원 풍부한 언어에만 현실적으로 가능**
  - 세계적으로 약 7,000개의 언어 중 대부분은 labeled data가 부족하여 순수 supervised 학습은 비현실적
- 이로 인해 unlabeled speech data를 효과적으로 활용하는 방법에 대한 연구가 증가

### 기존 접근 방식
- **Self-training (자기지도 학습)**
  - labeled data로 학습한 초기 모델을 이용해 unlabeled data에 pseudo-label 생성
  - 이를 이용해 모델을 재학습 (retrain)
- **Unsupervised Pre-training (비지도 사전학습)**
  - 레이블 없는 음성으로 표현을 학습(pre-train)
  - 그 후 소량의 labeled data로 fine-tuning
 
### 본 논문의 제안
- 본 논문에서는 위 두 가지 접근 방식을 결합
  - Pre-training: wav2vec 2.0 기반으로 representation 학습
  - Self-training: pseudo-label 생성 및 모델 재학습
 
### 실험 설정
- 모델: wav2vec 2.0 + Self-training 방식 (Kahn et al., Xu et al.)
- 전략
  - pseudo-labeled data로 모델을 처음부터 학습하거나
  - pre-trained model을 fine-tuning함
- 데이터 : 동일한 unlabeled data를 사용해 비교 (통제 조건 설정)

### 주요 실험 결과
- 실험 데이터셋: 전체 Librispeech / 소량 레이블 조건의 Libri-light (10분, 1시간 등)
- 결과: self-training과 pre-training은 실제로 상호보완적임을 확인
- 10분 labeled + LibriVox unlabeled: WER 2.8% (clean) / 4.8% (other)
  - pre-training 단독 대비 25%, 40% 상대 성능 향상
- 언어 모델 없이 acoustic model만 사용: WER 3.7% / 6.5%
  - self-training이 **pseudo-label 생성에 사용된 LM의 정보를 모델에 주입(distill)** 했다는 가설을 뒷받침
- 960시간의 labeled data 사용 시: WER 1.5% / 3.1% 달성 (SOTA 수준)


<br>  
  
## 2. Background
### 2.1 Unsupervised Pre-training Model
