# Self-training and Pre-training are Complementary for Speech Recognition
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



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
