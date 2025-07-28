# Self-training and Pre-training are Complementary for Speech Recognition
## 요약 정리
### Problem
- 음성 인식(ASR) 모델은 높은 성능을 위해 **대규모 labeled data가 필요하지만**, **이는 대부분의 저자원 언어에 대해 현실적이지 않음**
- 최근 Self-training (pseudo-labeling 기반)과 Unsupervised Pre-training (예: wav2vec 2.0)이 각각 효과를 보여주었음
- 그러나 두 방법이 **비슷한 정보를 학습**하는지, 또는 **상호보완적이며 함께 사용할 수 있는지는 미지수**

### Contributions
- **Pre-training (wav2vec 2.0)** 과 **Self-training (pseudo-labeling)** 을 같은 unlabeled data에 대해 결합 실험
- 다양한 labeled data 양 (10min~960h)에서 두 접근법의 조합 효과 분석
- **언어 모델 없이도 self-training이 LM 정보를 distill**할 수 있음을 실험적으로 증명
- Self-training 효과는 labeled data 절대량보다 **unlabeled:labeled 비율에 더 의존함을 분석**

### Method
- **Pre-training**: wav2vec 2.0으로 unlabeled speech에서 representation 학습
- **Fine-tuning**: 소량의 labeled data로 acoustic model 조정
- **Pseudo-labeling**: fine-tuned 모델 + LM으로 unlabeled data에 label 생성
- **Final Training**: pseudo-labeled data와 원래 labeled data로 모델 학습
- **실험: 두 가지 방식 비교**
  - seq2seq 모델을 pseudo-label로 처음부터 학습 (s2s scratch)
  - pre-trained model을 pseudo-label로 fine-tune (ctc ft) 

### Experiments & Setup
- **Labeled data**: LibriSpeech (10min, 1h, 10h, 100h, 960h)
- **Unlabeled data**: LibriSpeech 960h, LibriVox 53k h
- **평가 데이터셋**: test-clean, test-other
- **모델**
  - wav2vec 2.0 LARGE (300M)
  - Transformer LM (20-layer)
  - 최종 모델: 36-layer Transformer seq2seq
- **Decoding**: 4-gram LM + Transformer LM rescoring 

### Results
- 10min labeled + LibriVox unlabeled
  - WER 2.8% (clean), 4.8% (other) ← 기존 960h supervised 모델 수준
  - wav2vec 2.0 단독 대비 최대 **40% 상대 성능 향상**
- 960h labeled 사용 시: WER 1.5% / 3.1% (SOTA 수준)
- 언어 모델 없이도 self-trained 모델은 WER 6.5% 유지 (vs. 38.7% for baseline)
  - pseudo-label에 포함된 LM 정보가 모델에 distillation됨
- 성능 개선은 **labeled 절대량보다는 labeled:unlabeled 비율에 따라 결정**

### Limitations
- 대부분의 실험은 **영어 기반 데이터셋(LibriSpeech, LibriVox)에만 한정됨**
- pseudo-label filtering, iterative self-training 등 세부 기법은 생략하거나 **단일 설정만 사용**
- 모델 크기가 크고, 학습 및 디코딩에 **연산 자원이 많이 소모됨**

### Insights & Idea
- **Self-training과 Pre-training은 겹치는 게 아니라 서로 보완** → 함께 쓸 때 가장 효과적
- **pseudo-label 생성 시 strong LM**을 활용하면, final model에 LM 정보를 distill 가능 → inference 시 LM 없이도 높은 성능
- 초저자원 환경에서도 실용 가능한 ASR 구축 가능성 제시
  - 예: 10분 labeled data로도 SOTA 수준에 근접 
- 실제 저자원 언어 확장, 학습 비용 절감에 실질적 기여 가능

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
#### wav2vec 2.0 구조
- **Convolutional Feature Encoder f: X → Z**
  - 입력: Raw audio waveform (X)
  - 출력: Latent representations (z₁, …, z_T)
    - 각 z_t는 약 25ms 오디오 단위이며, stride는 20ms
    - 음성을 일정 구간으로 분할하여 표현 
- **Transformer Encoder g: Z → C**
  - 입력: Latent z 시퀀스
  - 출력: Context representations (c₁, …, c_T)
    - BERT-style Transformer 구조
    - 시간적 문맥 정보 반영 
- **Quantization Module Z → Q**
  - 학습 시 z_t를 discrete token q_t로 변환
  - 방식
    - Gumbel-Softmax 사용
    - 2개의 codebook(G=2), 각 320개 항목(V=320)
    - 선택된 두 항목을 concat하여 q_t 생성
      - q_t는 contrastive learning의 target 역할

#### 학습 방식
- 마스크된(masked) Feature Encoder의 출력에 대한 대조 학습(contrastive task) 방식으로 학습
  - 마스킹: 학습 중 일부 z_t 시퀀스를 mask함 (10개 연속 time-step)
  - 목표: Transformer 출력 c_t와 정답 q_t를 가깝게 만들고, 100개의 distractor q̃와는 멀게 학습 
  - 손실 함수: $$\log \frac{\exp(\text{sim}(c_t, q_t))}{\sum_{q̃} \exp(\text{sim}(c_t, q̃))}$$
    - sim(a, b)는 cosine similarity

### 2.2 Self-training Approach
#### Self-training
- 레이블이 없는 데이터를 자동으로 라벨링(pseudo-labeling) 해서 모델을 학습에 활용하는 방법
- **초기 모델 훈련**
  - 사용할 수 있는 소량의 labeled data로 acoustic model을 먼저 훈련함
  - 예: wav2vec 2.0을 fine-tuning해서 기본 성능 확보
- **Pseudo-labeling (의사 라벨링)**
  - 훈련된 초기 모델을 이용해 **unlabeled data에 대해 라벨을 예측함**
  - 이때 **language model도 함께 사용**해서 라벨의 품질을 향상시킴
  - 결과: “가짜” 레이블이 붙은 음성 데이터셋 생성
- **최종 모델 재학습**
  - pseudo-labeled data + 원래의 labeled data를 합쳐서 새 acoustic model을 학습

#### Iterative self-training?
- 일부 기존 연구는 **pseudo-label 생성 → 모델 재학습 → 다시 label 생성**을 여러 번 반복 (multi-round)
- 본 논문은 단 **1**회만 수행 (single iteration)
  - 이유: 실험 효율성을 위해
  - 핵심 목표는 “pre-training과 self-training이 상호보완적인지” 보기 위함이므로, 1회로도 충분하다고 판단
 
#### Pseudo-label filtering (안 한 부분)
- 어떤 연구들은 pseudo-label의 품질을 높이기 위해 **신뢰도 낮은 예측을 제거하기도 함**
- **본 논문에서는 적용하지 않고 future work로 남겨둠**

### 2.3 Combining the two Approaches
#### 제안된 결합 방식
- **Pre-training**
  - wav2vec 2.0 모델을 unlabeled data에 대해 사전학습
- **Fine-tuning**
  - 소량의 labeled data로 pre-trained 모델을 fine-tuning
  - 즉, 기존 self-training 방식에서 초기 모델로 사용하던 supervised-only 모델을, pre-trained 모델로 교체
- **Pseudo-labeling**
  - fine-tuned 모델을 사용해 unlabeled data에 대해 pseudo-label 생성
  - labeled data 없이 새로운 학습 데이터셋 생성
- **Final model training**
  - 생성된 pseudo-labeled data + 기존 labeled data를 합쳐서 최종 ASR 모델을 학습
 
#### 다른 실험 설정
- 위와는 별도로, pre-trained 모델을 직접 pseudo-labeled data에 fine-tune하는 방식도 실험함
- 기존 labeled data를 사용하지 않고, **pseudo-labeled data만으로 fine-tuning**
  - labeled data가 전혀 없는 언어 환경을 시뮬레이션하려는 목적
 


<br>  
  
## 3. Experimental Setup
### 3.1 Datasets
- **Unlabeled data (pre-training + self-training에 사용)**
  - LibriSpeech (LS-960): 960시간 분량, 텍스트 제거 후 사용
  - LibriVox (LV-60k): 약 53,200시간, Libri-light 기준 전처리
- **Labeled data 설정 (5가지)**
  - 10분 (train-10min)
  - 1시간 (train-1h)
  - 10시간 (train-10h)
  - 100시간 (train-clean-100)
  - 960시간 (전체 LibriSpeech)
- **평가 데이터셋**
  - dev-clean, dev-other, test-clean, test-other

### 3.2 Pre-trained Models
- 모델은 모두 **fairseq** 라이브러리 기반
- **wav2vec 2.0 LARGE** 모델 사용
  - **24-layer Transformer (BERT 구조)**
  - hidden dim = 1024, FFN dim = 4096
  - 총 약 3억 파라미터
- Convolutional encoder 구성
  - 7개 layer, stride = (5,2,2,2,2,2,2), receptive field = 25ms
- Fine-tuning: CTC loss + 문자 단위 예측

### 3.3 Self-training
- pseudo-labeling에 사용하는 모델: wav2vec 2.0 fine-tuned 버전
- **라벨링 방법**
  - Beam search (beam size=800)로 후보 텍스트 생성 (acoustic model + 4-gram LM)
  - 후보 50개 추려서 **Transformer LM으로 rescoring**
    - rescore: 1차 후보 문장 리스트에 대해 더 강력한 LM으로 점수를 다시 매겨 최종 선택 
- **LM 구성**
  - Transformer LM: 20 layers, dim = 1280, FFN dim = 6144
- 디코딩 파라미터 튜닝
  - LM weight, word insertion penalty 등 랜덤 탐색으로 조절 (128회 샘플링)
 
### 3.4 Final Model
- 최종 모델 아키텍처
  - 입력: log-Mel filterbank feature
  - 전처리: 4-layer convolutional frontend
  - 본체: 36-layer Transformer encoder
    - dim = 768, FFN = 3072, heads = 4
    - 총 약 3억 파라미터
- 출력 vocabulary
  - 10k wordpiece (WP): 전체 960h 사용할 경우
    - wordpiece: 글자보다 크고 단어보다 작은 서브워드 단위. 예: “playing” → “play + ing” 
  - 5k WP: 100h 이하 labeled일 경우
- 디코딩
  - beam size 50
  - 4-gram LM + Transformer LM rescoring 사용
 


<br>  
  
## 4. Result
### 4.1 Low-Resource Labeled Data
- 목표 : labeled data가 매우 적을 때(10분, 1시간, 10시간), **Pre-training, Self-training, 둘의 조합**이 어떻게 다른지 비교
- **실험 조건**
  - unlabeled data: LS-960 or LibriVox (LV-60k)
  - baseline: wav2vec 2.0만 사용한 pre-training 모델
  - self-training 적용 방식
    - s2s scratch: pseudo-labeled data로 seq2seq 모델을 처음부터 학습
    - ctc ft: wav2vec 2.0을 pseudo-label로 fine-tuning
- 주요 결과
  <img width="407" height="462" alt="image" src="https://github.com/user-attachments/assets/dba57263-5ae5-49e4-a319-8e49e6925e45" />

  - Self-training을 추가하면 성능이 대폭 향상됨
  - wav2vec 2.0 단독보다 최대 40% 상대 성능 향상
  - ctc fine-tune > seq2seq from scratch 경향
 
### 4.2 High-Resource Labeled Data
- 100h, 960h 수준의 labeled data가 있을 때, self-training이 여전히 도움이 되는지 평가
<img width="415" height="513" alt="image" src="https://github.com/user-attachments/assets/4ac9e997-b2c9-46ca-8bca-cfad80c8c59f" />

- labeled data가 많을수록 성능은 좋아지지만, self-training의 상대적 효과는 줄어듦
- 결론: self-training은 **low-resource 상황에서 가장 효과적**, high-resource에서도 여전히 소폭 개선 가능

### 4.3 Results without a Language Model
- 언어 모델을 사용하지 않고도 self-training이 효과적인가?
  - 즉, acoustic model 자체가 LM 정보를 내재화했는가?
<img width="417" height="348" alt="image" src="https://github.com/user-attachments/assets/d678196d-9266-409e-9553-b4da492fdb6d" />

- wav2vec 2.0 단독 모델은 **LM 없을 때 성능 급락 (38.7%)**
- 반면, Self-training을 하면 LM 없이도 WER 6.5%로 유지됨
  - 이유: pseudo-label 생성할 때 strong LM을 사용했기 때문에, 그 정보가 self-trained 모델에 distillation됨 (즉, 학습됨)
 



<br>  
  
## 5. Analysis
- Self-training의 효과는 **labeled 데이터 양보다는 비율에 더 좌우됨**
  - 즉, unlabeled:labeled 비율이 일정하면, 개선폭도 비슷
- unlabeled가 많고 labeled가 적을수록 self-training의 힘이 커짐


<br>  
  
## 6. Conclusion
- Unsupervised pre-training(wav2vec 2.0)과 pseudo-label 기반 self-training은 상호보완적

### 주요 기여
- 두 방법의 결합이 **low-resource 환경에서 매우 강력한 성능** 달성
  - 10분 labeled data만으로 과거 960시간 모델 수준 도달
  - WER 2.8% / 4.8% 달성 (test-clean / test-other)
- Self-training은 pseudo-label 생성 시 사용된 LM의 정보를 acoustic model에 distill하는 역할
  - inference 시 LM 없이도 강력한 성능 확보 가능
- 개선 정도는 labeled 양이 아닌 **labeled/unlabeled 비율**에 따라 결정
