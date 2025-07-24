# Self-Training for End-to-End Speech Recognition
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### Self-Training  
- unlabeled data에 대해 pseudo-label(가짜 정답)을 생성하고, 그 데이터를 마치 labeled data처럼 다시 학습에 사용하는 **semi-supervised learning 기법**
- Why?
  - 대규모 speech 데이터의 transcribe는 비용과 시간이 많이 듬
  - unlabeled audio는 많지만 label 부족 → 이를 효율적으로 활용하기 위한 방법
#### Semi-Supervised Learning (SSL)
- labeled data는 소수, unlabeled data는 다수 있는 상황에서 학습
- Self-Training은 SSL의 대표적인 방법 중 하나

### 주요 기여
- 강력한 baseline acoustic model + language model로 pseudo-label 생성
- seq2seq 모델의 특유 오류에 특화된 filtering 방식 제안
- pseudo-label 다양성을 위한 novel ensemble 방법 제안
  
### 주요 실험 결과(LibriSpeech corpus 기준)
- noisy speech: WER 33.9% 개선
- clean speech: baseline 대비 oracle과의 gap의 59.3% 회복
- 기존 방법보다 최소 93.8% 상대적 성능 우위


<br>  
  
## 1. Introduction
### ASR 시스템의 데이터 의존성
- ASR 시스템을 구축하기 위해서는 **대규모 전사 데이터 필요**
- 특히 end-to-end 모델은 학습 데이터가 줄어들 경우 성능이 급격히 저하되는 특성 존재
  - 데이터 의존성이 더욱 큼

### SSL의 필요성
- 대규모 음성 데이터를 사람이 직접 전사하는 과정은 비용과 시간 소모가 매우 큼
- unlabeled (또는 unpaired) audio와 text 데이터를 활용하는 다양한 준지도 학습 기법이 연구되고 있음

### 본 연구의 기여
- **강력한 Baseline 모델 기반 Pseudo-label 생성**
  - 소량의 레이블링된 데이터셋으로 acoustic model을 학습
  - 대규모 텍스트 코퍼스로 훈련된 language model(LM)과 결합해 안정적인 decoding으로 pseudo-label 생성
- **Filtering 메커니즘 제안**
  - seq2seq 모델에서 흔히 발생하는 반복(looping), 조기 종료(early stopping) 현상을 감지하고 제거하는 휴리스틱 기반 필터링과 신뢰도(confidence) 기반 필터링을 설계
- **Pseudo-label 다양성 확보를 위한 앙상블 전략**
  - 여러 모델로부터 다양한 pseudo-label을 생성하고 학습에 활용하여 모델이 noisy label에 **과도하게 확신(over-confident)** 하는 현상을 완화

### 주요 실험 결과
- **Noisy Speech Setting**
  - self-training으로 WER **33.9% 상대적 개선**
- **Clean Speech Setting**
  - baseline과 oracle 간 gap의 **59.3% 회복**



<br>  
  
## 2. Model
- 본 논문에서 사용된 시퀀스-투-시퀀스(sequence-to-sequence) 모델은 어텐션(attention) 메커니즘을 포함하는 인코더-디코더(encoder-decoder) 아키텍처를 기반으로 함
### 입력 및 출력 정의
- 입력 : $`X = [X_1, X_2, …, X_T]`$
  - 음성의 frame 시퀀스 (예: Mel-spectrogram)
- 출력 : $`Y = [y_1, y_2, …, y_U]`$
  - 대응되는 텍스트 (예: 문자/word-piece 시퀀스)

### Encoder
- $`[K\ V]= \text{encode}(X) \quad (1)\`$
- 입력 음성 프레임 $`X = [X_1, \ldots, X_T]`$를 키(keys) $`K = [K_1, \ldots, K_T]`$ 와 값(values) $`V = [V_1, \ldots, V_T]`$ 로 구성된 은닉 표현(hidden representation)으로 변환
- TDS(Time-Depth Separable) 블록을 사용하는 완전 컨볼루션(fully convolutional) 인코더를 사용
  - **Fully convolutional 인코더**
    - RNN 없이 전체를 convolution 연산만으로 구성한 인코더
    - 병렬처리가 가능해서 학습이 빠르고 효율적임
    - But, 일반 1D convolution 연산은 **시간 축(time)** 과 **채널 축(depth)** 을 한꺼번에 처리하기에 연상량이 크고 비효율적일 수 있음
  - **TDS (Time-Depth Separable)**
    - **시간(time)** 과 **깊이(depth)** 를 분리해서 처리하는 convolution 구조
    - ex) 시간 축 방향으로만 합성곱 수행 -> 각 시간 위치에 대해 채널 간 정보 추출 -> 채널간 feature mixing ...
    - 장기 의존성 부족할 수 있음

### Decoder
- $`Q_u = \text{RNN}(y_{u-1}, Q_{u-1})`$
  - RNN을 사용하여 이전 토큰 $`y_{u-1}`$ 과 쿼리 벡터(query vector) $`Q_{u-1}`$ 를 인코딩해 다음 쿼리 벡터 $`Q_u`$ 생성
- $`S_u = \text{attend}(Q_u, K, V)`$
  - Q,K,V 어텐션 메커니증을 통해 요약 벡터(summary vector) $`S_u`$ 생성
  - $`\text{attend}(K, V, Q) = V \cdot \text{softmax}\left( \frac{1}{\sqrt{d}} K^T Q\right)`$
     - 여기서 \(d\)는 키 \(K\), 쿼리 \(Q\), 값 \(V\)의 은닉 차원(hidden dimension)
     - $`\sqrt{d}`$로 나누는 것은 스케일링을 위한 것 
- $`P(y_u | X, y_{u}) = h(S_u, Q_u)`$
  - 최종적으로 요약 벡터 $`S_u`$와 쿼리 벡터 $`Q_u`$를 사용하여 출력 토큰(output tokens)에 대한 확률 분포 $`P(y_u | X, y_{u})`$를 계산

### Inference - Beam Search를 이용한 추론
- 모델이 새로운 음성 입력에 대해 적합한 텍스트 전사를 찾아내는 추론 과정
- **음향 모델(AM)** 과 **외부 언어 모델(LM)** 을 조합하여, **beam search** 알고리즘을 통해 가장 높은 점수를 받는 텍스트 가설 $`\bar{Y}`$ 선택
  - $`\text{P}_{\text{AM}}`$: 음성 특징(\(X\))으로부터 텍스트(\(Y\))의 확률을 계산하는 모델
  - $`\text{P}_{\text{LM}}`$: 텍스트 시퀀스(\(Y\))의 언어적 자연스러움을 평가하는 외부 언어 모델
- $`\bar{Y} = \text{argmax}_Y \log P_{AM}(Y | X) + \alpha \log P_{LM}(Y) + \beta|Y|`$
  - $`\bar{Y}`$ : 최종 선택된 전사 결과 (가장 높은 점수를 받은 텍스트 시퀀스)
  - $`\arg\max_Y`$ : 가능한 모든 \(Y\) 중 점수를 최대화하는 것을 선택
  - $`\log P_{\text{AM}}(Y|X)`$
    - 음성 특징 \(X\)가 주어졌을 때 텍스트 시퀀스 \(Y\)를 생성할 로그 확률
    - 입력 음성이 텍스트와 얼마나 잘 일치하는지
  - $`\log P_{\text{LM}}(Y)`$
    - 외부 언어 모델(PLM)이 예측한 확률
    - 문법적으로, 의미적으로 자연스러운 문장인지
  - $`\alpha`$ : 언어 모델 가중치 (LM weight) - 클수록 언어 모델의 영향 증가
  - $`\beta |Y|`$ : 토큰 삽입 항
    - \(|Y|\)는 텍스트 시퀀스 \(Y\)에 포함된 토큰(예: 단어 또는 서브워드)의 개수
    - $`\beta`$는 가중치, 조기 종료 문제를 방지하기 위해 사용
      - 전사된 시퀀스의 길이에 비례하여 점수를 조정하여, 모델이 너무 짧은 전사를 선호하는 경향을 완화 


### Inference - 조기 종료 문제 방지
- 특정 확률 조건을 만족할 때만 문장 종료(EOS: End-of-Sentence) 토큰을 제안
- $`\log P_u(\text{EOS} | y_{u}) &gt; \gamma \cdot \max_{c \neq \text{EOS}} \log P_u(c | y_{&lt;u})`$



