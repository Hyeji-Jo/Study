# Self-Training for End-to-End Speech Recognition
## 요약 정리
### Problem
- End-to-End ASR 시스템은 성능 향상을 위해 대규모의 라벨링된 데이터가 필요
  - 실제 현장에서는 전사 작업의 비용/시간 부담으로 인해 라벨 데이터 수집이 어려움
- 기존의 Semi-Supervised Learning 기법들은 hybrid 모델 중심
  - E2E에서는 복잡한 구조(TTS, cycle-consistency 등)를 요구

### Contributions
- 강력한 acoustic + language 모델 기반 pseudo-label 생성
- seq2seq 모델 특유의 오류(looping, early stopping)를 해결하는 필터링 설계
  - Heuristic 기반 + Confidence 기반 필터링 조합
- pseudo-label 다양성을 확보하는 앙상블(Sample Ensemble) 전략 제안
  - inference가 아닌 학습 데이터 자체를 ensemble  

### Method
#### 전체 구조
1. **Acoustic Model (AM)**: 소량의 labeled data로 학습
2. **Language Model (LM)**: 대규모 unpaired 텍스트로 학습
3. **Pseudo-label 생성**: AM + LM을 결합하여 unlabeled audio에 대해 가짜 전사 생성
4. **Filtering**: noisy pseudo-label 제거 (Heuristic + Confidence 기반)
5. **Ensemble**: 다양한 pseudo-label 생성을 위해 여러 모델 사용 (Sample Ensemble 방식)
6. **재학습**: labeled data + filtered pseudo-label data로 모델 재학습

#### 모델 구조
- **Encoder**: Fully Convolutional 구조 (TDS blocks 사용)
  - 시간(time)과 채널(depth)을 분리해서 처리 → 병렬화 가능, 학습 효율 좋음
- **Decoder**: Attention 기반 RNN 구조
  - 이전 토큰 정보와 encoder 출력에 집중하여 다음 토큰 예측

#### Filtering 기법
- **Heuristic-based**
  - 반복되는 n-gram 제거 (ex. “thank you thank you thank you”)
  - EOS 토큰이 너무 일찍 나오는 조기 종료 가설 제거 
- **Confidence-based**
  - 로그 확률을 시퀀스 길이로 나눈 값 사용
 

### Experiments
#### 데이터 구성
| 유형 | 세트 이름 | 시간 | 용도 |
|------|-----------|------|------|
| Labeled Audio | train-clean-100 | 100h | AM 학습 |
| Unlabeled Audio (Clean) | train-clean-360 | 360h | pseudo-label 생성 |
| Unlabeled Audio (Noisy) | train-other-500 | 500h | noisy 환경 성능 측정 |
| Evaluation | dev/test clean/other | - | WER 측정 |

#### 모델 구조
- **Encoder**: 9개 TDS 블록 (3 그룹: 채널 수 10/14/16, kernel size 21)
- **Target Token**: SentencePiece로 생성된 5,000개 word piece 사용

#### 학습 설정
| 항목 | 값 |
|------|----|
| Optimizer | SGD (no momentum) |
| Initial LR | 0.05 |
| LR decay | every 40/80 epochs ×0.5 |
| Epochs | 200 |
| Baseline | 1 GPU, batch size 16 |
| Self-Training | 8 GPUs |

#### 기타 설정
- Dropout: 20%
- Label smoothing: 10%
- Random sampling: 1%
- Word piece sampling: 1%
- Teacher forcing + soft-window pretraining 적용
- 프레임워크: **wav2letter++**


### Results
#### Filtering의 효과 (WER ↓)
| 환경 | 필터링 없음 | Heuristic | Heuristic + Confidence |
|------|--------------|-----------|------------------------|
| Dev-clean | 6.18% | 6.01% | **5.84%** |
| Dev-other | 24.1% | 21.4% | **18.95%** |

#### Ensemble의 효과
| 앙상블 모델 수 | Dev-clean WER | Dev-other WER |
|----------------|----------------|----------------|
| 1 | 5.84% | 21.86% |
| 4 | **5.41%** | **20.31%** |
| 6 | 5.36% | **18.95%** (+13.7%) |

#### 기존 연구 대비 WER Recovery Rate (WRR)
| 방법 | Test-clean WER | WRR |
|------|----------------|-----|
| Cycle TTE | 21.5% | 27.6% |
| ASR+TTS | 17.5% | 38.0% |
| **본 논문 (Ensemble)** | **9.62%** | **76.2%** |


### Insights
1. **Self-Training은 단순하지만 강력한 SSL 전략**
   - 복잡한 cycle-consistency나 TTS 없이도 높은 성능 확보 가능
   - End-to-End ASR에 효과적임을 실험적으로 입증

2. **Filtering은 필수 구성 요소**
   - pseudo-label은 noisy할 수밖에 없음
   - Filtering 없이는 오히려 성능 저하 가능
   - 특히 noisy 환경에서 필터링 없이는 학습 안정성도 낮음

3. **Sample Ensemble은 고성능을 추론 비용 없이 달성할 수 있는 전략**
   - inference ensemble은 느리고 비용이 큼
   - 학습 시점에서 pseudo-label 다양성 확보 → 더 robust한 모델 생성

4. **실제 적용 가능성**
   - 소량의 labeled data + 대량의 unlabeled 음성 + 텍스트만 있으면  
     → Low-resource 환경에서도 성능 개선 가능
   - 모델 구조도 간단하고, wav2letter++나 ESPnet 등에서 쉽게 재현 가능

5. **후속 연구 아이디어**
   - confidence score를 soft label로 활용 (semi-KD 구조)
   - ensemble된 pseudo-label에 weighting 주기
   - domain adaptation 적용 (noisy → clean transfer)



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
- $`[K\ V]= \text{encode}(X)`$
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
- $`\log P_u(\text{EOS} | y_{<u}) > \gamma \cdot \max_{c \neq \text{EOS}} \log P_u(c | y_{<u})`$
  - $`\log P_u(\text{EOS} \mid y_{<u})`$ : 현재 시점 \(u\)에서 이전에 생성된 토큰 y<u가 주어졌을 때, 문장 종료(EOS) 토큰이 나타날 로그 확률
  - $`\max_{c \ne \text{EOS}} \log P_u(c \mid y_{<u})`$ : EOS를 제외한 다른 모든 토큰 \(c\) 중 가장 높은 확률
  - $`\gamma`$ : 문장 종료 임계값(hyperparameter) / EOS가 다른 토큰보다 **γ배 이상 확신** 있을 때만 허용


<br>  
  
## 3. Semi-Supervised Self-Training
### 지도 학습(Supervised Learning) 설정
- 사용 데이터 : 레이블이 지정된 데이터셋 $`D = \{(X_1, Y_1), ..., (X_n, Y_n)\}`$
- $`\sum_{(X, Y) \in D} \log P(Y \mid X)`$
  - X : 입력 음성 시퀀스
  - Y : 대응되는 텍스트 전사
  - $`\log P(Y \mid X)`$
    - 모델이 \(X\)로부터 \(Y\)를 생성할 확률의 로그값
    - 모델이 Y를 얼마나 정확하게 예측하는지
  - $`\sum`$ : 학습 데이터셋 \(D\)에 있는 모든 음성-전사 쌍에 대해 합산

### 준지도 학습(Semi-supervised Learning) 설정
- **사용 데이터**
  - 레이블이 지정된 데이터셋 \(D\)
  - 레이블이 지정되지 않은 오디오 데이터셋 $`\mathcal{X}\)
  - 짝지어지지 않은 텍스트 데이터셋 $`\mathcal{Y}`$
- **Self-Training 학습 절차**
  - **Acoustic Model (AM)** 학습
    - 레이블 있는 \(D\)로 AM 학습
  - **Language Model (LM)** 학습
    - 짝지어지지 않은 텍스트 $`\mathcal{Y}`$로 LM 학습
  - **Pseudo-label 생성**
    - AM + LM을 조합 (cf. 수식 (6): inference 수식)
    - 레이블 없는 오디오 $`X_i \in \mathcal{X}\)`$에 대해 pseudo-label $`\bar{Y}_i`$ 생성
    - 결과적으로 가상 데이터셋 $`\bar{D} = \{(X_i, \bar{Y}_i)\}\)`$ 생성
  - **최종 모델 학습**
    - 원래의 \(D\)와 가상 데이터 $`\bar{D}\)`$를 함께 학습에 사용
    - $`\sum_{(X,Y) \in D} \log P(Y | X) + \sum_{(X, \bar{Y}) \in \bar{D}} \log P(\bar{Y} | X)`$
     - 첫 번째 항은 지도 학습과 동일하게 정답 레이블에 대한 우도를 최대화
       - **우도(Likelihood)** : 어떤 데이터가 주어졌을 때, 현재 모델이 그 데이터를 얼마나 잘 설명하는지를 나타내는 값
       - 즉, 모델이 정답에 더 높은 확률을 부여하도록 학습하는 것
       - log를 쓰면 곱셈이 덧셈으로 바뀌고 계산이 더 안정적이기 때문 - 여러 시점의 확률을 곱할 때 underflow 방지
     - 두 번째 항은 가상 레이블 데이터셋 \(\bar{D}\)에 대해 가상 레이블 \(\bar{Y}\)의 우도를 최대화


### Filtering
- **필요성** : 유사 레이블 데이터셋 $`\bar{D}`$는 정확하지 않은(noisy) 전사(transcriptions)를 포함 가능
  - noisy label이 포함되면 오히려 모델 성능이 저하될 수 있음 
- **목적** : **데이터의 양과 노이즈 수준 사이의 균형**을 맞춰, 학습 효과를 극대화

#### Heuristic-based Filtering (휴리스틱 기반 필터링)
- 정의: 수학적으로 완벽하진 않지만, **경험적으로 효과 있는 규칙(rule)**
  - 즉, 사람이 보기에 “틀렸다고 확신하진 못하지만 이상해 보여서 빼는” 방식 
- seq2seq 모델의 **전형적인 오류 패턴**을 고려한 rule-based 필터링
- **Looping 오류 필터링**
  - 특정 n-gram이 **지나치게 반복**되는 경우 (예: 4-gram이 2회 이상)
  - 무한 반복처럼 보이는 잘못된 전사를 제거 
- **Early Stopping 오류 필터링**
  - beam search 중 **EOS(End-of-Sentence)**를 제대로 찾지 못한 가설 제거
  - 완전한 문장이 나오지 않은 경우 학습에 사용하지 않음

#### Confidence-based Filtering (신뢰도 기반 필터링)
- 각 pseudo-label에 대해 모델의 **예측 신뢰도**를 계산하여, 신뢰도가 낮은 샘플은 제거
- $`\text{ConfidenceScore}(\bar{Y}_i) = \frac{\log P_{\text{AM}}(\bar{Y}_i \mid X_i)}{|\bar{Y}_i|}`$
  - $`\bar{Y}_i`$ : i번째 입력 X_i 에 대해 생성된 pseudo-label
  - $`P_{\text{AM}}(\bar{Y}_i \mid X_i)`$ : 모델이 이 전사를 생성한 확률
  - $`|\bar{Y}_i|`$ : 전사된 token (예: word pieces)의 개수
  - 이 점수는 **log-likelihood를 길이로 정규화**한 값 - 길이가 길거나 짧더라도 확률이 낮은 전사는 제거됨

### Ensembles
- **필요성**
  - ASR에서는 **여러 모델의 출력을 결합하면** 단일 모델보다 **WER(Word Error Rate)가 낮아짐**
  - Self-Training에서도 마찬가지로, **여러 모델이 만든 전사를 조합하면** 더 좋은 의사 레이블을 만들 수 있음
- **기존 방식의 한계**
  - 단순히 여러 모델의 출력을 디코딩할 때 결합하면 → **계산량이 급격히 증가** 
#### 제안 방법 : 샘플 앙상블 (Sample Ensemble)
- 추론 시간에 모델들을 조합하는 대신, **학습 데이터 자체를 다양하게 만드는 접근**
- **\(M\)개의 acoustic model을 서로 다른 초기값으로 학습**
- 각 모델로부터 독립적인 pseudo-label 데이터셋 $`\bar{D}_m`$ 생성
- 모든 pseudo-label 세트를 **동등하게 가중합**하여 학습에 활용
- $`\sum_{(X,Y) \in D} \log P(Y \mid X) + \frac{1}{M} \sum_{m=1}^{M} \sum_{(X, \bar{Y}) \in \bar{D}_m} \log P(\bar{Y} \mid X)`$
  - D : 레이블이 있는 데이터셋
  - $`\bar{D}_m`$ : m번째 모델이 생성한 pseudo-label 세트
  - M : 모델 수
  - $`P(\bar{Y} \mid X)`$ : pseudo-label의 확률 (AM 기준)
- **학습 시 동작 방식**
  - 학습 중 **에포크마다 \(M\)개 모델 중 하나의 pseudo-label을 무작위 샘플링**하여 사용
  - 이렇게 하면 같은 음성 \(X\)에 대해 **다양한 전사**를 학습하게 되어 **과도한 확신(overconfidence)**을 방지
- **장점**
  - **다양성 증가** : 같은 발화에 대해 다양한 pseudo-label 제공
  - **노이즈 완화** : 하나의 잘못된 전사에 과적합하지 않음 
  - **성능 향상** : 특히 noisy 환경에서 큰 WER 개선 효과
  - **필터링과 호환** : heuristic/confidence 기반 filtering과 결합 가능
 
#### Confidence Score 계산 (필터링 연계)
- 앙상블에서 생성된 pseudo-label에 대해 신뢰도 기반 필터링도 적용함
- $`\text{ConfidenceScore}(\bar{Y}_i) = \frac{\log P_{\text{AM}}(\bar{Y}_i \mid X_i)}{|\bar{Y}_i|}`$

#### 핵심 기여
- **추론 속도를 희생하지 않고** 다양한 pseudo-label을 얻는 효율적인 샘플 앙상블 제안
- 모델 간 다양성을 활용해 overfitting과 label noise에 대한 강인함 향상
- 기존 방법 (예: Cycle-consistency training, ASR+TTS 등) 대비 **최대 93.8% 높은 WER Recovery Rate(WRR)** 달성  





<br>  
  
## 4. Experiments
### Data
- **주요 데이터셋** : LibriSpeech
  - **라벨링된 데이터 (Paired Data)**
    - "train-clean-100" 세트 : 100시간 분량의 깨끗한 음성 데이터
  - **라벨링되지 않은 음성 데이터 (Unpaired Audio Data)**
    - 깨끗한 음성(Clean Speech) 설정: "train-clean-360" 세트
    - 잡음이 있는 음성(Noisy Speech) 설정: "train-other-500" 세트
  - 평가용 데이터는 **LibriSpeech의 공식 dev/test 세트**를 사용
    - **clean / noisy 환경 모두에서 성능 측정** 

- 언어 모델(Language Model, LM) 학습용 텍스트
  - **LibriSpeech의 14,476권 도서**에서 파생된 텍스트 사용
  - **실제 환경 반영**을 위해, acoustic model과 겹치는 도서 997권은 **제외**함
  - 텍스트 전처리
    | 처리 항목 | 설명 |
    |-----------|------|
    | 문장 분할 | **NLTK (Natural Language Toolkit)** 사용 |
    | 소문자화 | 모든 텍스트를 lower-case로 변환 |
    | 구두점 제거 | 아포스트로피(`'`) 제외한 **모든 punctuation 제거** |
    | 하이픈 | 공백(`-` → `' '`)으로 대체 |
    | 비표준어 | **정규화하지 않음** (원문 그대로 유지) 

### Experimental Setting
- **인코더 구조**
  - 총 **9개의 TDS(Time-Depth Separable) 블록**으로 구성
  - **3개 그룹**으로 나뉘고 각 그룹은 아래와 같은 채널 수를 가짐
    | 그룹 | 채널 수 | 커널 너비 |
    |------|----------|------------|
    | 1    | 10       | 21         |
    | 2    | 14       | 21         |
    | 3    | 16       | 21         |
  - 인코더 아키텍처는 [Sequence-to-Sequence Speech Recognition with Time-Depth Separable Convolutions] 논문 구조를 따름

- **타겟 토큰 생성 (Word Piece)**
  - train-clean-100`의 전사 데이터를 기반으로 **5,000개의 word pieces(단어 조각)**을 생성
  - **SentencePiece** 툴킷 사용
    - 언어에 독립적인 서브워드 토크나이저
  - 이 word piece들이 모델의 **출력 단위(token)**로 사용됨  

- **학습 세팅**
  - **사전 학습 및 Regularization**
    - soft-window pre-training 적용
    - Teacher-forcing 기반 학습
    - Dropout: 20%
    - Random Sampling: 1%
    - Label Smoothing: 10%
    - Word Piece Sampling: 1%
  - **GPU 사용 및 Epoch 설정**
    - Baseline 학습: 1 GPU, batch size = 16
    - Pseudo-label 학습: 8 GPU 사용
  - **Optimizer 설정**
    - 옵티마이저: **SGD (momentum 없음)**
    - 학습률: 초기값 0.05
    - Learning Rate decay
      - 1 GPU: 40 에포크마다 ×0.5
      - 8 GPU: 80 에포크마다 ×0.5
    - 전체 학습: **200 에포크**
  - **프레임워크**
    - 전체 학습은 **wav2letter++** 프레임워크에서 수행   

- **외부 언어 모델 (Language Model, LM) 및 추론 설정**
  - LM: **word-piece 기반 convolutional LM (ConvLM)** 사용
    - LM도 동일한 구조와 방식으로 학습
  - **pseudo-label 생성 전**, beam search의 **hyperparameter를 dev set에서 사전 튜닝**
  - **pseudo-label과 paired data를 함께 학습할 때**, 기존 모델에서 이어받는 방식(fine-tuning)이 아닌, **random initialization**으로 다시 시작


### Results
#### Importance of Filtering
- **Heuristic 필터링**과 **Confidence-based 필터링**을 적용했을 때, pseudo-label 품질과 모델 WER이 어떻게 달라지는지를 측정
- 주요 결과
  | 설정 | 필터링 없음 | Heuristic | Heuristic + Confidence |
  |------|--------------|-----------|------------------------|
  | Dev-clean | WER 6.18% | 6.01% | **5.84%** |
  | Dev-other | WER 24.1% | 21.4% | **18.95%** |
  - Clean 환경에서는 **10%만 필터링해도 성능 개선** (→ 과한 필터링은 데이터 부족 유발)
  - Noisy 환경에서는 **최대 60% 필터링**이 최적 성능 (→ 잡음 제거 효과 큼)
  - 필터링은 특히 **잡음 많은 상황에서 pseudo-label의 품질을 높이는 데 핵심적**

#### Model Ensembles
- 앙상블 모델 수를 늘려가며 WER 측정 (1~6개 모델)
  | 모델 수 | Dev-clean WER | Dev-other WER |
  |----------|----------------|----------------|
  | 1        | 5.84%          | 21.86%         |
  | 4        | **5.41%**      | **20.31%**     |
  | 6        | 5.36% (소폭 향상) | 18.95% (**+13.7% 개선**) |
  - 앙상블 모델 수가 많을수록 전사 다양성 증가 → **overconfidence 방지**, 성능 향상
  - 특히 noisy 환경에서 **filtering과 ensemble을 함께 사용할 때 큰 시너지** 발생

#### Comparison with Literature
- WER Recovery Rate (WRR) 정의 : $`\text{WRR} = \frac{\text{Baseline WER} - \text{Semi-Supervised WER}}{\text{Baseline WER} - \text{Oracle WER}}`$
- 결과 비교 (100h paired + 360h clean unpaired 사용 기준)
  | 방법 | Test-clean WER | WRR |
  |------|----------------|-----|
  | Cycle TTE [9] | 21.5% | 27.6% |
  | ASR+TTS [10]  | 17.5% | 38.0% |
  | **본 논문 (Ensemble)** | **9.62%** | **76.2%** |
  - **기존 best 대비 93.8% 상대 성능 향상**
  - 이는 filtering + ensemble + strong baseline의 조합 효과

#### 핵심 정리
| 요소 | 성능 기여도 |
|------|--------------|
| **Filtering** | noisy 환경에서 pseudo-label 품질 개선 핵심 |
| **Ensemble** | 다양성 확보 → overfitting 방지 |
| **결합 효과** | clean/noisy 모두에서 WER 크게 감소 |
| **대비 기존 연구** | SOTA 대비 최대 93.8% WRR 향상 |





<br>  
  
## 5. Related Work
### 하이브리드 시스템에서의 Self-Training
- 대부분의 self-training 연구는 **전통적인 hybrid 모델 (HMM + GMM/DNN)** 구조에서 수행됨
- 주요 초점은 **pseudo-label 품질 개선**, 특히 **confidence-based filtering**, **다중 시스템 간 agreement** 사용
- 대표적인 기법
  | 기법 | 설명 |
  |------|------|
  | Confidence Filtering [4, 5] | 모델이 자신 없는 예측은 제외 |
  | Agreement-Based Selection [20] | 여러 모델이 **동일한 결과를 낼 때만 신뢰**하여 선택 |
  | Frame-level Selection [6, 7] | 문장 전체가 아닌 **프레임 단위로 필터링** 수행 |
  | Soft Targets [21] | 정답 대신 모델의 확률분포를 사용 (Knowledge Distillation 방식 유사) | 
- pseudo-label이 정확하지 않을 수 있다는 점을 고려하여 어떤 데이터가 학습에 도움이 되는지를 **선택(selection)**하는 데 집중함

### Self-Training in End-to-End ASR (최근 연구 흐름)
#### Text-to-Speech 기반 방식
- **ASR + TTS (Back-Translation)**
  - 텍스트 → TTS → ASR → 복원되는지 확인 (Cycle-consistency 사용)
- **Back-Translation**
  - Unpaired 텍스트를 음성으로 바꾼 후 ASR에 학습
#### Embedding 기반 접근
- **Inter-domain Loss**
  - 음성과 텍스트를 같은 embedding 공간으로 학습    

### 본 논문의 차별점과 기여
| 기존 방식의 한계 | 본 논문의 기여 |
|------------------|------------------|
| 복잡한 구조 (TTS 필요, cycle training 등) | 단순한 self-training으로 구현 가능 |
| 비교적 낮은 WRR | 최대 93.8% 높은 WER 회복률 달성 |
| small text corpus 사용 | 대규모 LM + strong encoder 활용 |
| soft target 방식 위주 | hard pseudo-label + filtering/ensemble 조합 |




<br>  
  
## 6. Conclusion
### 주요 성과 요약
- **강력한 베이스라인 모델**을 기반으로 Self-Training 수행
- **필터링 기법** 도입
  - seq2seq 오류 패턴(반복, 조기종료)에 특화된 **Heuristic filtering**
  - 신뢰도 기반 **Confidence filtering**
- **Pseudo-label 다양성 확보**를 위한 **모델 앙상블 전략** 제안
- 이 세 가지 요소의 결합으로 Self-Training의 효과를 극대화함

### 실험 결과 요약
- **LibriSpeech 데이터셋**에서 WER 성능 크게 향상
- **Noisy 환경**에서도 필터링 + 앙상블로 강력한 성능 유지
- 기존 SOTA 준지도 학습 기법(Cycle-consistency, ASR+TTS 등) 대비 **최대 93.8% 더 높은 WER 회복률(WRR)** 달성

### 연구의 의의
- **복잡한 구성 없이도** strong baseline, 간단한 filtering & ensemble만으로도 SOTA 수준 성능을 낼 수 있음을 입증
- **Self-Training의 재현 가능한 benchmark**로, 향후 준지도 학습 연구들이 비교할 수 있는 기반 제공














 
