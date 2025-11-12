# Neural utterance confidence measure for RNN-transducers and two pass models
## 요약 정리
### Problem
- On-device **E2E ASR**에서 실서비스 운영을 위해 **예측 신뢰도(confidence)** 가 필요하지만, 모델의 **자체 점수(posteriors/beam score)** 만으로는 **정확한 신뢰도 추정이 어려움**
- **Streaming RNN-T**는 지연(latency)은 낮지만 전체 문맥 활용이 제한되어 **정확도·신뢰도 한계** 존재
- **2-pass(RNN-T + LAS)** 는 정확도를 높일 수 있으나, **언제 2nd pass(서버)를 사용할지** 결정할 **신뢰도 모델**이 필요

### Contributions
- **Neural Utterance Confidence Measure (NCM)** 을 **RNN-T 및 2-pass** 환경으로 **확장/일반화**
- **Predictor features** 를 RNN-T(Trans/Pred/Joint)와 2-pass(LAS Enc/Dec, Beam Scores)로부터 체계적으로 결합·분석
- **Distributed ASR** 관점의 **CS(비용절감)–RIER(오류증가)** 지표로 **실용성 검증**
- **LAS 디코더 feature** 가 신뢰도 추정에 **가장 강력**함을 실증


### Method
- **목표**: 발화 단위 **Accept(1)/Reject(0)** 판별하는 **경량 이진 분류기**(2-layer FFN, ReLU)
- **입력 특징**
  - **RNN-T**: Trans(인코더 평균), Pred(디코더 평균), Joint(**top-K logits**)
  - **2-pass**: Enc(추가 BiLSTM 인코더 평균), **Dec(LAS 디코더 top-K logits)**, **Beam Scores**(RNN-T+LAS)
  - **Multi-beam**(선택): 경쟁 가설의 Joint/Dec
- **시퀀스 피처 집약**: 단순 평균보다 나은 **self-attention 가중 평균** → **concat → FFN**


### Experiments & Setup
- **ASR 백본**
  - **RNN-T**: Trans(6×uniLSTM, 1536; 전반부 3층 max-pool×2 → 8× 다운샘플), Pred(2×LSTM 1536), Joint+Softmax
  - **2-pass**: RNN-T 인코더 **재사용** + **BiLSTM(1536) 1층 추가**, **LAS 디코더(Emb 512, 1×LSTM 1536, Softmax)**  
    (LAS는 **rescoring 모드**로 NCM feature 추출)
- **데이터/훈련**: 10k h 내부 영어 코퍼스 공동 학습, powermel 입력, Keras 기반 in-house 트레이너
- **NCM 데이터**: dev 20k(훈련), prod test 1.5k(평가)  
  - **라벨**: ASR 가설이 **참조 전사와 정확히 일치** → 1, 아니면 0
- **평가 지표**: **AUC, EER, NCE**, **CS–RIER 커브**

### Results
- **ASR 자체 점수 한계**: RNN-T/2-pass의 단순 점수는 **AUC 낮고 EER 높음** → **외부 NCM 필요**
- **Beam Scores만으로도 강함**: **CS 최대 75%**(RIER 0%) 가능, **RIER 5–10% 허용 시 CS 최대 92%**
- **Feature 조합 인사이트**
  - **+ RNN-T Joint**: 성능 **소폭 저하**(blank 다량으로 혼선)
  - **+ 2-pass Dec(LAS)**: **가장 큰 향상**(문맥 활용, 변별력↑)
  - **Joint + Dec 동시**: **중간 수준**(중복/간섭)
  - **Pred 사용**: **향상 없음**
  - **All features**: **Dec 단독 ≈ All**(Dec가 핵심 정보원)
- **Multi-beam**: **이득 없음**(RNN-T beam이 blank 정렬로 **유사 가설** 생성 → 정보 중복)

### Limitations
- **내부 영문 코퍼스** 중심(일반화/도메인 전이 검증 제한)
- **Exact-match 라벨링**: 부분 정답/부분 오류에 대한 연속적 신뢰 모형 아님(경계 사례 민감)
- **2-pass = LAS 가정**: 다른 오프라인 디코더로의 전이 평가 부족
- **Rescoring 중심**: 2nd pass **full decoding** 상황의 NCM 효과 보고 제한
- **Multi-beam 무효**: RNN-T 탐색/정렬 특성 의존(전략 바뀌면 결과 달라질 수 있음)

### Insights & Idea
- **외부 신뢰도 모델은 필수**: posterior/beam만으로는 부족. **경량 FFN NCM**이 서버 전송 결정 등 **운영 의사결정**에 직접 기여
- **LAS 디코더 feature가 핵심**: **top-K logits**가 가장 강력한 predictor — **문맥 정보를 품은 디코더 신호**가 신뢰도 판별에 최적
- **RNN-T Joint의 한계**: blank/정렬 다양성으로 **혼선** 발생 → Dec(LAS)로 보완해야 함
- **CS–RIER로 정책 운용**: 목표 품질선(예: **RIER ≤ 5%**) 하 **CS 극대화** → 배터리/서버비/latency 최적화
- **실무 적용 아이디어**
  1) **On-device 게이팅**: NCM ≥ τ → **로컬 확정**, < τ → **클라우드 2-pass**
  2) **사용자/도메인별 τ 튜닝**: CS–RIER 커브 기반 **동적 임계값**
  3) **Feature 비용 고려형 NCM**: 추출 비용 낮은 **beam/Dec 중심**으로 추론 경량화
  4) **Partial-credit 라벨**(CER/WER 기반 soft label) 도입 → **칼리브레이션 개선**


<br>  
  
## 1. INTRODUCTION
### 1.1 연구 배경
- 최근 **End-to-End (E2E) 음성 인식 시스템(ASR)** 의 성능이 크게 향상되었고, 기기(On-device) 수준에서도 실시간 인식이 가능 [1, 2, 3] 
  - 그러나 이러한 시스템을 **실제 서비스 환경(production setup)** 에 안정적으로 배포하는 것은 여전히 도전적인 과제
- 배포된 모델이 생성한 **예측 결과의 신뢰도(confidence)** 를 정확히 추정하는 것은 매우 중요
  - 이는 음성 명령 처리, 사용자 피드백, 클라우드 후처리 등 **후속 작업(downstream task)** 의 품질과 비용 효율에 직접적인 영향을 미치기 때문
- 즉, ASR 시스템이 예측한 문장이 얼마나 “믿을 수 있는가”를 정량적으로 평가할 방법이 필요


### 1.2 E2E ASR의 변화와 RNN-T의 부상
- 기존의 E2E ASR 시스템은 주로 다음 세 가지 접근으로 발전
  - **CTC (Connectionist Temporal Classification)** [4]
  - **Attention-based 모델** [5]
  - **RNN-Transducer (RNN-T)** [6]

- 이 중 **RNN-T** 모델은 온라인(스트리밍) 음성 인식에 적합하며, 최근 **학계와 산업계 모두에서 가장 활발히 연구되는 프레임워크**
- 본 논문은 RNN-T 시스템을 단순 인식 모델로 다루지 않고, **“품질 추정(quality estimation)”** 관점에서 연구를 확장
- 즉, 모델이 산출한 결과의 **정확도를 신경망 기반으로 예측하는 방법**을 제안


### 1.3 On-device ASR의 제약과 기존 접근의 한계
- On-device ASR 시스템은 **낮은 지연 시간(low latency)** 과 **낮은 연산 비용(computation cost)** 을 동시에 만족해야 함
- 이를 해결하기 위한 여러 접근법 존재

| 접근 방식 | 장점 | 한계 |
|------------|------|------|
| **모델 최적화** | 성능 향상 가능 | 구현 복잡, 시간 소모 |
| **모델 압축 (compression)** | 메모리 절약 | 정확도 저하 |
| **분산 ASR (distributed ASR)** | 클라우드 자원 활용 | 인프라 비용, 지연 증가 |

- 최근에는 이러한 문제를 완화하기 위한 대안으로, **2-pass 아키텍처(two-pass architecture)** 가 제안
- 이 방식은 1차 스트리밍 모델(RNN-T)과 2차 정교한 오프라인 디코더(attention-based decoder)를 결합하여 정확도를 높이면서도 latency 최소화 가능


### 1.4 본 논문의 목표 및 기여
- 본 연구는 위와 같은 **2-pass 구조의 ASR 모델**에 대해 **신경망 기반 신뢰도 측정(Neural Confidence Measure; NCM)** 을 적용하는 방법 제안

  - **RNN-T와 2-pass 모델의 내부 feature 조합**을 입력으로 사용하여 confidence score을 계산  
  - NCM을 **binary classification task (Accept / Reject)** 로 학습  
  - 기존의 AUC, EER 등의 일반 지표뿐 아니라  
    **분산 ASR 환경에 맞춘 custom metric** 을 활용하여 실험적 평가 수행

- 즉, RNN-T와 2-pass 모델이 생성한 다양한 내부 표현(feature)을 신경망 입력으로 사용해 예측 문장의 신뢰도를 자동으로 학습·판별


### 1.5 Section Summary
- E2E ASR의 성능은 향상되었지만, **결과 신뢰도 추정이 여전히 미해결 문제**임  
- On-device 제약 환경에서는 **정확한 신뢰도 평가가 서비스 품질과 비용에 직결**  
- 기존 Attention 기반 신뢰도 연구를 **RNN-T 및 2-pass 구조로 확장한 최초의 시도**
- 제안 방법은 **Neural Confidence Model** 을 통해 **예측 품질을 정량화**하고 **distributed ASR 환경에서 효율적 자원 사용**을 가능하게 함
- [14]Utterance confidence measure for end-to-end speech recognition with applications to distributed speech recognition scenario 연구에서 NCM 논문을 최초 제안
  - 본 논문은 이 연구를 직접 확장한 것
  - [14]: MoChA 기반 ASR에서 NCM 제안
  - 본 논문: RNN-T 및 2-pass 구조에서도 NCM이 동작하도록 일반화 및 확장 
- 2-pass architecture for E2E ASR was proposed in [16]
  - Two-Pass End-to-End Speech Recognition 논문에서 제안한 2-pass구조 사용
  - Google Research팀 1st pass: RNN-T, 2nd pass: LAS 

<br>  
  
## 2. RELATION TO PRIOR WORK
### 2.1 기존 음성 인식 시스템의 신뢰도 측정
- 기존 ASR 시스템에서는 **단어(word) 단위의 신뢰도 측정(confidence measure)** 이 활발히 연구되어 왔음 [7]
- 이러한 접근법은 크게 세 가지 범주로 구분됨

  a) **발화 검증 (Utterance Verification)**: 전체 발화의 정확성을 확인하는 방식으로, 인식 결과의 전반적인 신뢰도를 평가

  b) **사후 확률 기반 (Posterior Probability-based)**: 인식된 단어의 사후 확률(posteriors)을 직접 활용하여 신뢰도를 계산

  c) **예측자 특징 기반 (Predictor Feature-based)**: ASR 모델에서 추출한 다양한 내부 특징(feature)을 조합하여 **분류기(classifier)** 를 학습시키는 방법

- 이 중에서도 **예측자 특징 기반 방법**이 최근 가장 널리 사용되고 있음 [8, 9]
  - 여러 연구에서는 이러한 feature 조합을 입력으로 받아 단어의 정확/부정확 여부를 분류하는 **신경망 기반(binary) 분류기**를 학습 [10, 11, 12, 13] 
  - 즉, 모델 내부 정보(음향, 언어적, 확률적 신호)를 **“예측 신뢰도”로 변환하는 학습적 접근**이 주류


### 2.2 End-to-End ASR의 신뢰도 측정 연구
- 최근 **End-to-End (E2E) ASR 환경**에서도 발화 수준의 신뢰도 측정 연구가 등장했음  
- 대표적으로 [14]에서는 **Monotonic Chunkwise Attention (MoChA)** [1, 15] 기반 E2E ASR 모델을 대상으로  
  **신경망 기반 예측자 특징 모델(neural predictor-feature model)** 을 학습하여 신뢰도 추정
- 해당 연구 결과, predictor feature 기반 접근법이 기존의 단순 posterior probability 기반보다 훨씬 우수한 성능을 보임 
- 즉, **attention-based E2E ASR에서도 neural confidence model이 효과적임**을 실증적으로 보여줌


### 2.3 본 논문의 확장과 차별점
- 본 논문은 위의 [14] 연구를 **RNN-Transducer (RNN-T)** 및 **2-pass 프레임워크**로 확장함
- 기존의 MoChA 기반 모델은 **attention alignment**를 사용하지만, RNN-T는 **streaming 구조**를 기반
  - 2-pass 시스템은 **RNN-T(online) + Attention-based decoder(offline)** 의 **이질적 결합 구조**를 가짐
- 따라서 [14]의 방법을 그대로 적용하기에는 구조적 차이가 존재하며, 이러한 차이를 고려한 **새로운 feature 조합과 neural classifier 설계**가 필요함


<br>  
  
## 3. END-TO-END SPEECH RECOGNITION
### 3.1 RNN-Transducer (RNN-T)
- **RNN-T** [6]는 온라인(스트리밍) 음성 인식에 널리 사용되는 대표적인 E2E 구조 
- Attention 기반 모델([5], [15])이 디코딩 전에 전체 오디오 또는 큰 청크를 필요로 하는 것과 달리, **RNN-T는 완전한 스트리밍 디코딩**이 가능
- 모델은 세 개의 주요 네트워크로 구성됨

| 구성 요소 | 설명 |
|------------|------|
| **Transcription Network (인코더)** | 입력 음성 신호를 인코딩하여 음향 정보를 추출 |
| **Prediction Network (디코더)** | 이전에 예측된 레이블을 인코딩하여 언어적 맥락을 모델링 |
| **Joint Network (조인트 네트워크)** | 두 출력을 결합해 어휘 전체에 대한 확률 분포 생성 |

- 전체 확률 분포는 다음과 같이 표현됨
  - $`h_t = \text{Trans}(x_t, h_{t-1})`$ 
  - $`g_u = \text{Pred}(y_{u-1}, g_{u-1})`$
  - $`p_{t,u} = \text{Joint}(h_t, g_u)`$


여기서  
- $`x_t`$: 입력 오디오 프레임  
- $`h_t`$: Transcription Network의 상태  
- $`y_{u-1}`$: 이전 단계에서 예측된 레이블  
- $`g_u`$: Prediction Network의 상태  
- $`p_{t,u}`$: 어휘 전체에 대한 예측 확률 분포

- 즉, RNN-T는 음향 정보(Transcription)와 언어 정보(Prediction)를 결합하여, **실시간 스트리밍 인식이 가능한 E2E 구조**를 제공


### 3.2 2-Pass End-to-End 음성 인식
- **2-Pass 아키텍처** [16]는 스트리밍 기반 RNN-T 위에 더 높은 정확도의 오프라인 디코더를 결합하여 **인식 품질을 향상**시키는 구조
  - 약간의 지연(latency)을 허용하는 대신, 전체 전사(transcription)의 정확도를 높일 수 있음
  - 온라인 모델은 입력 음성을 들어오는 즉시 처리하기 때문에 실시간으로 단어 출력 가능
  - 오프라인 디코더는 전체 음성 신호를 본 뒤에 디코딩하므로 문맥 정보가 풍부 -> 정확도 높음
- 처리 과정은 다음과 같음

| 단계 | 설명 |
|------|------|
| **1st Pass** | RNN-T(또는 유사한 스트리밍 모델)가 입력 오디오에 대해 초기 디코딩 수행 |
| **2nd Pass** | LAS (Listen, Attend and Spell) [5] 기반 디코더가 결과를 재평가 또는 재디코딩 |
| **모드** | (a) *Rescoring*: 1st Pass 출력을 재점수화<br>(b) *Beam Search*: 입력 오디오로부터 완전 디코딩 수행 |

- 실제 구현에서는 RNN-T의 **Transcription Network 위에 추가 LSTM layer**를 쌓아 **오디오 인코더(encoder)** 로 재사용하며, 이를 LAS 디코더가 활용

-즉, 2-Pass 모델은 RNN-T의 실시간성과 Attention 디코더의 정확도를 결합하여, **정확도 향상과 latency 최소화의 균형**을 달성


### 3.3 섹션 요약
- RNN-T는 **스트리밍 기반 E2E ASR의 대표 구조**, 2-Pass 모델은 **정확도 향상을 위한 하이브리드 구조**
- 두 프레임워크의 내부 네트워크(Transcription, Prediction, Joint, Encoder, Decoder)는 모두 본 논문에서 제안하는 **Neural Confidence Measure (NCM)** 의 입력 feature로 활용
- 즉, 이 섹션은 **NCM이 어떤 모델 구성요소의 출력을 기반으로 학습되는지**를 이해하기 위한 이론적 기반을 제공
- LAS 사용 이유
  - 대표적인 attention 기반 오프라인 디코더
  - RNN-T 인코더의 출력 차원과 LAS 인코더 입력 차원이 자연스럽게 일치
  - RNN-T의 Transcription Network 출력을 LAS 디코더의 입력 인코딩으로 그대로 사용한다는 뜻
  - [16] 논문 이후, Google, Samsung, Apple 등 대부분의 2-pass 연구는 “RNN-T + LAS” 조합을 표준으로 채택 


<br>  
  
## 4. NEURAL UTTERANCE CONFIDENCE MEASURE
### 4.1 개요
- 이 섹션에서는 **음성 인식(ASR)** 모델의 예측에 대한 신뢰도 점수를 계산하는 **신경망 기반 접근 방식(NCM, Neural Utterance Confidence Measure)** 을 설명
- 본 연구는 **End-to-End (E2E)** 모델, 특히 **RNN-Transducer (RNN-T)** 및 **2-pass 모델**의 출력에 대해 **발화 수준(utterance-level) 신뢰도(confidence score)** 를 계산하는 방법 제안


### 4.2 기존 연구 확장 및 필요성
- 본 연구는 **“Utterance Confidence Measure for End-to-End Speech Recognition”** [14]의 접근법을 **RNN-T** 및 **2-pass 모델**로 확장
- 기존 [14] 연구는 **Monotonic Chunkwise Attention (MoChA)** [15] 기반 **attention decoder**의 예측에 대해 신뢰도 계산 
- 하지만 **RNN-T**는 MoChA와는 **다른 목적 함수(Objective Function)** 로 훈련되고, **디코더 구조 및 feature 특성이 근본적으로 다르기 때문에** 동일한 방법을 그대로 적용할 수 없음
- 따라서, **RNN-T 및 2-pass 모델에 맞는 새로운 feature 조합과 neural confidence framework** 가 필요


### 4.3 NCM의 학습 목표
- NCM 모델은 음성 인식 결과(hypothesis)가 **정확한지(accept)** 혹은 **부정확한지(reject)** 를 판별하는 **이진 분류(binary classification)** 모델로 학습
- 입력은 ASR 모델에서 추출한 다양한 feature들이며, 출력은 `1(accept)` 또는 `0(reject)` 로 설정  
- 즉, NCM은 “이 발화 결과를 신뢰할 수 있는가?”를 학습적으로 판단하는 신경망


### 4.4 2-pass 모델 확장의 장점
- **2-pass 구조**는 RNN-T보다 훨씬 **풍부한 feature set**을 제공
- 이러한 feature는 신뢰도 측정에서 사용되는 **predictor feature-based 모델**의 성능을 크게 향상[14]
- 따라서 본 논문에서는 **RNN-T의 내부 feature** 뿐만 아니라, **2-pass(LAS 기반 디코더)** 의 encoder/decoder feature를 함께 활용


### 4.5 NCM 입력 특징 (Input Features)

| Feature | 출처 네트워크 | 설명 |
|----------|----------------|------|
| **Transcription Network Output (Trans)** | RNN-T Encoder | 입력 음성의 **음향 요약(acoustic summary)**. 선행 연구 [8]에서 음향 임베딩이 confidence 추정에 중요한 역할을 한다고 보고됨. [14]에서도 이 feature를 명시적으로 전달할 때 성능이 향상됨. |
| **Prediction Network Output (Pred)** | RNN-T Decoder | 각 timestep에서 이전 예측된 레이블의 임베딩을 출력. 디코딩된 결과에 대한 언어적 정보를 포함. [14]에서는 사용되지 않았던 feature. |
| **Joint Network Output (Joint)** | RNN-T Joint Network | Softmax 이전의 최종 디코더 출력(logits). 각 디코딩된 토큰에 대해 **상위 K개의 로짓(top-K logits)** 만 사용하여 메모리 효율 개선. |
| **2nd Pass Encoder Output (Enc)** | 2-pass Encoder | RNN-T Transcription Network 위에 추가된 LSTM layer의 출력. Trans feature와 유사하게 인코딩된 음향 요약 정보를 제공. |
| **2nd Pass Decoder Output (Dec)** | 2-pass LAS Decoder | LAS 디코더의 출력 logits. Joint feature와 유사하게 각 디코딩된 토큰에 대한 top-K logits을 수집. |
| **Beam Scores (Scores)** | RNN-T & LAS Decoders | 각 빔(beam)에 대한 로그 확률 점수(log-probability score). 기존 연구 [14]에서도 신뢰도 측정에 효과적임이 확인됨. |
| **Multi-beam Features** | RNN-T & LAS | 여러 빔의 Joint 및 Decoder feature를 함께 활용. [14], [17]에서는 **경쟁 가설(competing hypotheses)** 이 confidence 예측에 중요한 정보를 제공함을 보고함. |


### 4.6 섹션 요약
- NCM은 RNN-T 및 2-pass ASR 모델의 출력을 기반으로 **발화 신뢰도(utterance confidence)** 를 추정하는 신경망 모델  
- 기존 attention 기반 모델의 신뢰도 측정 기법을 **RNN-T/2-pass 구조로 일반화**한 최초의 연구
- 다양한 내부 feature 조합(Trans, Pred, Joint, Enc, Dec, Scores, Multi-beam)을 통해 **정확하고 안정적인 confidence score 계산** 가능
- 2-pass 구조의 richer feature set이 confidence 추정의 품질을 크게 개선



<br>  
  
## 5. EXPERIMENTS
### 5.1 End-to-End Speech Recognition (ASR Setup)

#### RNN-T (Streaming Model)
- **Transcription Network (Encoder)**  
  - 6 × unidirectional LSTM, 각 레이어 hidden size = 1536  
  - 첫 3개 LSTM 출력에 **max pooling ×2** 적용 → 시계열 길이 **8배 축소**(효율성↑, 정확도 유지)
- **Prediction Network (Decoder)**  
  - 2 × LSTM, hidden size = 1536
- **Joint Network**  
  - Transcription/Prediction 출력을 결합 → feed-forward + **Softmax**

#### 2-Pass Model (Second-Pass Decoder)
- **인코더 재활용**: RNN-T **Transcription Network**를 **그대로 재사용**
- **추가 인코더 레이어**: 위에 **1 × BiLSTM(1536 cells)** 추가
- **LAS Decoder (Rescoring Mode)**  
  - Embedding dim = 512, 1 × LSTM(1536), Softmax  
  - 본 논문에서는 **rescoring mode**로 사용하여 **NCM 학습용 feature** 추출

#### Training Details
- **Joint training**: RNN-T와 2-pass를 **10k h internal English corpus**로 공동 학습  
- **Input feature**: Powermel  
- **Framework**: TensorFlow Keras 기반 **in-house trainer**

#### 위 설정은 이후 NCM(§5.2)과 입력 특징(§4.5)의 **feature 추출 파이프라인**을 구성하는 기반


### 5.2 Neural Confidence Measure (NCM)

#### 구조 (Binary Classifier)
- **목표**: 예측된 발화가 **Accept(1)** / **Reject(0)** 인지 판별  
- **아키텍처**: 2 × feed-forward layers, hidden size = 64, **ReLU**  
- [14]의 모델 구성과 유사한 **경량 신경 분류기**

#### 입력 특징 (from RNN-T & 2-Pass)
- **Trans**: Transcription network의 프레임별 출력 → 평균하여 **1536-d** 벡터
- **Pred**: Prediction network의 심볼별 출력 → 평균하여 **1536-d** 벡터
- **Joint**: RNN-T joint의 **pre-softmax logits** → 각 토큰의 **top-K logits**
- **Enc**: 2-pass 인코더(BiLSTM) 출력 → Trans와 유사하게 요약 사용
- **Dec**: 2-pass **LAS decoder logits** → 각 토큰의 **top-K logits**
- **Scores**: RNN-T 및 2-pass의 **beam log-probabilities**  
  - **beam width = 4** → 4-d 벡터
- **Multi-beam**: 경쟁 가설의 Joint/Dec feature도 선택적으로 포함

#### 특징 처리 & 출력
- **시퀀스형 feature**(예: Joint/Dec)는  
  - 단순 평균보다 성능이 근소하게 좋았던 **self-attention 기반 weighted averaging**으로 고정 길이 벡터화  
- 모든 feature를 **concatenate → binary classifier 입력**  
- 출력은 **Accept(1) / Reject(0)**, 추론 시 **P(positive)** 를 신뢰도로 사용


### 5.3 NCM Dataset

- **Train(dev) 구성**: ASR 훈련 시 사용한 **devset 20k utterances** 사용  
  - 전체 훈련 셋에서 **무작위 20%**로 구성된 devset
- **Feature 추출**: **저사양 ASR**을 beam search 모드(2-pass on/off 모두)로 실행하여 §5.2의 입력 feature(Trans/Pred/Joint/Enc/Dec/Scores 등)를 수집
- **레이블링 규칙**  
  - ASR 가설이 **참조 전사와 정확히 일치**하면 **target = 1 (Accept)**  
  - 일치하지 않으면 **target = 0 (Reject)**
- **Test 구성**: **in-house Production dataset, 1.5k utterances**


### 5.4 Evaluation Protocol

#### Conventional Metrics
- **AUC (ROC Area)**: 1에 가까울수록 우수  
- **EER (Equal Error Rate)**: 낮을수록 우수  
- **NCE (Normalized Cross Entropy)**: 본 논문 표기(↑) 기준으로 **클수록** 우수

#### Distributed ASR 지표 (Application-Oriented)
- **CS (Cost Saving)**: 온디바이스에서 처리(=클라우드 미전송)로 절약된 **연산 비용 비율** → **높을수록** 우수
- **RIER (Relative Increase in Error Rate)**: NCM 의사결정으로 인한 **WER 상대 증가율** → **낮을수록** 우수
- **목적**: **latency/비용과 정확도** 사이의 실제 트레이드오프를 가시화  
- [14]의 **CS-vs-RIER** 평가 방식을 **동일하게** 채택


### 5.5 섹션 요약
- 본 실험 설정은 **RNN-T(Streaming) + 2-Pass(LAS)** 조합으로 feature를 추출하고, **경량 신경 분류기(NCM)** 로 발화 신뢰도 판별
- **Train/Dev/Test** 구성과 **Conventional + Application 지표**를 함께 사용하여, **정확도와 비용 효율**을 동시에 평가




<br>  
  
## 6. RESULTS
### 6.1 개요
- 본 섹션은 제안된 **신경망 기반 발화 신뢰도 측정(NCM)** 모델의 성능을 실험적으로 검증
- RNN-Transducer(RNN-T) 및 2-pass 모델의 다양한 **feature 조합**이 NCM 성능에 미치는 영향 분석


### 6.2 ASR 모델 자체 신뢰도 점수의 한계
- Table 1의 상단 두 행에서 볼 수 있듯이, **RNN-T** 및 **2-pass** 모델이 자체적으로 산출한 confidence score는 **AUC/NCE가 낮고 EER이 높음** → 신뢰도 분류 능력이 제한적 
- 따라서, 별도의 **신경망 기반 외부 confidence model (NCM)** 을 훈련해야 함이 명확히 드러남


### 6.3 Beam Scores의 효과
- NCM 모델이 **beam scores만** 입력으로 사용하더라도 **분산 ASR 환경(distributed ASR)** 에서 탁월한 성능을 보임
- 결합된 전체 WER(Word Error Rate)을 악화시키지 않으면서 **최대 75% Cost Saving (CS)** 달성
- 5~10%의 작은 오류율 증가(RIER)를 허용하면 **최대 92% Cost Saving** 가능 — 이는 기존 연구 [14]보다 높은 수치


### 6.4 Feature 조합에 따른 성능 변화
| 실험 조건 | 관찰 결과 | 해석 |
|------------|------------|------|
| **Joint Network Features 추가 (RNN-T)** | 성능 약간 저하 | Joint 출력에 **Blank label**이 많이 포함되어 혼란을 유발 |
| **Decoder Features 추가 (2-pass)** | 성능 향상 | Attention 기반 디코더 피처가 **더 높은 변별력**을 제공 |
| **Joint + Decoder Features 동시 사용** | 중간 수준 성능 | 두 feature가 상호 중복적 → 완전한 시너지 아님 |
| **Prediction Network Features 사용** | 성능 향상 없음 | Non-blank label encoding 정보는 신뢰도 예측에 크게 기여하지 않음 |
| **모든 Feature 통합 사용** | Decoder feature만 사용 시와 유사하거나 약간 낮음 | Decoder feature가 핵심적인 정보원임을 시사 |

- Attention 기반 디코더(LAS)의 출력 feature가 RNN-T 내부 feature보다 **신뢰도 예측에 훨씬 효과적**


### 6.5 Multi-beam Features의 영향
- 여러 beam의 feature를 함께 사용해도 **성능 향상 없음** 
- 이는 기존 연구 [14]와 다른 결과이며, **RNN-T beam search에서 Blank label이 빈번히 예측되어 가설들이 매우 유사**하게 형성되었기 때문으로 분석됨
- 즉, 다중 beam은 **추가 정보보다 혼선(confusion)** 을 초래함


### 6.6 요약
- **NCM > ASR 자체 confidence**: 외부 신경망 학습이 필수  
- **Beam Scores만으로도 높은 비용 절감** (CS 최대 92%)  
- **2-pass Decoder feature**가 가장 유용한 predictor  
- **Multi-beam 효과는 제한적**, RNN-T 구조 특성상 redundancy 존재  
- 즉, 2-pass 모델에서 얻을 수 있는 **풍부한 feature**, 특히 **Attention-based decoder의 feature**가 신경망 기반 신뢰도 추정(NCM)의 성능과 **분산 ASR 효율성**을 크게 개선



<br>  
  
## 7. CONCLUSION
### 7.1 연구 요약
- 본 논문은 [14]에서 제안된 **Neural Utterance Confidence Measure (NCM)** 기법을 **RNN-Transducer (RNN-T)** 및 **2-pass** 프레임워크로 확장한 연구
- 두 구조의 근본적인 차이(Streaming vs Attention)를 고려하여 기존 방법을 단순 적용하는 대신, 새로운 **feature 조합 및 평가 환경**을 설계


### 7.2 주요 기여 및 발견
- **기존 연구 확장**  
  - Attention 기반 MoChA 모델에 한정되었던 NCM을 **RNN-T 및 2-pass 모델**에서도 적용 가능하도록 일반화
- **새로운 feature 추가**  
  - 기존 predictor feature에 더 다양한 내부 표현(Trans, Pred, Joint, Enc, Dec, Scores)을 포함
- **정량적 검증**  
  - 다양한 feature 조합 실험을 통해 **Decoder feature가 가장 유효**함을 입증
- **2-pass 구조의 장점**  
  - 인식 정확도 향상뿐 아니라 **confidence 추정에도 유리**  
  - 분산 ASR 시나리오에서 **높은 Cost Saving 효과(>90%)**

### 7.3 연구 의의
- Attention 기반 모델에서만 검증되던 NCM의 효과를 **Streaming 기반 ASR (RNN-T)** 로 확장한 **최초의 연구**. 
- 실시간 인식이 필요한 on-device 환경에서도 **신뢰도 기반 후처리 및 서버 선택(decision-making)** 이 가능함을 보여줌
- 실제 배포 가능한 **효율적이고 신뢰성 있는 ASR 품질 추정 방법론** 제시


### 7.4 향후 연구 방향
- **다국어(multilingual)** 및 **잡음 환경(noisy scenario)** 에서의 NCM 일반화  
- **Non-streaming attention 모델**과의 hybrid confidence fusion  
- 실제 **클라우드–디바이스 연동 환경**에서의 latency/accuracy 최적화 연구

















