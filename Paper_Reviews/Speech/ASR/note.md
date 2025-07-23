
## 1) Self-Training for End-to-End Speech Recognition (4pages)
### Self-Training  
- unlabeled data에 대해 pseudo-label(가짜 정답)을 생성하고, 그 데이터를 마치 labeled data처럼 다시 학습에 사용하는 semi-supervised learning 기법
- Why?
  - 대규모 speech 데이터의 transcribe는 비용과 시간이 많이 듬
  - unlabeled audio는 많지만 label 부족 → 이를 효율적으로 활용하기 위한 방법
  
### 주요 기여
- 강력한 baseline acoustic model + language model로 pseudo-label 생성
- seq2seq 모델의 특유 오류에 특화된 filtering 방식 제안
- pseudo-label 다양성을 위한 novel ensemble 방법 제안
  
### 주요 실험 결과(LibriSpeech corpus 기준)
- noisy speech: WER 33.9% 개선
- clean speech: baseline 대비 oracle과의 gap의 59.3% 회복
- 기존 방법보다 최소 93.8% 상대적 성능 우위

## 2) Self-training and Pre-training are Complementary for Speech Recognition (4pages)
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


## 3) Unsupervised Domain Adaptation for Speech Recognition via Uncertainty Driven Self-Training (4pages)
### 문제 정의
- ASR 시스템은 훈련 데이터와 실제 테스트 데이터(다른 도메인) 간 차이로 인해 성능이 저하됨
- ex) 깨끗한 뉴스 음성(WSJ)으로 학습 후, TED나 전화 통화(SWBD)에서 성능 하락
  
### 기존 Self-Training 방법의 한계
- teacher model이 unlabeled target data에 pseudo-label 부여 → student model 학습
- 한계: pseudo-label이 noisy하면 성능이 악화됨
- 기존 work들은 domain mismatch 없는 조건에서 filtering 없이 사용
  
### 제안 방법 : DUST(Dropout-based Uncertainty-driven Self-Training)
- **핵심 아이디어**
  - dropout 설정을 다르게 하여 얻은 여러 prediction들 간의 일치도를 통해 모델의 prediction 불확실성을 측정
  - **불확실성 높은 pseudo-label은 학습에서 제외**
- **장점**
  - Filtering 없는 **ST 대비 ASR 성능 크게 향상**
  - 학습 데이터셋 크기가 줄어 **training time 단축**
  
### 싦험
- Dataset
  - Source: WSJ (깨끗한 read speech)
  - Target: TED-LIUM 3 (강연), SWITCHBOARD (전화 대화)
- 결과
  - WSJ → TED: 최대 80% WER recovery
  - WSJ → SWBD: 최대 65% WER recovery 


## 4) Domain Adaptive Self-supervised Training of Automatic Speech Recognition (4pages)
### 문제 정의
- ASR 시스템의 도메인 적응
  - 다른 억양, 환경에서 성능 저하
  - 라벨 없는(target domain의 unlabeled) 데이터를 활용하여 ASR 모델 성능을 개선할 수 있는 방법 필요
  
### 기존 방법의 한계
- Self-supervised Learning (SSL) ASR은 많은 unlabeled 데이터로 representation을 학습
- 도메인 mismatch가 큰 경우 target domain에 대한 성능 저하 발생
- 단순히 target domain 데이터를 pre-training에 추가하는 것만으로는 한계

### 제안 방법
- **자기지도학습(SSL) + 반지도학습(semi-supervised learning)** 의 조합
- Target domain unlabeled data를 SSL Pre-training에 활용
  - or Fine-tuning에 활용 (semi-supervised pseudo-labeling)
  - 또는 두 단계에 모두 사용

### 실험 세팅
- 도메인 = 영어 억양(Accents)
  - 미국식 억양 (in-domain)
  - 비영어권 화자의 영어, 영국 억양, 인도 억양 (target domains)
- 평가 metric: Word Error Rate (WER)
- baseline: SSL로 학습된 wav2vec 2.0 기반 ASR 모델

### 주요 결과
- 단일 도메인 실험
  - WER 2.7% ~ 41.8% 상대적 감소 (도메인 mismatch 정도에 따라 다름)
- 다중 도메인 실험
  - 평균 8% WER 감소  

## 5) Consistency Based Unsupervised Self-training For ASR Personalisation (6pages)
### 문제 배경
- 현대 ASR(Automatic Speech Recognition) 시스템은 대규모 사용자 데이터를 기반으로 학습되지만, 훈련 시 보지 못한 개별 사용자에게는 성능 저하 발생
- 주요 원인: domain shift
  - 사용자 발화 특성 (억양, 발음 등)
  - 환경적 음향 조건 (소음, 울림 등)

### 기존 방법
- ASR personalisation  
  - 개별 사용자 데이터로 모델을 fine-tuning 하여 성능 개선
  - 대부분 기존 방법은 **labelled user data 필요**
- 라벨 없는 상황에서의 개인화(unsupervised personalisation)는 특히 어려움
  - 데이터 수량 부족
  - 녹음 품질 문제

### 제안 방법
- pseudo-labeling + consistency 기반 training
- 라벨 없는 user data에서 안정적인 학습 가능

### 주요 성과
- 17.3% WER 감소 (training data 기준)
- 8.1% WER 감소 (held-out test data 기준)
- 기존 SOTA 방법들보다 우수한 성능


## 6) Self-training ASR Guided by Unsupervised ASR Teacher (4pages)
### 문제 정의
- Self-training은 음성 인식에서 성능을 개선하는 방법으로 각광받고 있음
- 첫 teacher model 학습에 라벨링된 데이터가 필요
- 소량의 라벨링 데이터로 학습된 teacher는 과적합으로 인해 noise가 섞인 pseudo-label을 생성
  - unseen data에서 성능 저하
  
### 기존 방법의 한계
- 라벨 의존성 : 초기 teacher 학습에 supervised data 필요
- teacher quality 문제 : 첫 teacher가 small labeled dataset으로 overfitting
  - noisy pseudo-targets 생성
- 비용 문제 : 기존 방법은 multi-stage training 구조 → training 비용 ↑
  
### 제안 방법
- **UASR(Unsupervised ASR) teacher 도입**
  - 라벨 없이(unpaired speech & text) 학습 가능한 teacher로 시작
  - labeled data dependency 제거
- **중간 층 phonetic supervision**
  - teacher의 phonetic 정보가 student(Data2vec2)의 intermediate layer로 distillation
  - 상위 layer pseudo-target에 phonetic + contextual 정보 강화
- 결과적으로 more ASR-friendly한 pseudo-target 생성 → WER 개선
  
### 실험 세팅
- 데이터셋 : LibriSpeech
  - pre-training: LS 960h (라벨 없이)
  - fine-tuning: LS 100h (소량 labeled)
- 평가 지표 : Word Error Rate (WER)
- 비교 baseline : Data2vec2 (SOTA self-supervised model)
  
### 주요 결과
- test-clean: 8.9% WER relative reduction (Data2vec2 대비)
- test-other: 4.3% WER relative reduction
- 라벨 없이 pre-train한 teacher로도 SOTA 수준 달성
  
### 논문 장점 및 한계
- **장점**
  - 라벨링 데이터 필요 없는 self-training 구조 → 실용적
  - 기존 Data2vec2보다 ASR 친화적 pseudo-target 생성 가능
  - 성능 우수 (SOTA 기록 달성)
- **한계**
  - teacher로 UASR을 쓰지만, GAN 기반 UASR은 training 안정성 이슈 (unstable training)
  - intermediate layer 선택의 중요성 → 실험적 hyperparameter tuning 필요


## 7) EFFUSE: Efficient Self-Supervised Feature Fusion for E2E ASR in Low Resource and Multilingual Scenarios (4pages)
### 문제 정의
- SSL 모델은 저자원/다국어 ASR에서 강력한 성능을 보여주지만, **단일 SSL 모델은 한계 존재**
  - 특히 영어 단일 talker 환경에서 훈련된 모델은 타 언어/환경에 한계
- 이를 보완하기 위해 여러 SSL 모델의 **feature fusion(특징 융합)** 사용됨
  - 여러 SSL 모델을 융합하면 **파라미터 수가 크게 늘어 계산 비용이 높아짐**

### 기존 방법의 한계
- 다수 SSL 모델을 **단순히 결합(fusion) → 성능은 좋아짐**
- 단점: **모델이 무거워짐 (파라미터 수 ↑, 연산량 ↑, latency ↑)**

### 제안 방법 EFFUSE
- 하나의 SSL 모델만 사용하여 **다른 SSL 모델의 feature를 “예측(predict)”**
- 여러 모델의 feature를 직접 계산하지 않고, 한 모델의 feature를 기반으로 다른 모델 feature를 재구성
  - **경량화 + 성능 유지**
- Prediction 기반 **feature fusion 구조**
- SUPERB benchmark에서 baseline SSL보다 +6.3% score 개선
- 기존 fusion 모델 대비 파라미터 약 49% 절감 (평균 317M param 감소)

## 8) Speech Self-Supervised Learning Using Diffusion Model Synthetic Data (17pages)
### 배경 및 문제 상황
- **Self-Supervised Learning(SSL - 자기지도학습)**
  - 최근 음성 SSL은 labeling 없는 음성만으로 representation을 잘 학습 가능
  - ASR 등 downstream task에서 labeled data 요구량 감소
  - EX) HuBERT, Wav2Vec2.0
- **문제점**
  - 여전히 대규모 비주석 corpus 필요 (~1000시간 이상)
  - 저자원 언어(low-resource languages): 데이터 자체 부족
  - 프라이버시 문제: 데이터 수집 곤란
  - 기존 데이터 증강은 단순 noise 추가 등으로 prosody, speaker, content의 다양성을 잘 확장하지 못함 
   
### 제안 방법 : DIFFS4L(Diffusion Synthetic Speech Self-Supervised Learning)
- **아이디어**
  - 제한된 real 데이터로 diffusion model 학습
  - 다양한 variation을 갖는 synthetic speech 생성
    - 새로운 prosody(운율)
    - 새로운 speaker
    - 새로운 content (의미 없는 babble 포함)
  - Real + Synthetic data로 SSL 모델 사전학습
- **특징**
  - Diffusion model은 기존 generative model(WaveNet 등)보다 **데이터 분포를 더 잘 모델링 가능**
  - synthetic data에서 다양성(prosody, speaker, content) 제공
    - SSL 정보 효율성 증대

### 실험 결과
- English ASR Task
  - HuBERT pretrained model의 WER 6.26%p 감소
  - 26.4% relative improvement    
- 놀라운 발견
  - synthetic babble(의미 없는 음성)조차 SSL 성능 개선에 기여!
  - 기존 augment 방법보다 더 효과적
- 코드 공개 : https://github.com/Hertin/DiffS4L 

## 9) Data-Filtering Methods for Self-Training of Automatic Speech Recognition Systems (5pages)
### Self-Training이란?
- labeled data가 부족할 때, unlabeled speech를 자동 전사 → 학습 데이터에 추가
- 과정
  - 초기 ASR 시스템 준비 (labeled data로 학습)
  - unlabeled speech 데이터 입력 → 자동 전사 생성
  - 이 데이터를 labeled corpus에 추가하여 retraining 
- 문제점
  - 자동 전사가 오류를 포함 → 잘못된 라벨 데이터로 학습될 위험
  - 정확한 전사만 골라서 사용해야 함 → “Data Filtering” 필요
   
### 제안 방법 및 비교
- **Confidence score 기반 filtering**
  - ASR 시스템이 각 단어에 대해 confidence score 제공 (posterior probability)
  - 특정 threshold(예: 0.95) 이상만 선택
  - 간단하고 self-contained
  - 단점: 이미 “잘 알아듣는” 데이터만 선택 → 새로운 정보가 부족
- **Multiple hypotheses 기반 filtering**
  - 서로 다른 ASR 시스템 2개 사용 → 각 system이 동일한 transcriptions 예측한 부분만 선택
  - 서로 다른 오류 패턴을 이용 → “agreement” = 신뢰 가능
  - 다양한 source 기반 보강
  - 추가 ASR 시스템 필요 → 자원 요구
- **Approximate transcript 기반 filtering**
  - 뉴스 headline, 자막, script 등 “대략적인 텍스트”와 alignment
  - 공통 부분만 선택
  - 소량이지만 다양하고 noise-robust 데이터 확보
  - noisy 환경에서 효과적
  - Approximate text 필요 (모든 도메인에 적용 어려움)
  
### 성능 비교
- 기존 baseline 대비 최대 25% 상대적 WER 개선
- 세 방법 중 approximate transcript 기반 방법이 가장 효과적
  - 데이터 양은 적지만 품질 좋음
  - 특히 degraded/noisy speech에서 성능 향상 뚜렷
- 여러 방법을 합쳐도 approx 하나보다 좋지 않음
  - 이유: confi, multi는 “seed ASR가 이미 잘 하는 부분”에 bias
 
### 논문의 의의 및 한계
- **의의**
  - 다양한 filtering 방법 직접 비교 → 실질적 적용 지침 제공
  - Approximate transcript 활용법 실증
  - Romanian ASR에서 state-of-the-art 달성
- **한계**
  - Approximate transcript가 있어야 가능 (일반화 제한)
  - Romanian 데이터에만 실험됨 → 다른 언어에서 바로 적용 보장 X 

## 10) Unsupervised Fine-Tuning Data Selection for ASR Using Self-Supervised Speech Models (4pages)
### 연구 배경
- **Self-Supervised Learning (SSL)**
  - 라벨 없이 unlabeled speech data를 대량으로 활용
  - pre-training으로 강력한 음성 representation 학습
  - HuBERT, wav2vec 2.0, small labeled data로도 good performance
- **문제 상황**
  - 실제 적용 상황에서는 “전사 예산(transcription budget)“이 제한됨
  - unlabeled pool에서 어떤 subset을 transcribe해서 fine-tune할지를 결정하는 것은 중요한 과제

### 연구 목표
- 제한된 예산 하에서 좋은 ASR 성능을 위한 “데이터 선택(criteria)” 탐구
- “비지도(unsupervised) 방식”으로 unlabeled data 중 어떤 데이터를 선택할지 연구

### 주요 연구 내용
- 다양한 데이터 특성 평가
  - 화자 다양성(speaker diversity)
  - 성별 편향(gender bias)
  - 주제 다양성(topic diversity)
- 두 가지 새로운 방법 제안
  - **Pre-training loss 기반 데이터 선택**
    - HuBERT pre-training 과정의 loss (masked/unmasked cross-entropy)로 utterance의 informativeness를 측정
  - **PBPE(Perplexity of Byte Pair Encoded clustered units)**
    - HuBERT cluster label → run-length encoding → BPE tokenization → language model perplexity로 utterance score화
- Random selection과 비교
  - 얼마나 competitive한지 평가
- Correlation 분석
  - 선택된 subset의 특성과 성능(WER) 관계 분석
  - token diversity, speaker diversity, topic diversity의 중요성 실증   

## 11) Robust Speech Recognition via Large-Scale Weak Supervision (24pages)
### 연구 목표
- 기존 음성 인식 시스템은 fine-tuning 필요
  - 특정 데이터셋/도메인에 최적화되지만 일반화 어려움
- Whisper는 fine-tuning 없이 다양한 상황에서 robust하게 작동하는 universal speech recognition model을 만들고자 함 

### 핵심 방법
- 인터넷에서 수집한 transcript-paired audio 68만 시간으로 대규모 weak supervision 학습
- 다국어(multilingual) + 다중 작업(multitask) 학습
  - Speech recognition
  - Speech translation
  - Voice activity detection 등

### 주요 성과
- zero-shot setting에서도 기존 fully-supervised SOTA 모델에 필적하거나 뛰어남
- 인간 수준의 정확성과 robustness에 근접
- 모델과 inference code를 오픈소스화하여 커뮤니티 활용 가능


## 12) Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models (24pages)
### 연구 문제 정의
- 대형 ASR foundation models (Whisper 등)도 domain shift 상황(잡음, 억양 등)에서는 성능 저하 발생
- 라벨 없는 타겟 도메인 데이터만으로 모델을 효과적으로 적응시키는 방법 필요

### 기존 방법의 한계
- 기존 Unsupervised Domain Adaptation(UDA) 방법
- 대부분 라벨된 소스 데이터와 비라벨 타겟 데이터 둘 다 필요
- 현실에서는 소스 데이터 접근 불가능한 경우 많음 (보안, 저장, 프라이버시 문제 등)
- Softmax 기반 confidence score만으로 pseudo-label quality 판단
  - 신뢰성 떨어짐 (over-confidence 문제)
 
### 제안 방법 : STAR (Self-TAught Recognizer)
- Source-free UDA: 소스 데이터 없이, unlabeled 타겟 데이터만 활용
- Token-level pseudo-label quality 평가
  - 디코딩 중 self-attention 정보를 활용
  - pseudo-label의 품질을 더 잘 평가할 수 있는 새로운 지표 설계
- 효과적인 informed finetuning
  - 높은 품질 pseudo-label만 선별하여 모델 업데이트
- 모델 일반성
  - Whisper 외 Canary, SeamlessM4T 등 다른 speech foundation model에도 적용 가능

### 실험 세팅
- 다양한 target domain (잡음, 억양 등 14개 domain)에서 실험
- unlabeled data만 사용
- 평가 metric: WER(Word Error Rate)

### 주요 결과
- 평균 13.5% WER 상대 개선
- 일부 domain에서는 supervised adaptation upper bound에 근접
- 1시간 미만 unlabeled data만 필요 (데이터 효율성 높음)
- catastrophic forgetting 방지 효과도 확인
- **장점**
  - 소스 데이터 필요 없음 → practical, privacy-friendly
  - pseudo-label quality를 token 단위로 정교하게 평가
  - Whisper 등 다양한 model에 적용 가능
  - 작은 데이터만으로도 빠르게 적응 가능


## 13) Large Language Models are Efficient Learners of Noise-Robust Speech Recognition (20pages)
### 연구 문제 정의
- 기존 LLM 기반 GER은 깨끗한 음성에는 잘 동작하지만, 노이즈 환경에서의 강인성은 부족
- ASR에서 나온 N-best 후보 리스트가 noisy할 때 LLM으로 robust한 수정(correction)을 할 수 있을까?
- LLM에 “노이즈 상태”를 알려줘서 더 잘 고치게 할 방법 필요

### 기존 방법의 한계
- LM rescoring은 단순히 hypothesis의 점수 재조정 → 근본적인 수정은 어려움
- 기존 GER은 LLM의 강력한 언어 능력만으로 수정 → 노이즈 환경에서는 불안정
- 오디오 encoder의 audio embedding을 LLM에 바로 넣으면 cross-modal gap 때문에 tuning 성능 저하

### 제안 방법 : RobustGER
- **Language-space Noise Embedding (LSNE)**
  - N-best list의 diversity를 기반으로 언어적 noise condition 표현
  - Noise가 심할수록 → N-best diversity 커짐
  - Utterance-level diversity (전체 문장 의미 차이)
  - Token-level diversity (편집거리 중심 token 차이)
- **Audio Noise Distillation via MINE**
  - Mutual Information Neural Estimation(MINE) 활용
  - audio embedding의 noise 정보 → language embedding에 distill 
- **최종 Framework**
  - LLM fine-tuning 과정에서 LSNE를 noise conditioner로 투입
  - 목표: $Y = M_{H2T}(Y_N; -E_{LN})$

### 주요 실험 결과
- 최대 53.9% WER 감소
- Token-level embedding이 큰 효과 (WER metric에 직접 연관)
- Clean set에서도 30% WER 감소 → 일반화도 잘됨

### 논문의 장점 및 한계
- **장점**
  - LLM의 기존 능력을 Noise-robust ASR task로 효과적으로 확장
  - 모든 정보가 “언어적 표현” 안에서 처리 → cross-modal 문제 회피
  - 데이터 효율성 우수: 작은 training data로도 significant improvement
- **한계**
  - Context-based language correction과 noise-aware denoising trade-off 존재
  - CHiME-4에서 상대적으로 더 큰 개선 → dataset/domain bias 가능성 


## 14) It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition (12pages)
### 연구 문제 정의
- 기존 ASR 시스템은 N-best hypothesis list를 LLM이 받아 text-based 방식으로 오류를 수정(GER: Generative Error Correction)
- 하지만 LLM이 음향 정보를 사용하지 않기 때문에 데이터 불확실성이 증가하는 문제 존재

### 기존 방식의 한계
- LLM은 음향 정보 없이 text-only 기반으로만 학습되었음
- speech 신호 내 중요한 정보가 손실
- 기존 GER 방식은 이 한계를 그대로 안고 있음

### 제안 방법 : UADF(Uncertainty-Aware Dynamic Fusion)
- auto-regressive decoding 과정에서 late fusion 방식으로 동작
- LLM의 token-level 결정에 대해 uncertainty(불확실성)를 측정하고, 필요할 때 acoustic 정보를 동적으로 통합
- 구성 방안
  - LLM의 token-level 결정 분석 및 calibration
  - acoustic modality의 정보와 동적 융합 (dynamic fusion)
 
### 주요 효과
- 다양한 ASR task에서 기존 fusion 메커니즘보다 우수한 성능 달성
- WER(Word Error Rate) 개선
- 데이터 불확실성 완화 및 단일 modality 의존에서 오는 generalization 문제 해결
- audio-visual speech recognition에도 쉽게 적용 가능


## 15) Improving Speech Recognition with Prompt-based Contextualized ASR and LLM-based Re-predictor (4pages)
### 문제 배경
- 최근 ASR(자동 음성 인식) 시스템은 콜센터, 가상비서 등에서 널리 사용
- 하지만 발화 조건 악화, 문맥 정보 부족, 희귀 단어 인식 어려움 등의 한계 존재

### 제안 방법
- LLM(대형 언어 모델)과 prompt 메커니즘 통합
- 사전학습된 text encoder + task-specific text adapter로 문맥 정보를 효과적으로 반영
- LLM 기반 re-prediction으로 기존 n-best 결과 대신 개선된 최종 transcription 출력

### 실험 결과
- 기존 baseline ASR 대비 평균 상대적 WER 감소
  - 전통적 task: 27% ↓
  - 발화-문맥 task: 30% ↓
  - 단어-바이어싱 task: 33% ↓ 

## 16) Effective Text Adaptation for LLM-based ASR through Soft Prompt Fine-Tuning (5pages)
### 연구 배경
- 기존 ASR은 acoustic model + language model의 결합 구조였는데
- 최근에는 **LLM에 audio embedding을 prompt로 넣고 전사를 생성**하는 방식으로 발전
- **문제 상황**
  - 특정 도메인(예: 음악, 챗봇 등 entity-heavy domain)에서는 여전히 domain adaptation 필요
  - Domain adaptation을 위해서는 text-audio paired 데이터가 필요
    - text-only corpus만 있을 때 어떻게 효과적으로 LLM 기반 ASR을 domain adaptation할 수 있을까?
  - LLM 기반 ASR은 training 시 audio embedding을 prompt로 받는데
    - text-only corpus로 fine-tune하면 prompt가 없어서 condition mismatch 발생
    - 효과적 domain adaptation 어려움 
   
### 제안 방법
- 2단계 Soft Prompt Fine-Tuning
- Soft Prompt 학습
  - 도메인-specific pseudo audio embedding $S_{\zeta}$ 를 학습
  - audio encoder는 freeze
- Decoder Fine-Tune
  - 학습된 soft prompt S_{\zeta} 를 prompt로 사용하여 decoder fine-tune
- Inference : 실제 audio가 있으므로 soft prompt 사용 X

### 성과
- 성능 개선
  - 최대 9% Word Error Rate(WER) 감소
  - 최대 18% Entity Error Rate(EER) 감소
- Language Model(LM) fusion 추가 효과
  - soft prompt fine-tuned 모델 + domain-specific LM fusion
  - 추가 2-5% EER 개선  


## 17) Self-Train Before You Transcribe (4pages)
### 배경 및 문제
- ASR 시스템은 훈련 데이터와 테스트 데이터의 도메인이 다를 경우(domain mismatch) 성능이 크게 저하됨
- 기존 self-training 방법은 별도의 unlabeled target domain 데이터가 필요

### 제안 방법
- **Test-Time Adaptation (TTA) 방식**
  - 테스트 시점에서 테스트 recording 자체에 noisy student teacher training (NST)를 적용
  - 별도 adaptation dataset 필요 없음 → 데이터 수집 비용/노력 절감
- **Dynamic evaluation analogy**
  - 발화(utterance) 경계 넘어 context transfer → 긴 recording에서 유리
  - Local context 활용 → 모델이 현재 recording에 빠르게 적응 가능
 
### 주요 성과
- 다양한 dataset에서 최대 32.2% WER 개선
- 기존 self-training (separate adaptation data 사용)보다 더 큰 효과
