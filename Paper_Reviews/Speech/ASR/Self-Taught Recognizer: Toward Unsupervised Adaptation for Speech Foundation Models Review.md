# Self-Taught Recognizer: Toward Unsupervised Adaptation for Speech Foundation Models

## 요약 정리
### Problem
- 대형 ASR foundation model (예: Whisper, SeamlessM4T 등)은 다양한 도메인(잡음, 억양 등)에서 성능이 떨어지는 **domain shift 문제에 취약**
- 실제 서비스 환경에서는 **소스 도메인 데이터를 보존하거나 다시 사용하는 것이 어렵거나 불가능함** (프라이버시, 보안, 저장 비용 문제)
- 따라서 **source-free**, 즉 **소스 데이터 없이 unlabeled target data만으로 모델을 효과적으로 적응시키는 방법이 필요함**

### Contributions
1. **Source-free UDA 설정에서의 self-training 기반 ASR 적응 방식 제안**
  - Whisper 등 사전학습 모델만 활용, 소스 데이터 불필요
2. pseudo-label의 품질을 평가하기 위한 새로운 **token-level 지표 제안**
  - Softmax 기반 confidence의 한계를 보완하기 위해 attention 기반 정보를 결합한 STAR 지표 $$S_l$$ 설계
3. **Token-level re-weighting 및 utterance-level filtering을 통한 informed finetuning 전략**
  - 신뢰도 높은 토큰과 문장만 선택적으로 학습에 반영
4. **광범위한 도메인(노이즈, 악센트, 시나리오 등)에서 높은 데이터 효율성과 일반성 확인**
  - 다양한 대형 음성 모델과 음성 번역 태스크로의 확장 가능성 입증

### Method
- **Self-training 기반 도메인 적응**
  - unlabeled target 음성으로부터 pseudo-label 생성
  - token-level로 pseudo-label 품질 평가 → **STAR score $$S_l$$ 계산**
  - 각 토큰에 대해 신뢰도 기반 가중치 부여 → informed finetuning
- **STAR indicator 설계**
  - **Confidence score $$C_l$$**: 안정적이지만 over-confidence 문제
  - **Attentive score $$A_l$$**: 더 신뢰 가능하지만 수치 불안정성
  - 둘을 결합하여 상황에 따라 동적으로 weighting: $$S_l = S_{conflict} + S_{consistent}$$
- **Utterance-level filtering**
  - Gaussian noise를 주입해 모델의 예측 편차로 불확실성을 측정
  - 불확실성 높은 문장 제거 → 학습에 해로운 샘플 제거

### Experiments & Setup
- **모델**: Whisper-Large-V3 (15억 파라미터), OWSM, Canary, Parakeet 등
- **데이터셋 도메인**
  - **Noise**: CHiME-4, LibriSpeech-Freesound, RATS
	- **Accent**: CommonVoice (African, Australian, Indian, Singaporean)
  - **Scenario**: TED-LIUM, Switchboard, LRS2, ATIS, CORAAL
- **Metrics**: WER (Word Error Rate), BLEU (for translation)
- **세부 설정**
	- 1시간 미만의 unlabeled audio, learning rate 1e-5, 2 epochs, Adam optimizer
  - token-level loss re-weighting, utterance-level filtering percentile α=20

### Results
- **평균 13.5%의 WER 상대 개선**
-	일부 도메인에서는 supervised upper bound와 유사한 성능 도달
-	**1시간 미만 unlabeled data로도 학습 가능** (200–500 문장 수준)
-	**Catastrophic forgetting 없이 기존 도메인 성능 유지 또는 향상**
-	**Whisper 외 다양한 speech foundation model에도 적용 가능**
-	**음성 번역 task (FLEURS)에서도 BLEU 점수 1.2~2.2 개선**

### Limitations
-	STAR는 attention 기반 Transformer 구조에 특화되어 있음 → RNN 기반 모델에는 적용 어려움
-	매우 어려운 domain (예: RATS radio)에서는 성능 개선 한계 존재
-	STAR 지표의 성능은 hyperparameter 설정 (λ, τ)에 민감할 수 있음
-	반복 self-training에서 pseudo-label 품질이 나빠질 경우 성능 저하 가능성 있음


### Insights & Idea
-	기존 self-training이 신뢰도 불확실한 pseudo-label에 모두 동일한 가중치를 부여했다면, STAR는 **“토큰 단위의 품질 평가 + 재가중(weighting)”** 을 통해 더 안전하고 효과적인 자기학습을 실현
-	attention matrix가 단순한 예측 정보가 아닌, 언어적 의미와 문맥 정보를 품고 있음을 활용한 점에서 기계 번역 등 다른 seq2seq task로의 응용 가능성 높음
-	source-free UDA라는 현실적이고 시급한 문제를 해결하는 매우 practical한 접근법
-	Whisper 등 범용 대형 음성 모델을 특정 도메인에 경량화된 방식으로 빠르게 적응 가능 → 실제 제품 배포 및 개인화 시스템에서 유용


<br>  
  
## 0. Abstract
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




<br>  
  
## 1. Introduction
### 연구 문제 정의
- 사람의 음성은 개인의 특성, 억양, 발화 스타일, 배경 소음 등으로 인해 다양하고 예측 불가능한 환경으로 인해 복잡함
- 이로 인해 ASR 시스템은 **도메인 간 차이(domain shift)** 에 매우 취약

### 최근 ASR 기술 발전 및 한계
- Whisper (OpenAI), SeamlessM4T (Meta), Canary (NVIDIA) 등 **대규모 사전학습 기반 ASR 모델**들이 공개
- 이 모델들은 **거대한 범용 데이터에 대해 학습**되어 있지만, **특정 도메인에서는 여전히 성능이 낮음**

### 도메인 적응 어려움
- 타겟 도메인 데이터에 라벨 붙이기는 시간·비용이 큼
- 기존 UDA(Unsupervised Domain Adaptation) 방법은 소스 도메인 라벨/데이터를 요구 → 현실성 낮음
  - UDA: 라벨이 없는 타겟 도메인 데이터만 가지고, 소스 도메인에서 학습된 모델을 타겟 도메인에 적응시키는 기술

### 소스 프리(Source-Free) UDA 및 자기 학습(Self-Training) 개념
- 인간이 익숙하지 않은 음성 도메인에 직면했을 때 **'자기 주도적' 능력**을 보이는 것처럼, ASR 시스템도 유사하게 학습할 수 있음
  - 첫째, 사전 학습된 모델이 타겟 도메인 데이터에 대해 '의사 라벨(pseudo label)'을 생성
  - 둘째, 이 의사 라벨이 있는 데이터를 신뢰도와 함께 사용하여 모델을 적응 
- 본 논문에서는 인간의 음성 인식 과정과 유사하게 어떠한 소스 데이터 없이도 사전 학습된 Whisper 모델 활용
  - **타겟 도메인의 소량의 라벨링되지 않은 데이터를 사용**하여 도메인별 음성 인식기로 적응시키는 '소스 프리 UDA' 문제 연구
  - 소스 데이터 사용 없이 ASR 모델을 적응시켜 광범위한 컴퓨팅 자원 소비를 피하고
  - 적은 양의 음성 샘플만으로 타겟 도메인에서 ASR 성능을 크게 향상시킬 수 있다는 점에서 중요한 가치를 지님
 
### 제안 방법 : STAR(Self-TAught Recognizer)
- **핵심 질문**: "Ground-truth 라벨이 없는 상황에서 자기 학습을 안내하기 위해 의사 라벨의 품질을 어떻게 평가할 것인가?"
  - 기존 ASR 모델의 **'신뢰도 점수(confidence scores)'** 는 소프트맥스(softmax) 함수에서 파생된 유사 사후 확률(pseudo posterior probabilities)로 근사화되며, 이는 **'과신(over-confident)' 문제로 인해 신뢰할 수 없을 수 있음**
- STAR는 자동 회귀 디코딩(auto-regressive decoding) 중 얻어지는 자기 어텐션(self-attention) 행렬 활용
  - 이 어텐션 행렬은 음성 입력에 기반하며 언어적 적합성에 초점을 맞추므로 의사 라벨 품질 측정에 더 신뢰할 수 있는 지표가 될 수 있음
  - 그러나 어텐션 스코어(attentive score)는 수치적 불안정성을 보일 수 있음
- STAR는 이러한 문제들을 해결하기 위해 신뢰도 점수와 어텐션 스코어의 장점을 통합하는 새로운 지표 제안
  - 이 지표는 이후의 미세 조정(finetuning) 프로세스를 안내하여 '지시적 적응(instructive adaptation)'을 수행
  - 지시적 적응: 모델을 무작위가 아닌, 신뢰도 높은 pseudo-label을 기반으로 학습시키는 방식

### 주요 결과
- 14개 타겟 도메인에 걸쳐 평균 13.5%의 상대적인 단어 오류율(Word Error Rate, WER) 감소를 달성했으며, 때로는 지도 학습 적응의 상한 성능에 근접
- 소스 도메인 데이터를 다시 불러오지 않고도 적응된 모델이 '파멸적 망각(catastrophic forgetting)' 문제에 빠지는 것을 방지
  - 파멸적 망각: 딥러닝 모델이 새로운 도메인에 적응하면서, **기존에 잘하던 도메인 성능이 급격히 저하되는 현상**
- 1시간 미만의 라벨링되지 않은 데이터만으로도 높은 데이터 효율성을 보이며, 인식 및 번역 작업에서 다른 대규모 음성 모델에도 원활하게 적용 가능

### 본 논문의 기여
- 실제 애플리케이션에 가장 근접한 설정인 ASR의 소스 프리 UDA에 중점을 두어, 사전 학습된 음성 파운데이션 모델과 라벨링되지 않은 음성 샘플만을 사용하여 특정 타겟 도메인에 적응시키는 방법을 제시함
- 소음, 억양 및 특정 시나리오를 포함한 광범위한 타겟 도메인에서 음성 파운데이션 모델의 도메인별 능력을 향상시킴
- 데이터 효율성 및 일반성 분석을 통해 음성 비서의 점진적 업데이트와 같은 실제 애플리케이션에서의 잠재력을 보여줌





<br>  
  
## 2. Related Work
### ASR에서의 비지도 도메인 적응 (Unsupervised Domain Adaptation, UDA)
- 타겟 도메인에서 **라벨링된 음성(transcriptions)** 확보는 시간과 비용이 많이 듬
- 따라서 기존 연구들은 **소스 도메인(label 포함) 데이터**를 기반으로, **라벨 없는 타겟 도메인 데이터**에 적응하는 방법 고안

- **주요 기법**
  - 타켓 음성 시뮬레이션: 타겟 도메인 스타일의 데이터를 합성해서 적응
  - 적대적 학습 (Adversarial Learning): 도메인 불변 표현을 학습하여 domain shift 최소화
  - Teacher-Student Learning: source 모델이 label 제공 → target 모델 학습
  - Self-supervised 모델 활용: wav2vec2 등으로 pseudo-label 생성하여 라벨 없이 적응
- **한계**
  - 위 접근법들은 대부분 **소스 도메인의 레이블이 필요**하므로 완전한 비지도 학습이 아님 → **semi-supervised에 가까움**
  - 소스 데이터 접근 자체가 어려운 경우가 많음 → **현실 적용 제약 존재**

### 소스 프리 비지도 도메인 적응 (Source-free Unsupervised Domain Adaptation)
- **소스 데이터를 완전히 배제**하고, 사전 학습된 모델만을 가지고 **라벨 없는 타겟 도메인에 적응하는 방법론**
- **필요성**
  - 소스 데이터에 개인정보나 민감 정보 포함 가능성
  - 저장/전달/재사용의 법적·윤리적 제약 → 소스 없이 적응하는 방법이 중요
- **주요 접근법**
  - Self-supervised Knowledge Distillation: 사전 학습된 모델의 내부 지식만 활용하여 적응
  - Contrastive Learning: 타겟 도메인 간 유사·비유사 샘플 간 간극을 학습
  - Hidden Structure Mining: 라벨 없이 데이터 간 구조 관계를 추출하여 학습
  - Uncertainty-guided Adaptation: 모델의 예측 불확실성을 활용해 신뢰도 높은 샘플 선택 (→ STAR가 속한 범주)
 
- 최근에는 **auto-regressive decoder 기반 불확실성 추정**에 대한 연구가 많아졌으나,
→ **Source-Free UDA에는 거의 적용되지 않음**

### STAR의 위치 및 차별성
- **문제 인식**
  - Whisper 등 speech foundation model은 사전 학습에 수십만 시간의 데이터를 사용
    - 소스 도메인의 범위가 넓고 복잡
  - 이 데이터를 보존·공유·재학습하는 것이 현실적으로 불가능
- **STAR의 핵심 아이디어**
  - 소스 데이터가 없는 환경에서, **모델이 생성한 pseudo-label의 품질을 평가**
  - 신뢰할 수 있는 토큰 중심으로 학습을 유도 → **지시적 자기학습 (instructive self-training)**
- **효과**
  - source 데이터 불필요
  - ground-truth vs pseudo-label 적응 성능 차이를 크게 줄임
  - 현실적인 Source-free UDA 달성
  - ASR 기반 실제 서비스에 적용 가능성↑ (예: 음성 비서 도메인 적응) 




<br>  
  
## 3. Methodology
### 3.1 Problem Setup
#### ASR 공식화 (ASR Formulation)
- ASR 모델 f는 음성 입력 $$x \in \mathbb{R}^T$$를 텍스트 시퀀스 $$y \in \mathbb{R}^L$$로 변환
- 학습은 **teacher-forcing + cross-entropy loss**로 이루어짐
  - $$\mathcal{L}{\text{ASR}}(x, y) = \sum{l=1}^{L} - \log P_{\theta} (y_l|y_{1:l-1}, x)$$
- 핵심: **순차적(auto-regressive) 예측**으로 **이전 토큰에 기반해 다음 토큰 예측**

#### UDA 설정
- 목적: 라벨 없는 **타겟 도메인 데이터** $$X^{(t)}$$ 만 사용해, **소스 모델 $$f^{(s)}$$** 를 **타겟 도메인 $$D^{(t)}$$** 에 맞게 적응
- Source-free UDA 시나리오
  - 소스 도메인 데이터 $$\{X^{(s)}, Y^{(s)}\}$$는 사용할 수 없음
  - 사전 학습된 모델 $$f^{(s)}$$만 주어진 상황에서 **라벨 없는 음성 $$X^{(t)}$$** 만으로 적응해야 함 

#### Self-Training 전략 (STAR의 핵심 구조)
- **Pseudo-label 생성**
  - unlabeled 데이터 $$X^{(t)} = \{x_i^{(t)}\}_{i=1}^{N^{(t)}}$$를 소스 모델 $$f^{(s)}$$에 넣어 pseudo-label $$\hat{Y}^{(t)} = \{\hat{y}i^{(t)}\}{i=1}^{N^{(t)}}$$ 생성
- **Informed Finetuning (정보 기반 미세 조정)**
  - $$\{X^{(t)}, \hat{Y}^{(t)}\}$$를 이용해 모델을 타겟 도메인으로 적응시키는 학습 수행
  - $$\mathcal{L}{\text{ST}}(X^{(t)}, \hat{Y}^{(t)}) = \sum{i=1}^{N^{(t)}} \mathcal{L}_{\text{ASR}}(x_i^{(t)}, \hat{y}_i^{(t)})$$


### 3.2 Token-level Assessment and Re-weighting
#### 배경 및 목표
- Self-training에서 생성된 pseudo-label의 품질은 다양
- 따라서 **각 토큰별로 품질을 평가하고, 좋은 토큰은 크게, 나쁜 토큰은 작게 학습에 반영해야 함**
- 이 유사 레이블의 품질을 평가하고, 이를 바탕으로 모델 업데이트를 안내하는 "지표 (indicator)"를 개발하는 것이 이 섹션의 주요 목표
- 이는 토큰 레벨 (단어 또는 하위 단어 단위)과 발화 레벨 (전체 문장 단위)에서 모두 이루어짐

#### Confidence Score의 한계
- $$C_l = \max P(\hat{y}_l|\hat{y}_{1:l-1}, x, \theta^*)$$
  - $$\hat{y}{1:l-1}$$: 이전에 예측된 토큰들
  - $$\theta^*$$: 모델의 학습된 파라미터
- 기존의 모델에서는 각 토큰 예측에 대한 "confidence score" $$\(C_l\)$$ (확신 점수)를 사용
- 이는 자동 회귀 디코딩 (auto-regressive decoding) 시 이전 예측된 토큰들에 기반하여 현재 토큰 $$\hat{y}_l$$의 후방 확률 (posterior probability) 중 가장 높은 값으로 정의
- **STAR 기반 재가중 학습 손실**: $$eLASR(x, \hat{y}) = \sum_{l=1}^{L} -\log P_{\theta}(\hat{y}_l|\hat{y}_{l-1:1}, x) \cdot C_l$$
  - L: 출력 텍스트 시퀀스의 길이
- **문제점**
  - 자동 회귀 디코딩에서 확신 점수가 예측 정확도를 정확하게 반영하지 못하는 경향 존재
  - 오류 누적 (error accumulation) 및 과도한 확신 (over-confident) 문제 때문 

#### Attentive Score
- $$A_l = \sum_{j=4}^{l} W_{l,j} + \sum_{i=l+1}^{L} W_{i,l}$$
  - 자동 회귀 디코딩 과정에서 얻어지는 "self-attention matrix" W를 활용하여 "attentive score" $$A_l$$를 제안
  - 여기서 $$W_{l,j}$$는 l번째 토큰과 j번째 토큰 사이의 어텐션 가중치
  - 첫 3개 토큰은 프롬프트이므로 4번째부터 시작
- **현재 토큰이 과거/미래 토큰과 얼마나 의미적으로 연결**되는지 측정
- **Figure2**
  - 정확하게 디코딩된 토큰은 **어텐션 가중치가 대각선에 집중**
  - 잘못된 토큰은 특정 프롬프트 토큰에 집중되는 경향을 보임 
- **문제점**
  - 하지만 $$A_l$$은 "for"와 "housing"처럼 모두 올바른 토큰임에도 불구하고 점수가 크게 다르게 나타나는 등 수치적 안정성이 떨어지는 문제 존재
  - 전치사나 명사처럼 단어의 역할이 다를 때 어텐션 패턴이 달라지기 때문

#### STAR Indicator (신뢰성 및 안정성 확보)
- 저자들은 $$C_l$$의 안정성과 $$A_l$$의 신뢰성을 통합하여 새로운 지표인 "STAR indicator" $$S_l$$를 제안
- **충돌 시 (conflict)**: $$C_l$$과 $$A_l$$이 유사 레이블에 대해 상충되는 값을 보일 때
  - 신뢰성이 높은 $$A_l$$ 지표로 선택
  - $$S_{conf} = [\sigma(A_l^2/C_l - \lambda) + \sigma(C_l^2/A_l - \lambda)] \cdot A_l$$
  - $$\lambda$$: 임계값 (hyperparameter)
  - $$A_l^2/C_l$$ 또는 $$C_l^2/A_l$$가 $$\lambda$$보다 클 때 충돌로 간주
- **일관 시 (consistent)**: $$C_l$$의 안정성을 활용하여 $$A_l$$의 스케일 조정
  - Focal Loss에서 영감을 받음
  - $$S_{cons} = [\sigma(\lambda - A_l^2/C_l) \cdot \sigma(\lambda - C_l^2/A_l)] \cdot A_l \cdot e^{(C_l - A_l)/\tau}$$
  - $$\tau$$: 온도 (temperature) 파라미터
- **최종 STAR Indicator**: 두 경우를 결합하여 최종 $$S_l$$을 정의
  - $$S_l = S_{conf} + S_{cons}$$ 
- **학습에 활용**: 최종적으로, 이 $$S_l$$ 지표는 재가중치 ASR 손실 함수에 사용되어 모델의 학습 과정을 안내
  - $$eLASR(x, \hat{y}) = \sum_{l=1}^{L} -\log P_{\theta}(\hat{y}_l|\hat{y}_{l-1:1}, x) \cdot S_l$$
  - Figure 5에 나타난 것처럼 신뢰성과 안정성을 모두 갖추어 유사 레이블 품질을 평가하고 학습을 효과적으로 안내하는 데 기여



### 3.3 Utterance-level Filtering
#### 목표
- 전체적인 품질이 낮은 예측된 발화(utterance)를 제거하여 이후의 모델 적응 과정에 해로운 영향을 미 미치지 않도록 하는 것
- 이 논문에서는 의사 레이블(pseudo label)의 발화 수준(utterance-level) 품질을 평가하기 위한 몇 가지 접근 방식을 소개
  - 불확실성(uncertainty)이 높을수록 일반적으로 생성된 시퀀스의 품질이 낮음을 의미

#### Monte Carlo Sampling (Gaussian Noise Disturbances)
- 모델의 불확실성을 측정하기 위해 사용
- Whisper 모델은 학습 시 dropout을 사용하지 않으므로, 이 논문에서는 Gaussian 노이즈를 사용하여 모델 가중치를 무작위로 교란하는 방식을 제안
- **과정**
  - 입력 음성 $$x$$에 대해 한 번의 정방향 디코딩(forward decoding)을 수행하여 기본 전사(base transcription) $$\hat{y}$$를 얻음
  - Whisper 모델의 가중치를 Gaussian 노이즈로 무작위로 교란하고, 이 디코딩 과정을 K번 반복하여 의사 전사 목록 $$\{\hat{y}_k \}^K_{k=1}$$를 생성
  - 각 의사 전사 $$\hat{y}_k$$와 기본 전사 $$\hat{y}$$ 사이의 편집 거리(Edit Distance, ED)를 계산
    - 이는 교란이 Whisper 디코딩에 미치는 영향 
- **수식**
  - $$U(x, \hat{y}) = \frac{1}{K} \sum_{k=1}^{K} ED(\hat{y}, \hat{y}_k)$$
  - $$U(x, \hat{y})$$: 입력 x와 기본 전사 $$\hat{y}$$에 대한 모델의 불확실성(또는 강건성의 역수)
    - 값이 클수록 불확실성이 높습니다
  - K: Gaussian 노이즈로 교란된 디코딩 반복 횟수
  - $$ED(\hat{y}, \hat{y}_k)$$: 기본 전사 $$\hat{y}$$와 k번째 의사 전사 $$\hat{y}_k$$ 간의 편집 거리

- K개의 의사 레이블을 얻은 후, 그들의 다양성을 조사하여 모델의 불확실성을 추가로 평가
  - 만약 목록에 반복이 많다면, 모델이 해당 음성을 전사하는 데 더 확신이 있다는 것을 의미
- 이 논문에서는 중복 제거 후의 발화 수 l과 $$U(x, \hat{y})$$의 수치 곱을 최종 발화 수준 품질로 사용
  - 이 점수를 바탕으로 하위 $$\alpha\%$$의 샘플(불확실성이 높은 샘플)을 제거 

#### Beam Search Decoding 및 Consensus Decoding (대안)
- 이 논문에서는 Beam Search Decoding과 Consensus Decoding Finding consensus in speech recognition을 발화 수준 필터링의 대안으로도 실험
  - Table 8의 결과에 따르면, Gaussian 노이즈를 사용한 방법이 Beam Search Decoding이나 Consensus Decoding보다 약간 더 나은 성능을 보임 




<br>  
  
## 4. Experimental Setup
### 4.1 ASR Domains
- STAR의 일반적인 효과를 검증하기 위해 다양한 ASR(Automatic Speech Recognition) 도메인에서 실험 진행
- **노이즈 음성(Noisy Speech)**: CHiME-4, LibriSpeech-FreeSound, RATS 데이터셋 사용
  - 이 데이터셋들은 버스, 카페, 보행자 구역, 길거리 교차로, 웅성거림(babble), 자동차, 공항 소음 등 다양한 유형의 노이즈와 무선 통신 노이즈 포함
- **악센트 음성(Speaker Accents)**: CommonVoice 데이터셋에서 아프리카, 호주, 인도, 싱가포르 악센트의 영어 음성 데이터 사용
- **특정 시나리오(Specific Scenarios)**: BBC 강연, TED 강연, 전화 통화와 같은 특정 시나리오

### 4.2 Configurations
- 주요 실험에는 15억 개의 파라미터를 가진 Whisper-Large-V3 모델 사용
- 이 모델은 680k 시간의 웹 스케일 데이터로 사전 학습됨
- 모델은 Adam 옵티마이저를 사용하여 1e−5의 초기 학습률로 2 epoch 동안 파인튜닝됨
- 배치 크기는 1로 설정되었고, 16단계의 그래디언트 축적(gradient accumulation) 사용
- 하이퍼파라미터는 임계값 $$\lambda$$를 2, 온도 $$\tau$$를 10으로 설정
- 발화 수준 필터링(utterance-level filtering)의 백분율 $$\alpha$$는 20으로 설정되었으며, 이는 다양한 데이터셋에서 일관된 효과를 보임


<br>  
  
## 5. Results and Analysis
### 5.1 Effectiveness of STAR
#### Main Results(주요 결과)
- STAR는 노이즈, 악센트 및 특정 시나리오를 포함한 다양한 ASR(자동 음성 인식) 도메인에서 Whisper 모델의 성능을 향상
- 특히, 14개 대상 도메인에서 평균 13.5%의 상대적인 단어 오류율(WER) 감소 달성
- 일부 도메인에서는 STAR가 심지어 실제 레이블을 사용한 지도 학습(supervised adaptation)의 상한 성능에 근접하기도 함

#### Analysis of Catastrophic Forgetting(재앙적 망각)
- 기존의 Source-free ASR 적응에서 흔히 발생하는 재앙적 망각 문제(이전에 학습한 지식을 잃어버리는 현상)를 STAR가 방지하며, 오히려 다른 도메인에서의 성능까지 향상시키는 것을 보여줌

#### Analysis of Indicators(지표 분석)
- **신뢰도 점수(Confidence Score)**: 신경망이 예측하는 후방 확률의 가장 높은 값
  - 하지만, 이는 예측 정확도를 정확하게 반영하지 못하며, 특히 자동 회귀 디코딩(auto-regressive decoding)에서 오류 누적 및 전파로 인해 신뢰할 수 없는 경향 존재
- **주의 점수(Attentive Score)**: 자동 회귀 디코딩 중에 얻어지는 자기-주의(self-attention) 행렬 W 기반
  - 이는 음성 입력 및 언어적 수용 가능성(linguistic acceptability)과 직접적인 관련이 있어 신뢰도가 더 높음
  - 이는 현재 토큰과 모든 의사 토큰 간의 전역 의미론적 상관관계를 나타냄
- **STAR 지표(STAR Indicator)**: 신뢰도 점수 Cl과 주의 점수 Al의 장점을 통합하여 신뢰성과 안정성을 모두 갖춘 새로운 지표를 제안
  - 충돌하는 경우(두 점수가 모순될 때)에는 Al을 선택하고, 일관된 경우(두 점수가 일관될 때)에는 Cl의 안정성을 사용하여 Al의 스케일 조정
  - 이는 Sl = Sconfl + Sconsl로 표현
  - 이 STAR 지표를 사용하여 손실 함수 eLASR(x, ˆy) = Σl=1^L − log Pθ (ˆyl|ˆyl−1:1, x) ∗ Sl과 같이 각 토큰에 다른 가중치를 부여하여 informed finetuning을 수행
- STAR 지표는 pseudo-label 품질을 평가하는 데 있어 신뢰성과 안정성을 모두 제공하여 finetuning 프로세스를 효과적으로 안내

- **발화 수준 필터링(Utterance-level Filtering)**: 낮은 품질의 pseudo-label이 포함된 발화(utterances)를 제거하여 후속 적응에 해로운 영향을 줄이는 것을 목표
  - 모델의 불확실성을 평가하고, 불확실성이 높은 샘플을 제거



### 5.2 Generality of STAR
- **다양한 음성 파운데이션 모델로의 일반화**: STAR는 Whisper 모델뿐만 아니라 OWSM, Canary, Parakeet-TDT와 같은 다른 유명한 음성 파운데이션 모델에도 일관된 성능 향상(10% 이상의 상대적 WER 감소)을 보임
- **음성 번역(ST) 작업으로의 일반화**: ASR 외에 음성 번역 작업(FLEURS X→En 데이터셋)에서도 평균 1.2 BLEU 이상의 향상을 보여, STAR가 다른 sequence-to-sequence 작업으로도 확장될 가능성이 있음을 입증


### 5.3 Ablation Study
- **데이터 효율성(Data Efficiency)**: STAR 적응에 필요한 비레이블링 데이터 양을 분석한 결과, 최적의 성능을 달성하기 위해 200~500 문장(1시간 미만의 비레이블링 음성 데이터)만 필요
  - 이는 실제 시나리오에서 데이터 수집 및 레이블링에 드는 시간과 노력을 크게 절약할 수 있음을 의미
- **모델 크기(Model Size)**: 다양한 Whisper 모델 크기(base.en부터 large-v3까지)에 STAR를 적용한 결과, 모든 크기에서 일관되고 상당한 성능 향상을 보여주며, 이는 자원 제약이 있는 환경에서의 STAR의 잠재력 시사
- **파인튜닝 접근 방식(Finetuning Approach)**: 전체 파인튜닝, 인코더-온리, 디코더-온리, LoRA(Low-Rank Adaptation) 등 다양한 파인튜닝 접근 방식이 유사한 효과를 보이며, 이는 다양한 설정에서 유연한 선택지 제공



<br>  
  
## 6. Conclusion
- 이 논문은 Self-TAught Recognizer(STAR)라는 새로운 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 프레임워크 제안

- **STAR는 비지도 적응 프레임워크**: STAR는 레이블이 없는 데이터를 활용하여 노이즈나 억양과 같은 다양한 대상 도메인에서 자동 음성 인식(ASR) 시스템의 견고성 향상
- **새로운 품질 지표 도입**: STAR는 거짓 레이블(pseudo labels)의 품질을 평가하고 모델 업데이트를 안내하는 새로운 지표 제시
  - 이는 ground truth(실제 정답 레이블) 없이 토큰 수준의 품질을 평가할 수 있도록 도움
- **탁월한 성능 향상을 보입니다**: STAR는 14개 대상 도메인에서 평균 13.5%의 상대적인 단어 오류율(Word Error Rate, WER) 감소를 달성했으며, 일부 경우에서는 지도 학습(supervised adaptation)의 상한선 성능에 근접
- **치명적인 망각 문제 방지**: STAR는 원본 도메인 데이터를 다시 불러오지 않고도 적응된 모델이 치명적인 망각(catastrophic forgetting) 문제에 빠지는 것을 방지
- **높은 데이터 효율성**: STAR는 1시간 미만의 레이블 없는 데이터만으로도 최상의 성능을 달성할 수 있어 데이터 효율성이 매우 높음
- 뛰어난 범용성: STAR는 음성 인식(recognition) 및 음성 번역(translation) 작업에서 다른 대규모 음성 모델에도 원활하게 적용될 수 있음을 보여줌



















