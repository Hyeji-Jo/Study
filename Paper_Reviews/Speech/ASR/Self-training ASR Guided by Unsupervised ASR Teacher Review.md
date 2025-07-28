# Self-training ASR Guided by Unsupervised ASR Teacher
## 요약 정리
### Problem
- 기존 Self-training 방식
  - 초기 teacher를 labeled data로 학습해야 하므로 **라벨 의존성이 있음**
  - 소량의 라벨로 학습된 teacher는 **overfitting → noisy pseudo-label** 생성
  - **multi-stage 반복 학습 구조로 인해 학습 비용/시간 부담**이 큼 
- 또한 SSL 방식(예: Data2vec2)은 pseudo-target이 어떤 정보를 담고 있는지 해석이 어려움
  
### Contributions
- UASR(Wav2vec-u2)를 teacher로 사용하는 self-training 구조 제안

### Method


### Experiments & Setup


### Results


### Limitations


### Insights & Idea


<br>  
  
## 0. Abstract
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
    - phonetic : 사람이 말하는 **소리 단위(음소: phoneme)** 에 관한 정보
      - cat -> /k/ /æ/ /t/ 
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
 



<br>  
  
## 1. Introduction
### Self-supervised Learning (SSL)
- 최근 음성 인식(ASR)에서는 라벨 없이 학습할 수 있는 **Self-supervised Learning (SSL)** 기법들이 큰 성과를 보임
  - **Wav2vec2**: contrastive loss 사용, 음성 특징을 분리
  - **HuBERT**: MFCC 기반 클러스터로 음운 정보 학습
  - **Data2vec2**: self-distillation 방식으로 context representation 예측
- 라벨 없이 음성에서 의미 있는 표현을 학습하지만, **pseudo-target의 정보가 불분명하다는 한계 존재**

### Self-training
- 초기 ASR 모델을 작은 labeled 데이터로 학습
- 이 모델을 teacher로 사용해서 **unlabeled 데이터에 pseudo-label 생성**
- 이 pseudo-label로 student 학습 → 반복
- 장점 : pseudo-label이 실제 텍스트이므로 정보가 명확함 → ASR에 유리함
- 단점 : 초기 teacher가 overfitting되면 나쁜 pseudo-label 생성

### 제안 핵심
- 기존 self-training의 문제점 2가지를 해결!
- **문제 1 - 초기 teacher가 labeled data를 필요로 함**
  - UASR(비지도 음성 인식) 모델을 teacher로 사용 (labeled data 없이도 학습 가능)
- **문제 2 - 초기 teacher가 overfitting → noise label 발생**
  - phonetic supervision을 intermediate layer에 적용해서 student(Data2vec2)가 더 robust하게 학습되도록 설계
 


<br>  
  
## 2. Backgrounds
### 2.1 Self-supervised Learning: Data2vec2
- 음성 인식을 위한 **자기지도 학습(Self-supervised learning, SSL)** 모델의 한 종류
- **자기-증류(self-distillation)** 전략을 사용하여 학습하며, 학생(student) 모델과 교사(teacher) 모델이 **지속적으로 업데이트되는 방식**
  - **Self-distillation** : 현재 student 모델이 과거의 자기 자신(teacher)을 모방하면서 학습하는 방식
    - teacher는 실제 라벨 없이도 target을 제공
    - student는 teacher가 만든 타겟을 따라하도록 학습
    - 그래서 labeled data가 필요 없음! → **self-supervised**

#### 모델 구조
- Transformer 인코더 기반
- **학생 모델 (Student)**
  - 음성 입력을 받아 representation을 학습하는 주체
  - **입력**: 음성 파형 X
  - **CNN 인코더**: $$X \rightarrow Z = [z_1, z_2, \dots, z_T]$$
    - 여기서 Z는 **잠재 음성 표현 (latent representation)**, T는 프레임 수
    - waveform 같은 raw audio를 바로 Transformer에 넣을 수 없기에 CNN으로 특징을 추출 후 Transformer에 삽입
  - **Masking**: Z의 일부 프레임을 **무작위 마스킹**
    - 마스킹된 상태로 Transformer에 입력됨
  - **Transformer 인코더**: 자기-어텐션(Self-attention)을 사용해 **local + global 관계를 학습**
    - 문맥 표현(Context Representation) $$C = [c_1, …, c_T]$$
  - **디코더**: 이 문맥 표현을 최종 예측값 $$f_t(X)$$로 변환
- **교사 모델 (Teacher)**
  - student의 과거 상태를 따라가며, 학습 목표(의사 타겟)를 생성
  - 구조는 **student와 동일한 Transformer 기반**
  - 하지만 직접 학습하지 않음!
    - 대신 **학생 모델의 파라미터를 지수 이동 평균(EMA) 방식으로 업데이트**

#### 지수 이동 평균 (EMA) 수식
- $$\Delta \leftarrow \tau \Delta + (1 - \tau) \theta$$
  - $$\Delta$$: teacher의 파라미터
  - $$\theta$$: student의 현재 파라미터
  - $$\tau$$: 업데이트 비율 (보통 0.999 등, 거의 변화 없게 설정) 
- 불안정한 학습(mode collapse)을 방지하고, 더 부드러운 타겟 생성 가능

#### 의사 타겟(Pseudo-target) 생성 방식
- teacher의 Transformer에서 상위 K개 레이어의 출력을 평균해서 예측해야 할 **문맥화된 타겟 y_t** 을 생성
  - Transformer의 레이어는 층마다 역할이 다름
    - 초기 층 (lower layer): waveform의 저수준 패턴 (예: 에너지, 주파수 등)
    - 중간 층: 음소, 운율, 짧은 시간 범위의 관계
    - 상위 층 (top layers): 더 긴 문맥 정보 (long-term dependency), 단어 수준 의미, 문장 전체 구조 등
    - **즉, 상위 레이어일수록 더 많은 시간 범위를 고려한 추상적인 표현을 갖게 됨**  
- $$y_t = \frac{1}{K} \sum_{l = L - K + 1}^{L} \hat{a}_t^l$$
  - $$\hat{a}_t^l$$: teacher의 l번째 Transformer 레이어 출력
  - L: 총 레이어 수 
- 단순한 local 정보만이 아니라 **long-term dependency가 반영된 타겟을 생성**

#### 손실 함수 (Loss Function)
- 학생 모델은 마스킹된 프레임에 대해 teacher가 만든 pseudo-target과 예측값 사이의 **MSE(Mean Square Error) 오차를 최소화**하도록 학습
- $$\mathcal{L}_{SSL}(y_t, f_t(X)) = (y_t - f_t(X))^2$$
- 즉, student가 teacher가 만든 표현을 정확히 재현하도록 유도


### 2.2 Unsupervised Speech Recognition: Wav2vec-u2
- 짝을 이루지 않은(unpaired) 음성 데이터와 텍스트 데이터를 가지고 ASR 학습하는 방법
  - 기존 supervised ASR은 음성과 정답 텍스트 쌍이 필요했지만
  - UASR은 그게 없어도 학습 가능
 
#### Wav2vec-u2의 구조
- GAN (Generative Adversarial Network) 구조를 따름
- **Generator (G)**
  - 음성 입력으로부터 **phoneme sequence를 생성**함 (fake sample)
- **Discriminator (D)**
  - Generator가 만든 phoneme이 **진짜인지 가짜인지 판별함**

#### 학습 흐름
- 음성 → SSL 모델 → context representation C
  - SSL로 사전 학습된 Wav2vec2에서 뽑은 Transformer 출력값을 사용함
  - 즉, C는 **이미 phonetic 정보가 어느 정도 담긴 feature**
- Generator: C → fake phoneme sequence G(C)
- 텍스트 → 외부 phonemizer → real phoneme sequence P_r
- Discriminator는 G(C)와 P_r를 비교하며 둘을 구분하도록 학습됨
- 즉, G는 점점 더 자연스러운 phoneme sequence를 생성하게 됨

#### Loss
- GAN loss + 보조 손실(regularization)을 더한 형태
- $$\min_G \max_D \; \mathbb{E}_{P_r}[\log D(P_r)] - \mathbb{E}_C[\log(1 - D(G(C)))] \\ - \lambda L_{gp} + \gamma L_{sp} + \eta L_{pd}$$
  - 첫 두 항 - 전형적인 GAN loss (real vs fake 분류)
  - $$L_{gp}$$ - gradient penalty: D의 안정적 학습 보장
  - $$L_{sp}$$ - smoothness penalty: phoneme 예측이 부드럽게 연결되도록
  - $$L_{pd}$$ - phoneme diversity: 다양한 음소를 생성하도록 유도
  - 하이퍼파라미터 $$\lambda, \gamma, \eta$$는 각 손실 항의 중요도를 조절
 


<br>  
  
## 3. Proposed Method
### 3.1 UASR Teacher-guided Self-training
<img width="377" height="109" alt="image" src="https://github.com/user-attachments/assets/36a8af28-2fa5-426e-839a-d75d1a1267e8" />

#### 기존 방식의 한계
- 초기 teacher가 labeled data로 학습됨 -> 소량의 데이터에 과적합(overfitting) 발생
- pseudo-label이 noisy -> student 학습에 악영향
- 반복적인 multi-stage 학습 필요 -> 시간과 비용이 많이 듦

#### 제안된 방법 : UASR로 시작하는 Self-training
- unlabeled 음성과 text → Wav2vec-u2(UASR)로 teacher 학습
- 이 teacher가 student에게 phonetic 정보를 지도
- student(Data2vec2)는 이 정보를 활용해 representation 학습


### 3.2 Phonetic Supervision on Intermediate Layer
<img width="377" height="368" alt="image" src="https://github.com/user-attachments/assets/6022afd3-cc45-4fae-93a7-d7acb2ce0d91" />

#### 왜 해당 방식이 필요한가?
- **일반 self-distillation의 한계**
  - 기존 Data2vec2는 teacher의 상위 layer 출력을 target으로 사용함
  - 그러나 이 target이 무슨 정보를 담고 있는지는 해석 불가능함 (음운 정보인지, 의미인지 알 수 없음)
- **UASR teacher가 가진 phonetic 정보를 중간 층에 직접 supervision**
  - ASR은 음성 → phoneme → 단어의 과정을 따르므로,
  - 중간 representation에서 phoneme-level 정보가 잘 표현되는 게 중요

#### 기존 Data2vec2 loss (self-distillation)
- $$\mathcal{L}_{SSL}(y_t, f_t(X)) = (y_t - f_t(X))^2$$
  - teacher가 생성한 high-level context target $$y_t$$
  - student가 예측한 $$f_t(X)$$와 비교 (MSE loss) 

#### intermediate loss (phonetic supervision)
- $$\mathcal{L}_{distill}(y_t^{low}, f_t^{low}(X)) = (y_t^{low} - f_t^{low}(X))^2$$
  - $$y_t^{low}$$: UASR teacher가 생성한 phoneme-level 중간 표현
  - $$f_t^{low}(X)$$: student의 중간 transformer layer 출력값을 decoding한 결과 

#### 최종 Loss
- $$\mathcal{L}{total} = \mathcal{L}{SSL} + \kappa \cdot \mathcal{L}_{distill}$$
  - $$\kappa$$: 두 손실 간의 균형을 조절하는 가중치 

#### 모델 파라미터 업데이트 방식
- 상위 층 (L-th 이상) : $$\theta_{high} \gets \theta_{high} + \alpha \cdot \nabla \mathcal{L}_{SSL}$$
- 중간층 이하 (L-th 이하) : $$\theta_{low} \gets \theta_{low} + \alpha \cdot \nabla \mathcal{L}{SSL} + \beta \cdot \nabla \mathcal{L}{distill}$$
- 즉, **중간 층은 phonetic 정보 + 전체 목적 모두를 고려**해 학습되고, **상위 층은 기존 context target만 따라감**

#### 왜 intermediate layer인가?
- Transformer는 **layer가 높아질수록 정보의 추상화 수준이 올라감**
- 논문은 **phonetic 정보가 중간 층(L-th)에 가장 잘 나타난다고 주장**함
- 따라서, 그 층을 phonetic 정보의 “경계”로 정의하고 supervision을 줌



<br>  
  
## 4. Experimental Settings
### 4.1 Database
- **Pre-training용 unlabeled 데이터**
  - LibriSpeech 960시간 전체 사용
  - train-clean-100, train-clean-360, train-other-500
- **Fine-tuning용 labeled 데이터**
  - LibriSpeech 100시간만 사용 (train-clean-100)
  - low-resource 환경 실험 (라벨 부족 상황 가정)
- **평가 데이터셋**
  - dev-clean / dev-other (validation)
  - test-clean / test-other (최종 평가용)   

### 4.2 Training UASR Teacher
#### 기본 구조
- **Wav2vec-u2** 모델을 사용
- 사전 학습된 Wav2vec2 Base (Fairseq) 모델을 기반으로 구성
- generator와 discriminator는 논문 [22] 구조 그대로 사용

#### 학습 데이터
- 음성: LibriSpeech 960시간
- 텍스트: LibriSpeech Language Model (LS-LM) corpus
- Phonemizer: g2p-en 사용 (텍스트를 phoneme sequence로 변환)
- Silence token 추가: phoneme 사이에 무작위로 silence 추가 (50% 확률)

#### GAN 학습의 어려움
- **그래서 10번 학습해서 가장 좋은 모델 선택**
  - 기준: dev-other에서 phone error rate (PER) 평가
  - best PER = 21.54% (voice activity detection 적용 시) 

### 4.3 Pre-training and Fine-tuning
#### Student 모델 설정 (Data2vec2 기반)
- CNN-based waveform encoder (downsampling ratio: 320)
- Transformer 12층
- 최종 decoder: conv layer 1개

#### Pre-training 세팅
- GPU: Tesla V100 16GB × 16개
- 배치 크기: 1GPU당 약 62.5초 분량 음성
- 총 학습 스텝: 400k
- 하이퍼파라미터: Data2vec2 논문과 동일

#### Intermediate decoder 설정
- 구조 및 padding은 final decoder와 동일
- 중간층 **supervision에 사용할 UASR target은 instance normalization 적용**
- 논문에서 선택한 intermediate layer: **Transformer의 4번째 layer**
- 가중치 계수 \kappa = 0.1

#### Fine-tuning 세팅
- 100h 라벨 데이터로 CTC loss 기반 fine-tuning
- GPU: Tesla V100 8개
- 메모리 문제로 batch size 조정
- dev-other에서 WER 기준으로 best model 선택



<br>  
  
## 5. Experimental Results
### 5.1 Effectiveness of Intermediate Loss
- teacher 모델 유형 비교
  - CTC-finetuned: 라벨된 데이터로 학습한 일반 ASR 모델
  - Wav2vec-u2: UASR 기반 teacher
- loss 구성 비교
  - L_distill only: 중간층만 supervision (SSL 없음)
  - SSL + L_distill: 본 논문 방식
  - intermediate distill: 중간층에 supervision 적용 (논문 제안 핵심)  
<img width="381" height="79" alt="image" src="https://github.com/user-attachments/assets/4868eb80-60b1-44b3-bb39-b9bfaae9062a" />
 
  - UASR teacher만으로도 꽤 괜찮은 결과 (4.1 / 8.9)
  - intermediate layer + SSL 구조가 가장 성능이 좋음
  - CTC로 라벨이 있는 teacher보다도 좋은 결과! -> 즉, 라벨 없이도 더 나은 성능 가능함

### 5.2 Explicitness of Pseudo-targets
- 서로 다른 방식으로 만든 타겟이 student 성능에 미치는 영향 비교
- 중간층 distillation 위치는 고정 (4번째 layer)
- 다만 타겟을 어떻게 만들었는지가 다름
  - None: baseline Data2vec2
  - Contrastive: W2v-BERT 스타일 (discrete target + contrastive loss)
  - CTC-finetuned: supervised teacher 사용
  - Wav2vec-u2: 제안 방식 
<img width="379" height="90" alt="image" src="https://github.com/user-attachments/assets/3bb9da42-8045-4c76-a39f-389f002aa972" />

  - Data2vec2보다 contrastive loss를 쓴 W2v-BERT류는 오히려 성능 하락
  - Wav2vec-u2가 만든 타겟이 더 ASR-friendly
  - phonetic 정보가 포함된 pseudo-target이 가장 도움 됨
 
### 5.3 Low-resource Speech Recognition
- Pre-training: LS 960h (unlabeled)
- Fine-tuning: LS 100h (labeled)
- Decoding: CTC greedy decoding only (language model 없이)
<img width="380" height="266" alt="image" src="https://github.com/user-attachments/assets/6b4e6474-149f-4d94-a092-5b2dce16f09c" />

  - Data2vec2 대비 8.9% / 4.3% 상대적 WER 감소
  - 기존 self-training 기법(PBERT, ASBERT)보다도 우수
  - 제안 방식은 학습 전 과정에서 라벨 없이 teacher 구성




<br>  
  
## 6. Conclusions
### 논문에서 제안한 핵심 아이디어
- **라벨 없이도 self-training을 가능**하게 하는 새로운 방식 제안
- 기존 self-training은 항상 처음에 labeled data로 teacher를 학습해야 했음
- 이 때문에 **overfitting + noisy label 문제가 발생**

### 실험 결과
- phonetic supervision을 중간 층에 추가
  - 기존 SSL보다 더 ASR-friendly한 표현 학습 가능
  - 기존 supervised self-training teacher보다도 더 나은 성능 달성
- **low-resource 조건(100h labeled data)** 에서도 SOTA 결과를 기록
  - 특히, 전체 pre-training에서 라벨을 전혀 쓰지 않고도 가능 


