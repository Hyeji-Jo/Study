# SoftCorrect: Error correction with soft detection for automatic speech recognition
## 요약 정리
### Problem
- ASR 시스템이 낮은 WER을 달성했음에도, **여전히 소수의 오류 단어**는 존재함  
- 이때 중요한 건 **정확한 단어는 유지하고, 오류 단어만 선택적으로 수정하는 것**

- 기존 방식들의 문제점
  - **암묵적 오류 탐지**: 오류 위치에 대한 명확한 신호 부족 → 모델 학습 어려움
  - **명시적 오류 탐지**: 오류 탐지가 실패할 경우 **정확한 단어를 잘못 수정하는 오류** 발생 가능
- 따라서 오류 단어를 정확하고 유연하게 탐지하고, 해당 단어에만 집중하여 수정하는 방식이 필요

### Contributions
- **Soft Error Detection**  
  - 언어 모델 확률 기반으로 각 단어의 오류 가능성을 판단  
  - **GT token**을 추가한 anti-copy LM loss 도입
- **Constrained CTC Loss**  
  - **탐지된 오류 단어만 3회 복제**하여 디코더가 해당 위치에만 집중  
  - 정확한 단어는 anchor처럼 그대로 유지
- **병렬 생성 지원 (NAR)**: 빠른 속도 유지
- **다수 후보(n-best)** 기반 후보 선택 및 오류 판단
- **최고 수준 CER 감소 달성 (AISHELL-1, Aidatatang)**


### Method
- 전체 구조
  - Encoder (탐지기)
    - 다수 후보를 정렬하여 입력
    - 각 위치의 단어가 문맥상 자연스러운지 **확률로 soft하게 판단**
  - Decoder (수정기)
    - **탐지된 오류 단어만 복제**
    - **Constrained CTC Loss** 기반으로 병렬 교정 수행
- 핵심 모듈
  | 모듈 | 역할 |
  |------|------|
  | **Anti-Copy LM Loss** | GT token 추가 → trivial copy 방지 + 문맥 기반 오류 탐지 학습 |
  | **Constrained CTC Loss** | 오류 단어만 복제 → 빠르고 정확한 수정 수행 |
- 추가 기법
  - **ASR beam search 후보(n-best)** 정렬 후 voting 기반 후보 선택  
  - **ASR 음향 확률 + LM 확률** 결합 → 오류 판단 신뢰도 향상

### Experiments
<img width="585" alt="image" src="https://github.com/user-attachments/assets/9f9daca6-2ff4-41bf-bc11-dbcb29d356e8" />

- 기존 명시적 탐지(NAR) 방식보다 **더 높은 정확도**
- AR 기반 모델보다 **빠르면서도 정확**
- **Aidatatang**처럼 오류 탐지가 어려운 환경에서도 **안정적으로 효과 발휘**


<br>  
  
## 0. Abstract
### 오류 교정의 목표
- ASR 모델이 생성한 문장에서 **잘못된 단어만을 수정하는 것**
  
### 연구 배경 및 목적
- 최근 ASR 시스템은 일반적으로 **낮은** 단어 오류율(**WER**)을 보임
  - 하지만 **여전히 일부 단어 오류 존재**
- **정확한 단어는 그대로 두고, 오류 단어만 수정해야 함**
- 따라서 정확히 오류 단어를 탐지하는 것이 핵심 과제
### 기존 방법의 한계
- **암묵적 오류 탐지**
  - target-source attention 또는 CTC(connectionist temporal classification) loss를 통해 간접적으로 파악
  - 어떤 단어가 오류인지 명확한 신호를 제공하지 않음 
- **명시적 오류 탐지**
  - 삭제/치환/삽입 오류를 명확히 위치와 함께 지정하는 방식
  - 탐지 정확도가 낮기 때문에 오히려 새로운 오류 유발 가능
### 제안 방법 : SoftCorrect
- 명시적/암묵적 탐지의 장점 결합
- 주요 구성 요소
  - 1) 언어 모델 기반으로 각 단어가 문맥상 적절한지 **확률로 판단**
  - 2) Constrained CTC loss를 적용하여 **오류로 탐지된 단어들만 복제(duplicate)** 함으로써 디코더가 오류 단어에만 집중하도록 함
- 암묵적 방법 대비 명확한 오류 위치 제공 및 명시적 방법 대비 구체적인 유형 구분하지 않아도 됨
### 실험 결과
  - AISHELL-1: 문자 오류율(CER) 26.1% 감소
  - Aidatatang: CER 9.4% 감소
  - 동시에 병렬 생성(parallel generation) 방식으로 빠른 속도도 유지


<br>
  
## 1. Introduction
### Soft Error Detection의 구현 방법
- 기존 binary 분류 대신 **언어 모델의 확률(logits) 사용**
  - 기존 명시적 탐지 방식 중 일부 연구는 binary classification을 활용 
  - 문맥 이해/어휘적 자연성을 더 잘 반영
- 기존 LM 방식들(GPT, BERT)은 부적절함
  - GPT: 단방향이라 문맥 부족
  - BERT: 양방향이지만 느림 (N-pass 필요)
    - BERT는 원래 **“masked language model”** -> 그 MASK 위치에 올 정답 토큰을 예측하는 게 목적
    - 즉, 특정 위치 하나만 예측하도록 학습되어 모든 토큰 위치에 대해 확률을 알고 싶다면 각 위치를 [MASK]로 바꿔서 N번 추론해야 함 
- 따라서 새로운 언어 모델 손실 함수(anti-copy LM loss)를 도입하여 효율적 확률 추정 가능

### Constrained CTC
- 기존 CTC
  - 오류 여부와 관계없이 **모든 토큰을** 여러 번 중복해 디코더에 입력하고 학습함
  - 모델이 정확한 토큰과 오류 토큰을 구분하지 못해, 올바른 단어까지 잘못 수정할 가능성이 있음
  - 전체 토큰을 **모두 정렬 대상**으로 처리하므로 **연산량이 많고, 디코딩 속도가 느림**
- **Constrained CTC**
  - **오류로 탐지된 토큰만 중복**하고, 나머지 정확한 토큰은 수정 없이 그대로 유지되도록 처리함
  - 모델이 **수정이 필요한 부분에만 집중해서 학습**할 수 있어 오류 교정의 정확도가 높아짐
  - 중복된 토큰 수가 줄어들고, 정렬 대상이 제한되어 **전체 처리 속도가 빨라짐**
 
### ASR 후보 활용 (N-Best 활용)
- **ASR beam search** 결과의 다수 후보 이용
  - 더 나은 후보 문장 선택
  - 선택된 후보 내에서 오류 토큰 탐지
- Beam Search
  - 가능한 모든 단어 확률을 계산하고, 상위 K개 선택
  - 다음 토큰을 예측할 때, 각 후보 경로를 확장해서 또 상위 K개를 유지
  - 마지막까지 반복하여 K개의 후보 문장을 생성 
- 오류 감지 신뢰도를 높이기 위해 **ASR의 음향 확률 + 언어 모델 확률을 결합**
  - ASR의 음향 확률 : 아 움송아 툭정 단어일 확률
  - 언어 모델 확률 : 앞뒤 문맥을 고려했을 때, 이 단어가 나올 확률

  
  
<br>
  
## 2. Background
### 오류 교정의 발전
- 초기에는 **통계적 기계 번역(SMT)** 기반 접근 사용
- 이후에는 Transformer 기반의 자기회귀(AR) 신경망 모델로 발전
  - AR : 출력 시퀀스를 앞에서부터 순차적으로 한 토큰씩 생성하며, 각 토큰은 이전 토큰들에 의존해서 생성됨 
- 최근에는 추론 속도를 개선한 비자기회귀(NAR) 모델이 각광받고 있으며, duration predictor를 사용해 병렬 디코딩이 가능하고 정확도도 우수
  - **NAR** : **출력 전체를 동시에 예측**하며, 각 토큰은 다른 토큰에 의존하지 않고 **병렬로 생성**
    - NAR 모델의 경우 출력 길이를 정하거나 토큰 위치를 예측하기 어려움
  - **duration predictor** : 각 입력 토큰이 몇 개의 출력 토큰으로 확장될지 예측
    - **출력 길이와 위치가 고정**되면, 디코더는 각 위치의 출력 토큰을 **독립적으로 동시에 예측 가능**   
- SoftCorrect는 이러한 비자기회귀 기반의 교정 모델로, 빠르고 정확한 수정이 목표

### ASR 다중 후보 활용 (n-best)
- 대부분의 ASR 시스템은 beam search를 통해 **여러 개의 후보 문장(n-best)** 을 생성
- **후보 간 동일 위치의 불일치는 오류 가능성이 높은 위치를 시사**
  - 같은 위치에서 단어가 서로 다르게 나오는 부분이 있다면 오류가 있을 가능성 높음
- **후보들을 정렬하여 같은 길이로 만든 뒤**, encoder 입력으로 활용
- 다중 후보의 차이를 통해 오류 위치를 탐지하고, 더 나은 후보 문장을 선택하는 방식으로 정확도 높임

### 오류 탐지 방식: 명시적 vs. 암묵적
- Explicit - 명시적 정렬
  - source와 target 문장을 편집 거리 기반으로 정렬하고, 각 토큰의 duration 예측
  - duration 값에 따라 삭제/삽입/치환 오류 판단
    - 0:삭제 오류 / 1:유지or치환 / >=2:삽입 오류 
  - 하지만 duration 예측이 어렵고, 잘못된 판단 시 **새 오류 유발 가능성 존재**

- Implicit - 암묵적 정렬
  - attention 또는 CTC 기반 구조를 활용해 **오류 위치를 명시하지 않고 간접 학습**
  - 특히 CTC는 병렬 디코딩이 가능하고 duration 예측 없이 학습 가능
  - 오류 위치에 대한 명확한 신호가 없다는 한계 존재


  
<br>
  
## 3. SoftCorrect
<img width="792" alt="image" src="https://github.com/user-attachments/assets/deb6f1ca-5406-4bbe-af97-56d6bf6f015d" />
  
### 1) System Overview
#### 다중 후보 정렬 및 입력 생성
- ASR beam search로부터 얻은 다중 후보 문장 활용
- 각 후보 문장은 길이가 다를 수 있으므로, **동일한 길이로 정렬**
- 정렬된 결과는 각 위치마다 여러 후보 단어가 나열된 형태
- 이를 임베딩한 뒤 위치별로 concat하여 선형 계층(linear layer)의 입력으로 사용

#### 오류 탐지기 (Encoder)
- 표준 Transformer Encoder 구조로 각 토큰에 대한 확률 생성
- encoder의 hidden state는 임베딩 행렬과 곱해져서 **전체 어휘(vocabulary)**에 대한 확률 분포를 산출
- 즉, 특정 위치에서 **정답 토큰 E와 잘못된 토큰 E′**에 대한 확률을 동시에 비교 가능

#### 후보 문장 선택
- **그냥 입력 토큰을 그대로 복사하는 방식(trivial copy)** 으로 학습하지 않도록
  - **anti-copy** 언어 모델 손실 함수를 사용해 encoder 학습
- 정답만 예측하는 게 아니라, **“다른 후보 중 이 토큰이 더 자연스러운가?”** 를 평가하도록 학습
- encoder가 출력한 확률 분포를 기반으로, **각 위치마다 가장 확률이 높은 토큰을 선택**

#### 오류 토큰 판단 (확률 결합)
- 선택된 문장의 각 토큰에 대해, **encoder 확률(LM 기반)과 ASR 음향 확률을 가중합(weighted sum)**
- 확률이 설정된 임계값(threshold)보다 낮으면, 해당 토큰을 오류로 탐지

#### 오류 교정기 (Decoder)
- 선택된 후보 문장을 입력으로 받아, 수정된 최종 문장을 출력
- 오류로 탐지된 토큰만 3번 중복하여 decoder에 입력
- 정확한 토큰은 anchor로 간주되어 위치 고정, 수정되지 않음
- constrained CTC loss로 학습

### 2) Anti-Copy Language Modeling for Detection
- encoder를 언어 모델처럼 학습시키되, trival copy(입력 그대로 복사) 현상을 방지하기 위한 손실 함수(anti-copy LM loss) 도입
<img width="650" alt="image" src="https://github.com/user-attachments/assets/b03c3c20-8936-4ae0-8d0a-24ed6cb80779" />
  
#### Anti-Copy Language Model Loss 구성
1. Cross-Entropy Term (기본 LM 학습)
  - 문맥을 보고 정답 토큰의 확률이 가장 높아지도록 학습
  - 정답 토큰 $`y_t`$ 이 전체 vocabulary V + GT 중에서 가장 높은 확률을 갖도록 softmax 학습
2. Regularization Term (복사 억제)
  - GT (가짜 정답) 토큰을 새로 추가
  - 정답 토큰을 제외한 나머지 단어들과 GT 중에서 → GT의 확률이 가장 높아지도록 학습
  - 정답을 모르더라도 GT를 선택하게 유도함으로써 입력 복사에 의존하지 않는 학습 유도
    
#### 손실 함수 수식
<img width="414" alt="image" src="https://github.com/user-attachments/assets/c7f07f50-5bce-4701-8154-17a76e68bb94" />

- $`H_t`$ : 위치 t에서의 encoder 출력
- $`W_i`$ : softmax weight 행렬
- $`y_t`$ : 위치 t의 정답 토큰
- $`V`$ : 원래 vocabulary
- $`\lambda`$ : 정규화 항의 가중치 (논문에서는 1.0 사용)
  
#### 효과
- 정답은 맞추되, 단순히 입력을 복사하는 경향(trivial copy)을 억제함
- encoder가 입력 후보 간 문맥상의 차이를 구별할 수 있게 됨
- 결과적으로 오류 탐지 및 후보 선택에 더 정교한 확률 분포를 제공함
<img width="445" alt="image" src="https://github.com/user-attachments/assets/76462079-a91f-41de-bbfa-757d83f0e19f" />


### 3) Constrained CTC Loss for Correction
#### 기존 CTC Loss의 문제
- 입력의 모든 토큰을 복제해서 정렬과 수정 대상으로 사용함
- 정확한 토큰도 수정 대상이 됨 → **정확한 단어가 오히려 바뀌는 오류 발생 가능**
- 모든 위치에서 정렬을 하려 하니 **연산량 증가 → 속도 느려짐**

#### Constrained CTC Loss
<img width="466" alt="image" src="https://github.com/user-attachments/assets/f9d78cb8-a7d4-45f6-be60-f4bc33d6a3a2" />

- **오류로 탐지된 토큰만 3회 중복하여 입력**으로 넣고
- 정확한 토큰은 수정 없이 anchor처럼 고정시킴
- 디코더는 이 구조를 기반으로 학습함

#### Why Constrained CTC Loss
<img width="442" alt="image" src="https://github.com/user-attachments/assets/42786146-37f9-4c7b-b40e-2d7da245b161" />

- 기존 CTC는 전체 입력에서 정렬 경로를 자유롭게 찾음
- SoftCorrect의 Constrained CTC는 다음처럼 제약함
  - 정확한 토큰 위치는 고정
  - 중복된 오류 토큰 부분만 정렬 허용
- 결과: 어디를 고치고 어디는 그대로 둘지 명확히 구분됨

#### 추론 시 처리 방식
- 오류 토큰 위치에서만 softmax 수행하여 **수정된 토큰 선택**
- 그 외 위치(정확한 토큰)는 수정 없이 그대로 출력
- 중복과 blank는 CTC 방식에 따라 제거함

#### 추가 설계: 오류 탐지 실수에 대한 보완
- 오류 탐지기가 정확한 토큰을 실수로 오류로 판단하는 상황 대비
- 학습 중, 정확한 토큰의 5%를 임의로 오류로 간주(pseudo error)
  - **decoder가 정확한 토큰을 잘못 고치지 않도록 학습**
- 이로써 decoder가 robust(강인)하게 학습됨


  
<br>
  
## 4. Experimental Setup
### 1) Datasets and ASR Model
#### 데이터셋
- 모두 중국어 음성 인식 데이터셋
- 실험은 ASR → 오류 교정 순서로 진행
| 데이터셋     | 학습 (train) | 개발 (dev) | 테스트 (test) |
|--------------|--------------|-------------|----------------|
| AISHELL-1    | 150시간       | 10시간       | 5시간          |
| Aidatatang   | 140시간       | 20시간       | 40시간         |

#### ASR 모델 구성
- 아키텍처: Conformer (SOTA)
- 성능 개선 기법
  - SpecAugment (스펙트럼 왜곡)
  - Speed perturbation (속도 변화)
  - 언어 모델과 joint decoding

#### 교정 모델용 학습 데이터
- ASR 모델로 음성을 텍스트로 변환하여 오류가 포함된 문장 획득 → 교정 모델 학습에 사용
- 추가로, 4억 개의 비평렬 텍스트를 이용해 **가짜 오류 데이터(pseudo data)** 를 생성하여 사전학습(pretraining) 진행

  
### 2) Baseline Systems
#### 암묵적 오류 탐지 모델 (Implicit)
- AR Correct
  - Transformer 기반 AR encoder-decoder
- AR N-Best
  - n-best 후보를 정렬하여 AR 모델에 입력
- 오류 위치를 명시적으로 예측하지 않고, attention을 통해 암묵적으로 학습

#### 명시적 오류 탐지 모델 (Explicit)
- FastCorrect
  - duration 예측 기반 병렬 디코딩 NAR 모델
- FastCorrect 2
  - FastCorrect + n-best 후보 사용
- 각 토큰에 대해 duration을 예측하여 오류 유형(삽입/삭제/치환 등)을 명시적으로 파악

####  기타 결합 방식
- Rescore
  - 12-layer Transformer로 n-best 후보 중 가장 자연스러운 문장을 선택
- FC + Rescore
  - FastCorrect로 먼저 수정 → Rescore
- Rescore + FC
  - Rescore 먼저 수행 → 이후 FastCorrect로 수정

  
<br>
  

## 5. Result
### 1) Accuracy and Latency
<img width="696" alt="image" src="https://github.com/user-attachments/assets/f7d5c2d5-df66-4549-9727-4a32e2d297a4" />

- CER (Character Error Rate): 원본 문장과 교정 문장 간 문자 오류율
- CERR (CER Reduction): ASR 원문 대비 오류율 감소율
- Latency: 한 문장을 처리하는 데 걸리는 시간 (ms/sentence)
- 다양한 기존 모델들과 비교하여, SoftCorrect는 **가장 높은 CERR (오류율 감소율)** + 추론 속도(latency)도 매우 빠름

### 2) Ablation Studies
<img width="709" alt="image" src="https://github.com/user-attachments/assets/67f5200e-3240-46a2-960b-c75e2326531d" />

- SoftCorrect의 구성 요소(soft detection, anti-copy loss, constrained CTC)의 효과를 구성별로 제거해 확인
- anti-copy LM loss는 단순 복사 대신 문맥 기반 오류 판단 능력을 높여줌
- constrained CTC는 오류 토큰에만 수정 집중 → 정확도 향상 + 불필요한 수정 방지
- GPT-style, binary 분류 방식은 문맥 반영이 부족하거나 비효율적임


### 3) Method Analyses
<img width="729" alt="image" src="https://github.com/user-attachments/assets/29064989-39e8-415c-9d01-d536d295b1fe" />

- SoftCorrect가 기존 AR/NAR 모델보다 더 잘 오류를 찾고, 더 잘 고치는지 정량적으로 비교
- 평가 지표
  - P_det (Precision of Detection) : 오류로 탐지한 것 중 실제 오류인 비율
  - R_det (Recall of Detection) : 실제 오류 중 탐지된 비율
  - F1_det : 탐지 정밀도와 재현율의 조화 평균
  - P_cor (Precision of Correction) : 수정한 것 중 실제로 정답으로 바뀐 비율

- SoftCorrect는 탐지 정밀도와 재현율 모두 우수 → 높은 F1 score
- 수정한 토큰 중 정답으로 바뀐 비율(P_cor)도 가장 높음
- Aidatatang처럼 어려운 데이터셋에서도 성능 안정성 유지

  
<br>
  

## 6. Conclusion
#### 기존 한계 및 제안
- 기존의 오류 탐지 방식은 다음과 같은 한계 존재
  - 명시적 탐지: 오류 유형 분류가 어렵고 실패 시 성능 악화
  - 암묵적 탐지: 명확한 위치 신호가 없어 모델 학습이 비효율적
- 이를 해결하기 위해, **SoftCorrect는 soft error detection** 방식을 제안함

#### 주요 구성요소
- Encoder
  - 후보 문장 중 가장 자연스러운 문장을 선택하고, 각 토큰이 오류인지 확률적으로 판단
- Anti-Copy LM Loss
  - trivial copy 방지를 통해 문맥 기반 오류 탐지 능력 강화
- Decoder
  - 탐지된 오류 토큰만 수정, 나머지는 그대로 유지
- Constrained CTC Loss
  - 수정 대상을 제한해 정확도 상승 + 연산량 감소

#### 실험 결과 요약
- AISHELL-1: 26.1% CER 감소
- Aidatatang: 9.4% CER 감소
- 기존 AR/NAR/Rescore 방식보다 더 높은 정확도 + 더 빠른 속도 달성

  
