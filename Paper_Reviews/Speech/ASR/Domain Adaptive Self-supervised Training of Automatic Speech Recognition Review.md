# Domain Adaptive Self-supervised Training of Automatic Speech Recognition
## 요약 정리
### Problem
- **도메인 불일치(domain mismatch)** 문제로 ASR 시스템 성능 저하 발생
  - 예: 다양한 억양(영국, 인도, 비원어민 등)
- **라벨이 없는 도메인 데이터(unlabeled target domain data)**는 현실적으로 수집이 쉬움
- unlabeled 데이터를 효과적으로 활용하여 **ASR 모델의 도메인 적응 성능을 개선하는 방법 필요**


### Contributions
1. **Self-supervised + Semi-supervised** 조합을 통한 도메인 적응 전략 제안
2. **Wav2vec 2.0 기반 SSL 모델**에 target domain 데이터 활용
   - (1) pre-training에만,
   - (2) fine-tuning에만,
   - (3) 둘 다 사용하는 조합 실험
3. **단일 및 다중 도메인** 설정 모두에서 정량적 성능 개선 검증
4. **pseudo-label 기반 fine-tuning만으로도 oracle 수준에 근접**


### Method
- **모델 아키텍처**: Wav2vec 2.0 BASE  
  - Feature encoder (6 conv layers), Context network (12 Transformer layers)
  - Product Quantization + Gumbel Softmax + Contrastive loss
- **Fine-tuning**: pseudo-label 기반 CTC loss 학습
- **Decoding**: Beam search + Transformer LM (Librispeech 기반)



### Experiments & Setup
- **도메인**
  - D0: 미국 억양 (Librispeech) – in-domain
  - D1: 비원어민 영어 (L2-ARCTIC)
  - D2: 영국 억양 (British Isles)
  - D3: 인도 억양 (NPTEL)
- **모델 구성**
  - pre-train, fine-tune, both (with/without pseudo-label)
- **평가 지표**: WER (Word Error Rate)


### Results
- **단일 도메인**
  - 최대 41.8% WER 감소 (pre-train + fine-tune 조합)
  - D3 (인도 억양)의 경우, WER 35.7% → 17.8%
- **다중 도메인**
  - 평균 WER
    - `D0`: 14.6%  
    - `D0-AUG-D1,2,3-FT`: **9.2%**  
    - `D0-FT-Oracle`: **5.9%**


### Limitations
- pseudo-label 품질이 낮은 경우 성능 악화 가능 (e.g., D0-FT-D3)
- 오라클 성능과는 일부 간극 존재
- 다양한 noise, 어린이 음성 등 더 폭넓은 도메인에 대한 추가 실험 필요


### Insights & Idea
- **SSL 단독 학습은 도메인 적응에 한계**, semi-supervised fine-tuning이 이를 보완
- **다중 도메인 학습**은 모델 generalization 향상에 매우 효과적
- unlabeled speech 활용은 실제 제품/서비스(스마트 스피커, AI 콜센터)에서 매우 실용적인 전략
- 본 접근법은 **noise adaptation, 채널 mismatch 등 다른 도메인 적응 문제에도 확장 가능**

<br>  
  
## 0. Abstract
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




<br>  
  
## 1. Introduction
### 문제 정의
- ASR 시스템은 특정 도메인(예: 미국식 억양)에서 학습되면, 다른 억양/환경에서 성능 저하 발생
- 그러나 실제 사용 환경에서는 다양한 억양, 화자, 상황에서 발화가 이루어짐
  - **도메인 불일치 문제(domain mismatch) 발생** 

### 기존 접근 방식 및 한계
- **SSL (Self-supervised Learning)**
  - unlabeled speech로 **representation을 학습 (예: Wav2vec 2.0)**
  - 이후 fine-tuning을 통해 ASR 모델로 전환
- **한계**
  - SSL 모델을 **다른 도메인에서 바로 적용하면 성능 저하**
  - 기존 연구는 **target domain 데이터를 pre-training에만 사용했으며**, **fine-tuning에서는 적극 활용하지 않음**

### 제안 아이디어
- **“pre-training + semi-supervised fine-tuning”의 결합으로 도메인 적응력을 향상**
- **Pre-training**
  - target domain의 unlabeled data를 포함하여 **SSL 모델 사전학습**
- **Fine-tuning**
  - 사전학습된 모델을 기반으로, **pseudo-label 생성**
  - 이 pseudo-label을 사용해 **semi-supervised fine-tuning 수행**
 
### 연구 실험 환경
- In-domain (기준 도메인): Librispeech (미국식 억양)
- Target domains: L2-ARCTIC (비원어민 영어), British Isles (영국 억양), NPTEL (인도 억양)
- 사용 데이터 종류: 모두 라벨 없는(unlabeled) 데이터
- 평가 지표: Word Error Rate (WER)





<br>  
  
## 2. Related Works
### Unlabeled target domain data 활용한 SSL 사전학습
- 테스트 도메인과 유사한 unlabeled 데이터를 pre-training에 포함시키면 ASR 성능이 좋아진다는 것을 보임
- 다양한 도메인을 포함시켜 학습하면, 완전히 새로운 도메인에도 강건한 모델이 가능하다는 결과도 있음

### SSL 모델로부터 pseudo-label 생성해 fine-tuning
- SSL로 학습된 ASR 모델을 사용하여 pseudo-label을 생성하고, 그것을 사용해 **Transformer 기반 end-to-end ASR 모델을 지도학습**
- 원래의 SSL 모델 성능을 초월하는 결과를 얻음

### Semi-supervised ASR 연구들
- teacher 모델이 pseudo-label 생성
- student 모델은 pseudo-label과 ground truth 혼합하여 학습
- 추가 기법
  - confidence 낮은 pseudo-label 제거
  - pseudo-label 반복 refinement (iterative labeling) 수행

### 본 논문의 차별점
- target domain 데이터를
  - **사전학습(pre-training)** 에 넣거나
  - **반지도학습 fine-tuning**에 넣거나
  - **두 단계 모두에 사용하는 다양한 조합**을 실험
- 본 논문은 **pure SSL 기반 모델을 fine-tuning**하는 점에서 차별됨
  - Supervised: 라벨된 데이터(음성 + 정답 텍스트)를 이용해 처음부터 끝까지 모델을 학습
  - SSL (Self-supervised Learning): 라벨 없는 음성 데이터만으로 특징 표현(feature representation)을 학습한 후, downstream task에 fine-tuning
    - **Downstream task**: **사전학습(pre-training)** 된 모델이 실제로 성능을 발휘해야 하는 **최종 목적의 작업**
    - 즉, **미리 학습된 표현(representation)** 을 가지고 **실제 우리가 풀고 싶은 문제(task)에 적용하는 것**
  - Semi-supervised: 일부 라벨된 데이터 + 나머지는 pseudo-label로 구성해서 학습
  - Pure SSL 기반 모델: 사전학습(pre-training) 단계에서 라벨 없이 SSL만 사용한 모델
    - supervised 데이터는 fine-tuning 단계에서만 등장함

 

<br>  
  
## 3. Domain adaptive self-supervised training of ASR
### 학습 데이터 활용 전략
- 사전학습(pre-training): **대규모의 레이블 없는 데이터**
- 미세조정(fine-tuning): **소량의 레이블링된 데이터**

- **target domain의 unlabeled 데이터**
  - SSL 사전학습에 포함하거나
  - pseudo-label 생성 후 semi-supervised fine-tuning에 활용하거나
  - 둘 다 적용 가능 

### Wav2vec 2.0 SSL 모델 활용
- **Feature Encoder**
  - 원시 음성 파형을 입력받아 특징을 추출하는 역할
  - 여러 블록으로 구성되며, 각 블록은 Temporal Convolution, Layer Normalization, GELU(Gaussian Error Linear Unit) 활성화 함수로 구성
    - **Temporal Convolution (시간축 합성곱)**: 시간에 따라 변화하는 시계열 신호인 음성을 다루는 데 적합한 **1D Convolution 연산**
      - 이 과정을 통해 짧은 구간의 주파수 패턴이나 리듬 정보 추출
    - **GELU (Gaussian Error Linear Unit) 활성화 함수**: 입력값이 작으면 거의 0, 크면 1로 활성화되지만, 그 사이에서는 부드럽게 작동하는 비선형 함수
      - ReLU와 비슷하지만, 확률적 요소가 가미되어 **더 부드러운 결정 경계 생성**
      - **“너무 작거나 너무 큰 값은 줄이고, 중간 크기 값만 반영”**   
  - 원시 파형은 0 평균 및 단위 분산으로 정규화된 후 인코더에 입력
    - 원시 파형 정규화: 음성 파형 데이터를 모델에 넣기 전에, **평균이 0, 표준편차가 1**이 되도록 스케일을 조정하는 작업 
- **컨텍스트 네트워크(Context Network)**
  - 특징 인코더의 출력을 받아 처리하는 부분으로, Transformer 아키텍처
  - 특징 인코더의 출력은 Product Quantization을 통해 유한한 음성 표현 집합으로 이산화
    - **Product Quantization (PQ)**: **연속적인 벡터 표현(continuous vector)** 을 이산적인 코드로 바꾸는(quantize) 기술
      - PQ는 feature 벡터를 여러 개의 부분으로 쪼개고, 각 부분을 미리 정해둔 코드북에서 가장 가까운 벡터로 대체 
    - Quantization 모듈은 Gumbel softmax를 사용하여 코드북에서 엔트리를 선택
      - **Gumbel Softmax**: 연속값(softmax)을 통해 **“이산적인 선택”** 을 가능하게 해주는 기술
        - 샘플링을 흉내내어 hard하게 하나의 값을 선택한 것처럼 동작함
      - **코드북에서 엔트리를 선택**: 미리 정의된 **대표 벡터 집합(=코드북)** 에서 **입력 벡터와 가장 유사한 코드(entry)** 를 선택
  - Relative Positional Encoding은 Convolutional Layer를 통해 구현
  - **사전 학습 중에는 Contrastive Loss를 최소화**하는 방식으로 음성 표현이 학습
  - 모델 구조는 특징 인코더 6개의 Convolutional Layers, 컨텍스트 네트워크 12개의 Transformer Layers로 구성

### Fine-tuning
- 사전 학습된 Wav2vec 2.0 모델의 컨텍스트 네트워크 위에 무작위로 초기화된 선형 투사(linear projection) 레이어를 추가하여 ASR 작업에 맞게 미세 조정
  - linear projection은 음성 데이터를 미리 정의된 C개의 클래스(어휘)로 매핑
  - 본 연구에서는 영어 문자를 출력 단위로 사용하며, 총 29개의 문자 클래스와 1개의 단어 경계 토큰(word boundary token)을 포함
- 모델 파라미터는 **CTC(Connectionist Temporal Classification) 손실**을 최소화하여 최적화

### Decoding
- Beam Search + Transformer 언어 모델(Language Model, LM) 사용
  - LM은 Librispeech 정답 텍스트로 학습된 word-level Transformer
- Pre-training, Fine-tuning, Decoding은 모두 (Wav2vec 2.0 BASE 모델) 설정 그대로 사용 



<br>  
  
## 4. Experiments
### 4.1 Single domain
#### 실험 목적
- 단일 target domain의 unlabeled data를 사용하는 다양한 방식 비교
- pre-training / fine-tuning / 둘 다 적용하는 조합의 효과 분석

#### 도메인 정의

| 도메인 | 설명 | 표기 |
|--------|------|------|
| D0 | 미국 억양 (Librispeech) | 기준 도메인 |
| D1 | 비원어민 억양 (L2-ARCTIC) | target |
| D2 | 영국 억양 (British Isles) | target |
| D3 | 인도 억양 (NPTEL 강의 음성) | target |


#### 모델 구성 (Table 1 기준)

| 모델 이름 | 설명 |
|-----------|------|
| `D0` | Base 모델 (Librispeech만 사용) |
| `D0-FT-Di` | Di 도메인을 fine-tuning만 사용 (pseudo-label 활용) |
| `D0-AUG-Di` | Di 도메인을 pre-training에만 사용 |
| `D0-AUG-Di-FT` | Di를 pre-train + fine-tune에 모두 사용 |
| `D0-FT-Oracle-Di` | Di 도메인의 **정답 라벨**로 fine-tuning (성능 상한선 비교용) |

#### 주요 결과 요약

- **L2-ARCTIC (D1)**
  - `D0`: 16.1%  
  - `D0-AUG-D1-FT`: **7.9%** → **50.9% WER 감소**

- **British Isles (D2)**
  - `D0`: 14.4%  
  - `D0-AUG-D2-FT`: **10.1%**

- **NPTEL (D3)**
  - `D0`: 35.7%  
  - `D0-AUG-D3`: **18.3%**, `D0-AUG-D3-FT`: **17.8%**
  - D0-FT-D3는 오히려 성능 저하 발생 (pseudo-label 품질 문제)


### 4.2 Multiple domains
#### 실험 목적
- D1, D2, D3 도메인을 **동시에 사용**할 때의 성능 비교
- 다양한 억양에 robust한 ASR 성능 달성 가능 여부 검증

#### 실험 조건

| 모델 | 설명 |
|------|------|
| `D0-AUG-D1,2,3` | 모든 도메인을 pre-training에 포함 |
| `D0-AUG-D1,2,3-FT` | 위 모델을 기반으로 semi-supervised fine-tuning 수행 |
| `D0-FT-Oracle-D1,2,3` | 모든 도메인 정답 라벨 사용 (oracle 기준) |


#### 평균 WER 결과

| 모델 | 평균 WER |
|------|-----------|
| `D0` | 14.6 |
| `D0-AUG-D3` (최고 단일) | 11.5 |
| `D0-AUG-D1,2,3` | 10.0 |
| `D0-AUG-D1,2,3-FT` | **9.2** |
| `D0-FT-Oracle-D1,2,3` | **5.9** (성능 하한선)


#### 결론
- 다중 도메인 학습은 일반화에 유리
- fine-tuning을 추가하면 성능이 더 향상됨
- pseudo-label 기반 fine-tuning만으로도 oracle에 근접



<br>  
  
## 5. Conclusion
### 주요 접근 방식
- 레이블이 없는(unlabeled) 타겟 도메인 데이터를 활용하여 도메인 적응형 ASR을 구현하는 새로운 방법 제시
- SSL(Self-supervised Pre-training)과 준지도 학습(semi-supervised fine-tuning) 결합

### 실험 설정 및 평가 방식
- 다양한 영어 악센트(accent) 데이터를 도메인 데이터로 사용
- 타겟 도메인 데이터를 다음 세 가지 방식으로 사용하여 ASR 성능 향상 평가
  - SSL 사전 학습(pre-training) 단계에서만 사용
  - 준지도 미세 조정(fine-tuning) 단계에서만 사용
  - 두 단계 모두 사용 

### 주요 결과
- **단일 도메인 시나리오**
  - 도메인 불일치가 클수록 semi-supervised fine-tuning의 효과가 더 커졌음   
- **다중 도메인 시나리오**
  - 다양한 억양에 대해 robust한 모델 성능 확보가 가능
  - 평균 WER을 낮추는 데에도 효과적 
- **결론**
  - unlabeled domain data를 활용한 semi-supervised fine-tuning은 **self-supervised learning의 도메인 적응 능력을 보완**
  - 본 접근법은 accent recognition뿐 아니라 noise robustness 등 다른 도메인 적응 시나리오에도 적용 가능








