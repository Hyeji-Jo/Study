# LoRA: Low-Rank Adaptation of Large Language Models
## 요약 정리
### Problem
- **Full Fine-Tuning의 비효율성**: 초거대 LLM(예: GPT-3 175B)을 태스크마다 전부 재학습·배포하는 것은 **메모리/저장/운영 비용이 과도**함
- **기존 PEFT의 한계**
  - **Adapter**: 모델 깊이 증가 → **추론 지연(latency)**, **분산 동기화 부담**
  - **Prefix/Prompt**: **유효 시퀀스 길이 감소**, 파라미터 늘려도 **성능 비단조**
- **요구사항**
  1. 파라미터·메모리 절감  
  2. 추론 지연 없이 배포  
  3. Full FT급 성능 유지  


### Contributions
- **LoRA (Low-Rank Adaptation)** 제안: 사전학습 가중치 $`W_0`$ **동결**, 업데이트를 **저랭크 행렬**로만 학습
- **추론 지연 0**: 배포 시 $`W = W_0 + BA`$ 로 **사전 병합** → 기존 연산 경로 유지
- **폭넓은 검증**: RoBERTa / DeBERTa / GPT-2 / GPT-3 등 전 범위에서 Full FT에 **준하거나 상회**
- **이론적 통찰**: 다운스트림 적응의 **내재적 랭크(intrinsic rank)** 가 낮음을 실증 (작은 $`r`$ 로도 SOTA 달성)  
- **실무성**: 태스크별 LoRA 모듈만 교체 → **빠른 전환·저장 효율** 확보 


### Method
- **핵심 파라미터화**
  - 업데이트를 $`\Delta W = BA`$ 로 제한
    - $`B \in \mathbb{R}^{d \times r}`$, $`A \in \mathbb{R}^{r \times k}`$, $`r \ll \min(d,k)`$
  - **Forward**: $`h = W_0x + \Delta Wx = W_0x + BAx`$ (추론 시 $`W_0 + BA`$ 병합 가능)
- **초기화 & 스케일링**
  - $`A \sim \mathcal{N}(0,\sigma^2)`$, $`B=0`$ → 시작 시 $`\Delta W=0`$ (사전학습 성능 유지)
  - 스케일 $`\alpha / r`$ 적용 (보통 $`\alpha=r`$ 로 단순화)  
- **적용 위치 (Transformer)**
  - 주로 **Self-Attention의 $`W_q`$, $`W_v`$** 에 적용 → **효율 대비 성능 영향 최대**  
  - 여러 행렬에 **낮은 $`r`$ 분산 적용**이 한 행렬 고랭크보다 효과적  


### Experiments & Setup
- **모델**: RoBERTa(base/large), DeBERTa-XXL(1.5B), GPT-2(med/large), GPT-3(175B)  
- **태스크**: GLUE, E2E / WebNLG / DART, WikiSQL, MNLI, SAMSum 등  
- **베이스라인**: Full FT, BitFit, Prefix-Embedding / Layer, Adapter(H/L/P/D)  
- **LoRA 설정**: 주로 **$`W_q`$, $`W_v`$** 에 적용, $`r`$ 은 1–8 범위에서 실험  


### Results
- **효율성**: GPT-3 기준  
  - 학습 파라미터 **10,000× 감소**  
  - GPU 메모리 **3× 절감**
- **성능**  
  - 대부분 태스크에서 **Full FT ≈ LoRA (= 또는 ↑)**  
  - Adapter / Prefix 대비 **우수하고 안정적**
- **추론 지연**  
  - Adapter 대비 **현저히 낮음 (= 기존과 동일 경로)** → **온라인 서비스 친화적**
- **스케일·안정성**  
  - Prefix는 파라미터 증가 시 **비단조 성능**, LoRA는 **단조적·안정적 향상**
- **데이터 효율**  
  - 저자원 (수백~수천 샘플) 환경에서도 **높은 데이터 효율성** 유지  

### Limitations
- **모듈 병합 배포 시 멀티태스크 처리 어려움** → 태스크별 LoRA 모듈 전환 필요  
- **적용 위치 선택이 휴리스틱 의존** → 자동화된 기준 연구 필요  
- **초저랭크 설정 ($`r`$ 매우 작을 때)** 은 태스크/레이어에 따라 **표현력 부족 가능**  


### Insights & Idea
- **저랭크 가설 (Intrinsic Rank Hypothesis)**  
  - 다운스트림 적응에 필요한 변화 $`\Delta W`$ 는 **저차원(subspace)** 에 집중  
  - 실증: $`r = 1 \sim 4`$ 만으로도 상위 성능, $`r`$ 확대 효과 한계  
- **LoRA가 학습하는 것**  
  - $`\Delta W`$ 는 $`W_0`$ 의 상위 성분을 “복사”하지 않고, **덜 강조된 태스크 특이 방향을 ‘증폭(amplify)’**.  
  - **증폭 계수**: $`r = 4`$ 에서 약 **20×**, 필요한 방향을 강하게 증폭해 성능 확보  
- **실무 포인트**  
  - $`W_q + W_v`$ 조합 + 낮은 $`r`$ 분산 적용이 최적의 효율·성능 균형  
  - 배포 전 **$`W = W_0 + BA`$ 병합**으로 **레이턴시 0 유지**, 태스크 전환은 **모듈 교체**로 해결  
- **미래 연구 방향**  
  - **PEFT 결합 (Compacter / Prefix 등)**  
  - **적용 위치 자동 탐색**  
  - **$`W_0`$ 의 랭크·중복성 구조 분석** 

### 기타
- **Forward**: $`h = W_0x + \frac{\alpha}{r}BAx`$ (학습 시), 배포 시 $`W \leftarrow W_0 + BA`$
- **학습 파라미터 규모**: 기존 $`O(dk)`$ → LoRA $`O(r(d+k))`$
- **현장 기본 설정 가이드**
  - **적용 위치**: $`W_q, W_v`$ 우선 → 필요 시 $`W_o`$ 추가 고려  
  - **랭크 $`r`$**: $`r=4`$ 전후로 시작 → 작게 설정 후 필요 시 확장  
  - **스케일 $`\alpha`$**: $`\alpha = r`$ 기본값, LR만 튜닝 



<br>  
  
## 0. Abstract
- 이 논문의 초록은 LoRA(Low-Rank Adaptation) 방법론을 제안하며, **대규모 언어 모델(LLM)을 특정 작업에 효율적으로 적응(adaptation)시키는 과정에서 발생하는 문제점들을 해결**하고자 함

### 기존 파인튜닝의 문제점
- **NLP의 패러다임 변화**: 최근 자연어 처리(NLP) 분야는 **대규모 데이터로 사전 학습된 모델을 특정 작업이나 도메인에 맞게 조정(adaptation)하는 방식으로 발전**
- 모델 크기가 커질수록 모든 파라미터를 재학습하는 **Full Fine-Tuning** 은 **비현실적**
- 예: GPT-3 (175B 파라미터)
  - 각 작업마다 독립적인 모델을 파인튜닝하여 배포하는 것은 **막대한 비용 및 자원 낭비**

### LoRA(Low-Rank Adaptation) 제안
- 사전학습된 가중치 $`W_0`$ **동결(freeze)**
- 대신 각 Transformer 레이어에 저랭크 행렬 A, B를 삽입해 업데이트를 $`W = W_0 + \Delta W = W_0 + BA`$로 표현
- 이러한 접근 방식은 다운스트림 작업을 위한 학습 가능한 파라미터의 수 감소

### 효과
- GPT-3 기준, 학습 파라미터 **10,000배 감소**, GPU 메모리 **3배 절감**.
- RoBERTa, DeBERTa, GPT-2, GPT-3 등에서 full fine-tuning과 동등하거나 우수한 성능
- Adapter 방식과 달리 **추가 추론 지연(latency)** 없음



<br>  
  
## 1. Introduction
### 기존 접근법의 한계 (Limitations)
| 접근법 | 개요 | 한계점 |
|--------|------|--------|
| **Full Fine-Tuning** | 전체 파라미터를 업데이트 | 파라미터 수 많고 학습 비용 ↑ |
| **Adapter Layers** (Houlsby et al., 2019) | Transformer 블록 사이에 작은 모듈 추가 | 추론 지연(latency) 증가 |
| **Prefix-Tuning** (Li & Liang, 2021) | 입력 앞에 학습 가능한 embedding 삽입 | 시퀀스 길이 감소, 최적화 어려움 |

- 기존 방법들은 **효율성과 성능 사이의 trade-off**가 존재

### LoRA(Low-Rank Adaptation)의 제안
- LoRA(Low-Rank Adaptation)는 **저랭크 행렬(low-rank matrices)** 을 통해 모델 적응 수행
- 사전학습된 가중치 $`W_0`$ 는 **동결(frozen)** 상태로 유지하고, 각 레이어의 dense layer에 **학습 가능한 두 행렬 $`A`$, $`B`$** 를 삽입
- $`h = W_0 x + \Delta W x = W_0 x + BAx`$
- A는 랜덤 가우시안으로 초기화되고, BBBB는 0으로 초기화되므로, 학습 시작 시 $`\Delta W = BA`$는 0이 되어 사전 학습된 모델의 초기 성능을 유지

### LoRA(Low-Rank Adaptation) 방법론의 핵심 아이디어
<img width="191" height="174" alt="image" src="https://github.com/user-attachments/assets/615b19eb-8012-4a0b-9bad-184fe9167b5d" />

| 구성 요소 | 설명 |
|------------|------|
| **$`x`$ / $`h`$** | 입력과 출력 벡터 |
| **$`W_0`$** | 사전학습된 Transformer 가중치 (동결됨) |
| **$`A, B`$** | 학습 가능한 저랭크 행렬 |
| **초기화 방식** | $`A \sim \mathcal{N}(0, \sigma^2)`$, $`B = 0`$ → 시작 시 $`\Delta W = 0`$ |
- 즉, 학습 초기에 모델의 원래 출력이 그대로 유지되고, 점진적으로 $`BA`$가 학습되어 fine-tuning 효과

### LoRA의 장점 (Advantages)
| 항목 | 설명 |
|------|------|
| **파라미터 효율성** | GPT-3 기준 학습 파라미터 수 **10,000배 감소** |
| **메모리 절감** | GPU 메모리 사용량 **3배 감소** |
| **추론 지연 없음** | 학습 후 $`W = W_0 + BA`$를 미리 병합하여 latency 없음 |
| **빠른 학습** | 대부분 파라미터가 고정되어 gradient 계산량 감소 |
| **유연성** | 하나의 base 모델에 여러 LoRA 모듈을 교체하여 사용 가능 |
| **결합 가능성** | Prefix-tuning 등 다른 PEFT 기법과 병행 가능 (orthogonal) |



<br>  
  
## 2. Problem Statement
### 배경 (Pre-train → Adapt)
- **패러다임**: 범용 말뭉치로 **사전학습**한 LLM을, 각 **다운스트림 태스크**에 맞게 **적응(adaptation)**
- **문제**: 모델이 거대해질수록(예: GPT-3 175B) 매 태스크마다 **전체 파라미터를 재학습·배포**하는 **Full Fine-tuning**은 저장/메모리/운영 비용이 과도함

### Full Fine-tuning의 목적함수 (식 1)
- 사전학습 가중치 $`\Phi_0`$ 로 초기화 후, 전체 파라미터 $`\Phi = \Phi_0 + \Delta\Phi`$ 를 직접 최적화
- **식 (1)**: $`\max_{\Phi} \sum_{(x,y)\in Z}\ \sum_{t=1}^{|y|}\ \log P_{\Phi}\!\left(y_t \mid x,\ y_{<t}\right)`$

  - **기호**
  - $`Z=\{(x_i,y_i)\}_{i=1}^N`$: 학습 데이터(컨텍스트 $`x`$, 타깃 $`y`$)
  - $`P_{\Phi}(y_t\mid x, y_{<t})`$: 자기회귀 LM의 조건부 확률
  - $`\Delta\Phi`$: 태스크별로 학습되는 **증분 파라미터**

- **핵심 한계**
  - 매 태스크에서 $`|\Delta\Phi| \approx |\Phi_0|`$ → **업데이트/저장 크기 = 원본 모델 크기**. 
  - 초거대 모델에서 **비현실적**

### LoRA의 문제 재정의 (식 2)
- 전체 $`\Phi`$ 대신 **작은 파라미터 집합** $`\Theta`$ 로 **증분** $`\Delta\Phi`$ 를 **간접 파라미터화**
- **식 (2)**: $`\max_{\Theta} \sum_{(x,y)\in Z}\ \sum_{t=1}^{|y|}\ \log p_{\Phi_0+\Delta\Phi(\Theta)}\!\left(y_t \mid x,\ y_{<t}\right)`$

- **핵심 아이디어**
  - $`\Phi_0`$ **동결**, $`\Delta\Phi(\Theta)`$ 만 학습
  - LoRA에서는 $`\Delta\Phi`$ 를 **저랭크 분해**로 표현: $`\Delta W = BA`$, $`B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll \min(d,k)`$
  - Forward 예: $`h = W_0 x + \Delta W x = W_0 x + BAx`$ ( $`W_0`$ 고정, $`A,B`$ 학습 )

- **효과**
  - $`|\Theta| \ll |\Phi_0|`$ (실무 보고치: **$`\sim 0.01\%`$ 수준**까지 축소 가능)  
  - **학습/저장/메모리** 부담 급감, 태스크별 모듈화 용이

### 식 (1) vs 식 (2) — 무엇이 달라졌나?
| 항목 | 식 (1): Full FT | 식 (2): LoRA |
|---|---|---|
| 최적화 변수 | $`\Phi`$ (전체) | $`\Theta`$ (아주 작음) |
| 사전학습 가중치 | 함께 업데이트 | **동결** |
| 태스크별 추가 저장 | **모델 전체 크기** | **저랭크 모듈만** (수 MB~수십 MB) |
| 추론 지연 | 없음 | **없음** (배포 시 $`W_0+BA`$ 병합 가능) |

- **식 (1)**: $`\max_{\Phi} \sum \log P_{\Phi}(y_t\mid x,y_{<t})`$ — 전체 파라미터 직접 최적화 → $`|\Delta\Phi|\approx|\Phi_0|`$
- **식 (2)**: $`\max_{\Theta} \sum \log p_{\Phi_0+\Delta\Phi(\Theta)}(\cdot)`$ — $`\Phi_0`$ 동결, **저랭크 업데이트** $`BA`$ 만 학습
- LoRA 핵심: **저차원(낮은 랭크) 변화만으로도 태스크 적응 가능**하다는 가설(= **intrinsic rank**)



<br>  
  
## 3. Aren’t Existing Solutions Good Enough?
### 기존 두 축
1) **Adapter Layers**  
- Transformer 블록 사이에 작은 **병목(bottleneck)** 모듈을 삽입해 태스크 특이 표현을 학습
- 파라미터는 적지만 **모델 깊이**를 증가시킴

2) **Prompt/Prefix Tuning**  
- 입력 프롬프트(또는 중간 활성화)에 **학습 가능한 벡터/토큰**을 붙여 적응
- **본체 가중치**는 그대로 두고, 입력 측에서 태스크 특이성을 유도


### Adapter의 한계
- **추론 지연(latency)↑**
  - 어댑터는 **추가 계층**이므로 **순차 연산 경로**가 길어짐 → 실시간/온라인 추론에서 치명적
- **병렬 효율 저하**
  - 작은 배치(예: 배치=1)에서 레이어 순차성 때문에 **GPU 활용도**가 떨어짐
  - (논문 Table 1 요지) **배치 1 기준 지연이 크게 증가**
- **모델 샤딩 악화**
  - 다중 GPU 분산 시, 깊이 증가로 **AllReduce/Broadcast** 등 **동기화 비용**이 늘어 병목 심화


### Prompt/Prefix Tuning의 한계
- **최적화 난이도**
  - 학습 가능한 프롬프트 길이를 늘려도 **성능이 단조 증가하지 않음**(plateau/불안정)
- **컨텍스트 길이 소모**
  - 프롬프트 토큰이 **유효 시퀀스 길이**를 잠식 → 실제 입력 처리 **컨텍스트 여유 감소**


### 왜 LoRA인가?
- **추론 경로에 깊이 추가 없음**
  - LoRA는 $`\Delta W`$ 를 **저랭크 분해** $`BA`$ 로 파라미터화하여 **기존 선형변환에 흡수** 가능
  - 배포 시 $`W \!=\! W_0 \!+\! BA`$ 를 **사전 병합**하면 **추가 연산·지연이 없다**
- **하드웨어 친화성**
  - 기존 **텐서 연산 경로** 유지 → **작은 배치**에서도 병렬 효율 손실이 작음
  - **샤딩 전략**도 기존과 동일하게 적용 가능(동기화 경로 증가 없음)
- **컨텍스트 길이 보존**
  - 입력 토큰을 차지하지 않음 → **가용 시퀀스 길이** 감소 문제 無
- **표현력 vs 효율성 균형**
  - $`r \ll \min(d,k)`$ 인 $`BA`$ 만 학습하여 **파라미터·메모리** 절감, 성능은 **Full FT에 근접/상회**


### 비교 한 눈에
| 방법 | 학습 파라미터 | 추론 지연 | 컨텍스트 길이 | 분산/샤딩 | 메모리/저장 |
|---|---|---|---|---|---|
| Full FT | **전체** $`\Phi`$ | 없음 | 보존 | 기존과 동일 | **매 태스크 모델 전체** 필요 |
| Adapter | 소수(모듈) | **증가**(깊이↑) | 보존 | **동기화 비용↑** | 모듈만 저장 가능 |
| Prefix/Prompt | 소수(프롬프트) | 경미 | **감소**(프롬프트가 차지) | 보통 | 프롬프트만 저장 |
| **LoRA** | **저랭크** $`BA`$ | **없음**(사전 병합) | 보존 | 기존과 동일 | **매우 작음**(모듈만) |



<br>  
  
## 4. Our Method — LoRA (Low-Rank Adaptation)
- 이 섹션은 **LoRA의 핵심 설계 원리**와 **수학적 구조**, 그리고 **실질적 장점 및 한계**를 체계적으로 설명한다
- LoRA는 **대규모 언어모델(LLM)** 의 파라미터 효율적 적응(PEFT)을 위해 설계된, **저랭크 업데이트(low-rank update)** 기반 방법이다

### 핵심 아이디어: Low-Rank Parametrized Update Matrices
- 일반적인 Dense Layer의 가중치 행렬 $`W_0 \in \mathbb{R}^{d \times k}`$ 는 **Full-Rank**이다
- 그러나 선행연구 (*Intrinsic Dimensionality Explains the Effectiveness of LM Fine-Tuning*) 에 따르면, **사전학습 모델의 업데이트 공간은 낮은 내재적 차원(intrinsic dimension)** 을 가진다
- 따라서 모델의 가중치 변화 $`\Delta W`$ 역시 **저랭크(low-rank)** 로 근사 가능하다는 가설을 세운다

#### 수식 표현
- $`W = W_0 + \Delta W = W_0 + BA`$
- $`B \in \mathbb{R}^{d \times r}`$, $`A \in \mathbb{R}^{r \times k}`$,  단 $`r \ll \min(d, k)`$
- 학습 시 $`W_0`$ 는 **동결(freeze)**, $`A, B`$ 만 **훈련 대상**

### Forward Pass
- 입력 $`x`$ 에 대해: $`h = W_0x + \Delta Wx = W_0x + BAx`$

| 구성요소 | 의미 |
|-----------|------|
| $`x`$ | 입력 벡터 |
| $`W_0x`$ | 사전학습된 가중치의 출력 |
| $`BAx`$ | LoRA 모듈의 추가 출력 (업데이트된 부분) |

#### 초기화
- $`A \sim \mathcal{N}(0, \sigma^2)`$, $`B = 0`$ → 학습 초기 $`\Delta W = 0`$
- 모델은 **사전학습 상태의 출력 유지**로 시작

#### 스케일링
- $`\Delta Wx`$ 에 스케일 계수 $`\alpha / r`$ 적용: $`\Delta W x = \frac{\alpha}{r} BAx`$
- $`\alpha`$ 는 상수로, 학습률(lr) 조정 효과를 제공 
- 논문에서는 기본적으로 $`\alpha = r`$ 로 설정해 별도 튜닝 불필요

### LoRA의 주요 장점 (Advantages)
| 구분 | 설명 |
|------|------|
| **Full Fine-tuning 일반화** | 모든 레이어에 적용하고 $`r`$ 을 충분히 키우면 Full FT 수준의 표현력 복원 가능 |
| **추가 추론 지연 없음** | 배포 시 $`W = W_0 + BA`$ 를 **미리 병합**하여 기존 연산 경로 그대로 사용 |
| **모듈 교체 용이성** | $`W_0`$ 는 고정, 태스크별 LoRA 모듈($`A, B`$)만 교체 가능 |
| **저장 효율성** | 체크포인트 크기 최대 **10,000배 축소** (GPT-3, $`r=4`$ 기준) |

### Transformer에의 적용 (Applying LoRA)
- Transformer에는 Self-Attention 모듈 내 가중치 4개  
  - $`(W_q, W_k, W_v, W_o)`$ 와 MLP 모듈 내 2개 가중치 존재
- LoRA는 주로 **Self-Attention의 $`W_q`$(Query), $`W_v`$(Value)** 에만 적용
  - 이유
    - 가장 **task-specific signal**이 많이 반영되는 위치  
    - **효율성 대비 성능 영향이 큼**
- MLP 모듈은 단순성 및 효율성 고려로 고정


### 실질적 이점 및 한계 (Practical Benefits & Limitations)

#### 주요 이점
- **VRAM 사용량 절감**  
  - 옵티마이저 상태를 $`W_0`$ 에 대해 저장할 필요가 없어 **최대 2/3 감소**. 
  - GPT-3 175B: **1.2TB → 350GB**
- **훈련 속도 향상**  
  - 전체 파라미터 중 극히 일부만 업데이트 → **약 25% 속도 개선**
- **Task Switching 용이**  
  - $`W_0`$ 유지, $`A,B`$ 교체만으로 다른 태스크로 전환

#### 한계
- $`W_0+BA`$ 를 병합해 배포하는 경우, 서로 다른 $`A,B`$ 세트를 동시에 사용하는 **멀티태스크 배치처리**가 어렵다
- 단, 지연이 중요치 않다면 병합 없이 모듈별로 동적 선택 가능

### 정리
| 항목 | 기존 Fine-tuning | LoRA 방식 |
|------|------------------|------------|
| 파라미터 업데이트 | $`\Delta W`$ (Full-rank) | $`\Delta W = BA`$ (Low-rank) |
| 학습 대상 | 전체 $`W_0`$ | 저랭크 행렬 $`A,B`$ |
| 학습 파라미터 수 | $`O(dk)`$ | $`O(dr + rk) \approx O(r(d+k))`$ |
| 추론 시 latency | 없음 | 없음 (사전 병합) |

### 연구 확장과 파생 연구
| 연구 | 핵심 아이디어 |
|------|----------------|
| **FLoRA (2023)** | LoRA를 **연합 학습(Federated Learning)** 환경에 적용 → 이기종 클라이언트 간 효율적 협업 학습 |
| **Compacter (2021)** | Adapter 구조에 **저랭크 하이퍼컴플렉스 연산**을 결합하여 파라미터 효율성 극대화 |
| **QLoRA (2023)** | LoRA + 4-bit 양자화 → 대형 모델의 저자원 fine-tuning 실현 |


<br>  
  
## 5. Empirical Experiments
- 이 섹션은 **LoRA(Low-Rank Adaptation)** 의 성능을 **RoBERTa, DeBERTa, GPT-2, GPT-3** 등 다양한 규모의 언어 모델에 적용해 검증한 **실험적 평가(Experiments)** 를 다룬다
- 목표는 LoRA가 기존 방법(Full Fine-tuning, Adapter, Prefix-tuning 등) 대비 **성능·효율성·추론비용** 측면에서 얼마나 우수한지를 실증하는 것

### 5.1 실험 개요 (Overview)
#### 평가 대상 모델 및 태스크
| 모델 | 유형 | 주요 벤치마크 / 태스크 |
|------|------|------------------------|
| **RoBERTa (base/large)** | NLU | GLUE benchmark |
| **DeBERTa XXL (1.5B)** | NLU | GLUE / SuperGLUE |
| **GPT-2 (medium/large)** | NLG | E2E NLG, WebNLG, DART |
| **GPT-3 (175B)** | NLU & NLG | WikiSQL, MNLI-matched, SAMSum |

- 평가 목표
  - LoRA가 기존의 Full Fine-tuning 및 다른 PEFT 방법(예: Adapter, Prefix-tuning, BitFit) 대비 **적은 학습 파라미터로 동등하거나 더 나은 성능**을 낼 수 있는지 검증
 
### 5.2 비교 기준 (Baselines)
| 방법 | 핵심 개념 | 학습 파라미터 |
|------|------------|---------------|
| **Full Fine-Tuning (FT)** | 전체 파라미터 $`\Phi`$ 업데이트 | $`|\Phi_0|`$ |
| **BitFit (Bias-only)** | bias만 학습 | 매우 적음 |
| **Prefix-Embedding (PreEmbed)** | 입력 프롬프트 토큰 임베딩 학습 | $`|\Theta| = d_{model}(l_p + l_i)`$ |
| **Prefix-Layer (PreLayer)** | 각 레이어 활성화 학습 | $`|\Theta| = L \cdot d_{model}(l_p + l_i)`$ |
| **Adapter (AdapterH/L/P/D)** | Dense layer 사이 bottleneck 모듈 삽입 | $`|\Theta| = \hat{L}_{Adpt}(2d_{model}r + r + d_{model}) + 2\hat{L}_{LN}d_{model}`$ |
| **LoRA (Ours)** | 가중치 업데이트 $`\Delta W = BA`$ 로 저랭크 학습 | $`|\Theta| = 2 \cdot \hat{L}_{LoRA} \cdot d_{model} \cdot r`$ |

- **LoRA**는 대부분의 실험에서 **$`W_q`$, $`W_v`$** (Query, Value)에만 적용하여 효율성과 성능 균형 확보

### 5.3 주요 결과 (Main Results)
#### **성능 및 효율성**
- LoRA는 RoBERTa, DeBERTa, GPT-2, GPT-3 전 모델에서 **Full Fine-tuning과 동등하거나 더 높은 성능**을 달성
- GPT-3 175B 기준
  - 학습 가능한 파라미터 수 **10,000배 감소**
  - GPU 메모리 요구량 **3배 절감**
  - 추론 시 **지연 시간 없음**

#### **추론 효율**
- Adapter류는 레이어 깊이가 증가해 latency가 늘어나지만, LoRA는 $`W = W_0 + BA`$ 로 **추론 시 병합**되어 **지연 없음**

#### **성능 안정성**
- Prefix-tuning은 파라미터 수를 늘릴수록 성능이 **비단조적(non-monotonic)** 으로 변함
- 반면, LoRA는 **확장성(scalability)** 과 **안정적 학습**을 보임
  - Figure 2: LoRA는 파라미터 증가에도 성능이 꾸준히 향상
 

### 5.4 RoBERTa 실험 (GLUE Benchmark)
- **모델:** RoBERTa base (125M), large (355M)
- **비교:** Full FT, BitFit, AdapterL/H/P/D
- **결과:** LoRA는 Full FT 대비 **동등 혹은 상회하는 성능**을, Adapter 대비 **적은 파라미터로 더 높은 점수**를 달성 (Table 2 상단)
- **설정:** 동일 배치(128), 동일 시퀀스 길이 적용으로 **공정 비교 보장**

### 5.5 DeBERTa XXL (1.5B Parameters)
- **목표:** LoRA가 초대형 모델에서도 Full FT 수준 성능을 달성 가능한지 검증
- **결과 (Table 2 하단)**
  - Full FT (1,500M) ≈ LoRA (4.7M trainable params)  
  - 즉, **99.7% 파라미터 절감에도 동일 성능 유지**
- **의의:** 대형 모델 적응 시 LoRA가 **비용 효율적이고 실용적**임을 입증

### 5.6 GPT-2 (Medium / Large) — NLG 실험
- **벤치마크:** E2E NLG Challenge (main), WebNLG, DART (appendix)
- **비교:** AdapterL/H, PreLayer, Full FT
- **결과 (Table 3)** - GPT-2 Medium 기준
  - LoRA (0.35M params) → BLEU 70.4점  
  - Full FT (354.9M) / AdapterL (11.1M)보다 더 우수
- **결론:** LoRA는 **적은 파라미터로 높은 생성 품질**을 유지하며, NLG 영역에서도 안정적으로 작동


### 5.7 GPT-3 175B — Large-Scale Evaluation
#### Motivation
- GPT-3급 초대형 모델은 Full Fine-tuning 자체가 **비현실적**
- LoRA는 이를 **실현 가능한 수준의 비용으로** 학습·배포 가능케 함

#### 실험 설정
- **태스크:** WikiSQL (SQL 생성), MNLI (추론), SAMSum (요약)
- **비교:** FT, BitFit, Prefix-tuning, AdapterH 등  
- **파라미터:** LoRA는 $`r=8`$ 로 설정, $`W_q, W_v`$ 만 학습

#### 결과 (Table 4, Figure 2)
| 태스크 | Full FT | LoRA | Prefix | Adapter |
|--------|----------|------|---------|----------|
| WikiSQL | ≈ | **= / ↑** | ↓ | ↓ |
| MNLI | ≈ | **=** | ↓ | ↓ |
| SAMSum | ≈ | **↑** | ↓ | ↓ |

- LoRA는 **Full FT 성능에 근접하거나 초과**
- Prefix류는 파라미터 수가 커질수록 오히려 성능이 **감소**
- LoRA는 파라미터 수 증가에 따라 **성능이 단조적 증가(monotonic)**

#### 데이터 효율성 (Appendix F.3)
- 데이터가 적은 환경 (MNLI 100~10K 샘플)에서도 LoRA는 Full FT, Adapter 대비 **높은 데이터 효율성**을 유지





<br>  
  
## 6. Related Works
### 6.1 Transformer Language Models
- **Transformer (Vaswani et al., 2017)** 는 NLP의 표준 아키텍처로 자리잡으며, **사전 학습(Pre-training) → 미세 조정(Fine-tuning)** 패러다임을 확립
- **GPT-2 (Radford et al., 2019)**, **GPT-3 (Brown et al., 2020)** 이후 모델 크기가 수십억~수천억 파라미터 규모로 확장되며, Full Fine-tuning의 **비용·저장·배포 한계**가 두드러짐
- 이로 인해, **“모든 파라미터를 업데이트하지 않고도”** 특정 태스크에 적응할 수 있는 효율적 접근법 연구가 활발히 진행됨

### 6.2 Prompt Engineering & Fine-Tuning
| 방법 | 개념 | 한계 |
|------|------|------|
| **Prompt Engineering** | GPT-3처럼 소수의 예시만으로 학습 없이 태스크 수행 | 프롬프트 설계에 따라 성능이 불안정 (“prompt hacking” 필요) |
| **Full Fine-tuning** | 모든 파라미터($`\Phi`$)를 업데이트하여 성능 극대화 | GPT-3급 모델에서는 **비현실적** — 저장·메모리·시간 문제 |

- LoRA는 이 두 극단 사이에서, **“적은 학습 파라미터로 Full FT에 준하는 성능”**을 목표로 함

### 6.3 Parameter-Efficient Adaptation (PEFT)
- 파라미터 효율적 미세 조정 연구는 **“어떤 부분만 학습할지”**에 초점을 맞춤

#### (1) Adapter Layers
- 기존 레이어 사이에 **작은 bottleneck 모듈** 삽입  → 태스크별 적응 수행 (Houlsby et al., 2019)
- 장점: 파라미터 효율적  
- 단점: **추론 시 깊이 증가 → latency 상승**
- **LoRA와 차이점**  
  - LoRA는 $`\Delta W = BA`$ 를 원래 $`W_0`$ 에 **병합(merge)** 할 수 있어 **추론 시 추가 연산이 없음**

#### (2) Prefix / Prompt Tuning
- 입력 앞단에 학습 가능한 임베딩(“가상 프롬프트”)을 붙이는 방식  
- 장점: 원본 모델 구조 변경 없음  
- 단점
  - **유효 시퀀스 길이 감소**
  - 파라미터 증가 → 성능이 **비단조적(non-monotonic)** 변화  
- **LoRA의 장점:** 입력 길이 보존 + 안정적 확장성 확보

### 6.4 Low-Rank Structures in Deep Learning
- 과도하게 파라미터화된 신경망은 실제로 **낮은 내재적 차원(intrinsic dimension)** 을 가진다는 연구 결과 다수 존재  
- LoRA는 이 이론을 확장하여 **“가중치 변화($`\Delta W`$) 또한 저랭크(low-rank) 공간에 존재한다”**는 가설을 제시
- 기존 연구들은 학습 시점에서 low-rank 제약을 직접 부여했지만, **LoRA는 사전 학습된 $`W_0`$ 를 고정하고, 업데이트 $`\Delta W = BA`$ 만 학습한다는 점에서 독창적**




<br>  
  
## 7. Understanding The Low-Rank Updates
### 7.1 어떤 가중치 행렬에 LoRA를 적용해야 할까?
- GPT-3 175B를 사용해, 제한된 학습 파라미터(약 1,800만 개) 내에서  **Transformer의 어떤 가중치 행렬에 LoRA를 넣는 게 가장 효율적인지** 탐구

#### 실험 설정
- Self-Attention의 네 가지 가중치 행렬
  - $`W_q`$ (Query), $`W_k`$ (Key), $`W_v`$ (Value), $`W_o`$ (Output)
- 한 가중치만 적용 시 $`r=8`$, 두 가중치에 적용 시 $`r=4`$로 조정
- 전체 96개 레이어에 동일하게 적용

#### 결과 (Table 5)
| 적용 대상 | 성능 요약 |
|------------|------------|
| $`W_q`$, $`W_k`$ 단독 | 성능 낮음 |
| $`W_v`$, $`W_o`$ 단독 | 상대적으로 양호 |
| **$`W_q`$ + $`W_v`$** | **최고 성능 달성** |
| $`W_q, W_k, W_v, W_o`$ (모두, $`r=2`$) | $`W_q+W_v`$ ($`r=4`$)와 비슷한 성능 |

#### 결론
- LoRA는 **한 행렬에 높은 $`r`$을 주기보다, 여러 행렬에 낮은 $`r`$을 분산 적용**하는 것이 효과적
- 즉, 여러 $`\Delta W = BA`$ 조합이 다양한 정보 방향을 포착


### 7.2 LoRA의 랭크 $`r`$은 얼마나 커야 할까?

#### 개념 복습
- LoRA는 가중치 업데이트를 $`\Delta W = BA`$로 표현하며, 여기서 $`r`$은 두 행렬의 “병목 차원(rank)”
- 즉, $`r`$이 작을수록 학습 파라미터 수 감소

#### 실험 결과 (Table 6)
- **$`r=1`$만으로도 경쟁력 있는 성능 달성**
- $`W_q`$ 단독보다 $`W_q + W_v`$ 조합이 훨씬 좋음
- 랭크를 크게 늘려도 성능 향상이 거의 없음

#### 부분공간(subspace) 분석 (Figure 3)
- $`r=8`$과 $`r=64`$로 학습된 LoRA 행렬의 특이값 분해(SVD) 결과 비교
- 상위 특이벡터(subspace) 방향이 **절반 이상 겹침 (similarity ≥ 0.5)**
- 이는 **핵심 정보가 저차원 공간(low-rank space)에 집중되어 있음**을 보여줌

#### 결론
- LoRA는 “가중치 변화($`\Delta W`$)” 자체가 낮은 **내재적 랭크(intrinsic rank)** 를 가짐
- 즉, 모델을 새로운 작업에 맞추는 데 필요한 변화는 **저차원에서도 충분히 표현 가능**

### 7.3 LoRA는 실제로 무엇을 학습하는가?
#### 연구 질문
1. $`\Delta W`$는 사전학습된 가중치 $`W`$와 어떤 관계일까?  
2. $`\Delta W`$는 단순한 보정값인가, 아니면 새로운 의미를 추가하는가?

#### 🔬 실험 결과
1. **상관관계**  
   - $`\Delta W`$는 무작위 행렬보다 $`W`$와 강한 상관성
   - 하지만 $`W`$의 상위 singular direction(주성분)을 반복하지 않음
   - 즉, **$`W`$에서 덜 강조된 feature 방향을 강화(task-specific amplification)**

2. **증폭 계수 (Amplification Factor):**  
   - 정의:  $`\text{Amp} = \frac{||\Delta W||_F}{||U^T W V^T||_F}`$  
     - 여기서 $`U, V`$는 $`\Delta W`$의 SVD 특이벡터
   - $`r=4`$ → 약 **20배 증폭 (6.91 / 0.32)**  
   - $`r=64`$ → 약 **2배** (노이즈 증가로 선택적 증폭 약화)

3. **해석**  
   - $`\Delta W`$는 $`W`$의 기존 feature를 “복사(copy)”하는 게 아니라, **태스크별로 필요한 방향을 선택적으로 크게 증폭(amplify)** 하는 역할
  



<br>  
  
## 8. Conclusion And Future Work
### 주요 결론 (Conclusion)
#### 기존 Fine-tuning의 한계
- 대규모 언어 모델(LLM)을 **전체 fine-tuning**하는 것은 현실적으로 비효율적
- 이유
  - 하드웨어 요구량이 너무 큼  
  - 저장 공간 및 배포 비용이 과도함  
  - 태스크(task)마다 별도 모델을 유지해야 함 → 전환 비용 증가

#### LoRA의 핵심 장점
| 항목 | 설명 |
|------|------|
| **효율성** | 사전 학습된 가중치 $`W_0`$는 고정하고, 작은 저랭크 행렬 $`A`$, $`B`$만 학습 |
| **추론 지연 없음** | 배포 시 $`W = W_0 + BA`$로 병합할 수 있어 추가 연산이 필요 없음 |
| **입력 길이 보존** | Prefix-tuning처럼 입력 시퀀스 길이를 줄이지 않음 |
| **모델 품질 유지** | Full fine-tuning 수준의 성능 달성 |
| **파라미터 공유 가능** | 대부분의 파라미터($`W_0`$)를 여러 태스크에서 재사용 가능 |
| **빠른 전환성** | 태스크별로 $`A, B`$ 행렬만 교체 → 즉시 새로운 작업으로 전환 가능 |

#### 범용성
- LoRA는 Transformer 기반 언어 모델뿐 아니라, **Dense Layer를 포함한 대부분의 신경망 구조**에도 일반화 가능
- 즉, 비전(vision), 음성(speech) 등 다른 도메인에도 응용할 수 있는 잠재력 존재

### 향후 연구 방향 (Future Work)

#### 다른 PEFT 방법들과의 결합
- LoRA는 **다른 파라미터 효율적 방법(PEFT)** 들과 결합 가능
- 예시 
  - **COMPACTER** (크로네커 곱 기반)와 결합 시 더 높은 효율성 기대
  - **Prefix-tuning**, **Adapter**와 직교적으로 조합 가능 → 성능 보완 효과

#### 모델 적응 메커니즘 이해
- Fine-tuning이나 LoRA가 **사전학습 모델의 feature를 어떻게 변화시켜 성능을 높이는지**  
  - 그 근본 원리는 아직 명확하지 않음
- LoRA는 전체 Fine-tuning보다 구조가 단순해, **모델 적응의 본질을 연구하기 위한 좋은 분석 틀(tractable framework)** 로 활용될 수 있음

#### LoRA 적용 위치의 체계화
- 현재는 어떤 가중치($`W_q, W_k, W_v, W_o`$ 등)에 LoRA를 적용할지 **경험적으로 선택**함
- 앞으로는 데이터 기반 분석이나 이론적 근거에 따라 **“최적의 적용 위치”를 자동으로 찾는 방법**을 개발할 필요가 있음

#### 가중치 랭크(rank)에 대한 근본 연구
- LoRA의 업데이트 행렬 $`\Delta W`$가 **낮은 랭크(rank-deficient)** 를 가진다는 점은, 사전 학습된 가중치 $`W_0`$ 자체도 랭크가 부족할 수 있음을 시사
- 이는 **대규모 모델의 구조적 중복(redundancy)** 과 **학습 효율성의 본질**을 탐구하는 중요한 단서가 될 수 있음




