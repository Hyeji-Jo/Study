# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
## 요약 정리
### Problem
- **In-Context Learning (ICL)** 은 few-shot 예시(demonstrations)만으로 새로운 태스크를 수행하는 능력  
- 하지만 **ICL이 왜 작동하는가**에 대한 근본적 이해가 부족함
- ICL은 소수의 데모만으로 새 태스크를 수행하지만 **왜/어떻게 작동하는지**가 불명확
  - 모델이 정말 데모의 **입력–레이블 매핑(input–label mapping)** 을 학습하는가?  
  - 아니면 단순히 **형식(format)** 과 **분포(distribution)** 를 이용하는가?  
- 초거대 LM(GPT-3 등)에 매 태스크별 fine-tuning은 **비용·운영상 비현실적** → ICL 메커니즘 이해가 필수 


### Contributions
1. **대규모 실증 분석**
   - 12개 LM (774M–175B) × 26개 데이터셋 → ICL 성능의 원인 요소를 분리·검증  
2. **반직관적 발견**
   - 데모의 **정답 라벨을 무작위로 바꿔도 성능 하락 0–5%p 수준**
     - Classification 평균 **2.6%p 감소**
     - Multi-choice 평균 **1.7%p 감소**
     - MetaICL: 단 **0.1–0.9%p 감소**
3. **핵심 3요소 규명**
   1. **Input Distribution (입력 분포)** — 인분포일수록 ICL 이득 ↑ (OOD 시 3–16%p ↓)  
   2. **Label Space (레이블 공간)** — 실제 레이블 vs 무작위 영어단어 차이 5–16%p  
   3. **Pairing Format (입력–레이블 형식)** — 형식이 깨지면 성능 붕괴  
4. **Ablation Study**
   - 정확한 라벨 비율(0~100%), 샷 수(k), 템플릿(minimal vs manual) 변화에도 결과 일관  
   - $`k ≥ 8`$ 이후 수익 체감(샷 효율 포화)
5. **MetaICL 분석**
   - 메타 트레이닝은 정답 매핑보다 **형식/분포 신호**에 더 의존하도록 모델을 유도


### Method
### 설정 비교
| 설정 | 설명 | 예측식 |
|------|------|---------|
| Zero-shot | 데모 없음 | $`\arg\max_{y\in C}P(y\mid x)`$ |
| Gold-label | 정답 데모 | $`\arg\max_{y\in C}P(y\mid x_1,y_1,…,x_k,y_k,x)`$ |
| Random-label | 무작위 레이블 데모 | $`\arg\max_{y\in C}P(y\mid x_1,\tilde y_1,…,x_k,\tilde y_k,x)`$ |

- **Direct model:** $`P(y|x)`$ 계산 → 레이블 공간에 민감  
- **Channel model:** $`P(x|y)`$ 계산 → 입력 분포에 민감  

### Ablation
- **정답 비율 조정:** 0%~100% (모두 틀려도 zero-shot보다 우수)  
- **샷 수($k$):** {0,4,8,16,32}, $`k≥8`$부터 성능 포화  
- **템플릿:** minimal vs manual → 언어적 복잡도보다 형식 일관성이 핵심 


### Experiments & Setup
| 항목 | 내용 |
|------|------|
| **모델** | GPT-2 Large, MetaICL(774M), GPT-J 6B, fairseq 6.7B/13B, GPT-3 175B |
| **구조** | 모두 decoder-only dense LM |
| **데이터셋** | 총 26개 (GLUE, SuperGLUE, 기타 도메인) |
| **평가 지표** | Classification: Macro-F1 / Multi-choice: Accuracy |
| **데모 수** | $`k=16`$, 균등 샘플링, 5개 랜덤 시드 |
| **추론 방식** | Direct vs Channel (Noisy Channel LM Prompting) |


### Results
| 항목 | 주요 수치 | 해석 |
|------|------------|------|
| Random vs Gold | 0–5%p↓ (평균 2.6/1.7%) | 정답 매핑 영향 작음 |
| MetaICL | 0.1–0.9%p↓ | 형식 신호 활용 극대화 |
| OOD 입력 | 3–16%p↓ | 입력 분포 중요 |
| 랜덤 영어 라벨 | 5–16%p↓ (Direct), 0–2%p (Channel) | Label Space 역할 다름 |
| 형식 제거(no labels 등) | 성능 붕괴 (≈ zero-shot) | 구조적 신호 상실 |
| 형식 유지만 | ICL 이득의 75–95% 유지 | Format이 핵심 |


### Limitations
1. **작업 범위 제한**
   - Classification·Multi-choice 중심 → Generation 등 open-set에는 미확인  
   - 합성(synthetic) 과제에서는 정답 매핑 중요도 ↑ 가능 (*Rong, 2021*)
2. **데이터셋별 변동성**
   - *Financial PhraseBank*에서 **최대 14%p 차이** 보고 (*Kim et al., 2022*)  
   - 해당 연구는 랜덤·부정(negated) 레이블을 보간(interpolation) → 본 논문보다 세밀한 설정  
3. **생성/CoT 일반화 한계**
   - *Madaan & Yazdanbakhsh (2022)*:  
     - 무작위 추론(rationale)은 CoT 성능 저하  
     - 그러나 ‘틀린 구조적 추론’은 영향 적음 → **형식의 중요성 재확인**


### Insights & Idea
- **핵심 통찰**: ICL은 데모에서 **새 매핑을 학습**하기보다, 사전학습 지식을 **형식·분포·출력공간** 신호로 **정렬·호출**  
- **프롬프트 설계 가이드**  
  | 원칙 | 설명 |
  |------|------|
  | **형식 유지** | 입력→레이블의 짝 구조를 명확히. “문장 → 답변” 형태 유지 |
  | **인분포 입력** | 테스트와 유사한 스타일/도메인의 문장 사용 |
  | **레이블 공간 명시** | 가능한 출력 후보(`positive`, `negative`, `neutral`)를 제시 |
  | **샷 효율** | $`k≥8`$ 이후 이득 적음 → 소수 대표 예시로 충분 |
  | **Meta-training** | 형식·분포 민감도를 키워 ICL 효과 증폭 가능 | 
- **연구 아이디어**  
  - 사전학습에 내재된 입력–레이블 매핑을 **검색/편집/프롬프트**로 더 안정적으로 호출  
  - **생성 태스크**에서도 분포·형식만으로 ICL 이득을 재현하는 프로토콜 설계  
  - ICL + **소량 라벨 튜닝(PEFT/LoRA)** 의 하이브리드로 **완전히 새로운 매핑** 태스크 대응


<br>  
  
## 0. Abstract
- 이 논문은 **In-Context Learning (ICL)** 이 실제로 **어떻게 작동하는가**를 실험적으로 분석한 연구
- ICL은 **대규모 언어모델 (Large Language Model; LLM)** 이 몇 개의 **입력–레이블 쌍(demonstrations)** 에 조건화하여, **추론만으로 새로운 태스크를 수행**하는 능력을 말한다.  
  - 예를 들어, “이 문장은 긍정 / 부정”과 같은 몇 가지 예시를 보여주면, 모델이 추가 학습 없이도 새로운 문장의 감정을 분류할 수 있음
- 그러나 기존 연구들은 ICL의 **작동 원리**와 **데모의 어떤 요소가 성능 향상에 기여하는지**에 대한 명확한 이해를 제공하지 못함
- 즉, 모델이 실제로 **입력과 라벨 간의 관계를 학습하는지**, 아니면 단순히 형식적 패턴을 따라가는지에 대한 분석이 부족

### 핵심 발견
- 연구팀은 GPT-3를 포함한 **12개의 대형 언어모델**을 대상으로 실험 수행
- 그 결과, 데모 내의 **정답 라벨(ground truth labels)** 을 **무작위(random)** 로 바꿔도 모델의 성능이 거의 떨어지지 않는다는 사실 발견 

- 이는 기존의 직관을 뒤집는 결과로, 모델이 데모의 **입력–레이블 매핑(input–label mapping)** 자체를 학습하는 것이 아니라는 점을 시사
- 대신 모델은 다음 세 가지 요소에 의해 크게 영향을 받는다

  1. **Label Space (레이블 공간)**  
     - 가능한 라벨들의 집합 (예: `positive`, `negative`, `neutral`)  
     - 모델이 출력할 수 있는 답변의 범주를 규정함
  
  2. **Input Distribution (입력 분포)**  
     - 데모로 주어진 문장들의 도메인, 언어적 스타일, 분포적 특성
     - 입력의 유형이 모델의 추론 방향을 결정하는 중요한 단서로 작용
  
  3. **Format (형식적 구조)**  
     - “입력 → 라벨”의 짝 구조를 가진 문맥 포맷
     - 모델이 태스크 수행 모드로 전환되도록 유도하는 구조적 신호

### 연구의 기여
- 이 연구는 **“ICL은 실제로 무엇을 학습하는가?”** 라는 근본적인 질문에 처음으로 **정량적이고 실험적인 근거** 제시

- **정답 데모가 없어도 성능이 유지**된다는 결과를 통해, 모델이 입력–라벨 대응관계를 학습하기보다는 **분포적 패턴과 문맥 구조**에 반응함을 입증
- ICL은 단순한 few-shot 학습이 아니라, 사전학습(pretraining) 중 내재된 지식을 **형식(format signal)** 을 통해 **활성화(retrieval)** 하는 과정으로 해석 가능
- 결과적으로, ICL은 **추론 기반 메타학습 (inference-based meta learning)** 의 한 형태로 이해될 수 있음



<br>  
  
## 1. Introduction
### In-Context Learning (ICL) 개요
- **대규모 언어모델 (Large Language Models; LMs)** 은 소수의 입력–레이블 쌍(demonstrations)에 조건을 부여하여 **추론만으로 새로운 작업을 수행** 
  - 이러한 능력을 **In-Context Learning (ICL)** 
  - 예: GPT-3는 단 몇 개의 예시만으로도 감정분류나 문장추론과 같은 태스크를 수행할 수 있음 (“Language Models are Few-Shot Learners”)
- ICL은 다양한 자연어 처리 작업에서 **zero-shot 추론보다 훨씬 높은 성능**을 보이지만, 모델이 **실제로 어떻게 학습하는지**, 그리고 **demonstrations의 어떤 속성이 성능에 기여하는지**는 명확히 이해되지 않음

### 반직관적 핵심 발견
- 논문은 “ICL에서 **정답 레이블(ground truth label)** 은 반드시 필요하지 않다”는 반직관적인 결과를 제시
- 데모의 라벨을 무작위(random)로 바꿔도 성능 저하가 거의 없으며, 이는 **GPT-3를 포함한 12개 이상의 모델**에서 일관되게 관찰됨
- 이 결과는 모델이 데모 내의 **입력–레이블 매핑(input–label mapping)** 자체를 활용하지 않고, **다른 요인들에 의존하여 태스크를 수행**하고 있음을 보여줌

### 성능에 기여하는 핵심 요소
- 논문은 ICL의 성능을 결정짓는 세 가지 주요 요인 식별

1. **Label Space (레이블 공간)**  
   - 가능한 레이블들의 집합(예: `positive`, `negative`, `neutral`)이 제공됨으로써 모델이 **출력 가능한 답변의 범주** 인식 가능

2. **Input Distribution (입력 분포)**  
   - 데모에 포함된 문장들이 특정 도메인(뉴스, 대화, 리뷰 등)의 **언어적 특성과 분포** 반영
   - 모델은 이러한 입력 분포를 통해 **문맥적 특성을 학습하거나 추론 방향을 조정**

3. **Format (전체 형식 구조)**  
   - 입력–레이블 쌍이 제시되는 **일관된 문맥 구조(sequence format)** 자체가 모델에게 “이런 형태의 입력에는 이런 종류의 출력을 해야 한다”는 **구조적 시그널(structural cue)** 을 제공


### 새로운 이해와 연구 질문
- 이러한 분석은 **ICL이 왜, 어떻게 작동하는가**에 대한 새로운 관점 제시
- 특히, **MetaICL** 과 같이 ICL 목적 함수로 **메타 학습(meta-training)** 된 모델들은 입력–레이블 매핑보다는 **형식(format)** 과 같은 **단순한 구조적 요소를 주로 활용**하는 경향 존재
- 결과적으로, ICL은 모델이 데모를 통해 “새로운 태스크를 학습”한다기보다, **이미 사전학습(pretraining) 단계에서 내재된 지식을 호출(retrieve)** 하는 과정으로 볼 수 있음

### 기존 이해에 대한 도전
- 이 논문은 ICL의 기존 가설에 도전
  - “모델이 데모에 포함된 입력–레이블 쌍을 통해 새로운 태스크를 명시적으로 학습한다”
- 대신, 모델이 **사전훈련(pretraining)** 단계에서 이미 획득한 지식을 활용하여 데모를 **작업 위치 지정(task location)** 신호로 사용한다는 가능성 제시  
- 이러한 관점은 *Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm* 등의 선행연구에서 제시된 “데모는 학습(learning)이 아니라, **작업 위치(task location)** 를 위한 것”이라는 주장과 일치




<br>  
  
## 2. Related Work
### 대규모 언어모델(Large Language Models, LMs)의 발전과 한계
- **대규모 언어모델 (LMs)** 은 다양한 자연어처리(NLP) 과제에서 탁월한 성능
  - 대표적으로 BERT, GPT, RoBERTa, T5, BART 등 존재
- 새로운 태스크에 LMs를 적용하기 위한 전통적인 방법은 **Fine-tuning**
  - 즉, 사전학습된 언어모델을 각 태스크별로 추가 학습시키는 방식 
- 그러나 **수십억(≥10B)** 개 이상의 파라미터를 가진 초거대 모델(GPT-3 등)은 매 태스크마다 파라미터를 업데이트하고 저장하는 것이 **비용적으로나 계산적으로 비현실적**
- 이러한 한계를 해결하기 위해, **추론(inference)** 단계에서만 새로운 태스크를 학습하는 **ICL** 개념 등장

### In-Context Learning(ICL)의 등장
- *Language Models are Few-Shot Learners* (Brown et al., 2020) 논문에서 **In-Context Learning(ICL)** 이 새로운 학습 패러다임 처음 제안  
- ICL은 **gradient 업데이트 없이**, 소수의 **입력–레이블 쌍(demonstrations)** 에 조건을 부여(conditioning)하여 **추론만으로 새로운 작업을 수행**
- Figure 2의 개념 예시
  - Circulation revenue has increased by 5% in Finland → Positive
  - Paying off the national debt will be extremely painful. → Negative
  - 위와 같이 데모 몇 개를 문맥에 제시하면, 모델은 학습 없이 새로운 입력의 레이블 예측 가능

### 기존 ICL 연구 동향
- ICL 도입 이후 활발한 연구가 이루어졌으며, 주요 연구 흐름은 다음 네 가지로 구분

1. **문제 공식화 개선 (Problem Formulation)**  
 - ICL 문제를 더 효과적으로 정의하고 안정화하는 연구  
 - 예시: *Calibrate Before Use (Zhao et al., 2021)*,  
   *Surface Form Competition (Holtzman et al., 2021)*,  
   *Noisy Channel Language Model Prompting (Min et al., 2021a)*

2. **Demonstration 선택 전략 (Example Selection)**  
 - 어떤 예시를 데모로 넣을지, 순서를 어떻게 정할지에 대한 연구  
 - 예시: *What Makes Good In-context Examples for GPT-3 (Liu et al., 2021)*,  
   *Fantastically Ordered Prompts (Lu et al., 2021)*,  
   *Learning to Retrieve Prompts (Rubin et al., 2021)*

3. **메타학습 기반 ICL (Meta-training)**  
 - ICL을 명시적인 학습 목표(objective)로 두고, 다양한 태스크를 통해 **메타 학습(meta-learning)** 을 수행  
 - 예시: *MetaICL (Min et al., 2021b)*, *Meta-learning via Language Model In-context Tuning (Chen et al., 2021)*

4. **Instruction-following (지시문 기반 학습)**  
 - 태스크를 예시 대신 **자연어 설명(instruction)** 으로 제공하여 모델이 이를 따르도록 학습  
 - 예시: *Finetuned Language Models are Zero-Shot Learners (Wei et al., 2022a)*, 
   *Multitask Prompted Training (Sanh et al., 2022)*,  
   *Cross-task Generalization via Instructions (Mishra et al., 2021b)*

### ICL의 취약성과 한계 보고
- 일부 연구에서는 ICL의 **취약성(brittleness)** 과 **과도한 민감성(over-sensitivity)** 을 지적
- 예시: *Fantastically Ordered Prompts (Lu et al., 2021)*,  
  *Calibrate Before Use (Zhao et al., 2021)*  
- 데모 순서나 표현이 약간만 바뀌어도 결과가 크게 달라지는 현상이 관찰

### ICL 작동 원리에 대한 이해 부족
- 위의 연구들이 ICL의 성능 향상 방법을 제시하긴 했지만, **“왜 ICL이 작동하는가?”** 라는 근본적인 질문에는 충분히 답하지 못함
- 일부 관련 연구
  - *An Explanation of In-context Learning as Implicit Bayesian Inference (Xie et al., 2022)*  
    - ICL을 **베이지안 추론(Bayesian inference)** 으로 해석
  - *Impact of Pretraining Term Frequencies on Few-shot Reasoning (Razeghi et al., 2022)*  
    - 사전학습 데이터의 **용어 빈도(term frequency)** 가 ICL 성능과 밀접히 연관됨을 보임
  
### 본 논문의 기여
- 본 논문은 ICL이 **zero-shot 추론보다 더 높은 성능을 보이는 이유**를 **실증적으로 분석한 최초의 연구 중 하나**
- Demonstrations 내의 **ground truth input–label 매핑이 실제로는 거의 영향을 주지 않는다**는 점을 밝혔으며, 데모를 구성하는 다양한 요소(입력 분포, 라벨 공간, 형식 구조)가 **각각 어떤 영향을 미치는지 정량적으로 측정** 
- 이를 통해 “ICL은 실제로 무엇을 학습하는가?”라는 근본적인 질문에 대해 새로운 실험적 근거 제시



<br>  
  
## 3. Experimental Setup
### 사용된 모델 (Models)
- 총 **12개의 언어모델(Language Models; LMs)** 을 실험에 사용
  - **GPT-2 Large (774M)**  
  - **MetaICL (774M)** — GPT-2 Large를 기반으로 **ICL 목적 함수로 메타 학습(meta-trained)** 된 모델  
  - **GPT-J (6B)**  
  - **fairseq 6.7B, fairseq 13B**  
  - **GPT-3 (175B)**  
- 모든 모델은 **decoder-only 구조의 dense LM** 아키텍처
- 모델 크기는 **774M → 175B**에 이르기까지 다양하며, 이를 통해 모델 크기에 따른 일반화 및 성능 차이를 함께 분석
- 특히, **MetaICL** 은 사전학습(pre-training)된 LM을 ICL 태스크 중심으로 재학습한 모델로, 일반 LM이 보이지 않는 ICL의 메커니즘을 파악하는 데 중요한 비교 기준으로 사용

### 추론 방식 (Inference Methods)
- 각 LM은 두 가지 방식으로 평가
  1. **Direct 방식**  
     - 입력(`x`)을 조건으로 하여 레이블(`y`)의 확률 $`P(y|x)`$ 를 계산
  2. **Channel 방식**  
     - 반대로, 레이블(`y`)을 조건으로 입력(`x`)의 확률 $`P(x|y)`$ 를 계산한 뒤 역으로 추론  
- 이 두 방식은 *Noisy Channel Language Model Prompting for Few-Shot Text Classification*  (Min et al., 2021a) 에서 제안된 접근법을 따른다
- 동일한 모델이라도 두 방식 간에 ICL 성능 차이가 존재하며, 이를 통해 모델의 조건화 방식이 결과에 미치는 영향을 함께 분석

### 평가 데이터 (Evaluation Data)
- 총 **26개의 데이터셋** 에 대해 평가를 수행 
  - 분류(Classification) 및 다중선택(Multi-choice) 태스크 모두 포함  
  - 전체 목록은 **부록 A(Appendix A)** 에 수록되어 있다
- **데이터셋 선정 기준:**
  1. **Low-resource datasets** — 학습 예제가 **10K 미만**인 데이터셋 중심으로 구성  
  2. **표준 벤치마크 포함** — GLUE (*A Multi-Task Benchmark for NLU*) 및 SuperGLUE (*A Stickier Benchmark for General-Purpose NLU Systems*) 포함  
  3. **다양한 도메인** — 과학(Science), 금융(Finance), 소셜미디어(SNS) 등 다양한 텍스트 도메인 포괄  

### 기타 세부 사항 (Other Details)
- **데모 예시 개수 ($k$)**  
  - 기본적으로 $`k = 16`$ 개의 예시를 데모로 사용  
  - 예시는 학습 데이터에서 **균등 샘플링(uniform sampling)** 하여 구성
- **랜덤 시드(Random Seeds)**  
  - 데모 예시를 선택할 때 **5개의 랜덤 시드(seed)** 를 사용  
  - 모든 실험은 각 시드별로 5회 반복 수행  
  - 단, **fairseq 13B** 와 **GPT-3** 는 자원 한계로 **6개의 데이터셋 × 3개의 시드** 로 제한하여 수행
- **평가 지표 (Metrics)**  
  - **Classification:** Macro-F1 (클래스 불균형에 더 적합)  
  - **Multi-choice:** Accuracy (정답률)
- **입력 템플릿 (Templates)**  
  - 입력 시퀀스는 **최소한의 템플릿(minimal templates)** 을 사용하여 구성  
  - 수동(manual) 템플릿도 실험적으로 탐색했으나, 일관적으로 더 높은 성능을 보이지 않아 기본적으로 minimal template을 유지  
  - 예시 템플릿은 부록 B(Appendix B)에 제공


### 연구적 의의
- 본 실험 설정은 **ICL이 왜 성능 향상을 보이는지**를 밝히기 위한 **최초의 체계적 실험적 기반** 제공  
- 이후 연구인 *Ground-Truth Labels Matter: A Deeper Look into Input-Label Demonstrations* (Kim et al., 2022) 등에서도 이 실험 구조를 토대로 분석이 확장
- 특히, 이 논문은 **“ICL의 성능 향상이 실제 학습 때문인가?”** 라는 근본적 질문에 접근하기 위한 **정량적 분석의 초석** 마련



<br>  
  
## 4. Ground Truth Matters Little
- 이 장에서는 **In-Context Learning (ICL)** 에서 데모(demonstrations)의 **정답 입력–레이블 매핑(ground truth input–label mapping)** 이 실제로 얼마나 중요한지를 실험적으로 분석  
- 즉, 데모에서 입력($x$)과 레이블($y$)의 대응 관계가 정확해야만 ICL이 잘 작동하는지, 혹은 다른 요소들이 더 큰 영향을 미치는지를 평가

### 4.1 Gold Labels vs. Random Labels
#### 실험 목적
- 데모에 포함된 입력–레이블 쌍이 **정확한 매핑**인지 여부가 ICL 성능에 미치는 영향을 정량적으로 측정  
- 즉, 모델이 데모로부터 **입력–레이블 관계 자체를 학습하는지**, 아니면 **형식적 패턴과 분포적 특성**만을 활용하는지를 검증

#### 비교 방법
- 세 가지 조건에서 모델 성능을 비교

1. **No Demonstrations (데모 없음)**  
   - 일반적인 **zero-shot 추론** 방식  
   - 입력 $x$ 만 보고 가능한 레이블 집합 $C$ 중에서 가장 확률이 높은 $y$를 선택  
   - 수식: $`\operatorname{argmax}_{y \in C} P(y|x)`$

2. **Demonstrations w/ Gold Labels (정답 레이블 데모)**  
   - 전통적인 ICL 방식  
   - $k$개의 올바른 입력–레이블 쌍 $`(x_1, y_1), \dots, (x_k, y_k)`$ 을 데모로 사용하고,  
     테스트 입력 $x$를 추가하여 예측을 수행 
   - 수식: $`\operatorname{argmax}_{y \in C} P(y|x_1, y_1, \dots, x_k, y_k, x)`$

3. **Demonstrations w/ Random Labels (무작위 레이블 데모)**  
   - 각 입력 $`x_i`$ 에 대한 정답 $`y_i`$ 대신, 가능한 레이블 집합 $C$에서 무작위로 샘플링된 $`\tilde{y_i}`$를 사용  
   - 데모는 $(x_1, \tilde{y_1}), \dots, (x_k, \tilde{y_k})$ 형태를 가지며, 입력 분포, 레이블 공간, 전체 형식은 유지하되 **입력–레이블 대응만 깨진 상태**
   - 수식: $`\operatorname{argmax}_{y \in C} P(y|x_1, \tilde{y_1}, \dots, x_k, \tilde{y_k}, x)`$

#### 주요 결과
- **정답 레이블 데모의 효과**  
  - “No demonstrations”에 비해 “Demonstrations w/ gold labels”은 모든 모델에서 **유의미한 성능 향상** 
  - 이는 기존 연구(*Language Models are Few-Shot Learners*)의 결과와 일치하며, ICL이 zero-shot보다 효과적임을 다시 확인

- **무작위 레이블 데모의 놀라운 성능**  
  - Gold label 데모와 Random label 데모를 비교했을 때, **무작위 레이블을 사용해도 성능 하락은 미미**
  - 대부분의 모델에서 **0–5%** 이내의 절대 성능 감소에 그쳤으며,  
    - Classification: 평균 **2.6% 감소**  
    - Multi-choice: 평균 **1.7% 감소**
  - **GPT-3를 포함한 12개의 모델** 전반에서 일관된 경향을 보였다

- **MetaICL 모델의 특이점**  
  - MetaICL(메타 학습된 LM)은 무작위 레이블 사용 시 성능 하락이 **0.1–0.9% 절대 감소**에 불과
  - 이는 MetaICL이 데모의 **입력–레이블 매핑을 거의 무시하고**, **형식적 구조(format)나 분포 정보(distributional cues)** 만으로 태스크를 수행한다는 점을 시사


#### 해석 및 의미
- 모델은 데모에 제시된 **정확한 입력–레이블 매핑에 의존하지 않아도** 작업을 수행 
- 즉, 데모의 올바른 정답쌍 $(x, y)$ 을 학습하는 것이 아니라, **사전학습(pretraining)** 을 통해 이미 내재된 지식을 활용하거나, 데모의 **입력 분포**, **레이블 공간**, **문맥 형식(format)** 으로부터 “기대되는 입력–레이블 관계”를 **복구(recover)** 하는 것으로 보임
- 이 결과는 ICL이 “데모로부터 새로운 태스크를 학습한다”는 기존의 직관에 도전하며, **대규모 언어모델이 추론만으로 얼마나 많은 정보를 재활성화할 수 있는가**에 대한 새로운 연구 방향을 제시

#### 관련 연구 및 확장 논의
- **Ground-truth Labels Matter: A Deeper Look into Input-Label Demonstrations (Kim et al., 2022)**  
  - 무작위 레이블뿐 아니라 ‘부정된(negated) 레이블’을 사용했을 때 성능이 급격히 저하됨을 발견  
  - 이는 “모든 레이블이 중요하지 않다”기보다, **의미적 관련성(semantic relevance)** 이 완전히 깨질 경우 성능이 영향을 받음을 보여줌

- **Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango (Madaan & Yazdanbakhsh, 2022)**  
  - Chain-of-Thought(CoT) 프롬프팅에서도 유사한 분석을 수행  
  - 무작위 추론 경로는 성능을 크게 낮추지만, **‘틀린 논리 구조’** 등은 의외로 성능에 큰 영향을 주지 않음  
  - 이는 **“정답의 내용보다 형식의 일관성이 더 중요하다”** 는 본 논문의 통찰을 CoT 영역으로 확장한 사례

### 4.2 Ablations
- 이 절에서는 **Ablation Study** 를 통해 ICL에서 데모의 구성 요소들이 성능에 어떻게 기여하는지를 보다 정밀하게 분석 
- Ablation은 시스템의 특정 요소를 제거하거나 변형하여 전체 성능에 미치는 영향을 관찰하는 실험 기법

#### 1. 정확한 레이블 비율 실험  
- *(Does the number of correct labels matter?)*

- 데모 내 **정확한 레이블 비율($a$%)** 을 0%–100%로 다양하게 조정하여 실험  
  - “Demonstrations w/ $a$\% correct labels”는 $`k \times (a/100)`$ 개의 올바른 쌍과 $`k \times (1 - a/100)`$ 개의 잘못된 쌍으로 구성
- 결과
  - **정확한 레이블 비율**이 높아져도 성능 향상은 미미 
  - **0% (모두 틀린 레이블)** 인 경우에도 no-demonstration(=zero-shot)보다 훨씬 뛰어난 성능을 유지  
- 해석
  - 모델은 **정답 매핑의 정확도보다는** **입력의 분포, 레이블 공간, 형식적 구조** 등 다른 요인들을 더 강하게 활용

#### 2. 데모 예시 수($k$) 변경 실험  
- *(Is the result consistent with varying k?)*

- 데모의 개수 $k$ 를 {0, 4, 8, 16, 32}로 변화시키며 실험  
- 결과
  - Gold labels와 Random labels 모두에서 $`k \ge 8`$ 이후 성능 향상이 거의 정체됨  
  - 이는 **ICL이 소수의 예시만으로도 입력 분포 및 형식을 파악**할 수 있음을 시사  
- 해석
  - 모델은 많은 예시보다, **일부 대표적인 분포적·구조적 신호**로 충분히 태스크를 유추할 수 있음

#### 3. 템플릿 종류 변경 실험  
- *(Is the result consistent with better templates?)*

- 기본적으로 사용된 **minimal templates** 외에, **수동(manual) 템플릿**을 적용하여 실험  
- 결과
  - Manual template을 사용해도 Gold → Random label 대체 시 성능 저하는 여전히 미미  
  - 즉, **템플릿의 정교함 자체가 핵심 요인은 아님**
- 해석
  - 모델은 **템플릿의 언어적 복잡성보다는** **입출력 구조의 일관성(format consistency)** 에 더 민감




<br>  
  
## 5. Why does In-Context Learning work?
- 이 장은 데모(demonstrations)에서 **무엇이** 인컨텍스트 러닝(ICL) 성능을 결정하는지 분해하여 분석한다 
- 핵심 결론은 **정답 입력–레이블 매핑 자체는 덜 중요**하고, 다음 세 요소가 주효하다는 것  
  
  1. 입력 텍스트의 **분포**  
  2. 가능한 출력의 **레이블 공간**  
  3. 입력과 레이블의 **페어링 형식(format)**


### 5.1 Distribution of the Input Text
**설정**  
- 데모의 입력 $`x_1,\dots,x_k`$ 를 원 데이터셋 대신 외부 코퍼스에서 무작위 샘플로 치환  
- 레이블 공간과 전체 형식은 유지하여 **입력 분포만** 바꿈

**결과**  
- 여러 모델에서 인분포(in-distribution) 입력을 쓸 때 대비 **3–16%p 성능 하락**  
- Direct GPT-J의 일부 멀티초이스 태스크는 **데모 없음**보다도 더 악화  
- 단, Direct MetaICL은 영향이 상대적으로 작음

**해석**  
- 데모는 모델에게 “어떤 종류의 텍스트가 들어올지”를 알려 **사전학습 지식을 올바른 영역으로 호출**하게 만든다  
- 인분포 입력이 없으면 ICL 이득이 크게 줄어든다

### 5.2 Label Space
**설정**  
- 원래 레이블 대신, 크기만 동일한 무작위 영어 단어 집합 $`C_{\text{rand}}`$ 에서 레이블을 치환  
- 입력 분포와 형식은 유지

**결과**  
- **Direct 모델**: 실제 레이블 공간의 임의 라벨 vs 무작위 영어 단어 라벨 사이에 **5–16%p 하락**  
- **Channel 모델**: 출력(레이블) 공간 제거의 영향이 **0–2%p 이내**로 작거나 경우에 따라 증가

**해석**  
- Direct는 “무엇을 출력해야 하는가”를 직접 생성하므로 **레이블 공간 정보**가 중요  
- Channel은 $`P(x|y)`$ 형태라 출력 어휘의 구조를 덜 필요로 함


### 5.3 Pairing Format
**설정**  
- 입력–레이블 **페어링 형식 자체**를 변형  
  - **no labels**: 입력만 나열  
  - **labels only**: 레이블만 나열

**결과**  
- 페어링 형식을 제거하면 **데모 없음과 비슷하거나 더 나쁨**  
- 반면, 입력이 OOD이고 레이블이 무작위 영어 단어여도 **입력–레이블을 짝지어 제시**하면  
  - Direct MetaICL: 분류에서 ICL 이득의 **~95% 유지**, 멀티초이스에서 **~82% 유지**  
  - Channel: ICL 이득의 **~75–87% 유지**

**해석**  
- 모델은 “입력 뒤에 레이블이 온다”는 **형식적 시그널**을 모방하여 테스트 예시를 완성  
- 형식은 ICL에서 **핵심 구조적 단서**다


### 5.4 Effect of Meta-training (MetaICL)

**관찰**  
- MetaICL은 ICL 목적을 명시적으로 학습한 모델  
- 정답 매핑의 영향이 **거의 사라지고**, **형식/분포 신호**만으로도 높은 성능

**가설**  
- **형식 신호**가 **입력–레이블 정합성**보다 활용이 쉬움  
- Direct는 **레이블 공간** 신호를, Channel은 **입력 분포** 신호를 더 잘 활용  
- 메타트레이닝은 모델이 **더 단순한 단서**를 **일관되게** 잡도록 유도

## 요약 포인트
- 정답 매핑 없이도 ICL 성능 저하가 작음 → 모델은 **형식·분포·레이블 공간**을 주로 사용  
- 인분포 입력과 명확한 레이블 공간, 그리고 **입력–레이블 페어링 형식 유지**가 핵심  
- MetaICL처럼 ICL 목적에 특화된 모델은 위 단서를 **거의 독점적으로** 활용


## 수식 참고
- Zero-shot 예측: $`\arg\max_{y \in C} P(y\mid x)`$

- $k$-shot ICL 예측 (gold): $`\arg\max_{y \in C} P\!\left(y \mid x_1,y_1,\dots,x_k,y_k,x\right)`$

- $k$-shot ICL 예측 (random): $`\arg\max_{y \in C} P\!\left(y \mid x_1,\tilde y_1,\dots,x_k,\tilde y_k,x\right)`$




<br>  
  
## 6. Discussion & Implications
### 핵심 결과 요약
- **Ground truth 매핑의 낮은 중요성**  
  - 데모의 정답 레이블을 무작위로 대체해도 성능 저하는 작음  
  - 데모 내 **정확한 입력–레이블 매핑**보다 다른 요인이 더 중요
- **성능 향상의 원동력**  
  - ICL의 이득은 주로 **입력 분포**, **레이블 공간**, **전반적 형식(format)** 에서 기인
- **형식의 결정적 역할**  
  - 적절한 **입력–레이블 페어링 형식**만 유지해도, 입력만 또는 레이블만으로 **ICL 이득의 최대 95%**까지 유지 가능
- **Meta-training의 증폭 효과**  
  - MetaICL 등 메타 트레이닝된 모델은 **정답 매핑보다 형식** 등 단순한 신호를 더욱 강하게 활용

### 모델이 테스트 시점에 ‘학습’하는가?
- **엄격한 정의(새 매핑 학습)**  
  - 결과는 LLM이 데모로 **새로운 입력–레이블 매핑**을 학습한다기보다, **사전학습(pretraining) 지식**을 호출해 활용함을 시사
- **넓은 정의(분포·레이블·형식 적응)**  
  - 데모가 제공하는 **입력 분포**, **레이블 공간**, **형식**에 모델이 적응함으로써 예측이 개선된다면 이를 **테스트 시점 학습**으로 볼 여지는 있음

### LLM의 능력(Capacity)과 함의
- LLM은 언어모델링 목표만으로도 **암묵적 입력–레이블 연관**(예: 긍정 리뷰 ↔ “positive”)을 상당 부분 내재했을 수 있음  
  - Prompt Programming for LLMs: Beyond the Few-shot Paradigm와 맥을 같이 함
- **시사점**  
  - 사전학습에서 거의 접하지 않은 **완전히 새로운 매핑**이 필요한 과제라면, ICL만으로는 충분치 않을 수 있음
  - 이 경우 **명시적 미세조정(fine-tuning)** 이나 **새 목표 함수 설계**가 필요

### Instruction-following 모델과의 연결
- 데모와 지시문(instruction)은 공통적으로 **모델의 기존 능력을 회복·정렬**시키는 역할을 할 수 있음 
- “Do Prompt-Based Models Really Understand the Meaning of Their Prompts?”는 **관련 없거나 애매한 지시**도 성능을 비슷하게 끌어올릴 수 있음을 보여, **정답적 의미**보다 **형식·구조적 신호**의 힘을 뒷받침

### 높아진 Zero-shot 기준선
- 레이블이 없는 입력에 **무작위 레이블을 페어링**한 데모만으로도 **거의 $`k`$-shot 수준**을 달성 가능 → 실제 **zero-shot 기준선**이 생각보다 높을 수 있음  
- 미래 연구는 **레이블 없는 데이터 가정 완화**를 통해 **zero-shot 성능 상향** 가능성을 탐색할 수 있음

### 한계(매우 중요)
- **작업 범위**: 실제 자연어 입력의 분류·다지선다에 집중  
  - **합성 과제**나 제약된 입력에서는 정답 레이블의 중요도가 더 클 수 있음
- **데이터셋별 변동성**: 평균적으로는 강건하나, 예컨대 **Financial PhraseBank** 등  
  - 특정 데이터셋에서는 무작위 레이블 사용 시 성능 저하가 더 큼
- **생성(Generation) 과제로의 일반화**: 출력 분포를 유지하면서 입력–출력 대응을 제거하는  
  설계가 쉽지 않음
  - 개방형 생성에서는 **정답성의 중요도**가 더 커질 가능성
- **Chain-of-Thought(CoT)**: “Text and Patterns: It Takes Two to Tango”는 **무작위 추론(rationale)** 이 CoT 성능을 떨어뜨리나, **틀린 방정식 등 일부 반사실적 구조**는 치명적이지 않을 수 있음을 보여 **형식의 상대적 중요성**을 시사

### 연구·실무 시사점
- **형식 우선**: 입력–정답 **페어링 형식**을 일관되게 유지  
- **인분포 입력 확보**: 테스트와 유사한 도메인·스타일의 입력을 데모에 넣기  
- **레이블 공간 명세**: 가능한 출력 집합을 명확히 제시(Direct계 모델에서 특히 중요)  
- **샷 효율**: $`k \ge 8`$ 이후 수익 체감 → **소수의 대표 예시**로 포맷·분포·출력공간을 명확화  
- **메타 트레이닝/튜닝**: 형식·분포 신호에 대한 **민감도**를 높여 ICL 효과를 증폭

### 향후 과제(Research Agenda)
- **저장된 매핑의 추출**: 사전학습으로 내재된 입력–레이블 매핑을 **프롬프트/검색/편집(editing)** 으로 더 안정적으로 호출하는 방법
- **새 목표 함수**: **더 넓은 작업 의미**를 학습하도록 하는 사전학습/미세조정 목표 설계  
- **하이브리드 접근**: ICL + **소량 라벨 튜닝**(PEFT/LoRA/IA3 등)으로 새로운 매핑이 필요한 태스크에 대응
- **생성 과제 일반화**: 출력 분포를 교란하지 않고 **형식·분포** 조작만으로 ICL 이득을 재현하는 프로토콜 개발




<br>  
  
## 7. Limitations
### 1. 다루는 작업 및 데이터셋의 종류
- 본 연구는 **자연어 입력 기반의 기존 NLP 벤치마크 작업**에 초점을 맞추고 있다
  - 예: 감정분석, 자연어추론(NLI), 질의응답(QA) 등.
- 그러나 *Extrapolating to Unnatural Language Processing with GPT-3’s In-context Learning: The Good, the Bad, and the Mysterious* (Rong, 2021)에서는 **합성(synthetic) 또는 비자연어적 입력**을 사용하는 작업에서는 모델이 실제(ground truth) 레이블 정보를 **더 직접적으로 활용**할 가능성이 높음을 지적
- 따라서 본 논문의 결론(“정답 매핑은 덜 중요하다”)이 **모든 작업 유형에 일반화되지는 않을 수 있음**을 유념

### 2. 데이터셋별 성능 차이
- 본 논문은 여러 데이터셋에 대한 **평균 성능(macro-level)** 분석을 중심
  - 그러나 개별 데이터셋에서는 **무작위 레이블과 정답 레이블 간의 차이**가 크게 벌어지는 경우도 존재
- 예를 들어, *Ground-truth Labels Matter: A Deeper Look into Input-Label Demonstrations* (Kim et al., 2022)에서는  
  **Financial PhraseBank** 데이터셋에서 **최대 14%의 성능 차이**를 보고
  → 즉, **데이터셋의 도메인 특성이나 언어적 구조**에 따라 실제 레이블의 중요도가 달라질 수 있다
- 또한 Kim et al. (2022)은 본 논문과 달리 **랜덤 레이블**뿐 아니라 **부정된(negated) 레이블**을 포함해 두 경우의 성능을 **보간(interpolation)** 하는 방식을 사용  
  - 반면 본 논문은 레이블을 **균일 샘플링(uniform random)** 방식으로 처리
  - 이러한 차이는 “모델이 실제 레이블 정보를 어느 정도 필요로 하는가?”라는 근본 질문에 대한 해석에 영향을 미칠 수 있다

### 3. 작업 범위의 제한 (Classification & Multi-choice 중심)
- 본 연구의 실험은 **분류(classification)** 및 **다중 선택(multi-choice)** 태스크에 한정되어 있다
  - 따라서 **생성(generation)** 과 같은 **개방형(open-set)** 작업으로의 확장은 간단하지 않다
- 생성 작업에서는 입력–출력 관계를 일부러 왜곡시키면서도 **출력 분포(output distribution)** 를 유지하는 데모 설계가 훨씬 복잡하기 때문
- 예를 들어, *Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango* (Madaan & Yazdanbakhsh, 2022)는 **Chain-of-Thought(CoT)** 프롬프팅에서  
  - **무작위 추론(random rationale)** 은 성능을 크게 떨어뜨리지만,  
  - **잘못된 방정식과 같은 반사실적(counterfactual) 추론** 은 예상만큼 성능을 저하시키지 않음을 보여주었다
  - 이는 생성형 태스크에서 데모의 **정답성보다 구조와 형식의 일관성**이 여전히 핵심적 요인일 수 있음을 시사




























