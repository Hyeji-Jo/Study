# Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?
## 요약 정리
### Problem


### Contributions


### Method


### Experiments & Setup


### Results


### Limitations


### Insights & Idea



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
- 데모의 라벨을 무작위(random)로 바꿔도 성능 저하가 거의 없으며,  
  이는 **GPT-3를 포함한 12개 이상의 모델**에서 일관되게 관찰됨
- 이 결과는 모델이 데모 내의 **입력–레이블 매핑(input–label mapping)** 자체를 활용하지 않고,  
  **다른 요인들에 의존하여 태스크를 수행**하고 있음을 보여줌
















