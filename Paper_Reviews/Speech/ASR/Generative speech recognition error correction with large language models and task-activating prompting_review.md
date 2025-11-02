# Generative speech recognition error correction with large language models and task-activating prompting
## 요약 정리
### Problem (기존 ASR의 한계)
- 자동 음성 인식(ASR)은 말소리를 텍스트로 변환하지만, 여전히 단어가 빠지거나(누락), 다른 단어로 잘못 인식되거나(치환), 문맥에 어색한 전사가 나오는 문제 존재
- 기존 접근 방식
  - (1) ASR 모델 자체를 더 크게 만들거나
  - (2) ASR이 낸 상위 N개의 후보 전사(N-best hypotheses) 중에서 언어 모델이 가장 좋아 보이는 후보를 골라주는 재채점(rescoring) 모델을 붙이는 방식
- 하지만 이 방식은 한계 존재
  - 온디바이스나 실시간 환경에서는 언어 모델 크기가 수천만~수억 파라미터 수준으로 제한되므로 복잡한 문맥을 제대로 다루지 못함
  - 도메인이 바뀌면(예: 항공 질의 vs 뉴스 읽기) 기존 모델은 그 도메인에 맞게 다시 fine-tuning이 필요함
  - 기존 재채점 모델은 "후보 중에서 고르기"만 할 수 있고, 후보 자체의 오류(빠진 단어, 비문)를 적극적으로 고쳐주지는 못함
- 즉, "ASR이 낸 텍스트 결과를 사람처럼 읽고 수정해 줄 존재"가 필요하지만, 이걸 ASR 파이프라인에 제대로 붙여서 안정적으로 성능 향상을 증명한 연구는 부족

> 이 논문은 바로 이 문제를 겨냥
> **사전 학습된 대규모 언어 모델(LLM)을 이용해서, ASR 결과를 자동으로 교정하고 재채점해서 실제로 오류율(WER)을 크게 낮출 수 있는가?**


### Contributions (이 논문이 새로 한 일)
1. **LLM을 ASR의 두 번째 패스(second-pass 후처리기)로 사용하는 두 개의 파이프라인을 제안하고 분석**
   - Pipeline 1: LLM이 먼저 ASR의 N-best 후보 전사를 교정하고, 그 결과를 기존 재채점 모델(예: RescoreBERT)에 넣어 최종 선택
   - Pipeline 2: 프롬프트 설계를 통해 LLM 자체가 곧바로 "재채점자 + 오류 수정자"가 되어서 최종 전사를 생성/선택

2. **Task-Activating Prompting (TAP)** 라는 새로운 프롬프팅 전략 제안
   - TAP은 LLM에게 "너는 지금 ASR 오류 교정/재채점 역할을 수행 중이며, 기준은 이것이며, 출력 형식은 이것"이라는 태스크 인식을 단계적으로 심어줌
   - 단순히 "이 문장을 고쳐줘"보다 훨씬 구체적으로 역할을 부여하고, LLM의 작업 모드를 활성화

3. **Fine-tuning 없이 (frozen LLM 상태로) 프롬프팅만으로도 성능이 얼마나 좋아지는지**, 그리고 **추가로 LoRA 등 파라미터 효율적 미세 조정(PEFT)을 얹으면 어디까지 더 좋아지는지**를 정량적으로 비교

4. ATIS(항공 여행 질의), WSJ(뉴스 읽기) 등 서로 다른 도메인 데이터셋에서, Zero-shot / Few-shot / TAP / PEFT 등 다양한 설정으로 Word Error Rate(WER) 감소 효과를 체계적으로 보고

5. "LLM은 기존 후보 중 하나를 고르는 것"을 넘어서, **후보 자체를 더 자연스러운/정확한 전사로 바꿔서 N-best 오라클조차 이길 수 있다**는 것을 실제로 보임
   - 기존 전통적 재채점기(rescorer)가 할 수 없던 영역까지 확장

### Method (접근 방식 요약)
- 이 논문은 LLM을 이용해 ASR의 N-best 출력을 후처리(post-processing)하는 두 가지 파이프라인을 정의

#### 1. Pipeline 1 (P1): "LLM 교정 → 전통적 재채점"
- Step 1: ASR이 만든 상위 N개의 후보 전사를 LLM(사전학습된 상태, 파라미터 고정)이 읽고 문법 오류, 누락된 단어, 의미적 어색함 등을 먼저 수정
- Step 2: 이렇게 교정된 후보들을 기존 재채점 모델(예: RescoreBERT / ALBERT-base-v2 기반 LM)에 넣어 최종 1개 후보 선택
- 재채점 모델은 두 가지 손실로 학습
  - Masked Language Model (MLM) 손실 $`L_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i \mid \text{masked input})`$  
    - 문맥적으로 "이 단어가 맞는가?"를 학습
  - MWER(Minimum Word Error Rate) 손실  
    - 실제 목표인 단어 오류율(WER)을 직접 줄이도록 최적화

- 해석: P1은 기존 ASR 파이프라인을 크게 바꾸지 않고, LLM을 "전사 보정기"로 앞단에 붙여주는 실용적인 강화 버전

#### 2. Pipeline 2 (P2): "LLM이 직접 재채점자 + 생성형 수정자"
- LLM에게 프롬프트만으로 역할을 부여한다. 이때 핵심이 **Task-Activating Prompting (TAP)**
- TAP은 다음 순서를 따른다
  1. 태스크 설명 단계: "ASR이 무엇인지, 재채점이 무엇인지"를 LLM에게 스스로 설명시키며 역할을 활성화
  2. 예시 제공: N-best 후보와 정답 예시를 보여주며, 어떤 출력이 좋은 출력인지 규칙과 포맷을 학습시킴
  3. 실제 질의: 새로운 N-best 후보를 주고 "최종적으로 올바른 전사를 생성하라"라고 지시
- 결과적으로 LLM이
  - 각 후보의 의미/문맥을 비교하고
  - 오류를 직접 수정하거나 보완하고
  - 최종 답안을 생성(또는 점수까지 매김)
- 여기서는 별도의 재채점 모델이 필요 없고 LLM이 곧바로 최종 전사를 출력

#### 3. 추가 테크닉
- Few-shot / One-shot in-context prompting: 프롬프트에 실제 예시를 몇 개 넣어서 LLM이 그 스타일을 그대로 적용하게 함
- Zero-shot reasoning / Chain-of-Thought 스타일 프롬프트: "Let's think step by step" 같이 중간 추론 과정을 활성화해 더 정확한 판단을 유도
- 파라미터 효율적 미세 조정(PEFT)도 실험
  - LoRA(Low-Rank Adaptation), Residual Adapter, Prefix Tuning 등
  - 이 방식은 전체 LLM을 다시 학습시키지 않고도 도메인 적응 가능

### Experiments & Setup (실험 설정)
- **ASR 백본 (first-pass recognizer)**
  - Conformer 기반 RNN-Transducer (~760M 파라미터)
  - LibriSpeech, GigaSpeech, VoxPopuli, Libri-Light 등 대규모 음성 데이터로 학습
  - 외부 언어 모델 없이도 WER이 test-clean 2.45%, test-other 5.45% 수준
  - 역할: 입력 음성을 텍스트로 바꾼 상위 10개 후보(N-best hypotheses)를 만들어낸다
    - 이 N-best가 LLM 후처리의 입력이 된다

- **데이터셋**
  - **ATIS**: 항공편/여행 질의 음성 (도메인 특화된 대화 질의)
  - **WSJ**: Wall Street Journal 기사 낭독 음성 (일반/뉴스 도메인)
  - 서로 성격이 다른 두 도메인에서 검증함으로써, 제안한 방식이 특정 도메인에만 특화된 꼼수가 아닌지(즉, 일반화력이 있는지)를 점검

- **LLM들**
  - GPT-2 (1.5B), OpenLLaMA (13B), BLOOM (176B), InstructGPT (175B)
  - 작은 모델부터 초대형 모델까지 스케일을 바꿔가며 비교
  - InstructGPT는 인간 피드백(RLHF) 기반으로 학습되어 지시 따르기, 제로샷 수행 능력이 강함

- **평가 지표**
  - Word Error Rate (WER): ASR 품질을 비교할 때 표준적으로 사용되는 지표
  - 목표는 "기존 first-pass ASR 결과 대비 WER이 얼마나 줄었는가"를 보는 것

### Results (핵심 실험 결과)
1. **Pipeline 1 (LLM 교정 + 재채점)의 효과**
   - LLM이 먼저 후보 전사를 언어적으로 정제해주면, 그 후단의 재채점기(RescoreBERT 등)가 더 좋은 결정을 내릴 수 있게 된다
   - 결과적으로 WER이 단계적으로 감소
   - 예: 기존 시스템의 WER이 약 11.3%에서 8.7%까지 내려가는 등,LLM 보정이 "기존 파이프라인에 덧댄 추가 레이어"로서 유의미한 성능 향상
   - 결론: P1은 기존 ASR 파이프라인에 비교적 쉽게 붙일 수 있는 현실적인 개선책

2. **Pipeline 2 (TAP + LLM 직접 재채점)의 효과**
   - 프롬프트(TAP 등)만으로 동결된 InstructGPT(175B 파라미터)를 태스크 전용 재채점기로 만들었을 때
     - ATIS에서 first-pass 대비 WER 31% 감소
     - WSJ에서 first-pass 대비 WER 38% 감소
   - 특히 "Let's think step by step"과 같은 Zero-shot reasoning 스타일 프롬프트나 Few-shot 예시, TAP 기반 프롬프팅이 가장 큰 향상을 만듦
   - 반면 GPT-2(1.5B) 같은 작은 모델은 이런 생성형 오류 교정에서 충분한 개선을 내기 어려움
     - 모델 스케일 자체가 중요한 요인임을 시사.=

3. **Parameter-Efficient Fine-Tuning (PEFT)까지 결합했을 때**
   - OpenLLaMA + LoRA 등 일부만 미세 조정하는 방식으로 추가 적응을 했을 때
     - ATIS: 최대 86% 상대적 WER 감소
     - WSJ: 최대 80% 상대적 WER 감소
   - 이 수치는 "N-best oracle" 수준보다도 낮은 오류율을 달성한 경우 존재
     - N-best oracle = 사람이 N-best 후보 중 최적의 답만 골랐다고 가정할 때의 최소 오류율
     - 즉, LLM은 단순히 "후보 중 고르기"가 아니라, 기존 후보를 더 낫게 재작성/수정해서 오라클조차 넘는다
   - 이는 기존 재채점기(단순 후보 선택기)로는 불가능했던 영역

4. **SOTA와의 비교**
   - 본 논문은 이 방식이 Google USM 등 최첨단(SoTA) 대규모 ASR 시스템과 비교해도 경쟁력 있는 수준임을 보고
   - 중요한 점: 이 논문이 사용한 ASR 백엔드는 비교적 표준적인 Conformer-RNN-T인데도, LLM 후처리만으로 SOTA급 WER에 근접하거나 그 이상까지 도달 가능
   - 해석: "ASR 자체를 무식하게 키우지 않아도 된다"는 가능성을 보여줌

### Limitations (한계)
- 현재 LLM은 주로 "텍스트 후보"만 보고 판단
  - 즉, 음성 신호(발음 유사도, 잡음, 억양 등) 같은 음향 정보(acoustic evidence)를 직접적으로 활용하지는 않음
- LLM 크기가 곧 성능과 연결
  - 작은 모델(GPT-2 수준)은 TAP나 reasoning prompt만으로는 대형 모델(InstructGPT 수준)과 같은 품질을 내기 어려움
- 프롬프트 설계 의존성 큼 
  - TAP, Chain-of-Thought 등 고품질 프롬프트 구성이 성능에 직접 영향을 줌
  - 아직 "자동으로 최적 프롬프트를 만들고 유지"하는 체계는 제안 단계에 가까움
- 실제 서비스/제품 환경 관점에서는 지연 시간(latency), 추론 비용(cloud LLM 호출 비용), 개인정보(민감 음성 데이터 유출 우려) 등 추가 이슈 존재
  - 논문은 주로 WER 개선에 집중하고 있으며, 실시간성 문제 자체는 직접 해결하지 않음


### Insights & Idea (이 논문이 주는 핵심 메시지)
1. **ASR 파이프라인 재정의**  
   - "음향 모델이 초안(transcript draft)을 만들고, LLM이 그걸 언어적으로 교정/선택한다"는 2단계 구조가 매우 강력하다는 것을 보임
   - 앞으로 ASR은 단일 거대 음향 모델만의 문제가 아니라, "음향 전사 + LLM 언어 복원"의 조합 문제로 볼 수 있음

2. **프롬프트 = 역할 부여**  
   - TAP은 LLM에게 단순 편집기가 아니라 "ASR 재채점기"라는 정체성을 부여
   - 즉, 파라미터를 건드리지 않고(=학습 없이) 프롬프트만으로 역할을 명확하게 지정하면 LLM이 그 역할에 맞춰 동작할 수 있음을 실험적으로 증명

3. **N-best 오라클 한계 돌파**  
   - 기존 재채점기는 후보 중에서만 고를 수 있었지만, LLM은 후보를 더 나은 문장으로 만들어버릴 수 있음
   - 따라서 이제 ASR 후처리는 단순 "랭킹 문제"가 아니라 "생성적 오류 복원(generative error correction)" 문제로 확장

4. **다음 단계: 멀티모달 LLM ASR**  
   - 앞으로는 텍스트 기반 교정을 넘어서, 음향 신호 정보(음향 신뢰도, 발음 유사도 등)를 LLM에 직접 연결시키는 방향이 유망
   - 그렇게 되면 LLM은 "언어적으로 자연스러운가?"뿐 아니라 "실제로 그렇게 들렸는가?"까지 동시에 고려할 수 있게 됨
   - 즉, 음성-텍스트 융합형 LLM 기반 ASR로 진화할 가능성을 보여줌


<br>  
  
## 0. Abstract
### 핵심 개요
- 대규모 언어 모델(LLM)을 음성 인식(ASR) 후처리 장치로 활용하여 재채점(rescoring) 및 오류 수정(error correction)을 수행하는 방법을 탐구
- 즉, “ASR이 말소리를 글자로 바꾼 결과를 LLM이 다시 보고 문맥적으로 고쳐주는” 접근

## 연구 목표
- LLM이 **ASR 후처리(post-processing)** 단계에서 **재채점**과 **오류 수정**을 수행할 수 있는지를 검증
- 특히, **미세 조정(fine-tuning)** 없이 **프롬프트 설계(prompting)** 만으로 이러한 기능을 실현할 수 있는지 탐구

### 1. LLM의 ASR 후처리 활용
- LLM을 기존 ASR 시스템의 **두 번째 단계(second pass)** 로 사용하여 N-best 후보 문장(ASR이 예측한 상위 몇 개의 문장)을 다시 검토하고 수정
- 이 과정에서 **LLM은 문맥적 이해를 통해 문법 오류나 의미 오류를 보정**
- 이는 LLM을 **미세 조정(fine-tuning) 없이** 이러한 작업에 활용하는 데 중점

### 2. 다양한 프롬프팅 전략 평가
- **Instruction Prompting:** LLM이 “이 문장을 교정해줘”와 같은 **지시(prompt)** 를 받아 작업을 수행하도록 설계
- **Zero-shot / Few-shot In-Context Learning (ICL):** 예시 없이 또는 소수의 예시만 주어도 LLM이 맥락을 학습하는 능력 평가 
- **Task-Activating Prompting (TAP) 제안:** 인과적 지시(causal instruction) + 시연(demonstration)을 결합하여 LLM이 **작업(task)을 명시적으로 인식하고 수행하도록 활성화**하는 새로운 방식

### 3. 동결된(frozen) LLM의 성능
- ASR 시스템의 1차 출력 결과(N-best 후보)에 대해 **동결된 LLM**(즉, 파라미터를 업데이트하지 않은 상태)만으로 수행한 재채점이 **도메인 튜닝된 언어모델(domain-tuned LM)** 과 비슷하거나 더 나은 성능을 보임
- 평가 데이터셋: **ATIS (항공 여행 질의)**, **WSJ (뉴스 읽기 음성)**

### 4. 프롬프팅 + 미세 조정의 결합 효과
- LLM 프롬프트 기반 접근과 미세 조정(fine-tuning)을 병행하면 **N-best oracle 수준 이하의 오류율**(즉, 이상적인 성능)을 달성 
- 이는 LLM의 **일반화 능력(generalization power)** 이 매우 강력함을 보여줌

<br>  
  
## 1. INTRODUCTION
### 연구 배경과 문제의식
- 최근 **대규모 언어 모델(Large Language Models, LLMs)**은 단순한 텍스트 생성 도구를 넘어, **지시문(task description)**이나 소수의 **입출력 예시(input-output pairs)**만으로도 새로운 작업을 수행하는 **In-Context Learning (ICL)** 능력을 보여주고 있다 

- *ICL이란?*  
  - 모델이 사전 학습된 상태에서, 추가 학습(fine-tuning) 없이 입력된 예시만 보고 새로운 작업의 규칙을 “추론”하는 능력
  - 1,000억 개 이상의 파라미터를 가진 사전 학습된 LLM들은 비지도 학습(pre-training)만으로도 강력한 ICL 성능을 보여왔다

### ASR 시스템의 한계
- In-Context Learning이 다양한 작업에서 좋은 성능을 보였지만, ASR 작업에서의 상호작용이나 이점에 대한 연구는 제한적
- 그러나 **음성 인식(Automatic Speech Recognition, ASR)** 분야에서는 이러한 ICL 능력을 직접적으로 활용한 사례가 거의 없다
- 기존 ASR 시스템의 한계
  - “zero-shot learning” 능력을 제대로 이용하지 못함  
  - 모델의 파라미터 수를 늘려도 (예: 10B 이상) 대화체 음성이나 복잡한 도메인에서 높은 성능 달성 어려움  
  - RNN-transducer 모델에 외부 언어모델(LM)을 결합해보았지만, 온디바이스(on-device) 환경 제약으로 크기가 10M~100M 수준에 머무름
  - 즉, **LLM의 강력한 언어 이해 능력을 ASR에 직접 결합하는 실험**은 부족

### 본 연구의 접근 방식
- 이 연구는 위의 한계를 해결하기 위해 **클라우드 기반(second-pass) LLM 후처리 파이프라인**을 제안
- 이는 LLM의 ICL 능력을 활용
- “ASR이 생성한 N-best 후보 문장을 LLM이 다시 보고 문맥적으로 교정하거나 가장 자연스러운 후보를 선택(rescoring)”
- **N-best Hypothesis (N-베스트 가설)**
  - 이는 ASR 시스템의 1차 패스(first pass) 결과
  - 음성 입력에 대한 상위 N개의 가능한 텍스트 전사(transcription) 후보 목록
  - 일반적으로 이러한 가설에는 ASR 오류가 포함될 수 있음 

### (a) Pipeline 1 — LLM 기반 오류 수정 + 표준 재채점
1. **Frozen LLM Correction (고정된 LLM 오류 수정)**  
   - N-best Hypothesis를 입력으로 받음
   - **"Frozen"** 이라는 용어는 LLM이 이미 사전 학습(pre-trained)된 상태로 파라미터가 고정되어 있으며, 특정 작업에 대해 추가적인 미세 조정(fine-tuning) 없이 사용됨을 의미
   - 이 단계에서는 LLM이 입력된 N-best 가설의 문법적 오류나 삭제 오류 등을 수정하여 가설의 품질을 개선 
2. **Standard Rescoring (표준 재채점)**  
   - Frozen LLM에 의해 오류가 수정된 N-best 가설을 입력으로 받음
   - 기존의 학습된 별도의 언어 모델(LM)이나 **RescoreBERT** 같은 모델이 MWER 손실 기반으로 재채점 수행
   - 불꽃 아이콘은 이 모듈이 특정 작업에 대해 학습되거나 미세 조정되었음을 의미
   - 목표는 수정된 가설들 중에서 실제 정답에 가장 가까운 가설을 재순위 매겨(rerank) 최종 ASR 출력을 결정하는 것

> 이 방식은 기존 ASR 구조를 그대로 두면서, LLM의 언어 이해력을 추가하는 “보정 레이어” 역할
> 표준 재점수 매기기 시스템에 LLM 기반의 에러 교정(error correction) 과정을 삽입하여 ASR 첫 번째 통과(first pass) 가설(hypotheses)을 후처리

### (b) Pipeline 2 — Task-Activating Prompting (TAP)
1. **Frozen LLM Initialization (고정된 LLM 초기화)**  
   - N-best Hypothesis를 입력으로 받음
   - 여기서도 LLM은 파라미터가 고정된 "Frozen" 상태로 사용
   - **Task-Activating Prompting (TAP)** 을 통해 LLM은 특정 ASR 오류 수정 및 재채점 작업에 대한 지시를 받아 "초기화"
   - 예: “아래 후보 중에서 의미적으로 가장 자연스러운 문장을 선택하라.”  
2. **Generative Error Correction (생성적 오류 수정)**  
   - TAP으로 초기화된 LLM이 직접 N-best 가설을 기반으로 오류를 수정
   - 더 나아가 각 가설에 대한 언어 모델 점수(LM score)를 제공하여 재채점 및 최종 전사를 생성
   - 불꽃 아이콘은 LLM이 Prompting을 통해 활성화되어 작업을 수행하며, 경우에 따라 어댑터(adapter)를 이용한 미세 조정(fine-tuning)을 통해 성능을 더욱 향상시킬 수 있음을 나타냄
   - 이 방식은 LLM의 제로샷(zero-shot) 또는 퓨샷(few-shot) 학습 능력을 활용하여 미세 조정 없이도 높은 성능을 달성할 수 있음을 목표로 함

> 즉, LLM이 직접 “rescorer + corrector” 역할을 수행하는 완전한 생성적 접근
> "task-activating prompting"이라는 새로운 방법을 사용하여 고정된 LLM을 작업 지향적인 명령으로 초기화
> N-best ASR 가설 목록을 LLM의 입력으로 포맷하여 "in-context learning initialization" 또는 "in-domain fine-tuning"을 통해 음성 전사를 개선


### Task-Activating Prompting (TAP)
- 이 기술은 LLM이 특정 작업을 수행하도록 "활성화"하고 "안내"하는 데 사용되는 새로운 인컨텍스트 학습(in-context learning) 전략

- **핵심 개념**
  - LLM에게 단계적 지시를 제공 → “작업 맥락”을 인식하도록 유도  
  - 예시(demonstration) → 테스트 입력 순으로 구성된 대화형 프롬프트  
  - fine-tuning 없이도 LLM 내부의 task representation을 끌어내는 방식  

> 📈 TAP은 Pipeline 1의 LLM Correction 단계와  
> Pipeline 2의 Initialization 단계 모두에 적용 가능


### 연구 목표 요약
1. LLM의 **In-Context Learning** 능력을 ASR 후처리에 활용
2. **두 가지 파이프라인(Pipeline 1, 2)** 을 통해 오류 수정 및 재채점 효율 향상 검증
3. **Task-Activating Prompting (TAP)** 을 이용해 fine-tuning 없이도 LLM이 ASR 개선에 기여할 수 있음을 보임
4. 기존 domain-tuned 모델과의 **성능 비교 실험**을 통해 접근의 효과 검증

### 핵심 인사이트 (for future me)
- 단순히 모델 크기를 키우는 대신, **LLM의 언어적 일반화 능력**을 이용해 ASR을 후처리
- **TAP은 LLM의 “작업 이해력”을 자극하는 프롬프팅 전략**  
- 이 접근은 **fine-tuning 비용 없이도** 기존 ASR을 보완할 수 있는 “클라우드 보정 모듈” 형태의 새로운 설계 철학을 보여줌



<br>  
  
## 2. RELATED WORK

- 이 섹션에서는 본 연구와 밀접하게 관련된 세 가지 주요 연구 흐름을 다룬다 
1. **LLM 기반 후처리를 통한 가설 품질 개선**  
2. **음향 및 언어 모델링에서의 Zero-shot 학습**  
3. **정보 프롬프팅 기반 In-Context Learning (ICL)**  

### 2.1 LLM 기반 가설 후처리 개선 (*LLM-based Post-processing to Improve Hypotheses*)
- ASR 시스템에서 생성된 초기 가설(first-pass hypotheses)의 문법적 오류나 누락 오류를 수정하여 정확도를 높이는 LLM 기반 후처리 기술에 대해 설명
- 이러한 기술은 사전에 훈련된 LLM(pretrained LLM)이 가진 풍부한 문맥 정보를 활용한다는 특징 존재
- **Liao et al. [9]**: ASR 전사의 가독성을 높이는 **문장 교정(post-editing)** 기법을 제안
- **N-best T5 [11]**: **T5 인코더-디코더 구조**를 활용하여 N-best 후보 전사들을 입력받고, **차별적 학습(discriminative training)** 기반으로 **재점수(rescoring)** 수행  

### 2.2 음향 및 언어 모델링을 위한 Zero-shot 학습 (*Zero-shot Learning for Acoustic and Language Modeling*)
- 언어 모델이 **명시적인 예시 없이도(zero-shot)** 새로운 작업을 수행할 수 있다는 점은  
  - 최근 연구([3], [12], [13])에서 반복적으로 입증됨
- 하지만 이러한 zero-shot 및 few-shot 언어 모델링 기법은 종종 사전 훈련된 모델의 재배포를 필요로 하는 fine-tuning에 의존하는 경향 존재
- 즉, 기존 연구는 zero-shot 학습의 잠재력을 보여주었으나, **fine-tuning 없는 실질적 zero-shot ASR 적용**은 여전히 미해결 과제

### 2.3 정보 프롬프팅 기반 In-context Learning (*In-context Learning Based on Information Prompting*)
- **In-context Learning (ICL)** 은 LLM이 프롬프트 내부의 **맥락(Context)** 만을 이용해 새로운 작업을 학습하는 능력
  - 이 개념은 Brown et al. [1]과 Dai et al. [14]의 연구에서 공식적으로 제시
  - fine-tuning 없이도 단일 또는 소수의 프롬프트(prompt)를 제공하여 도메인에 구애받지 않는 추론(domain-agnostic inference)을 수행

- **Min et al. [2]**는 ICL 프레임워크에서 ground truth 데모(demonstrations)가 예상보다 적은 영향을 미치며, 올바른 프롬프팅 전략을 선택하면 frozen pretrained LLM 자체에서 외부 정보 이득을 추출할 수 있음을 시사
  - 즉, LLM 내부에 이미 학습된 지식만으로도 외부 fine-tuning 없이 정보 이득(information gain)을 얻을 수 있음을 입증

- 이후 연구들은 ICL의 추론 한계를 극복하기 위해 **Chain-of-Thought (CoT) prompting [15]** 을 제안  
  - CoT는 “단계적 사고(step-by-step reasoning)”를 유도하여, 복잡한 추론 문제에서도 LLM이 중간 논리를 생성

- 또한 **Kojima et al. [16]** 은 LLM이 단 하나의 자연어 프롬프트만 주어져도 **zero-shot reasoner (추론기)** 로 동작할 수 있음을 입증

### 2.4 본 연구의 차별점
- 기존 ICL 및 zero-shot 연구들은 대부분 자연어 추론(NLI), 수학적 연산, 질의응답 등의 영역에 집중 
- 반면, 본 연구는 **ASR 재점수(rescoring)** 라는 비정형적이고 복합적인 음성-텍스트 문제에 **ICL을 최초로 적용**
- 즉, “LLM의 In-context 학습 능력이 음성 인식의 오류를 교정하고, fine-tuning 없이도 ASR 성능을 실질적으로 향상시킬 수 있다” 는 점을 **실험적으로 입증한 첫 시도**



<br>  
  
## 3. METHOD
### 3.1 인컨텍스트 학습 (In-Context Learning, ICL)

### 3.1.1 정의
- ICL은 LLM이 **추가 파인튜닝 없이**, 입력 프롬프트 안에 주어진 정보(작업 설명, input-output 예시 등)를 통해 새로운 작업을 수행하는 능력
- 즉, 모델의 파라미터를 수정하지 않고, 프롬프트라는 “맥락(context)”만 보고 그 작업을 수행 
- 논문에서는 이 능력이 사전 학습(pretraining) 중 학습된 **장거리 문맥 이해 능력(long-range coherence)**에서 비롯된다고 설명

### 3.1.2 이론적 해석 (베이즈 관점)
- LLM이 생성하는 토큰 시퀀스 $`o_1, \dots, o_T`$의 확률은 다음과 같이 쓸 수 있다: $`p_{\text{prompt}}(o_1, \dots, o_T) = \int_{\theta \in \Theta} p(o_1, \dots, o_T \mid \theta) \, p(\theta) \, d\theta`$
  - $`\theta`$: 잠재적인 “작업/도메인 설정” 혹은 “프롬프트가 유도한 규칙”  
  - $`p(o_1, \dots, o_T \mid \theta)`$: 특정 작업 $`\theta`$에서 문장이 생성될 확률  
  - $`p(\theta)`$: 해당 작업이 등장할 사전 확률

- 즉, 모델은 프롬프트를 보고 “지금은 이런 작업($`\theta^*`$)을 해야 하는구나”라고 추론하고, 그에 맞는 출력 생성
- 입력–출력 관점에서 보면: $`y_{\text{test}} \sim p_{\text{prompt}}(y \mid x_{\text{test}}, \theta^*)`$
  - $`x_{\text{test}}`$: LLM에 주어진 입력 (예: ASR 후보 문장들)  
  - $`\theta^*`$: 프롬프트로 유도된 태스크 정의  
  - $`p_{\text{prompt}}(\cdot)`$: 프롬프트 조건부 출력 분포  
➡ **결론:** ICL은 프롬프트를 통해 $`\theta^*`$라는 작업 정의를 “활성화”하고, 그 작업 방식대로 출력을 생성하는 과정

### 3.1.3 인컨텍스트 학습 방식 (Prompting Variants)
- 논문에서는 ASR 후처리에 적용 가능한 여러 프롬프팅 전략을 Figure 2로 분류

#### (a) Zero-shot prompting
- 모델에게 단순히 “이 작업을 수행하라”는 지시만 제공
> 예시: “다음 후보 문장 중 가장 자연스러운 전사를 선택하라.”
- 예시 없이 즉시 수행
- 도메인 정보 반영이 제한적

#### (b) Zero-shot domain-hint prompting
- 프롬프트에 **도메인 힌트(domain hint)** 를 추가
> 예시: “이 대화는 항공권 예약 상황이다. 올바른 전사를 선택하라.”
- 도메인 정보가 포함된 경우 모델은 해당 분야의 어휘(공항 코드, 도시명 등)를 더 적절히 선택 가능
- 수식으로는 다음처럼 표현:$`r_y(y_{\text{test}}) \sim p_{\text{prompt}}(r_y(y) \mid r_x(x_{\text{test}}), \theta^*)`$
- 여기서 $`r_x`$, $`r_y`$는 입력·출력에 도메인 맥락을 덧붙이는 래핑 함수(wrapper function)

#### (c) Zero-shot reasoning (Chain-of-Thought style)
- 모델에 “step-by-step으로 생각해보자” 같은 문장을 추가하여 **중간 추론 과정**을 먼저 출력하게 만듦
> 예시: “각 후보 문장을 분석하고, 가장 문맥상 자연스러운 문장을 단계적으로 선택하라.”
- 이 방식은 Chain-of-Thought(CoT) prompting [15] 및 “LLMs are Zero-shot Reasoners” [16] 접근과 유사

#### (d) Few-shot / One-shot in-context learning
- 모델에 실제 예시(input-output pair)를 제공

> 예시:  
> - “다음 10개의 후보는 ASR 결과이다.”  
> - “정답은 이것이다.”  
> - “이제 새로운 입력에 대해 같은 작업을 수행하라.”

- **One-shot**: 예시 1개  
- **Few-shot**: 예시 여러 개  
- 데이터 누출 방지를 위해 학습·테스트 데이터는 분리

### 3.1.4 H2T (Hypotheses-to-Transcription) 학습 목적 함수
- 논문은 ASR의 N-best 후보 리스트를 실제 전사로 직접 매핑하도록 하는 목적 함수를 제안 
- 이를 **H2T (Hypotheses-to-Transcription) loss**라고 부름
  - $`L_{\text{H2T}} = \sum_{i=1}^{N} \Big(- \log P(y^* \mid x_i, \Theta)+ \lambda \cdot \text{MSE}\big(s_i,\; P(y^* \mid x_i, \Theta)\big)\Big)`$
  - $`x_i`$: ASR의 i번째 후보 전사  
  - $`y^*`$: 실제 정답 전사  
  - $`\Theta`$: 모델 파라미터  
  - $`P(y^* \mid x_i, \Theta)`$: 후보가 정답일 확률  
  - $`s_i`$: 기존 ASR 시스템의 점수 (posterior score 등)  
  - $`\lambda`$: 가중치 계수 (논문에서는 $`\lambda = 0.01`$ 사용)

- 이 손실은 다음 두 목표를 함께 만족시키려 한다
  1. 정답 전사를 높은 확률로 예측 ($`-\log P(y^* \mid x_i, \Theta)`$)  
  2. LLM 점수와 ASR 점수가 크게 어긋나지 않게 조정 (MSE 항)

### 3.2 Task-Activating Prompting (TAP)
- TAP은 LLM이 수행할 태스크를 명시적으로 인식하도록 만드는 새로운 프롬프팅 프레임워크
- 단순히 한 문장을 주는 대신, **질문–이해–예시–실행**의 순서로 컨텍스트를 확장

#### 단계별 구성 (Figure 3)
1. **Task 이해 단계**  
   - “ASR이 무엇인지 알고 있나요?”  
   - “ASR 재점수(rescoring)가 무엇인지 설명해주세요.” → 모델이 작업 개념을 스스로 정의하도록 유도

2. **예시 제공 단계 (Demonstration)**  
   - 실제 N-best 후보와 정답을 보여줌

3. **실제 질의 단계 (Query)**  
   - 새로운 N-best 후보를 주고 정답 전사를 생성하도록 요청

- 이 과정을 통해 모델은 “단순 문장 교정기”가 아니라 “ASR 결과를 재채점하고 최적 후보를 선택하는 평가자”로 동작
- TAP은 미세 조정 없이도 LLM의 작업 적응력을 크게 향상시킴

### 3.3 파라미터 효율적 미세 조정 (Parameter-Efficient Fine-Tuning)
- ICL과 TAP만으로도 효과가 있지만, 논문은 추가로 **Parameter-efficient fine-tuning (PEFT)** 기법 고려
  - **LoRA (Low-Rank Adaptation)**: 작은 저랭크 행렬만 학습해 파라미터 효율적으로 적응
  - **Residual Adapters**: 레이어 출력에 작은 보정 모듈을 추가
  - **Prefix Tuning**: 학습 가능한 벡터를 입력 앞에 붙여 프롬프트 효과를 강화

- 이 기법들은 전체 모델을 다시 학습하지 않고도 도메인 적응(domain adaptation)을 가능하게 함

### 3.4 섹션 요약
1. LLM은 파인튜닝 없이 프롬프트만으로 작업을 수행할 수 있다 (ICL)
2. 다양한 프롬프팅 기법(zero-shot, domain-hint, reasoning, few-shot)이 ASR 후처리에 사용된다
3. H2T loss는 언어적 정확성과 ASR 신뢰도 점수를 함께 고려한다
4. TAP은 LLM의 “태스크 활성화”를 단계적으로 유도하는 프로토콜이다
5. 필요 시 LoRA 등 PEFT로 도메인 적응을 수행할 수 있다




<br>  
  
## 4. EXPERIMENTS AND RESULTS
- 이 섹션에서는 **LLM 기반 ASR 후처리 시스템**의 성능을 평가하기 위한 실험 설정, 모델 구성, 데이터셋, 그리고 주요 결과를 다룸
- 평가는 **단어 오류율 (Word Error Rate, WER)** 을 중심으로 진행되며, Pipeline 1 (LLM 보정 + 재채점)과 Pipeline 2 (LLM 직접 재채점) 두 접근법을 비교

### 4.1 실험 개요
- **목표:** ASR 2차 재채점 시스템의 최종 WER 개선 평가
- **절차:**  
  1. 사전 학습된 ASR 모델을 사용해 오디오를 디코딩하고 상위 10개의 가설(N-best)을 생성  
  2. LLM 기반 오류 수정(P1)과 LLM 직접 재채점(P2)을 수행  
  3. 각 접근 방식의 WER을 비교 (Figure 1 참고)

### 4.2 First-pass ASR Model
#### 모델 구성
- **Architecture:** Conformer 기반 RNN-Transducer [13, 19]  
- **파라미터 수:** 760M  
- **학습 데이터:**
  - LibriSpeech [8] — 960시간  
  - GigaSpeech [27] — 10,000시간  
  - VoxPopuli [25] — 24,000시간  
  - Libri-Light [23, 9] — wav2vec2 사전 학습 데이터  

#### 역할 및 성능
- 역할: 오디오 입력을 디코딩하여 **N-best hypotheses (상위 10개 전사 후보)** 생성  
- 성능 (외부 LM 미사용)
  - Test-clean: 2.45% WER  
  - Test-other: 5.45% WER

### 4.3 Pipeline 1 (P1) — LLM Error Correction + Rescoring
- **목적**
  - LLM을 사용해 1차 ASR 결과의 오류를 수정한 뒤, RescoreBERT와 같은 표준 재채점 모델을 결합하여 최종 WER을 낮추는 것이 목표

#### 훈련 절차
##### (1) MLM (Masked Language Model) 손실 기반 적응
- 입력 문장에서 일부 단어를 마스킹하고, 모델이 이를 예측하도록 훈련
- 수식:$`L_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(w_i \mid \text{masked input})`$
  - $`w_i`$: 마스킹된 토큰  
  - 목적: 문맥적 관계 학습  
##### (2) MWER (Minimum Word Error Rate) 훈련
- 모델의 실제 목적(WER 최소화)을 직접 반영하는 훈련 방식 
- 정답 전사와 비교하여 단어 오류율을 최소화하는 방향으로 최적화

#### 사용 모델
- **Rescoring LM:** ALBERT-base-v2 (4.8M 파라미터)  
- **절차**  
  1. LLM이 N-best 후보의 오류 수정  
  2. 수정된 후보를 입력으로 RescoreBERT 훈련  
  3. 최종 재채점 수행  

- P1은 LLM의 오류 교정 능력과 기존 재채점 시스템을 결합하여 ASR의 후처리 성능을 향상


### 4.4 실험에 사용된 LLMs
- 다양한 규모와 훈련 방식의 LLM을 사용하여 ASR 오류 수정과 재채점 능력의 차이를 비교

| 모델 | 파라미터 수 | 특성 | 훈련 데이터 |
|------|--------------|------|--------------|
| **GPT-2** | 1.5B | Causal LM, In-context prompting의 기본 LLM | Wikipedia, Common Crawl |
| **OpenLLaMA** | 13B | LLaMA 구조의 오픈소스 재현, Decoder-only | RedPajama |
| **BLOOM** | 176B | 46개 언어 + 13개 프로그래밍 언어, 오픈 대규모 LM | 1.61TB HuggingFace 데이터 |
| **InstructGPT** | 175B | GPT-3 기반, RLHF로 훈련, 인간 피드백 반영 | 비공개 OpenAI 데이터 |

> **비교 목적**  
> - 모델 크기(1B ~ 176B)가 성능에 미치는 영향  
> - RLHF (Human Feedback) 훈련 효과 분석  
> - Fine-tuning 없이도 In-context Learning 성능 확인


### 4.5 실험 데이터셋
| 데이터셋 | 설명 | 규모 | 목적 |
|-----------|------|------|------|
| **ATIS** (Airline Travel Information System) | 항공 여행 정보 질의 음성 | 4978 train / 893 test 발화 | 도메인 특화 질의 응답 평가 |
| **WSJ** (Wall Street Journal) | 뉴스 읽기 음성 및 텍스트 | train-si284 / 93dev | 일반 도메인 인식 성능 검증 |

> 두 데이터셋은 LLM 기반 ASR 후처리의 **일반화 능력**을 평가하는 데 사용됨

### 4.6 Pipeline 1 Results
#### 단계별 처리 과정
1. **Stage 0**  
   - *LLM Correction (NC)* — N-best 가설의 오류 수정  
2. **Stage 1**  
   - *Fine-tuned RescoreBERT (MLM loss)* — 수정된 후보에 기반하여 RescoreBERT 미세 조정  
3. **Stage 2**  
   - *MWER training* — 단어 오류율 직접 최소화  

#### 주요 결과 (Table 1, Figure 4)
- **LLM Correction 효과**  
  - LLM을 통한 전처리 후 N-best oracle WER이 개선됨
  - 특히 InstructGPT가 가장 큰 성능 향상

- **기존 재채점과의 시너지:**  
  - 기본 RescoreBERT: 11.3% → 8.7% WER  
  - + LLM Correction: 추가적인 성능 향상  
  - LLM이 기존 파이프라인의 한계를 넘는 **추가적 부스트 효과** 제공

> 💡 **결론**  
> Pipeline 1은 LLM의 오류 수정 능력과 기존 재채점 모델을 결합해 표준 ASR 시스템 대비 더 낮은 오류율 달성


### 4.7 Pipeline 2 Results — Generative Error Correction with TAP
#### 개요
- Pipeline 2는 **Task-Activating Prompting (TAP)** 을 사용하여 LLM이 직접 N-best 가설을 재채점하고 오류를 수정하도록 함

#### (1) Zero-shot Learning 결과
- **LLM 규모의 영향**  
  - GPT-2 (1.5B): 4-gram baseline보다 낮은 성능  
  - BLOOM (176B), InstructGPT (175B): 탁월한 성능  
- **InstructGPT**  
  - 19.7% 상대적 WER 감소 (Table 3)  
  - Zero-shot reasoning 프롬프트 (“Let’s think step by step”) 사용 시 최고 성능  
  - 추론 유도 프롬프트가 LLM의 오류 교정 품질을 향상시킴

#### (2) Few-shot Learning 결과
- **데모 샘플 수 증가 효과**  
  - InstructGPT에서 샘플 수를 1 → 12개로 늘리자 성능 지속 향상  
- **Prompting 방식 비교**  
  - One-by-one prompting: 각 발화마다 모델 초기화  
  - In-context prompting: 대화 기록 누적  
  → In-context 방식이 훨씬 우수 (Figure 5)

#### (3) In-domain Fine-tuning 결과
- **LoRA / Residual Adapters**  
  - 전체 미세 조정보다 더 효율적이고 높은 성능 (Table 4)
- **OpenLLaMA + LoRA 조합**  
  - ATIS: 86%  
  - WSJ: 80% 상대적 WER 감소  
  - N-best oracle 오류율보다 낮은 수준 달성

> 📘 **해석**  
> LLM은 사전 학습된 지식을 바탕으로 ASR 출력 오류를 직접 교정할 수 있으며, fine-tuning 없이도 탁월한 성능을 보임


### 4.8 결론 요약
| 구분 | 핵심 발견 |
|------|------------|
| **P1 (LLM Correction + Rescoring)** | 기존 재채점 시스템과 결합 시 WER 대폭 감소 |
| **P2 (Generative TAP)** | LLM이 직접 오류 수정 및 점수 산출 수행 가능 |
| **프롬프트 전략** | Zero-shot reasoning, In-context prompting이 가장 효과적 |
| **미세 조정 방식** | LoRA 등 PEFT 기법이 full fine-tuning보다 효율적 |



<br>  
  
## 5. CONCLUSIONS
- 이 섹션은 대규모 언어 모델(LLM)의 **In-Context Learning (ICL)** 능력을 활용해 자동 음성 인식(ASR) 시스템의 출력 품질을 향상시키는 본 연구의 핵심 결과를 요약

### 5.1 문제 설정과 접근 요약
- 본 연구의 목표는 **ASR 1차 패스(first-pass) 출력의 N-best 후보**를 후처리 단계에서 더 정확하게 만들기 위한 새로운 방법을 제안하고 검증하는 것
- 이를 위해, 사전 학습된 LLM의 ICL 능력을 활용하여 **추가적인 full fine-tuning 없이**도 ASR 오류를 수정하고 재채점(rescoring)할 수 있는지를 분석
- 연구는 두 가지 파이프라인을 제시

  1. **Pipeline 1 (P1)**  
     - 사전 학습된 LLM이 먼저 N-best 후보 전사 내 오류를 수정 (오타, 누락된 단어, 문맥적으로 부자연스러운 표현 등)
     - 이후, 기존 ASR 재채점기(예: RescoreBERT류의 언어 모델)를 사용해 최종 후보를 선택
     - 즉, "LLM 교정 → 전통적 재채점" 구조
  
  2. **Pipeline 2 (P2)**  
     - 프롬프트 설계를 통해 **동결된(frozen) LLM 자체가 직접 재채점자(rescorer)로 동작**하도록 만듦
     - 이 방식은 LLM이 N-best 후보를 보고 자체적으로 더 나은 전사를 생성/선택하고 점수화까지 수행하도록 유도 
     - 핵심은 LLM이 단순 교정기(post-editor)가 아니라 최종 결정기(decider) 역할까지 맡는다는 점


### 5.2 Pipeline 2의 성능
- Pipeline 2는 특히 LLM의 인컨텍스트 능력을 극대화하도록 설계된 프롬프팅 기법들을 사용
  - **Chain-of-Thought (CoT)** 스타일의 추론 유도 프롬프트  
  - **예시 기반 프롬프트 (example prompting / few-shot prompting)**  
  - 본 논문에서 제안한 **Task-Activating Prompting (TAP)**

- 이러한 프롬프트 전략을 적용한 결과
- **동결된 InstructGPT (175B 파라미터)** 만으로도  
  - ATIS 데이터셋에서 1차 패스 ASR 대비 **31% WER 감소**  
  - WSJ 데이터셋에서 **38% WER 감소**  

- 이 성능은 **기존에 fine-tuning된 GPT-2 기반 LM보다도 우수**
  - 즉, 단순히 모델을 미세 조정한 작은 LM보다, 파라미터를 건드리지 않은 대규모 LLM + 올바른 프롬프팅이 더 효과적일 수 있음

- **핵심 의미:** 프롬프트만으로도 (즉, 파라미터 업데이트 없이도) 대규모 LLM은 ASR의 N-best 후보를 상당히 안정적으로 교정 및 재채점 가능


### 5.3 Fine-tuning의 추가 효과
- 연구는 프롬프팅만으로 끝나지 않고, **파라미터 효율적 미세 조정(parameter-efficient fine-tuning)** 이 더 높은 성능 향상을 줄 수 있다는 점도 보였다.

- 특히 **OpenLLaMA + LoRA (Low-Rank Adaptation)** 조합에서
  - ATIS에서 **86% 상대적 WER 감소**
  - WSJ에서 **80% 상대적 WER 감소**

- 이 결과는 **N-best oracle 오류율보다 더 낮은 수준**까지 도달
  - N-best oracle은 “사람이 N-best 후보 중 최적 해답을 직접 골랐을 때 가능한 최소 오류율”이므로, 그보다 낮다는 건 LLM이 원래 N-best 후보에 없던 더 나은 전사를 생성하거나 강하게 수정하고 있음을 의미

- 즉, LLM은 단순히 "후보 중에 누가 제일 나아?"만 하는 게 아니라, "후보를 더 나은 문장으로 직접 재구성"할 수 있다는 걸 보여줌
- 즉, **생성적(error-corrective) 역할**을 가짐

### 5.4 비교 대상 대비 의의
- 본 연구의 방식은 거대 전문 ASR 시스템 (예: Google USM 기반의 state-of-the-art 모델)과 비교해도 경쟁력 있는 결과를 보였다고 보고
- 특히, 이 논문에서 사용한 ASR 백엔드는 비교적 “표준적인” Conformer-RNN-T 계열 모델인데도, LLM 기반 후처리를 결합함으로써 SOTA 수준 또는 그 이상의 WER을 달성한 사례가 관찰

- **중요한 포인트**  
  - 이건 “ASR 모델 자체를 초거대화하지 않고도” (즉, 음향/인식 모델을 계속 키우지 않고도)  
  - LLM을 후처리 스택에 붙여서 품질을 극적으로 끌어올릴 수 있음을 의미
  - 실용적인 관점에서 비용-효율성이 높음

### 5.5 한계와 향후 방향
- 논문은 다음과 같은 후속 연구 방향을 제시
1. **LLM에 음향 정보(파형/스펙트럼 레벨 정보)를 더 직접적으로 통합하기**  
   - 현재 접근은 기본적으로 텍스트 기반 후처리에 가까움
   - 향후에는 음향 표현(acoustic representations) 자체를 LLM에 주어 (예: 음향 신뢰도나 발음 유사도 같은 정보)  
     - 더 정교한 오류 수정과 재채점이 가능하도록 할 수 있다고 제안

2. **TAP 스타일의 프롬프팅 구조를 더 일반화**  
   - TAP은 LLM에게 역할을 "단계적으로 활성화"시키는 프로토콜
   - 이 구조를 다른 도메인(예: 의료 음성 기록, 고객센터 콜 로그 등)에서도 재사용 가능한지 탐구할 가치 존재























