# Effective Text Adaptation for LLM-based ASR through Soft Prompt Fine-Tuning

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
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
