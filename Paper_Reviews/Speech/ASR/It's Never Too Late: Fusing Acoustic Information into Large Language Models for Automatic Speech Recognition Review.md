# It's Never Too Late: Fusing Acoustic Information into Large Language Models for Automatic Speech Recognition

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
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
