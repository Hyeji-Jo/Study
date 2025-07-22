# Improving Speech Recognition with Prompt-based Contextualized ASR and LLM-based Re-predictor

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 문제 배경
- 최근 ASR(자동 음성 인식) 시스템은 콜센터, 가상비서 등에서 널리 사용
- 하지만 발화 조건 악화, 문맥 정보 부족, 희귀 단어 인식 어려움 등의 한계 존재

### 제안 방법
- LLM(대형 언어 모델)과 prompt 메커니즘 통합
- 사전학습된 text encoder + task-specific text adapter로 문맥 정보를 효과적으로 반영
- LLM 기반 re-prediction으로 기존 n-best 결과 대신 개선된 최종 transcription 출력

### 실험 결과
- 기존 baseline ASR 대비 평균 상대적 WER 감소
  - 전통적 task: 27% ↓
  - 발화-문맥 task: 30% ↓
  - 단어-바이어싱 task: 33% ↓ 
