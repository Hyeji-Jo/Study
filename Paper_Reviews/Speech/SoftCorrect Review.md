# SoftCorrect: Error correction with soft detection for automatic speech recognition

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



## 요약 정리
### Problem
- ASR 시스템의 낮은 WER에도 불구하고 남아있는 오류 단어만을 선택적으로 수정해야 함
- 기존 방식들의 문제점
  - 암묵적 오류 탐지: 정확한 오류 위치 신호 부족
	- 명시적 오류 탐지: 오류 탐지 실패 시 오히려 잘못된 수정 발생 가능
- 따라서 오류 단어를 정확하고 유연하게 탐지하고, 해당 단어에만 집중하여 수정하는 방식이 필요

### Contributions
- Soft Error Detection : 언어 모델 기반 확률로 단어 오류 여부를 판단 (GT token 도입)
- Constrained CTC Loss : 탐지된 오류 단어만 중복하여 디코딩 대상에 포함시킴
- 병렬 생성 기반으로 속도 유지
- 다수 후보(beam search results)를 활용한 투표 기반 후보 선택
- AISHELL-1/Aidatatang 데이터셋 기준 최고 CER 개선

### Method
- 전체 구조
  - Encoder (탐지기): soft하게 오류 단어 판단 (확률값으로)
	- Decoder (수정기): 오류 단어만 복제 후 CTC로 수정
- 핵심 모듈
  - Anti-Copy Language Model Loss : 복사 학습을 피하기 위해 GT token을 추가한 변형된 cross-entropy loss
  - Constrained CTC Loss : 오류 단어만 3회 복제 → 나머지는 그대로 디코딩 → 빠르고 정밀한 수정
- 추가 기법
  - ASR beam search 후보 다수 사용
  - ASR 모델의 음향 확률과 encoder 확률을 융합해 오류 판단

### Experiments
<img width="585" alt="image" src="https://github.com/user-attachments/assets/9f9daca6-2ff4-41bf-bc11-dbcb29d356e8" />

- 명시적 탐지 기반보다 더 높은 정확도
- AR 모델보다 빠르고 정확
- Aidatatang처럼 탐지가 어려운 데이터셋에서도 효과적
