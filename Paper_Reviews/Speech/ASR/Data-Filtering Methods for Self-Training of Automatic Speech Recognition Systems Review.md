# Data-Filtering Methods for Self-Training of Automatic Speech Recognition Systems

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### Self-Training이란?
- labeled data가 부족할 때, unlabeled speech를 자동 전사 → 학습 데이터에 추가
- 과정
  - 초기 ASR 시스템 준비 (labeled data로 학습)
  - unlabeled speech 데이터 입력 → 자동 전사 생성
  - 이 데이터를 labeled corpus에 추가하여 retraining 
- 문제점
  - 자동 전사가 오류를 포함 → 잘못된 라벨 데이터로 학습될 위험
  - 정확한 전사만 골라서 사용해야 함 → “Data Filtering” 필요
   
### 제안 방법 및 비교
- **Confidence score 기반 filtering**
  - ASR 시스템이 각 단어에 대해 confidence score 제공 (posterior probability)
  - 특정 threshold(예: 0.95) 이상만 선택
  - 간단하고 self-contained
  - 단점: 이미 “잘 알아듣는” 데이터만 선택 → 새로운 정보가 부족
- **Multiple hypotheses 기반 filtering**
  - 서로 다른 ASR 시스템 2개 사용 → 각 system이 동일한 transcriptions 예측한 부분만 선택
  - 서로 다른 오류 패턴을 이용 → “agreement” = 신뢰 가능
  - 다양한 source 기반 보강
  - 추가 ASR 시스템 필요 → 자원 요구
- **Approximate transcript 기반 filtering**
  - 뉴스 headline, 자막, script 등 “대략적인 텍스트”와 alignment
  - 공통 부분만 선택
  - 소량이지만 다양하고 noise-robust 데이터 확보
  - noisy 환경에서 효과적
  - Approximate text 필요 (모든 도메인에 적용 어려움)
  
### 성능 비교
- 기존 baseline 대비 최대 25% 상대적 WER 개선
- 세 방법 중 approximate transcript 기반 방법이 가장 효과적
  - 데이터 양은 적지만 품질 좋음
  - 특히 degraded/noisy speech에서 성능 향상 뚜렷
- 여러 방법을 합쳐도 approx 하나보다 좋지 않음
  - 이유: confi, multi는 “seed ASR가 이미 잘 하는 부분”에 bias
 
### 논문의 의의 및 한계
- **의의**
  - 다양한 filtering 방법 직접 비교 → 실질적 적용 지침 제공
  - Approximate transcript 활용법 실증
  - Romanian ASR에서 state-of-the-art 달성
- **한계**
  - Approximate transcript가 있어야 가능 (일반화 제한)
  - Romanian 데이터에만 실험됨 → 다른 언어에서 바로 적용 보장 X 
