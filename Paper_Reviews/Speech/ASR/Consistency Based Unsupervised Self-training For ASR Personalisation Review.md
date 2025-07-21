# Consistency Based Unsupervised Self-training For ASR Personalisation
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 문제 배경
- 현대 ASR(Automatic Speech Recognition) 시스템은 대규모 사용자 데이터를 기반으로 학습되지만, 훈련 시 보지 못한 개별 사용자에게는 성능 저하 발생
- 주요 원인: domain shift
  - 사용자 발화 특성 (억양, 발음 등)
  - 환경적 음향 조건 (소음, 울림 등)

### 기존 방법
- ASR personalisation  
  - 개별 사용자 데이터로 모델을 fine-tuning 하여 성능 개선
  - 대부분 기존 방법은 **labelled user data 필요**
- 라벨 없는 상황에서의 개인화(unsupervised personalisation)는 특히 어려움
  - 데이터 수량 부족
  - 녹음 품질 문제

### 제안 방법
- pseudo-labeling + consistency 기반 training
- 라벨 없는 user data에서 안정적인 학습 가능

### 주요 성과
- 17.3% WER 감소 (training data 기준)
- 8.1% WER 감소 (held-out test data 기준)
- 기존 SOTA 방법들보다 우수한 성능

