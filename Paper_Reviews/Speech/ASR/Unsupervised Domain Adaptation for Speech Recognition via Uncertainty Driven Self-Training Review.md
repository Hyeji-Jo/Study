# Unsupervised Domain Adaptation for Speech Recognition via Uncertainty Driven Self-Training
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 문제 정의
- ASR 시스템은 훈련 데이터와 실제 테스트 데이터(다른 도메인) 간 차이로 인해 성능이 저하됨
- ex) 깨끗한 뉴스 음성(WSJ)으로 학습 후, TED나 전화 통화(SWBD)에서 성능 하락
  
### 기존 Self-Training 방법의 한계
- teacher model이 unlabeled target data에 pseudo-label 부여 → student model 학습
- 한계: pseudo-label이 noisy하면 성능이 악화됨
- 기존 work들은 domain mismatch 없는 조건에서 filtering 없이 사용
  
### 제안 방법 : DUST(Dropout-based Uncertainty-driven Self-Training)
- **핵심 아이디어**
  - dropout 설정을 다르게 하여 얻은 여러 prediction들 간의 일치도를 통해 모델의 prediction 불확실성을 측정
  - **불확실성 높은 pseudo-label은 학습에서 제외**
- **장점**
  - Filtering 없는 **ST 대비 ASR 성능 크게 향상**
  - 학습 데이터셋 크기가 줄어 **training time 단축**
  
### 싦험
- Dataset
  - Source: WSJ (깨끗한 read speech)
  - Target: TED-LIUM 3 (강연), SWITCHBOARD (전화 대화)
- 결과
  - WSJ → TED: 최대 80% WER recovery
  - WSJ → SWBD: 최대 65% WER recovery 
