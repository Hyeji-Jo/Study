# Self-Train Before You Transcribe

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 배경 및 문제
- ASR 시스템은 훈련 데이터와 테스트 데이터의 도메인이 다를 경우(domain mismatch) 성능이 크게 저하됨
- 기존 self-training 방법은 별도의 unlabeled target domain 데이터가 필요

### 제안 방법
- **Test-Time Adaptation (TTA) 방식**
  - 테스트 시점에서 테스트 recording 자체에 noisy student teacher training (NST)를 적용
  - 별도 adaptation dataset 필요 없음 → 데이터 수집 비용/노력 절감
- **Dynamic evaluation analogy**
  - 발화(utterance) 경계 넘어 context transfer → 긴 recording에서 유리
  - Local context 활용 → 모델이 현재 recording에 빠르게 적응 가능
 
### 주요 성과
- 다양한 dataset에서 최대 32.2% WER 개선
- 기존 self-training (separate adaptation data 사용)보다 더 큰 효과
