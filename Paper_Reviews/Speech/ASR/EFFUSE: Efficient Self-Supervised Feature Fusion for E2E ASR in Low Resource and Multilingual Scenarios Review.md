# EFFUSE: Efficient Self-Supervised Feature Fusion for E2E ASR in Low Resource and Multilingual Scenarios
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 문제 정의
- SSL 모델은 저자원/다국어 ASR에서 강력한 성능을 보여주지만, **단일 SSL 모델은 한계 존재**
  - 특히 영어 단일 talker 환경에서 훈련된 모델은 타 언어/환경에 한계
- 이를 보완하기 위해 여러 SSL 모델의 **feature fusion(특징 융합)** 사용됨
  - 여러 SSL 모델을 융합하면 **파라미터 수가 크게 늘어 계산 비용이 높아짐**

### 기존 방법의 한계
- 다수 SSL 모델을 **단순히 결합(fusion) → 성능은 좋아짐**
- 단점: **모델이 무거워짐 (파라미터 수 ↑, 연산량 ↑, latency ↑)**

### 제안 방법 EFFUSE
- 하나의 SSL 모델만 사용하여 **다른 SSL 모델의 feature를 “예측(predict)”**
- 여러 모델의 feature를 직접 계산하지 않고, 한 모델의 feature를 기반으로 다른 모델 feature를 재구성
  - **경량화 + 성능 유지**
- Prediction 기반 **feature fusion 구조**
- SUPERB benchmark에서 baseline SSL보다 +6.3% score 개선
- 기존 fusion 모델 대비 파라미터 약 49% 절감 (평균 317M param 감소)
