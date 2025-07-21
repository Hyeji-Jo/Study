# Domain Adaptive Self-supervised Training of Automatic Speech Recognition
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 문제 정의
- ASR 시스템의 도메인 적응
  - 다른 억양, 환경에서 성능 저하
  - 라벨 없는(target domain의 unlabeled) 데이터를 활용하여 ASR 모델 성능을 개선할 수 있는 방법 필요
  
### 기존 방법의 한계
- Self-supervised Learning (SSL) ASR은 많은 unlabeled 데이터로 representation을 학습
- 도메인 mismatch가 큰 경우 target domain에 대한 성능 저하 발생
- 단순히 target domain 데이터를 pre-training에 추가하는 것만으로는 한계

### 제안 방법
- **자기지도학습(SSL) + 반지도학습(semi-supervised learning)** 의 조합
- Target domain unlabeled data를 SSL Pre-training에 활용
  - or Fine-tuning에 활용 (semi-supervised pseudo-labeling)
  - 또는 두 단계에 모두 사용

### 실험 세팅
- 도메인 = 영어 억양(Accents)
  - 미국식 억양 (in-domain)
  - 비영어권 화자의 영어, 영국 억양, 인도 억양 (target domains)
- 평가 metric: Word Error Rate (WER)
- baseline: SSL로 학습된 wav2vec 2.0 기반 ASR 모델

### 주요 결과
- 단일 도메인 실험
  - WER 2.7% ~ 41.8% 상대적 감소 (도메인 mismatch 정도에 따라 다름)
- 다중 도메인 실험
  - 평균 8% WER 감소  
