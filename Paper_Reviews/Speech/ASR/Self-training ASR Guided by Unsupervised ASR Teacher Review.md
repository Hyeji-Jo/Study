# Self-training ASR Guided by Unsupervised ASR Teacher
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 문제 정의
- Self-training은 음성 인식에서 성능을 개선하는 방법으로 각광받고 있음
- 첫 teacher model 학습에 라벨링된 데이터가 필요
- 소량의 라벨링 데이터로 학습된 teacher는 과적합으로 인해 noise가 섞인 pseudo-label을 생성
  - unseen data에서 성능 저하
  
### 기존 방법의 한계
- 라벨 의존성 : 초기 teacher 학습에 supervised data 필요
- teacher quality 문제 : 첫 teacher가 small labeled dataset으로 overfitting
  - noisy pseudo-targets 생성
- 비용 문제 : 기존 방법은 multi-stage training 구조 → training 비용 ↑
  
### 제안 방법
- **UASR(Unsupervised ASR) teacher 도입**
  - 라벨 없이(unpaired speech & text) 학습 가능한 teacher로 시작
  - labeled data dependency 제거
- **중간 층 phonetic supervision**
  - teacher의 phonetic 정보가 student(Data2vec2)의 intermediate layer로 distillation
  - 상위 layer pseudo-target에 phonetic + contextual 정보 강화
- 결과적으로 more ASR-friendly한 pseudo-target 생성 → WER 개선
  
### 실험 세팅
- 데이터셋 : LibriSpeech
  - pre-training: LS 960h (라벨 없이)
  - fine-tuning: LS 100h (소량 labeled)
- 평가 지표 : Word Error Rate (WER)
- 비교 baseline : Data2vec2 (SOTA self-supervised model)
  
### 주요 결과
- test-clean: 8.9% WER relative reduction (Data2vec2 대비)
- test-other: 4.3% WER relative reduction
- 라벨 없이 pre-train한 teacher로도 SOTA 수준 달성
  
### 논문 장점 및 한계
- **장점**
  - 라벨링 데이터 필요 없는 self-training 구조 → 실용적
  - 기존 Data2vec2보다 ASR 친화적 pseudo-target 생성 가능
  - 성능 우수 (SOTA 기록 달성)
- **한계**
  - teacher로 UASR을 쓰지만, GAN 기반 UASR은 training 안정성 이슈 (unstable training)
  - intermediate layer 선택의 중요성 → 실험적 hyperparameter tuning 필요
