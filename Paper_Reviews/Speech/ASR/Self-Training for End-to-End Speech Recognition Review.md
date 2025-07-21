# Self-Training for End-to-End Speech Recognition
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### Self-Training  
- unlabeled data에 대해 pseudo-label(가짜 정답)을 생성하고, 그 데이터를 마치 labeled data처럼 다시 학습에 사용하는 semi-supervised learning 기법
- Why?
  - 대규모 speech 데이터의 transcribe는 비용과 시간이 많이 듬
  - unlabeled audio는 많지만 label 부족 → 이를 효율적으로 활용하기 위한 방법
  
### 주요 기여
- 강력한 baseline acoustic model + language model로 pseudo-label 생성
- seq2seq 모델의 특유 오류에 특화된 filtering 방식 제안
- pseudo-label 다양성을 위한 novel ensemble 방법 제안
  
### 주요 실험 결과(LibriSpeech corpus 기준)
- noisy speech: WER 33.9% 개선
- clean speech: baseline 대비 oracle과의 gap의 59.3% 회복
- 기존 방법보다 최소 93.8% 상대적 성능 우위
