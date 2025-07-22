# Unsupervised Fine-Tuning Data Selection for ASR Using Self-Supervised Speech Models

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 연구 배경
- **Self-Supervised Learning (SSL)**
  - 라벨 없이 unlabeled speech data를 대량으로 활용
  - pre-training으로 강력한 음성 representation 학습
  - HuBERT, wav2vec 2.0, small labeled data로도 good performance
- **문제 상황**
  - 실제 적용 상황에서는 “전사 예산(transcription budget)“이 제한됨
  - unlabeled pool에서 어떤 subset을 transcribe해서 fine-tune할지를 결정하는 것은 중요한 과제

### 연구 목표
- 제한된 예산 하에서 좋은 ASR 성능을 위한 “데이터 선택(criteria)” 탐구
- “비지도(unsupervised) 방식”으로 unlabeled data 중 어떤 데이터를 선택할지 연구

### 주요 연구 내용
- 다양한 데이터 특성 평가
  - 화자 다양성(speaker diversity)
  - 성별 편향(gender bias)
  - 주제 다양성(topic diversity)
- 두 가지 새로운 방법 제안
  - **Pre-training loss 기반 데이터 선택**
    - HuBERT pre-training 과정의 loss (masked/unmasked cross-entropy)로 utterance의 informativeness를 측정
  - **PBPE(Perplexity of Byte Pair Encoded clustered units)**
    - HuBERT cluster label → run-length encoding → BPE tokenization → language model perplexity로 utterance score화
- Random selection과 비교
  - 얼마나 competitive한지 평가
- Correlation 분석
  - 선택된 subset의 특성과 성능(WER) 관계 분석
  - token diversity, speaker diversity, topic diversity의 중요성 실증   
