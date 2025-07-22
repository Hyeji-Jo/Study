# Robust Speech Recognition via Large-Scale Weak Supervision

## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 연구 목표
- 기존 음성 인식 시스템은 fine-tuning 필요
  - 특정 데이터셋/도메인에 최적화되지만 일반화 어려움
- Whisper는 fine-tuning 없이 다양한 상황에서 robust하게 작동하는 universal speech recognition model을 만들고자 함 

### 핵심 방법
- 인터넷에서 수집한 transcript-paired audio 68만 시간으로 대규모 weak supervision 학습
- 다국어(multilingual) + 다중 작업(multitask) 학습
  - Speech recognition
  - Speech translation
  - Voice activity detection 등

### 주요 성과
- zero-shot setting에서도 기존 fully-supervised SOTA 모델에 필적하거나 뛰어남
- 인간 수준의 정확성과 robustness에 근접
- 모델과 inference code를 오픈소스화하여 커뮤니티 활용 가능
