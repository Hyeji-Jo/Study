# Speech Self-Supervised Learning Using Diffusion Model Synthetic Data
## 요약 정리
### Problem


### Contributions


### Method


### Experiments



<br>  
  
## 0. Abstract
### 배경 및 문제 상황
- **Self-Supervised Learning(SSL - 자기지도학습)**
  - 최근 음성 SSL은 labeling 없는 음성만으로 representation을 잘 학습 가능
  - ASR 등 downstream task에서 labeled data 요구량 감소
  - EX) HuBERT, Wav2Vec2.0
- **문제점**
  - 여전히 대규모 비주석 corpus 필요 (~1000시간 이상)
  - 저자원 언어(low-resource languages): 데이터 자체 부족
  - 프라이버시 문제: 데이터 수집 곤란
  - 기존 데이터 증강은 단순 noise 추가 등으로 prosody, speaker, content의 다양성을 잘 확장하지 못함 
   
### 제안 방법 : DIFFS4L(Diffusion Synthetic Speech Self-Supervised Learning)
- **아이디어**
  - 제한된 real 데이터로 diffusion model 학습
  - 다양한 variation을 갖는 synthetic speech 생성
    - 새로운 prosody(운율)
    - 새로운 speaker
    - 새로운 content (의미 없는 babble 포함)
  - Real + Synthetic data로 SSL 모델 사전학습
- **특징**
  - Diffusion model은 기존 generative model(WaveNet 등)보다 **데이터 분포를 더 잘 모델링 가능**
  - synthetic data에서 다양성(prosody, speaker, content) 제공
    - SSL 정보 효율성 증대

### 실험 결과
- English ASR Task
  - HuBERT pretrained model의 WER 6.26%p 감소
  - 26.4% relative improvement    
- 놀라운 발견
  - synthetic babble(의미 없는 음성)조차 SSL 성능 개선에 기여!
  - 기존 augment 방법보다 더 효과적
- 코드 공개 : https://github.com/Hertin/DiffS4L 
