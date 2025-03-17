# 0. 화자 분할 (Speaker Diarization)이란?
- 오디오를 **화자 단위**로 구분하여 **“누가 언제 말했는가?”(Who Speaks When?)** 를 결정하는 작업

# 1. NeMo 기반 Speaker Diarization 시스템 구성
<img width="705" alt="image" src="https://github.com/user-attachments/assets/50c256eb-3fc6-46d3-b0f7-e2df7bbfac1d" />

## 1) 음성 활동 검출 (Voice Activity Detection, VAD)
- VAD 모델을 사용하여 음성이 있는 구간을 찾고 타임스탬프 생성
- 배경 소음은 무시하고, 말하는 구간만 식별
- NeMo에서는 **MarbleNet** 모델을 사용하여 VAD 수행
- **두 가지 방식의 VAD 지원** : **Oracle VAD (오라클 VAD) / System VAD (시스템 VAD)**
  
### Oracle VAD (오라클 VAD)
- 실제(ground-truth) 음성/비음성 라벨을 사용하여 VAD를 수행하는 방식
- 즉, 이미 제공된 정답 데이터(음성이 있는 부분과 없는 부분의 **타임스탬프**)를 활용 -> **NeMo의 사전 학습된 화자 임베딩 추출 모델 활용**
- 오라클 VAD를 사용하면 VAD 모델의 성능에 영향을 받지 않고 화자 분할을 평가할 수 있음
  
### System VAD (시스템 VAD)
- 실제 VAD 모델을 사용하여 음성/비음성 라벨을 생성하는 방식
- 즉, NeMo의 VAD 모델이 음성이 있는 구간을 직접 예측하여 이를 기반으로 화자 분할 수행
- **Ground-truth 데이터가 없을 때** 사용 가능

---
  
## 2) 분할 (Segmentation)
- VAD에서 찾은 음성 구간을 더 작은 조각으로 나누어 분석
  
### 균일한 분할 (Uniform Segmentation)
- VAD(음성 활동 검출) 모듈을 거친 후, 음성을 여러 개의 짧은 세그먼트(0.5~3.0초) 로 분할
- 각 세그먼트에서 화자 임베딩(Speaker Embedding)을 추출
- 각 세그먼트의 임베딩을 통해 해당 구간에서 화자의 특징(프로필, Representation) 을 얻을 수 있음
  
### 세그먼트 길이의 트레이드오프 (Trade-off: Long vs Short Segment Length)
- 세그먼트 길이를 설정할 때 **화자 표현(Representation) 품질**과 **시간 해상도(Temporal Resolution)** 사이의 트레이드오프가 존재
  - 긴 세그먼트 (2 ~ 3초 이상)
    - 화자의 특성을 더 명확하게 추출할 수 있으며 보다 일관된 화자 표현 가능
    - 2~3초 동안 한 명의 화자로 단정할 경우 오류 발생
    - 화자가 바뀌는 지점을 정확히 잡아내기 어려움
  - 짧은 세그먼트 (0.2 ~ 0.5초)
    - 시간 해상도가 높아져 보다 세밀한 분석 가능
    - 너무 짧아 신뢰할 수 있는 화자 특징 추출이 어려움 
  
### 다중 스케일 분할 (Multi-scale Segmentation)
<img width="699" alt="image" src="https://github.com/user-attachments/assets/864783e3-2fb6-4373-8829-ba9389cb2f64" />

- **NeMo** 화자 분할 파이프라인에서는 이러한 문제를 해결하기 위해 다중 스케일 접근법 사용
- 여러 개의 세그먼트 길이를 사용하여 분석하고, **각 스케일에서 얻은 결과를 융합(Fusion)하여 최종 결론 도출**
  - 여러 개의 서로 다른 세그먼트 길이를 조합하여 화자를 분석
  - 각각의 세그먼트 길이에서 얻은 affinity(유사도) 값을 결합(Fuse)하여 최종 결과 생성 
- 가장 짧은 세그먼트 길이를 갖는 스케일을 **“기본 스케일(Base Scale)”**
  -  기본 스케일은 가장 높은 스케일 인덱스(Highest Scale Index) 로 지정
  -  최종적인 의사 결정은 기본 스케일의 세그먼트 범위를 기준으로 수행

### 다중 스케일 분할에서 스케일 간 매핑(Mapping among Scales)
- 다중 스케일 분할 과정에서, 각 스케일 간 매핑을 계산
- 각 세그먼트의 중심점(middle point)을 기준으로 다른 스케일의 세그먼트와 연결(매칭)
  - 각 세그먼트의 중심점(Anchor Point) 을 계산
  - 가장 가까운 다른 스케일의 중심점과 연결하여 매핑을 수행
  - 서로 다른 스케일에서 얻은 정보를 효과적으로 결합 
- 제공된 이미지(설명 속 그림)에서, 파란색 윤곽선(blue outline)이 다중 스케일 분할과 매핑이 결정되는 방식을 보여줌

### 다중 스케일 화자 분할(Multi-scale Diarization)
- NeMo에서는 기본적으로 diar_infer_telephonic.yaml 설정 파일을 사용하며, 이 설정에는 5개의 스케일(세그먼트 길이)이 포함
- 다섯 개의 세그먼트 길이(Window Length)는 1.5초, 1.25초, 1.0초, 0.75초, 0.5초
- 50%의 오버랩(Overlap) 비율을 가지며, 모든 스케일에 동일한 가중치(Equal Weights)를 적용
  - 다중 스케일에서 Affinity Matrix(유사도 행렬)를 결합할 때 가중치가 적용
  - 그러나 Affinity Matrix는 정규화(Normalization)되므로 가중치 값의 비율만 중요
  - ex) [1,1,1,1,1] 와 [0.5,0.5,0.5,0.5,0.5] 는 동일한 결과를 생성 
  
---
  
## 3) 화자 임베딩 추출 (Speaker Embedding Extraction)
- TitaNet-L 모델을 사용하여 각 화자의 고유한 특징을 벡터(임베딩)로 변환
  
### Affinity Matrix(유사도 행렬)
- 각 세그먼트의 화자 임베딩(Speaker Embedding) 벡터 간 유사도를 나타내는 행렬
  - 코사인 유사도(Cosine Similarity) 를 사용하여 각 세그먼트 간의 유사도 계산
  - 은 화자일 가능성이 높은 세그먼트들은 높은 유사도
- 각 스케일(세그먼트 길이별)에서 개별적인 Affinity Matrix를 계산
- 그런 다음, 각 Affinity Matrix에 가중치를 적용한 후 합산(Weighted Sum)하여 최종 Affinity Matrix를 생성
  - 이때 가중치는 multiscale_weights 라는 파라미터를 사용하여 조절
- **가중합 계산 과정**
  - 각 세그먼트 길이별로 Affinity Matrix를 계산 (코사인 유사도 기반)
  - 각 Affinity Matrix에 가중치를 곱하여 가중합을 수행
  - 최종적으로 결합된 Affinity Matrix를 클러스터링 알고리즘에 입력 
  
---
  
## 4) 클러스터링 (Clustering)
- 화자 임베딩 벡터를 군집화하여 **화자의 수를 추정**
- 같은 화자가 말한 구간을 같은 그룹으로 묶음

### 클러스터링을 통한 화자 그룹 및 화자 수 예측
  - 가중합이 적용된 최종 Affinity Matrix를 클러스터링 알고리즘에 입력
    - 같은 화자인 세그먼트들을 그룹화(Grouping)
    - 화자의 수를 예측(Speaker Counting)
  - 화자 오류율(DER, Diarization Error Rate)을 줄이고, 더 정확한 화자 수를 추정
  
---
  
## 5) 신경망 기반 다이어라이저 (Neural Diarizer)
- MSDD (Multi-Scale Diarization Decoder)를 활용하여 클러스터링 결과를 기반으로 최종 화자 라벨 생성
- 여러 화자가 동시에 말하는 중첩 음성(overlap speech) 도 처리 가능
- 입력 데이터로는 특징(feature) 또는 오디오(audio input) 를 사용
- 기존의 Clustering 기반 화자 분할과 대비되는 개념  
  
| 구분 | Clustering Diarizer | Neural Diarizer |
|------|---------------------|----------------------|
| **학습 여부** | ❌ 학습 불가능 (비학습 방식) | ✅ 학습 가능 (신경망 기반) |
| **방식** | - 화자 임베딩 벡터를 사용하여 **K-means, Spectral Clustering** 등을 수행하여 화자를 구분 | - **Neural Network**를 사용하여 직접 화자 레이블을 예측 |
| **장점** | - 데이터 없이도 사용 가능 <br> - 상대적으로 빠른 수행 속도 | - **화자 중첩(Overlapping Speech) 처리 가능** <br> - **더 높은 정확도(Improved Accuracy)** <br> - **화자 임베딩 모델과 함께 학습 가능** |
| **단점** | - **화자 중첩(Overlapping Speech) 처리가 불가능** <br> - **정확도가 한계** | - 모델 학습이 필요하므로 **데이터셋이 있어야 함** <br> - 상대적으로 높은 연산 비용 |
  
### Neural Diarizer의 필요성
- Overlap-aware Diarization (화자 중첩 처리)
  - 여러 명의 화자가 동시에 말하는 상황을 고려하여 화자 분할 수행
  - 기존의 클러스터링 기반 기법은 한 시점에 한 명의 화자만 할당할 수 있지만, Neural Diarizer는 다중 화자 예측

- 다중 화자 데이터셋을 활용한 훈련 (Joint Training with Speaker Embedding Models)
  - Multi-speaker dataset을 활용하여 화자 임베딩 모델과 함께 공동 학습(Joint Training)이 가능
  - 즉, 화자 임베딩 모델과 Neural Diarizer가 동시에 최적화됨 
   
### MSDD (Multi-scale Diarization Decoder)
- Neural Diarizer의 한 종류
- Clustering 기반 화자 분할과 함께 사용되며, 먼저 클러스터링을 수행하여 예상 화자 프로필(Speaker Profile)과 예상 화자 수를 추정한 후, 이를 기반으로 Neural Diarizer가 보다 정밀한 분할을 수행
- 모델의 학습과 추론 과정(Training & Inference)이 다름
- **MSDD 모델의 학습(Training)**
  - “쌍(pairwise) 단위 모델”을 사용하여 학습
  - 두 명의 화자(Two-speaker) 단위로 모델을 학습하며, 두 명 이상의 화자가 있는 데이터에서는 무작위로 두 명을 선택하여 학습
  - 즉, 화자가 3명 이상이어도, 항상 두 명씩 조합하여 학습
- **MSDD 모델의 추론(Inference)**
  - 추론 과정에서도 모든 가능한 화자 쌍(pair)을 만들고 결과를 평균(Average)하여 최종 예측을 수행
  - 각 화자 쌍에 대해 Sigmoid 출력을 계산한 후, 최종적으로 이 값을 평균 내어 최종 화자 분할 결과를 생성
- **장점**
  - 특정한 화자 수에 종속되지 않으며, 쌍(pair) 단위 모델을 사용하여 동적으로 화자 수를 처리
  - Neural Diarizer의 장점을 활용하여 여러 화자가 동시에 말하는 구간도 처리   
  
---
  
## 6) 결과 출력 (Speaker Labels)
- 최종적으로 화자 레이블을 생성하여 오디오 내에서 누가 언제 말했는지 표시
