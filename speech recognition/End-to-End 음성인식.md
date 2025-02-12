# End-to-End ASR
<img width="818" alt="image" src="https://github.com/user-attachments/assets/0092af4a-42e2-4bc2-86e5-98070c8b8542">

![image](https://github.com/user-attachments/assets/dea3f1b8-350b-4bde-addb-1f1aa3e72363)

![IMG_B3A7B651B333-1](https://github.com/user-attachments/assets/da895908-5235-493e-b270-be0780c95d9a)


# End-to-End ASR 등장 배경(핵심 아이디어, 누가 어떻게 만들었는지)
- 2010년대에 들어서면서 **심층 신경망(DNN)** 이 음성인식에 도입되기 시작했지만, 여전히 GMM-HMM 구조와 결합된 형태
- 음성 데이터를 바로 텍스트로 변환하는 하나의 신경망 모델을 사용해 여러 모듈을 통합하는 방식 -> 이 방식은 복잡한 중간 단계를 없애고, 음성에서 바로 텍스트로 변환하는 과정을 단순화


# End-to-End ASR 분류 - 논문별 카테고리
## 1) An Overview of End-to-End Automatic Speech Recognition(2019)
- **End-to-End 모델은 3가지 분류**
  - **CTC(Connectionist Temporal Classification) 기반 모델**
  - **RNN-Transducer 모델**
  - **Attention 기반 모델**

- **LVCSR(Large Vocabulary Continuous Speech Recognition)**
  - 대규모 어휘 연속 음성 인식을 의미
  - 일상적인 대화나 연설처럼 흐름이 끊기지 않는 긴 문장 내에서 많은 단어를 인식해야 함
  - 소규모 어휘를 처리하는 시스템과는 달리 어휘 크기, 연속성, 발음의 변이와 같은 도전 과제를 해결 


- 구조
  - 입력 시퀀스 (X = {x1,···, xT}): 음성 입력.
  - 인코더 (Encoder): 입력된 음성 시퀀스를 특징 시퀀스로 변환.
  - 얼라이너 (Aligner): 특징 시퀀스와 언어를 정렬.
  - 디코더 (Decoder): 최종 결과로 단어나 철자 시퀀스를 출력.


## 2) End-to-End Speech Recognition: A Survey
- "End-to-End"라는 용어는 **"과정의 모든 단계를 포함한다"**는 의미를 지니며
- E2E 모델은 하나의 신경망 구조로 통합되어 있으며, 다양한 언어나 도메인에서 더 빠른 개발이 가능하고, 메모리 사용량과 전력 소비를 줄이는 데 유리
- 사전 학습된 모듈 없이 처음부터 끝까지 통합된 학습을 수행하며, 하나의 목표에 맞춰 단일 패스로 인식하는 시스템
- **E2E의 여러 해석**
  - 이론적으로는 음향 모델과 언어 모델의 구분이 없어지지만, 실제로는 외부 언어 모델을 사용해 성능을 향상시키는 경우가 많아 완전한 E2E 모델이라 하기 어려움
  - 외부의 지식을 사용하지 않고 순수하게 데이터를 통해 처음부터 학습하는 방식으로 해석 가능하나 대규모의 사전 학습된 모델이나 추가적인 학습 전략(예: 자가 지도 학습)을 사용하는 경우가 많음
  - 전체 단어 또는 문자 단위의 직접적인 어휘 모델링을 사용하지만, 일부 언어에서는 훈련 데이터가 부족해 이러한 모델링이 어렵다는 문제점 존재
  - E2E 시스템은 모든 구성 요소를 통합하여 단일 모델로 학습하는 점에서 유리하지만, 실질적으로는 외부 자원 및 중간 과정의 도움을 받는 경우가 많아 완벽한 E2E 모델이라고 보기 어려울 때가 많음

- 음성 인식은 음성 신호를 입력(X)으로 받고 이를 단어나 문자 등의 시퀀스(C)로 변환하는 시퀀스 분류 문제
- E2E 모델은 이 입력과 출력 간의 관계, 즉 **정렬 문제(alignment problem)** 를 처리하는 방식에 따라 분류
- **E2E 모델 분류의 기초**
  - **명시적 정렬(explicit alignment)**
    - 입력과 출력 사이의 정렬을 명시적인 **잠재 변수(latent variable)** 로 모델링
    - 입력 음성과 출력 텍스트 간의 정렬을 계산
    - CTC (Connectionist Temporal Classification), RNN-T (Recurrent Neural Network Transducer)
    - 기존 과거의 HMM, Hybrid 모델도 여기에 해당
  - **암시적 정렬(implicit alignment)**
    - 입력과 출력 간의 정렬을 명시적으로 정의하지 않고 모델이 스스로 정렬을 학습
    - Attention 메커니즘을 통해 입력의 특정 부분과 출력 라벨 간의 관계를 학습
    - 명시적 정렬 모델보다 우수한 성능을 보여주나 실시간 처리에서는 사용하기 어렵다는 한계 존재
      - 스트리밍 모델을 위해 스트리밍 가능한 암시적 정렬 모델 개발됨
      - **Neural Transducer (NT)** 와 **Monotonic Attention**
      - Neural Transducer나 Monotonic Attention 같은 모델들은 **명시적 정렬 모델과 암시적 정렬 모델의 장점을 결합한 하이브리드 형태**
    - Attention-based Encoder-Decoder (AED), 또는 LAS (Listen, Attend and Spell) 모델
- **E2E 모델은 크게 Encoder 모듈과 Decoder 모듈로 나뉨**
  - **Encoder 모듈**
    - 입력 음성 시퀀스를 더 높은 차원의 표현으로 변환
    - 입력 음성 신호는 주로 D-차원의 음향 프레임으로 표현(길이가 가변적인 입력 시퀀스)
  - **Decoder 모듈**
    - Encoder의 출력을 기반으로 출력 라벨을 생성
    - 출력 라벨(문자, 단어, 서브워드 등)의 확률 분포를 계산
    - 이전에 예측된 라벨들(출력의 이전 부분)에 의존하여 다음 라벨을 예측(조건부 확률 형태)

- E2E ASR에서는 발음 사전이 필요하지 않음
  - 대신, 모델이 음소 대신 문자(character) 또는 서브워드(subword) 단위로 음성 데이터를 직접 학습하여 단어를 예측
- 언어 모델이 내부적으로 통합되어 있으며, 음향 모델과 함께 학습
  - 하지만, 여전히 외부 언어 모델(예: shallow fusion)을 추가적으로 사용가능
- 외부 언어모델 통합 방법
  - 문맥적 오류를 보정하고, 더 자연스러운 텍스트 시퀀스를 생성하는 데 도움
  - **Shallow Fusion, Deep Fusion, Cold Fusion**
  - Shallow Fusion
    - 음성 신호에서 생성한 출력에 외부 언어 모델을 결합
    - 텍스트 시퀀스를 예측할 때, 외부 언어 모델에서 제공하는 확률 분포와 결합
    - 디코딩 과정에서 언어 모델의 확률을 가중치로 적용하여, 텍스트 시퀀스의 선택을 조정
  - Deep Fusion
    -  **은닉 상태(hidden states)** 와 **언어 모델의 상태**를 결합
    -  학습 과정에서 E2E 모델과 언어 모델이 동시에 작동
  - Cold Fusion
    - E2E 모델에 **사전 통합(pre-integration)** 하여 학습 과정에서부터 두 모델을 결합
    - 언어 모델의 출력을 E2E 모델에 보조 정보로 제공
    - 훈련 중에 언어 모델과 음향 모델을 동시에 학습하게 하여, 언어 모델의 영향을 더 자연스럽게 반영
