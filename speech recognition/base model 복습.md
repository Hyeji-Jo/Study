## CNN
- 특징
    - 이미지 분석에 주로 사용됨
    - DNN과 달리 커널 합성곱을 통해 이미지의 위치정보를 살릴 수 있음
- 구조
    - convolution layer - 커널 합성곱 연산을 통해 이미지의 특징을 추출 → feature map 생성
    - pooling layer - max pooling, average pooling 2 종류로 feature map의 크기를 줄여 연산량을 줄이고, 과적합 방지
    - fully connected layer - 추출한 특징을 Flatten한 후 최종 출력 생성


## RNN
- 특징
    - 각 시점에서 입력값과 이전 상태의 값을 받아 새로운 상태 생성
    - 순환되는 시계열, 음성, 텍스트 데이터에 적합
    - 시퀀스가 길어지게되면 기울기 소실 문제 및 장기 종속성 문제 발생
- 구조
    - 입력 - 시퀀스 데이터가 한 번에 하나씩 들어감
    - recurrent layer - 이전 시점의 정보를 다음 시점으로 전달해서 순환구조 형성
    - 출력 - 시퀀스의 각 시점에 대한 예측값 출력


## LSTM
- 특징
  - RNN의 단점을 보완하여 장기 종속성 문제 해결
  - 기울기 소실 문제를 완화함으로써 긴 시퀀스에서도 학습 가능
  - 셀 상태(cell state)와 게이트 구조를 사용해 정보의 흐름을 조절
- 구조
  - 셀 상태(cell state): 중요한 정보를 오랜 기간 유지하며 필요 없을 때 버릴 수 있는 경로
  - 게이트 (gate) 구조: 정보를 선택적으로 통과시켜 정보 손실을 최소화하고, 장기 기억과 단기 기억을 분리함
  - 입력 게이트: 현재 입력이 얼마나 셀 상태에 영향을 줄지 결정
  - 포겟 게이트 (forget gate): 이전 셀 상태 중 어떤 정보를 잊을지 결정
  - 출력 게이트: 현재 셀 상태에서 어떤 정보를 출력할지 결정
  - 이전 상태와 새로운 입력이 결합: 이전 시점의 상태와 새로운 입력을 바탕으로 다음 상태로의 정보를 전달

  
## GRU
- 특징
  - LSTM보다 간단한 구조로 설계되어 연산량이 적음
  - 기울기 소실 문제를 해결하면서도 LSTM보다 효율적
  - 셀 상태 대신 업데이트된 상태로 다음 시점에 정보를 전달
  - 게이트 수가 적어 연산 효율성이 높음 (입력, 출력 게이트가 통합됨)
- 구조
  - 업데이트 게이트 (update gate): 현재 상태가 얼마나 갱신될지 결정, 이전 상태와 새 입력을 적절히 반영
  - 리셋 게이트 (reset gate): 이전 상태를 얼마나 기억할지 결정, 시퀀스가 길어져도 이전 정보를 잊지 않도록 함
  - 셀 상태가 없음: 상태 자체가 이전 정보를 계속 전달하며, 게이트들이 이를 조절


### LSTM vs GRU
- 구조: LSTM은 3개의 게이트(입력, 포겟, 출력 게이트)를 사용하지만, GRU는 2개의 게이트(업데이트, 리셋 게이트)를 사용하여 상대적으로 단순함.
- 연산량: GRU는 구조가 간단해서 연산 속도가 빠르고 메모리 효율적임.
- 성능: 특정 문제에 따라 LSTM이 더 나은 성능을 보일 수 있지만, 많은 경우 GRU도 충분히 강력함.


## Transformer
- 특징
    - RNN처럼 데이터를 순차적으로 처리하지 않고, self-attention으로 시퀀스를 병렬 처리
    - 인코더-디코더 구조를 가짐
- 구조
    - 입력 - RNN과 달리 시퀀스 순서 정보가 없으므로 + positional encoding을 통해 문장내 단어들의 상대적 위치 임베딩 제공
    - 인코더
        - self-attention - 입력 시퀀스 내의 단어들 간의 관계를 학습, 중요한 단어에 더 많은 가중치 부여
        - feedforward layer - 단어의 표현을 더욱 풍부하게 하기 위해 비선형 변환을 적용
    - 디코더
        - masked-self-attention - 이전 시점까지의 단어들을 보고 다음 단어 추론
        - encoder-decoder-attention - 인코더에서 받은 정보를 바탕으로 디코더가 문맥을 더 잘 파악할 수 있도록 인코더에서 나온 출력값과 입력 시퀀스 관계 학습 (디코더의 Q와 인코더의 K,V 사용)
            - 인코더에서 나온 출력값과 디코더의 입력 사이의 상관 관계를 학습하여, 디코더가 더 정확한 출력 데이터를 생성할 수 있게 도와줌
        - feedforward layer - 각 시점에서의 단어가 문장 내 어떤 단어와 연관성이 있는지 도출 및 최종 결과 도출
    - 출력 - 디코더에서 최종적으로 예측된 단어 출력, 확률값에 기반하여 최종 단어 생성 및 시퀀스 출력
