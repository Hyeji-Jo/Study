# 개요
- seq2seq 모델은 인코더에서 입력 시퀀스를 context vector라는 하나의 고정된 크기의 벡터 표현으로 압축하고, 디코더는 이 context vector를 통해 출력 시퀀스를 만들어냄  
- RNN 기반의 seq2seq 모델의 문제점  
  - 하나의 고정된 크기의 벡터에 모든 정보를 압축하려고 하니 정보 손실 발생  
  - RNN의 고질적인 문제인 **기울기 소실(vanishing gradient)** 문제 존재  
  - 기계 번역 분야에서 입력 문장이 길면 번역 품질이 떨어지는 현상 발생  
- 이를 위한 대안으로 Attention 기법 탄생  

## 아이디어  
- 디코더에서 출력 단어를 예측하는 매 시점마다, 인코더에서 전체 입력 문장을 다시 한 번 참고한다는 점  
- 단, 해당 시점에서 예측해야할 단어와 **연관이 있는 입력 단어 부분을 좀 더 집중(attention)** 해서 보겠다는 의미  

## 어텐션 함수(Attention Function)  
- **Attention(Q, K, V) = Attention Value**  
  - 주어진 '**쿼리(Query)'에 대해서 모든 '키(Key)'와의 유사도를 각각 구함**  
  - 이 **유사도를** 키와 맵핑되어있는 각각의 **'값(Value)'에 반영**  
  - 그리고 유사도가 반영된 **'값(Value)'을 모두 더해서 리턴**  
  - Q = Query : t 시점의 디코더 셀에서의 은닉 상태  
  - K = Keys : 모든 시점의 인코더 셀의 은닉 상태들  
  - V = Values : 모든 시점의 인코더 셀의 은닉 상태들  

# 닷-프로덕트 어텐션(Dot-Product Attention)  
<img width="591" alt="image" src="https://github.com/user-attachments/assets/a06aba2f-6197-4b08-8145-cca1df965615">  
  
- 디코더의 세번째 LSTM 셀에서 출력 단어를 예측할 때, 어텐션 메커니즘을 사용하는 모습  
  - 디코더의 세번째 LSTM 셀은 출력 단어를 예측하기 위해서 인코더의 모든 입력 단어들의 정보를 다시 한번 참고  
- 인코더에 softmax 함수 존재  
  - softmax 함수를 통해 나온 결과값은 각각이 출력 단어를 예측할 때 얼마나 도움이 되는지의 정도를 수치화한 값  
  - 위의 그림의 빨간 직사각형의 크기  
  - 각 입력 단어가 디코더에 도움이 되는 정도를 수치화해 이를 하나의 정보로 담아서 디코더로 전송 -> 초록색 삼각형  

## 1) 어텐션 스코어(Attention Score) 계산  
<img width="628" alt="image" src="https://github.com/user-attachments/assets/19bde94b-3c74-49c3-860e-5532237bd872">    
  
- 조건  
  - 인코더의 시점(time step)을 각각 1, 2, ... N  
  - 인코더의 은닉 상태(hidden state)를 각각 h_1, h_2, ... , h_n  
  - 디코더의 현재 시점(time step) t에서의 디코더의 은닉 상태(hidden state)를 s<sub>t</sub>  
  - 인코더의 은닉 상태와 디코더의 은닉 상태의 차원이 같다고 가정  
  
- 시점 t에서 **출력 단어를 예측하기 위해서 디코더의 셀은 두 개의 입력값을 필요로 함**  
  - 이전 시점인 **t-1의 은닉 상태**  
  - 이전 시점 **t-1에 나온 출력 단어**  
- 출력 단어 예측에 **어텐션 값(Attention Value)** 도 추가로 필요로 함  
  - t번째 단어를 예측하기 위한 어텐션 값 = **$$a_t$$**  

- **어텐션 스코어(Attention Score)**  
  - 현재 디코더의 **시점 t에서 단어를 예측하기 위해**, **인코더의 모든 은닉 상태** 각각이 **디코더의 현 시점의 은닉 상태 $$s_t$$와 얼마나 유사한지** 판단하는 스코어값  
  - 해당 스코어 값을 구하기 위해 **s_t를 전치(transpose)** 하고 각 **은닉 상태와 내적(dot product) 실시**  
  - 모든 어텐션 스코어 값은 **스칼라**  
  - **0~1 사이의 값**을 가짐
<img width="352" alt="image" src="https://github.com/user-attachments/assets/5e38a554-faf1-468a-84be-1730fab010e4">  

- s_t와 인코더의 모든 은닉 상태의 **어텐션 스코어의 모음값을 $$e^t$$**로 정의  

## 2) 소프트맥스(softmax) 함수를 통해 어텐션 분포(Attention Distribution) 구하기  
<img width="660" alt="image" src="https://github.com/user-attachments/assets/ae850d4a-2305-47cb-b3fd-bd356b90c50d">  

- **$$e^t$$ 에 소프트맥스 함수를 적용하여, 모든 값을 합하면 1이 되는 확률 분포 = 어텐션 분포(Attention Distribution)**  
- **각각의 값 = 어텐션 가중치(Attention Weight)**  
- 어텐션 가중치가 클수록 직사각형의 크기가 큼  
- 디코더의 시점 t에서의 어텐션 가중치의 모음값인 어텐션 분포를 $$\alpha^t$$ 이라고 할 때  
  - $$\alpha^t = softmax(e^t)$$  

## 3) 각 인코더의 어텐션 가중치와 은닉 상태를 가중합하여 어텐션 값(Attention Value) 구하기
<img width="609" alt="image" src="https://github.com/user-attachments/assets/206c25c6-9444-4db7-97d1-3cf2209e5c2e">  
  
- 어텐션의 최종 결과값을 얻기 위해 각 **인코더의 은닉 상태와 가중치값들을 곱하고, 최종적으로 모두 더함**  
- 즉, 가중합(Weighted Sum) 진행  
$$a_t = \sum_{i=1}^{N} \alpha_{i}^{t} h_{i}$$  
- 이러한 어텐션 값(= $$a_t$$ )은 종종 **인코더의 문맥을 포함하고 있다고 하여 컨텍스트 벡터(context vector)라고도 불림**  
  - seq2seq에서 인코더의 마지막 은닉 상태를 컨텍스트 벡터라고 부르는 것과 대조됨  

## 4) 어텐션 값과 디코더의 t 시점의 은닉 상태 연결(Concatenate)
<img width="688" alt="image" src="https://github.com/user-attachments/assets/c36dc955-7206-4eed-ba8c-14e68f3d805d">  
  
- $$a_t$$와 $$s_t$$를 결합하여 하나의 벡터로 만드는 작업을 수행 -> $$v_t$$  
- 해당 $$v_t$$를 $$\hat{y}$$ 예측 연산의 입력으로 사용하므로서 인코더로부터 얻은 정보를 활용하여 $$\hat{y}$$을 더 잘 예측할 수 있게 됨  

## 5) 출력층 연산의 입력이 되는 $$\tilde{s}t$$를 계산
<img width="492" alt="image" src="https://github.com/user-attachments/assets/d9ce036d-46ec-4078-9af8-6a37cdc8a016">  
  
- $$v_t$$를 가중치 행렬과 곱한 후에 하이퍼볼릭탄젠트 함수를 지나도록 하여 출력층 연산을 위한 새로운 벡터인 $$\tilde{s}t$$ 도출  
- 어텐션 매커니즘에서 출력층의 입력 = $$\tilde{s}t$$  

## 6) $$\tilde{s}t$$ 를 출력층의 입력으로 사용  
- 출력층의 입력으로 사용하여 예측 벡터 도출  
  $$\hat{y_t} = \text{Softmax}(W_y \tilde{s_t} + b_y)$$  
  






  
