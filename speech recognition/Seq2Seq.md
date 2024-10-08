# 소개
- 2014년 발표된 언어모델
- 자연어처리 역사상 가장 큰 발명 중의 하나
- 기존 문장을 수치로 바꾸는것 뿐만 아니라 여기서 생성된 수치를 활용해 다시 문장을 생성할 수 있는 모델
- ex) 한국어 문장을 넣어 영어 문장을 생성하도록 하는 것
- 입력 문장을 처리하는 인코더 부분과 이를 받아 다시 문장을 생성하는 디코더에 RNN 활용
  - 문장이 길어질수록 이전 정보를 기억하기 힘들다는 RNN의 고질적인 문제 가짐
  - 뒤에 Attention이라는게 추가되면서 길이에 구애받지 않는 번역이 가능해지게 됨
  - 하지만 여전히 RNN 기반의 모델은 인간의 번역보다 퀄리티가 떨어짐
## 특징
- 입력과 출력의 시퀀스 길이 불일치
  - 예를들어, 한 문장을 다른 언어로 변환할 때 입력과 출력 시퀀스의 길이가 다를 수 있음
- **교사 강요**
  - 학습 시 디코더는 실제 정답을 입력으로 사용할 확률을 제어하는 교사 강제 전략 사용이 가능
  - 이를 통해 모델이 더 빠르게 수렴할 수 있음
- 단일 모델이 아님
  - RNN, LSTM과 같은 순환 신경망을 기반으로 구축
  - 구조적 아키텍처
- 활용
  - 챗봇과 기계번역에 주로 사용됨
- **Input**
  - 형태: [batch_size, input_sequence_length]
  - 예: 단어 인덱스 시퀀스, [3, 7, 15, 4] (배치 크기가 1인 경우), [3, 7, 15, 4, 2, 0] (패딩된 경우)
- **Output**
  - 형태: [batch_size, output_sequence_length]
  - 예: 번역된 단어 인덱스 시퀀스, [1, 5, 2, 7] (배치 크기가 1인 경우), [1, 5, 2, 7, 0, 0] (패딩된 경우)
- 임베딩
  - 단어의 의적 유사성을 반영할 수 있음
    - 예를 들어, "king"과 "queen" 같은 단어들은 임베딩 공간에서 서로 가까운 위치에 배치
  - 계산 효율성을 크게 향상시킴
  - 임베딩을 통해 모델이 단어 간의 패턴을 인식할 수 있음
    - 문법적 패턴 : 단어가 문장에서 차지하는 위치와 역할을 반영
    - 의미적 관계 : 단어 간의 유사성, 반의어, 동의어 관계 반영 
  

# Attention
- 디코더에서 출력 단어를 예측하는 매 시점(time step)마다 인코더의 전체 입력 문장을 다시 한 번 참고한다는 것
- 해당 시점에서 예측해야 할 단어와 가장 연관이 있는 단어를 좀 더 집중해서 보겠다
- 디코더의 현재 타임스텝의 출력값(hidden state)에 가중치를 곱해 쿼리를 만들고
  이를 인코더 전체의 타입스텝 출력값과 내적(dot product)해 예측해야 하는 단어를 잘 참조할 수 있도록 역전파를 통해 해당 가중치들을 학습시키는 것

# Sequence to Sequence Learning with Neural Networks (NIPS 2014)
## 딥러닝 기반의 기계 번역 발전 과정
- 본 논문에서는 LSTM을 활용한 효율적인 **Seq2Seq** 기계 번역 아키텍처를 제안
  -  **Seq2Seq**는 딥러닝 기반 기계 번역의 돌파구와 같은 역할을 수행
  -  **Transformer(2017)** 가 나오기 전까지 state-of-the-art로 사용
  -  Sequence -> 일반적으로 하나의 문장을 의미(문장 = 각각의 단어(token)들로 이루어짐)
  -  **context vector는 크기가 고정됨**
     <img alt="image" src="https://github.com/user-attachments/assets/34cd1446-d79e-4b36-9849-d490ada5d727">

- 2021년 기준으로 **최신 고성능 모델들은 Transformer 아키텍처를 기반**으로 함
  - **GPT** : **Transformer의 디코더(Decoder)** 아키텍처 활용
  - **BERT** : **Transformer의 인코더(Encoder)** 아키텍처 활용
  <img alt="image" src="https://github.com/user-attachments/assets/11faec3f-ed01-4149-98d2-7a0e2f89cbaa">

## 자연어 처리를 위한 기초 수학 : 언어 모델(Language Model)
- **언어 모델** : **문장(시퀀스)에 확률을 부여**하는 모델
- 언어 모델을 가지고 있으면 특정한 상황에서의 적절한 문장이나 단어를 예측할 수 있음
  - **기계 번역 예시**
    - P(난 널 사랑해|I love you) > P(난 널 싫어해|I love you)
  - **다음 단어 예측 예시**
    - P(먹었다|나는 밥을) > P(싸웠다|나는 밥을)
  - 해당 알고리즘을 활용하여 검색어 추천 알고리즘을 구축할 수 있음
- 하나의 **문장(W)** 은 여러 개의 **단어(w)** 로 구성됨
  - **결합 확률 분포**라고도 말함
  - P(W) = P(w1, w2, w3, ..., wn) 
  - P(친구와 친하게 지낸다) = P(친구와, 친하게, 지낸다)
- **연쇄 법칙(Chain Rule)**
  - P(w1, w2, w3, ..., wn) = P(w1)*P(w2|w1)*P(w3|w1,w2),...,P(wn|w1,w2,....,wn-1)
  - P(친구와 친하게 지낸다) = P(친구와)*P(친하게|친구와)*P(지낸다|친구와 친하게)
- **전통적인 통계적 언어 모델**은 카운트 기반의 접근을 사용
  - P(지낸다|친구와 친하게) = count(친구와 친하게 지낸다)/count(친구와 친하게)
  - 현실 세계에서 모든 문장에 대한 확률을 가지고 있으려면 매우 방대한 양의 데이터가 필요하며, 긴 문장은 처리하기 매우 어려움
  - 현실적인 해결책으로 **N-gram 언어 모델**이 사용됨
    - 인접한 일부 단어만 고려하는 아이디어
    - ex) P(먹었다|나는 공부를 마치고 집에서 밥을) -> 해당 경우 '집에서 밥을'만 고려하여 '먹었다'의 확률 도출

## 전통적인 RNN 기반의 번역 과정
- 전통적인 초창기 RNN 기반의 기계 번역은 입력과 출력의 크기가 같다고 가정
  <img alt="image" src="https://github.com/user-attachments/assets/962e3d41-e9b6-492e-a1ba-28631bd4cabf">
- 단어 하나 하나에 대해 개별적으로 결과를 도출할 경우 정확한 결과를 유추하기 어려움

## RNN 기반의 Sequence to Sequence 개요
- 전통적인 초창기 RNN 기반의 언어 모델에 다양한 한계점 존재
  - 이를 해결하기 위해 **인코더**가 고정된 크기의 **문맥 벡터(context vector)** 를 추출하도록 함
  - 이후에 문맥 벡터로부터 **디코더**가 번역 결과 추론
  - 본 Seq2Seq 논문에서는 LSTM을 이용해 문맥 벡터를 추출하도록 하여 성능 향상
    - 인코더의 마지막 hidden state만을 context vector로 사용
    - 가장 마지막에 추가된 hidden state가 전체 문맥의 내용을 담고 있는 벡터이기 때문
- 인코더와 디코더는 **서로 다른 파라미터(가중치)** 를 가짐
- 종료 시점 : <eos> = end of sequence

## Seq2Seq의 성능 개선 포인트
- 기본적인 RNN 대신 **LSTM**을 활용했을 때 더 높은 정확도를 보임
- 실제 학습 및 테스트 과정에서 **입력 문장의 순서를 거꾸로 했을 때 더 높은 정확도**를 보임
  - 출력 문장의 순서는 바꾸지 않음 

## Seq2Seq 구현 shape
- Encoder
  - Input = src = [src length, batch size]
  - embedded = [src length, batch size, embedding dim]
  - outputs = [src length, batch size, hidden dim * n directions]
  - hidden = [n layers * n directions, batch size, hidden dim]
  - cell = [n layers * n directions, batch size, hidden dim]
  - outputs are always from the top hidden layer

- Seq2Seq
  - Encoder 결과
    - src = [src length, batch size]
    - trg = [trg length, batch size]
  - hidden = [n layers * n directions, batch size, hidden dim]
  - cell = [n layers * n directions, batch size, hidden dim]
  - first input to the decoder is the <sos> tokens
  - output = [batch size, output dim]
  - hidden = [n layers, batch size, hidden dim]
  - cell = [n layers, batch size, hidden dim]
  - input = [batch size]
  
- Decoder
  - Seq2Seq 결과
    - input = [batch size]
    - hidden = [n layers * n directions, batch size, hidden dim]
    - cell = [n layers * n directions, batch size, hidden dim]
    - n directions in the decoder will both always be 1, therefore:
    - hidden = [n layers, batch size, hidden dim]
    - context = [n layers, batch size, hidden dim]
  
  - input = input.unsqueeze(0) 후 input = [1, batch size]
  - embedded = [1, batch size, embedding dim]
  - output = [seq length, batch size, hidden dim * n directions]
  - hidden = [n layers * n directions, batch size, hidden dim]
  - cell = [n layers * n directions, batch size, hidden dim]
  - seq length and n directions will always be 1 in this decoder, therefore:
  - output = [1, batch size, hidden dim]
  - hidden = [n layers, batch size, hidden dim]
  - cell = [n layers, batch size, hidden dim]
  - prediction = [batch size, output dim]
