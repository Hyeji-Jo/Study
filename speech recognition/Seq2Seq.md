# 소개
- 2014년 발표된 언어모델
- 자연어처리 역사상 가장 큰 발명 중의 하나
- 기존 문장을 수치로 바꾸는것 뿐만 아니라 여기서 생성된 수치를 활용해 다시 문장을 생성할 수 있는 모델
- ex) 한국어 문장을 넣어 영어 문장을 생성하도록 하는 것
- 입력 문장을 처리하는 인코더 부분과 이를 받아 다시 문장을 생성하는 디코더에 RNN 활용
  - 문장이 길어질수록 이전 정보를 기억하기 힘들다는 RNN의 고질적인 문제 가짐
  - 뒤에 Attention이라는게 추가되면서 길이에 구애받지 않는 번역이 가능해지게 됨
  - 하지만 여전히 RNN 기반의 모델은 인간의 번역보다 퀄리티가 떨어짐

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

- 2021년 기준으로 최신 고성능 모델들은 Transformer 아키텍처를 기반으로 함
  - GPT : Transformer의 디코더(Decoder) 아키텍처 활용
  - BERT : Transformer의 인코더(Encoder) 아키텍처 활용
  <img alt="image" src="https://github.com/user-attachments/assets/11faec3f-ed01-4149-98d2-7a0e2f89cbaa">


 
  

