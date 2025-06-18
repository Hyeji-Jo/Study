- BERT로부터 문장 벡터를 얻는 방법
  - BERT의 **[CLS] 토큰의 출력 벡터**를 문장 벡터로 간주
  - 모든 단어의 출력 벡터에 대해서 **평균 풀링**을 수행한 벡터를 문장 벡터로 간주
  - 모든 단어의 출력 벡터에 대해서 **맥스 풀링**을 수행한 벡터를 문장 벡터로 간주

- SBERT는 기본적으로 BERT의 문장 임베딩의 성능을 우수하게 개선시킨 모델
  - BERT의 문장 임베딩을 응용하여 BERT를 파인 튜닝

## 1) 문장 쌍 분류 태스크로 파인 튜닝
<img width="612" alt="image" src="https://github.com/user-attachments/assets/9d7e80f7-6928-4041-9107-c1d1d5567b03">

- 두 개의 문장이 주어지면 수반(entailment) 관계인지, 모순(contradiction) 관계인지, 중립(neutral) 관계인지를 맞추는 문제
  - 문장 A와 B 각각을 BERT의 입력으로 넣고, 평균 풀링 또는 맥스 풀링을 통해 각각에 대한 문장 임베딩 벡터를 얻음
  - 2 벡터의 차이를 구하고, 해당 세 가지 벡터를 연결
  - 벡터의 차원이 n이라면, 세 개의 벡터를 연결한 벡터 h의 차원은 3n
  <img width="343" alt="image" src="https://github.com/user-attachments/assets/6f8c8cea-fee4-46bb-ac7c-b5ecf494e395">


  - 분류하고자 하는 클래스의 개수가 k라면, 가중치 행렬 3n * k의 크기를 가지는 행렬을 곱한 후 소프트맥스 함수 통과
## 2) 문장 쌍 회귀 태스크로 파인 튜닝
- 두 개의 문장으로부터 의미적 유사성을 구하는 문제
  <img width="325" alt="image" src="https://github.com/user-attachments/assets/76b9bc2a-215d-43f2-82d6-f990e0b21d5e">

- 두개의 문장 임베딩 벡터의 코사인 유사도 도출
- 해당 유사도와 레이블 유사도와의 평균 제곱 오차(MSE)를 최소화하는 방식으로 학습
