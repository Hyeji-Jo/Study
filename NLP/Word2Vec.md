# 0. 워드 임베딩(Word Embedding)  
- 컴퓨터가 이해할 수 있도록 텍스트를 적절히 숫자로 변환  
  - 현재에는 각 단어를 인공 신경망 학습을 통해 벡터화 하는 워드 임베딩 방법 존재  
- 단어를 벡터로 표현하는 방법  

## 1) 희소 표현(Sparse Representation)  
- **타켓 단어의 인덱스 값만 1이고, 나머지 인덱스는 전부 0**으로 표현 -> 원-핫 벡터  
- 벡터 또는 행렬(matrix)의 값의 대부분이 0으로 표현되는 방법  
- 원-핫 벡터, DTM는 희소 벡터  
- Ex) 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0] # 이때 1 뒤의 0의 수는 9995개. 차원은 10,000  

- 문제점  
  - 단어의 개수가 늘어나면 **벡터의 차원이 한없이 커짐**  
  - 이러한 벡터 표현은 **공간적 낭비**를 불러일으킴  
  - **단어의 의미를 표현하지 못함**  
    
## 2) 밀집 표현(Dense Representation)  
- 희소 표현과 반대되는 표현  
  - 밀집 표현은 벡터의 차원을 단어 집합의 크기로 상정하지 않음  
- **사용자가 설정한 값**으로 모든 단어의 **벡터 표현의 차원을 맞춤**  
  - 또한 해당 과정에서 0,1 뿐만 아니라 **실수값**을 가짐  
- Ex) 강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...] # 이 벡터의 차원은 128  

## 3) 워드 임베딩(Word Embedding)  
- **단어를 밀집 벡터의 형태로 표현**하는 방법  
- 밀집 벡터를 워드 임베딩 과정을 통해 나온 결과라고 하여 **임베딩 벡터**라고 함  
- 워드 임베딩 방법론으로는 LSA, Word2Vec, FastText, Glove등이 존재  

<img width="482" alt="image" src="https://github.com/user-attachments/assets/cf4de8a8-9257-40d0-840f-f90c44288785">

# 1. 워드투벡터(Word2Vec)  
- 원-핫 벡터는 단어 벡터 간 유의미한 유사도를 계산할 수 없다는 단점 존재  
- **단어 벡터 간 유의미한 유사도를 반영할 수 있도록 단어의 의미를 수치화**해야함  

## 1) 분산 표현(Distributed Representation)  
- 희소 표현의 단점  
  - 각 단어 벡터간 유의미한 유사성을 표현할 수 없다는 단점 존재  
  - 대안으로 **단어의 의미를 다차원 공간에 벡터화**하는 방법 -> **분산 표현**  
- 기본적으로 **분포 가설(distributional hypothesis)이라는 가정 하에 만들어진 표현 방법**  
  - **'비슷한 문맥에서 등장하는 단어들은 비슷한 의미를 가진다'** 라는 가정  
- 분포 가성에 따라 해당 내용을 가진 텍스트의 단어들을 벡터화한다면 해당 단어 벡터들은 유사한 벡터값을 가짐  
- **정리**  
  - 희소 표현 = 고차원에 각 차원이 분리된 표현 방법  
  - 분산 표현 = 저차원에 **단어의 의미를 여러 차원에다가 분산하여 표현**  
    - 이런 표현 방법을 사용하면 **단어 벡터간 유의미한 유사도 계산 가능**  
    
## 2) CBOW(Continuous Bag of Words)  
- Word2Vec은 2가지 학습 방식 존재  
  - CBOW(Continuous Bag of Words)  
    - 주변에 있는 단어들을 입력으로 중간에 있는 단어들을 예측하는 방법   
  - Skip-Gram  
    - 중간에 있는 단어들을 입력으로 주변 단어들을 예측하는 방법  
    
- 예문 : "The fat cat sat on the mat"  
  - ['The', 'fat', 'cat', 'on', 'the', 'mat']으로부터 sat을 예측하는 것은 CBOW가 하는 일  
  - 예측해야하는 단어 sat = **중심 단어(center word)**  
  - 예측에 사용되는 단어들 = **주변 단어(context word)**  
  - 중심 단어를 예측하기 위해서 앞, 뒤로 몇 개의 단어를 볼지를 결정해야 하는데 이 범위 = **윈도우(window)**  
    - 윈도우 크기가 정해지면 윈도우를 옆으로 움직여서 주변 단어와 중심 단어의 선택을 변경해가며 학습을 위한 데이터셋 생성  
    - 이 방법을 **슬라이딩 윈도우(sliding window)** 라 함  
    
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/59553c35-e6d3-448d-b3dd-4c62de2c1a05">
  
  - **Word2Vec에서 입력은 모두 원-핫 벡터**가 되어야 함  
  - 즉, 위 그림은 결국 **CBOW를 위한 전체 데이터 셋**  
  - Word2Vec은 **은닉층이 1개인 얕은 신경망(shallow neural network)**  
  - 일반적인 은닉층과는 달리 **활성화 함수가 존재하지 않음**  
  - 룩업 테이블이라는 **연산을 담당하는 층으로 투사층(projection layer)** 이라고 부르기도 함  
  <img width="538" alt="image" src="https://github.com/user-attachments/assets/8a5527e8-02c4-436b-ab63-08c041f45387">

  - 투사층의 크기가 M = 임베딩하고 난 벡터의 차원  
  - 입력층과 투사층 사이의 가중치 W는 V × M 행렬  
  - 출력층사이의 가중치 W'는 M × V 행렬  
  - V는 단어 집합의 크기  
  - **주의! : W, W'는 전치한 것이 아니라, 서로 다른 행렬**   
  - 훈련 전에 이 가중치 행렬 W와 W'는 랜덤 값, 중심 단어를 더 정확히 맞추기 위해 계속해서 이 W와 W'를 학습해가는 구조  
    - 그 이유가 여기서 **lookup해온 W의 각 행벡터가** Word2Vec **학습 후에는 각 단어의 M차원의 임베딩 벡터로 간주되기 때문**   
  <img width="549" alt="image" src="https://github.com/user-attachments/assets/d48fe79a-c0a4-4a73-bde8-0e0344e91636">

  - 주변 단어의 원-핫 벡터에 대해서 가중치 W 곱함 = 결과 벡터들  
  - 결과 벡터들은 투사층에서 만나 이 벡터들의 평균인 벡터를 구함  
    - 만약 윈도우 크기 n=2라면, **입력 벡터의 총 개수는 2n**이므로 중간 단어를 예측하기 위해서는 총 4개가 입력 벡터로 사용  
  - **투사층에서 벡터의 평균을 구하는 부분은 CBOW가 Skip-Gram과 다른 차이점**  
    - Skip-Gram은 입력이 중심 단어 하나이기때문에 투사층에서 벡터의 평균을 구하지 않음  
  <img width="544" alt="image" src="https://github.com/user-attachments/assets/e59c70df-4865-4815-944a-0d8057c050c9">

  - 구해진 평균 벡터는 두번째 가중치 행렬 W'와 곱해짐  
  - CBOW는 **소프트맥스(softmax)** 함수를 지나면서 벡터의 **각 원소들의 값은 0과 1사이의 실수**로, 총 **합은 1**이 됨  
  - 두 **벡터값의 오차를 줄이기위해** CBOW는 **손실 함수(loss function)로 크로스 엔트로피(cross-entropy) 함수**를 사용  
    - 입력값 = 크로스 엔트로피 함수에 중심 단어인 원-핫 벡터와 스코어 벡터  
    - 역전파(Back Propagation)를 수행하면 W와 W'가 학습됨  
    - 학습이 다 되었다면 M차원의 크기를 갖는 W의 행렬의 행을 각 단어의 임베딩 벡터로 사용하거나 W와 W' 행렬 두 가지 모두를 가지고 임베딩 벡터를 사용  

## 3) Skip-gram  
- 중심 단어에서 주변 단어를 예측  
<img width="539" alt="image" src="https://github.com/user-attachments/assets/a9565dfe-26ad-41b6-b060-d2eea722ec9a">

<img width="489" alt="image" src="https://github.com/user-attachments/assets/32423056-499a-476e-9541-2b21f63886ef">

- 여러 논문에서 성능 비교를 진행했을 때 **전반적으로 Skip-gram이** CBOW보다 **성능이 좋다**고 알려짐  

## 4) NNLM Vs. Word2Vec  
<img width="489" alt="image" src="https://github.com/user-attachments/assets/52f519f4-f95e-48aa-8e9d-b241f5c9ab55">

- NNLM은 단어 벡터 간 유사도를 구할 수 있도록 워드 임베딩의 개념을 도입  
- **워드 임베딩 자체에 집중**하여 NNLM의 **느린 학습 속도와 정확도를 개선**하여 탄생한 것이 **Word2Vec**  
- NNLM  
  - 다음 단어를 예측하는 언어 모델이 목적이므로 다음 단어를 예측  
  - 예측 단어의 이전 단어들만을 참고  
- Word2Vec  
  - 워드 임베딩 자체가 목적이므로 다음 단어가 아닌 중심 단어를 예측  
  - 예측 단어의 전, 후 단어들을 모두 참고  
  - NNLM에 존재하던 활성화 함수가 있는 은닉층을 제거  
  - 학습 속도에서 강점을 가지는 이유는 **은닉층을 제거한 것뿐만 아니라 추가적으로 사용되는 기법들** 덕분  
    - **계층적 소프트맥스(hierarchical softmax)** 와 **네거티브 샘플링(negative sampling)**  
   
## 5) 네거티브 샘플링(negative sampling)  
- Word2Vec의 출력층  
  - 소프트맥스 함수를 지난 단어 집합 크기의 벡터와 실제값인 원-핫 벡터와의 오차 구함  
  - 이로부터 임베딩 테이블에 있는 모든 단어에 대한 임베딩 벡터 값을 업데이트  
  - 만약, 단어 집합의 크기가 커진다면 굉장히 무거운 작업  

- 네거티브 샘플링  
  - Word2Vec이 학습 과정에서 전체 단어 집합이 아니라 일부 단어 집합에만 집중할 수 있도록 하는 방법  
  - 하나의 중심 단어에 대해서 전체 단어 집합보다 훨씬 작은 단어 집합을 만들어놓고 마지막 단계를 이진 분류 문제로 변환  
  - **주변 단어들을 긍정(positive)**, 랜덤으로 **샘플링 된 단어들을 부정(negative)** 으로 레이블링한다면 **이진 분류 문제를 위한 데이터셋**  
- 네거티브 샘플링 Skip-Gram  
  - 중심 단어와 주변 단어가 모두 입력이 되고, 이 두 단어가 실제로 윈도우 크기 내에 존재하는 이웃 관계인지 그 확률을 예측  
  <img width="309" alt="image" src="https://github.com/user-attachments/assets/debcb9e2-1520-4659-853d-9f8eeb83575e">
  <img width="548" alt="image" src="https://github.com/user-attachments/assets/befc4181-0983-4a34-810b-f607fc5c4ad6">
  <img width="544" alt="image" src="https://github.com/user-attachments/assets/f23ea20b-8c6b-4940-914c-51b952abd5f8">

  - 실제로는 입력1(중심 단어)와 주변 단어 관계가 아닌 단어들을 입력2로 삼기 위해서 단어 집합에서 랜덤으로 선택한 단어들을 입력2로 설정  
  - 입력1과 입력2가 실제로 **윈도우 크기 내에서 이웃 관계인 경우에는 레이블이 1**, **아닌 경우에는 레이블이 0**인 데이터셋  
  - 두 임베딩 테이블은 훈련 데이터의 단어 집합의 크기를 가지므로 크기가 같음  
  <img width="510" alt="image" src="https://github.com/user-attachments/assets/b9c1b1be-2148-4b65-a682-9567a74ff0b1">

  - 두 테이블 중 하나는 입력 1인 중심 단어의 테이블 룩업을 위한 임베딩 테이블  
  - 하나는 입력 2인 주변 단어의 테이블 룩업을 위한 임베딩 테이블  
  <img width="511" alt="image" src="https://github.com/user-attachments/assets/87617f49-cd89-4a30-aa6e-8bf8f91dcdf3">

  - **모델의 예측값 = 중심 단어와 주변 단어의 내적값**  
  - 레이블과의 오차로부터 역전파하여 중심 단어와 주변 단어의 임베딩 벡터값을 업데이트  




  




