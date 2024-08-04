# Transformer 소개  
- 구글이 자연어처리를 위해 2017년 발표한 모델  
- ChatGPT 역시 트랜스포머 기반 모델  
- 자연어처리 뿐만 아니라 컴퓨터 비전이나 음성 인식 등 다른 분야에도 적용  
- **인코더와 디코더를 모두 어텐션으로 구현한 모델**  
  - Attention 과정을 여러 레이어에서 반복   
- **Seq2Se에 비해 문장의 길이에 대한 제약이 없어짐**  
- 인코더가 인풋 문장을 보다 잘 이해하고 디코더가 자신이 앞서 생성한 단어들에 대해서도 보다 더 잘 이해할 수 있는 모델  
- **RNN이나 CNN을 전혀 필요로 하지 않음**  
  - 대신 **Positional Encoding**을 활용하여 문장내 각각의 단어들에 대한 **순서의 정보를 알려줌**   

## 1) 트랜스포머 구조∙활용에 따른 언어모델의 분화  
- 2021년 기준 최신 고성능 모델들은 Transformer 아키텍처를 기반으로 함  
- 인코더만 활용  
  - BERT (트랜스포머의 인코더 12개로 구성 - 자연어 이해에 강점을 보이는 모델)  
- 인코더 + 디코더 구조  
  - BART  
  - T5  
- 디코더만 활용  
  - GPT  
  - XLNet  

## 2) Seq2Seq 모델의 한계  
- seq2seq 개념 요약  
  - 인코더-디코더 구조로 구성  
  - 인코더 : 입력 시퀀스를 하나의 벡터 표현으로 압축  
  - 디코더 : 해당 벡터 표현을 통해 출력 시퀀스 생성  
- 단점  
  - 인코더가 입력 시퀀스를 **하나의 벡터로 압축하는 과정에서 입력 시퀀스의 정보가 일부 손실됨**    
  - 해당 단점을 보정하기 위해 **Attention 활용**  
  - Attention을 RNN의 보정을 위한 용도로서 사용하는 것이 아니라 **Attention만으로 인코더와 디코더를 만드는법 생각**  
  <img width="1191" alt="image" src="https://github.com/user-attachments/assets/ebde630c-641a-4b26-b3d6-45ee3e1cdc44">   
  
  <img width="712" alt="image" src="https://github.com/user-attachments/assets/e1cd09a0-899e-4c76-b98f-1e35458b5c00">  
  

# 구조와 원리  
## 1) 인코더 + 디코더 구조  
- 인코더  
  - 입력 : 텍스트 또는 음성의 특징(피처) 벡터  
  - 출력 : 입력의 숨겨진 표현(hidden representation)  
  - 구성 : Self-Attention과 피드포워드 뉴럴 네트워크  
    - Self-Attention : 각각의 단어가 서로에게 어떤 연관성을 가지고 있는지를 구하기 위해 사용  
- 디코더  
  - 입력 : 인코더의 출력과 이전 디코더 출력(디코더는 순차적으로 작동)  
  - 출력 : 최종 결과물(ex) 번역된 문장)  
  - 구성 : Self-Attention, Encoder-Decoder Attention, 피드 포워드 뉴럴 네트워크  
    - Encoder-Decoder Attention : 디코더가 인코더의 전체 출력 시퀀스에서 중요한 부분을 선택적으로 참조하여 더 나은 출력을 생성 할 수 있게 함  

## 2) 인코더  
<img width="448" alt="image" src="https://github.com/user-attachments/assets/ac6e07d6-02f3-4620-a642-9f067a5cd81f">  
  
1. **입력 값 임베딩** (트랜스포머 이전의 전통적인 임베딩은 해당 단계만 실행)  
2. RNN을 사용하지 않으려면 **위치 정보를 포함하고 있는 임베딩을 사용**해야 함  
  - 이를 위해 **Positional Encoding** 사용  
3. 임베딩이 끝난 후 **어텐션(Attention)을 진행**  
4. 성능 향상을 위해 **잔여 학습(Residual Learing)을 사용**  
5. **어텐션(Attention)과 정규화(Normalization) 과정 반복**  
  - **각 레이어는 서로 다른 파라미터**를 가짐  
<img width="331" alt="image" src="https://github.com/user-attachments/assets/adbbf628-a584-4e50-aabd-84bd2051540e">  
  
## 3) 인코더와 디코더  
<img width="732" alt="image" src="https://github.com/user-attachments/assets/f18367cb-5692-422c-8d76-8bf7be7cb9ef">   

- 트랜스포머에서는 **마지막 인코더 레이어의 출력**이 모든 디코더 레이어에 입력됨  
  - 인코더의 각 레이어별로 나온 모든 출력값을 받지 않음  
  - **n_layers = 4**일 때의 예시  
  <img width="665" alt="image" src="https://github.com/user-attachments/assets/4b289b85-0ce1-4037-8fc8-d16e6e95ff20">  
  
- 디코더에서 2개의 Attention이 사용됨  
  1. Self-Attention(아래)  
    - 각각의 단어들이 서로가 서로에게 어떤 **가중치**를 가지는지 계산  
    - 출력되는 문장에 대한 전반적인 표현을 구축  
  2. Encoder-Decoder Attention(위)  
    - Encoder에 대한 정보를 attention 할 수 있도록 만듦  
    - 각각의 출력 단어가 encoder의 출력 정보를 받아와 사용할 수 있음  
    - 각각의 **출력되고 있는 단어가 소스 문장에서의 어떤 단어와 연관성이 있는지 도출**  
- 트랜스포머에서도 인코더(Encoder)와 디코더(Decoder)의 구조를 따름  
  - 이때 **RNN을 사용하지 않으며, 인코더와 디코더를 다수 사용**한다는 점이 특징  
  - LSTM이나 RNN 등은 입력단어의 개수만큼 반복적으로 인코더 레이어를 거쳐서 매번 Hidden state 생성  
    - 그리고 이전 타임스텝의 hidden state를 다음 타임스텝으로 전달하여 정보를 유지  
    - 모든 타임스템은 이전 타임스템의 계산 결과에 의존하므로, 계산이 순차적으로 일어나야 함   
  - Transformer의 경우 입력 단어 자체가 하나로 연결되어 한번에 입력이 되고 한번에 그에 대한 attention 값 도출 -> 계산 복잡도가 낮음  
    - 긴 시퀀스나 복잡한 상호작용이 필요한 상황에서 더 효율적   

  <img width="577" alt="image" src="https://github.com/user-attachments/assets/3c5adfe8-34a4-4817-b162-d062afbdc77e">  

## 4) Attention  
- 인코더와 디코더는 **Multi-Head Attention 레이어**를 사용  
<img width="711" alt="image" src="https://github.com/user-attachments/assets/5bca7db8-a1a7-4f4f-903f-0ca58cb5f3b1">  
<img width="710" alt="ima. e" src="https://github.com/user-attachments/assets/b8e6268c-008c-46cc-83fa-b5e068c9066a">  

- **Scaled Dot-Product Attention**
  - 어텐션을 위해 쿼리(Query), 키(Key), 값(Value) 필요  
  - 각 단어의 임베딩(Embedding)을 이용해 생성 가능  
    <img width="570" alt="image" src="https://github.com/user-attachments/assets/0a86f52f-b90d-4a8d-b052-6d28a2eb630a">

  <img width="722" alt="image" src="https://github.com/user-attachments/assets/af57b686-993d-44d3-9abf-c1f1dbce7a35">  

  <img width="650" alt="image" src="https://github.com/user-attachments/assets/e6dec723-1363-4142-bd99-f75f1e8cc058">  

  <img width="667" alt="image" src="https://github.com/user-attachments/assets/cb5a1f40-1d09-4cce-a888-01b5f6f52bd7">  
  
<img width="684" alt="image" src="https://github.com/user-attachments/assets/5cafd692-5598-432f-8855-c5db3f432603">  
  
- **Multi-Head Attention**
  <img width="723" alt="image" src="https://github.com/user-attachments/assets/9ad5ad39-d8a6-458f-bea4-94d8f85918d5">  

  <img width="667" alt="image" src="https://github.com/user-attachments/assets/85d9eb12-7d07-4564-b616-ec764b9a8d1d">





  

- **3가지 종류의 Attention 존재**  
  - Encoder Self-Attention  
    - 입력 데이터의 각 부분이 서로 어떻게 관련되어 있는지 파악  
    - 모델이 문장이나 음성 데이터의 맥락을 이해하는 데 중요한 역할   
  - Masked Decoder Self-Attention  
  - Encoder-Decoder Attention  
- 입력 문장을 토큰화해 사전을 만들고 토큰을 정수에 매핑시켜 임베딩 층을 통과하면 모델이 학습하기 위한 토큰들의 임베딩 값이 만들어 짐  
  - 트랜스포머는 단어를 표현하는 임베딩 벡터와 모델 내 입출력 값이 모두 같은 512 차원  
- 첫 번째 인코딩 츨에서는 **입력된 문장의 토큰들끼리 유사도 계산**  
  - 512차원을 n개 head로 나눠서 학습함 (= **Multi-head Attention**)  
 
  




 
# 기존 모델과의 차이점
- Transformer가 등장하기 전 NLP 분야에서는 주로 RNN, LSTM, GRU 등의 모델이 사용됨
- **RNN**
  - 순차적 데이터를 처리하기 위해 설계된 신경망
  - 먼 과거의 정보를 효과적으로 활용하지 못하는 한계 존재
- **LSTM**
  - RNN의 장거리 의존성 한계를 해결하기 위해 개발된 모델
  - 정보를 선택적으로 기억하고 잊을 수 있는 구조로 긴 시계열 데이터의 예측 정확도를 높임
- **GRU**
  - LSTM을 단순화한 모델
  - 비슷한 성능을 보이면서 계산 효율성을 개선함
- **Transformer**
  - 기존의 모델들은 순차적으로 데이터를 처리하나 transformer의 경우 문장의 모든 단어를 동시에 처리할 수 있음 **(병렬 처리)**
  - Self-Attention 메커니즘을 통해 문장 내 모든 단어 간의 관계를 직접적으로 계산 **(장거리 의존성 문제 해결)**
  - 병렬 처리가 가능해 문장 길이에 덜 민감, 다만 Self-Attention 연산의 복잡도는 문장 길이의 제곱에 비례 **(계산 복잡도 완화)**
  - 기존의 RNN 계열 모델은 이전 상태를 기억하기 위한 내부 메모리를 사용했으나 transformer의 경우 별도의 메모리 구조 없이 어텐션 메커니즘으로 정보 처리 **(메모리 사용 절약)**
 
# Transformers in Speech Recognition
- 전체 문장을 한 번에 분석하므로 음성을 이해하는 데 장점 존재
- 음성 신호를 텍스트로 변환하는데 사용
  1. 음성 입력 : 음성 신호 입력 받기
  2. 전처리 : 음성 신호를 작은 청크(chunk)로 나누고, 각 청크의 특징(ex) 스팩트로그램) 추출
  3. 트랜스포머 모델 적용
    - 인코더 : 음성 특징을 입력으로 받아 숨겨진 표현을 생성
    - 디코더 : 숨겨진 표현을 기반으로 텍스트 생성
 
# 기타 개념
## 1) Residual Learning
- ResNet으로 널리 알려져 있음
- 신경망이 직접 목표 출력 값을 예측하는 대신, 목표 출력과 입력 사이의 차이(residual)를 예측하도록 하는 것
- **배경**
  - 일반적으로 네트워크가 깊어질수록 훈련이 어려워지는 문제 존재
  - 이러한 문제는 주로 기울기 소실(gradient vanishing)과 기울기 폭발(gradient exploding)로 인해 발생
  - 해당 문제를 해결하기 위해 도입됨
- **Residual Block**
  - Residual Learning의 핵심 구성 요소
  - 입력 x를 바로 다음 블록으로 전달하면서, x를 여러 레이어를 거친 출력 F(x)에 더함
  - F(x)는 입력 x에 대한 변화 학습
  - 이렇게 하면 모델은 x에 대한 변화를 학습하는 데 집중할 수 있음
  - 해당 방식은 신경망이 더 쉽게 학습하고, 더 깊은 네트워크도 효율적으로 훈련할 수 있게 함 
