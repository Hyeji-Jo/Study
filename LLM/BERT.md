- 트랜스포머 계열의 사전 훈련된 언어 모델
- 2018년에 구글이 공개한 사전 훈련된 모델

# 1. NLP에서의 사전 훈련(Pre-training)

## 1) 사전 훈련된 워드 임베딩
- 이전의 Word2Vec, FastText와 같은 워드 임베딩 방법론 존재
- 임베딩을 사용하는 방법 2가지 존재
  - 임베딩 층(Embedding layer)을 랜덤 초기화하여 처음부터 학습하는 방법
  - 사전에 학습된 임베딩 벡터들을 가져와 사용하는 방법
    - 태스크에 사용하기 위한 데이터가 적다면, 성능 향상을 기대해볼 수 있음
  - 2가지 방법 모두 하나의 단어가 하나의 벡터값으로 맵핑됨
    - 문맥을 고려하지 못하여 다의어나 동음이의어를 구분 못하는 문제점 존재
    - 한국어에는 '사과'가 다양한 의미를 가지고 있음
- 사전 훈련된 언어 모델을 사용하므로서 한계 극복
  - ELMo나 BERT 등

## 2) 사전 훈련된 언어 모델
<img width="785" alt="image" src="https://github.com/user-attachments/assets/bb3ae35d-7b25-4cba-9ffc-8181834d380b">

- 2015년 구글은 LSTM 언어 모델을 학습하고나서 텍스트 분류에 추가하는 학습 방법 보임
  - 사전 훈련된 언어 모델의 강점은 학습 전 사람이 별도 레이블을 지정해줄 필요가 없다는 점
- 방대한 텍스트로 LSTM 언어 모델을 학습해두고, 다른 테스크에서 높은 성능을 얻기 위해 사용하는 방법으로 ELMo와 같은 아이디어도 존재
  - 순방향 언어 모델과 역방향 언어 모델을 각각 따로 학습, 사전 학습된 언어 모델로부터 임베딩 값을 얻음
  - 다의어를 구분할 수 없었던 문데점 해결 가능
- OpenAI의 GPT-1
  - 트랜스포머 디코더로 총 12개의 층을 쌓은 후에 방대한 텍스트 데이터를 학습 시킴
  - 다양한 태스크를 위해 추가 학습을 진행하였을 때, 다양한 태스크에서 높은 성능을 얻을 수 있음
<img width="769" alt="image" src="https://github.com/user-attachments/assets/c5296528-1006-4bf8-86cb-76362e742401">

- 언어의 실제 문맥은 양방향
  - 하지만 이전 단어들로부터 다음 단어를 예측하는 언어 모델의 특성으로 인해 양방향 언어 모델을 사용할 수 없음
- 양방향 구조를 도입하기 위해 2018년 새로운 구조의 언어 모델 탄생
  - 마스크드 언어 모델
 
## 3) 마스크드 언어 모델(Masked Language Model)
- 입력 텍스트의 단어 집합의 15%의 단어를 랜덤으로 마스킹
  - 원래의 단어가 무엇이었는지 모르게 함
- 그리고 인공 신경망에게 마스킹된 단어들을 예측하도록 함  

# 2. BERT

## 1) BERT의 개요
<img width="759" alt="image" src="https://github.com/user-attachments/assets/23184621-6c4a-4378-8b68-7d77f0c1c130">

- **트랜스포머**를 이용해 구현되었으며, 위키피디아(25억 단어)와 BooksCorpus(8억 단어)와 같은 **레이블이 없는 텍스트 데이터로 사전 훈련된 언어 모델**
- 레이블이 없는 방대한 데이터로 사전 훈련된 모델을 가지고, 레이블이 있는 다른 작업에서 추가 훈련과 함께 하이퍼파라미터를 재조정하여 사용하면 성능이 높게 나오는 기존의 사례를 참고하였기에 성능이 높음
- 다른 작업에 대해 파라미터 재조정을 위한 **추가 훈련 과정을 파인 튜닝(Fine-tuning)** 이라 함
  - 위의 그림이 예시
  - 해당 경우, BERT가 언어 모델 사전 학습 과정에서 얻은 지식을 활용할 수 있으므로 더 좋은 성능을 얻을 수 있음

## 2) BERT의 크기
<img width="743" alt="image" src="https://github.com/user-attachments/assets/863f7d03-6122-4d8e-b23c-aa01146f1517">

- 트랜스포머 인코더 층의 수를 L, d_model의 크기를 D, 셀프 어텐션 헤드의 수를 A
  - BERT-Base : L=12, D=768, A=12 : 110M개의 파라미터
  - BERT-Large : L=24, D=1024, A=16 : 340M개의 파라미터

## 3) BERT의 문맥을 반영한 임베딩(Contextual Embedding)
- BERT는 문맥을 반영한 임베딩을 사용
<img width="425" alt="image" src="https://github.com/user-attachments/assets/e2b0caec-8abc-4fb7-a0e7-66f44246f7f8">

- 입력 : 임베딩 층(Embedding layer)를 지난 임베딩 벡터들 (D 차원)
<img width="743" alt="image" src="https://github.com/user-attachments/assets/04984683-8fec-4f0f-bf80-2e6081564f25">

- BERT의 연산을 거친 후의 출력 임베딩은 문장의 문맥을 모두 참고한 문맥을 반영한 임베딩
  - 모든 단어를 참고하고 있다는 것을 점선의 화살표로 표현
- 하나의 단어가 모든 단어를 참고하는 연산은 BERT의 12개의 층에서 전부 이뤄지는 연산
<img width="799" alt="image" src="https://github.com/user-attachments/assets/1c392287-c96e-47ce-b997-403ccd2add48">

- self-attention을 통해 모든 단어를 참고하여 문맥을 반영

## 4) BERT의 서브워드 토크나이저 : WordPiece
- **BERT는** 단어보다 더 작은 단위로 쪼개는 **서브워드 토크나이저 사용**
- **글자로부터 서브워드들을 병합해가는 방식으로 최종 단어 집합을 만드는 것**
- 자주 등장하는 단어는 그대로 단어 집합에 추가
- 자주 등장하지 않는 단어의 경우에는 더 작은 단위인 서브워드로 분리되어 서브워드들이 단어 집합에 추가됨
- 만들어진 단어 집합들을 기반으로 토큰화 수행
- **수행 과정**
  - 준비물 : 이미 훈련 데이터로부터 만들어진 단어 집합
  1. 토큰이 단어 집합에 존재
    - 해당 토큰을 분리하지 않음
  
  2. 토큰이 단어 집합에 존재하지 않음
    - 해당 토큰을 서브워드로 분리
    - 해당 토큰의 첫번째 서브워드를 제외한 나머지 서브워드들은 앞에 "##"를 붙인 것을 토큰으로 함
  ```py
  import pandas as pd
  from transformers import BertTokenizer
  
  tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") # Bert-base의 토크나이저
  result = tokenizer.tokenize('Here is the sentence I want embeddings for.')
  print(result) # ['here', 'is', 'the', 'sentence', 'i', 'want', 'em', '##bed', '##ding', '##s', 'for', '.']
  ```

## 5) 포지션 임베딩(Position Embedding)
- 트랜스포머에서는 포지셔널 인코딩(Positional Encoding)이라는 방법을 통해서 단어의 위치 정보를 표현
  - 사인 함수와 코사인 함수를 사용하여 위치에 따라 다른 값을 가지는 행렬을 만들어 단어 벡터들과 더하는 방법
- **BERT**의 경우 위치 정보를 사인 함수와 코사인 함수가 아닌 **학습을 통해 얻는 포지션 임베딩(Position Embedding)이라는 방법을 사용**
<img width="509" alt="image" src="https://github.com/user-attachments/assets/8967e082-e23d-449f-adef-5857a1f798eb">

- WordPiece Embedding = 단어 임베딩 = 실질적인 입력
- Position Embedding
  - 위치 정보를 위한 임베딩 층(Embedding layer)을 하나 더 사용
  - 실제 BERT에서는 문장의 최대 길이를 512로 하고 있으므로, 총 512개의 포지션 임베딩 벡터가 학습
  - 결론적으로 현재 설명한 내용을 기준으로는 BERT에서는 총 두 개의 임베딩 층이 사용

## 6) BERT의 사전 훈련(Pre-training)
<img width="997" alt="image" src="https://github.com/user-attachments/assets/2ff611b3-c9ec-4359-a164-e79cee33f995">

- ELMo
  - 정방향 LSTM과 역방향 LSTM을 각각 훈련시키는 방식으로 양방향 언어 모델
- GPT-1
  - 트랜스포머의 디코더를 이전 단어들로부터 다음 단어를 예측하는 방식으로 단방향 언어 모델
- **BERT**
  - GPT와 달리 가장 좌측 그림의 BERT는 화살표가 양방향으로 뻗어나가는 모습을 보여줌
  - **마스크드 언어 모델(Masked Language Model)을 통해 양방향성을 얻었기 때문**
  - **사전 훈련 방법은 크게 2가지**
    - 마스크드 언어 모델
    - 다음 문장 예측(Next sentence prediction, NSP)

### 마스크드 언어 모델(Masked Language Model, MLM)
- BERT는 인공 신경망의 입력으로 들어가는 입력 텍스트의 15%의 단어를 랜덤으로 마스킹(Masking)
- 그리고 인공 신경망에게 이 가려진 단어들을(Masked words) 예측하도록 함
- 정확히는 전부 [MASK]로 변경하지는 않고, 랜덤으로 선택된 15%의 단어들은 다시 다음과 같은 비율로 규칙이 적용
  - 80%의 단어들은 [MASK]로 변경
    - Ex) The man went to the store → The man went to the **[MASK]**
  - 10%의 단어들은 랜덤으로 단어가 변경
    - Ex) The man went to the **store** → The man went to the **dog**
  - 10%의 단어들은 동일
    - Ex) The man went to the store → The man went to the store
  - [MASK]만 사용할 경우에는 [MASK] 토큰이 파인 튜닝 단계에서는 나타나지 않으므로 사전 학습 단계와 파인 튜닝 단계에서의 불일치가 발생하는 문제 존재
  <img width="560" alt="image" src="https://github.com/user-attachments/assets/a1a3513e-8cf3-440d-8b5c-3fb563a41996">

<img width="566" alt="image" src="https://github.com/user-attachments/assets/207973bb-45de-44c9-8d51-731a9c607649">

  - BERT는 랜덤 단어 'king'으로 변경된 토큰에 대해서도 원래 단어가 무엇인지
  - 변경되지 않은 단어 'play'에 대해서도 원래 단어가 무엇인지를 예측해야 함
  - 'play'는 변경되지 않았지만 BERT 입장에서는 이것이 변경된 단어인지 아닌지 모르므로 마찬가지로 원래 단어를 예측해야 함
- BERT는 마스크드 언어 모델 외에도 다음 문장 예측이라는 또 다른 태스크를 학습 함

### 다음 문장 예측(Next Sentence Prediction, NSP)
- BERT는 두 개의 문장을 준 후에 이 문장이 이어지는 문장인지 아닌지를 맞추는 방식으로 훈련
  - 50:50 비율로 실제 이어지는 두 개의 문장과 랜덤으로 이어붙인 두 개의 문장을 주고 훈련
<img width="612" alt="image" src="https://github.com/user-attachments/assets/f9684a91-d992-4788-baeb-fefe6ade2065">

- BERT의 입력으로 넣을 때에는 [SEP]라는 특별 토큰을 사용해서 문장을 구분
- 그리고 이 두 문장이 실제 이어지는 문장인지 아닌지를 [CLS] 토큰의 위치의 출력층에서 이진 분류 문제를 풀도록 함
  - [CLS] 토큰은 BERT가 분류 문제를 풀기 위해 추가된 특별 토큰
- 위의 그림에서 나타난 것과 같이 **마스크드 언어 모델과 다음 문장 예측은 loss를 합하여 학습이 동시에 이뤄짐**
- BERT가 언어 모델 외에도 다음 문장 예측이라는 태스크를 학습하는 이유
  - QA(Question Answering)나 NLI(Natural Language Inference)와 같이 두 문장의 관계를 이해하는 것이 중요한 태스크들이 있기에 
 

## 7) 세그먼트 임베딩(Segment Embedding)
<img width="658" alt="image" src="https://github.com/user-attachments/assets/1bbcd5ca-d06f-4e4c-9d6a-96a190d21f6d">

- BERT는 QA(질문,본문) 등과 같은 두 개의 문장 입력이 필요한 태스크를 풀기도 함
  - 문장 구분을 위해 BERT는 **세그먼트 임베딩**이라는 또 다른 임베딩 층(Embedding layer)을 사용
  - 첫번째 문장에는 Sentence 0 임베딩, 두번째 문장에는 Sentence 1 임베딩을 더해주는 방식
  - 임베딩 벡터는 두 개만 사용
- **BERT는 총 3개의 임베딩 층이 사용됨**
  - **WordPiece Embedding**
    - 실질적인 **입력**이 되는 워드 임베딩
    - 임베딩 벡터의 종류는 **단어 집합의 크기**로 30,522개
  - **Position Embedding**
    - **위치 정보를 학습**하기 위한 임베딩
    - 임베딩 벡터의 종류는 **문장의 최대 길이**인 512개
  - **Segment Embedding**
    - **두 개의 문장을 구분**하기 위한 임베딩
    - 임베딩 벡터의 종류는 문장의 최대 개수인 **2**개
- BERT가 두 개의 문장을 입력받을 필요가 없는 경우도 있음
  - 예를 들어 네이버 영화 리뷰 분류나 IMDB 리뷰 분류와 같은 감성 분류 태스크
  - 이 경우에는 BERT의 전체 입력에 Sentence 0 임베딩만을 더함

## 8) BERT를 파인 튜닝(Fine-tuning)하기
### 하나의 텍스트에 대한 텍스트 분류 유형(Single Text Classification)
<img width="467" alt="image" src="https://github.com/user-attachments/assets/732b93eb-defe-4844-ba54-5f33b31f676f">

- 하나의 문서에 대한 텍스트 분류 유형
  - ex) 영화 리뷰 감성 분류, 로이터 뉴스 분류
- 문서의 시작에 [CLS] 라는 토큰을 입력
- 텍스트 분류 문제를 풀기 위해서 [CLS] 토큰의 위치의 출력층에서 밀집층(Dense layer) 층들을 추가하여 분류에 대한 예측 수행

### 하나의 텍스트에 대한 태깅 작업(Tagging)
<img width="457" alt="image" src="https://github.com/user-attachments/assets/5090b29e-16c2-4b6d-9b3e-f8a34b14c464">

- 대표적으로 문장의 각 단어에 품사를 태깅하는 품사 태깅 작업과 개체를 태깅하는 개체명 인식 작업 존재
- 출력층에서는 입력 텍스트의 각 토큰의 위치에 밀집층을 사용하여 분류에 대한 예측 수행

### 텍스트의 쌍에 대한 분류 또는 회귀 문제(Text Pair Classification or Regression)
<img width="707" alt="image" src="https://github.com/user-attachments/assets/e75f6238-f833-489d-be25-0e41915a3ad1">

- 대표적인 예로 자연어 추론 태스크 존재
  - 자연어 추론 문제란, 두 문장이 주어졌을 때, 하나의 문장이 다른 문장과 논리적으로 어떤 관계에 있는지를 분류
  - 유형으로는 모순 관계(contradiction), 함의 관계(entailment), 중립 관계(neutral) 존재
- 입력 텍스트가 1개가 아니므로, 텍스트 사이에 [SEP] 토큰을 집어넣고 수행
- Sentence 0 임베딩과 Sentence 1 임베딩이라는 두 종류의 세그먼트 임베딩을 모두 사용하여 문서를 구분 

### 질의 응답(Question Answering)
<img width="694" alt="image" src="https://github.com/user-attachments/assets/0e8b60db-9856-45da-92c7-b58361e36b91">

- BERT로 QA를 풀기 위해서 질문과 본문이라는 두 개의 텍스트의 쌍을 입력

## 9) 그 외 기타
- 훈련 데이터는 위키피디아(25억 단어)와 BooksCorpus(8억 단어) ≈ 33억 단어
- WordPiece 토크나이저로 토큰화를 수행 후 15% 비율에 대해서 마스크드 언어 모델 학습
- 두 문장 Sentence A와 B의 합한 길이. 즉, 최대 입력의 길이는 512로 제한
- 100만 step 훈련 ≈ (총 합 33억 단어 코퍼스에 대해 40 에포크 학습)
- 옵티마이저 : 아담(Adam)
- 학습률(learning rate) : 
- 가중치 감소(Weight Decay) : L2 정규화로 0.01 적용
- 드롭 아웃 : 모든 레이어에 대해서 0.1 적용
- 활성화 함수 : relu 함수가 아닌 gelu 함수
- 배치 크기(Batch size) : 256

## 10) 어텐션 마스크(Attention Mask)
<img width="580" alt="image" src="https://github.com/user-attachments/assets/0be8affe-1c4f-42bb-8011-87032a5ec91a">

- BERT를 사용하기 위해 어텐션 마스크라는 시퀀스 입력이 추가로 필요
- 어텐션 마스크
  - 불필요하게 패딩 토큰에 대해서 어텐션을 하지 않도록 알려주는 입력
  - 실제 단어와 패딩 토큰을 구분할 수 있도록
  - 0과 1 두 가지 값
    - 1 : 해당 토큰은 실제 단어이므로 마스킹을 하지 않는다라는 의미
    - 0 : 해당 토큰은 패딩 토큰이므로 마스킹을 한다는 의미  





 


