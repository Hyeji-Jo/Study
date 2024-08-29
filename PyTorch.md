# 1. PyTorch 기본
## 1) 프레임워크란?
- 프로그램을 다룸에 있어, 공통적으로 사용되는 기능들을 표준화된 소스코드로 만들어 사용할 수 있도록 제공하는 것

## 2) 딥러닝 프레임워크의 종류와 특징
- Google에서 개발된 **Tensorflow**
- Meta(구 Facebook)에서 개발된 **PyTorch**
<img width="814" alt="image" src="https://github.com/user-attachments/assets/af83db5f-fa81-41a4-ab45-3ab54f15e28c">

## 3) Define and Run vs Define by Run
- **TensorFlow**
  -  **Define and Run** 특징
  -  실행할 계산에 관련된 그래프를 미리 다 정의하여 올려놓고
  -  그래프에 투입될 데이터들을 집어넣어 연산을 수행
  -  한번 실행이 된 상태에서 에러가 나면 찾기 어려움
- **PyTorch**
  -  **Define by Run** 특징
  -  **연산이 이루어지는 시점에서 동적으로 그래프를 만들어 연산 수행**
  -  조금 더 낮은 단위의 연산들로 구성
  -  **디버깅 및 구조 설계의 세분화 가능**
  -  **Numpy** 구조를 가지는 Tensor 객체로 array 표현
  -  **자동미분을 지원**하여 DL 연산 지원
  -  다양한 형태의 DL을 지원하는 함수와 모델을 지원

## 4) Tensor
- numpy에서 ndarray, 파이썬 내장 객체의 list와 유사한 배열 객체
- list보다 ndarray와 가장 유사
- 다차원 Arrays를 표현하는 PyTorch 클래스
- GPU를 이용한 연산 지원
- 텐서는 스칼라, 벡터, 행렬을 일반화한 개념으로 임의의 차원을 가지는 다차원 배열(multi - dimensional array)
- 스칼라, 벡터, 행렬은 각각 0, 1, 2차원 텐서의 개념으로 볼 수 있음
- 딥러닝과 머신러닝 분야에서는 고차원의 텐서를 데이터로 다루기 떄문에 중요한 데이터 구조

```py
# numpy - ndarray
import numpy as np
n_array = np.arange(10).reshape(2,5)
print(n_array)
# [[0 1 2 3 4]
#  [5 6 7 8 9]]
print('ndim :', n_array.ndim, ' shape:', n_array.shape) # n_dim : 2  shape: (2, 5)

# pytorch - tensor
import torch
t_array = torch.FloatTensor(n_array)
print(t_array)
# tensor([[0., 1., 2., 3., 4.],
#         [5., 6., 7., 8., 9.]])
print('ndim :', t_array.ndim, ' shape:', t_array.shape) # ndim : 2  shape: torch.Size([2, 5])
```

- Array to Tensor
  - Tensor 생성은 list나 ndarray를 사용 가능
  ```py
  # data to tensor
  data = [[3,5], [10,5]]
  x_data = torch.tensor(data)
  x_data

  # ndarray to tensor
  nd_array_ex = np.array(data)
  tensor_array = torch.from_numpy(nd_array_ex)
  tensor_array

  # 결과물 동일
  # tensor([[ 3,  5],
  #        [10,  5]])
  ```
- Tensor가 가질수 있는 data 타입은 numpy와 동일


## 5) Squeeze vs Unsqueeze
- 앞으로 딥러닝을 학습하면서, 다양한 행렬 연산들을 수행하게 됨
  - **행렬 연산시 가장 중요한것은 행렬의 크기**
  - 행렬의 크기가 동일해야 함
- 해당 함수를 통해 행렬의 '차원'을 하나 더 높이거나 줄일 수 있음
- ex) [1,2,3,4] 와 같은 행렬을 [[1,2,3,4]] 또는 [[1],[2],[3],[4]] 와 같이 변환 할 수 있음
- **view** : reshape과 동일하게 tensor의 shape을 변환
- **squeeze** : 차원의 개수가 1인 차원을 삭제(압축)
- **unsqueeze** : 차원의 개수가 1인 차원을 추가


## 6) Broadcasting
- torch를 통해 행렬을 연산할때, 두 행렬의 크기가 달라도 계산이 되는 경우 종종 존재
- 이 경우 작은 크기의 행렬이 큰 크기의 행렬의 차원으로 Broadcasting 됨
- 예시
  - A(2X4) : [[1,2,3,4],[5,6,7,8]]
  - B(1X4) : [1,2,3,4]
  - A+B = [[2,4,6,8],[6,8,10,12]]
    - 크기가 작은 B행렬이 크기가 큰 A행렬로 Broadcasting됨

## 7) Matrix multiplication
- Broadcasting을 수행할 때, 행렬의 곱은?
- **내적**을 계산할때만 **dot** 사용
- tensor에는 **행렬 곱**을 위한 **mm**, matmul 두가지의 메서드가 존재
  - mm 메서드의 경우 Broadcasting을 지원하지 않음
  - matmul 메서드의 경우 지원
 
## 8) nn.Functional
- 다양한 function들을 한번에 모아서 사용 할 수 있도록 하는 모듈
- 이를 통해서 sigmoid, tanh, cross entropy 등 다양한 함수들을 직접 구현하지 않아도 사용 가능

## 9) AutoGrad
- PyTorch의 핵심 기능인 자동 미분
- 어떤 tensor에 대한 연산 정보들을 기억했다가 자동으로 미분해주는 기능
  - Tensor 객체의 requires_grad 인자를 True로 만들어 주면 됨 

# 2. PyTorch 모델의 구성 요소

## 1) torch.nn.Module 
- 딥러닝 모델은 복잡한 구조로 되어있는 것처럼 보이지만, 몇 개의 코어 블록의 반복
- torch의 nn.Module은 모든 layer의 Base Class
- Input, Output, Forward, Backward 정의
- 학습의 대상이 되는 parameter(tensor) 정의

### nn.Parameter
- Tensor 객체의 상속 객체
- **nn.Module 내에 attribute**가 될 때는 **required_grad = True**로 지정되어 학습 대상이 되는 Tensor
- 우리가 직접 지정할 일은 잘 없음
  - 대부분의 layer에는 weights 값들이 지정되어 있음
 
## 2) Backward
- Layer에 있는 Parameter들의 미분을 수행
- Forward의 결과값 (model의 output=예측치)과 실제값간의 차이(loss)에 대해 미분을 수행
- 해당 값으로 Parameter 업데이트
- 실제 backward는 Module 단계에서 직접 지정가능
- Module에서 backward와 optimizer 오버라이딩
  - 사용자가 직접 미분 수식을 써야하는 부담
  - 쓸일은 없으나 순서는 이해할 필요 있음
- 순서
  - (Forward 과정을 통해 얻어진) Model의 출력값(Output)과 정답값(Label)의 차이를 Loss Function을 통해 계산
  - Backward는 Loss 값을 활용해 미분을 수행
  - back-propagation을 통해 Parameter(weight)를 update

### Loss function
- 입력값(input)에 대한 Model의 출력값(Output)과 정답값(Label)의 차이를 비교해 나타내는 함수
  - **MSE**(Mean Squared Error) Loss
    - **회귀**문제에서 많이 쓰이는 함수
    - 출력값(Output)과 정답값(Label)의 차이의 제곱에 대해 측정
  - **CE**(Cross Entropy) Loss
    - **분류**문제에서 많이 쓰이는 함수

## 3) PyTorch Dataset 이론
<img width="741" alt="image" src="https://github.com/user-attachments/assets/6008fcbe-1b14-4532-8ab9-551e43157e25">

### Dataset 클래스
- 데이터 입력 형태를 정의하는 클래스
- 데이터를 입력하는 방식의 표준화
- Image, Text, Audio 등에 따라 다른 입력정의
```py
import torch
from torch.utils.data import Dataset
class CustomDataset(Dataset):

  # 초기 데이터 생성 방법을 지정
  def __init__(self, text, labels):
    self.labels = labels
    self.data = text

  # 데이터의 전체 길이 return
  def __len__(self):
    return len(self.labels)

  # index 값을 주었을 때 반환되는 데이터의 형태 (X, y)
  def __getitem__(self, idx):
    label = self.labels[idx]
    text = self.data[idx]
    sample = {"Text": text, "Class": label}
    return sample
```
- **유의점**
  - 데이터 형태에 따라 각 함수를 다르게 정의
  - 모든 것을 데이터 생성 시점에 처리할 필요는 없음
    - image의 Tensor 변화는 학습에 필요한 시점에 변환
  - 데이터 셋에 대한 표준화된 처리방법 제공 필요
     - 후속 연구자 또는 동료에게는 빛과 같은 존재
  - 최근에는 HuggingFace등 표준화된 라이브러리 사용    

### DataLoader 클래스
- Data의 Batch를 생성해주는 클래스
- 학습직전(GPU feed전) 데이터의 변환을 책임
- Tensor로 변환 + Batch 처리가 메인 업무
- 병렬적인 데이터 전처리 코드의 고민 필요
```py
text = ['Happy', 'Amazing', 'Sad', 'Unhappy', 'Claim']
labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']
MyDataset = CustomDataset(text, labels)

from torch.utils.data import DataLoader
MyDataLoader = DataLoader(MyDataset, batch_size=2, shuffle=True)
next(iter(MyDataLoader))
```
- **파라미터**
  - **sampler**
    - index를 컨트롤하는 방법
    - SequentialSampler : 항상 같은 순서
    - RandomSampler : 랜덤, replacemetn 여부 선택 가능, 개수 선택 가능
    - SubsetRandomSampler : 랜덤 리스트, 위와 두 조건 불가능
    - WeigthRandomSampler : 가중치에 따른 확률
    - BatchSampler : batch단위로 sampling 가능
    - DistributedSampler : 분산처리 (torch.nn.parallel.DistributedDataParallel과 함께 사용)
  - **collate_fn**
    - map-style 데이터셋에서 sample list를 batch 단위로 바꾸기 위해 필요한 기능
    - zero-padding이나 Variable Size 데이터 등 데이터 사이즈를 맞추기 위해 많이 사용 


# 3. PyTorch 모델 관리하기

## 1) 모델 저장하기
- **model.save()**
  - 학습의 결과를 저장하기 위한 함수
  - 모델 형태(architecture)와 파라미터를 저장
  - 모델 학습 중간 과정의 저장을 통해 최선의 결과모델을 선택
  - 만들어진 모델을 외부 연구자와 공유하여 학습 재연성 향상

## 2) 모델 체크포인트 만들기
- 학습의 중간 결과를 저장하여 최선의 결과를 선택
- earlystopping 기법 사용시 이전 학습의 결과물을 저장
- loss와 metric 값을 지속적으로 확인 저장
- 일반적으로 epoch, loss, metric을 함께 저장하여 확인
- colab에서 지속적인 학습을 위해 필요

## 3) 모델 학습 시켜보기
- **Transfer learning**
  - 다른 데이터셋으로 만든 모델을 현재 데이터에 적용
  - 일반적으로 대용량 데이터셋으로 만들어진 모델의 성능 높음
  - 현재의 DL에서는 가장 일반적인 학습 기법
  - backbone architecture가 잘 학습된 모델에서 일부분만 변경하여 학습 수행
- **Freezing**
  <img width="573" alt="image" src="https://github.com/user-attachments/assets/8b5dbbe8-65ee-4a41-97d7-8c88eec23e0f">

  - pretrained model을 활용시 모델의 일부분을 frozen 시킴
  - 모델의 **학습 파라미터들을 update 하지 않고 얼려둔 상태**
  - 이를 통해 필요한 만큼의 파라미터를 효율적으로 update하여 연산속도를 올리고 빠른 결과를 내는 것이 목적
  - Auto Grad의 **requires_grad=False**가 되면 freezing 하는 것

# 4. Advanced PyTorch
## 1) PyTorch 프로젝트 구조 이해하기
- 개발 초기 단계에서는 대화식 개발 과정이 유리
  - 학습과정과 디버깅 등 지속적인 확인
- 배포 및 공유 단계에서는 notebook 공유의 어려움
  - 쉬운 재현의 어려움, 실행순서 꼬임
- DL 코드도 하나의 프로그램
  - 개발 용이성 확보와 유지보수 향상 필요
- 다양한 프로젝트 템플릿 존재
  - 사용자 필요에 따라 수정하여 사용
  - 실행, 데이터, 모델, 설정, 로깅 등 다양한 모듈을 분리하여 프로젝트 템플릿화 

### Machine learning project template (머신러닝 프로젝트 템플릿)
- AI 모델이 만들어지는 과정을 정형화한 것
- 머신러닝 프로젝트 템플릿 = 머신러닝 모델이 개발되는 과정
  - (1)데이터 전처리 -> (2)모델 아키텍쳐 구축 -> (3)모델 학습 -> (4)모델 검증 -> (5)모델 배포

### OOP (Object Oriented Programming, 객체지향 프로그래밍)
- 프로그램 코드를 하나의 "객체"로 생각하는 방식의 구현
  - 객체 : 다른 객체와 상호작용 하고 값을 주고 받으며 행동하는 하나의 주체를 의미
  - "이름" 은 서로를 구분하는 값으로 서로 다른 경우가 많을 것, 이런 경우 객체를 생성할 때 "이름" 이라는 속성을 부여해 하나의 "인스턴스"를 생성
  - 각각의 인스턴스들은 "클래스"로부터 공통적인 속성들을 부여받고 인스턴스마다의 특징을 생성 시점에 반영하도록 프로그래밍

## 2) Monitoring tools for PyTorch
