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
- Tensor 생성은 list나 ndarray를 사용 가능


## 5) Squeeze vs Unsqueeze
- 앞으로 딥러닝을 학습하면서, 다양한 행렬 연산들을 수행하게 됨
  - **행렬 연산시 가장 중요한것은 행렬의 크기**
  - 행렬의 크기가 동일해야 함
- 해당 함수를 통해 행렬의 '차원'을 하나 더 높이거나 줄일 수 있음
- ex) [1,2,3,4] 와 같은 행렬을 [[1,2,3,4]] 또는 [[1],[2],[3],[4]] 와 같이 변환 할 수 있음

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
- tensor에는 행렬 곱을 위한 mm, matmul 두가지의 메서드가 존재
  - mm 메서드의 경우 Broadcasting을 지원하지 않음
  - matmul 메서드의 경우 지원
 
## 8) nn.Functional
- 다양한 function들을 한번에 모아서 사용 할 수 있도록 하는 모듈
- 이를 통해서 sigmoid, tanh, cross entropy 등 다양한 함수들을 직접 구현하지 않아도 사용 가능

## 9) AutoGrad
- PyTorch의 핵심 기능인 자동 미분
- 어떤 tensor에 대한 연산 정보들을 기억했다가 자동으로 미분해주는 기능
  - Tensor 객체의 requires_grad 인자를 True로 만들어 주면 됨 
