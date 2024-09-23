# Neural Networks 소개 및 로지스틱 회귀

작성자: 조혜지
사용언어: Python
강의명: ML Bootcamp Korea 2023, 딥러닝 1단계 : 신경망과 딥러닝
투입기간: 2023/08/30 → 2023/09/05

/목차

---

# 1. Deep Learning 소개

## 1. Neural Network (신경망이란?)

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled.png)

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%201.png)

- 신경망의 단일 뉴런인 작은 원이 옆의 파란 함수 구성

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%202.png)

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%203.png)

- 신경망 정의
    - 위와 같은 과정을 딥러닝(Deep learning , 심층 신경망 학습)으로 진행하게 되면 가족구성원의 수, 보행성, 학교 수준 등의 변수를 스스로 만들어 학습
    - 즉, 우리가 알고 싶은 변수 x(주택의 정보)만 입력하면 y(주택의 가격)을 예측할 수 있음

<aside>
💡 Every input layer feature is interconnected with every hidden layer feature
→ **은닉층의 노드들은 모든 입력층의 노드들의 영향을 받는다**

</aside>

### ※ ReLU(Rectified Linear Unit) 함수

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%204.png)

<aside>
💡 Rectified란 '정류된' 이라는 뜻으로 흐름을 차단한다고 보면 됨

**ReLU** 함수도 **x 가 0 이하일 때**를 차단하여 아무 값도 출력하지 않고 **0 을 출력**
→ 이러한 점에서 전류가 차단된다고 보임
**x가 양수 값**의 경우 **값을 그대로** 내보내는 함수

</aside>

## 2. Supervised Learning with Neural Network (신경망 지도학습)

### 1) 신경망 예시 및 활용사례

![신경망 지도학습의 활용사례](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%205.png)

신경망 지도학습의 활용사례

![신경망 예시](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%206.png)

신경망 예시

- 신경망 종류
    1. 일반 신경망 - Standard NN
    2. 합성곱 신경망 - Convolutional NN
    3. 순환 신셩망 - Recurrent NN

### 2) 활용 데이터 종류

- 구조화된 데이터/**정형 데이터** - Structured Data
    1. 데이터 베이스
- 비구조화된 데이터/**비정형 데이터** - Unstructured Data
    1. 오디오, 음성 파일
    2. 이미지
    3. 텍스트

## 3. Deep Learning의 급부상 이유

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%207.png)

- **데이터의 양이 많아질수록, 신경망이 복잡할수록 딥러닝 학습의 성능은 올라감**
    - 데이터의 양이 적을때는 어느 신경망이 확실히 우월한지 배열이 잘 되어 있지 않기에 
    본인이 사용하고 싶은거 사용하면 됨
    - 데이터의 양 → **레이블 되어있는 데이터의 양**

1. 데이터 - Data
    1. 이전보다 많은 양의 데이터
2. 컴퓨터 - Computation
    1. CPU
    2. GPU
3. 알고리즘 - Algorithms
    1. 활성함수의 변화(Sigmoid → ReLU)

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%208.png)

![Sigmoid](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%209.png)

Sigmoid

![ReLU](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%204.png)

ReLU

- 시그모이드 함수는 **함수의 기울기가 0에 가까운 값이 됨**
    - Gradient Descent(경사 하강법) 도입의 경우 기울기가 0이면 개체가 느린 속도로 변화
    - 즉, **학습 속도가 매우 느려짐**
- ReLU
    - **기울기가 모든 양수에 대해 1**이 됨
    - 기울기가 0으로 줄어드는 확률은 급격히 감소

---

# 2. Logistic Regression (로지스틱 회귀)

## 1. Binary Classification (이진 분류)

![1 (고양이) VS 0 (고양이X)](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2010.png)

1 (고양이) VS 0 (고양이X)

### ※ 로지스틱 회귀 함수 표기법

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2011.png)

## 2. Logistic Regression (로지스틱 회귀)

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2012.png)

- Sigmoid 함수
    - z의 값이 커지면 → 1에 가까움
    - z의 값이 작아지면 → 0에 가까움
    

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2013.png)

- 로지스틱 회귀분석의 목표
    - **$w^T$와 $b$를 수정함으로써 분류 기준이 되는 적절한 선을 찾는것**
- 매개 변수(Parameters)
    - $w$ : 가중치 (weight)
    - $b$ : 편향 (bias)
- Sigmoid 함수 활용 이유
    - $0\le \hat{y} \le1$로 값을 맞춰주기 위해 
    시그모이드 함수 사용
    - y축을 기준으로 0.5보다 크면 고양이 사진, 작으면 고양이 사진이 아닌것으로 판별

## 3. Logistic Regression Cost Function 
(로지스틱 회귀 비용함수)

### 1) 손실 함수 (Loss function)

- 입력 특성(x)에 대한 예측값($\hat{y}$)과 정답값(y) 차이의 제곱
    - $L(\hat{y},y)=\frac{1}{2}(\hat{y}-y)^2$
    - 보통의 손실 함수 식

![지역 최소값](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2014.png)

지역 최소값

- 그러나 로지스틱 회귀에서 이러한 손실 함수를 
사용하면 **지역 최소값**에 빠질 수 있음
    - **$L(\hat{y},y)=-(ylog\hat{y} +  (1-y)log(1-\hat{y}))$**
    - **y가 1일 경우** → $L(\hat{y},1)=-log\hat{y}$
        - $log\hat{y}$과  $\hat{y}$가 커야함 → sigmoid 함수는 
        $0\le \hat{y} \le1$ → **$\hat{y}$은 1**에 수렴
    - **y가 0일 경우** → $L(\hat{y},0)=-log(1-\hat{y})$
        - $\hat{y}$이 작아야함 → sigmoid 함수는 $0\le \hat{y} \le1$ → **$\hat{y}$은 0**에 수렴

### 2) 비용 함수 (Cost function)

- 모든 입력에 대해 계산한 **손실 함수의 평균 값**으로 계산 가능
- **$J(w,b) =\frac{1}{m}\sum^{i=m}_{i=1}L(\hat{y}^{(i)},y^{(i)})= -\frac{1}{m}\sum^{i=m}_{i=1}(y^{(i)}log\hat{y}^{(i)}+(1-y^{(i)})log(1-\hat{y}^{(i)}))$**

## 4. Gradient Descent (경사 하강법)

![경사 하강법](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2015.png)

경사 하강법

- w와 b를 찾고 싶다면 $J(w,b)$를 최소화 하는것
- 함수의 최솟값을 모르기에 임의의 점을 골라 시작
- 오차(함숫값)가 가장 적은 **가장 아래의 빨간 점(Global Optimum)**

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2016.png)

- $w : w-\alpha\frac{dJ(w,b)}{dw}$
- $b : b-\alpha\frac{dJ(w,b)}{db}$
- $\frac{dJ(w)}{dw}$ : 도함수라고 하며, 미분을 통해 구한 값, dw라고 표기하기도 함
- $\alpha$ : 학습률
- 만약 dw >0이면 기존 w값 보다 작은 방향으로 업데이트, dw<0이면 기존 w값 보다 큰 방향으로 업데이트
- $dw =\frac{\partial J(w,b)}{\partial w}$ : 함수의 기울기가 w 방향으로 얼만큼 변했는지
- $db =\frac{\partial J(w,b)}{\partial b}$ : 함수의 기울기가 b 방향으로 얼만큼 변했는지

## 5. Derivatives (도함수)

- 어떤 함수 a 도함수 = 직선의 기울기
- $\frac{d}{da}f(a)$
- 기울기 혹은 도함수는 함수의 위치에 따라 달라질 수 있음

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2017.png)

## 6. Computation Graph (계산 그래프)

### 1) $J(a,b,c) =3(a+bc)$의 계산 그래프 만드는 과정

- u = bc
- v = a+u
- J = av

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2018.png)

- 위와 같이 **순방향** 방식은 **함숫값**을 구하는데 사용하며, **역방향**의 방식은 **미분값**을 구하는데 사용

### 2) 계산 그래프의 미분

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2019.png)

- 도함수의 계산은 왼쪽에서 오른쪽으로 진행
- $\frac{dJ}{dv} =?=3$
- $\frac{dJ}{da} =3=\frac{dJ}{dv}\frac{dv}{da}$=3*1
- $\frac{dv}{da} =1$

## 7. 로지스틱 회귀 경사 하강법

### 1) 로지스틱 회귀 정리

- $z = w^Tx+b$
- $\hat{y}=a=\sigma(z)$
- **$L(a,y)=-(yloga +  (1-y)log(1-a))$**

### 2) 로지스틱 회귀 미분

- $z = w_1x_1+w_2x_2+b$ **→** $a=\sigma(z)$ **→** $L(a,y)$
1. $L(a,y)$ →  $a=\sigma(z)$ 
    - $da=\frac{dL(a,y)}{da}=-\frac{y}{a}+\frac{1-y}{1-a}$
2. $a=\sigma(z)$ → $z = w_1x_1+w_2x_2+b$ 
    - $dz=\frac{dL(a,y)}{dz}=a-y=\frac{dL}{da}\frac{da}{dz}$

### 3) m개 샘플의 경사 하강법

- 비용 함수 **$J(w,b) =\frac{1}{m}\sum^{i=m}_{i=1}L(a^{(i)},y^{(i)})$**
    
    $a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)})=\sigma(w^Tx^{(i)}+b)$
    

![로지스틱 회귀에서의 비용함수 코드화](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2020.png)

로지스틱 회귀에서의 비용함수 코드화

- 현재 코드에서는 n=2지만, 특성의 개수가 많아진다면 for문 처리를 해야함
- 그렇게 된다면 이중 for문으로 나타나게 되고 이로인해 계산 속도가 느려짐

---

# 3. Python and Vectorization(파이썬 벡터화)

## 1. Vectorization(벡터화)

### 1) 개념 및 예제

- 코딩에서 명시적 for문을 제거하는 기술

![벡터화 VS 비벡터화](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2021.png)

벡터화 VS 비벡터화

- 벡터화인 경우가 비벡터화인 경우보다 계산 속도가 빠르다

```python
import numpy as np

a = np.array([1,2,3,4])
print(a) # [1 2 3 4]
```

```python
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c) # 250170.62904592504
print('Vectorized version : ' + str(1000*(toc-tic)) +'ms') 
# Vectorized version : 1.4641284942626953ms

c= 0
tic = time.time()
for i in range(1000000):
  c += a[i]*b[i]
toc = time.time()

print(c) # 250170.62904593535
print('For loop : '+str(1000*(toc-tic))+'ms')
# For loop : 530.8642387390137ms
```

- np.dot(a,b)는 a,b를 벡터화 해주는 역할
- code 실습을 통해 비교해봐도 벡터화 버전이 훨씬 빠르다
- 이렇게 **벡터의 형태로 계산하게 되면 병렬적 연산**을 하게 되는데 이러한 경우 **CPU보다 GPU가 더 유리**
- 컴퓨터의 계산 효율성을 위해 가능하면 ‘for loop’를 피하는것이 좋음

![벡터화 VS 비벡터화 예제2 ](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2022.png)

벡터화 VS 비벡터화 예제2 

![벡터화 VS 비벡터화 예제2](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2023.png)

벡터화 VS 비벡터화 예제2

```python
# non-vectorization
u = np.zeros((n,1))
for i in range(n):
  u[i] = math.exp(v[i])

# vectorization
import numpy as np
u = np.exp(v)
```

- np.zeros((n,1)) → 0으로 이루어진 n*1 행렬 생성
- np.exp(v) → 요소별 지수 계산
- np.log(v) → 요소별 로그값 계산
- np.abs(v) → 절대값 계산
- np.maximum(v,0) → 모든 요소에 대한 최대값을 0으로 설정
- v**2 → 제곱 계산
- 1/v → 역수

### 2) 로지스틱 회귀 벡터화

![로지스틱 회귀 벡터화](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2024.png)

로지스틱 회귀 벡터화

- 벡터화 적용을 통해 for문 하나만 남은 상태

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2025.png)

### 3) 로지스틱 회귀 경사하강법

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2026.png)

![로지스틱 회귀 경사 하강 벡터화 완성](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2027.png)

로지스틱 회귀 경사 하강 벡터화 완성

## 2. Broadcasting in Python (파이썬 브로드 캐스팅)

- Numpy 배열에서 **차원의 크기가 서로 다른** 배열에서도 **산술 연산을 가능하게 하는 원리**
    - 두 배열의 차원(ndim)이 같지 않다면 차원이 더 **높은 배열과 같은 차원의 배열로 인식**
    - 차원의 크기가 1일때 가능

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2028.png)

```python
import numpy as np

A = np.array([[56.0, 0.0, 4.4, 68.0],
             [1.2, 104.0, 52.0, 8.0],
             [1.8, 135.0, 99.0, 0.9]])

print(A)
'''
[[ 56.    0.    4.4  68. ]
 [  1.2 104.   52.    8. ]
 [  1.8 135.   99.    0.9]]
'''

cal = A.sum(axis=0)
print(cal)
# [ 59.  239.  155.4  76.9]

cal.reshape(1,4) # reshape를 활용한 행렬화
# array([[ 59. , 239. , 155.4,  76.9]])

percentage = 100*A/cal.reshape(1,4)
print(percentage)
'''
[[94.91525424  0.          2.83140283 88.42652796]
 [ 2.03389831 43.51464435 33.46203346 10.40312094]
 [ 3.05084746 56.48535565 63.70656371  1.17035111]]
'''
```

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2029.png)

```python
A = np.array([1,2,3,4])
A+100
# array([101, 102, 103, 104])
```

![Untitled](Neural%20Networks%20%E1%84%89%E1%85%A9%E1%84%80%E1%85%A2%20%E1%84%86%E1%85%B5%E1%86%BE%20%E1%84%85%E1%85%A9%E1%84%8C%E1%85%B5%E1%84%89%E1%85%B3%E1%84%90%E1%85%B5%E1%86%A8%20%E1%84%92%E1%85%AC%E1%84%80%E1%85%B1%2088ffdb194cb74d8794a8e7f0944748b8/Untitled%2030.png)

```python
A = np.array([[1,2,3],[4,5,6]])
B = np.array([100,200,300])
A+B
'''
array([[101, 202, 303],
       [104, 205, 306]])
'''
```

```python
# rank 1 array (Don't use)
import numpy as np
a = np.random.randn(5)
print(a)
# [-0.93997085  1.37828891  0.23178588 -0.97728334 -1.68825958]
print(a.shape)
# (5,)
print(a.T)
# [-0.93997085  1.37828891  0.23178588 -0.97728334 -1.68825958]
print(np.dot(a,a.T))
# 6.642253350337608
```

```python
# array
import numpy as np
a = np.random.randn(5,1) # 열 벡터
# a = np.random.randn(1,5) # 행 벡터
print(a)
'''
[[-1.21270946]
 [ 1.24917393]
 [-1.55620123]
 [-0.85252435]
 [ 0.31968055]]
'''
print(a.T)
# [[-1.21270946  1.24917393 -1.55620123 -0.85252435  0.31968055]]
print(np.dot(a,a.T))
'''
[[ 1.47066423 -1.51488504  1.88721995  1.03386434 -0.38767962]
 [-1.51488504  1.5604355  -1.943966   -1.06495119  0.3993366 ]
 [ 1.88721995 -1.943966    2.42176227  1.32669944 -0.49748726]
 [ 1.03386434 -1.06495119  1.32669944  0.72679776 -0.27253545]
 [-0.38767962  0.3993366  -0.49748726 -0.27253545  0.10219565]]
'''
```

## 3. Quiz

- 문제 풀이 및 실습
    
    ### 1) Consider the following code snippet:
    
    a.shape = (3,4)
    
    b.shape = (4,1)
    
    for i in range(3):
    for j in range(4):
    c[i][j] = a[i][j] + b[j]
    
    How do you vectorize this? → c = a + b.T
    
    ### 2) sigmoid 함수 코드 → $sigmoid(x) = \frac{1}{1+e^{-x}}$
    
    ```python
    import math
    from public_tests import *
    
    # GRADED FUNCTION: basic_sigmoid
    
    def basic_sigmoid(x):
        """
        Compute sigmoid of x.
    
        Arguments:
        x -- A scalar
    
        Return:
        s -- sigmoid(x)
        """
        # (≈ 1 line of code)
        # s = 
        # YOUR CODE STARTS HERE
        s = 1/(1+math.exp(-x))
    		# s = 1/(1+np.exp(-x)) 
        
        # YOUR CODE ENDS HERE
        
        return s
    ```
    
    ```python
    # GRADED FUNCTION: sigmoid_derivative
    
    def sigmoid_derivative(x):
        """
        Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
        You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
        
        Arguments:
        x -- A scalar or numpy array
    
        Return:
        ds -- Your computed gradient.
        """
        
        #(≈ 2 lines of code)
        # s = 
        # ds = 
        # YOUR CODE STARTS HERE
        s = 1/(1+np.exp(-x))
        ds = s*(1-s)
        
        # YOUR CODE ENDS HERE
        
        return ds
    ```