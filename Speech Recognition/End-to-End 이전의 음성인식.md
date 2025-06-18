# 과거 음성인식 시스템의 발전과 한계
<img width="1019" alt="image" src="https://github.com/user-attachments/assets/63a14901-fa96-441d-82cb-78e10d7b2224">

![image](https://github.com/user-attachments/assets/d6f01631-7fd1-4051-9d13-10486e305fcd)

![IMG_8E96AF62D4FD-1](https://github.com/user-attachments/assets/0ae8391f-8cc8-4e97-a661-47b130a3b997)


## 1) HMM 모델
![image](https://github.com/user-attachments/assets/ba275934-bb00-4283-8134-52f57f4ba13c)
![image](https://github.com/user-attachments/assets/ab5e010e-5ead-4475-a97d-d5c4f205cf76)

### Markov model
- 마르코프 모델은 state로 이루어진 Sequence를 상태 전이 확률 행렬로 표현하는 것
- **Markov 가정** : 시간 **t에서 관측은 가장 최근 r개의 관측에만 의존**한다는 가정
  - 한 상태에서 다른 상태로의 전이는 이전 상태의 긴 이력을 필요치 않다는 가정
  - $\[P(S_{t} | S_{t-1}, S_{t-2}, \dots, S_{1}) = P(S_{t} | S_{t-1})\]$
- **Hidden Markov Model** : 관측이 불가능한 process를 관측이 가능한 다른 process로 추정하는 이중 확률처리 모델
  - 관측이 가능한 Observable과 관측이 불가능한 Hidden state 구분하기
  
### 개념
  - **확률적으로 상태간 전이가 일어나는 시퀀스 데이터를 모델링한 알고리즘, 현재 상태는 이전 상태에만 의존한다는 Markov 가정을 따름**
  - 주어진 관측 데이터에서, 가장 가능성 높은 hidden state 시퀀스를 추론하는것이 목표
  - Hidden state가 존재하고, 시퀀스 데이터의 확률적 패턴을 학습해야 하는 경우에 사용됨
  ![image](https://github.com/user-attachments/assets/4a259f9b-a70d-49ee-a44e-9f60316343df)

### 주요 구성 요소(Parameters)
![IMG_44B586691076-1](https://github.com/user-attachments/assets/1be6fc07-cc33-4a55-a10f-be3ed67d7d23)


### HMM 알고리즘 - 동작과정
- **Forward Algorithm**
  - 주어진 모델에서 관측된 시퀀스가 나올 확률을 계산
  - HMM의 상태와 전이 확률을 기반으로, 각 시간 단계에서 관측된 데이터가 발생할 확률을 계산
  - 초기 상태에서 시작하여 각 상태에 대한 확률을 누적하여 최종적인 관측 시퀀스의 확률을 얻음
  - 특정 음성이 주어졌을 때, 그 음성이 특정 상태(예: 음소)에서 나올 확률을 계산하고자 할 때 사용
- **Viterbi Algorithm**
  - 가장 가능성 있는 숨겨진 상태 시퀀스를 찾는 알고리즘, 실제 음소들이 발음된 시퀀스를 추정하는 데 사용
  - 주어진 관측 시퀀스에 대해, 각 시간 단계에서 가장 높은 확률을 가지는 상태를 선택(동적 프로그래밍 사용)
  - 최종적으로, 선택된 상태 시퀀스가 가장 높은 확률을 가진 숨겨진 상태 경로를 제공
  - 실제 발음된 음소의 시퀀스를 추정하고자 할 때, 즉 어떤 단어가 발음되었는지 알아내고자 할 때 사용
- **Baum-Welch Algorithm**
  - 모델의 파라미터를 학습하는 데 사용되는 알고리즘으로, EM(Expectation-Maximization) 알고리즘의 한 형태
  - HMM의 파라미터(전이 확률, 방출 확률 등)를 학습
  - EM(Expectation-Maximization) 알고리즘을 기반으로 하며, 관측된 데이터와 모델의 파라미터를 반복적으로 업데이트하여 최적의 파라미터를 찾음
  - E 단계에서 현재 파라미터를 사용하여 각 상태의 확률 분포를 계산하고, M 단계에서 이러한 확률을 기반으로 모델의 파라미터를 업데이트


### HMM의 한계
- 입력과 출력을 수동으로 정렬해야해 함
- 마르코프 가정을 따르기에 멀리 떨어진 과거 정보 반영 어려움
- Forward 알고리즘이 순차적으로 실행되기에 병렬 연산 어려움



## 2) GMM 모델
![IMG_45FE64BB363C-1](https://github.com/user-attachments/assets/04c34685-12d6-4302-9937-720f0372381e)


## 3) HMM-GMM 모델
![IMG_273AE024D4CF-1](https://github.com/user-attachments/assets/02facddd-8abc-4842-b75e-da12b11b5772)

  
