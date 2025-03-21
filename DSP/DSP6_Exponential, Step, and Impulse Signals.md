- Exponentials (지수 함수)
  - 신호의 decay 또는 growth 표현
  - Exponentially damped sinusoids
  - Example - Guitar
- Steps (계단 함수)
  - 급격한 변화 또는 스위친(switch) 표현 
- Impulses (임펄스 함수)
  - 작은 섭동(perturbation) 또는 시스템 테스트 표현

## 1. Exponentials(지수 함수)
### 1) Continuous-Time Exponentials
![image](https://github.com/user-attachments/assets/c6b30303-917a-4b82-941d-419574ae7495)

- $\[x(t) = A e^{-bt} \quad \text{또는} \quad x(t) = A \exp(-bt)\]$
  - A : 초기 값 (Initial Value)
  - b : 지수 함수의 속도 결정 계수 (Rate Constant)
    - b > 0 : 감쇠(Decay)
      - t가 증가함에 따라 x(t)는 점점 감소
      - 초기 값 A에서 시작하여 0으로 수렴
      - ex) 충격 후 에너지 소실, RC 회로 방전 
    - b < 0 : 성장(Growth)
      - t가 증가함에 따라 x(t)는 점점 증가
      - 초기 값 A에서 시작하여 무한대로 발산
      - ex) 금융의 복리 성장, 감염병 확산

### 2) Discrete-Time Exponentials
![image](https://github.com/user-attachments/assets/a106e9b5-127e-4f2b-84ea-a56deb162b64)

- $\[x[n] = A e^{-bn} \quad \text{또는} \quad x[n] = A \alpha^n, \quad \text{where } \alpha = e^{-b}\]$
  - A : 초기 값 (Initial Value)
  - b : 감쇠 또는 성장 계수 (Decay or Growth Rate)
  - $\( \alpha \)$: $\(\alpha = e^{-b}\)$, 성장/감쇠를 조정하는 값
    - $\( 0 < \alpha < 1 \)$ : decay
      - n이 증가함에 따라 x[n]의 크기는 점점 줄어듬
      - 초기 값에서 시작해서 0으로 수렴 
    - $\( \alpha > 1 \)$ : growth
      - n이 증가함에 따라 x[n]의 크기는 점점 커짐
      - 초기 값에서 시작하여 무한대로 발산 

### 3) Exponentially Damped Sinusoids (지수적으로 감쇠된 정현파)
![image](https://github.com/user-attachments/assets/b9b3016a-fe7a-4990-bcd2-544403a8d341)

- Continuous-Time Exponentially Damped Sinusoids
  - $\[x(t) = A e^{-bt} \cos(\omega_0 t)\]$
    - A : 초기 진폭 (Initial Amplitude)
    - b > 0 : 감쇠 계수 (Damping Factor)
      - 시간 t가 증가할수록 진폭이 점점 줄어듬 
    - $\( \omega_0 \)$ : 각주파수 (Angular Frequency)
- Discrete-Time Exponentially Damped Sinusoids
  - $\[x[n] = A \alpha^n \cos(\Omega_0 n), \quad \text{where } 0 < \alpha < 1\]$
    - $\( \alpha = e^{-b} \)$ : 이산 시간 감쇠 계수
      - $\( 0 < \alpha < 1 \)$ - 샘플 간 감쇠가 발생하며, n이 증가할수록 진폭이 감소
      - 이산 샘플들이 점점 축으로 가까워지며 진폭이 줄어듬
    - $\( \Omega_0 \)$ : 이산 각주파수 (Discrete Angular Frequency)

## 2. Steps
![image](https://github.com/user-attachments/assets/1e496e99-00a2-4d04-b81b-5a84dab53708)

- **이산 시간 계단 함수** 
  - $\( s[n] \)$ : 시간 n 에 대한 이산 시간 신호
  - $\( n < 0 \)$ : 값이 0 으로 유지
  - $\( n \geq 0 \)$ : 값이 1 로 변함
  - 점으로 표현된 이산 신호

- **연속 시간 계단 함수** 
  - $\( s(t) \)$ : 시간 t 에 대한 연속 시간 신호
   - $\( t < 0 \)$ : 값이 0 으로 유지
   - $\( t \geq 0 \)$ : 값이 1 로 변함
   - 선으로 표현된 연속 신호

## 3. Impulses
![image](https://github.com/user-attachments/assets/89d93e45-6484-4c2a-ae64-10f7b1fb49c6)


## 1) Continuous-Time Impulses
![image](https://github.com/user-attachments/assets/4c203cd5-ff7b-4f49-8bcd-8f48a23d72bd)

- 연속 시간 임펄스 함수 $\( \delta(t) \)$는 **일반화된 함수 (Generalized Function)** 로 취급
- 실제로 $\( \delta(t) \)$는 $\(\Delta \to 0\)$으로 수렴하는 특정 함수 $\( f_\Delta(t) \)$의 극한으로 정의
- $\( \delta(t) \)$는 **적분 내에서만 의미**를 가지며, 단독으로는 정의되지 않음
  - 이를 통해 특정 시점의 신호를 모델링할 수 있음
- 시프팅 성질 (Sifting Property)
  - 임펄스 함수의 가장 중요한 성질로, 특정 시점 $\( t_0 \)$에서의 함수 값을 추출
  - 임펄스 함수는 특정 시점에서의 함수 값을 "골라내는" 역할
