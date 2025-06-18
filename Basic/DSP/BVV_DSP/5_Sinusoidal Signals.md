- Sinusoids
- Continuous and discrete-time frequency
- Examples of sinusoids
- Non-uniqueness of discrete-time sinusoids

## 1. Sinusoids(정현파)
- 일반적으로 코사인(cosine) 또는 사인(sine)신호를 정현파 신호, 정현파라고 함
- 자연에서 흔히 나타나는 현상
  - 빛의 특정 색깔 (Light of a given color) : 특정 파장의 빛은 사인파로 표현
  - 마이크로파 (Microwaves) : 전자기파는 사인파 형태로 전파
  - 진동 운동 (Oscillatory Motion) : 진자 운동, 감쇠되지 않은 스프링-질량 시스템 등에서 발생
- 정현파의 합으로 모든 신호를 표현 가능
  - Fourier Series에 따르면, 임의의 신호는 다양한 주파수와 진폭의 사인파들의 합으로 표현될 수 있음
  - 이를 통해 복잡한 신호를 단순화하고 분석하기 쉽게 만든다
- 주파수 개념이 우리의 경험에 깊이 관여
  - 주파수는 사운드, 라디오 신호, 빛의 색깔 등에서 직관적으로 이해할 수 있는 개념
  - 신호의 성격을 이해하고 분석하는 데 중요한 도구

## 2. Continuous-Time Sinusoids
![image](https://github.com/user-attachments/assets/c108fae1-25bc-42c9-a13e-b632c3a7f333)

- $\[x(t) = A \cos(\omega_0 t + \phi)\]$
  - $\( A \)$ : 진폭 (Amplitude)
    - 신호의 최고점과 최저점 간의 차이를 정의
    - 그래프에서 정현파의 높이 결정
  - $\( \omega_0 \)$ : 각주파수 (Angular Frequency)
    - 단위 시간당 신호의 변화를 나타냄
    - 주기 $\( T_0 \)$ 와 반비례 관계
      - $\[T_0 = \frac{2\pi}{\omega_0}\]$
      - $\( T_0 \)$ : 신호가 한 번 반복되는 데 걸리는 시간
  - $\( \phi \)$ : 위상 (Phase Shift)
    - 신호의 시작 위치를 나타냄
    - t=0에서의 신호 값을 기준으로 설정

## 3. Discrete-Time Sinusoids
![image](https://github.com/user-attachments/assets/2608290a-24e4-4e5d-9162-63eff10b16e9)

- $\[x[n] = A \cos(\Omega_0 n + \phi)\]$
  - $\( A \)$ : 진폭 (Amplitude)  
  - $\( \Omega_0 \)$ : 이산 각주파수 (Discrete Angular Frequency)
    - 샘플 간 신호 변화 속도를 정의
    - 주기 $\( N \)$와 반비례 관계
      - $\[N = \frac{2\pi}{\Omega_0}\]$
      - $\( N \)$ : 신호가 한 번 반복되기 위해 필요한 샘플 수
  - $\( \phi \)$ : 위상 (Phase Shift)

## 4. Frequency
- **정의**
  - Frequency = 1 / Period (주파수 = 1 / 주기)
  - 주파수는 신호의 반복 속도를 나타냄
  - 주기는 신호가 한 번 반복되는 데 걸리는 시간(CT) 또는 샘플 개수(DT)를 의미
- **CT (Continuous-Time, 연속 시간 신호)**
  - $\[f_0 = \frac{1}{T_0} \quad \text{(cycles/sec or Hz)}\]$
  - $\[\omega_0 = 2\pi f_0 = \frac{2\pi}{T_0} \quad \text{(rads/sec)}\]$ 
  - 주기 $\( T_0 \)$는 초(Seconds)로 측정
  - 주파수는 **라디안/초(rads/sec)** 또는 **주기/초(cycles/sec, Hz)** 로 표현
  - 연속 시간 신호에서는 주기 $\( T_0 \)$에 따라 $\( f_0 \)$와 $\( \omega_0 \)$를 계산
- **DT (Discrete-Time, 이산 시간 신호**
  - $\[F_0 = \frac{1}{N} \quad \text{(cycles/sample)}\]$
  - $\[\Omega_0 = 2\pi F_0 = \frac{2\pi}{N} \quad \text{(rads/sample or rads)}\]$ 
  - 주기 $\( N \)$는 샘플(Samples)로 측정
  - 주파수는 **라디안/샘플(rads/sample)** 또는 **주기/샘플(cycles/sample)**로 표현
  - 이산 시간 신호에서는 샘플 주기 $\( N \)$에 따라 $\( F_0 \)$와 $\( \Omega_0 \)$를 계산
- $\( 2\pi \)$는 주파수를 라디안 단위로 변환하는 데 사용

## 5. Relating Continuous and Discrete-Time Frequency
![image](https://github.com/user-attachments/assets/7c7aae38-c9e1-488e-8b6a-212f3d8bb342)

- 연속 시간 사인파의 샘플이 이산 시간 사인파와 같아지도록 관계를 정의
  - $\[x[n] = x(t)\big|_{t = nT}\]$
  - $\[x[n] = A \cos(\Omega n + \phi) = A \cos(\omega n T + \phi) = A \cos((\omega T) n + \phi)\]$
    - $\( \Omega = \omega T \)$
    - $\(\Omega\)$ : 이산 각주파수 (rads/sample)
    - $\(\omega\)$ : 연속 각주파수 (rads/sec)
    - $\( T \)$ : 샘플링 간격 (Sampling Interval, sec/sample)
- 연속 시간 주파수에서 이산 시간 주파수로 변환
  - $\[\Omega = 2\pi fT\]$
  - $\( f \)$ : 연속 시간 주파수 (cycles/sec or Hz)

### Example
- Musical Notes (음악적 음표)
  - 옥타브당 12개의 음표 또는 주파수의 두 배 (12 notes per octave or doubling of frequency)
    - 한 옥타브 내에는 12개의 음이 있으며, 옥타브가 올라갈수록 주파수는 두 배
  - 주파수는 $\( 2^{1/12} \)$의 배수로 간격이 나뉨
    - 이는 균등한 12음계(equal temperament)를 형성
  - C-major scale (C 메이저 스케일)
    - C: 523.28 Hz
    - D: 587.36 Hz
    - E: 659.28 Hz
    - F: 698.48 Hz
    - G: 784 Hz
    - A: 880 Hz
    - B: 987.84 Hz
- Sum of Sinusoids with Different Frequencies
  - 여러 주파수의 정현파를 합성하여 특정 음을 표현
  - ex) 색소폰으로 연주한 A 음표는 다양한 주파수의 사인파 합으로 나타낼 수 있음

## 6. Uniqueness (고유성)
- Continuous-Time (연속 시간 신호)
  - 서로 다른 주파수를 가진 연속 시간 사인파는 서로 다른 신호를 생성
  - $\[\text{If } \omega_1 \neq \omega_2, \text{ then } \sin(\omega_1 t) \neq \sin(\omega_2 t)$
  - 연속 시간 신호에서는 서로 다른 주파수는 항상 고유한 신호를 나타냄
- Discrete-Time (이산 시간 신호)
  - 이산 시간 신호에서 주파수가 $\( m2\pi \)$만큼 이동하면 동일한 신호가 됨
    - 이는 **주파수 중복(Aliasing)** 과 관련있으며, 주파수 표현이 제한적임을 보여줌
  - $\[\text{If } \Omega_2 = \Omega_1 + m2\pi, \text{ then } \sin(\Omega_1 n) = \sin(\Omega_2 n)\]$
- $\[\sin((\Omega_1 + m2\pi)n) = \sin(\Omega_1 n)\cos(m2\pi n) + \cos(\Omega_1 n)\sin(m2\pi n)\]$
  
