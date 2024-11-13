# 푸리에 변환(Fourier transform)
![image](https://github.com/user-attachments/assets/bc4e8fc3-46d3-4248-a2a6-00a2b4d66706)

![image](https://github.com/user-attachments/assets/e5b6cc82-92b7-4adc-915b-1430a2763f3f)

- 임의의 입력 신호를 다양한 주파수를 갖는 주기함수(복수 지수함수)들의 합으로 분해하여 표현하는 것
- 비주기적인 신호(또는 무한히 긴 주기를 가진 신호)를 주파수 성분으로 분석하기 위한 기법
  - 푸리에 급수는 주기적인 신호를 대상으로, 비주기적 신호는 푸리에 변환 
- 그리고 각 주기함수들의 진폭과 위상을 구하는 과정
  - 주기(period) : 파동이 한번 진동하는데 걸리는 시간 또는 그 길이
  - 주파수(frequency) : 1초동안의 진동횟수  

- 푸리에 급수 : $$y(t)=\sum_{k=-\infty}^\infty A_k \, \exp \left( i\cdot 2\pi\frac{k}{T} t \right)$$
  - 주기가 T인 신호를 기본 주파수 $\frac{1}{T}$의 정수배 주파수 성분으로 분해
  - 주파수 성분이 이산적
  - 주파수에서의 신호 성분을 나타내는 복소수 값 : y(t) 
  - 주기 함수들의 수 : $k$는 $-\infty ~ \infty$의 범위
  - 사인함수의 진폭 : $A_k$
  - 회전하는 속도 : $\exp()$ 괄호 안의 값

- 진폭 수식 : $$A_k = \frac{1}{T} \int_{-\frac{T}{2}}^\frac{T}{2} f(t) \ \exp \left( -i\cdot 2\pi \frac{k}{T} t \right)dt$$
  - 각 주파수의 성분이 신호에서 어느 정도의 크기로 기여하는지 나타내는 것
  - 시계방향으로 돌기 때문에 $\exp()$ 괄호 안의 값이 **마이너스**
  - 원형으로 회전하는 복소수 $\exp()$에 f(t)를 곱해 커졌다, 작아졌다를 반복 -> 원에 그래프가 감긴 형태 생성
  - $\frac{1}{T}$을 통해 복소수로서 모두 더한 값을 점의 개수로 나눔 (평균을 구하는것)
  - 주기 함수의 합으로 표현된다고 했는데, 왜 지수함수 형태로 작성되어 있나?
    - 오일러 공식 : $$e^{i\theta} = \cos{\theta} + i\sin{\theta}$$
      - 다른 표현 : $$\exp \left( i\cdot 2\pi\frac{k}{T} t \right) = \cos\left({2\pi\frac{k}{T}}\right) + i\sin\left({2\pi\frac{k}{T}}\right)$$
      - 사인 함수와 코사인 함수를 각각 다루지 않고, 하나의 복소수로 통합(cos:복소수의 실수부, sin:복소수의 허수부)
      - 사인 함수와 코사인 함수는 각도에 따라 독립적으로 변하므로, 방정식이나 계산을 수행할 때 각 함수별 따로 처리해야함
        - 예를 들어, 미분이나 적분을 할 때 따로따로 계산해야 하므로 복잡도 상승
      - 복소 지수 함수를 사용하므로서 미분과 적분이 단순해짐
        - 복소 지수 함수의 미분은 지수가 곱해지기만 하면됨
        - $\[\frac{d}{dt} e^{i \omega t} = i \omega e^{i \omega t}\]$
        - $\[\frac{d}{dt} \cos(\omega t) = -\omega \sin(\omega t)\]$, $\[\frac{d}{dt} \sin(\omega t) = \omega \cos(\omega t)\]$
        - 적분은 지수 형태 그대로 남음
        - $\[\int e^{i \omega t}dt = \frac{e^{i \omega t}}{i \omega}\]$
        - $\[\int \cos(\omega t)dt = \frac{\sin(\omega t)}{\omega}\]$, $\[\int \sin(\omega t)dt = -\frac{\cos(\omega t)}{\omega}\]$
      
- 푸리에 변환 : $\[X(f) = \int_{-\infty}^{\infty} f(t) e^{-i 2 \pi f t} \, dt\]$
  - 주파수 $\( f \)$에서의 신호 성분 : $\( X(f) \)$
  - 시간 영역의 신호 : $\( f(t) \)$
  - 주파수 $\( f \)$에 해당하는 복소 지수 함수 : $\( e^{-i 2 \pi f t} \)$

- 진폭 수식 : $\[X(f) = \int_{-\infty}^{\infty} f(t) e^{-i 2 \pi f t}dt\]$

