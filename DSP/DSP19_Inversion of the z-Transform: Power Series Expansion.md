
# Z-변환은 멱급수 (Power Series)
![image](https://github.com/user-attachments/assets/59cdcd46-c194-47f8-804f-c2b639b09db4)

## Z-변환 정의
- Z-변환 $\(X(z)\)$은 멱급수 형태로 표현됩니다
  - $\[  X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}  \]$
  - 이를 멱급수로 전개하면: $\[  X(z) = \cdots + x[-2]z^{2} + x[-1]z^{1} + x[0] + x[1]z^{-1} + x[2]z^{-2} + \cdots\]$

## Z-변환 역변환 절차
1. **함수를 멱급수 형태로 변환**
   - $\(X(z)\)$을 $z^{-n}$의 멱급수로 표현합니다
2. **계수 확인**
   - $\(x[n]\)$은 $z^{-n}$의 계수로 결정됩니다

## 예제
- $\[  X(z) = 2z^{5} + z^{3} - z^{2} + 1 + 3z^{-1} - 4z^{-4}  \]$
- 이때
  - $\(x[5] = 2\)$
  - $\(x[3] = 1\)$
  - $\(x[2] = -1\)$
  - $\(x[0] = 1\)$
  - $\(x[-1] = 3\)$
  - $\(x[-4] = -4\)$

- 따라서, 신호 $\(x[n]\)$은 다음과 같이 표현됩니다
  - $\[  x[n] = 2\delta[n+5] + \delta[n+3] - \delta[n+2] + \delta[n] + 3\delta[n-1] - 4\delta[n-4]  \]$

## 요약
1. Z-변환은 멱급수로 표현되며, 각 항의 계수는 신호 $\(x[n]\)$의 값과 직접적으로 연관됩니다
2. 역변환은 멱급수 전개를 통해 각 $\(z^{-n}\)$의 계수를 확인하여 신호를 복원합니다

# 멱급수 전개를 통한 초월 함수의 역변환
![image](https://github.com/user-attachments/assets/c92cd209-d01e-4f8c-9ffb-b6444d79738e)

## 멱급수 전개를 이용한 Z-변환 역변환
- Z-변환 $\(X(z)\)$이 초월 함수(Transcendental Function)로 표현되는 경우, 멱급수 전개를 사용하여 역변환을 수행할 수 있습니다.

## 예제
- $\[  X(z) = \exp\left(-2z^{-1}\right)  \]$

- **지수 함수의 멱급수 표현**:  $\[ \exp(x) = \sum_{n=0}^{\infty} \frac{x^n}{n!} \]$

- $X(z)$의 멱급수 전개:  $\[ X(z) = \sum_{n=0}^{\infty} \frac{(-2z^{-1})^n}{n!} =  \sum_{n=0}^{\infty} \frac{(-2)^n}{n!} z^{-n} \]$

- 신호 $x[n]$은 $z^{-n}$의 계수로 결정됩니다:  $\[ x[n] = \frac{(-2)^n}{n!} u[n] \]$


## 결과
- $\[ x[n] = \frac{(-2)^n}{n!} u[n] \]$

## 요약
1. 초월 함수로 표현된 Z-변환의 역변환은 멱급수 전개를 통해 수행됩니다
2. 지수 함수의 멱급수 표현을 이용해 각 항의 계수를 계산하여 $x[n]$을 구합니다
3. 이 방법은 초월 함수를 포함하는 복잡한 Z-변환을 신호로 복원하는 데 유용합니다


# 장제법(Long Division)을 통한 유리 함수 Z-변환의 역변환
![image](https://github.com/user-attachments/assets/971010c4-9c2c-4196-8f4e-7cb2a10d22b6)

## Z-변환 역변환
- 유리 함수 형태의 $X(z)$는 장제법(Long Division)을 사용하여 멱급수로 전개한 뒤 역변환할 수 있습니다.

---

## 예제
- $\[ X(z) = \frac{1 - z^{-1}}{1 - \frac{1}{2}z^{-1}}, \, |z| > \frac{1}{2} \]$

- **장제법을 사용한 멱급수 전개**:  
  $\[ X(z) = 1 - \frac{1}{2}z^{-1} - \frac{1}{4}z^{-2} - \frac{1}{8}z^{-3} - \frac{1}{16}z^{-4} + \cdots = \sum_{n=0}^{\infty} \left(-\frac{1}{2}\right)^n z^{-n}. \]$

- $x[n]$의 값은 다음과 같습니다:
  - $n < 0$인 경우: $x[n] = 0$
  - $n = 0$인 경우: $x[n] = 1$
  - $n = 1$인 경우: $x[n] = -\frac{1}{2}$
  - $n = 2$인 경우: $x[n] = -\frac{1}{4}$
  - $n = 3$인 경우: $x[n] = -\frac{1}{8}$
  - $n = 4$인 경우: $x[n] = -\frac{1}{16}$
  - 기타: $x[n] = \vdots$
  
- 최종 결과: $\[ x[n] = \delta[n] - \left(\frac{1}{2}\right)^n u[n] \]$

## 요약
1. 장제법(Long Division)을 사용하여 $X(z)$를 멱급수로 전개합니다
2. 전개된 멱급수에서 각 항의 계수를 이용해 $x[n]$을 계산합니다
3. 유리 함수 형태의 Z-변환은 이러한 방법을 통해 간단히 역변환할 수 있습니다
