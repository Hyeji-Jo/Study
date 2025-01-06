# 극(Poles)과 영점(Zeros)

![image](https://github.com/user-attachments/assets/94959988-be5a-43c4-974c-e5a45261f509)

## 1. 개요
- **Z-변환**에서 가장 유용하고 중요한 형태는 **유리 함수(Rational Functions)** 입니다: $\[X(z) = \frac{P(z)}{Q(z)}\]$
  - \(P(z)\), \(Q(z)\): \(z\)에 대한 다항식

## 2. 극(Poles)와 영점(Zeros)의 정의
1. **영점(Zeros)**
   - \(X(z) = 0\)이 되는 \(z\)의 값들.
   - 다항식 \(P(z)\)의 근(roots)이 영점에 해당:$\[P(z) = 0 \quad \Rightarrow \quad z \text{ is a zero.}\]$

2. **극(Poles)**
   - $\(X(z) \to \infty\)$가 되는 \(z\)의 값들
   - 다항식 \(Q(z)\)의 근(roots)이 극에 해당:$\[Q(z) = 0 \quad \Rightarrow \quad z \text{ is a pole.}\]$

## 3. 극과 영점의 추가 조건
- **$\(z = \infty\)$에서의 극과 영점**
  - $\(z \to \infty\)$에서의 극 또는 영점 여부는 \(P(z)\)와 \(Q(z)\)의 차수에 의해 결정됩니다
  - $\[\text{차수 관계: } \text{Order of } Q(z) \neq \text{Order of } P(z).\]$


# 예제: 극(Poles)과 영점(Zeros)의 분석

![image](https://github.com/user-attachments/assets/7c7e8e0f-a674-4de9-918c-219b135dbbd8)

## 1. 사례 1: $\(x[n] = \alpha^n u[n]\)$
- **Z-변환**:$\[X(z) = \frac{z}{z - \alpha}, \quad \text{ROC: } |z| > |\alpha|\]$

- **특성**
  - **영점(Zero)**:$\[z = 0\]$
  - **극(Pole)**:$\[z = \alpha\]$

- **Z-평면 시각화**
  - 영점 \(z = 0\)에서 표시 (●)
  - 극 $\(z = \alpha\)$에서 표시 (×)
  - ROC는 $\( |z| > |\alpha| \)$인 영역


## 2. 사례 2: $\(x[n] = -\alpha^n u[-n-1]\)$
- **Z-변환**:$\[X(z) = \frac{z}{z - \alpha}, \quad \text{ROC: } |z| < |\alpha|\]$

- **특성**
  - **영점(Zero)**:$\[z = 0\]$
  - **극(Pole)**:$\[z = \alpha\]$

- **Z-평면 시각화**
  - 영점 \(z = 0\)에서 표시 (●)
  - 극 $\(z = \alpha\)$에서 표시 (×)
  - ROC는 $\( |z| < |\alpha| \)$인 영역

## 3. ROC에 따른 차이점
- **사례 1**: $\( |z| > |\alpha| \)$
  - 신호가 $\(n \geq 0\)$에 대해 정의됨 (단위 계단 신호 \(u[n]\))
- **사례 2**: $\( |z| < |\alpha| \)$
  - 신호가 \(n < 0\)에 대해 정의됨 (역 단위 계단 신호 \(u[-n-1]\))

 
# 예제 3: 복합 신호의 Z-변환 분석

![image](https://github.com/user-attachments/assets/4ea4aa56-5ba5-4155-9051-9b783ab5af1e)

## 1. 신호 정의
- 주어진 신호:$\[x[n] = \left(\frac{1}{4}\right)^n u[n] + \left(-\frac{1}{2}\right)^n u[n]\]$
- Z-변환:$\[X(z) = \frac{z}{z - \frac{1}{4}} + \frac{z}{z + \frac{1}{2}}, \quad \text{ROC: } |z| > \frac{1}{2}\]$


## 2. Z-변환 계산
- Z-변환을 하나로 합치기:$\[X(z) = \frac{z}{z - \frac{1}{4}} + \frac{z}{z + \frac{1}{2}}\]$
- 통분하여 계산:$\[X(z) = \frac{z^2 + \frac{1}{2}z}{(z - \frac{1}{4})} + \frac{z^2 - \frac{1}{4}z}{(z + \frac{1}{2})}\]$
- 최종 정리:$\[X(z) = \frac{2z(z + \frac{1}{8})}{(z - \frac{1}{4})(z + \frac{1}{2})}, \quad \text{ROC: } |z| > \frac{1}{2}\]$


## 3. 극(Poles)와 영점(Zeros)
- **영점(Zeros)**:$\[z = 0, \quad z = -\frac{1}{8}\]$
- **극(Poles)**:$\[z = \frac{1}{4}, \quad z = -\frac{1}{2}\]$

## 4. Z-평면 시각화
- **영점(Zeros)**
  - $\(z = 0\)$: 원점에서 표시 (●)
  - $\(z = -\frac{1}{8}\)$: 음의 실수 축에서 표시 (●)
- **극(Poles)**
  - $\(z = \frac{1}{4}\)$: 양의 실수 축에서 표시 (×)
  - $\(z = -\frac{1}{2}\)$: 음의 실수 축에서 표시 (×)
- **ROC**
  - $\( |z| > \frac{1}{2} \)$: Z-평면 상에서 반지름이 $\(\frac{1}{2}\)$인 원 외부의 영역
 

# 예제 4: 복합 신호의 Z-변환 분석

![image](https://github.com/user-attachments/assets/04d6e2af-c45b-4183-b52c-7028bf3c87fb)


## 1. 신호 정의
- 주어진 신호:$\[x[n] = \left(\frac{1}{4}\right)^n u[n] - \left(-\frac{1}{2}\right)^n u[-n-1]\]$
- Z-변환:$\[X(z) = \frac{z}{z - \frac{1}{4}} + \frac{z}{z + \frac{1}{2}}, \quad \text{ROC: } \frac{1}{4} < |z| < \frac{1}{2}\]$

## 2. Z-변환 계산
- Z-변환의 합:$\[X(z) = \frac{z}{z - \frac{1}{4}} + \frac{z}{z + \frac{1}{2}}\]$
- 통분하여 계산:$\[X(z) = \frac{z(z + \frac{1}{8})}{(z - \frac{1}{4})(z + \frac{1}{2})}, \quad \text{ROC: } \frac{1}{4} < |z| < \frac{1}{2}\]$

## 3. 극(Poles)와 영점(Zeros)
- **영점(Zeros)**:$\[z = 0, \quad z = -\frac{1}{8}\]$
- **극(Poles)**:$\[z = \frac{1}{4}, \quad z = -\frac{1}{2}\]$

## 4. Z-평면 시각화
- **영점(Zeros)**
  - \(z = 0\): 원점에서 표시 (●)
  - $\(z = -\frac{1}{8}\)$: 음의 실수 축에서 표시 (●)
- **극(Poles)**
  - $\(z = \frac{1}{4}\)$: 양의 실수 축에서 표시 (×)
  - $\(z = -\frac{1}{2}\)$: 음의 실수 축에서 표시 (×)
- **ROC**
  - $\( \frac{1}{4} < |z| < \frac{1}{2} \)$: Z-평면 상에서 두 극 사이의 고리 형태

## 5. 결과 요약
1. 신호 $\(x[n] = \left(\frac{1}{4}\right)^n u[n] - \left(-\frac{1}{2}\right)^n u[-n-1]\)$의 Z-변환은
   - $\[X(z) = \frac{z(z + \frac{1}{8})}{(z - \frac{1}{4})(z + \frac{1}{2})}, \quad \text{ROC: } \frac{1}{4} < |z| < \frac{1}{2}\]$
2. 극과 영점은 각각 \(z\)-평면에서 신호의 특성을 나타냄.
3. ROC는 Z-평면 상에서 두 극 사이의 고리 형태로 정의됩니다.


# 예제 5: 유한 길이 지수 신호의 Z-변환 분석

![image](https://github.com/user-attachments/assets/4a55176a-a673-4660-bc21-5b8705b96f6b)

## 1. 신호 정의
- 주어진 신호: $\[ x[n] = \begin{cases} \alpha^n & 0 \leq n \leq N-1 \\ 0 & \text{otherwise} \end{cases} \]$

## 2. Z-변환 계산
- Z-변환:$\[X(z) = \sum_{n=0}^{N-1} \alpha^n z^{-n} = \sum_{n=0}^{N-1} (\alpha z^{-1})^n\]$
- 기하급수 합 공식 적용:$\[X(z) = \frac{1 - (\alpha z^{-1})^N}{1 - \alpha z^{-1}}\]$
- 정리된 형태:$\[X(z) = \frac{z^N - \alpha^N}{z^{N-1} (z - \alpha)}\]$

## 3. 수렴 영역(ROC)
- 수렴 조건:$\[\sum_{n=0}^{N-1} |\alpha z^{-1}|^n < \infty\]$
- ROC
  - $\( |z| < \infty \)$ (전체 \(z\)-평면에서 \(z = 0\) 제외)

## 4. 극(Poles)와 영점(Zeros)
- **영점(Zeros)**:$\[z_k = \alpha e^{j \frac{2\pi k}{N}}, \quad k = 0, 1, 2, \dots, N-1\]$
  - $\(z_k\)$: 원형적으로 배치된 \(N\)개의 영점

- **극(Poles)**:$\[z = 0 \quad (\text{\(N-1\) 중복}) \quad \text{and} \quad z = \alpha\]$
  - 극 $\(z = \alpha\)$는 영점 $\(z = \alpha\)$와 상쇄됨

## 5. Z-평면 시각화
- **영점(Zeros)**
  - $\(z_k = \alpha e^{j \frac{2\pi k}{N}}\)$: Z-평면에서 원형으로 분포
- **극(Poles)**
  - \(z = 0\): 원점에서 \(N-1\) 중복 극 표시
  - $\(z = \alpha\)$: 영점에 의해 상쇄됨
- **ROC**
  - 전체 \(z\)-평면에서 \(z = 0\)을 제외한 영역
 

## 6. 요약
1. 신호 $\(x[n] = \alpha^n\)$의 유한 길이 버전의 Z-변환:$\[X(z) = \frac{z^N - \alpha^N}{z^{N-1} (z - \alpha)}\]$
2. ROC는 \(z = 0\)을 제외한 전체 \(z\)-평면입니다
3. 극과 영점은 신호의 주파수 특성과 시간 도메인 특성을 결정합니다


# 예제 6: $\(z = \infty\)$에서의 극(Pole)과 영점(Zero)

![image](https://github.com/user-attachments/assets/ed5e562b-a61f-4b30-b4b0-932ba9ff89fe)

## 1. 예제 6-1: $\(X(z) = \frac{z+1}{(z+2)(z-1)}\)$
- **영점(Zeros)**:$\[z = -1\]$
- **극(Poles)**:$\[z = -2, \quad z = 1\]$
- **특징**:
  - $\(z \to \infty\)$일 때:$\[\lim_{z \to \infty} X(z) = \lim_{z \to \infty} \frac{1}{z} = 0\]$
    - $\(z = \infty\)$는 영점(Zero)입니다.

## 2. 예제 6-2: $\(X(z) = \frac{(z+2)(z-1)}{z+1}\)$
- **영점(Zeros)**:$\[z = -2, \quad z = 1\]$
- **극(Poles)**:$\[z = -1\]$
- **특징**:
  - $\(z \to \infty\)$일 때:$\[\lim_{z \to \infty} X(z) \to \infty\]$
    - $\(z = \infty\)$는 극(Pole)입니다

## 3. 일반화된 규칙: $\(z = \infty\)$에서의 극과 영점
- Z-변환이 유리 함수 형태일 때:$\[X(z) = \frac{P(z)}{Q(z)}\]$
  - \(P(z)\): 분자의 차수 \(M\)
  - \(Q(z)\): 분모의 차수 \(N\)

1. **\(N > M\)**
   - $\(z = \infty\)$는 **\(N-M\)개의 영점(Zero)**를 가짐
2. **\(M > N\)**
   - $\(z = \infty\)$는 **\(M-N\)개의 극(Pole)**을 가짐

## 4. 요약
1. $\(z = \infty\)$에서의 극과 영점은 함수의 분자와 분모의 차수 관계에 따라 결정됩니다
2. 차수 \(M\)과 \(N\)의 차이에 따라
   - \(N > M\): $\(z = \infty\)$에서 영점(Zero)
   - \(M > N\): $\(z = \infty\)$에서 극(Pole)
3. 이는 시스템의 안정성과 주파수 응답을 분석하는 데 중요한 정보입니다
