# 수렴 영역 (Region of Convergence, ROC)

![image](https://github.com/user-attachments/assets/ee81357d-16ad-45b9-8fa6-fcdaf262a84e)

## 1. ROC의 정의
- **ROC**: 신호 \(x[n]\)의 Z-변환 \(X(z)\)가 **수렴(존재)**하는 \(z\)의 집합
- Z-변환 정의:$\[X(z) = \sum_{n=-\infty}^\infty x[n] z^{-n}\]$


## 2. ROC 분석

### 사례 1: 지연 (Delay)
- 신호 $\(w[n] = \delta[n-n_0]\)$의 Z-변환:$\[W(z) = \sum_{n=-\infty}^\infty \delta[n-n_0] z^{-n} = z^{-n_0}\]$

- **특징**
  - \(n_0 > 0\) (지연이 있을 때): \(z = 0\) 제외
  - \(n_0 < 0\) (지연이 음수일 때): \(z = \infty\) 제외

- **결론**
  - $\[ \forall n_0: \begin{cases}  z \neq 0, & \text{if } n_0 > 0 \\ z \neq \infty, & \text{if } n_0 < 0 \end{cases} \]$

## 3. 신호와 Z-변환 관계
- 지연 신호 $\(\delta[n-n_0]\)$는 Z-변환에서 $\(z^{-n_0}\)$로 변환
  - $\[\delta[n-n_0] \xrightarrow{Z} z^{-n_0}\]$

## 4. 요약
1. ROC는 Z-변환이 수렴하는 복소수 영역을 정의합니다
2. 신호의 특성 (예: 지연)은 ROC에 영향을 미칩니다
   - $\(n_0 > 0\)$: $\(z = 0\)$ 제외
   - $\(n_0 < 0\)$: $\(z = \infty\)$ 제외
3. ROC는 디지털 신호 처리와 안정성 분석에 중요합니다.


# 사례 2: 지수 신호 $\(x[n] = \alpha^n u[n]\)$의 Z-변환과 수렴 영역 (ROC)

![image](https://github.com/user-attachments/assets/7a43e027-fda7-41bf-95cb-79ed6bb30e38)

## 1. 신호와 Z-변환
- 신호:$\[x[n] = \alpha^n u[n]\]$
  - 여기서 \(u[n]\)은 단위 계단 함수 (Unit Step Function).

- Z-변환:$\[X(z) = \sum_{n=0}^\infty \alpha^n z^{-n} = \sum_{n=0}^\infty (\alpha z^{-1})^n\]$

- 기하급수 합을 이용한 결과:$\[X(z) = \frac{1}{1 - \alpha z^{-1}} = \frac{z}{z - \alpha}, \quad \text{provided } |\alpha z^{-1}| < 1 \, (\text{즉, } |z| > |\alpha|)\]$

## 2. 수렴 조건 (ROC)
- Z-변환이 수렴하기 위한 조건:$\[|\alpha z^{-1}| < 1 \quad \Rightarrow \quad |z| > |\alpha|\]$

- **ROC**:
  - 원점에서 반지름이 $\(|\alpha|\)$인 원 외부의 영역.
  - $\( |z| > |\alpha| \)$.

## 3. ROC의 시각화
- **Z-평면**:
  - $\(z = \alpha\)$에서 극(Pole)이 위치.
  - ROC는 극을 포함하지 않으며, $\( |z| > |\alpha| \)$ 영역입니다.

## 4. 신호의 다양한 $\(\alpha\)$ 값에 따른 특징
1. $\(0 < \alpha < 1\)$: 신호가 \(n\)에 따라 지수적으로 감소.
2. $\(\alpha > 1\)$: 신호가 \(n\)에 따라 지수적으로 증가.
3. $\(-1 < \alpha < 0\)$: 신호가 번갈아가며 감소.
4. $\(\alpha < -1\)$: 신호가 번갈아가며 증가.


# 사례 3: 지수 신호 $\(y[n] = -\alpha^n u[-n-1]\)$의 Z-변환과 수렴 영역 (ROC)

![image](https://github.com/user-attachments/assets/2ca9d293-5ae7-46fd-8a56-05298f9cdeb8)

## 1. 신호와 Z-변환
- 신호:$\[y[n] = -\alpha^n u[-n-1]\]$
  - 여기서 $\(u[-n-1]\)$은 역 단위 계단 함수 (Reversed Unit Step Function).

- Z-변환:$\[Y(z) = \sum_{n=-\infty}^\infty y[n] z^{-n} = \sum_{n=-\infty}^{-1} (-\alpha^n) z^{-n}\]$

- 지수 신호의 특성을 이용한 계산:$\[Y(z) = -\sum_{n=-\infty}^{-1} (\alpha^{-1}z)^n\]$

- 기하급수 합을 이용한 결과:$\[Y(z) = \frac{\alpha}{z - \alpha} + 1 \quad \text{(for \(|z| < |\alpha|\))}.\]$

## 2. 결과 요약
- Z-변환:$\[Y(z) = \frac{z}{z - \alpha}, \quad \text{ROC: } |z| < |\alpha|\]$

- Z-변환 결과는 이전 사례 $\(x[n] = \alpha^n u[n]\)$의 결과와 동일하지만, **수렴 영역(ROC)**이 다릅니다
  - **이전 사례**: $\( |z| > |\alpha| \)$
  - **현재 사례**: $\( |z| < |\alpha| \)$

## 3. 중요한 관찰
- Z-변환은 **ROC가 없으면 고유하지 않습니다**
  - 동일한 수식 형태를 가진 \(X(z)\)와 \(Y(z)\)는 ROC에 따라 다른 신호를 나타냅니다.
- ROC는 Z-변환의 고유성을 보장하며, 신호의 안정성과 인과성을 결정합니다.


## 4. ROC의 시각화
- **Z-평면**
  - $\(z = \alpha\)$에서 극(Pole)이 위치
  - ROC는 극 내부의 영역 $\( |z| < |\alpha| \)$입니다


# Z-변환과 수렴 영역 (ROC: Region of Convergence)

![image](https://github.com/user-attachments/assets/f2f258ce-5068-4aa4-b7a1-fcf4619caeba)

## 1. Z-변환의 정의와 수렴 조건
- Z-변환:$\[X(z) = \sum_{n=-\infty}^\infty x[n] z^{-n}\]$

- **수렴 조건**:
  - 급수가 수렴하려면 **절대 합 가능성(Absolutely Summable)**이 필요:$\[\sum_{n=-\infty}^\infty |x[n] z^{-n}| = \sum_{n=-\infty}^\infty |x[n]| |z^{-n}| < \infty\]$

## 2. 수렴 영역(ROC)
- ROC는 $\( |z| \)$에 따라 결정됩니다
  - Z-평면에서 $\( |z| \)$ 값에 따라 원 또는 고리 형태로 나타남


## 3. 예제: $\(g[n] = \left(\frac{1}{4}\right)^n u[n] - \left(\frac{1}{2}\right)^n u[-n-1]\)$

### (1) Z-변환 계산
1. 첫 번째 항 $\( \left(\frac{1}{4}\right)^n u[n] \)$:$\[Z\text{-변환}: \frac{z}{z - \frac{1}{4}}, \quad |z| > \frac{1}{4}\]$

2. 두 번째 항 $\( -\left(\frac{1}{2}\right)^n u[-n-1] \)$:$\[Z\text{-변환}: \frac{z}{z - \frac{1}{2}}, \quad |z| < \frac{1}{2}\]$

3. 전체 Z-변환:$\[G(z) = \frac{z}{z - \frac{1}{4}} + \frac{z}{z - \frac{1}{2}}\]$

### (2) ROC 분석
- 두 항의 ROC가 겹치는 영역:$\[\frac{1}{4} < |z| < \frac{1}{2}\]$
- 이는 Z-평면에서 중심이 원점인 고리 형태로 나타남

## 4. ROC의 시각화
- **Z-평면**:
  - $\(z = \frac{1}{4}\)$와 $\(z = \frac{1}{2}\)$에서 극(Pole)이 위치
  - ROC는 두 극 사이의 영역 $\(\frac{1}{4} < |z| < \frac{1}{2}\)$입니다


# 예제: 신호 $\(x[n] = u[n] + \left(-\frac{3}{4}\right)^n u[-n-1]\)$의 Z-변환 분석

![image](https://github.com/user-attachments/assets/149e502f-ec2f-4e81-91d3-756432a81b98)

## 1. 문제 정의
- 신호:$\[x[n] = u[n] + \left(-\frac{3}{4}\right)^n u[-n-1]\]$

- Z-변환을 구하기 위해, 신호를 Z-변환 쌍을 이용해 표현:$\[x[n] = u[n] + \delta[n] - \left(-\frac{3}{4}\right)^n u[-n-1]\]$

## 2. 개별 항의 Z-변환
1. **단위 계단 신호 \(u[n]\):** $\[Z\text{-변환}: \frac{z}{z-1}, \quad \text{ROC: } |z| \geq 1\]$

2. **단위 임펄스 신호 $\(\delta[n]\)$:** $\[Z\text{-변환}: 1\]$

3. **역 단위 계단 신호 $\(\left(-\frac{3}{4}\right)^n u[-n-1]\)$:** $\[Z\text{-변환}: \frac{z}{z+\frac{3}{4}}, \quad \text{ROC: } |z| < \frac{3}{4}\]$

## 3. 수렴 영역(ROC)의 분석
- ROC
  - $\(u[n]\)$: $\( |z| \geq 1 \)$
  - $\(\left(-\frac{3}{4}\right)^n u[-n-1]\)$: $\( |z| < \frac{3}{4} \)$

- 두 ROC는 **서로 겹치지 않음 (Incompatible)**
  - $\( |z| \geq 1 \)$와 $\( |z| < \frac{3}{4} \)$는 교집합이 존재하지 않음.

## 4. 결과
- Z-변환 \(X(z)\)는 **존재하지 않음**:$\[X(z) \text{ does not exist!}\]$
