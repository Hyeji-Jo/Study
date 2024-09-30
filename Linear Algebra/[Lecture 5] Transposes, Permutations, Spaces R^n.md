## Permutations P  : execute row exchanges
![image](https://github.com/user-attachments/assets/6d878b8a-6c97-4453-96a8-ee3520631ee0)


- numerical accuracy 때문에 0에 가까운 pivot을 좋아하지 않는다

### 0에 가까운 pivot을 좋아하지 않는 이유
> 작은 피벗 값은 수치적 계산에서 부동소수점 오류를 증가시키고, 행렬의 조건수를 악화시켜 수치적 불안정을 초래하며  
> 이는 오차를 크게 만들 수 있어 계산의 정확도를 떨어뜨리기 때문에, 작은 피벗 값은 선호되지 않음  
- 행렬 분해 과정에서 **부동소수점 연산(floating point arithmetic)** 에서 오차가 증폭될 가능성을 높임
  - 부동소수점 시스템에서는 숫자의 유효 자리수가 제한되어 있기 때문에, 매우 작은 수나 큰 수와의 연산은 **라운딩 오류(rounding error)** 를 유발 
-  피벗이 0에 가까워지면 행렬의 **조건수(condition number)** 가 커짐
  - 조건수가 큰 행렬은 **ill-conditioned**라고 부르며, 이는 입력 데이터에 작은 변화가 생겼을 때 결과값이 크게 변할 수 있음을 의미
  - **시스템의 민감도가 증가하여 작은 부동소수점 오류가 전체 결과에 큰 영향을 줄 수 있음**
- 피벗 값이 작으면 역행렬을 구할 때 나눗셈 과정에서 매우 큰 수들이 나타날 수 있으며, 이로 인해 수치적 오류가 커질 수 있음
- 가우스 소거법(Gaussian Elimination)을 사용하여 연립방정식을 풀 때, 피벗을 기준으로 행렬의 나머지 부분을 변환할 때 발생하는 오차가 커짐
- **해결 방법: 피벗팅(Pivoting)**
  - **부분 피벗팅(partial pivoting)** 이나 **완전 피벗팅(full pivoting)** 같은 기법을 사용
  - 각 단계에서 피벗으로 사용할 값이 0에 가깝지 않도록 행이나 열을 재배열하는 방법

## Transposes
![image](https://github.com/user-attachments/assets/7640dbab-b3ce-4f63-a4fc-a6ea35e2fc3d)


## Vector Space
> **같은 공간에 속한 벡터들은 서로가 서로의 선형결합에 의해 표현될 수 있어야 한다**
- 벡터 공간 내에 존재하는 임의의 **벡터 v와 w**는 그 둘을 더해도 **(v+w)** 그 결과가 반드시 **같은 벡터 공간에 존재**해야 한다. 
- 벡터 공간 내에 존재하는 임의의 벡터 **v**에 **임의의 상수 c를 곱해도 (cv)** 그 결과가 반드시 **같은 벡터 공간에 존재**해야 한다.
- 벡터 공간 내에 존재하는 임의의 **벡터 v, w**와 **임의의 상수 c, d**에 대해 **모든 경우의 cv+dw 조합**(각 벡터에 임의의 상수를 곱한 뒤 더하는, 즉 **선형 결합(Linear Combination)**)결과가 반드시 **같은 벡터 공간에 존재**해야 한다.
![image](https://github.com/user-attachments/assets/2aa1db25-7d67-4fc8-819e-dd3d6a8b9911)
![image](https://github.com/user-attachments/assets/9f2f8e0b-0ff2-49ce-aa85-88d0e895972f)


## Subspace of R<sup>2<sub>  : a vector space inside R<sup>2<sub>
- 부분 공간 역시 벡터 공간이기 때문에 벡터 공간의 조건을 모두 만족해야함
- R<sup>2<sub> 부분 공간이 될 수 있는 목록
  - all of R<sup>2<sub> : R<sup>2<sub> 공간 전체는 그 자체로 자신의 부분 공간이다. (가장 큰 부분 공간)
  - any line through zero vector [0 0]' : 영벡터를 지나는 직선 * 단, 이 직선이 1차원 공간을 의미하는 것은 아니다. R<sup>2<sub>에 속한 직선이므로 두 개의 성분을 갖기 때문이다.
  - zero vector [0 0]' : 영벡터는 부분 공간의 성질을 모두 만족한다. (가장 작은 부분 공간)
