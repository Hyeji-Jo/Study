
# Signal Processing: System Overview
<img width="789" alt="image" src="https://github.com/user-attachments/assets/61323e98-bcd3-4ed6-bbcb-a2a18998628f" />

## 1. System의 역할
- 시스템 \(H\)는 입력 신호 \(x(t)\)를 출력 신호 \(y(t)\)로 변환
- 입력과 출력 신호는 시간 연속형 (\(x(t), y(t)\)) 또는 이산형 (\(x[n], y[n]\))일 수 있습니다


## 2. 시스템의 주요 목적
### 1) 물리적 현상을 모델링 (Model Physical Phenomenon)
- **예시: 무선 통신에서의 다중 경로 전파(Multipath Propagation)**  
  - 신호는 물리적 장애물(빌딩 등)로 인해 반사 신호가 섞임
  - 수학적으로 다음과 같이 표현됩니다: $\[y(t) = x(t) + \alpha x(t-\tau)\]$
    - $\(\alpha\)$ : 감쇠 계수 $(\(|\alpha| < 1\))$
    - $\(\tau\)$ : 시간 지연(Time Delay)

### 2) 원하는 효과 구현 (Implement Desired Effect)
- 시스템은 특정 입력 신호를 원하는 출력 신호로 변환하는 데 사용
- **예시: 신호 복원 과정**
  - $\[z(t) = y(t) - \alpha y(t-\tau) + \alpha^2 y(t-2\tau) - \alpha^3 y(t-3\tau)\]$
  - 위 과정을 통해 \(z(t)\)는 원 신호 \(x(t)\)에 가까워짐


## 3. 주요 응용
- **다중 경로 모델링** : 실제 세계의 물리적 신호 왜곡 현상을 수학적으로 모델링
- **신호 복원** : 입력 신호에서 원하는 정보를 추출하거나 원래 신호를 복원

### 수식 설명
1. **모델링된 물리적 신호** : $\[y(t) = x(t) + \alpha x(t-\tau)\]$

2. **신호 복원 과정** : $\[z(t) = x(t) \approx y(t) - \alpha y(t-\tau) + \alpha^2 y(t-2\tau) - \alpha^3 y(t-3\tau)\]$
