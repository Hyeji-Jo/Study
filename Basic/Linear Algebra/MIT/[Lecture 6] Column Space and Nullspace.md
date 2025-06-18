## Vector Spaces and Subspaces
- $$P \cup L$$은 부분 공간인가? : **Nope!**
  - P에 속하지만, L에는 속하지 않는 벡터와 반대로 L에 속하지만, P에는 속하지 않는 벡터의 선형결합은 $$P \cup L$$이라는 공간에 속하지 않을 수 있기 때문
- $$P \cap L$$은 부분 공간인가? : **Yes!**
  - 집합에 속하는 벡터가 영벡터 [0 0 0]'뿐이기 때문에, 영벡터는 부분 공간에 해당하므로 $$P \cap L$$는 부분 공간이라고 할 수 있음

## Column Space of A
![image](https://github.com/user-attachments/assets/fc8a7b6f-3f32-4205-98be-184fcd47dac1)
![image](https://github.com/user-attachments/assets/de083dd6-448f-459e-a306-a7642e99f2c4)

- A의 첫 번째, 두 번째 column 벡터는 상호 독립적(Independent)이나, 3번째 column은 기존 두 개의 column에 종속적(Dependent)이다.
  - 이는 앞의 두 개의 column의 합을 통해 col3를 정의할 수 있기 때문
 
## Null Space
![image](https://github.com/user-attachments/assets/eaa8f6e4-9300-4533-bce4-bcc46c5d8448)
- Null space 역시 부분 공간임을 의미
  - N(A)는 해 x로 이루어진 공간이므로 x의 차원인 벡터 공간의 부분 공간이라는 점 
