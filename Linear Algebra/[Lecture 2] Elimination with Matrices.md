## Elimination
- The way every software solves equations
- row를 제거하며, 식을 풀어나감
![image](https://github.com/user-attachments/assets/834c1889-2144-4568-9204-ba3464bfa8eb)

![image](https://github.com/user-attachments/assets/ca8f3d28-f277-462d-b1e0-014998572b99)

  ### How can we fail?
  - 만약 첫번째 row, 첫번째 col이 0이라면?
    - Nope! 그때는 단순히 row switch 하면 됨!
    - 일시적인 오류 발생
  - change할 row가 없는데 pivot 위치가 0일 경우
    - **Fail!**(x invertible = 역행렬이 불가능하게 됨)
    - 완전한 오류 발생
  
## Matrices
