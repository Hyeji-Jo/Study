![IMG_2E7937396CA4-1](https://github.com/user-attachments/assets/e0f659c4-1f2a-4d39-966b-0016af122080)

## Solvability : Condition on b
![IMG_457EE1F63169-1](https://github.com/user-attachments/assets/2bae8832-571f-406a-b74e-4269a20deee6)

## Ax=b의 Complete solution
![IMG_50659574598A-1](https://github.com/user-attachments/assets/191fe4b3-511c-422d-9030-fa6957495e8f)

## mxn matrix A of rank r
![IMG_C4A679EBB2FF-1](https://github.com/user-attachments/assets/d537b4ea-b6f9-4e00-bfee-adc3ca74f67c)

![IMG_3C64E356C54B-1](https://github.com/user-attachments/assets/1e3b82f5-f05b-41d3-896e-d4b0ae61bd7f)

|                           | Full column rank (r = n < m)                  | Full row rank (r = m < n)                  | Full rank (r = m = n)                     | r < m, r < n                                  |
|---------------------------|-----------------------------------------------|-------------------------------------------|--------------------------------------------|-----------------------------------------------|
| rref                      | R = \[I 
0\]                                   | R = \[I F\]                               | R = I (invertible)                         | R = \[I F 
0 0\]                               |
| N(A)                      | only zero vector (no free variable)           | Exists (free variable이 있기 때문에)      | only zero vector (no free variable)        | Exists (free variable이 있기 때문에)         |
| Solution Ax = b           | 0 or only 1 solution (unique solution : comb of columns of A) | 무수히 많음 - 모든 b에 대해 solution이 있다. (free variable이 있기 때문에) | unique solution (no free variable)          | no solution or infinite solutions             |
