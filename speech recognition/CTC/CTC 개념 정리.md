# 1. 정의
<img width="1027" alt="image" src="https://github.com/user-attachments/assets/9cc00313-d122-46ed-8234-1682ed69d5ff" />

# 2. 핵심개념
<img width="937" alt="image" src="https://github.com/user-attachments/assets/fa0d65e4-d2ce-4d45-bceb-16dc3733dace" />

### CTC + RNN 모델 동작과정 
1) 음성특징벡터 추출
2) RNN을 사용하여 시퀀스 데이터 처리(LSTM, GRU)
3) CTC Loss 사용하여 정렬없이 출력 시퀀스 예측
4) Decoding 과정에서 Black Token을 제거하여 최종 텍스트 생성

# 3. 한계점
<img width="910" alt="image" src="https://github.com/user-attachments/assets/bee6bbe2-2232-49db-a72e-145867a8a4c6" />

<img width="899" alt="image" src="https://github.com/user-attachments/assets/c264ebb7-836e-40a4-88b8-bf2f4fe55f23" />
