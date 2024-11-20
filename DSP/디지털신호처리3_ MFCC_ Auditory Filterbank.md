# MFCC(Mel-Frequency Cepstral Coefficients)
- **정의**
  - 음성 신호의 특징을 추출하는 기술 중 하나
  - 서로 독립적인 주요 음성 특성을 포함
- **배경**
  - Mel spectrum 혹은 Log Mel spectrum은 Feature내 변수 간 상관관계가 존재함
    - Filter Back는 모두 Overlapping 되어 있기 때문에 에너지들 사이에 상관관계 존재
    - 멜 필터 뱅크는 삼각형 형태로 구성되며, 각 필터가 인접 대역과 겹치고,
      - 이로 인해 한 주파수 대역의 값이 인접한 대역폭 값에 영향을 주게됨
    ![image](https://github.com/user-attachments/assets/90820750-9cc7-4938-9951-03308ba92a02)

    - Mel Spectrum이나 Log Mel Spectrum은 멜 필터를 적용하여 주파수 대역을 인간 청각에 맞게 변환한 스펙트럼 
  - 변수간의 종속성을 없애기 위해 역푸리에 변환(Discrete Cosine Transform, DCT) 적용
    - DCT는 신호를 주파수 공간에서 에너지 밀도로 변환하면서 상관관계 해소 
  - 이렇게 얻어진 값이 MFCC
- **단계**
  - STFT(Short Time Fourier Transform)에 의해 주어진 음성 신호를 작은 프레임 단위로 나누어서 주파수 영역의 데이터로 변환
  - Mel Filter Bank로 멜 스펙트럼을 계산
  - Mel spectrum에 log 적용
  - Mel-log-spectrum list 전체에 DCT(Discrete Cosine Transfrom)적용
  - 이를 이용하여 해당 프레임의 특징을 추출
  - 이때 추출된 특징은 일반적으로 MFCC 계수라고 부름
- **단점**
  - 그러나 Mel spectrum 혹은 Log Mel spectrum 대비 버려지는 정보가 많음
  - 최근 딥러닝 기반 모델들에서는 MFCC보다는 Mel spectrum, Log Mel spectrum이 더 널리 사용됨
```py
# mfcc (DCT)
## MFCC 계산
mfcc = librosa.feature.mfcc(S=log_mel_S, n_mfcc=13) #-> 일반적으로 음성 분석에서 13개의 계수 사용
## 메모리 절약을 위해 데이터 형식 변경
mfcc = mfcc.astype(np.float32)    # to save the memory (64 to 32 bits)
## 시각화
plt.figure(figsize=(12,4))
librosa.display.specshow(mfcc)
```


# 기타
## DCT(Discrete Cosine Transfrom)
- 신호나 데이터를 주파수 영역으로 변환하는 기법 중 하나
- n개의 데이터를 n개의 코사인 함수의 합으로 표현하여 데이터의 양을 줄이는 방식
  - 저주파수에 에너지가 집중되고 고주파수 영역에 에너지가 감소 
