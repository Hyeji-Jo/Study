# 논문 정리 : End-to-End Speech Recognition: A Survey (T-ASLP 2024)

## Abstract
- **딥러닝이 도입된 ASR** 모델이 도입되지 않은 모델과 비교해 **단어 오류율(WER)이 50%이상 감소**
  - **WER** : 음성인식 시스템이 인식한 텍스트가 정답과 얼마나 다른지를 측정하는 평가 지표
    - \text{WER} = \frac{S + D + I}{N} 
    - 추가, 삭제, 대체된 단어 수를 전체 정답 단어 수로 나누어 계산
    - 낮을수록 좋은 값
      
- 현재는 End-to-End(E2E)모델이 음성인식의 주 방식
